"""Regression tests for the rules learned from real platform rejections.

Each test names the submission that motivated it.  These are the checks that
should have run *before* spending a submission slot.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from competition.adaptation import constexprs, failures, safety


def errors_for(source: str) -> list[str]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source)
        return safety.check_source(path)


HEADER = "import torch\nimport triton\nimport triton.language as tl\n\n"


class ModuleLevelContainerRule(unittest.TestCase):
    """Task 111, 10-02 07:04: every chip rejected on a module-level dict."""

    def test_module_level_dict_is_rejected(self) -> None:
        errors = errors_for(HEADER + "_PLANS = {}\n")
        self.assertTrue(
            any("module-level mutable container" in e for e in errors), errors
        )
        self.assertTrue(any("_PLANS" in e for e in errors), errors)

    def test_module_level_set_is_rejected(self) -> None:
        errors = errors_for(HEADER + "_SEEN = set()\n")
        self.assertTrue(
            any("module-level mutable container" in e for e in errors), errors
        )

    def test_annotated_module_level_dict_is_rejected(self) -> None:
        errors = errors_for(HEADER + "_PLANS: dict = {}\n")
        self.assertTrue(
            any("module-level mutable container" in e for e in errors), errors
        )

    def test_module_level_list_is_allowed(self) -> None:
        """Task 112 ships a module-level ``__all__`` list and passes."""
        self.assertEqual(errors_for(HEADER + '__all__ = ["op"]\n'), [])

    def test_scalar_constants_are_allowed(self) -> None:
        self.assertEqual(errors_for(HEADER + "_ROW_CAP = 1024\n"), [])

    def test_function_local_dict_is_allowed(self) -> None:
        errors = errors_for(
            HEADER + "def f():\n    cache = {}\n    return cache\n"
        )
        self.assertEqual(errors, [])


class OtherPlatformRules(unittest.TestCase):
    def test_try_is_rejected(self) -> None:
        errors = errors_for(
            HEADER
            + "def f():\n    try:\n        pass\n    except Exception:\n        pass\n"
        )
        self.assertTrue(any("try/except" in e for e in errors), errors)

    def test_torch_compute_is_rejected(self) -> None:
        self.assertTrue(
            any(
                "native torch compute" in e
                for e in errors_for(HEADER + "x = torch.zeros(3)\n")
            )
        )

    def test_literal_metadata_tensor_is_allowed(self) -> None:
        self.assertEqual(
            errors_for(
                HEADER + "x = torch.tensor([1, 2, 3], dtype=torch.int32)\n"
            ),
            [],
        )

    def test_computed_metadata_tensor_is_rejected(self) -> None:
        errors = errors_for(HEADER + "x = torch.tensor(dst.stride())\n")
        self.assertTrue(
            any("literal metadata buffer" in e for e in errors), errors
        )

    def test_tensor_method_compute_is_a_known_scope_limit(self) -> None:
        """Asserted, not assumed: ``torch.X(...)`` is what the scan sees.

        A reference-shaped ``dst.clone()`` is caught by the structural-delta
        gate instead; tightening this would also affect packages that call
        tensor methods legitimately.
        """
        self.assertEqual(errors_for(HEADER + "x = dst.clone()\n"), [])


class FailureLedger(unittest.TestCase):
    """Bytes that already failed must not be shipped at them again."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(
            failures,
            "path_for",
            lambda task: Path(self._tmp.name) / f"{task}.jsonl",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_blocks_when_the_target_is_still_routed_to_the_failed_bytes(
        self,
    ) -> None:
        failures.record(
            "t",
            {
                "kind": "compile",
                "targets": ["kunlunxin"],
                "members": {"generic.py": "aaa"},
                "message": "boom",
            },
        )
        blocked = failures.exact_repeat(
            "t", {"kunlunxin": "generic.py"}, {"generic.py": "aaa"}
        )
        self.assertEqual(len(blocked), 1)
        self.assertIn("kunlunxin", blocked[0])

    def test_a_dedicated_file_unblocks_the_target(self) -> None:
        failures.record(
            "t",
            {
                "kind": "compile",
                "targets": ["kunlunxin"],
                "members": {"generic.py": "aaa"},
                "message": "boom",
            },
        )
        routing = {"kunlunxin": "op_kunlunxin.py"}
        hashes = {"generic.py": "aaa", "op_kunlunxin.py": "bbb"}
        self.assertEqual(failures.exact_repeat("t", routing, hashes), [])

    def test_an_edit_to_the_routed_file_unblocks_it(self) -> None:
        failures.record(
            "t",
            {
                "kind": "compile",
                "targets": ["kunlunxin"],
                "members": {"generic.py": "aaa"},
                "message": "boom",
            },
        )
        self.assertEqual(
            failures.exact_repeat(
                "t", {"kunlunxin": "generic.py"}, {"generic.py": "ccc"}
            ),
            [],
        )

    def test_a_platform_dispatch_failure_never_blocks(self) -> None:
        """An infra failure is about the queue, not our bytes.

        Observed for real: Task 111's only missing chip (intl_b) failed twice
        with "评测任务提交失败，已停止重试" while passing at 8.25x on Task 112.
        Recording those entries made the gate reject the package whose whole
        point was to resubmit, so the retry -- the only possible remedy -- was
        forbidden by the ledger.
        """
        failures.record(
            "t",
            {
                "kind": "platform-infra",
                "targets": ["intl_b"],
                "members": {"generic.py": "aaa"},
                "message": "评测任务提交失败，已停止重试",
            },
        )
        self.assertEqual(
            failures.exact_repeat(
                "t", {"intl_b": "generic.py"}, {"generic.py": "aaa"}
            ),
            [],
        )

    def test_other_targets_are_unaffected(self) -> None:
        failures.record(
            "t",
            {
                "kind": "compile",
                "targets": ["kunlunxin"],
                "members": {"generic.py": "aaa"},
                "message": "boom",
            },
        )
        self.assertEqual(
            failures.exact_repeat(
                "t", {"ascend": "generic.py"}, {"generic.py": "aaa"}
            ),
            [],
        )

class ConstexprUnpackingRule(unittest.TestCase):
    """A tl.constexpr subscript returns a raw value; arange then refuses it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def write_temp(self, name: str, text: str) -> Path:
        path = Path(self._tmp.name) / name
        path.write_text(text, encoding="utf-8")
        return path


    def test_an_unpacked_constexpr_is_rejected(self) -> None:
        """The failure that cost a submission on 2026-10-02 08:07.

        ``constexpr.__getitem__`` returns the raw element, so a name bound from
        ``SHAPE[4]`` is a plain int and ``tl.arange`` refuses it on real
        hardware.  The gate must catch the pattern locally instead.
        """
        path = self.write_temp(
            "bad.py",
            """
import triton
import triton.language as tl


@triton.jit
def k(p, SHAPE: tl.constexpr):
    row_block = SHAPE[4]
    row = tl.arange(0, row_block)
""",
        )
        errors = constexprs.check_source(path)
        self.assertEqual(len(errors), 1)
        self.assertIn("tl.constexpr", errors[0])

    def test_a_rewrapped_constexpr_is_accepted(self) -> None:
        path = self.write_temp(
            "good.py",
            """
import triton
import triton.language as tl


@triton.jit
def k(p, SHAPE: tl.constexpr):
    row_block = tl.constexpr(SHAPE[4])
    row = tl.arange(0, row_block)
""",
        )
        self.assertEqual(constexprs.check_source(path), [])
