#!/usr/bin/env python3
"""Regression tests for compiler risks found by the historical Arc runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[3]
REVIEW_PATH = Path(__file__).with_name("review_candidate.py")
SPEC = importlib.util.spec_from_file_location("task78_review_candidate", REVIEW_PATH)
assert SPEC and SPEC.loader
review_candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_candidate)
GATE_PATH = Path(__file__).with_name("run_candidate_gate.py")
GATE_SPEC = importlib.util.spec_from_file_location("task78_run_candidate_gate", GATE_PATH)
assert GATE_SPEC and GATE_SPEC.loader
run_candidate_gate = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(run_candidate_gate)


def inspect_zip(name: str) -> dict:
    archive = ROOT / "competition" / "task78" / name
    with tempfile.TemporaryDirectory(prefix="task78-review-regression-") as directory:
        directory_path = Path(directory)
        with zipfile.ZipFile(archive) as package:
            package.extractall(directory_path)
        reports = {
            backend: review_candidate.inspect_source(
                directory_path / review_candidate.source_name(backend)
            )
            for backend in review_candidate.BACKENDS
        }
        blockers = [
            {"backend": backend, **finding}
            for backend, report in reports.items()
            for finding in report["blockers"]
        ]
        return {"passed": not blockers, "blockers": blockers}


class HistoricalArcFailureRegression(unittest.TestCase):
    def test_v21_catches_known_runtime_branches(self) -> None:
        report = inspect_zip("flagos-task78-v21.zip")
        failures = {
            (item["backend"], item["kind"])
            for item in report["blockers"]
        }
        self.assertFalse(report["passed"])
        self.assertIn(("iluvatar", "runtime-branch-in-jit"), failures)
        self.assertIn(("metax", "runtime-branch-in-jit"), failures)

    def test_v22_catches_known_target_compiler_failures(self) -> None:
        report = inspect_zip("flagos-task78-v22.zip")
        failures = {
            (item["backend"], item["kind"])
            for item in report["blockers"]
        }
        self.assertFalse(report["passed"])
        self.assertIn(("enflame", "invalid-num-warps"), failures)
        self.assertIn(("hygon", "hygon-cast-before-broadcast"), failures)
        self.assertNotIn(("enflame", "uncapped-next-power-of-two"), failures)


class WorkflowHardGateRegression(unittest.TestCase):
    def inspect_source(self, source: str, filename: str = "concat_and_cast_mha_k.py") -> dict:
        with tempfile.TemporaryDirectory(prefix="task78-review-source-") as directory:
            path = Path(directory) / filename
            path.write_text(source, encoding="utf-8")
            return review_candidate.inspect_source(path)

    def test_rejects_unknown_config_abi_option(self) -> None:
        report = self.inspect_source(
            "triton.Config({'BT': 1}, num_warps=4, num_stages=1, multibuffer=True)\n",
            "concat_and_cast_mha_k_ascend.py",
        )
        self.assertIn("unknown-config-option", {item["kind"] for item in report["blockers"]})

    def test_rejects_explicit_tile_kwargs_on_autotuned_launch(self) -> None:
        report = self.inspect_source(
            """
@triton.autotune(configs=[], key=[])
@triton.jit
def _kernel(out, BR: tl.constexpr, NRC: tl.constexpr):
    pass

def concat_and_cast_mha_k(out):
    _kernel[(1,)](out, BR=128, NRC=1)
""",
            "concat_and_cast_mha_k_enflame.py",
        )
        kinds = {item["kind"] for item in report["blockers"]}
        self.assertIn("autotune-explicit-tile-constexpr", kinds)

    def test_rejects_masked_negative_pointer_and_scalar_mask(self) -> None:
        report = self.inspect_source(
            """
@triton.jit
def _kernel(out, rope, cols, DN, RS2, H):
    head = tl.program_id(0)
    head_ok = head < H
    value = tl.load(rope + (cols - DN) * RS2,
                    mask=head_ok & (cols >= DN), other=0)
    tl.store(out + cols, value, mask=head_ok & (cols >= DN))
""",
            "concat_and_cast_mha_k_kunlunxin.py",
        )
        kinds = {item["kind"] for item in report["blockers"]}
        self.assertIn("masked-negative-pointer", kinds)
        self.assertIn("implicit-scalar-mask-broadcast", kinds)

    def test_rejects_fixed_loop_count_with_varying_autotune_tile(self) -> None:
        report = self.inspect_source(
            """
@triton.autotune(
    configs=[
        triton.Config({'BC': 256}, num_warps=4),
        triton.Config({'BC': 512}, num_warps=4),
    ], key=[])
@triton.jit
def _kernel(out, BC: tl.constexpr):
    pass

def concat_and_cast_mha_k(out, dn):
    nc = triton.cdiv(dn, 512)
    _kernel[(1,)](out)
""",
            "concat_and_cast_mha_k_iluvatar.py",
        )
        self.assertIn("autotune-tile-grid-mismatch",
                      {item["kind"] for item in report["blockers"]})

    def test_subagent_novel_finding_is_not_a_soft_warning(self) -> None:
        receipt = {
            "review_type": "read-only-subagent",
            "review_mode": "adversarial-read-only",
            "searched_for_novel_risks": True,
            "reviewer": "test-reviewer",
            "candidate": "candidate",
            "reviewed_source_sha256": {"default": "digest"},
            "backend_findings": {
                "default": {
                    "status": "unknown",
                    "notes": [],
                    "evidence": [],
                    "novel_findings": [{"kind": "new-risk"}],
                }
            },
            "blockers": [],
        }
        with tempfile.TemporaryDirectory(prefix="task78-review-receipt-") as directory:
            path = Path(directory) / "review.json"
            import json
            path.write_text(json.dumps(receipt), encoding="utf-8")
            result = run_candidate_gate.review_check(
                path, True, {"default": "digest"}, "candidate"
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["novel_finding_count"], 1)


if __name__ == "__main__":
    unittest.main()
