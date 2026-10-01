from __future__ import annotations
import tempfile
import unittest
import zipfile
from pathlib import Path
from competition.adaptation import compiler_scan as review_candidate
from competition.experiments.store import ROOT
from competition.adaptation.packaging import resolve_artifact

def inspect_zip(name):
    archive=resolve_artifact(ROOT / 'competition/task78',name)
    blockers=[]
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(archive) as stream:
            for backend in ('default','ascend','enflame','hygon','iluvatar','kunlunxin','metax'):
                name='concat_and_cast_mha_k'+('' if backend=='default' else '_'+backend)+'.py'
                path=Path(directory)/name
                path.write_bytes(stream.read(name))
                report=review_candidate.inspect_source(path,backend)
                blockers += [{'backend':backend,**f} for f in report['blockers']]
    return {'passed':not blockers,'blockers':blockers}

class StaticRulePatternRegression(unittest.TestCase):
    """These assert scanner behavior, not independent-agent capability."""

    def test_v21_contains_jit_branch_patterns_for_rule_regression(self) -> None:
        report = inspect_zip("flagos-task78-v21.zip")
        failures = {(item["backend"], item["kind"]) for item in report["blockers"]}
        self.assertFalse(report["passed"])
        self.assertIn(("iluvatar", "runtime-branch-in-jit"), failures)
        self.assertIn(("metax", "runtime-branch-in-jit"), failures)

    def test_v22_contains_known_target_risk_patterns_for_rule_regression(self) -> None:
        report = inspect_zip("flagos-task78-v22.zip")
        failures = {(item["backend"], item["kind"]) for item in report["blockers"]}
        self.assertFalse(report["passed"])
        self.assertIn(("enflame", "invalid-num-warps"), failures)
        self.assertIn(("hygon", "hygon-cast-before-broadcast"), failures)
        self.assertNotIn(("enflame", "uncapped-next-power-of-two"), failures)

class WorkflowHardGateRegression(unittest.TestCase):
    def inspect_source(self, source: str, filename: str = "concat_and_cast_mha_k.py") -> dict:
        with tempfile.TemporaryDirectory(prefix="task78-review-source-") as directory:
            path = Path(directory) / filename
            path.write_text(source, encoding="utf-8")
            return review_candidate.inspect_source(path, filename.removeprefix("concat_and_cast_mha_k").removesuffix(".py").lstrip("_") or "default")

    def test_rejects_unknown_config_abi_option(self) -> None:
        report = self.inspect_source(
            "triton.Config({'BT': 1}, num_warps=4, num_stages=1, multibuffer=True)\n",
            "concat_and_cast_mha_k_ascend.py",
        )
        self.assertIn("unknown-config-option", {item["kind"] for item in report["blockers"]})

    def test_rejects_explicit_tile_kwargs_duplicated_by_autotune_config(self) -> None:
        report = self.inspect_source(
            """
@triton.autotune(configs=[triton.Config({'BR': 128}, num_warps=4)], key=[])
@triton.jit
def _kernel(out, BR: tl.constexpr):
    pass

def concat_and_cast_mha_k(out):
    _kernel[(1,)](out, BR=128)
""",
            "concat_and_cast_mha_k_enflame.py",
        )
        self.assertIn("autotune-explicit-tile-constexpr", {item["kind"] for item in report["blockers"]})

    def test_allows_tile_constexpr_not_present_in_autotune_configs(self) -> None:
        report = self.inspect_source(
            """
@triton.autotune(configs=[triton.Config({'BM': 1}, num_warps=4)], key=[])
@triton.jit
def _kernel(out, BM: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr):
    pass

def concat_and_cast_mha_k(out):
    _kernel[(1,)](out, BN=128, BR=64)
""",
            "concat_and_cast_mha_k_ascend.py",
        )
        self.assertNotIn("autotune-explicit-tile-constexpr", {item["kind"] for item in report["blockers"]})


    def test_static_report_source_path_is_always_absolute(self) -> None:
        report = self.inspect_source("pass\n")
        self.assertEqual(report["path"], str(Path(report["path"]).resolve()))

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
        self.assertIn("autotune-tile-grid-mismatch", {item["kind"] for item in report["blockers"]})


class PackageTorchRuleRegression(unittest.TestCase):
    """One anti-cheat rule lives in two places; they had drifted.

    ``task111/test_task.py`` allows ``torch.tensor`` when its first argument is a
    literal list (the stride metadata buffer) and rejects anything computed from
    a tensor.  ``packaging.validate_package`` still carried the pre-tightening
    allowlist, which rejected every task111 candidate -- including ones the task
    itself permits -- so no task111 submission was possible at all.  These tests
    pin both halves of the rule so the copies cannot drift again.
    """

    OPERATOR = "conv_window_scatter_with_mask"
    CONTRACT = {
        "operator": OPERATOR,
        "entrypoint": "conv_window_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw)",
        "targets": ["iluvatar"],
        "package_members": ["conv_window_scatter_with_mask.py"],
    }

    def check(self, body: str):
        from competition.adaptation.packaging import make_package, validate_package

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            name = "conv_window_scatter_with_mask.py"
            (source / name).write_text(
                "import torch\n"
                "import triton\n"
                "import triton.language as tl\n\n"
                "@triton.jit\n"
                "def _k(ptr, N: tl.constexpr):\n"
                "    tl.store(ptr, tl.load(ptr) + N)\n\n"
                "def conv_window_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):\n"
                f"    {body}\n"
                "    return out\n"
            )
            archive = source / "package.zip"
            make_package(source, archive, self.CONTRACT)
            return validate_package(source, archive, self.CONTRACT)

    def test_literal_metadata_tensor_is_allowed(self) -> None:
        report = self.check(
            "out = torch.empty_like(dst)\n"
            "    tail = torch.tensor([dst.stride(0), src.stride(4)], dtype=torch.int32)"
        )
        self.assertTrue(report["passed"], report["errors"])

    def test_tensor_built_from_a_tensor_is_rejected(self) -> None:
        report = self.check("out = torch.empty_like(dst)\n    tail = torch.tensor(dst.stride())")
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("literal metadata buffer" in error for error in report["errors"]),
            report["errors"],
        )

    def test_other_torch_compute_is_still_rejected(self) -> None:
        report = self.check("out = torch.zeros(dst.shape)")
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("native torch compute call" in error for error in report["errors"]),
            report["errors"],
        )

    def test_scanner_only_sees_the_torch_namespace(self) -> None:
        """Known scope limit, asserted so it is visible rather than assumed away.

        The rule matches ``torch.<attr>(...)``; compute reached through a tensor
        *method* (``dst.clone()``, ``src.to(...)``) is not matched, which is why
        the reference-shaped implementation is caught by the structural-delta
        gate instead.  Tightening it is a deliberate separate change: it would
        also affect packages that legitimately call tensor methods.
        """
        report = self.check("out = dst.clone()")
        self.assertTrue(report["passed"], report["errors"])
