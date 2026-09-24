#!/usr/bin/env python3
"""Static-rule regressions and two-stage reviewer-gate protocol tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
            return review_candidate.inspect_source(path)

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

    def test_same_line_power_of_two_findings_have_distinct_source_identity(self) -> None:
        source = (
            "bn, br = triton.next_power_of_2(max(1, dn)), "
            "triton.next_power_of_2(max(1, dr))\n"
        )
        report = self.inspect_source(
            source
        )
        findings = [item for item in report["blockers"]
                    if item["kind"] == "uncapped-next-power-of-two"]
        ids = {
            run_candidate_gate.canonical_json_sha256({"target_id": "default", **item})
            for item in findings
        }
        self.assertEqual(len(findings), 2)
        self.assertEqual(len(ids), 2)
        expected_columns = {
            source.index("triton.next_power_of_2(max(1, dn))") + 1,
            source.index("triton.next_power_of_2(max(1, dr))") + 1,
        }
        self.assertEqual({item["column"] for item in findings}, expected_columns)
        self.assertEqual({item["expression"] for item in findings}, {
            "triton.next_power_of_2(max(1, dn))",
            "triton.next_power_of_2(max(1, dr))",
        })

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


class TwoStageReviewProtocolTests(unittest.TestCase):
    hashes = {"default": "candidate-digest"}
    baseline_hashes = {"default": "baseline-digest"}
    static = {
        "backends": {"default": {"findings": [{
            "severity": "warning", "kind": "power-of-two-bound", "line": 4,
            "message": "check bound",
        }]}}
    }

    def blind_receipt(self, **overrides):
        receipt = {
            "protocol_version": 3,
            "review_mode": "blind-independent",
            "reviewer_agent_id": "agent-A",
            "reviewer_name": "reviewer A",
            "candidate": "candidate",
            "baseline": "baseline",
            "reviewed_source_sha256": self.hashes,
            "reviewed_baseline_sha256": self.baseline_hashes,
            "backend_coverage": {"default": {
                "status": "complete", "checks_run": ["dataflow", "bounds"],
                "analysis_summary": ["rows map to independent output intervals"],
                "evidence": ["kernel.py:1-4"],
            }},
            "findings": [],
            "verdict": "pass",
            "limitations": ["target compiler not available"],
        }
        receipt.update(overrides)
        return receipt

    def test_protocol_requires_blind_and_distinct_reconciliation_receipts(self):
        with tempfile.TemporaryDirectory(prefix="task78-two-stage-") as directory:
            root = Path(directory)
            blind_path = root / "blind.json"
            final_path = root / "reconciliation.json"
            blind_path.write_text(json.dumps(self.blind_receipt()), encoding="utf-8")
            blind_digest = hashlib.sha256(blind_path.read_bytes()).hexdigest()
            static_id = "static:" + run_candidate_gate.canonical_json_sha256(
                {"target_id": "default", **self.static["backends"]["default"]["findings"][0]}
            )
            final = {
                "protocol_version": 3,
                "review_mode": "independent-reconciliation",
                "reviewer_agent_id": "agent-B",
                "reviewer_name": "reviewer B",
                "candidate": "candidate",
                "baseline": "baseline",
                "reviewed_source_sha256": self.hashes,
                "reviewed_baseline_sha256": self.baseline_hashes,
                "blind_receipt_sha256": blind_digest,
                "static_report_sha256": run_candidate_gate.canonical_json_sha256(self.static),
                "dispositions": [{
                    "finding_id": static_id, "decision": "residual",
                    "rationale": "tile is capped by the wrapper at the call site",
                }],
                "additional_findings": [],
                "verdict": "pass",
            }
            final_path.write_text(json.dumps(final), encoding="utf-8")
            result = run_candidate_gate.review_check(
                blind_path, final_path, self.static, True, self.hashes,
                self.baseline_hashes, "candidate", "baseline",
            )
        self.assertTrue(result["passed"], result)
        self.assertEqual(len(result["residuals"]), 1)

    def test_unresolved_or_missing_static_disposition_fails(self):
        with tempfile.TemporaryDirectory(prefix="task78-two-stage-") as directory:
            root = Path(directory)
            blind_path = root / "blind.json"
            final_path = root / "reconciliation.json"
            blind_path.write_text(json.dumps(self.blind_receipt()), encoding="utf-8")
            final = {
                "protocol_version": 3,
                "review_mode": "independent-reconciliation",
                "reviewer_agent_id": "agent-B",
                "reviewer_name": "reviewer B",
                "candidate": "candidate",
                "baseline": "baseline",
                "reviewed_source_sha256": self.hashes,
                "reviewed_baseline_sha256": self.baseline_hashes,
                "blind_receipt_sha256": hashlib.sha256(blind_path.read_bytes()).hexdigest(),
                "static_report_sha256": run_candidate_gate.canonical_json_sha256(self.static),
                "dispositions": [],
                "additional_findings": [],
                "verdict": "pass",
            }
            final_path.write_text(json.dumps(final), encoding="utf-8")
            result = run_candidate_gate.review_check(
                blind_path, final_path, self.static, True, self.hashes,
                self.baseline_hashes, "candidate", "baseline",
            )
        self.assertFalse(result["passed"])
        self.assertFalse(result["reconciliation_valid"])

    def test_unknown_backend_coverage_cannot_pass_complete_reconciliation(self):
        with tempfile.TemporaryDirectory(prefix="task78-two-stage-") as directory:
            root = Path(directory)
            blind_path = root / "blind.json"
            blind = self.blind_receipt(backend_coverage={"default": {
                "status": "unknown", "checks_run": ["dataflow"],
                "analysis_summary": ["not enough context"],
                "evidence": ["kernel.py:1"],
            }})
            blind_path.write_text(json.dumps(blind), encoding="utf-8")
            static_finding = self.static["backends"]["default"]["findings"][0]
            static_id = "static:" + run_candidate_gate.canonical_json_sha256(
                {"target_id": "default", **static_finding}
            )
            final_path = root / "reconciliation.json"
            final_path.write_text(json.dumps({
                "protocol_version": 3,
                "review_mode": "independent-reconciliation",
                "reviewer_agent_id": "agent-B",
                "reviewer_name": "reviewer B",
                "candidate": "candidate",
                "baseline": "baseline",
                "reviewed_source_sha256": self.hashes,
                "reviewed_baseline_sha256": self.baseline_hashes,
                "blind_receipt_sha256": hashlib.sha256(blind_path.read_bytes()).hexdigest(),
                "static_report_sha256": run_candidate_gate.canonical_json_sha256(self.static),
                "dispositions": [{
                    "finding_id": static_id, "decision": "residual",
                    "rationale": "warning was inspected against the wrapper bound",
                }],
                "additional_findings": [],
                "verdict": "pass",
            }), encoding="utf-8")
            result = run_candidate_gate.review_check(
                blind_path, final_path, self.static, True, self.hashes,
                self.baseline_hashes, "candidate", "baseline",
            )
        self.assertFalse(result["passed"])
        self.assertFalse(result["blind_review_valid"])

    def test_old_single_receipt_cannot_pass(self):
        with tempfile.TemporaryDirectory(prefix="task78-two-stage-") as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({
                "review_type": "read-only-subagent",
                "review_mode": "adversarial-read-only",
                "searched_for_novel_risks": True,
                "blockers": [],
            }), encoding="utf-8")
            result = run_candidate_gate.review_check(
                path, None, self.static, True, self.hashes,
                self.baseline_hashes, "candidate", "baseline",
            )
        self.assertFalse(result["passed"])

    def test_blocker_in_blind_receipt_cannot_be_softened(self):
        with tempfile.TemporaryDirectory(prefix="task78-two-stage-") as directory:
            root = Path(directory)
            blind_path = root / "blind.json"
            blind = self.blind_receipt(findings=[{
                "id": "A-1", "target_id": "default", "severity": "blocker",
                "location": "kernel.py:10", "mechanism": "bad bounds",
                "activation": "tail shape", "confidence": "high", "evidence": ["line 10"],
            }])
            blind_path.write_text(json.dumps(blind), encoding="utf-8")
            final_path = root / "reconciliation.json"
            final_path.write_text("{}", encoding="utf-8")
            result = run_candidate_gate.review_check(
                blind_path, final_path, self.static, True, self.hashes,
                self.baseline_hashes, "candidate", "baseline",
            )
        self.assertFalse(result["passed"])
        self.assertFalse(result["blind_review_valid"])


if __name__ == "__main__":
    unittest.main()
