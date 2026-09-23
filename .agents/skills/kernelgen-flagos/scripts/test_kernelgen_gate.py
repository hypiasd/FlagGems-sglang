import hashlib
import unittest

from kernelgen_gate import gate


CODE = "def public_op(x):\n    return x\n"
BASELINE = "def public_op(x):\n    return x + 1\n"


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def response(code=CODE):
    return {"triton_code": code, "success": True}


def evidence(candidate=CODE, baseline=BASELINE):
    signature = "x: shape=[32, 64], dtype=float16, strides=[64, 1]"
    return {
        "schema_version": 1,
        "run_id": "run-001",
        "target": {
            "backend": "test-backend",
            "device": "test-device",
            "compiler": "compiler-1",
            "runtime": "runtime-1",
        },
        "source_sha256": sha256(candidate),
        "baseline_source_sha256": sha256(baseline),
        "compile": {"success": True},
        "correctness": {
            "suite_id": "suite-001",
            "total_cases": 1,
            "passed_cases": 1,
            "cases": [{
                "case_id": "case-001",
                "input_signature": signature,
                "status": "passed",
            }],
        },
        "benchmark": {
            "method": "device-event",
            "warmup_runs": 10,
            "case_contract": {
                "case_set_id": "suite-001-cases-v1",
                "required_case_ids": ["case-001"],
                "score_rule": "geometric mean of per-case baseline/candidate medians",
            },
            "cases": [{
                "case_id": "case-001",
                "input_signature": signature,
                "baseline_ms": [2.0, 2.0, 2.0, 2.0, 2.0],
                "candidate_ms": [1.0, 1.0, 1.0, 1.0, 1.0],
            }],
        },
    }


class KernelGenGateTests(unittest.TestCase):
    def call_gate(self, phase="target", manifest=None, code=CODE,
                  candidate_source=CODE, baseline_source=BASELINE, **kwargs):
        return gate(
            response(code), phase, "public_op", manifest,
            candidate_source, baseline_source, **kwargs,
        )

    def test_candidate_artifact_is_only_generated(self):
        result = self.call_gate(phase="candidate", manifest=None)
        self.assertEqual(result["state"], "generated")

    def test_missing_target_manifest_is_inconclusive(self):
        result = self.call_gate(manifest=None)
        self.assertEqual(result["state"], "inconclusive")

    def test_exact_source_and_repeated_target_evidence_is_reported_only(self):
        result = self.call_gate(manifest=evidence())
        self.assertEqual(result["state"], "performance_reported_complete")
        self.assertAlmostEqual(result["diagnostic_geometric_mean_speedup"], 2.0)
        self.assertFalse(result["official_score_computed"])
        self.assertEqual(result["evidence_provenance"], "unverified_claim")
        self.assertTrue(result["task_level_case_coverage_complete"])

    def test_evidence_for_different_source_is_rejected(self):
        manifest = evidence()
        manifest["source_sha256"] = sha256("def public_op(y):\n    return y\n")
        result = self.call_gate(manifest=manifest)
        self.assertEqual(result["state"], "rejected")

    def test_unknown_manifest_schema_is_inconclusive(self):
        manifest = evidence()
        manifest["schema_version"] = 2
        result = self.call_gate(manifest=manifest)
        self.assertEqual(result["state"], "inconclusive")

    def test_failed_target_case_is_rejected(self):
        manifest = evidence()
        manifest["correctness"]["passed_cases"] = 0
        manifest["correctness"]["cases"][0]["status"] = "failed"
        result = self.call_gate(manifest=manifest)
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(result["correctness_state"], "failed")

    def test_too_few_timing_samples_does_not_claim_performance(self):
        manifest = evidence()
        manifest["benchmark"]["cases"][0]["baseline_ms"] = [2.0]
        manifest["benchmark"]["cases"][0]["candidate_ms"] = [1.0]
        result = self.call_gate(manifest=manifest)
        self.assertEqual(result["state"], "target_reported")
        self.assertEqual(result["performance_state"], "reported_incomplete")

    def test_benchmark_signature_must_match_correctness_signature(self):
        manifest = evidence()
        manifest["benchmark"]["cases"][0]["input_signature"] = "wrong shape"
        result = self.call_gate(manifest=manifest)
        self.assertEqual(result["state"], "target_reported")
        self.assertEqual(result["performance_state"], "reported_incomplete")

    def test_slow_candidate_is_measured_but_not_improvement(self):
        manifest = evidence()
        manifest["benchmark"]["cases"][0]["candidate_ms"] = [3.0] * 5
        result = self.call_gate(manifest=manifest)
        self.assertEqual(result["state"], "performance_reported_no_improvement")

    def test_noisy_samples_are_retained_but_flagged(self):
        manifest = evidence()
        manifest["benchmark"]["cases"][0]["candidate_ms"] = [1.0, 1.0, 1.0, 1.0, 4.0]
        result = self.call_gate(manifest=manifest)
        self.assertEqual(result["state"], "performance_reported_unstable")

    def test_unbound_case_subset_is_not_task_level_speedup(self):
        manifest = evidence()
        del manifest["benchmark"]["case_contract"]
        result = self.call_gate(manifest=manifest)
        self.assertEqual(result["state"], "performance_reported_subset")
        self.assertFalse(result["task_level_case_coverage_complete"])
        self.assertIn("diagnostic subset", result["reasons"][0])

    def test_raw_tool_result_hash_is_linked_but_not_authentication(self):
        manifest = evidence()
        raw = b'{"run":"real-or-not"}'
        import hashlib
        manifest["provenance"] = {
            "kind": "captured-live-tool-result",
            "provider": "example-runner",
            "invocation_id": "call-123",
            "raw_result_sha256": hashlib.sha256(raw).hexdigest(),
        }
        result = self.call_gate(manifest=manifest, raw_result_sha256=hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["evidence_provenance"], "raw_result_linked")
        self.assertEqual(result["state"], "performance_reported_complete")

    def test_wrong_raw_tool_result_hash_remains_unverified_claim(self):
        manifest = evidence()
        manifest["provenance"] = {
            "kind": "captured-live-tool-result",
            "provider": "example-runner",
            "invocation_id": "call-123",
            "raw_result_sha256": "wrong",
        }
        result = self.call_gate(manifest=manifest, raw_result_sha256="other")
        self.assertEqual(result["evidence_provenance"], "unverified_claim")


if __name__ == "__main__":
    unittest.main()
