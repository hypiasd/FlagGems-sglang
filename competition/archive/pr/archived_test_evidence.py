"""Offline tests for the evidence discipline.

Stdlib only, no GPU, no torch and no network.  Run with
``python -m unittest competition.archive.pr.test_evidence``.
"""

from __future__ import annotations

import unittest

from . import evidence as ev
from . import gates

TARGETS = ["iluvatar", "kunlunxin", "intl_a"]


def _bundle(**overrides) -> dict:
    bundle = {
        "schema_version": 2,
        "run_id": "run-1",
        "target": {
            "backend": "_kunlunxin",
            "device": "R200",
            "compiler": "xkcc-1",
            "runtime": "xre-1",
        },
        "provenance": {
            "kind": "captured-live-tool-result",
            "provider": "rented-device",
            "invocation_id": "job-7",
            "raw_result_sha256": "a" * 64,
        },
        "source_sha256": "b" * 64,
        "baseline_source_sha256": "c" * 64,
        "compile": {"success": True, "log_ref": "local"},
        "correctness": {
            "suite_id": "suite-1",
            "total_cases": 2,
            "passed_cases": 2,
            "case_contract": {
                "case_set_id": "official-plus-boundary-v1",
                "required_case_ids": ["tail", "strided"],
            },
            "cases": [
                {
                    "case_id": "tail",
                    "input_signature": "x: shape=[33,65], dtype=bf16",
                    "status": "passed",
                },
                {
                    "case_id": "strided",
                    "input_signature": "x: shape=[32,64], dtype=bf16",
                    "status": "passed",
                },
            ],
        },
    }
    bundle.update(overrides)
    return bundle


def _benchmark(bundle: dict, samples: int = 5, coverage: str = "all") -> dict:
    signatures = {
        case["case_id"]: case["input_signature"]
        for case in bundle["correctness"]["cases"]
    }
    case_ids = list(signatures)
    if coverage == "subset":
        case_ids = case_ids[:1]
    cases = []
    for case_id in case_ids:
        cases.append(
            {
                "case_id": case_id,
                "input_signature": signatures[case_id],
                "baseline_ms": [0.9 + 0.01 * i for i in range(samples)],
                "candidate_ms": [0.45 + 0.01 * i for i in range(samples)],
            }
        )
    return {
        "method": "device-event",
        "warmup_runs": 10,
        "case_contract": {
            "case_set_id": "score-set-v1",
            "required_case_ids": list(signatures),
            "score_rule": "geometric mean of per-case medians",
        },
        "cases": cases,
    }


def _record(status: str, **overrides) -> dict:
    record = {
        "record_id": "task103-20261001T1944+0800-d3d16c3d",
        "status": status,
        "submitted_at": "2026-10-01T19:44:56+08:00",
        "local_archive_sha256": "d" * 64,
        "aggregate_speedup": 4.80 if status == "completed" else None,
        "targets": {
            name: {
                "status": "pass",
                "speedup": 1.85,
                "source_sha256": "e" * 64,
            }
            for name in TARGETS
        },
    }
    record.update(overrides)
    return record


class CapabilityTest(unittest.TestCase):
    def test_registry_is_the_only_proof(self) -> None:
        self.assertEqual(
            ev.classify_capability(in_tool_registry=True),
            ev.CALLABLE,
        )
        self.assertEqual(
            ev.classify_capability(
                in_tool_registry=False, config_present=True
            ),
            ev.CONFIGURED_UNAVAILABLE,
        )
        self.assertEqual(
            ev.classify_capability(in_tool_registry=False, reachable=True),
            ev.CONFIGURED_UNAVAILABLE,
        )
        self.assertEqual(
            ev.classify_capability(in_tool_registry=False), ev.ABSENT
        )


class PromotionTest(unittest.TestCase):
    def test_no_target_evidence_sources(self) -> None:
        for state in ev.NO_TARGET_EVIDENCE_FROM:
            for target in ev.TARGET_STATES:
                with self.assertRaises(ev.EvidenceError):
                    ev.promote(state, target, _bundle())

    def test_target_state_needs_a_bundle(self) -> None:
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.promote("locally_validated", "target_validated")
        self.assertIn("inconclusive", str(caught.exception))

    def test_target_validated_with_complete_evidence(self) -> None:
        receipt = ev.promote(
            "locally_validated", "target_validated", _bundle()
        )
        self.assertEqual(receipt["state"], "target_validated")
        self.assertIn("2 correctness case", receipt["correctness"])

    def test_partial_coverage_stays_inconclusive(self) -> None:
        bundle = _bundle()
        bundle["correctness"]["cases"] = bundle["correctness"]["cases"][:1]
        bundle["correctness"]["total_cases"] = 1
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.promote("locally_validated", "target_validated", bundle)
        self.assertIn("inconclusive", str(caught.exception))

    def test_failed_case_is_rejected(self) -> None:
        bundle = _bundle()
        bundle["correctness"]["cases"][1]["status"] = "failed"
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.promote("locally_validated", "target_validated", bundle)
        self.assertIn("rejected", str(caught.exception))

    def test_compile_failure_is_rejected(self) -> None:
        bundle = _bundle(compile={"success": False})
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.promote("locally_validated", "target_validated", bundle)
        self.assertIn("rejected", str(caught.exception))

    def test_missing_identity_stays_inconclusive(self) -> None:
        bundle = _bundle()
        bundle["target"] = {"backend": "_kunlunxin"}
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.promote("locally_validated", "target_validated", bundle)
        self.assertIn("inconclusive", str(caught.exception))

    def test_measured_needs_repeated_samples(self) -> None:
        bundle = _bundle()
        bundle["benchmark"] = _benchmark(bundle, samples=3)
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.promote("target_validated", "measured", bundle)
        self.assertIn("inconclusive", str(caught.exception))

    def test_measured_with_complete_coverage(self) -> None:
        bundle = _bundle()
        bundle["benchmark"] = _benchmark(bundle)
        receipt = ev.promote("target_validated", "measured", bundle)
        self.assertEqual(receipt["state"], "measured")
        self.assertAlmostEqual(
            receipt["benchmark"]["per_case"]["tail"]["speedup"],
            0.92 / 0.47,
        )
        self.assertFalse(receipt["official_score_computed"])

    def test_subset_never_measures(self) -> None:
        bundle = _bundle()
        bundle["benchmark"] = _benchmark(bundle, coverage="subset")
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.promote("target_validated", "measured", bundle)
        self.assertIn("inconclusive", str(caught.exception))

    def test_noisy_samples_are_rejected(self) -> None:
        bundle = _bundle()
        bundle["benchmark"] = _benchmark(bundle)
        bundle["benchmark"]["cases"][0]["candidate_ms"] = [
            0.45,
            0.45,
            0.45,
            0.45,
            0.95,
        ]
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.promote("target_validated", "measured", bundle)
        self.assertIn("rejected", str(caught.exception))

    def test_benchmark_case_must_match_correctness(self) -> None:
        bundle = _bundle()
        bundle["benchmark"] = _benchmark(bundle)
        bundle["benchmark"]["cases"][0]["input_signature"] = "other"
        with self.assertRaises(ev.EvidenceError) as caught:
            ev.promote("target_validated", "measured", bundle)
        self.assertIn("rejected", str(caught.exception))

    def test_illegal_transition(self) -> None:
        with self.assertRaises(ev.EvidenceError):
            ev.promote("prepared", "submitted")
        with self.assertRaises(ev.EvidenceError):
            ev.promote("nonsense", "prepared")

    def test_package_and_submit_path(self) -> None:
        self.assertEqual(
            ev.promote("measured", "submission_candidate")["state"],
            "submission_candidate",
        )
        self.assertEqual(
            ev.promote("submission_candidate", "submitted")["state"],
            "submitted",
        )


class LabelTest(unittest.TestCase):
    def test_labels(self) -> None:
        self.assertEqual(
            ev.performance_label(
                coverage="complete",
                raw_result_preserved=False,
                invocation_id=None,
            )["label"],
            "reported_only",
        )
        self.assertEqual(
            ev.performance_label(
                coverage="complete",
                raw_result_preserved=True,
                invocation_id="job-1",
            )["label"],
            "performance_reported_complete",
        )
        self.assertEqual(
            ev.performance_label(
                coverage="subset",
                raw_result_preserved=True,
                invocation_id="job-1",
            )["label"],
            "performance_reported_subset",
        )
        for coverage in ("complete", "subset", "incomplete"):
            labelled = ev.performance_label(
                coverage=coverage,
                raw_result_preserved=True,
                invocation_id="job-1",
            )
            self.assertFalse(labelled["official_score_computed"])


class LifecycleTest(unittest.TestCase):
    def test_unknown_status_fails_closed(self) -> None:
        with self.assertRaises(ev.EvidenceError):
            ev.submission_lifecycle({"status": "maybe"})

    def test_evaluating_keeps_the_aggregate_null(self) -> None:
        record = _record("evaluating", aggregate_speedup=2.43)
        derived = ev.submission_lifecycle(record)
        self.assertTrue(derived["provisional"])
        self.assertTrue(derived["errors"])

    def test_evaluating_with_pending_target(self) -> None:
        record = _record("evaluating")
        record["targets"]["iluvatar"] = {"status": "pending"}
        derived = ev.submission_lifecycle(record)
        self.assertEqual(derived["stage"], "evaluating")
        self.assertEqual(derived["pending_targets"], ["iluvatar"])
        self.assertEqual(derived["errors"], [])

    def test_completed_requires_terminal_targets(self) -> None:
        record = _record("completed")
        record["targets"]["intl_a"] = {"status": "pending"}
        derived = ev.submission_lifecycle(record)
        self.assertTrue(derived["errors"])

    def test_terminal_record(self) -> None:
        derived = ev.submission_lifecycle(_record("completed"))
        self.assertEqual(derived["stage"], "completed")
        self.assertFalse(derived["provisional"])
        self.assertEqual(derived["errors"], [])


class RunRecordTest(unittest.TestCase):
    def test_template_is_null_and_reports_gaps(self) -> None:
        template = ev.run_record_template()
        self.assertEqual(set(template), set(ev.RUN_RECORD_FIELDS))
        self.assertTrue(all(value is None for value in template.values()))
        checked = ev.run_record_errors(template)
        self.assertEqual(
            sorted(checked["missing_required"]), sorted(ev.RUN_RECORD_REQUIRED)
        )
        self.assertIn(
            "evidence_state must be one of",
            " ".join(checked["errors"]),
        )

    def test_unknown_field_rejected(self) -> None:
        with self.assertRaises(ev.EvidenceError):
            ev.run_record_template(invented="x")

    def test_tool_name_requires_a_callable_service(self) -> None:
        record = ev.run_record_template(
            schema_version=1,
            run_id="run-1",
            operator="recompute_w_u",
            mode="optimize",
            service_state=ev.CONFIGURED_UNAVAILABLE,
            started_at="2026-10-01T20:00:00+08:00",
            contract_hash="a" * 64,
            candidate_source_hash="b" * 64,
            local_correctness="pass",
            evidence_state="locally_validated",
            decision="keep_candidate",
            reason="structure changed",
            tool_name="optimize_kernel",
        )
        checked = ev.run_record_errors(record)
        self.assertTrue(checked["errors"])
        self.assertIn(
            "config file is not evidence", " ".join(checked["errors"])
        )

    def test_retry_gets_a_new_run_id(self) -> None:
        record = ev.run_record_template(
            schema_version=1,
            run_id="run-1",
            parent_run_id="run-1",
            operator="recompute_w_u",
            mode="optimize",
            service_state=ev.CALLABLE,
            tool_name="optimize_kernel",
            started_at="2026-10-01T20:00:00+08:00",
            contract_hash="a" * 64,
            candidate_source_hash="b" * 64,
            local_correctness="pass",
            evidence_state="generated",
            decision="keep_candidate",
            reason="retry",
        )
        checked = ev.run_record_errors(record)
        self.assertEqual(checked["missing_required"], [])
        self.assertIn("self", " ".join(checked["errors"]))


def _receipt(stage: str, agent: str, **overrides) -> dict:
    receipt = {
        "stage": stage,
        "reviewer_agent_id": agent,
        "review_provenance": "platform-recorded",
        "candidate_source_hash": "b" * 64,
        "baseline_source_hash": "c" * 64,
    }
    receipt.update(overrides)
    return receipt


class ReviewReceiptTest(unittest.TestCase):
    def test_consistent_pair(self) -> None:
        stage_a = _receipt(
            "A",
            "agent-a",
            coverage={"kunlunxin": {"trace": "lines 40-88: mask widened"}},
            findings=[{"id": "f1", "severity": "high"}],
        )
        stage_b = _receipt("B", "agent-b", dispositions={"f1": "fixed"})
        checked = ev.review_receipt_errors(stage_a, stage_b)
        self.assertEqual(checked["errors"], [])
        self.assertEqual(checked["gaps"], [])
        self.assertFalse(checked["authenticated"])

    def test_same_reviewer_is_a_contradiction(self) -> None:
        stage_a = _receipt("A", "agent-a", coverage={"k": {"trace": "x"}})
        stage_b = _receipt("B", "agent-a")
        checked = ev.review_receipt_errors(stage_a, stage_b)
        self.assertIn("differ", " ".join(checked["errors"]))

    def test_missing_agent_ids_are_a_gap(self) -> None:
        stage_a = _receipt("A", "", coverage={"k": {"trace": "x"}})
        stage_b = _receipt("B", "")
        checked = ev.review_receipt_errors(stage_a, stage_b)
        self.assertEqual(checked["errors"], [])
        self.assertTrue(checked["gaps"])

    def test_hash_mismatch_is_a_contradiction(self) -> None:
        stage_a = _receipt("A", "a", coverage={"k": {"trace": "x"}})
        stage_b = _receipt("B", "b", candidate_source_hash="f" * 64)
        checked = ev.review_receipt_errors(stage_a, stage_b)
        self.assertIn("different candidate bytes", " ".join(checked["errors"]))

    def test_every_finding_needs_a_disposition(self) -> None:
        stage_a = _receipt(
            "A",
            "a",
            coverage={"k": {"trace": "x"}},
            findings=[{"id": "f1"}, {"id": "f2"}],
        )
        stage_b = _receipt("B", "b", dispositions={"f1": "fixed"})
        checked = ev.review_receipt_errors(stage_a, stage_b)
        self.assertIn("f2", " ".join(checked["errors"]))

    def test_labelled_coverage_is_not_a_trace(self) -> None:
        stage_a = _receipt("A", "a", coverage={"k": {"label": "checked"}})
        stage_b = _receipt("B", "b")
        checked = ev.review_receipt_errors(stage_a, stage_b)
        self.assertIn("trace", " ".join(checked["errors"]))

    def test_unverified_receipt_is_labelled(self) -> None:
        stage_a = _receipt(
            "A",
            "a",
            review_provenance="unverified_claim",
            coverage={"k": {"trace": "x"}},
        )
        stage_b = _receipt("B", "b")
        checked = ev.review_receipt_errors(stage_a, stage_b)
        self.assertTrue(any("unverified" in note for note in checked["notes"]))


class ReportTest(unittest.TestCase):
    def _release(self) -> dict:
        return {
            "targets": {name: {"source_sha256": "e" * 64} for name in TARGETS},
            "full_correctness": {"status": "blocked-device-unavailable"},
            "reviews": [],
        }

    def test_terminal_record_promotes_only_matching_sources(self) -> None:
        report = ev.derive_report(
            task_id="task103",
            adaptation_id="adapt-x",
            profile_targets=TARGETS,
            release=self._release(),
            records=[_record("completed")],
            package_sha256="d" * 64,
            routes=[
                {"name": "official_platform", "in_tool_registry": True},
                {"name": "rented_device", "config_present": True},
            ],
            local_gates={"ready_for_pr": True},
        )
        self.assertEqual(
            report["capability"]["official_platform"], ev.CALLABLE
        )
        self.assertEqual(
            report["capability"]["rented_device"], ev.CONFIGURED_UNAVAILABLE
        )
        for name in TARGETS:
            self.assertEqual(
                report["targets"][name]["state"], "target_validated"
            )
            self.assertEqual(
                report["targets"][name]["performance"]["label"],
                "performance_reported_complete",
            )
        self.assertEqual(report["missing_target_evidence"], [])
        self.assertEqual(report["official"]["best_aggregate_speedup"], 4.80)
        self.assertFalse(report["official"]["official_score_computed"])
        self.assertEqual(report["package_state"], "target_validated")

    def test_provisional_record_never_promotes(self) -> None:
        provisional = _record("evaluating")
        provisional["targets"]["kunlunxin"]["source_sha256"] = "e" * 64
        report = ev.derive_report(
            task_id="task103",
            adaptation_id="adapt-x",
            profile_targets=TARGETS,
            release=self._release(),
            records=[provisional],
            package_sha256="d" * 64,
            routes=[{"name": "official_platform", "in_tool_registry": True}],
            local_gates={"ready_for_pr": True},
        )
        for name in TARGETS:
            self.assertEqual(report["targets"][name]["state"], "inconclusive")
        self.assertEqual(
            report["official"]["provisional_records"][0]["record_id"],
            provisional["record_id"],
        )
        self.assertEqual(report["package_state"], "submission_candidate")

    def test_interim_row_is_superseded_not_provisional(self) -> None:
        interim = _record("evaluating")
        terminal = _record("completed")
        report = ev.derive_report(
            task_id="task103",
            adaptation_id="adapt-x",
            profile_targets=TARGETS,
            release=self._release(),
            records=[interim, terminal],
            package_sha256="d" * 64,
            routes=[{"name": "official_platform", "in_tool_registry": True}],
            local_gates={"ready_for_pr": True},
        )
        self.assertEqual(report["official"]["provisional_records"], [])
        self.assertEqual(len(report["official"]["superseded_records"]), 1)
        self.assertEqual(
            report["official"]["superseded_records"][0]["values"]["kunlunxin"],
            1.85,
        )
        self.assertEqual(
            report["targets"]["kunlunxin"]["state"], "target_validated"
        )
        self.assertTrue(any("superseded" in note for note in report["notes"]))

    def test_recorded_reviews_are_checked(self) -> None:
        release = self._release()
        release["reviews"] = [
            _receipt("A", "a", coverage={"k": {"trace": "x"}}),
            _receipt("B", "a"),
        ]
        report = ev.derive_report(
            task_id="task103",
            adaptation_id="adapt-x",
            profile_targets=TARGETS,
            release=release,
            records=[],
            package_sha256="d" * 64,
        )
        self.assertFalse(report["review"]["verified"])


class EvidenceGateTest(unittest.TestCase):
    def _report(self, **overrides) -> dict:
        report = {
            "capability": {"official_platform": ev.CALLABLE},
            "package_state": "target_validated",
            "targets": {},
            "best_per_target": {},
            "missing_target_evidence": [],
            "official": {
                "lifecycle": [],
                "provisional_records": [],
                "best_aggregate_speedup": None,
                "best_aggregate_record_id": None,
                "official_score_computed": False,
            },
            "review": {"errors": [], "gaps": [], "verified": True},
        }
        report.update(overrides)
        return report

    def test_clean_report_passes(self) -> None:
        gate = gates.evidence_gate(self._report())
        self.assertEqual(gate["status"], gates.PASS)
        self.assertFalse(gate["official_score_computed"])

    def test_contradiction_fails(self) -> None:
        report = self._report(
            official={
                "lifecycle": [
                    {"record_id": "r1", "errors": ["aggregate not null"]}
                ],
                "provisional_records": [],
                "official_score_computed": False,
            }
        )
        gate = gates.evidence_gate(report)
        self.assertEqual(gate["status"], gates.FAIL)

    def test_missing_target_evidence_is_unavailable(self) -> None:
        report = self._report(
            capability={"rented_device": ev.CONFIGURED_UNAVAILABLE},
            missing_target_evidence=["kunlunxin"],
            targets={"kunlunxin": {"basis": "no target execution evidence"}},
            package_state="submission_candidate",
        )
        gate = gates.evidence_gate(report)
        self.assertEqual(gate["status"], gates.UNAVAILABLE)
        self.assertIn("kunlunxin", gate["detail"])


if __name__ == "__main__":
    unittest.main()
