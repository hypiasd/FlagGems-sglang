"""Offline tests for the result-driven adjustment step.

Stdlib only.  The properties under test are the ones that make a result
actionable: the aggregate rule is verified rather than assumed, the priority
is ordered by marginal aggregate gain (so a low target with headroom outranks
a high target with none), a below-parity target is never excused by noise, and
the carry-forward plan names the file whose bytes produced the best value.

Run with ``python -m unittest competition.test_decide``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from competition import decide
from competition.experiments.store import digest

TARGETS = [
    "iluvatar",
    "metax",
    "hygon",
    "kunlunxin",
    "ascend",
    "intl_a",
    "intl_b",
]
OP = "recompute_w_u"


def record(record_id, submitted_at, scores, status="completed", sha=None):
    targets = {}
    for target in TARGETS:
        value = scores.get(target)
        item = {
            "status": "pass" if value is not None else "fail",
            "speedup": value,
        }
        if value is not None and sha:
            item["source_sha256"] = sha
        targets[target] = item
    passes = sum(1 for item in targets.values() if item["status"] == "pass")
    values = [
        item["speedup"]
        for item in targets.values()
        if item["status"] == "pass"
    ]
    aggregate = (
        sum(values) / len(values)
        if status == "completed" and passes == len(TARGETS)
        else None
    )
    return {
        "record_id": record_id,
        "submitted_at": submitted_at,
        "status": status,
        "excluded": False,
        "pass_count": passes,
        "aggregate_speedup": aggregate,
        "local_archive_sha256": "f" * 64,
        "targets": targets,
    }


class AggregateModelTest(unittest.TestCase):
    def test_mean_rule_is_verified_on_a_matching_record(self) -> None:
        model = decide.aggregate_model(
            [record("r1", "2026-10-01T10:00:00", {t: 4.0 for t in TARGETS})]
        )
        self.assertEqual(model["rule"], "arithmetic_mean_of_target_speedups")
        self.assertEqual(model["targets_per_record"], 7)
        self.assertIn("delta / targets_per_record", model["marginal_rule"])

    def test_rule_stays_unverified_when_a_record_disagrees(self) -> None:
        row = record("r1", "2026-10-01T10:00:00", {t: 4.0 for t in TARGETS})
        row["aggregate_speedup"] = 99.0
        model = decide.aggregate_model([row])
        self.assertEqual(model["rule"], "unverified")
        self.assertIn("do not rank", model["marginal_rule"])

    def test_provisional_observations_do_not_verify_the_rule(self) -> None:
        model = decide.aggregate_model(
            [
                record(
                    "r1",
                    "2026-10-01T10:00:00",
                    {t: 4.0 for t in TARGETS},
                    status="evaluating",
                )
            ]
        )
        self.assertEqual(model["rule"], "unverified")


def write_sources(local, adaptation_id, files):
    directory = Path(local) / "adaptations" / adaptation_id / "source"
    directory.mkdir(parents=True)
    (
        Path(local) / "adaptations" / adaptation_id / "adaptation.json"
    ).write_text(
        json.dumps({"task_id": "task103", "created_at": "2026-10-01T10:00:00"})
    )
    for name, body in files.items():
        (directory / name).write_text(body)


class DiagnoseTest(unittest.TestCase):
    def run_diagnose(self, records, files=None):
        with tempfile.TemporaryDirectory() as local:
            if files:
                write_sources(local, "adapt-a", files)
            return decide.diagnose("task103", local, records=records)

    def test_priority_orders_by_marginal_gain_not_by_value(self) -> None:
        scores = {t: 10.0 for t in TARGETS}
        scores["intl_a"] = 0.5
        scores["metax"] = 9.5
        result = self.run_diagnose(
            [record("r1", "2026-10-01T10:00:00", scores)]
        )
        self.assertEqual(result["priority"][0], "intl_a")
        self.assertIn("intl_a", result["rework_targets"])
        self.assertGreater(
            result["targets"]["intl_a"]["marginal_aggregate_gain_upper_bound"],
            result["targets"]["metax"]["marginal_aggregate_gain_upper_bound"],
        )

    def test_below_parity_is_not_excused_by_noise(self) -> None:
        first = {t: 4.0 for t in TARGETS}
        first["intl_a"] = 0.90
        second = {t: 4.0 for t in TARGETS}
        second["intl_a"] = 0.89
        result = self.run_diagnose(
            [
                record("r1", "2026-10-01T10:00:00", first, sha="a" * 64),
                record("r2", "2026-10-01T11:00:00", second, sha="a" * 64),
            ]
        )
        block = result["targets"]["intl_a"]
        self.assertTrue(block["below_baseline_parity"])
        self.assertEqual(block["action"], "rework_below_baseline")
        self.assertEqual(
            block["delta_vs_best_eligible"]["verdict"],
            "below_best_same_source",
        )

    def test_peer_median_gives_headroom_to_a_never_improved_target(
        self,
    ) -> None:
        scores = {t: 4.0 for t in TARGETS}
        scores["iluvatar"] = 0.2
        result = self.run_diagnose(
            [record("r1", "2026-10-01T10:00:00", scores)]
        )
        block = result["targets"]["iluvatar"]
        self.assertEqual(block["reference_kind"], "peer_median")
        self.assertAlmostEqual(block["headroom"], 3.8)
        self.assertEqual(block["action"], "rework_below_baseline")

    def test_provisional_observation_can_be_the_reference(self) -> None:
        done = {t: 4.0 for t in TARGETS}
        done["metax"] = 5.0
        pending = {t: 4.0 for t in TARGETS}
        pending["metax"] = 9.0
        result = self.run_diagnose(
            [
                record("r1", "2026-10-01T10:00:00", done),
                record(
                    "r2", "2026-10-01T11:00:00", pending, status="evaluating"
                ),
            ]
        )
        block = result["targets"]["metax"]
        self.assertEqual(block["reference_kind"], "provisional_observation")
        self.assertEqual(block["reference"], 9.0)
        self.assertEqual(block["action"], "rework_behind_reference")

    def test_a_target_at_the_reference_is_kept(self) -> None:
        scores = {t: 4.0 for t in TARGETS}
        result = self.run_diagnose(
            [record("r1", "2026-10-01T10:00:00", scores)]
        )
        for target in TARGETS:
            self.assertEqual(result["targets"][target]["action"], "keep")

    def test_missing_eligible_result_is_blocked(self) -> None:
        failed = {t: None for t in TARGETS}
        result = self.run_diagnose(
            [record("r1", "2026-10-01T10:00:00", failed, status="evaluating")]
        )
        self.assertTrue(
            all(
                block["action"] == "blocked_no_eligible_result"
                for block in result["targets"].values()
            )
        )
        self.assertEqual(result["rework_targets"], list(TARGETS))

    def test_carry_forward_names_the_file_behind_the_best_value(self) -> None:
        scores = {t: 4.0 for t in TARGETS}
        scores["kunlunxin"] = 1.5
        with tempfile.TemporaryDirectory() as local:
            write_sources(
                local,
                "adapt-a",
                {f"{OP}.py": "GENERIC", f"{OP}_kunlunxin.py": "KUNLUN"},
            )
            sha = digest(
                Path(local)
                / "adaptations"
                / "adapt-a"
                / "source"
                / f"{OP}_kunlunxin.py"
            )
            result = decide.diagnose(
                "task103",
                local,
                records=[record("r1", "2026-10-01T10:00:00", scores, sha=sha)],
            )
        plan = result["next_package"]["kunlunxin"]
        self.assertEqual(plan["carry_forward_file"], f"{OP}_kunlunxin.py")
        self.assertEqual(plan["from_adaptation"], "adapt-a")
        self.assertEqual(plan["carry_forward_sha256"], sha)

    def test_upper_bound_is_flagged_as_an_upper_bound(self) -> None:
        scores = {t: 4.0 for t in TARGETS}
        scores["intl_a"] = 0.5
        result = self.run_diagnose(
            [record("r1", "2026-10-01T10:00:00", scores)]
        )
        self.assertGreater(result["rework_aggregate_gain_upper_bound"], 0)
        self.assertIn("upper bounds", result["limitation"])
        self.assertIn("carry_forward_file", result["reuse_rule"])


if __name__ == "__main__":
    unittest.main()
