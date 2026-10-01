"""Offline tests for the cross-package per-target ledger.

Stdlib only: synthetic official records, no network and no device.  The
semantics under test are (a) a per-target best table that is independent of
the aggregate best, and (b) a measurement-resolution line that separates a
real version regression from re-measuring byte-identical source.

Run with ``python -m unittest competition.adaptation.test_ledger``.
"""

from __future__ import annotations

import unittest

from competition.adaptation import ledger

TARGETS = [
    "iluvatar",
    "metax",
    "hygon",
    "kunlunxin",
    "ascend",
    "intl_a",
    "intl_b",
]


def record(record_id, submitted_at, scores, sha="a" * 64, aggregate=1.0):
    """Build one official observation; ``scores`` maps target to speedup/None."""
    targets = {}
    for target in TARGETS:
        value = scores.get(target)
        if value is None:
            targets[target] = {"status": "fail", "speedup": None}
        else:
            targets[target] = {
                "status": "pass",
                "speedup": value,
                "source_sha256": sha,
            }
    passes = sum(1 for item in targets.values() if item["status"] == "pass")
    return {
        "record_id": record_id,
        "submitted_at": submitted_at,
        "status": "completed",
        "pass_count": passes,
        "aggregate_speedup": aggregate if passes == len(TARGETS) else None,
        "local_archive_sha256": "f" * 64,
        "targets": targets,
    }


def full(**overrides):
    scores = {target: 1.0 for target in TARGETS}
    scores.update(overrides)
    return scores


class BestTableTest(unittest.TestCase):
    def test_best_is_per_target_not_per_aggregate(self) -> None:
        records = [
            record(
                "r1",
                "2026-10-01T10:00:00+08:00",
                full(metax=5.01),
                aggregate=4.4,
            ),
            record(
                "r2",
                "2026-10-01T11:00:00+08:00",
                full(metax=4.98),
                aggregate=4.8,
            ),
        ]
        table = ledger.target_ledger("task103", records)["targets"]
        self.assertEqual(table["metax"]["best_eligible"]["speedup"], 5.01)
        self.assertEqual(table["metax"]["best_eligible"]["record_id"], "r1")
        self.assertEqual(table["metax"]["latest"]["speedup"], 4.98)

    def test_ineligible_package_is_not_the_eligible_best(self) -> None:
        strict = full()
        strict["kunlunxin"] = None
        records = [
            record("r1", "2026-10-01T10:00:00+08:00", strict, aggregate=None),
            record(
                "r2",
                "2026-10-01T11:00:00+08:00",
                full(metax=4.98),
                aggregate=4.8,
            ),
        ]
        table = ledger.target_ledger("task103", records)["targets"]["metax"]
        self.assertEqual(table["best_eligible"]["record_id"], "r2")
        self.assertEqual(table["best_observed"]["record_id"], "r2")

    def test_excluded_and_unfinished_records_are_ignored(self) -> None:
        evaluating = record("r1", "2026-10-01T10:00:00+08:00", full(metax=9.0))
        evaluating["status"] = "evaluating"
        evaluating["aggregate_speedup"] = None
        excluded = record("r2", "2026-10-01T11:00:00+08:00", full(metax=8.0))
        excluded["excluded"] = True
        records = [
            evaluating,
            excluded,
            record("r3", "2026-10-01T12:00:00+08:00", full(metax=4.0)),
        ]
        table = ledger.target_ledger("task103", records)["targets"]
        self.assertEqual(table["metax"]["best_observed"]["speedup"], 4.0)
        self.assertEqual(table["metax"]["best_observed"]["record_id"], "r3")

    def test_revision_keeps_the_last_one(self) -> None:
        interim = record(
            "r1", "2026-10-01T10:00:00+08:00", full(metax=4.0), aggregate=4.0
        )
        interim["revision"] = 1
        final = record(
            "r1", "2026-10-01T10:00:00+08:00", full(metax=4.9), aggregate=4.9
        )
        final["revision"] = 2
        table = ledger.target_ledger("task103", [interim, final])["targets"]
        self.assertEqual(table["metax"]["best_observed"]["speedup"], 4.9)


class ResolutionTest(unittest.TestCase):
    def test_same_source_spread_becomes_the_resolution(self) -> None:
        records = [
            record("r1", "2026-10-01T10:00:00+08:00", full(ascend=2.98)),
            record("r2", "2026-10-01T11:00:00+08:00", full(ascend=3.44)),
        ]
        table = ledger.target_ledger("task103", records)["targets"]["ascend"]
        self.assertEqual(table["source_versions"], 1)
        self.assertGreater(table["resolution"], 0.13)
        self.assertEqual(table["repeated_observations"][0]["samples"], 2)

    def test_single_observation_has_no_resolution(self) -> None:
        table = ledger.target_ledger(
            "task103", [record("r1", "2026-10-01T10:00:00+08:00", full())]
        )["targets"]["metax"]
        self.assertIsNone(table["resolution"])
        self.assertIn("no repeated observation", table["resolution_basis"])

    def test_small_delta_on_identical_source_is_re_measurement(self) -> None:
        records = [
            record("r1", "2026-10-01T10:00:00+08:00", full(metax=5.01)),
            record("r2", "2026-10-01T11:00:00+08:00", full(metax=5.00)),
            record("r3", "2026-10-01T12:00:00+08:00", full(metax=4.98)),
        ]
        delta = ledger.target_ledger("task103", records)["targets"]["metax"][
            "delta_vs_best_eligible"
        ]
        self.assertTrue(delta["same_source"])
        self.assertEqual(delta["verdict"], "below_best_same_source")
        self.assertIsNotNone(delta["observed_spread_for_latest_source"])

    def test_changed_source_regression_is_not_re_measurement(self) -> None:
        records = [
            record(
                "r1",
                "2026-10-01T10:00:00+08:00",
                full(kunlunxin=2.43),
                sha="b" * 64,
            ),
            record(
                "r2",
                "2026-10-01T11:00:00+08:00",
                full(kunlunxin=0.09),
                sha="c" * 64,
            ),
        ]
        delta = ledger.target_ledger("task103", records)["targets"][
            "kunlunxin"
        ]["delta_vs_best_eligible"]
        self.assertFalse(delta["same_source"])
        self.assertEqual(delta["verdict"], "below_best_changed_source")
        self.assertIsNone(delta["observed_spread_for_latest_source"])

    def test_spread_is_reported_per_source(self) -> None:
        records = [
            record(
                "r1",
                "2026-10-01T10:00:00+08:00",
                full(ascend=2.0),
                sha="d" * 64,
            ),
            record(
                "r2",
                "2026-10-01T11:00:00+08:00",
                full(ascend=3.44),
                sha="a" * 64,
            ),
            record(
                "r3",
                "2026-10-01T12:00:00+08:00",
                full(ascend=2.98),
                sha="a" * 64,
            ),
        ]
        table = ledger.target_ledger("task103", records)["targets"]["ascend"]
        self.assertEqual(table["source_versions"], 2)
        digests = {
            entry["source_sha256"] for entry in table["repeated_observations"]
        }
        self.assertEqual(digests, {"a" * 64})
        delta = table["delta_vs_best_eligible"]
        self.assertTrue(delta["same_source"])
        self.assertGreater(delta["observed_spread_for_latest_source"], 0.13)
        self.assertLess(delta["observed_spread_for_latest_source"], 0.15)

    def test_at_best_reports_no_regression(self) -> None:
        records = [
            record("r1", "2026-10-01T10:00:00+08:00", full(hygon=18.0)),
            record("r2", "2026-10-01T11:00:00+08:00", full(hygon=18.46)),
        ]
        delta = ledger.target_ledger("task103", records)["targets"]["hygon"][
            "delta_vs_best_eligible"
        ]
        self.assertEqual(delta["verdict"], "at_or_above_best")
        self.assertLessEqual(delta["absolute"], 0)


class ShapeTest(unittest.TestCase):
    def test_eligible_records_list_only_full_passes(self) -> None:
        strict = full()
        strict["kunlunxin"] = None
        records = [
            record("r1", "2026-10-01T10:00:00+08:00", strict, aggregate=None),
            record("r2", "2026-10-01T11:00:00+08:00", full(), aggregate=4.8),
        ]
        result = ledger.target_ledger("task103", records)
        self.assertEqual(result["eligible_records"], ["r2"])
        self.assertEqual(
            sorted(result["completed_observations"]), ["r1", "r2"]
        )
        self.assertIn("re-measurement", result["limitation"])

    def test_every_contract_target_is_present(self) -> None:
        result = ledger.target_ledger(
            "task103", [record("r1", "2026-10-01T10:00:00+08:00", full())]
        )
        self.assertEqual(sorted(result["targets"]), sorted(TARGETS))


if __name__ == "__main__":
    unittest.main()
