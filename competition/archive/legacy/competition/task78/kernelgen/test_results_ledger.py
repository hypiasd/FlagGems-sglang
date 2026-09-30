#!/usr/bin/env python3
"""Regression tests for the append-only Task 78 result ledger."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task78_results as results


class Task78ResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = results.load_ledger()
        cls.enriched = results.enrich(cls.rows)

    def test_backfill_preserves_all_25_rows_and_duplicate_v7(self):
        self.assertEqual(len(self.rows), 25)
        v7 = [row for row in self.rows if row["version"] == "v7"]
        self.assertEqual(len(v7), 2)
        self.assertNotEqual(v7[0]["record_id"], v7[1]["record_id"])

    def test_partial_runs_never_receive_aggregate(self):
        self.assertEqual(results.validate_ledger(self.rows), [])
        self.assertTrue(all(row["aggregate_speedup"] is None
                            for row in self.rows if row["pass_count"] < 8))
        complete = [row for row in self.rows if row["pass_count"] == 8]
        self.assertTrue(all(row["aggregate_speedup"] is not None for row in complete))
        self.assertEqual(max(row["aggregate_speedup"] for row in complete), 1.27)

    def test_champions_are_selected_per_target_not_by_one_whole_version(self):
        expected = {
            "iluvatar": (2.36, "v19"), "metax": (1.51, "v11"),
            "enflame": (0.60, "v16"), "hygon": (2.50, "v17"),
            "kunlunxin": (0.39, "v3"), "ascend": (0.33, "v13"),
            "intl_a": (1.64, "v21"), "intl_b": (1.75, "v23"),
        }
        actual = results.champions(self.rows)
        for target, pair in expected.items():
            self.assertEqual((actual[target]["score"], actual[target]["selected"]["version"]), pair)

    def test_identical_source_score_spread_is_flagged_without_discarding_values(self):
        same_ascend = [row for row in self.enriched
                       if row["version"] in {"v19", "v25"}]
        for row in same_ascend:
            self.assertEqual(row["targets"]["ascend"]["source_spread"], "flagged")
            self.assertEqual(row["targets"]["kunlunxin"]["source_spread"], "flagged")
            self.assertIn(row["targets"]["ascend"]["speedup"], (0.14, 0.23))
        self.assertEqual(len(same_ascend), 2)

    def test_generic_archive_fallback_is_part_of_source_identity(self):
        v3 = next(row for row in self.enriched if row["version"] == "v3")
        item = v3["targets"]["kunlunxin"]
        self.assertEqual(item["source_member"], "concat_and_cast_mha_k.py")
        self.assertEqual(item["source_sha256"],
                         "5714a25b3b07c96d2f3537539b29bf1c137849b5e60c2781f939e029be59b388")


if __name__ == "__main__":
    unittest.main()
