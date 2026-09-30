#!/usr/bin/env python3
"""Tests per-chip and shared-generic baseline source selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_mixed_candidate as prepare
import task78_results as results


class MixedCandidateSelectionTests(unittest.TestCase):
    def test_shared_generic_baseline_uses_joint_a_b_evidence(self):
        rows = results.load_ledger()
        row, stats = prepare.joint_generic_champion(rows, results.ROOT)
        self.assertEqual(row["version"], "v23")
        self.assertEqual(stats["per_target_median"]["intl_a"], 1.64)
        self.assertEqual(stats["per_target_median"]["intl_b"], 1.75)


if __name__ == "__main__":
    unittest.main()
