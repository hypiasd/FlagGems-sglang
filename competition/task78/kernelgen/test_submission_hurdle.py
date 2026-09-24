#!/usr/bin/env python3
"""Tests that the limited-quota release hurdle blocks small or weakly supported gains."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import submission_hurdle as hurdle
import task78_results as results


class SubmissionHurdleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = results.load_ledger()

    def forecast(self, expected: float, lower: float, confidence: str = "medium"):
        return {
            target: {
                "expected_delta_pct": expected,
                "lower_delta_pct": lower,
                "confidence": confidence,
                "evidence_basis": "historical_structural_comparison",
                "evidence": ["same target, source-bound official observation"],
            }
            for target in results.TARGETS
        }

    def test_material_gain_with_nonnegative_lower_bound_passes(self):
        report = hurdle.evaluate(
            {"performance_forecast": self.forecast(10, 0)}, self.rows, results.ROOT
        )
        self.assertTrue(report["passed"], report["errors"])
        self.assertGreaterEqual(report["projected_mean"], report["baseline_mean"] * 1.05)

    def test_small_gain_is_rejected(self):
        report = hurdle.evaluate(
            {"performance_forecast": self.forecast(2, 0)}, self.rows, results.ROOT
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("release hurdle" in error for error in report["errors"]))

    def test_five_percent_is_not_enough_if_it_misses_the_1_50_goal(self):
        # The current champion composite is about 1.38x: +5% reaches only
        # about 1.45x, still below the user's stated 1.50x competition goal.
        report = hurdle.evaluate(
            {"performance_forecast": self.forecast(5, 0)}, self.rows, results.ROOT
        )
        self.assertFalse(report["passed"])
        self.assertLess(report["projected_mean"], hurdle.MIN_ABSOLUTE_MEAN)

    def test_low_confidence_or_large_target_downside_is_rejected(self):
        forecast = self.forecast(10, 0, confidence="low")
        report = hurdle.evaluate({"performance_forecast": forecast}, self.rows, results.ROOT)
        self.assertFalse(report["passed"])
        forecast = self.forecast(10, -6)
        report = hurdle.evaluate({"performance_forecast": forecast}, self.rows, results.ROOT)
        self.assertFalse(report["passed"])
        self.assertTrue(any("lower-bound regression" in error for error in report["errors"]))

    def test_non_finite_forecast_is_rejected(self):
        forecast = self.forecast(10, 0)
        forecast["ascend"]["expected_delta_pct"] = float("nan")
        report = hurdle.evaluate({"performance_forecast": forecast}, self.rows, results.ROOT)
        self.assertFalse(report["passed"])
        self.assertTrue(any("finite numbers" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
