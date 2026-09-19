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


if __name__ == "__main__":
    unittest.main()
