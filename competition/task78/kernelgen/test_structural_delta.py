#!/usr/bin/env python3
"""Tests for the generic structural-delta gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from check_structural_delta import BACKENDS, source_name, validate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StructuralDeltaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="task78-structure-")
        self.root = Path(self.tmp.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.baseline.mkdir()
        self.candidate.mkdir()
        self.manifest_path = self.root / "hypotheses.json"
        self.baseline_text = "def concat_and_cast_mha_k(x):\n    return x + 1\n"
        self.manifest = {"source_method": "agent-authored", "backend_hypotheses": {}}

    def tearDown(self):
        self.tmp.cleanup()

    def write_sources(self, candidate_text: str):
        for backend in BACKENDS:
            name = source_name(backend)
            base_path = self.baseline / name
            candidate_path = self.candidate / name
            base_path.write_text(self.baseline_text, encoding="utf-8")
            candidate_path.write_text(candidate_text, encoding="utf-8")
            self.manifest["backend_hypotheses"][backend] = {
                "target_ids": ["intl_a", "intl_b"] if backend == "default" else [backend],
                "bottleneck_evidence": "recorded backend mapping and source path",
                "structural_change": "replace a scalar expression with a predicated two-path structure",
                "expected_mechanism": "specialize work by the runtime condition instead of executing one uniform expression",
                "falsifier": "official target result fails correctness or regresses versus its bound source",
                "baseline_sha256": digest(base_path),
                "source_sha256": digest(candidate_path),
            }
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_accepts_all_backend_structural_changes_bound_to_exact_hashes(self):
        self.write_sources("def concat_and_cast_mha_k(x):\n    if x:\n        return x + 1\n    return x\n")
        report = validate(self.candidate, self.baseline, self.manifest_path)
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(len(report["backends"]), 7)
        self.assertTrue(all(item["structural_delta"] for item in report["backends"].values()))

    def test_rejects_literal_only_edits_even_if_bytes_differ(self):
        self.write_sources("def concat_and_cast_mha_k(x):\n    return x + 2\n")
        report = validate(self.candidate, self.baseline, self.manifest_path)
        self.assertFalse(report["passed"])
        self.assertTrue(all("normalized AST is unchanged" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
