#!/usr/bin/env python3
"""Tests that release ZIP bytes match the reviewed Task 78 sources."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

from run_candidate_gate import BACKENDS, source_name
from validate_package import validate_package


class PackageValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="task78-package-gate-")
        self.root = Path(self.temp.name)
        self.source_dir = self.root / "candidate"
        self.source_dir.mkdir()
        self.sources = {}
        for index, backend in enumerate(BACKENDS):
            name = source_name(backend)
            data = f"# candidate {index}\n".encode("utf-8")
            (self.source_dir / name).write_bytes(data)
            self.sources[name] = data
        self.archive_path = self.root / "package.zip"

    def tearDown(self):
        self.temp.cleanup()

    def write_archive(self, entries=None):
        with zipfile.ZipFile(self.archive_path, "w") as archive:
            for name, data in (entries or self.sources).items():
                archive.writestr(name, data)

    def test_accepts_exact_seven_root_files(self):
        self.write_archive()
        result = validate_package(self.source_dir, self.archive_path)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["archive_file_count"], 7)
        self.assertEqual(len(result["file_sha256"]), 7)

    def test_rejects_source_mismatch(self):
        entries = dict(self.sources)
        name = source_name("enflame")
        entries[name] = b"different source"
        self.write_archive(entries)
        result = validate_package(self.source_dir, self.archive_path)
        self.assertFalse(result["passed"])
        self.assertEqual(result["hash_mismatches"], [name])

    def test_rejects_extra_or_nested_files(self):
        entries = dict(self.sources)
        entries["README.md"] = b"extra"
        self.write_archive(entries)
        result = validate_package(self.source_dir, self.archive_path)
        self.assertFalse(result["passed"])
        self.assertIn("README.md", result["extra_in_archive"])


if __name__ == "__main__":
    unittest.main()
