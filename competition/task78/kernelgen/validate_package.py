#!/usr/bin/env python3
"""Verify a Task 78 ZIP is byte-identical to the reviewed source directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from run_candidate_gate import BACKENDS, source_name


def validate_package(source_dir: Path, archive_path: Path) -> dict:
    expected_names = {source_name(backend) for backend in BACKENDS}
    expected = {}
    missing_sources = []
    for name in sorted(expected_names):
        path = source_dir / name
        if path.is_file():
            expected[name] = path.read_bytes()
        else:
            missing_sources.append(name)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            files = [name for name in names if not name.endswith("/")]
            duplicates = sorted({name for name in files if files.count(name) > 1})
            actual = {
                name: archive.read(name)
                for name in files
                if name in expected_names
            }
    except Exception as exc:
        return {
            "passed": False,
            "archive": str(archive_path),
            "error": f"could not read package ZIP: {exc}",
        }

    extra = sorted(set(files) - expected_names)
    missing = sorted(expected_names - set(files))
    mismatched = sorted(
        name for name in expected_names & set(actual)
        if expected.get(name) != actual.get(name)
    )
    file_hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in sorted(actual.items())
    }
    return {
        "passed": (
            not missing_sources and not extra and not missing
            and not duplicates and not mismatched
        ),
        "archive": str(archive_path),
        "source_dir": str(source_dir),
        "expected_file_count": len(expected_names),
        "archive_file_count": len(files),
        "missing_sources": missing_sources,
        "missing_from_archive": missing,
        "extra_in_archive": extra,
        "duplicate_entries": duplicates,
        "hash_mismatches": mismatched,
        "file_sha256": file_hashes,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    result = validate_package(args.source_dir, args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
