#!/usr/bin/env python3
"""Run deterministic pre-Arc gates for a Task 78 candidate directory.

This gate is intentionally local-only. A PASS means the candidate satisfies
the source contract and the CPU semantic model; it does not prove Triton
compilation or accelerator performance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile


BACKENDS = ("default", "ascend", "enflame", "hygon", "iluvatar", "kunlunxin", "metax")
PUBLIC = "concat_and_cast_mha_k"


def source_name(backend: str) -> str:
    suffix = "" if backend == "default" else f"_{backend}"
    return f"concat_and_cast_mha_k{suffix}.py"


def static_check(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    public = [node for node in functions if node.name == PUBLIC]
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            forbidden.append("try/except fallback")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                if node.func.attr in {"cat", "concat", "compile"}:
                    forbidden.append(f"torch.{node.func.attr}")
                if node.func.attr == "ops":
                    forbidden.append("torch.ops")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "public_count": len(public),
        "public_entry": len(public) == 1,
        "forbidden": sorted(set(forbidden)),
        "syntax": True,
    }


def review_check(review_path: Path | None, required: bool,
                 expected_hashes: dict[str, str]) -> dict:
    """Validate the read-only sub-agent review receipt.

    The gate cannot prove that a sub-agent really inspected the source, but it
    can prevent a candidate from being promoted without an explicit review
    receipt and can reject receipts that still contain unresolved blockers.
    The receipt is deliberately small and human-auditable.
    """
    if review_path is None:
        return {
            "required": required,
            "present": False,
            "passed": not required,
            "error": "review receipt was not supplied" if required else None,
        }
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "required": required,
            "present": False,
            "passed": False,
            "error": f"could not read review receipt: {exc}",
        }
    blockers = review.get("blockers")
    findings = review.get("backend_findings")
    observed_hashes = review.get("reviewed_source_sha256")
    hashes_match = (
        isinstance(observed_hashes, dict)
        and all(observed_hashes.get(backend) == digest
                for backend, digest in expected_hashes.items())
    )
    passed = (
        review.get("review_type") == "read-only-subagent"
        and review.get("candidate")
        and review.get("reviewer")
        and isinstance(findings, dict)
        and all(backend in findings for backend in expected_hashes)
        and isinstance(blockers, list)
        and not blockers
        and hashes_match
    )
    return {
        "required": required,
        "present": True,
        "passed": bool(passed),
        "path": str(review_path),
        "blocker_count": len(blockers) if isinstance(blockers, list) else None,
        "hashes_match": hashes_match,
        "error": None if passed else "review receipt is missing required fields or has blockers",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--review-json", type=Path,
                        help="read-only sub-agent review receipt")
    parser.add_argument("--require-review", action="store_true",
                        help="reject candidates without a passing review receipt")
    args = parser.parse_args(argv)
    source_dir = args.source_dir.resolve()
    static = []
    missing = []
    for backend in BACKENDS:
        path = source_dir / source_name(backend)
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            result = static_check(path)
        except Exception as exc:
            result = {"path": str(path), "syntax": False, "error": str(exc)}
        static.append(result)

    with tempfile.NamedTemporaryFile(prefix="task78-gate-", suffix=".json", delete=False) as handle:
        validator_json = Path(handle.name)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "validate_cpu.py"),
        "--source-dir",
        str(source_dir),
        "--json",
        str(validator_json),
        "--all",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    try:
        semantic = json.loads(validator_json.read_text(encoding="utf-8"))
    except Exception as exc:
        semantic = {"passed": False, "error": f"validator did not produce JSON: {exc}"}
    finally:
        validator_json.unlink(missing_ok=True)

    static_passed = (
        not missing
        and len(static) == len(BACKENDS)
        and all(item.get("syntax") and item.get("public_entry") and not item.get("forbidden")
                for item in static)
    )
    expected_hashes = {
        backend: item["sha256"]
        for backend, item in zip(BACKENDS, static)
        if item.get("sha256")
    }
    review = review_check(args.review_json, args.require_review, expected_hashes)
    passed = (
        static_passed
        and semantic.get("passed") is True
        and completed.returncode == 0
        and review["passed"]
    )
    result = {
        "passed": passed,
        "source_dir": str(source_dir),
        "missing": missing,
        "static": static,
        "semantic": semantic,
        "review": review,
        "validator_returncode": completed.returncode,
        "notice": "Local gate only; target compiler/device and performance remain unverified.",
    }
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed and completed.stdout:
        print("\n--- validator stdout ---\n" + completed.stdout, file=sys.stderr)
    if completed.stderr:
        print("\n--- validator stderr ---\n" + completed.stderr, file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
