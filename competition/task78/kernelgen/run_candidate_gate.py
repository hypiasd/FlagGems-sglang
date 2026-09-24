#!/usr/bin/env python3
"""Run deterministic pre-submission gates for a Task 78 candidate directory.

This gate is intentionally local-only. A PASS means the candidate satisfies
the source contract, the CPU semantic model, and the deterministic compiler-
risk scan; it does not prove Triton compilation or accelerator performance.
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

GENERIC_SCRIPTS = Path(__file__).resolve().parents[3] / ".agents/skills/kernelgen-flagos/scripts"
sys.path.insert(0, str(GENERIC_SCRIPTS))
import reviewer_protocol
import check_structural_delta


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


def canonical_json_sha256(value: object) -> str:
    return reviewer_protocol.canonical_json_sha256(value)


def source_hashes(source_dir: Path | None) -> tuple[dict[str, str], list[str]]:
    if source_dir is None:
        return {}, ["source directory was not supplied"]
    hashes = {}
    missing = []
    for backend in BACKENDS:
        path = source_dir / source_name(backend)
        if path.is_file():
            hashes[backend] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            missing.append(str(path))
    return hashes, missing


def review_check(blind_path: Path | None, reconciliation_path: Path | None,
                 static_report: dict, required: bool,
                 expected_hashes: dict[str, str], expected_baseline_hashes: dict[str, str],
                 expected_candidate: str, expected_baseline: str) -> dict:
    return reviewer_protocol.validate(
        blind_path, reconciliation_path, static_report, required,
        expected_hashes, expected_baseline_hashes, expected_candidate, expected_baseline,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--review-json", type=Path,
                        help="second independent sub-agent reconciliation receipt")
    parser.add_argument("--blind-review-json", type=Path,
                        help="first-pass receipt from a fresh reviewer who was not shown static findings")
    parser.add_argument("--baseline-source-dir", type=Path,
                        help="exact baseline source directory reviewed by both agents")
    parser.add_argument("--structural-manifest", type=Path,
                        help="per-backend optimization hypotheses bound to exact candidate/baseline hashes")
    parser.add_argument("--require-structural-delta", action="store_true",
                        help="reject a candidate unless every backend has a manifest-bound AST-shape delta")
    parser.add_argument("--require-review", action="store_true",
                        help="reject candidates without a passing review receipt")
    parser.add_argument("--compiler-review-json", type=Path,
                        help="save the automatic compiler-risk scan receipt")
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
            result = {"backend": backend, **static_check(path)}
        except Exception as exc:
            result = {"path": str(path), "syntax": False, "error": str(exc)}
        static.append(result)

    # Run the cheap deterministic compiler-risk scan before the expensive CPU
    # matrix.  A known ABI/pointer/autotune blocker should stop immediately,
    # rather than spending minutes validating a candidate that cannot be
    # promoted anyway.
    with tempfile.NamedTemporaryFile(prefix="task78-review-", suffix=".json", delete=False) as handle:
        compiler_review_json = Path(handle.name)
    compiler_review_command = [
        sys.executable,
        str(Path(__file__).with_name("review_candidate.py")),
        str(source_dir),
        "--json",
        str(compiler_review_json),
    ]
    compiler_review_completed = subprocess.run(
        compiler_review_command, text=True, capture_output=True
    )
    try:
        compiler_review = json.loads(compiler_review_json.read_text(encoding="utf-8"))
    except Exception as exc:
        compiler_review = {
            "passed": False,
            "error": f"compiler-risk scan did not produce JSON: {exc}",
        }
    finally:
        if args.compiler_review_json and "compiler_review" in locals():
            args.compiler_review_json.parent.mkdir(parents=True, exist_ok=True)
            args.compiler_review_json.write_text(
                json.dumps(compiler_review, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        compiler_review_json.unlink(missing_ok=True)

    compiler_review_passed = (
        compiler_review.get("passed") is True
        and compiler_review_completed.returncode == 0
    )
    if compiler_review_passed:
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
            "--autotune-sweep",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        try:
            semantic = json.loads(validator_json.read_text(encoding="utf-8"))
        except Exception as exc:
            semantic = {"passed": False, "error": f"validator did not produce JSON: {exc}"}
        finally:
            validator_json.unlink(missing_ok=True)
    else:
        completed = None
        semantic = {
            "passed": False,
            "skipped": True,
            "reason": "deterministic compiler-risk scan failed; semantic matrix not run",
        }

    static_passed = (
        not missing
        and len(static) == len(BACKENDS)
        and all(item.get("syntax") and item.get("public_entry") and not item.get("forbidden")
                for item in static)
    )
    expected_hashes = {
        item["backend"]: item["sha256"]
        for item in static
        if item.get("backend") and item.get("sha256")
    }
    expected_baseline_hashes, missing_baseline = source_hashes(
        args.baseline_source_dir.resolve() if args.baseline_source_dir else None
    )
    if args.require_structural_delta and args.baseline_source_dir and args.structural_manifest:
        structural = check_structural_delta.validate(
            source_dir,
            args.baseline_source_dir.resolve(),
            args.structural_manifest.resolve(),
        )
    elif args.require_structural_delta:
        structural = {
            "passed": False,
            "errors": ["--require-structural-delta needs --baseline-source-dir and --structural-manifest"],
        }
    else:
        structural = {"passed": True, "required": False}
    review = review_check(
        args.blind_review_json, args.review_json, compiler_review,
        args.require_review, expected_hashes, expected_baseline_hashes,
        source_dir.name,
        args.baseline_source_dir.name if args.baseline_source_dir else "",
    )
    review["missing_baseline_sources"] = missing_baseline
    if args.require_review and missing_baseline:
        review["passed"] = False
        review["error"] = "baseline source directory is incomplete"
    passed = (
        static_passed
        and semantic.get("passed") is True
        and compiler_review_passed
        and review["passed"]
        and structural.get("passed") is True
    )
    validator_returncode = completed.returncode if completed is not None else None
    result = {
        "passed": passed,
        "source_dir": str(source_dir),
        "missing": missing,
        "static": static,
        "semantic": semantic,
        "compiler_review": compiler_review,
        "review": review,
        "structural_delta": structural,
        "validator_returncode": validator_returncode,
        "compiler_review_returncode": compiler_review_completed.returncode,
        "notice": "Local gate only; target compiler/device and performance remain unverified.",
    }
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if completed is not None and not passed and completed.stdout:
        print("\n--- validator stdout ---\n" + completed.stdout, file=sys.stderr)
    if completed is not None and completed.stderr:
        print("\n--- validator stderr ---\n" + completed.stderr, file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
