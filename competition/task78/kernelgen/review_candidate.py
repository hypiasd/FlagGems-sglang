#!/usr/bin/env python3
"""Deterministic compiler-risk scan used before the read-only sub-agent gate.

This is not a Triton compiler.  It finds patterns that must be explicitly
reviewed before a scarce target submission: runtime branches in JIT kernels,
unbounded power-of-two aranges, and unbounded block constants.  Findings are
reported as blockers or warnings; a sub-agent must inspect the source and
produce the final review receipt consumed by run_candidate_gate.py.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


BACKENDS = ("default", "ascend", "enflame", "hygon", "iluvatar", "kunlunxin", "metax")


def source_name(backend: str) -> str:
    suffix = "" if backend == "default" else f"_{backend}"
    return f"concat_and_cast_mha_k{suffix}.py"


def _names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _is_jit_decorator(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
            if dec.value.id == "triton" and dec.attr == "jit":
                return True
        if isinstance(dec, ast.Name) and dec.id == "jit":
            return True
    return False


def _constexpr_params(node: ast.FunctionDef) -> set[str]:
    result = set()
    for arg in node.args.args:
        annotation = ast.unparse(arg.annotation) if arg.annotation else ""
        if "constexpr" in annotation:
            result.add(arg.arg)
    return result


def _compile_time_names(node: ast.FunctionDef) -> set[str]:
    """Approximate scalar constexpr propagation inside a JIT function.

    Triton commonly derives compile-time scalars such as ``tail`` from
    constexpr parameters.  Treat those as safe while keeping values derived
    from ``program_id`` or tensor expressions in the runtime set.
    """
    known = _constexpr_params(node)
    changed = True
    while changed:
        changed = False
        for item in ast.walk(node):
            if isinstance(item, (ast.Assign, ast.AnnAssign)):
                value = item.value
                if value is None or not _names(value).issubset(known):
                    continue
                targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in known:
                        known.add(target.id)
                        changed = True
    return known


def inspect_source(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and _is_jit_decorator(n)]:
        constexpr = _compile_time_names(fn)
        for item in ast.walk(fn):
            if not isinstance(item, ast.If):
                continue
            names = _names(item.test)
            runtime_names = sorted(names - constexpr)
            if runtime_names:
                findings.append({
                    "severity": "blocker",
                    "kind": "runtime-branch-in-jit",
                    "line": item.lineno,
                    "function": fn.name,
                    "names": runtime_names,
                    "message": "JIT control flow depends on runtime tensor/program-id values; require explicit compiler review or split static paths.",
                })

    for line_no, line in enumerate(source.splitlines(), 1):
        if re.search(r"tl\.arange\(\s*0\s*,\s*(?:BR|BN|BC|BH|HS)\s*\)", line):
            findings.append({
                "severity": "warning",
                "kind": "power-of-two-bound",
                "line": line_no,
                "message": "arange bound is a block constexpr; verify the wrapper caps it for target compiler limits.",
            })
        if "triton.next_power_of_2" in line and not re.search(r"min\s*\(", line):
            findings.append({
                "severity": "blocker",
                "kind": "uncapped-next-power-of-two",
                "line": line_no,
                "message": "next_power_of_2 result is not visibly capped; large dimensions can create invalid or very slow tiles.",
            })
        if "eviction_policy" in line or "cache_modifier" in line:
            findings.append({
                "severity": "warning",
                "kind": "target-cache-hint",
                "line": line_no,
                "message": "target-specific cache hint requires compiler evidence; do not infer support from CPU validation.",
            })

    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "findings": findings,
        "blockers": [f for f in findings if f["severity"] == "blocker"],
        "warnings": [f for f in findings if f["severity"] == "warning"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args(argv)
    reports = {}
    missing = []
    for backend in BACKENDS:
        path = args.source_dir / source_name(backend)
        if path.is_file():
            reports[backend] = inspect_source(path)
        else:
            missing.append(str(path))
    blockers = [
        {"backend": backend, **finding}
        for backend, report in reports.items()
        for finding in report["blockers"]
    ]
    result = {
        "passed": not missing and not blockers,
        "source_dir": str(args.source_dir.resolve()),
        "missing": missing,
        "backends": reports,
        "blockers": blockers,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
