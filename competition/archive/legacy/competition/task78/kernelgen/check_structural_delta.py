#!/usr/bin/env python3
"""Require a reviewed, non-constant-only structural delta for each backend."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

BACKENDS = ("default", "ascend", "enflame", "hygon", "iluvatar", "kunlunxin", "metax")
TARGETS_BY_BACKEND = {
    "default": ["intl_a", "intl_b"],
    "ascend": ["ascend"], "enflame": ["enflame"], "hygon": ["hygon"],
    "iluvatar": ["iluvatar"], "kunlunxin": ["kunlunxin"], "metax": ["metax"],
}


def source_name(backend: str) -> str:
    suffix = "" if backend == "default" else f"_{backend}"
    return f"concat_and_cast_mha_k{suffix}.py"


class ShapeNormalizer(ast.NodeTransformer):
    """Erase names/docstrings/literal values while retaining program shape."""

    def visit_Constant(self, node: ast.Constant):
        node.value = f"<constant:{type(node.value).__name__}>"
        return node

    def visit_Name(self, node: ast.Name):
        node.id = "_NAME_"
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = "_ARG_"
        node.annotation = self.visit(node.annotation) if node.annotation is not None else None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = "_FUNCTION_"
        return self._visit_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        node.name = "_FUNCTION_"
        return self._visit_body(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = "_CLASS_"
        return self._visit_body(node)

    def _visit_body(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return self.generic_visit(node)


def normalized_ast(path: Path) -> tuple[str, dict[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    normalized = ShapeNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    structure = ast.dump(normalized, include_attributes=False)
    counts = {
        "functions": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)),
        "for_loops": sum(isinstance(n, (ast.For, ast.AsyncFor)) for n in ast.walk(tree)),
        "while_loops": sum(isinstance(n, ast.While) for n in ast.walk(tree)),
        "branches": sum(isinstance(n, ast.If) for n in ast.walk(tree)),
        "loads": sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "load" for n in ast.walk(tree)),
        "stores": sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "store" for n in ast.walk(tree)),
        "launches": sum(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id.startswith("_") for n in ast.walk(tree)),
    }
    return hashlib.sha256(structure.encode("utf-8")).hexdigest(), counts


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(candidate_dir: Path, baseline_dir: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hypotheses = manifest.get("backend_hypotheses", {})
    errors = []
    if manifest.get("source_method") not in {"agent-authored", "kernelgen"}:
        errors.append("manifest source_method must explicitly identify agent-authored or kernelgen provenance")
    missing_hypotheses = sorted(set(BACKENDS) - set(hypotheses))
    unexpected_hypotheses = sorted(set(hypotheses) - set(BACKENDS))
    sources = {}
    for backend in BACKENDS:
        name = source_name(backend)
        candidate = candidate_dir / name
        baseline = baseline_dir / name
        if not candidate.is_file() or not baseline.is_file():
            errors.append(f"{backend}: missing candidate or baseline source")
            continue
        candidate_hash, baseline_hash = sha_file(candidate), sha_file(baseline)
        cand_shape, cand_counts = normalized_ast(candidate)
        base_shape, base_counts = normalized_ast(baseline)
        hypothesis = hypotheses.get(backend, {})
        required = ("target_ids", "bottleneck_evidence", "structural_change",
                    "expected_mechanism", "falsifier")
        absent = [key for key in required if not hypothesis.get(key)]
        if absent:
            errors.append(f"{backend}: hypothesis missing {', '.join(absent)}")
        if hypothesis.get("target_ids") != TARGETS_BY_BACKEND[backend]:
            errors.append(f"{backend}: target_ids must be {TARGETS_BY_BACKEND[backend]}")
        if candidate_hash == baseline_hash:
            errors.append(f"{backend}: candidate bytes equal its per-chip champion baseline")
        if cand_shape == base_shape:
            errors.append(f"{backend}: normalized AST is unchanged; likely only constants/comments/names changed")
        if hypothesis.get("source_sha256") != candidate_hash:
            errors.append(f"{backend}: hypothesis source_sha256 does not bind the candidate bytes")
        if hypothesis.get("baseline_sha256") != baseline_hash:
            errors.append(f"{backend}: hypothesis baseline_sha256 does not bind the champion bytes")
        sources[backend] = {
            "source": name,
            "source_sha256": candidate_hash,
            "baseline_sha256": baseline_hash,
            "candidate_ast_shape_sha256": cand_shape,
            "baseline_ast_shape_sha256": base_shape,
            "candidate_ast_counts": cand_counts,
            "baseline_ast_counts": base_counts,
            "structural_delta": cand_shape != base_shape,
        }
    if missing_hypotheses:
        errors.append(f"manifest missing backend hypotheses: {', '.join(missing_hypotheses)}")
    if unexpected_hypotheses:
        errors.append(f"manifest has unexpected backend hypotheses: {', '.join(unexpected_hypotheses)}")
    return {
        "passed": not errors,
        "candidate_dir": str(candidate_dir),
        "baseline_dir": str(baseline_dir),
        "manifest": str(manifest_path),
        "backends": sources,
        "errors": errors,
        "limitation": "AST shape difference rejects many constant-only edits but does not prove the claimed transformation is an optimization; source review and target measurements remain necessary.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate(args.candidate_dir, args.baseline_dir, args.manifest)
    except (OSError, ValueError, json.JSONDecodeError, SyntaxError) as exc:
        report = {"passed": False, "errors": [str(exc)]}
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
