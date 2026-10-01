"""Local gates for an upstream competition-PR bundle.

The authoritative gate is upstream CI (``basic-ci.yml``).  What we can do
locally is split into three tiers so a bundle is never claimed "ready" on
evidence we do not have:

* always runnable, stdlib only  -> structure / hygiene / AST preservation
* runnable when a toolchain is present -> black, isort, flake8
* runnable when a poisoned tree + deps are present -> upstream ci_checks,
  import smoke (needs torch+triton, so usually unavailable off-device)
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

from . import evidence as evidence_mod
from . import spec

PASS, FAIL, SKIPPED, UNAVAILABLE = "pass", "fail", "skipped", "unavailable"


def _gate(name, status, detail, **extra):
    return {"name": name, "status": status, "detail": detail, **extra}


def strip_prologue(tree: ast.Module) -> str:
    """Drop the module docstring and the added ``__all__`` assignment."""
    kept = []
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            continue
        kept.append(node)
    return ast.dump(
        ast.Module(body=kept, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )


def apply_renames(tree: ast.Module, mapping: dict[str, str]) -> ast.Module:
    """Normalise declared identifier renames so lint fixes stay comparable."""
    if not mapping:
        return tree
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            node.id = mapping[node.id]
        elif isinstance(node, ast.arg) and node.arg in mapping:
            node.arg = mapping[node.arg]
        elif isinstance(node, ast.keyword) and node.arg in mapping:
            node.arg = mapping[node.arg]
        elif isinstance(node, ast.Global) and mapping:
            node.names = [mapping.get(name, name) for name in node.names]
        elif isinstance(node, ast.Nonlocal):
            node.names = [mapping.get(name, name) for name in node.names]
    return tree


def ast_equal_modulo_prologue(
    awarded: str, bundled: str, renames: dict[str, str] | None = None
) -> tuple[bool, str]:
    try:
        left = strip_prologue(apply_renames(ast.parse(awarded), renames or {}))
        right = strip_prologue(ast.parse(bundled))
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    if left == right:
        note = (
            "implementation AST identical"
            if not renames
            else "implementation AST identical modulo declared renames"
        )
        return True, note
    return False, "implementation AST changed"


def structure_gate(
    bundle: dict, texts: dict[str, str], entrypoint: str
) -> dict:
    op = bundle["op"]
    errors, seen = [], []
    base = entrypoint.split("(", 1)[0].strip()
    arity = len(
        [
            p
            for p in entrypoint.split("(", 1)[1].rstrip(")").split(",")
            if p.strip()
        ]
    )
    described = {entry["path"] for entry in bundle["files"]}
    for rel, text in texts.items():
        if rel not in described:
            errors.append(f"{rel}: not described by the bundle manifest")
            continue
        tier, expected = spec.tier_path(op, _suffix_of(bundle, rel))
        if rel != expected:
            errors.append(f"{rel}: expected tier path {expected}")
        if Path(rel).stem != op:
            errors.append(f"{rel}: file stem must equal the op name {op!r}")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            errors.append(f"{rel}: cannot parse: {exc}")
            continue
        assigned = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "__all__"
                for t in node.targets
            )
        ]
        if not assigned:
            errors.append(f"{rel}: missing __all__")
        else:
            try:
                value = ast.literal_eval(assigned[0].value)
            except ValueError:
                value = None
            if value != [op]:
                errors.append(
                    f"{rel}: __all__ must be exactly [{op!r}], found {value!r}"
                )
        public = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == op
        ]
        if not public:
            errors.append(f"{rel}: no public def {op}(...)")
        elif len(public[0].args.args) != arity:
            errors.append(
                f"{rel}: {op} takes {len(public[0].args.args)} args, contract says {arity}"
            )
        if public and base != op:
            errors.append(
                f"{rel}: entrypoint name {base!r} does not match op {op!r}"
            )
        if tier == "generic" and any(
            isinstance(n, (ast.Import, ast.ImportFrom))
            and "runtime.backend" in ast.dump(n)
            for n in ast.walk(tree)
        ):
            errors.append(
                f"{rel}: generic tier must not import runtime.backend"
            )
        seen.append({"path": rel, "tier": tier})
    if not texts:
        errors.append("bundle contains no source files")
    return _gate(
        "structure",
        FAIL if errors else PASS,
        "; ".join(errors)
        or "tier paths, __all__, signatures and tier purity hold",
        files=seen,
    )


def hygiene_gate(texts: dict[str, str]) -> dict:
    errors = []
    for rel, text in texts.items():
        lines = text.split("\n")
        if not text.startswith(spec.HEADER):
            errors.append(f"{rel}: missing Apache 2.0 header")
        if "\r" in text:
            errors.append(f"{rel}: CRLF line endings")
        if "\t" in text:
            errors.append(f"{rel}: tab characters")
        if not text.endswith("\n") or text.endswith("\n\n"):
            errors.append(f"{rel}: file must end with exactly one newline")
        for number, line in enumerate(lines, 1):
            if line.rstrip() != line:
                errors.append(f"{rel}:{number}: trailing whitespace")
            if len(line) > spec.FLAKE8_MAX_LINE:
                errors.append(
                    f"{rel}:{number}: {len(line)} columns exceeds flake8's {spec.FLAKE8_MAX_LINE}"
                )
    return _gate(
        "hygiene",
        FAIL if errors else PASS,
        "; ".join(errors[:12])
        or "header, line width (<=120), whitespace and newline conventions hold",
    )


def preservation_gate(
    bundle: dict, awarded: dict[str, str], texts: dict[str, str]
) -> dict:
    errors, checked = [], 0
    by_path = {entry["path"]: entry["name"] for entry in bundle["files"]}
    for rel, text in texts.items():
        name = by_path.get(rel)
        source = awarded.get(name) if name else None
        if source is None:
            errors.append(f"{rel}: no awarded source recorded")
            continue
        equal, detail = ast_equal_modulo_prologue(
            source, text, bundle.get("renames")
        )
        checked += 1
        if not equal:
            errors.append(f"{rel}: {detail}")
    return _gate(
        "ast_preservation",
        FAIL if errors else PASS,
        "; ".join(errors)
        or f"{checked} file(s) keep the awarded implementation AST modulo header/__all__/declared renames",
    )


def _tool(bin_dir: Path | None, name: str) -> str | None:
    if bin_dir is not None:
        candidate = Path(bin_dir) / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


def style_gate(
    files: list[Path], bin_dir: Path | None, cwd: Path | None = None
) -> dict:
    missing = [
        name
        for name in ("black", "isort", "flake8")
        if _tool(bin_dir, name) is None
    ]
    if missing:
        return _gate(
            "style",
            UNAVAILABLE,
            "missing %s; install black==24.8.0, flake8==7.1.0, isort==5.12.0 "
            "and rerun with --tools <bin-dir>" % ", ".join(missing),
            commands=[
                f"black {' '.join(str(f) for f in files)}",
                "isort %s <files>" % " ".join(spec.ISORT_ARGS),
                f"flake8 {spec.FLAKE8_ARGS[0]} {spec.FLAKE8_ARGS[1]} <files>",
            ],
        )
    runs, failures = [], []
    commands = [
        (
            "black",
            [
                str(_tool(bin_dir, "black")),
                "--check",
                "--diff",
                *[str(f) for f in files],
            ],
        ),
        (
            "isort",
            [
                str(_tool(bin_dir, "isort")),
                "--check-only",
                *spec.ISORT_ARGS,
                *[str(f) for f in files],
            ],
        ),
        (
            "flake8",
            [
                str(_tool(bin_dir, "flake8")),
                *spec.FLAKE8_ARGS,
                *[str(f) for f in files],
            ],
        ),
    ]
    for name, argv in commands:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
        runs.append(
            {
                "tool": name,
                "returncode": proc.returncode,
                "output": (proc.stdout + proc.stderr).strip()[:2000],
            }
        )
        if proc.returncode != 0:
            failures.append(name)
    return _gate(
        "style",
        FAIL if failures else PASS,
        (
            f"failed: {', '.join(failures)}"
            if failures
            else "black/isort/flake8 clean"
        ),
        runs=runs,
    )


def repo_checks_gate(tree: Path, python: str, operators: list[str]) -> dict:
    """Run the upstream rule checks that apply to a competition PR.

    CONTRIBUTING section 9 waives tests and benchmarks for competition PRs and
    waives the operators.yaml entry too, so the marker/benchmark/yaml-driven
    checks are out of scope by rule rather than by convenience.  A check that
    cannot start because a dependency is missing is reported as unavailable,
    never as a pass.
    """
    if not (tree / "tools" / "ci_checks").is_dir():
        return _gate(
            "repo_checks",
            UNAVAILABLE,
            f"no upstream tree at {tree}; run `materialize` first",
        )
    applicable = [
        (["tools/ci_checks/check_init_exports.py"], None),
        (
            [
                "tools/ci_checks/check_kernelgen_tests.py",
                "--operators",
                json.dumps(operators),
            ],
            None,
        ),
        (["tools/ci_checks/check_operators_yaml.py", "--all"], "yaml"),
    ]
    out_of_scope = {
        "tools/ci_checks/check_operator_markers.py": "section 9 waives tests/test_<op>.py",
        "tools/ci_checks/check_aten_operators.py": "keyed off operators.yaml entries, which competition PRs do not add",
        "tools/ci_checks/check_api_logs.py": "warning-level and keyed off operators.yaml entries",
        "tools/ci_checks/check_performance_reference.py": "keyed off operators.yaml entries",
    }
    runs, failures, unavailable = [], [], []
    for argv, needs in applicable:
        script = tree / argv[0]
        if not script.is_file():
            unavailable.append(f"{argv[0]} (absent upstream)")
            continue
        if needs == "yaml":
            probe = subprocess.run(
                [python, "-c", "import yaml"],
                cwd=tree,
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0:
                unavailable.append(
                    f"{argv[0]} (pyyaml unavailable for {python})"
                )
                continue
        proc = subprocess.run(
            [python, *argv], cwd=tree, capture_output=True, text=True
        )
        output = (proc.stdout + proc.stderr).strip()
        runs.append(
            {
                "argv": argv,
                "returncode": proc.returncode,
                "output": output[:1500],
            }
        )
        if proc.returncode != 0 and "ModuleNotFoundError" in output:
            unavailable.append(f"{argv[0]} ({output.splitlines()[-1][:80]})")
        elif proc.returncode != 0:
            failures.append(argv[0])
    detail = (
        f"failed: {', '.join(failures)}"
        if failures
        else "applicable upstream rule checks clean"
    )
    if unavailable:
        detail += f"; unavailable: {', '.join(unavailable)}"
    detail += f"; out of scope for competition PRs: {', '.join(sorted(Path(k).name for k in out_of_scope))}"
    return _gate(
        "repo_checks",
        FAIL if failures else PASS,
        detail,
        runs=runs,
        out_of_scope=out_of_scope,
    )


def import_smoke_gate(tree: Path, python: str, op: str) -> dict:
    script = (
        "import flaggems_sglang as f;"
        f"ops = f.all_registered_ops();"
        f"assert {op!r} in ops, 'op not registered';"
        f"print('registered', {op!r}, f.{op}.__module__)"
    )
    proc = subprocess.run(
        [python, "-c", script], cwd=tree, capture_output=True, text=True
    )
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        missing = [
            line for line in output.splitlines() if "No module named" in line
        ]
        if missing:
            return _gate(
                "import_smoke",
                UNAVAILABLE,
                f"cannot import the package here ({missing[-1].strip()}); "
                f'run this on a device with torch+triton: {python} -c "{script}"',
            )
        return _gate(
            "import_smoke", FAIL, " | ".join(output.splitlines()[-3:])
        )
    return _gate("import_smoke", PASS, proc.stdout.strip()[:500])


def _suffix_of(bundle: dict, rel: str) -> str | None:
    for item in bundle["files"]:
        if item["path"] == rel:
            return item.get("suffix")
    return None


def evidence_gate(report: dict) -> dict:
    """Fold the evidence discipline into the gate summary.

    Contradictory recorded evidence fails.  Evidence that simply does not
    exist here (no callable target route, no review receipts, provisional
    platform values) keeps the gate ``unavailable``, so ``ready_for_pr`` still
    means "nothing we can run has failed" rather than "this is validated".
    """
    official = report.get("official") or {}
    errors = []
    for item in official.get("lifecycle") or []:
        errors.extend(
            f"{item.get('record_id')}: {problem}"
            for problem in item.get("errors") or []
        )
    review = report.get("review") or {}
    errors.extend(review.get("errors") or [])
    gaps = list(review.get("gaps") or [])
    capability = report.get("capability") or {}
    callable_routes = [
        name
        for name, state in capability.items()
        if state == evidence_mod.CALLABLE
    ]
    missing = list(report.get("missing_target_evidence") or [])
    provisional = official.get("provisional_records") or []
    if not callable_routes:
        gaps.append(
            "no callable target route; capability is "
            + ", ".join(f"{k}={v}" for k, v in sorted(capability.items()))
        )
    for name in missing:
        state = (report.get("targets") or {}).get(name) or {}
        gaps.append(f"{name}: {state.get('basis')}")
    if provisional:
        gaps.append(
            f"{len(provisional)} provisional platform record(s) quoted, "
            "never aggregated"
        )
    if errors:
        detail = "; ".join(errors[:12])
    elif gaps:
        detail = "unestablished here: " + "; ".join(gaps[:12])
    else:
        detail = "every target carries terminal per-source official evidence"
    return _gate(
        "evidence",
        FAIL if errors else (UNAVAILABLE if gaps else PASS),
        detail,
        capability=capability,
        package_state=report.get("package_state"),
        missing_target_evidence=missing,
        best_per_target=report.get("best_per_target") or {},
        best_aggregate_speedup=official.get("best_aggregate_speedup"),
        best_aggregate_record_id=official.get("best_aggregate_record_id"),
        provisional_records=[item.get("record_id") for item in provisional],
        superseded_records=[
            item.get("record_id")
            for item in official.get("superseded_records") or []
        ],
        review={
            "verified": review.get("verified"),
            "authenticated": review.get("authenticated"),
            "gaps": review.get("gaps") or [],
        },
        official_score_computed=bool(official.get("official_score_computed")),
    )


def summarize(gates: list[dict]) -> dict:
    failed = [g["name"] for g in gates if g["status"] == FAIL]
    unavailable = [
        g["name"] for g in gates if g["status"] in (UNAVAILABLE, SKIPPED)
    ]
    return {
        "gates": gates,
        "failed": failed,
        "not_run_here": unavailable,
        "ready_for_pr": not failed,
        "ready_claim": (
            "all locally runnable gates pass; unrun gates still need upstream CI"
            if not failed
            else "blocked"
        ),
    }
