#!/usr/bin/env python3
"""Deterministic compiler-risk scan used before the read-only sub-agent gate.

This is not a Triton compiler.  It finds patterns that must be explicitly
reviewed before a scarce target submission: runtime branches in JIT kernels,
unbounded power-of-two aranges, invalid launch configurations, and known
backend-sensitive shape conversions.  Findings are reported as blockers or
warnings; a sub-agent must inspect the source and produce the final review
receipt consumed by run_candidate_gate.py.
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


def _int_literal(node: ast.AST) -> int | None:
    """Return a small statically visible integer, otherwise ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _int_literal(node.operand)
        if value is not None:
            return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _literal_bindings(tree: ast.AST) -> dict[str, set[int]]:
    """Collect obvious scalar bindings used by launch configuration code.

    This intentionally handles assignments and literal ``for`` iterables.  It
    is enough to prove the common ``for nw in (1, 2, 4, 8)`` idiom without
    pretending that arbitrary Python data flow is statically understood.
    """
    bindings: dict[str, set[int]] = {}
    for item in ast.walk(tree):
        if isinstance(item, ast.Assign):
            value = _int_literal(item.value)
            if value is not None:
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        bindings.setdefault(target.id, set()).add(value)
        elif isinstance(item, ast.AnnAssign):
            value = _int_literal(item.value) if item.value is not None else None
            if value is not None and isinstance(item.target, ast.Name):
                bindings.setdefault(item.target.id, set()).add(value)
        elif isinstance(item, ast.For) and isinstance(item.target, ast.Name):
            values = item.iter.elts if isinstance(item.iter, (ast.Tuple, ast.List)) else []
            literals = {_int_literal(value) for value in values}
            literals.discard(None)
            if literals:
                bindings.setdefault(item.target.id, set()).update(literals)
    return bindings


def _triton_config_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "triton"
        and node.func.attr == "Config"
    )


def _config_option_findings(tree: ast.AST) -> list[dict]:
    """Reject config options outside the portable Triton ABI allowlist.

    The CPU model intentionally accepts arbitrary config options because it is
    not a target compiler.  That made ``multibuffer=True`` invisible until the
    Ascend backend compile.  Unknown options are therefore a hard gate: they need
    a target-version proof before a scarce submission, not a best-effort guess.
    """
    allowed = {"num_warps", "num_stages"}
    findings = []
    for node in [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _triton_config_call(n)]:
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in allowed:
                continue
            findings.append({
                "severity": "blocker",
                "kind": "unknown-config-option",
                "line": keyword.value.lineno,
                "option": keyword.arg,
                "message": (
                    f"triton.Config option {keyword.arg!r} is outside the portable ABI "
                    "allowlist; require target compiler/version evidence before submission."
                ),
            })
    return findings


def _autotuned_jit_names(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    result = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        has_jit = any(
            (isinstance(dec, ast.Attribute)
             and isinstance(dec.value, ast.Name)
             and dec.value.id == "triton" and dec.attr == "jit")
            or (isinstance(dec, ast.Name) and dec.id == "jit")
            for dec in fn.decorator_list
        )
        has_autotune = any(
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and isinstance(dec.func.value, ast.Name)
            and dec.func.value.id == "triton"
            and dec.func.attr == "autotune"
            for dec in fn.decorator_list
        )
        if has_jit and has_autotune:
            result[fn.name] = fn
    return result


def _autotune_config_keys(node: ast.FunctionDef) -> tuple[set[str], bool]:
    """Return literal Config keys and whether the full set was inspectable."""
    calls = [
        decorator
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "triton"
        and decorator.func.attr == "autotune"
    ]
    if len(calls) != 1:
        return set(), False
    configs = next((kw.value for kw in calls[0].keywords if kw.arg == "configs"), None)
    if not isinstance(configs, (ast.List, ast.Tuple)):
        return set(), False
    keys: set[str] = set()
    for config in configs.elts:
        if not isinstance(config, ast.Call) or not _triton_config_call(config):
            return keys, False
        if not config.args or not isinstance(config.args[0], ast.Dict):
            return keys, False
        for key in config.args[0].keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return keys, False
            keys.add(key.value)
    return keys, True


def _autotune_binding_findings(tree: ast.AST) -> list[dict]:
    """Catch explicit constexpr kwargs that duplicate literal autotune keys.

    Triton autotune implementations differ in how config kwargs are merged
    with launch kwargs.  A duplicate is concrete only when the launch keyword
    is also present in a config dictionary; merely passing another constexpr
    such as BN is safe when the configs tune BM only.  Unknown config layouts
    remain conservative for tile keywords.
    """
    tile_names = {"BT", "BH", "HS", "BN", "BC", "BR", "NRC"}
    autotuned = _autotuned_jit_names(tree)
    findings = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)):
            continue
        value = node.func.value
        if not isinstance(value, ast.Name) or value.id not in autotuned:
            continue
        constexpr = _constexpr_params(autotuned[value.id])
        config_keys, configs_inspected = _autotune_config_keys(autotuned[value.id])
        for keyword in node.keywords:
            if (keyword.arg in tile_names and keyword.arg in constexpr
                    and (keyword.arg in config_keys or not configs_inspected)):
                findings.append({
                    "severity": "blocker",
                    "kind": "autotune-explicit-tile-constexpr",
                    "line": keyword.value.lineno,
                    "function": value.id,
                    "argument": keyword.arg,
                    "message": (
                        f"autotuned kernel {value.id} receives tile constexpr "
                        f"{keyword.arg} explicitly; target autotune wrappers may merge "
                        "the same key and raise a duplicate-argument error."
                    ),
                })
    return findings


def _masked_pointer_findings(tree: ast.AST) -> list[dict]:
    """Reject masked loads whose pointer expression can be negative.

    A mask does not guarantee that every backend avoids evaluating or lowering
    an invalid pointer expression.  Normalize the index first (for example via
    tl.where) or use separate source paths.  This catches the Kunlunxin and
    Enflame ``cols - DN`` pattern that the CPU model cannot model faithfully.
    """
    findings = []
    for node in ast.walk(tree):
        if not _is_tl_load(node) or not node.args:
            continue
        pointer = ast.unparse(node.args[0])
        if not re.search(r"\b(?:col|cols|c|rc)\s*-\s*(?:DN|DR)\b", pointer):
            continue
        findings.append({
            "severity": "blocker",
            "kind": "masked-negative-pointer",
            "line": node.lineno,
            "pointer": pointer,
            "message": (
                "masked tl.load pointer contains a potentially negative column "
                "offset; materialize a nonnegative index before pointer arithmetic."
            ),
        })
    return findings


def _scalar_mask_findings(tree: ast.AST) -> list[dict]:
    """Flag scalar head predicates combined with vector load/store masks."""
    findings = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and _is_jit_decorator(n)]:
        scalar_masks = set()
        for item in ast.walk(fn):
            if not isinstance(item, ast.Assign) or not isinstance(item.value, ast.Compare):
                continue
            if len(item.targets) != 1 or not isinstance(item.targets[0], ast.Name):
                continue
            left = item.value.left
            if isinstance(left, ast.Name) and left.id in {"head", "token", "job"}:
                scalar_masks.add(item.targets[0].id)
        if not scalar_masks:
            continue
        for item in ast.walk(fn):
            if not (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)):
                continue
            if not (isinstance(item.func.value, ast.Name)
                    and item.func.value.id == "tl"
                    and item.func.attr in {"load", "store"}):
                continue
            mask_kw = next((kw for kw in item.keywords if kw.arg == "mask"), None)
            if mask_kw is None:
                continue
            if _names(mask_kw.value) & scalar_masks:
                findings.append({
                    "severity": "blocker",
                    "kind": "implicit-scalar-mask-broadcast",
                    "line": mask_kw.value.lineno,
                    "function": fn.name,
                    "message": (
                        "a scalar head/token predicate is combined directly with a "
                        "vector mask; materialize an explicit vector-shaped predicate "
                        "before target lowering."
                    ),
                })
    return findings


def _autotune_tile_grid_findings(tree: ast.AST) -> list[dict]:
    """Catch host-side loop counts fixed to a tile size not in autotune configs."""
    config_values: dict[str, set[int]] = {}
    for node in [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _triton_config_call(n)]:
        if not node.args or not isinstance(node.args[0], ast.Dict):
            continue
        for key, value in zip(node.args[0].keys, node.args[0].values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            literal = _int_literal(value)
            if literal is not None:
                config_values.setdefault(key.value, set()).add(literal)
    findings = []
    for tile_name, tile_values in config_values.items():
        if tile_name not in {"BC", "BN", "BR"} or len(tile_values) <= 1:
            continue
        if tile_name == "BC":
            match = re.search(r"triton\.cdiv\(\s*(?:dn|dr)\s*,\s*(\d+)\s*\)",
                              ast.unparse(tree))
            if match and tile_values != {int(match.group(1))}:
                findings.append({
                    "severity": "blocker",
                    "kind": "autotune-tile-grid-mismatch",
                    "line": match.string[:match.start()].count("\n") + 1,
                    "tile": tile_name,
                    "configured_tiles": sorted(tile_values),
                    "host_divisor": int(match.group(1)),
                    "message": (
                        f"autotune configs vary {tile_name}={sorted(tile_values)}, but host "
                        f"loop count is fixed at divisor {match.group(1)}; every selected "
                        "config must cover the same columns."
                    ),
                })
    return findings


def _num_warps_findings(tree: ast.AST, backend: str) -> list[dict]:
    """Audit every statically visible ``num_warps`` value.

    Triton accepts the Python object construction for values that a target
    compiler may reject.  In particular, Enflame rejects non-power-of-two
    warp counts.  Helper-based config construction (``_cfg(..., 12)``) is
    resolved through the helper's positional parameter as well.
    """
    findings: list[dict] = []
    bindings = _literal_bindings(tree)
    config_helpers: dict[str, int] = {}
    helper_param_names: set[str] = set()

    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _triton_config_call(n)]:
            keyword = next((kw for kw in call.keywords if kw.arg == "num_warps"), None)
            if keyword is None:
                continue
            if isinstance(keyword.value, ast.Name):
                params = [arg.arg for arg in fn.args.args]
                if keyword.value.id in params:
                    config_helpers[fn.name] = params.index(keyword.value.id)
                    helper_param_names.add(keyword.value.id)

    def check_value(value: int | None, line: int, context: str) -> None:
        if value is None:
            findings.append({
                "severity": "blocker",
                "kind": "unproven-num-warps",
                "line": line,
                "message": f"{context}: num_warps is not statically provable to be a positive power of two.",
            })
        elif not _is_power_of_two(value):
            findings.append({
                "severity": "blocker",
                "kind": "invalid-num-warps",
                "line": line,
                "value": value,
                "message": f"{context}: num_warps={value} is not a positive power of two.",
            })

    for node in [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _triton_config_call(n)]:
        keyword = next((kw for kw in node.keywords if kw.arg == "num_warps"), None)
        if keyword is None:
            continue
        value = _int_literal(keyword.value)
        if value is None and isinstance(keyword.value, ast.Name):
            if keyword.value.id in helper_param_names:
                # The helper call sites below carry the concrete values.
                continue
            values = bindings.get(keyword.value.id)
            if values and all(_is_power_of_two(item) for item in values):
                value = 1
            elif values:
                for item in sorted(values):
                    check_value(item, keyword.value.lineno, f"triton.Config")
                continue
        check_value(value, keyword.value.lineno, "triton.Config")

    for node in [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]:
        param_index = config_helpers.get(node.func.id)
        if param_index is None or param_index >= len(node.args):
            continue
        value_node = node.args[param_index]
        value = _int_literal(value_node)
        if value is None and isinstance(value_node, ast.Name):
            values = bindings.get(value_node.id)
            if values and all(_is_power_of_two(item) for item in values):
                continue
            if values:
                for item in sorted(values):
                    check_value(item, node.lineno, f"{node.func.id} helper")
                continue
        check_value(value, node.lineno, f"{node.func.id} helper")

    return findings


def _is_tl_load(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "load"
    )


def _is_load_cast(node: ast.AST) -> bool:
    """Match ``tl.load(...).to(COMMON)`` before a later broadcast."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
        and _is_tl_load(node.func.value)
    )


def _hygon_shape_findings(tree: ast.AST, backend: str) -> list[dict]:
    """Catch the Hygon lowering pattern that failed in the v22 FlagOS run."""
    if backend != "hygon":
        return []
    findings: list[dict] = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and _is_jit_decorator(n)]:
        load_cast_names: dict[str, int] = {}
        for item in ast.walk(fn):
            if isinstance(item, ast.Assign) and _is_load_cast(item.value):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        load_cast_names[target.id] = item.lineno
            if not (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)):
                continue
            if not (isinstance(item.func.value, ast.Name)
                    and item.func.value.id == "tl"
                    and item.func.attr == "broadcast_to"):
                continue
            if not item.args or not isinstance(item.args[0], ast.Name):
                continue
            source_name = item.args[0].id
            if source_name in load_cast_names:
                findings.append({
                    "severity": "blocker",
                    "kind": "hygon-cast-before-broadcast",
                    "line": item.lineno,
                    "source_line": load_cast_names[source_name],
                    "message": "Hygon-sensitive path casts a 1-D tl.load result before tl.broadcast_to; broadcast to the explicit 2-D store shape before casting, or use a backend-safe direct load.",
                })
    return findings


def _next_power_of_two_findings(tree: ast.AST) -> list[dict]:
    """Flag power-of-two tiles unless a visible ``min`` bound proves safety."""
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    bindings = _literal_bindings(tree)

    def visible_constant(node: ast.AST) -> int | None:
        value = _int_literal(node)
        if value is not None:
            return value
        if isinstance(node, ast.Name):
            values = bindings.get(node.id)
            if values and len(values) == 1:
                return next(iter(values))
        return None

    bounded_names: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not (isinstance(node.value.func, ast.Name) and node.value.func.id == "min"):
            continue
        limits = [value for value in (visible_constant(arg) for arg in node.value.args) if value is not None]
        if not limits:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                bounded_names[target.id] = min(limits)

    findings: list[dict] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "triton"
            and node.func.attr == "next_power_of_2"
        ):
            continue
        container = parent.get(node)
        bounded = False
        for _ in range(3):
            if (
                isinstance(container, ast.Call)
                and isinstance(container.func, ast.Name)
                and container.func.id == "min"
                and any(visible_constant(arg) is not None for arg in container.args)
            ):
                bounded = True
                break
            container = parent.get(container) if container is not None else None
        if not bounded and node.args and isinstance(node.args[0], ast.Name):
            bounded = node.args[0].id in bounded_names
        if not bounded:
            findings.append({
                "severity": "blocker",
                "kind": "uncapped-next-power-of-two",
                "line": node.lineno,
                "column": node.col_offset + 1,
                "expression": ast.unparse(node),
                "message": "next_power_of_2 result is not visibly capped; large dimensions can create invalid or very slow tiles.",
            })
    return findings


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
        if "eviction_policy" in line or "cache_modifier" in line:
            findings.append({
                "severity": "warning",
                "kind": "target-cache-hint",
                "line": line_no,
                "message": "target-specific cache hint requires compiler evidence; do not infer support from CPU validation.",
            })

    base_name = "concat_and_cast_mha_k"
    backend = "default" if path.stem == base_name else path.stem[len(base_name) + 1:]
    findings.extend(_next_power_of_two_findings(tree))
    findings.extend(_num_warps_findings(tree, backend))
    findings.extend(_hygon_shape_findings(tree, backend))
    findings.extend(_config_option_findings(tree))
    findings.extend(_autotune_binding_findings(tree))
    findings.extend(_masked_pointer_findings(tree))
    findings.extend(_scalar_mask_findings(tree))
    findings.extend(_autotune_tile_grid_findings(tree))

    return {
        "path": str(path.resolve()),
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
