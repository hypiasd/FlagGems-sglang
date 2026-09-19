#!/usr/bin/env python3
"""Apply deterministic acceptance gates to a saved KernelGen JSON response.

This script never executes generated Triton code. It checks the response
envelope, source syntax, public entry, and (for target phase) the presence of
non-empty correctness and performance evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any


def _decode_nested(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def unwrap_response(value: Any) -> dict[str, Any]:
    """Accept a saved structured result or an MCP JSON-RPC wrapper."""
    value = _decode_nested(value)
    if not isinstance(value, dict):
        return {}
    for key in ("structuredContent", "result"):
        nested = value.get(key)
        if isinstance(nested, dict):
            candidate = unwrap_response(nested)
            if candidate:
                return candidate
    content = value.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                candidate = unwrap_response(item.get("text"))
                if candidate:
                    return candidate
    return value


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def gate(data: dict[str, Any], phase: str, public_symbol: str,
         baseline_speedup: float | None) -> dict[str, Any]:
    reasons: list[str] = []
    code = data.get("triton_code") or data.get("code")
    if data.get("success") is False or data.get("error"):
        reasons.append("service reported failure")
    if not isinstance(code, str) or not code.strip():
        reasons.append("missing triton_code")
        code = ""

    tree = None
    if code:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            reasons.append(f"source syntax error: {exc.msg} at line {exc.lineno}")

    function_names = set()
    if tree is not None:
        function_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    if public_symbol not in function_names:
        reasons.append(f"missing public function: {public_symbol}")

    if tree is not None and any(isinstance(node, ast.Try) for node in ast.walk(tree)):
        reasons.append("try/except fallback is present")

    verification = data.get("verify_result")
    if phase == "target":
        if not isinstance(verification, dict):
            reasons.append("missing verify_result")
        else:
            total = verification.get("total_tests")
            passed = verification.get("passed_tests")
            if not isinstance(total, int) or total <= 0:
                reasons.append("verification has no executed tests")
            elif passed != total:
                reasons.append(f"verification incomplete: {passed}/{total}")

        performance = data.get("performance_result")
        speedup = performance.get("speedup") if isinstance(performance, dict) else None
        speedup = _number(speedup)
        if speedup is None:
            reasons.append("missing numeric performance result")
        elif baseline_speedup is not None and speedup <= baseline_speedup:
            reasons.append(f"speedup {speedup:g} does not beat baseline {baseline_speedup:g}")

    if reasons:
        deterministic = (
            "service reported failure",
            "missing triton_code",
            "source syntax",
            "missing public",
            "try/except",
        )
        state = "rejected" if any(reason.startswith(d) for reason in reasons
                                   for d in deterministic) else "inconclusive"
    elif phase == "target":
        state = "promotable"
    else:
        state = "generated"

    return {"state": state, "reasons": reasons, "public_symbol": public_symbol}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path, help="saved MCP JSON response")
    parser.add_argument("--phase", choices=("candidate", "target"), default="candidate")
    parser.add_argument("--public-symbol", required=True)
    parser.add_argument("--baseline-speedup", type=float)
    args = parser.parse_args(argv)

    try:
        raw = json.loads(args.response.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"state": "rejected", "reasons": [str(exc)]}), file=sys.stderr)
        return 2

    result = gate(unwrap_response(raw), args.phase, args.public_symbol,
                  args.baseline_speedup)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["state"] in ("generated", "promotable") else 1


if __name__ == "__main__":
    raise SystemExit(main())

