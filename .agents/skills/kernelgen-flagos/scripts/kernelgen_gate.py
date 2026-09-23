#!/usr/bin/env python3
"""Validate a saved KernelGen artifact and normalized target evidence.

This gate does not execute Triton or authenticate a target service. Candidate
mode checks the response/source envelope. Target mode checks consistency of
reported compile, correctness, and repeated timings against exact source files,
and labels those as reports/claims. It never makes a cross-target promotion
decision or computes an operator's official score.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple


def _decode_nested(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def unwrap_response(value: Any) -> Dict[str, Any]:
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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_samples(value: Any, minimum: int) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) < minimum:
        return None
    samples: List[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0:
            return None
        samples.append(number)
    return samples


def _relative_mad(samples: List[float]) -> float:
    center = statistics.median(samples)
    return statistics.median(abs(x - center) for x in samples) / center


def _max_relative_deviation(samples: List[float]) -> float:
    center = statistics.median(samples)
    return max(abs(x - center) for x in samples) / center


def _target_gate(
    data: Dict[str, Any],
    code: str,
    evidence: Optional[Dict[str, Any]],
    candidate_source: Optional[str],
    baseline_source: Optional[str],
    minimum_speedup: float,
    max_relative_noise: float,
    raw_result_sha256: Optional[str],
) -> Dict[str, Any]:
    reasons: List[str] = []
    if evidence is None:
        return {
            "state": "inconclusive",
            "correctness_state": "unknown",
            "performance_state": "unknown",
            "reasons": ["missing target-evidence manifest"],
        }

    schema_version = evidence.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
        reasons.append("missing or unsupported target-evidence schema_version")
    if not _nonempty_string(evidence.get("run_id")):
        reasons.append("missing target-evidence run_id")

    if candidate_source is None or baseline_source is None:
        return {
            "state": "inconclusive",
            "correctness_state": "unknown",
            "performance_state": "unknown",
            "reasons": ["target phase requires exact candidate and baseline source files"],
        }

    if candidate_source != code:
        return {
            "state": "rejected",
            "correctness_state": "unknown",
            "performance_state": "unknown",
            "reasons": ["candidate file differs from the source returned by KernelGen"],
        }

    target = evidence.get("target")
    if not isinstance(target, dict):
        target = {}
    target_fields = ("backend", "device", "compiler", "runtime")
    missing_target = [name for name in target_fields if not _nonempty_string(target.get(name))]
    if missing_target:
        reasons.append("target identity incomplete: " + ", ".join(missing_target))

    provenance = evidence.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance_linked = (
        provenance.get("kind") == "captured-live-tool-result"
        and _nonempty_string(provenance.get("provider"))
        and _nonempty_string(provenance.get("invocation_id"))
        and _nonempty_string(provenance.get("raw_result_sha256"))
        and provenance.get("raw_result_sha256") == raw_result_sha256
    )

    expected_candidate_hash = _sha256(candidate_source)
    expected_baseline_hash = _sha256(baseline_source)
    supplied_candidate_hash = evidence.get("source_sha256")
    supplied_baseline_hash = evidence.get("baseline_source_sha256")
    if not _nonempty_string(supplied_candidate_hash):
        reasons.append("missing candidate source hash")
    elif supplied_candidate_hash != expected_candidate_hash:
        return {
            "state": "rejected",
            "correctness_state": "unknown",
            "performance_state": "unknown",
            "reasons": ["target evidence source hash does not match candidate file"],
        }
    if not _nonempty_string(supplied_baseline_hash):
        reasons.append("missing baseline source hash")
    elif supplied_baseline_hash != expected_baseline_hash:
        return {
            "state": "rejected",
            "correctness_state": "unknown",
            "performance_state": "unknown",
            "reasons": ["target evidence baseline hash does not match baseline file"],
        }

    compile_result = evidence.get("compile")
    if not isinstance(compile_result, dict) or not isinstance(compile_result.get("success"), bool):
        reasons.append("missing target compile outcome")
        compile_state = "unknown"
    elif compile_result["success"] is False:
        return {
            "state": "rejected",
            "correctness_state": "unknown",
            "performance_state": "unknown",
            "reasons": ["target compilation failed"],
        }
    else:
        compile_state = "passed"

    correctness = evidence.get("correctness")
    if not isinstance(correctness, dict):
        correctness = {}
    suite_id = correctness.get("suite_id")
    total = correctness.get("total_cases")
    passed = correctness.get("passed_cases")
    cases = correctness.get("cases")
    if not _nonempty_string(suite_id):
        reasons.append("missing correctness suite identity")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        reasons.append("correctness suite executed zero or unknown cases")
    elif not isinstance(passed, int) or isinstance(passed, bool):
        reasons.append("missing correctness pass count")
    elif passed != total:
        return {
            "state": "rejected",
            "correctness_state": "failed",
            "performance_state": "unknown",
            "reasons": ["target correctness did not pass every executed case: " + str(passed) + "/" + str(total)],
        }

    case_ids = set()
    if not isinstance(cases, list) or not isinstance(total, int) or len(cases) != total:
        reasons.append("correctness manifest must enumerate every executed case and input signature")
    else:
        for case in cases:
            if not isinstance(case, dict):
                reasons.append("malformed correctness case")
                break
            case_id = case.get("case_id")
            signature = case.get("input_signature")
            if not _nonempty_string(case_id) or not _nonempty_string(signature):
                reasons.append("correctness case is missing case_id or input_signature")
                break
            if case_id in case_ids:
                reasons.append("duplicate correctness case_id: " + case_id)
                break
            case_ids.add(case_id)
            if case.get("status") not in ("pass", "passed"):
                return {
                    "state": "rejected",
                    "correctness_state": "failed",
                    "performance_state": "unknown",
                    "reasons": ["target case failed or has unknown status: " + case_id],
                }

    if reasons:
        return {
            "state": "inconclusive",
            "correctness_state": "unknown" if compile_state != "passed" else "incomplete",
            "performance_state": "unknown",
            "target": target,
            "reasons": reasons,
        }

    correctness_state = "passed"
    benchmark = evidence.get("benchmark")
    performance_reasons: List[str] = []
    if not isinstance(benchmark, dict):
        performance_reasons.append("missing benchmark record")
        benchmark = {}
    if not _nonempty_string(benchmark.get("method")):
        performance_reasons.append("missing benchmark method")
    warmups = benchmark.get("warmup_runs")
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 1:
        performance_reasons.append("benchmark must record at least one warm-up")

    benchmark_cases = benchmark.get("cases")
    measured: List[Dict[str, Any]] = []
    if not isinstance(benchmark_cases, list) or not benchmark_cases:
        performance_reasons.append("benchmark has no per-case raw timings")
    else:
        seen_bench_ids = set()
        for case in benchmark_cases:
            if not isinstance(case, dict):
                performance_reasons.append("malformed benchmark case")
                break
            case_id = case.get("case_id")
            signature = case.get("input_signature")
            if not _nonempty_string(case_id) or case_id not in case_ids:
                performance_reasons.append("benchmark case is not linked to a passed correctness case")
                break
            if case_id in seen_bench_ids:
                performance_reasons.append("duplicate benchmark case_id: " + case_id)
                break
            seen_bench_ids.add(case_id)
            matching = next((x for x in cases if x.get("case_id") == case_id), None)
            if not _nonempty_string(signature) or matching is None or signature != matching.get("input_signature"):
                performance_reasons.append("benchmark input signature differs from correctness case: " + case_id)
                break
            baseline_samples = _positive_samples(case.get("baseline_ms"), 5)
            candidate_samples = _positive_samples(case.get("candidate_ms"), 5)
            if baseline_samples is None or candidate_samples is None:
                performance_reasons.append("each benchmark case needs at least five positive raw baseline and candidate timings: " + case_id)
                break
            base_median = statistics.median(baseline_samples)
            candidate_median = statistics.median(candidate_samples)
            measured.append({
                "case_id": case_id,
                "baseline_median_ms": base_median,
                "candidate_median_ms": candidate_median,
                "speedup": base_median / candidate_median,
                "baseline_relative_mad": _relative_mad(baseline_samples),
                "candidate_relative_mad": _relative_mad(candidate_samples),
                "baseline_max_relative_deviation": _max_relative_deviation(baseline_samples),
                "candidate_max_relative_deviation": _max_relative_deviation(candidate_samples),
            })

    if performance_reasons:
        return {
            "state": "target_reported",
            "correctness_state": "reported_pass",
            "performance_state": "reported_incomplete",
            "evidence_provenance": "raw_result_linked" if provenance_linked else "unverified_claim",
            "task_level_case_coverage_complete": False,
            "target": target,
            "reasons": performance_reasons,
        }

    case_contract = benchmark.get("case_contract")
    if not isinstance(case_contract, dict):
        case_contract = {}
    required_case_ids = case_contract.get("required_case_ids")
    required_case_ids_valid = (
        isinstance(required_case_ids, list)
        and bool(required_case_ids)
        and all(_nonempty_string(case_id) for case_id in required_case_ids)
        and len(set(required_case_ids)) == len(required_case_ids)
        and _nonempty_string(case_contract.get("case_set_id"))
        and _nonempty_string(case_contract.get("score_rule"))
    )
    measured_case_ids = {case["case_id"] for case in measured}
    task_case_coverage_complete = (
        required_case_ids_valid
        and measured_case_ids == set(required_case_ids)
        and set(required_case_ids).issubset(case_ids)
    )

    speedups = [case["speedup"] for case in measured]
    geometric_mean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    unstable_cases = [
        case["case_id"] for case in measured
        if max(
            case["baseline_relative_mad"], case["candidate_relative_mad"],
            case["baseline_max_relative_deviation"],
            case["candidate_max_relative_deviation"],
        ) > max_relative_noise
    ]
    result: Dict[str, Any] = {
        "state": "performance_reported_complete" if task_case_coverage_complete else "performance_reported_subset",
        "correctness_state": "reported_pass",
        "performance_state": "reported_complete" if task_case_coverage_complete else "reported_subset",
        "evidence_provenance": "raw_result_linked" if provenance_linked else "unverified_claim",
        "task_level_case_coverage_complete": bool(task_case_coverage_complete),
        "target": target,
        "per_case": measured,
        "diagnostic_geometric_mean_speedup": geometric_mean,
        "official_score_computed": False,
        "reasons": [] if task_case_coverage_complete else [
            "no adapter-declared complete case set; aggregate is diagnostic subset only"
        ],
    }
    if unstable_cases:
        result["state"] = "performance_reported_unstable"
        result["performance_state"] = "reported_unstable"
        result["reasons"].append("relative MAD exceeds threshold for: " + ", ".join(unstable_cases))
    elif geometric_mean <= minimum_speedup:
        result["state"] = "performance_reported_no_improvement"
        result["performance_state"] = "reported_no_improvement"
        result["reasons"].append(
            "geometric-mean speedup " + format(geometric_mean, ".6g")
            + " does not exceed minimum " + format(minimum_speedup, ".6g")
        )
    return result


def gate(
    data: Dict[str, Any],
    phase: str,
    public_symbol: str,
    target_evidence: Optional[Dict[str, Any]] = None,
    candidate_source: Optional[str] = None,
    baseline_source: Optional[str] = None,
    minimum_speedup: float = 1.0,
    max_relative_noise: float = 0.10,
    raw_result_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    reasons: List[str] = []
    code = data.get("triton_code") or data.get("code")
    if data.get("success") is False or data.get("error"):
        reasons.append("service reported failure")
    if not isinstance(code, str) or not code.strip():
        reasons.append("missing source artifact")
        code = ""

    tree = None
    if code:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            reasons.append("source syntax error: " + exc.msg + " at line " + str(exc.lineno))

    function_names = set()
    if tree is not None:
        function_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    if public_symbol not in function_names:
        reasons.append("missing public function: " + public_symbol)
    if tree is not None and any(isinstance(node, ast.Try) for node in ast.walk(tree)):
        reasons.append("try/except fallback is present")

    if reasons:
        return {"state": "rejected", "reasons": reasons, "public_symbol": public_symbol}
    if phase == "candidate":
        return {"state": "generated", "reasons": [], "public_symbol": public_symbol}
    return _target_gate(
        data, code, target_evidence, candidate_source, baseline_source,
        minimum_speedup, max_relative_noise, raw_result_sha256,
    )


def _read_text(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.read_bytes().decode("utf-8")
    except OSError:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path, help="saved MCP JSON response")
    parser.add_argument("--phase", choices=("candidate", "target"), default="candidate")
    parser.add_argument("--public-symbol", required=True)
    parser.add_argument("--target-evidence", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--raw-target-result", type=Path,
                        help="unmodified raw result from the target service/tool")
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    parser.add_argument("--max-relative-noise", type=float, default=0.10)
    args = parser.parse_args(argv)
    if not math.isfinite(args.minimum_speedup) or args.minimum_speedup <= 0:
        parser.error("--minimum-speedup must be a finite positive number")
    if not math.isfinite(args.max_relative_noise) or args.max_relative_noise < 0:
        parser.error("--max-relative-noise must be a finite non-negative number")

    try:
        raw = json.loads(args.response.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"state": "rejected", "reasons": [str(exc)]}), file=sys.stderr)
        return 2

    evidence = None
    if args.target_evidence is not None:
        try:
            parsed = json.loads(args.target_evidence.read_text(encoding="utf-8"))
            evidence = parsed if isinstance(parsed, dict) else None
        except (OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"state": "inconclusive", "reasons": [str(exc)]}), file=sys.stderr)
            return 2

    raw_result_sha256 = None
    if args.raw_target_result is not None:
        try:
            raw_result_sha256 = hashlib.sha256(args.raw_target_result.read_bytes()).hexdigest()
        except OSError as exc:
            print(json.dumps({"state": "reported_only", "reasons": [str(exc)]}), file=sys.stderr)
            return 2
    result = gate(
        unwrap_response(raw), args.phase, args.public_symbol, evidence,
        _read_text(args.candidate_source), _read_text(args.baseline_source),
        args.minimum_speedup, args.max_relative_noise, raw_result_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    accepted_states = {
        "generated", "target_reported", "performance_reported_complete",
        "performance_reported_subset", "performance_reported_no_improvement",
        "performance_reported_unstable",
    }
    return 0 if result.get("state") in accepted_states else 1


if __name__ == "__main__":
    raise SystemExit(main())
