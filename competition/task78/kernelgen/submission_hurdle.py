#!/usr/bin/env python3
"""Score a pre-registered performance forecast against the chip-wise champion set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_mixed_candidate as candidate
import task78_results as results

TARGETS = results.TARGETS
MIN_RELATIVE_GAIN = 0.05
MIN_ABSOLUTE_MEAN = 1.50
MIN_TARGET_FLOOR_PCT = -5.0
ACCEPTED_EVIDENCE = {"same_target_measurement", "historical_structural_comparison"}


def baseline_composite(rows: list[dict], archive_root: Path) -> dict[str, float]:
    champions = results.champions(rows, archive_root)
    scores = {}
    for target in TARGETS:
        champion = champions[target]
        digest = champion["source_sha256"]
        repeated = [
            float(row["targets"][target]["speedup"])
            for row in rows
            if row["targets"][target]["status"] == "pass"
            and digest
            and results.target_source_hash(row, target, archive_root)[0] == digest
        ]
        scores[target] = statistics.median(repeated) if repeated else champion["score"]
    shared_row, shared = candidate.joint_generic_champion(rows, archive_root)
    del shared_row
    scores["intl_a"] = shared["per_target_median"]["intl_a"]
    scores["intl_b"] = shared["per_target_median"]["intl_b"]
    return scores


def evaluate(manifest: dict, rows: list[dict], archive_root: Path) -> dict:
    baseline = baseline_composite(rows, archive_root)
    forecast = manifest.get("performance_forecast", {})
    errors = []
    if set(forecast) != set(TARGETS):
        errors.append("performance_forecast must contain exactly all eight target IDs")
    expected_delta = {}
    lower_delta = {}
    for target in TARGETS:
        item = forecast.get(target, {})
        for field in ("expected_delta_pct", "lower_delta_pct", "confidence", "evidence_basis", "evidence"):
            if field not in item:
                errors.append(f"{target}: missing {field}")
        if not isinstance(item.get("expected_delta_pct"), (int, float)):
            continue
        if not isinstance(item.get("lower_delta_pct"), (int, float)):
            continue
        expected_delta[target] = float(item["expected_delta_pct"])
        lower_delta[target] = float(item["lower_delta_pct"])
        if not math.isfinite(expected_delta[target]) or not math.isfinite(lower_delta[target]):
            errors.append(f"{target}: forecast deltas must be finite numbers")
            continue
        if expected_delta[target] < lower_delta[target]:
            errors.append(f"{target}: expected delta is below its lower bound")
        if item.get("confidence") not in {"medium", "high"}:
            errors.append(f"{target}: confidence must be medium/high to spend a submission")
        if item.get("evidence_basis") not in ACCEPTED_EVIDENCE or not item.get("evidence"):
            errors.append(f"{target}: forecast needs target measurement or a historical structural comparison")
        if lower_delta[target] < MIN_TARGET_FLOOR_PCT:
            errors.append(f"{target}: lower-bound regression exceeds {MIN_TARGET_FLOOR_PCT:.0f}%")
    if errors:
        return {"passed": False, "errors": errors, "baseline_scores": baseline}
    if any(not math.isfinite(score) or score <= 0 for score in baseline.values()):
        return {"passed": False, "errors": ["champion composite contains non-finite or non-positive scores"],
                "baseline_scores": baseline}
    base_mean = sum(baseline.values()) / len(TARGETS)
    projected_scores = {target: baseline[target] * (1 + expected_delta[target] / 100)
                        for target in TARGETS}
    lower_scores = {target: baseline[target] * (1 + lower_delta[target] / 100)
                    for target in TARGETS}
    expected_mean = sum(projected_scores.values()) / len(TARGETS)
    lower_mean = sum(lower_scores.values()) / len(TARGETS)
    required_mean = max(base_mean * (1 + MIN_RELATIVE_GAIN), MIN_ABSOLUTE_MEAN)
    if expected_mean < required_mean:
        errors.append(
            f"expected aggregate {expected_mean:.4f}x is below the {required_mean:.4f}x release hurdle "
            f"(at least {MIN_RELATIVE_GAIN:.0%} above baseline and at least {MIN_ABSOLUTE_MEAN:.2f}x)"
        )
    if lower_mean < base_mean:
        errors.append(f"lower-bound aggregate {lower_mean:.4f}x does not preserve the champion composite {base_mean:.4f}x")
    return {
        "passed": not errors,
        "hurdle_relative_gain": MIN_RELATIVE_GAIN,
        "hurdle_absolute_mean": MIN_ABSOLUTE_MEAN,
        "baseline_scores": baseline,
        "baseline_mean": base_mean,
        "expected_deltas_pct": expected_delta,
        "lower_deltas_pct": lower_delta,
        "projected_scores": projected_scores,
        "projected_mean": expected_mean,
        "lower_bound_scores": lower_scores,
        "lower_bound_mean": lower_mean,
        "errors": errors,
        "limitation": "Forecasts are not measurements. Do not pass by inventing percentages; evidence and calibration against later official results are required.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--ledger", type=Path, default=results.LEDGER)
    parser.add_argument("--archive-root", type=Path, default=results.ROOT)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        rows = results.load_ledger(args.ledger)
        ledger_errors = results.validate_ledger(rows, args.archive_root)
        if ledger_errors:
            report = {"passed": False, "errors": ledger_errors}
        else:
            report = evaluate(manifest, rows, args.archive_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {"passed": False, "errors": [str(exc)]}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
