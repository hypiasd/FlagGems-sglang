"""Immutable official observations, with explicit revisions while evaluating."""
from __future__ import annotations
import json
import math
import statistics
from pathlib import Path
from competition.experiments.store import ROOT, profile


def rows(task):
    result = []
    for name in ("results.jsonl", "official-records.jsonl"):
        path = ROOT / "competition" / task / name
        if path.exists():
            result += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return result


def validate(row, contract):
    errors = []
    if not row.get("record_id") or not row.get("submitted_at"):
        errors.append("official record ID and submission time are required")
    # The lifecycle field is what target_ledger filters on; a row without it is
    # silently invisible to every per-target report, so it must be explicit.
    if row.get("status") not in {"submitted", "evaluating", "completed"}:
        errors.append("status must be one of submitted/evaluating/completed")
    if row.get("evidence_class") != "official-platform" or not row.get("evidence"):
        errors.append("official platform evidence is required; GPU experiments are not official scores")
    digest = row.get("local_archive_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        errors.append("package SHA-256 is required")
    if set(row.get("targets", {})) != set(contract["targets"]):
        errors.append("target matrix differs from this task's contract")
    passes = 0
    for target, item in row.get("targets", {}).items():
        score, state = item.get("speedup"), item.get("status")
        if state == "pass":
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score <= 0:
                errors.append(f"{target}: pass requires a finite positive score")
            else:
                passes += 1
        elif state not in {"fail", "evaluating", "pending"} or score is not None:
            errors.append(f"{target}: invalid status/score pair")
        source_hash = item.get("source_sha256")
        if source_hash is not None and (len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash)):
            errors.append(f"{target}: invalid source digest")
    if row.get("pass_count") != passes:
        errors.append("pass count mismatch")
    aggregate = row.get("aggregate_speedup")
    if aggregate is not None and (passes != len(contract["targets"]) or isinstance(aggregate, bool) or not isinstance(aggregate, (int, float)) or not math.isfinite(aggregate) or aggregate <= 0):
        errors.append("aggregate requires all targets to pass and a finite positive official value")
    return errors


def record(task, row):
    errors = validate(row, profile(task))
    if errors:
        raise ValueError("; ".join(errors))
    previous = [r for r in rows(task) if r["record_id"] == row["record_id"]]
    if previous:
        old = previous[-1]
        if {k: v for k, v in old.items() if k != "revision"} == {k: v for k, v in row.items() if k != "revision"}:
            return {"status": "unchanged", "record_id": row["record_id"]}
        if old.get("status") not in {"submitted", "evaluating"} or old.get("local_archive_sha256") != row["local_archive_sha256"]:
            raise ValueError("conflicting immutable official record")
        row = {**row, "revision": len(previous) + 1}
    path = ROOT / "competition" / task / "official-records.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"status": "recorded", "record_id": row["record_id"]}


def _scored(row, target):
    """One target's usable score observation, or None if it is not a pass."""
    item = (row.get("targets") or {}).get(target) or {}
    score = item.get("speedup")
    if item.get("status") != "pass" or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return None
    return {"speedup": score, "source_sha256": item.get("source_sha256"), "record_id": row["record_id"], "submitted_at": row["submitted_at"]}


def _eligible(row, targets):
    """An eligible package passed every target and has a finite aggregate."""
    aggregate = row.get("aggregate_speedup")
    return row.get("pass_count") == len(targets) and isinstance(aggregate, (int, float)) and not isinstance(aggregate, bool) and math.isfinite(aggregate) and aggregate > 0


def _closest(observations, key, better):
    best = None
    for item in observations:
        if item is None:
            continue
        if best is None or better(item[key], best[key]):
            best = item
    return best


def _resolution(observations):
    """Measurement resolution per source: repeated observations of identical bytes.

    Two packages that carry the exact same source for a target measure the same
    code twice.  Their spread is the platform's resolution for that target, so a
    difference smaller than it is not a version regression.  With one or two
    observations this is a warning line, not a statistical bound.
    """
    groups = {}
    for item in observations:
        if item is None or not item.get("source_sha256"):
            continue
        groups.setdefault(item["source_sha256"], []).append(item["speedup"])
    detail, widest = [], None
    for digest, values in sorted(groups.items()):
        if len(values) < 2:
            continue
        median = statistics.median(values)
        if median <= 0:
            continue
        entry = {
            "source_sha256": digest,
            "samples": len(values),
            "min": min(values),
            "max": max(values),
            "median": median,
            "relative_range": (max(values) - min(values)) / median,
            "relative_mad": max(abs(value - median) for value in values) / median,
        }
        detail.append(entry)
        if widest is None or entry["relative_range"] > widest["relative_range"]:
            widest = entry
    if widest is None:
        return None, "no repeated observation of an identical source", detail
    return widest["relative_range"], f"widest repeated spread over {widest['samples']} samples of one source ({widest['source_sha256'][:8]})", detail


def target_ledger(task, records=None):
    """Cross-package per-target best table plus the measurement resolution.

    The aggregate best answers "which single package scored highest"; it does
    not answer "which package held the best value for this chip".  This table
    does, and it separates a real version regression (the source changed) from
    platform re-measurement noise (the source is byte-identical).

    ``records`` exists so the table can be computed from a supplied observation
    list instead of the on-disk ledger; it does not change the semantics.
    """
    contract = profile(task)
    targets = list(contract["targets"])
    observations = records if records is not None else rows(task)
    latest = {row["record_id"]: row for row in observations}
    observations = [row for row in latest.values() if not row.get("excluded") and row.get("status") == "completed"]
    eligible = [row for row in observations if _eligible(row, targets)]
    table = {}
    for target in targets:
        seen = [_scored(row, target) for row in observations]
        usable = [item for item in seen if item]
        eligible_items = [item for item in usable if _eligible(latest[item["record_id"]], targets)]
        best_eligible = _closest(eligible_items, "speedup", lambda a, b: a > b)
        best_observed = _closest(usable, "speedup", lambda a, b: a > b)
        ordered = sorted(usable, key=lambda item: item["submitted_at"])
        current = ordered[-1] if ordered else None
        resolution, basis, detail = _resolution(seen)
        delta = None
        if current and best_eligible:
            difference = best_eligible["speedup"] - current["speedup"]
            same_source = bool(best_eligible["source_sha256"]) and best_eligible[
                "source_sha256"
            ] == current["source_sha256"]
            if difference <= 0:
                verdict = "at_or_above_best"
            elif same_source:
                # Byte-identical source measured again: this is re-measurement,
                # not a version regression.  The observed spread for that same
                # source is reported next to the delta so the reader can judge
                # the size, but it is not a statistical test.
                verdict = "below_best_same_source"
            else:
                verdict = "below_best_changed_source"
            spread = None
            for entry in detail:
                if entry["source_sha256"] == current["source_sha256"]:
                    spread = entry["relative_range"]
            delta = {
                "absolute": best_eligible["speedup"] - current["speedup"],
                "relative": (best_eligible["speedup"] - current["speedup"]) / best_eligible["speedup"],
                "same_source": same_source,
                "latest_source_sha256": current["source_sha256"],
                "best_source_sha256": best_eligible["source_sha256"],
                "observed_spread_for_latest_source": spread,
                "verdict": verdict,
            }
        table[target] = {
            "source_versions": len({item["source_sha256"] for item in usable if item.get("source_sha256")}),
            "best_eligible": best_eligible,
            "best_observed": best_observed,
            "latest": current,
            "delta_vs_best_eligible": delta,
            "resolution": resolution,
            "resolution_basis": basis,
            "repeated_observations": detail,
        }
    return {
        "task_id": task,
        "targets": table,
        "eligible_records": [
            row["record_id"]
            for row in sorted(eligible, key=lambda row: row["submitted_at"])
        ],
        "completed_observations": [
            row["record_id"]
            for row in sorted(observations, key=lambda row: row["submitted_at"])
        ],
        "limitation": (
            "one platform observation per package per target; the observed "
            "spread is computed from the same observations it is compared "
            "against, so 'below_best_same_source' means 'not distinguishable "
            "from re-measurement', not 'no regression'"
        ),
        "source": "official-platform",
    }


def summary(task):
    latest = {row["record_id"]: row for row in rows(task)}
    usable = [row for row in latest.values() if not row.get("excluded") and row.get("status") == "completed" and row.get("aggregate_speedup") is not None]
    best = max(usable, key=lambda row: row["aggregate_speedup"], default=None)
    return {"task_id": task, "record_count": len(latest), "observation_count": len(rows(task)), "best_complete_record": best, "targets": target_ledger(task)["targets"], "records": list(latest.values()), "source": "official-platform"}
