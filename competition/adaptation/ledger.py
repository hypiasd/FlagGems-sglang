"""Immutable official observations, with explicit revisions while evaluating."""
from __future__ import annotations
import json
import math
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


def summary(task):
    latest = {row["record_id"]: row for row in rows(task)}
    usable = [row for row in latest.values() if not row.get("excluded") and row.get("status") == "completed" and row.get("aggregate_speedup") is not None]
    best = max(usable, key=lambda row: row["aggregate_speedup"], default=None)
    return {"task_id": task, "record_count": len(latest), "observation_count": len(rows(task)), "best_complete_record": best, "records": list(latest.values()), "source": "official-platform"}
