#!/usr/bin/env python3
"""Create an isolated seven-file candidate from per-chip best-observed sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task78_results as results

RUNS = HERE / "candidates"
BACKENDS = ("default", "ascend", "enflame", "hygon", "iluvatar", "kunlunxin", "metax")


def output_name(backend: str) -> str:
    suffix = "" if backend == "default" else f"_{backend}"
    return f"concat_and_cast_mha_k{suffix}.py"


def source_bytes(archive_path: Path, backend: str) -> tuple[str, bytes]:
    specific = output_name(backend)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        member = specific if specific in names else output_name("default")
        if member not in names:
            raise ValueError(f"{archive_path}: neither {specific} nor generic source exists")
        return member, archive.read(member)


def joint_generic_champion(rows: list[dict], archive_root: Path) -> tuple[dict, dict]:
    """Choose one generic source using both international backends, not A alone."""
    enriched = results.enrich(rows, archive_root)
    groups: dict[str, dict] = {}
    for row in enriched:
        for target in ("intl_a", "intl_b"):
            item = row["targets"][target]
            digest = item.get("source_sha256")
            if item["status"] != "pass" or not digest:
                continue
            group = groups.setdefault(digest, {"intl_a": [], "intl_b": [], "rows": []})
            group[target].append(float(item["speedup"]))
            if target == "intl_a" and row not in group["rows"]:
                group["rows"].append(row)
    candidates = []
    for digest, group in groups.items():
        if not group["intl_a"] or not group["intl_b"]:
            continue
        # Median each backend first, so a one-off peak on either side does not
        # decide which shared source becomes the next baseline.
        medians = {target: sorted(group[target])[len(group[target]) // 2]
                   if len(group[target]) % 2 else
                   (sorted(group[target])[len(group[target]) // 2 - 1]
                    + sorted(group[target])[len(group[target]) // 2]) / 2
                   for target in ("intl_a", "intl_b")}
        candidates.append((sum(medians.values()) / 2, digest, group, medians))
    if not candidates:
        raise ValueError("no archived generic source has passing observations on both Intl A and B")
    _, digest, group, medians = max(
        candidates,
        key=lambda item: (item[0], min(row["submitted_at"] for row in item[2]["rows"])),
    )
    source_rows = [row for row in group["rows"]
                   if row["targets"]["intl_a"].get("source_sha256") == digest
                   and row["targets"]["intl_b"].get("source_sha256") == digest
                   and row["targets"]["intl_a"]["status"] == "pass"
                   and row["targets"]["intl_b"]["status"] == "pass"]
    if not source_rows:
        raise ValueError("shared generic source has no single package passing both targets")
    chosen = min(source_rows, key=lambda row: row["submitted_at"])
    return chosen, {"source_sha256": digest, "per_target_median": medians,
                   "objective_mean": sum(medians.values()) / 2}


def prepare(run_id: str, ledger_path: Path = results.LEDGER,
            archive_root: Path = results.ROOT) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id) or run_id in {".", ".."}:
        raise ValueError("run-id must be a simple 1-80 character directory name")
    run_root = RUNS / run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite candidate run: {run_root}")
    rows = results.load_ledger(ledger_path)
    errors = results.validate_ledger(rows, archive_root)
    if errors:
        raise ValueError("result ledger failed validation:\n" + "\n".join(errors))
    champions = results.champions(rows, archive_root)
    generic_row, generic_stats = joint_generic_champion(rows, archive_root)
    selections: dict[str, dict] = {}
    payloads: dict[str, bytes] = {}
    for backend in BACKENDS:
        target = backend if backend != "default" else "intl_a"
        champ = champions[target]
        row = generic_row if backend == "default" else champ["selected"]
        package = archive_root / row["artifact"]
        member, payload = source_bytes(package, backend)
        digest = hashlib.sha256(payload).hexdigest()
        expected_digest = generic_stats["source_sha256"] if backend == "default" else champ["source_sha256"]
        if expected_digest != digest:
            raise ValueError(f"champion/source resolution mismatch for {backend}: ledger {expected_digest} != {digest}")
        payloads[output_name(backend)] = payload
        selections[backend] = {
            "evaluation_target": ["intl_a", "intl_b"] if backend == "default" else target,
            "record_id": row["record_id"],
            "version": row["version"],
            "submitted_at": row["submitted_at"],
            "speedup": generic_stats["per_target_median"] if backend == "default" else champ["score"],
            "selection_objective_mean": generic_stats["objective_mean"] if backend == "default" else None,
            "artifact": row["artifact"],
            "archive_sha256": row["local_archive_sha256"],
            "archive_member_used": member,
            "source_sha256": digest,
            "same_source_observations": champ["same_source_observations"],
            "same_source_ratio": champ["same_source_ratio"],
            "champion_stability": champ["source_spread"],
        }
    baseline = run_root / "baseline"
    candidate = run_root / "source"
    baseline.mkdir(parents=True)
    candidate.mkdir()
    for name, payload in payloads.items():
        (baseline / name).write_bytes(payload)
        (candidate / name).write_bytes(payload)
    manifest = {
        "run_id": run_id,
        "source_method": "agent-authored starting point copied from archived per-target best-observed submissions; not KernelGen",
        "promotion_policy": "best-observed chip result is used as source baseline even if flagged provisional; retain anomaly and verify with new official measurement",
        "selected_sources": selections,
    }
    (run_root / "baseline-selections.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--ledger", type=Path, default=results.LEDGER)
    parser.add_argument("--archive-root", type=Path, default=results.ROOT)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.run_id, args.ledger, args.archive_root)
    except (ValueError, FileExistsError, OSError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
