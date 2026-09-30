from pathlib import Path
from competition.task78 import official_history as results

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
