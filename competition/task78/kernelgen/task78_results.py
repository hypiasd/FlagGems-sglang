#!/usr/bin/env python3
"""Validate, audit, and render the append-only Task 78 FlagOS score ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results.jsonl"
REPORT = ROOT / "results.md"
TARGETS = (
    "iluvatar", "metax", "enflame", "hygon", "kunlunxin", "ascend",
    "intl_a", "intl_b",
)
DISPLAY = {
    "iluvatar": "天数智芯", "metax": "沐曦", "enflame": "燧原",
    "hygon": "海光", "kunlunxin": "昆仑芯", "ascend": "华为",
    "intl_a": "国际通用 A", "intl_b": "国际通用 B",
}
BACKEND = {target: ("default" if target.startswith("intl_") else target)
           for target in TARGETS}
ANOMALY_RATIO = 1.35


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_ledger(path: Path = LEDGER) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        row["_line"] = line_number
        rows.append(row)
    return rows


def validate_ledger(rows: list[dict], archive_root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    for row in rows:
        rid = row.get("record_id")
        if not isinstance(rid, str) or not rid:
            errors.append(f"line {row.get('_line')}: missing record_id")
        elif rid in ids:
            errors.append(f"duplicate record_id: {rid}")
        ids.add(rid)
        targets = row.get("targets")
        if not isinstance(targets, dict) or set(targets) != set(TARGETS):
            errors.append(f"{rid}: target set must be exactly {', '.join(TARGETS)}")
            continue
        passes = 0
        for target, item in targets.items():
            status = item.get("status")
            score = item.get("speedup")
            if status == "pass" and isinstance(score, (int, float)) and score > 0:
                passes += 1
            elif status == "fail" and score is None:
                continue
            else:
                errors.append(f"{rid}/{target}: pass needs positive score; fail needs null score")
        if row.get("pass_count") != passes:
            errors.append(f"{rid}: pass_count={row.get('pass_count')} but target records contain {passes} passes")
        aggregate = row.get("aggregate_speedup")
        if passes == len(TARGETS):
            if not isinstance(aggregate, (int, float)) or aggregate <= 0:
                errors.append(f"{rid}: all-target pass requires an official aggregate")
            else:
                calculated = sum(targets[target]["speedup"] for target in TARGETS) / len(TARGETS)
                if abs(calculated - aggregate) > 0.02:
                    errors.append(f"{rid}: official aggregate {aggregate} is inconsistent with target mean {calculated:.4f}")
        elif aggregate is not None:
            errors.append(f"{rid}: partial result must not have a task aggregate")
        archive = archive_root / row.get("artifact", "")
        local_digest = row.get("local_archive_sha256")
        if not isinstance(local_digest, str) or len(local_digest) != 64:
            errors.append(f"{rid}: missing 64-character local archive SHA-256")
        if archive.is_file():
            actual_hash = sha256(archive.read_bytes())
            if actual_hash != local_digest:
                errors.append(f"{rid}: local archive hash mismatch ({actual_hash})")
        for target in TARGETS:
            source_digest = targets[target].get("source_sha256")
            if source_digest is not None and (not isinstance(source_digest, str) or len(source_digest) != 64):
                errors.append(f"{rid}/{target}: source_sha256 must be a 64-character hex digest")
            if archive.is_file() and source_digest:
                actual_source, _ = target_source_hash(row, target, archive_root)
                if actual_source != source_digest:
                    errors.append(f"{rid}/{target}: archived source hash mismatch ({actual_source})")
    return errors


def target_source_hash(row: dict, target: str, archive_root: Path = ROOT) -> tuple[str | None, str | None]:
    """Return (sha256, member name), using the submitted generic fallback."""
    archive_path = archive_root / row.get("artifact", "")
    if not archive_path.is_file():
        stored = row.get("targets", {}).get(target, {})
        return stored.get("source_sha256"), stored.get("source_member")
    backend = BACKEND[target]
    specialized = f"concat_and_cast_mha_k_{backend}.py" if backend != "default" else None
    generic = "concat_and_cast_mha_k.py"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            member = specialized if specialized and specialized in names else generic
            if member not in names:
                return None, None
            return sha256(archive.read(member)), member
    except (OSError, zipfile.BadZipFile):
        return None, None


def enrich(rows: list[dict], archive_root: Path = ROOT) -> list[dict]:
    observations: dict[tuple[str, str], list[tuple[dict, float]]] = {}
    for row in rows:
        for target, item in row["targets"].items():
            digest, member = target_source_hash(row, target, archive_root)
            item["source_sha256"] = digest
            item["source_member"] = member
            if digest and item["status"] == "pass":
                observations.setdefault((target, digest), []).append((row, float(item["speedup"])))
    for row in rows:
        for target, item in row["targets"].items():
            digest = item.get("source_sha256")
            if not digest:
                item["source_spread"] = "unknown"
                continue
            group = observations.get((target, digest), [])
            if not group:
                item["same_source_observations"] = 0
                item["same_source_ratio"] = None
                item["source_spread"] = "no_passing_observation"
                continue
            values = [score for _, score in group]
            ratio = max(values) / min(values) if min(values) > 0 else float("inf")
            item["same_source_observations"] = len(group)
            item["same_source_ratio"] = ratio
            item["source_spread"] = "flagged" if len(group) > 1 and ratio >= ANOMALY_RATIO else "within_threshold"
    return rows


def champions(rows: list[dict], archive_root: Path = ROOT) -> dict[str, dict]:
    enriched = enrich(rows, archive_root)
    result = {}
    for target in TARGETS:
        eligible = [row for row in enriched if row["targets"][target]["status"] == "pass"]
        if not eligible:
            continue
        best_score = max(row["targets"][target]["speedup"] for row in eligible)
        best_rows = [row for row in eligible if row["targets"][target]["speedup"] == best_score]
        selected = min(best_rows, key=lambda row: row["submitted_at"])
        data = selected["targets"][target]
        result[target] = {
            "score": best_score,
            "records": best_rows,
            "selected": selected,
            "source_sha256": data.get("source_sha256"),
            "source_member": data.get("source_member"),
            "same_source_observations": data.get("same_source_observations", 0),
            "same_source_ratio": data.get("same_source_ratio"),
            "source_spread": data.get("source_spread", "unknown"),
        }
    return result


def render(rows: list[dict], archive_root: Path = ROOT) -> str:
    rows = enrich(rows, archive_root)
    best_aggregate = max(
        (row for row in rows if row.get("pass_count") == len(TARGETS)
         and isinstance(row.get("aggregate_speedup"), (int, float))),
        key=lambda row: row["aggregate_speedup"],
    )
    best = champions(rows, archive_root)
    out = [
        "# Task 78 第 6 批：官方成绩账本",
        "",
        f"共 {len(rows)} 条官方流水。**全芯片有效总分最佳**：{best_aggregate['version']} "
        f"{best_aggregate['aggregate_speedup']:.2f}×（{best_aggregate['pass_count']}/8）。",
        "单芯片成绩按该芯片单独通过的官方结果统计，即便整包未达 8/8 也保留；"
        "它不构成有效整题平均分。",
        "",
        "## 芯片最佳观察值",
        "",
        "| 芯片 | 最佳加速比 | 来源版本/时间 | 源文件 | 源码 SHA-256 | 同源码观测 | 稳定性提示 |",
        "| --- | ---: | --- | --- | --- | ---: | --- |",
    ]
    for target in TARGETS:
        item = best[target]
        row = item["selected"]
        ratio = item["same_source_ratio"]
        stability = (f"跨度 {ratio:.2f}×，待复测" if item["source_spread"] == "flagged"
                     else "未见阈值级跨度" if ratio is not None else "源码身份未知")
        digest = item["source_sha256"] or "unknown"
        out.append(
            f"| {DISPLAY[target]} | {item['score']:.2f}× | {row['version']} / {row['submitted_at']} "
            f"| `{item['source_member'] or 'unknown'}` | `{digest}` "
            f"| {item['same_source_observations']} | {stability} |"
        )
    out += [
        "",
        f"## 同源码成绩离散（观察阈值：最高/最低 ≥ {ANOMALY_RATIO:.2f}）",
        "",
        "该标记仅提示同一源文件在不同官方提交中的结果不一致；不自动删除分数、"
        "不推断哪次一定错误，也不抹掉历史最佳。下一轮仍保留 best-observed 源码，"
        "但需在决策中把它标为暂定冠军。",
        "",
    ]
    seen = set()
    for row in rows:
        for target, value in row["targets"].items():
            digest = value.get("source_sha256")
            if value["status"] != "pass" or not digest or value.get("source_spread") != "flagged":
                continue
            key = (target, digest)
            if key in seen:
                continue
            seen.add(key)
            group = [r for r in rows if r["targets"][target].get("source_sha256") == digest
                     and r["targets"][target]["status"] == "pass"]
            values = [r["targets"][target]["speedup"] for r in group]
            out.append(f"- {DISPLAY[target]} `{digest[:12]}`："
                       f"{min(values):.2f}×–{max(values):.2f}×，"
                       f"{len(values)} 次观测，版本 {', '.join(r['version'] for r in group)}。")
    out += [
        "",
        "## 官方提交历史",
        "",
        "| 时间 | 版本/文件 | 天数 | 沐曦 | 燧原 | 海光 | 昆仑芯 | 华为 | 国际 A | 国际 B | 通过 | 全局平均 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        vals = []
        for target in TARGETS:
            item = row["targets"][target]
            vals.append(f"{item['speedup']:.2f}×" if item["status"] == "pass" else "Fail")
        aggregate = f"{row['aggregate_speedup']:.2f}×" if row["aggregate_speedup"] is not None else "—"
        out.append(
            f"| {row['submitted_at']} | {row['version']} / `{row['artifact']}` | "
            f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {vals[5]} | "
            f"{vals[6]} | {vals[7]} | {row['pass_count']}/8 | {aggregate} |"
        )
    out += [
        "",
        "## 数据口径",
        "",
        "- JSONL 是追加式原始账本；本页由 `kernelgen/task78_results.py render` 生成。",
        "- ZIP SHA-256 来自本机存档字节，FlagOS 未向页面暴露上传对象摘要；"
        "上传包与本机存档是否逐字节相同，除非另有证据，均不宣称已认证。",
        "- 第 6 批 25 条流水由 FlagOS 团队提交记录回填；v7.zip 出现两次，"
        "服务器端文件摘要未知，故两条记录分开保留。",
        "- 同源码离散是用于异常复核的工作阈值，不是统计学定论；硬件/平台重复评测仍是判断依据。",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "enrich", "render", "summary"))
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--archive-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    rows = load_ledger(args.ledger)
    errors = validate_ledger(rows, args.archive_root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if args.command == "verify":
        missing = [f"{row['record_id']}/{target}" for row in rows for target in TARGETS
                   if not row["targets"][target].get("source_sha256")]
        if missing:
            print("missing immutable source fingerprints: " + ", ".join(missing), file=sys.stderr)
            return 1
        print(f"PASS: {len(rows)} result records; package/source hashes and score invariants checked")
    elif args.command == "enrich":
        rows = enrich(rows, args.archive_root)
        missing = [f"{row['record_id']}/{target}" for row in rows for target in TARGETS
                   if not row["targets"][target].get("source_sha256")]
        if missing:
            print("cannot fingerprint source for: " + ", ".join(missing), file=sys.stderr)
            return 1
        lines = []
        for row in rows:
            row.pop("_line", None)
            lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        args.ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Added source-file fingerprints to {len(rows)} initial history records in {args.ledger}")
    elif args.command == "render":
        args.output.write_text(render(rows, args.archive_root), encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        for target, item in champions(rows, args.archive_root).items():
            row = item["selected"]
            stability = "provisional" if item["source_spread"] == "flagged" else "no flagged spread"
            print(f"{DISPLAY[target]}: {item['score']:.2f}x ({row['version']}, {stability}, {item['source_sha256'] or 'hash unknown'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
