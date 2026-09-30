"""CLI for frozen, individually reviewed GPU experiments."""
from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from pathlib import Path

from . import store
from .timing import MODES
from .transport import execute


def comparison(reports):
    if not reports or any(r.get("command") != "bench" or r.get("status") != "passed" for r in reports):
        raise ValueError("comparison requires complete bench reports")
    keys = ("name", "device_id", "capability", "torch", "triton", "cuda", "driver")
    signature = lambda r: ([r["device"].get(key) for key in keys], r.get("mode"), r.get("selected_cases"), r.get("seed"), r.get("budget_seconds"), r.get("protocol_sha256"))
    if any(signature(r) != signature(reports[0]) for r in reports[1:]):
        raise ValueError("device, software, seed, cases or timing policy differ; results remain separate")
    base = reports[0]
    index = {(r["case"], r["source"]): r for r in base["rows"]}
    return [{"run_id": report["run_id"], "cases": [
        {"case": row["case"], "source": row["source"], "historical_baseline_ratio": index[(row["case"], row["source"])]["measurement"]["candidate"]["median_ms"] / row["measurement"]["candidate"]["median_ms"],
         "note": "comparison across separate runs; inspect each run's same-process reference and spread"}
        for row in report["rows"]]} for report in reports[1:]]


def compact(result):
    if "hashes" in result:
        return {key: result[key] for key in ("run_id", "task_id", "hypothesis", "parent", "snapshot_sha256")}
    if "experiment" in result:
        return {"experiment": compact(result["experiment"]), "attempts": [compact(r) for r in result["attempts"]]}
    if "rows" not in result:
        return result
    summary = {key: value for key, value in result.items() if key not in {"rows", "selected_cases"}}
    summary["execution_count"] = len(result["rows"])
    if result.get("command") == "bench":
        summary["cases"] = [{"case": row["case"], "source": row["source"], "correct": row["correct"],
                             "measurement": row.get("measurement"), "error": row.get("error")} for row in result["rows"]]
    else:
        summary["failures"] = [row for row in result["rows"] if not row["correct"]][:5]
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--device", default="t4")
    new = sub.add_parser("new")
    new.add_argument("--task", required=True)
    new.add_argument("--source", "--source-dir", type=Path, required=True)
    new.add_argument("--hypothesis", required=True)
    new.add_argument("--parent")
    new.add_argument("--seed", type=int, default=0)
    new.add_argument("--requirements", type=Path, help="explicit implementation hardware requirements JSON; does not change dtype")
    for command in ("test", "bench", "profile"):
        p = sub.add_parser(command)
        p.add_argument("--run", required=True)
        p.add_argument("--device", default="t4")
        p.add_argument("--cases", nargs="+")
        if command == "test":
            p.add_argument("--all-sources", action="store_true")
        if command == "bench":
            p.add_argument("--mode", choices=MODES, default="quick")
            p.add_argument("--budget-seconds", type=float)
        if command == "profile":
            p.add_argument("--tool", choices=("ncu", "nsys"), required=True)
            p.add_argument("--ncu-mode", choices=("basic", "launch"), default="basic", help="launch mode only requests static launch/resource metrics")
    compare = sub.add_parser("compare")
    compare.add_argument("--runs", nargs="+", required=True)
    report = sub.add_parser("report")
    report.add_argument("--run", required=True)
    for command_parser in sub.choices.values():
        command_parser.add_argument("--json", action="store_true", help="print the full machine-readable report")
    args = parser.parse_args(argv)
    try:
        if args.command == "new":
            result = store.create(args.task, args.source, args.hypothesis, args.parent, args.seed, store.load_json(args.requirements) if args.requirements else None)
        elif args.command == "doctor":
            result = execute(args.device, {"command": "doctor"}, store.LOCAL / "doctor" / args.device, 45)
        elif args.command in {"report", "compare"}:
            if args.command == "report":
                root = store.run_path(args.run)
                result = {"experiment": store.verify(root), "attempts": store.attempts(root)}
            else:
                latest = []
                for run_id in args.runs:
                    store.verify(store.run_path(run_id))
                    benches = [r for r in store.attempts(store.run_path(run_id)) if r.get("command") == "bench"]
                    if not benches:
                        raise ValueError("run has no bench attempt")
                    latest.append(benches[-1])
                result = {"comparison": comparison(latest), "evidence_class": "gpu-experiment-nontarget"}
        else:
            root = store.run_path(args.run)
            manifest = store.verify(root)
            job = {"command": args.command, "run_id": args.run, "cases": args.cases}
            timeout = 600
            if args.command == "bench":
                seconds = MODES[args.mode]["budget"] if args.budget_seconds is None else args.budget_seconds
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("budget must be positive and finite")
                job.update(mode=args.mode, budget_seconds=seconds)
                timeout = MODES[args.mode]["timeout"]
            elif args.command == "test":
                job["all_sources"] = args.all_sources
            else:
                if not args.cases or len(args.cases) != 1:
                    raise ValueError("profile requires exactly one explicit --cases ID")
                job["tool"] = args.tool
                job["ncu_mode"] = args.ncu_mode
                timeout = 210
            attempt = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
            output = root / "attempts" / attempt
            store.event(root, "started", {"command": args.command, "attempt": attempt, "device": args.device})
            try:
                result = execute(args.device, job, output, timeout, root)
            except Exception:
                store.event(root, "interrupted", {"command": args.command, "attempt": attempt})
                raise
            result["snapshot_sha256"] = manifest["snapshot_sha256"]
            store.write_json(output / "report.json", result)
            store.event(root, "finished", {"command": args.command, "attempt": attempt, "status": result["status"]})
        print(json.dumps(result if args.json else compact(result), ensure_ascii=False, indent=2))
        return 0 if result.get("status", "passed") in {"passed", "ready"} else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
