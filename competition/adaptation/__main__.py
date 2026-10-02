"""Prepare competition adaptations; upload is handled explicitly in Chrome."""

from __future__ import annotations
import argparse
import gzip
import importlib
import json
import math
import shutil
import subprocess
import sys
import uuid
import urllib.request
from pathlib import Path
from competition.experiments import store
from competition import decide
from competition import members
from . import constexprs, failures, safety
from .compiler_scan import inspect_source
from .ledger import record, summary, target_ledger
from .packaging import make_package, validate_package
from .structure import normalized_ast


def prepare(run_id, baseline=None):
    root = store.run_path(run_id)
    experiment = store.verify(root)
    contract = store.load_json(root / "contract.json")
    layout = members.audit(
        contract["operator"],
        root / "source",
        contract["targets"],
        contract["package_members"],
    )
    if not layout["passed"]:
        raise ValueError(
            "the frozen run has an inconsistent package layout: "
            + "; ".join(layout["errors"])
        )
    identifier = "adapt-" + uuid.uuid4().hex[:12]
    dest = store.LOCAL / "adaptations" / identifier
    shutil.copytree(root / "source", dest / "source")
    shutil.copy2(root / "contract.json", dest / "contract.json")
    if baseline:
        (dest / "baseline").mkdir()
        for name in store.load_json(dest / "contract.json")["package_members"]:
            path = baseline / name
            if path.is_symlink():
                raise ValueError("baseline sources must be regular files")
            if path.is_file():
                shutil.copy2(path, dest / "baseline" / name)
    manifest = {
        "adaptation_id": identifier,
        "task_id": experiment["task_id"],
        "experiment_id": run_id,
        "experiment_snapshot_sha256": experiment["snapshot_sha256"],
        "status": "draft",
        "created_at": store.now(),
    }
    store.write_json(dest / "adaptation.json", manifest)
    package = make_package(
        dest / "source",
        dest / "package.zip",
        store.load_json(dest / "contract.json"),
    )
    return {
        **manifest,
        "package": package,
        "next": "edit the independent source if needed; provide release.json and run check before any upload",
    }


def publication_check(dest):
    contract = store.load_json(dest / "contract.json")
    manifest_path = dest / "release.json"
    errors, compiler, structure = [], {}, {}
    if not manifest_path.exists():
        return {
            "passed": False,
            "errors": [
                "release.json is missing: per-target hypotheses, forecasts, reviews and full correctness evidence are required"
            ],
        }
    manifest = store.load_json(manifest_path)
    if set(manifest.get("targets", {})) != set(contract["targets"]):
        errors.append(
            "release hypotheses must cover exactly this task's target matrix"
        )
    for target in contract["targets"]:
        item = manifest.get("targets", {}).get(target, {})
        name = item.get("source", "")
        if name not in contract["package_members"]:
            errors.append(f"{target}: invalid source member")
            continue
        source, baseline = dest / "source" / name, dest / "baseline" / name
        required = (
            "structural_change",
            "expected_mechanism",
            "bottleneck_evidence",
            "falsifier",
        )
        if any(not item.get(key) for key in required):
            errors.append(f"{target}: structural hypothesis is incomplete")
        if not source.exists() or not baseline.exists():
            errors.append(f"{target}: source/baseline snapshot missing")
            continue
        if item.get("source_sha256") != store.digest(source) or item.get(
            "baseline_sha256"
        ) != store.digest(baseline):
            errors.append(
                f"{target}: hypothesis hashes differ from source/baseline"
            )
        changed = normalized_ast(source)[0] != normalized_ast(baseline)[0]
        structure[target] = changed
        if not changed:
            errors.append(
                f"{target}: constant/comment-only or unchanged structure"
            )
        scan = inspect_source(
            source, target if not target.startswith("intl_") else "default"
        )
        compiler[target] = scan["findings"]
        if scan["blockers"]:
            errors.append(f"{target}: unresolved compiler blockers")
    carry_forward = decide.enforce(
        contract["task_id"], store.LOCAL, manifest, dest / "source"
    )
    errors.extend(carry_forward["errors"])
    reviews = manifest.get("reviews", [])
    review_ids = {
        item.get("invocation_id")
        for item in reviews
        if item.get("invocation_id")
    }
    if len(review_ids) < contract["review_policy"].get(
        "independent_reviews", 2
    ) or any(item.get("novel_findings") for item in reviews):
        errors.append(
            "independent review evidence is missing or contains unresolved findings"
        )
    full = manifest.get("full_correctness", {})
    if (
        full.get("status") != "passed"
        or not full.get("evidence")
        or full.get("source_hashes")
        != {p.name: store.digest(p) for p in (dest / "source").glob("*.py")}
    ):
        errors.append(
            "full correctness evidence must bind every adapted source"
        )
    policy = contract.get("publication_policy")
    if not policy:
        errors.append("this task has no verified publication policy yet")
    elif contract["task_id"] == "task78":
        from competition.task78 import release, official_history

        hurdle = release.evaluate(
            manifest, official_history.load_ledger(), official_history.ROOT
        )
        errors.extend(hurdle["errors"])
    else:
        goal = manifest.get("frozen_goal")
        projected = manifest.get("projected_aggregate")
        if (
            not manifest.get("forecast_evidence")
            or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                and v > 0
                for v in (goal, projected)
            )
            or projected <= goal
        ):
            errors.append(
                "task-specific frozen goal and evidenced forecast improvement are required"
            )
    return {
        "passed": not errors,
        "errors": errors,
        "compiler": compiler,
        "structural_delta": structure,
        "carry_forward": carry_forward,
        "limitation": "forecasts and local validation are not official target results",
    }


def inspect(task, refresh=False):
    contract = store.profile(task)
    result = {"task_id": task, "contract": contract}
    if refresh:
        url = (
            "https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/"
            + contract["operator"]
        )
        with urllib.request.urlopen(url, timeout=25) as response:
            data = response.read()
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        official = json.loads(data)["data"]
        evidence = (
            store.LOCAL
            / "contracts"
            / task
            / (store.now().replace(":", "-") + ".json")
        )
        store.write_json(
            evidence,
            {"source": url, "captured_at": store.now(), "data": official},
        )
        mapping = {
            "tianshu": "iluvatar",
            "muxi": "metax",
            "haiguang": "hygon",
            "huawei": "ascend",
            "card_a": "intl_a",
            "card_b": "intl_b",
        }
        targets = [mapping.get(t, t) for t in official["supported_gpus"]]
        result.update(
            official_targets=targets,
            contract_drift=set(targets) != set(contract["targets"]),
            evidence=evidence.relative_to(store.ROOT).as_posix(),
        )
    return result


def members_command(task, source=None, adaptation=None):
    """Show the package layout a task would ship, and any contract drift.

    The answer to "may this chip have its own file" is yes, and this command
    says which chips already have one, which share the generic module, and
    whether a recent package carried files the current contract no longer
    declares (the silent-drop case).
    """
    contract = store.profile(task)
    result = {
        "task_id": task,
        "drift": members.drift(task, contract, store.LOCAL),
    }
    if adaptation:
        dest = store.LOCAL / "adaptations" / adaptation
        contract = store.load_json(dest / "contract.json")
        source = dest / "source"
    if source:
        result["layout"] = members.audit(
            contract["operator"],
            source,
            contract["targets"],
            contract["package_members"],
        )
    else:
        result["layout"] = {
            "declared": contract.get("package_members"),
            "targets_without_dedicated": "provide --source to inspect a directory",
        }
    return result


def decide_command(task, next_package=False):
    """Report how the official results should change the next package."""
    result = decide.diagnose(task, store.LOCAL)
    if next_package:
        return {
            "task_id": task,
            "rework_targets": result["rework_targets"],
            "rework_aggregate_gain_upper_bound": result[
                "rework_aggregate_gain_upper_bound"
            ],
            "next_package": result["next_package"],
            "reuse_rule": result["reuse_rule"],
            "limitation": result["limitation"],
        }
    return result


def gate(run_id):
    """Everything that must hold *before* a package is uploaded.

    The submission platform is a slow, quota-limited oracle, so every rule it
    has ever enforced goes here instead of in the operator's memory: the member
    layout, the platform-safety rules, the CPU semantic model for **every**
    declared member (not just the generic one) and the failure ledger, which
    blocks re-uploading bytes that already failed.
    """
    prepared = prepare(run_id)
    dest = store.LOCAL / "adaptations" / prepared["adaptation_id"]
    contract = store.load_json(dest / "contract.json")
    source = dest / "source"
    task = contract["task_id"]
    declared = list(contract["package_members"])
    errors, members_report = [], []

    layout = members.audit(
        contract["operator"], source, contract["targets"], declared
    )
    errors.extend(layout["errors"])

    checker = store.ROOT / "competition" / task / "validate_cpu.py"
    for name in declared:
        errors.extend(safety.check_source(source / name))
        errors.extend(constexprs.check_source(source / name))
        entry = {
            "member": name,
            "role": "generic" if name == layout["generic"] else "dedicated",
            "sha256": store.digest(source / name),
        }
        if checker.is_file():
            proc = subprocess.run(
                [
                    sys.executable,
                    str(checker),
                    "--source",
                    str(source / name),
                    "--allow-rewrite",
                    "--allow-auxiliary",
                ],
                cwd=str(store.ROOT),
                capture_output=True,
                text=True,
            )
            entry["semantic"] = "pass" if proc.returncode == 0 else "FAIL"
            if proc.returncode != 0:
                errors.append(
                    f"{name}: CPU semantic validation failed: "
                    f"{(proc.stdout or proc.stderr).strip().splitlines()[-1:] }"
                )
        else:
            entry["semantic"] = f"skipped: {task} ships no validate_cpu.py"
        members_report.append(entry)

    hashes = {e["member"]: e["sha256"] for e in members_report}
    errors.extend(
        failures.exact_repeat(task, layout["target_sources"], hashes)
    )

    adaptation = store.load_json(dest / "adaptation.json")
    result = {
        "adaptation_id": prepared["adaptation_id"],
        "task_id": task,
        "package": prepared["package"],
        "members": members_report,
        "target_sources": layout["target_sources"],
        "warnings": layout["warnings"],
        "blocking": errors,
        "passed": not errors and prepared["package"]["passed"],
        "submit_package": str(dest / "package.zip"),
    }
    adaptation.update(
        status="ready" if result["passed"] else "blocked",
        package_sha256=prepared["package"]["archive_sha256"],
    )
    store.write_json(dest / "adaptation.json", adaptation)
    store.write_json(dest / "gate.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--task", required=True)
    p.add_argument("--refresh", action="store_true")
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", type=Path)
    p = sub.add_parser("check")
    p.add_argument("--adaptation", required=True)
    p.add_argument("--package-only", action="store_true")
    p = sub.add_parser("gate")
    p.add_argument("--run", required=True)
    p = sub.add_parser("record")
    p.add_argument("--task", required=True)
    p.add_argument("--input", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--task", required=True)
    p.add_argument("--targets", action="store_true")
    p = sub.add_parser("decide")
    p.add_argument("--task", required=True)
    p.add_argument("--next-package", action="store_true")
    p = sub.add_parser("members")
    p.add_argument("--task", required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--adaptation")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect(args.task, args.refresh)
        elif args.command == "prepare":
            result = prepare(args.run, args.baseline)
        elif args.command == "gate":
            result = gate(args.run)
        elif args.command == "record":
            result = record(args.task, store.load_json(args.input))
        elif args.command == "report":
            result = (
                target_ledger(args.task)
                if args.targets
                else summary(args.task)
            )
        elif args.command == "members":
            result = members_command(args.task, args.source, args.adaptation)
        elif args.command == "decide":
            result = decide_command(args.task, args.next_package)
        else:
            if (
                not args.adaptation.startswith("adapt-")
                or not args.adaptation[6:].isalnum()
            ):
                raise ValueError("invalid adaptation ID")
            dest = store.LOCAL / "adaptations" / args.adaptation
            contract = store.load_json(dest / "contract.json")
            package = make_package(
                dest / "source", dest / "package.zip", contract
            )
            publication = (
                None if args.package_only else publication_check(dest)
            )
            result = {
                "package": package,
                "publication": publication,
                "passed": package["passed"]
                and (args.package_only or publication["passed"]),
            }
            if args.package_only and (dest / "release.json").is_file():
                preview = decide.enforce(
                    contract["task_id"],
                    store.LOCAL,
                    store.load_json(dest / "release.json"),
                    dest / "source",
                )
                result["carry_forward_preview"] = {
                    "blocking": False,
                    "preview": preview,
                }
            manifest = store.load_json(dest / "adaptation.json")
            manifest.update(
                status=(
                    "package_checked"
                    if args.package_only and result["passed"]
                    else "ready" if result["passed"] else "blocked"
                ),
                package_sha256=package["archive_sha256"],
                source_hashes={
                    p.name: store.digest(p)
                    for p in (dest / "source").glob("*.py")
                },
            )
            store.write_json(dest / "adaptation.json", manifest)
            store.write_json(dest / "check.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("passed", True) else 1
    except (ValueError, OSError, KeyError) as exc:
        print(
            json.dumps(
                {"passed": False, "error": str(exc)}, ensure_ascii=False
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
