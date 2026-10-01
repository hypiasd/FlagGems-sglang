"""Turn an awarded competition submission into an upstream repository PR.

The platform channel (``competition.adaptation``) and this channel answer
different questions.  The platform wants one ZIP with ``<op>.py`` plus
``<op>_<chip>.py`` files and scores it on its own harness; the upstream
repository wants the same sources placed in the dispatcher tree, with an
Apache header and ``__all__``, passing ``basic-ci.yml``.

Usage::

    python -m competition.pr rules
    python -m competition.pr plan --task task103 --adaptation adapt-1ba6bd8a83ce
    python -m competition.pr bundle --task task103 --adaptation adapt-1ba6bd8a83ce
    python -m competition.pr materialize --task task103 --adaptation adapt-1ba6bd8a83ce
    python -m competition.pr check --task task103 --adaptation adapt-1ba6bd8a83ce
    python -m competition.pr evidence --task task103 --adaptation adapt-1ba6bd8a83ce

``evidence`` reuses the KernelGen reliability-gate methodology: it states, per
target, which evidence state we are entitled to claim, keeps provisional
platform readings out of any aggregate, and refuses to turn "no target
evidence" into a pass.  See ``competition/pr/evidence.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from competition.experiments.store import (
    LOCAL,
    ROOT,
    digest,
    load_json,
    write_json,
)

from . import bundle as bundle_mod
from . import description, evidence, gates, spec

DEFAULT_TOOLS = LOCAL / "venv-pr" / "bin"


def _dir(task: str, adaptation: str) -> Path:
    return bundle_mod.PR_ROOT / task / adaptation


def _records(task: str, package_sha256: str) -> list[dict]:
    """Official observations bound to this exact package digest.

    The platform exposes no record ID, so a record is bound to an archive by
    its SHA-256 and keyed by submission time.
    """
    path = ROOT / "competition" / task / "official-records.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            package_sha256
            and record.get("local_archive_sha256") != package_sha256
        ):
            continue
        rows.append(record)
    return rows


def _evidence_report(task: str, adaptation: str, dest: Path) -> dict:
    """Build the per-target evidence report from the local artifacts.

    Capability is classified rather than assumed: the official platform
    channel is ``callable`` only if its local driver exists, and the rented
    device route is ``configured_unavailable`` because a configured tunnel is
    not a working device.  Reachability is deliberately not probed here.
    """
    profile = load_json(ROOT / "competition" / task / "profile.json")
    source = LOCAL / "adaptations" / adaptation
    release = load_json(source / "release.json")
    archive = source / "package.zip"
    package_sha256 = digest(archive) if archive.is_file() else ""
    gates_path = dest / "gates.json"
    local_gates = load_json(gates_path) if gates_path.is_file() else {}
    routes = [
        {
            "name": "official_platform",
            "config_present": True,
            "in_tool_registry": (LOCAL / "browser" / "submit.mjs").is_file(),
            "reachable": bool(_records(task, package_sha256)),
        },
        {
            "name": "rented_device",
            "config_present": True,
            "in_tool_registry": False,
            "reachable": False,
        },
    ]
    return evidence.derive_report(
        task_id=task,
        adaptation_id=adaptation,
        profile_targets=list(profile["targets"]),
        release=release,
        records=_records(task, package_sha256),
        package_sha256=package_sha256,
        routes=routes,
        local_gates=local_gates,
        reviews=release.get("reviews") or [],
    )


def cmd_rules(args) -> dict:
    return spec.rules()


def cmd_plan(args) -> dict:
    profile = load_json(ROOT / "competition" / args.task / "profile.json")
    plan = {
        "task_id": args.task,
        "op": profile["operator"],
        "targets": profile["targets"],
        "files": [],
    }
    if args.adaptation:
        for path in sorted(
            (LOCAL / "adaptations" / args.adaptation / "source").glob("*.py")
        ):
            stem = path.stem
            suffix = (
                None
                if stem == profile["operator"]
                else stem[len(profile["operator"]) + 1 :]
            )
            tier, rel = spec.tier_path(profile["operator"], suffix)
            plan["files"].append(
                {
                    "name": path.name,
                    "suffix": suffix,
                    "tier": tier,
                    "path": rel,
                }
            )
    plan["not_required"] = list(spec.NOT_REQUIRED)
    return plan


def cmd_bundle(args) -> dict:
    tools = (
        Path(args.tools)
        if args.tools
        else (DEFAULT_TOOLS if DEFAULT_TOOLS.is_dir() else None)
    )
    renames = {}
    for item in args.rename or []:
        if "=" not in item:
            raise ValueError(f"--rename wants OLD=NEW, got {item!r}")
        old, new = item.split("=", 1)
        renames[old] = new
    built = bundle_mod.build(
        args.task,
        args.adaptation,
        op=args.op,
        tools=tools,
        submission_id=args.submission_id,
        note=args.note,
        renames=renames,
    )
    dest = _dir(args.task, args.adaptation)
    (dest / "PR.md").write_text(
        description.render(built, extra_constraints=args.constraint or [])
    )
    built, texts, awarded = bundle_mod.load(dest)
    results = [
        gates.structure_gate(
            built,
            texts,
            load_json(ROOT / "competition" / args.task / "profile.json")[
                "entrypoint"
            ],
        ),
        gates.hygiene_gate(texts),
        gates.preservation_gate(built, awarded, texts),
    ]
    summary = gates.summarize(results)
    write_json(dest / "gates.json", summary)
    return {
        "bundle": str(dest.relative_to(ROOT)),
        "op": built["op"],
        "files": [entry["path"] for entry in built["files"]],
        "formatting": built["formatting"],
        **summary,
    }


def cmd_check(args) -> dict:
    dest = _dir(args.task, args.adaptation)
    built, texts, awarded = bundle_mod.load(dest)
    profile = load_json(ROOT / "competition" / args.task / "profile.json")
    tools = (
        Path(args.tools)
        if args.tools
        else (DEFAULT_TOOLS if DEFAULT_TOOLS.is_dir() else None)
    )
    results = [
        gates.structure_gate(built, texts, profile["entrypoint"]),
        gates.hygiene_gate(texts),
        gates.preservation_gate(built, awarded, texts),
        gates.style_gate(
            [dest / "files" / e["path"] for e in built["files"]], tools
        ),
    ]
    tree = Path(args.tree) if args.tree else dest / "repo"
    python = args.python or str(tools / "python" if tools else "python3")
    if tree.is_dir():
        results.append(gates.repo_checks_gate(tree, python, [built["op"]]))
        results.append(gates.import_smoke_gate(tree, python, built["op"]))
    else:
        results.append(
            gates._gate(
                "repo_checks",
                gates.UNAVAILABLE,
                "run `materialize` first to obtain an upstream tree",
            )
        )
        results.append(
            gates._gate("import_smoke", gates.UNAVAILABLE, "no upstream tree")
        )
    base = gates.summarize(results)
    report = _evidence_report(args.task, args.adaptation, dest)
    report["local_gates"] = {
        "ready_for_pr": base["ready_for_pr"],
        "failed": base["failed"],
        "not_run_here": base["not_run_here"],
    }
    write_json(dest / "evidence.json", report)
    results.append(gates.evidence_gate(report))
    summary = gates.summarize(results)
    write_json(dest / "gates.json", summary)
    return summary


def cmd_evidence(args) -> dict:
    """Write the per-target evidence report and a run-record skeleton."""
    dest = _dir(args.task, args.adaptation)
    report = _evidence_report(args.task, args.adaptation, dest)
    write_json(dest / "evidence.json", report)
    if args.run_record:
        record = load_json(args.run_record)
        validation = evidence.run_record_errors(record)
        write_json(dest / "run-record.check.json", validation)
        report["run_record"] = validation
        write_json(dest / "evidence.json", report)
    else:
        write_json(
            dest / "run-record.template.json",
            evidence.run_record_template(),
        )
    gate = gates.evidence_gate(report)
    return {
        "evidence": f"{dest.relative_to(ROOT)}/evidence.json",
        "gate": gate,
        "package_state": report["package_state"],
        "capability": report["capability"],
        "missing_target_evidence": report["missing_target_evidence"],
        "best_aggregate_speedup": report["official"]["best_aggregate_speedup"],
        "provisional_records": [
            item["record_id"]
            for item in report["official"]["provisional_records"]
        ],
    }


def cmd_materialize(args) -> dict:
    dest = _dir(args.task, args.adaptation)
    tree = bundle_mod.materialize(dest, ref=args.ref)
    return {
        "tree": str(tree.relative_to(ROOT)),
        "next": f"python -m competition.pr check --task {args.task} --adaptation {args.adaptation}",
    }


def cmd_description(args) -> dict:
    dest = _dir(args.task, args.adaptation)
    built, _, _ = bundle_mod.load(dest)
    (dest / "PR.md").write_text(
        description.render(built, extra_constraints=args.constraint or [])
    )
    return {"path": str((dest / "PR.md").relative_to(ROOT))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("rules")
    p = sub.add_parser("plan")
    p.add_argument("--task", required=True)
    p.add_argument("--adaptation")

    p = sub.add_parser("bundle")
    p.add_argument("--task", required=True)
    p.add_argument("--adaptation", required=True)
    p.add_argument("--op")
    p.add_argument("--tools")
    p.add_argument("--submission-id")
    p.add_argument("--note")
    p.add_argument("--constraint", action="append")
    p.add_argument(
        "--rename",
        action="append",
        metavar="OLD=NEW",
        help="token-level identifier rename recorded in the bundle (for flake8 E741 and friends)",
    )

    p = sub.add_parser("check")
    p.add_argument("--task", required=True)
    p.add_argument("--adaptation", required=True)
    p.add_argument("--tools")
    p.add_argument("--tree")
    p.add_argument("--python")

    p = sub.add_parser("materialize")
    p.add_argument("--task", required=True)
    p.add_argument("--adaptation", required=True)
    p.add_argument("--ref", default="upstream/master")

    p = sub.add_parser("description")
    p.add_argument("--task", required=True)
    p.add_argument("--adaptation", required=True)
    p.add_argument("--constraint", action="append")

    p = sub.add_parser("evidence")
    p.add_argument("--task", required=True)
    p.add_argument("--adaptation", required=True)
    p.add_argument(
        "--run-record",
        help="validate an existing run record against the field discipline",
    )

    args = parser.parse_args(argv)
    handler = {
        "rules": cmd_rules,
        "plan": cmd_plan,
        "bundle": cmd_bundle,
        "check": cmd_check,
        "materialize": cmd_materialize,
        "description": cmd_description,
        "evidence": cmd_evidence,
    }[args.command]
    try:
        result = handler(args)
    except (ValueError, OSError, KeyError) as exc:
        print(
            json.dumps(
                {"passed": False, "error": str(exc)}, ensure_ascii=False
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "check" and result.get("failed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
