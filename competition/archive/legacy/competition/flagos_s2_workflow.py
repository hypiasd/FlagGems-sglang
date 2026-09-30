#!/usr/bin/env python3
"""Resumable, task-neutral checkpoint manager for FlagOS S2 kernel runs.

This tool records durable state and rejects unsafe lifecycle transitions. It
does not pretend that local reports prove target correctness or perform browser
uploads itself; the active Codex workflow supplies those independently
verifiable receipts through Chrome and the registered KernelGen operation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
COMPETITION = ROOT / "competition"
PROFILE_DIR = COMPETITION / "workflow" / "tasks"
STATE_ROOT = COMPETITION / ".autopilot"
RUN_ROOT = STATE_ROOT / "runs"
SESSION_AUTH = STATE_ROOT / "session-authorization.json"
TASK_INVENTORY = STATE_ROOT / "task-inventory.json"
TASK_CONTRACT_EVIDENCE = STATE_ROOT / "task-contract-evidence.json"
KERNELGEN_RUNTIME_CHECK = STATE_ROOT / "kernelgen-runtime-check.json"
STATES = {
    "planned", "tool_unavailable", "prepared", "generated",
    "locally_validated", "reviewed", "target_validated", "measured",
    "package_ready", "upload_armed", "submitted", "record_confirmed",
    "upload_aborted", "evaluating", "completed", "promoted", "rejected", "inconclusive",
}
QUEUE_PRIORITY_ORDER = {
    "resume_before_new_upload_or_iteration": 0,
    "record_terminal_result": 1,
    "resume_or_repair": 2,
    "resume_after_kernelgen_is_live": 2,
    "start_first_iteration": 3,
    "check_kernelgen_registry": 3,
    "onboard_task_contract": 4,
    "resolve_inconclusive_evidence": 5,
    "wait_for_kernelgen": 6,
    "wait_for_batch_to_open": 7,
    "refresh_open_task_status_in_chrome": 8,
    "goal_reached": 9,
}
REQUIRED_KERNELGEN_OPERATIONS = {
    "generate_kernel", "optimize_kernel", "specialize_kernel",
}
TRANSITIONS = {
    "planned": {"tool_unavailable", "prepared", "rejected", "inconclusive"},
    "tool_unavailable": {"tool_unavailable", "prepared", "rejected", "inconclusive"},
    "prepared": {"prepared", "generated", "rejected", "inconclusive"},
    "generated": {"locally_validated", "rejected", "inconclusive"},
    "locally_validated": {"reviewed", "rejected", "inconclusive"},
    "reviewed": {"target_validated", "package_ready", "inconclusive", "rejected"},
    "target_validated": {"measured", "rejected", "inconclusive"},
    "measured": {"package_ready", "promoted", "rejected", "inconclusive"},
    "package_ready": {"upload_armed", "rejected", "inconclusive"},
    "upload_armed": {"submitted", "record_confirmed", "upload_aborted", "inconclusive"},
    "upload_aborted": set(),
    "submitted": {"record_confirmed", "evaluating", "inconclusive"},
    "record_confirmed": {"evaluating", "completed", "inconclusive"},
    "evaluating": {"completed", "inconclusive"},
    "completed": {"target_validated", "rejected", "inconclusive"},
    "promoted": set(),
    "rejected": set(),
    "inconclusive": {"prepared", "rejected"},
}


class WorkflowError(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read JSON {path}: {exc}") from exc


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def profile_path(task_id: str) -> Path:
    if not re.fullmatch(r"task[0-9]+", task_id):
        raise WorkflowError("task id must look like task60")
    path = PROFILE_DIR / f"{task_id}.json"
    if not path.is_file():
        raise WorkflowError(
            f"no adapter for {task_id}; copy {PROFILE_DIR / 'template.json'} and fill the official contract"
        )
    return path


def validate_profile(profile: dict) -> list[str]:
    required = (
        "schema_version", "competition", "task_id", "task_name", "operator",
        "public_entrypoint", "source_files", "targets", "baseline", "local_gate",
        "review_policy", "score_policy", "package", "submission", "result_ledger",
    )
    missing = [key for key in required if key not in profile]
    errors = [f"missing profile field: {key}" for key in missing]
    if missing:
        return errors
    if profile["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if profile["competition"] != "flagos-s2":
        errors.append("competition must be flagos-s2")
    if not re.fullmatch(r"task[0-9]+", str(profile["task_id"])):
        errors.append("task_id must look like task60")
    if not isinstance(profile["operator"], str) or not profile["operator"] or not isinstance(profile["public_entrypoint"], str) or not profile["public_entrypoint"]:
        errors.append("operator and public_entrypoint must be nonempty")
    baseline = profile.get("baseline", {})
    baseline_mode = baseline.get("mode", "optimize_existing_sources") if isinstance(baseline, dict) else None
    if baseline_mode not in {"optimize_existing_sources", "generate_from_official_reference"}:
        errors.append("baseline.mode must be optimize_existing_sources or generate_from_official_reference")
    sources = profile["source_files"]
    if not isinstance(sources, list) or not sources or any(not isinstance(raw, str) for raw in sources):
        errors.append("source_files must be a nonempty list")
    else:
        if len(sources) != len(set(sources)):
            errors.append("source_files must not contain duplicates")
        for raw in sources:
            path = Path(raw)
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"source path must stay within the repo: {raw}")
            elif baseline_mode == "optimize_existing_sources" and not (ROOT / path).is_file():
                errors.append(f"source file does not exist: {raw}")
    targets = profile["targets"]
    if (not isinstance(targets, list) or not targets
            or any(not isinstance(target, str) or not target.strip() for target in targets)
            or (all(isinstance(target, str) for target in targets) and len(targets) != len(set(targets)))):
        errors.append("targets must be a nonempty list of unique target ids")
    for field in ("baseline", "local_gate", "review_policy", "score_policy", "package", "submission", "result_ledger"):
        if not isinstance(profile[field], dict) or not profile[field]:
            errors.append(f"{field} must be a nonempty object")
    baseline = profile.get("baseline", {})
    if isinstance(baseline, dict) and (not baseline.get("selection") or not baseline.get("ledger")
                                   or not baseline.get("candidate_snapshot")):
        errors.append("baseline must define selection, ledger, and immutable candidate snapshot")
    if isinstance(baseline, dict) and baseline_mode == "generate_from_official_reference":
        reference = baseline.get("reference_source")
        if not isinstance(reference, dict) or not reference.get("path") or not reference.get("sha256"):
            errors.append("generate_from_official_reference requires reference_source.path and reference_source.sha256")
        else:
            reference_path = Path(str(reference["path"]))
            if reference_path.is_absolute() or ".." in reference_path.parts:
                errors.append("baseline.reference_source.path must stay within the repo")
            elif not (ROOT / reference_path).is_file():
                errors.append(f"official reference source does not exist: {reference_path}")
            elif not re.fullmatch(r"[0-9a-f]{64}", str(reference["sha256"])):
                errors.append("baseline.reference_source.sha256 must be an exact lowercase SHA-256")
            elif sha256_bytes((ROOT / reference_path).read_bytes()) != reference["sha256"]:
                errors.append("baseline.reference_source.sha256 does not match the frozen official reference bytes")
    local_gate = profile.get("local_gate", {})
    if isinstance(local_gate, dict) and any(not local_gate.get(key) for key in ("command", "evidence_class", "contract")):
        errors.append("local_gate must define command, evidence_class, and semantic contract")
    review = profile.get("review_policy", {})
    if isinstance(review, dict) and (review.get("independent_reviews") != 2
                                     or review.get("distinct_invocation_ids") is not True):
        errors.append("review_policy must require two distinct independent review invocations")
    score = profile.get("score_policy", {})
    if isinstance(score, dict) and (not score.get("goal_resolution") or not score.get("release_floor")
                                    or not score.get("anomaly_rule")):
        errors.append("score_policy must define a frozen goal, release hurdle, and anomaly rule")
    submission = profile.get("submission", {})
    if isinstance(submission, dict) and submission.get("browser") != "chrome":
        errors.append("submission.browser must be chrome")
    if isinstance(submission, dict) and submission.get("batch_resolution") != "chrome-before-each-upload":
        errors.append("batch_resolution must read the selected batch from Chrome before every upload")
    if isinstance(score, dict) and score.get("aggregate_source") != "official-platform":
        errors.append("aggregate_source must be official-platform")
    package = profile.get("package", {})
    members = package.get("root_members") if isinstance(package, dict) else None
    if not isinstance(members, list) or not members or any(not isinstance(member, str) or not member for member in members):
        errors.append("package.root_members must list exact nonempty ZIP member names")
    elif len(members) != len(set(members)):
        errors.append("package.root_members must not contain duplicates")
    elif any(Path(member).name != member or not member.endswith(".py") for member in members):
        errors.append("package.root_members must be root-level UTF-8 .py filenames")
    elif (isinstance(sources, list) and sources and all(isinstance(raw, str) for raw in sources)
          and {Path(raw).name for raw in sources} != set(members)):
        errors.append("source_files basenames must match package.root_members exactly")
    target_source_map = profile.get("target_source_map")
    ledger_for_mapping = profile.get("result_ledger")
    ledger_format = ledger_for_mapping.get("format") if isinstance(ledger_for_mapping, dict) else None
    valid_targets = (
        isinstance(targets, list)
        and all(isinstance(target, str) and target.strip() for target in targets)
        and len(targets) == len(set(targets))
    )
    if (ledger_format == "generic-v1"
            and isinstance(members, list) and len(members) > 1):
        if (not isinstance(target_source_map, dict)
                or not valid_targets
                or set(target_source_map) != set(targets)
                or any(not isinstance(member, str) or member not in members
                       for member in target_source_map.values())):
            errors.append("multi-file generic-v1 adapters must map every target to an exact package root member")
    ledger = profile.get("result_ledger", {})
    if isinstance(ledger, dict) and (not isinstance(ledger.get("format"), str)
                                     or ledger.get("format") not in {"generic-v1", "task78-v1"}):
        errors.append("result_ledger.format must be generic-v1 or task78-v1")
    if isinstance(ledger, dict):
        for field in ("path", "report", "validator"):
            if field in ledger:
                ledger_path = Path(str(ledger[field]))
                if ledger_path.is_absolute() or ".." in ledger_path.parts:
                    errors.append(f"result_ledger.{field} must stay within the repo")
    if isinstance(ledger, dict) and ledger.get("format") == "task78-v1" and not ledger.get("validator"):
        errors.append("task78-v1 result ledger requires its validator adapter")
    def find_placeholders(value, path="profile"):
        found = []
        if isinstance(value, dict):
            for key, item in value.items():
                found.extend(find_placeholders(item, f"{path}.{key}"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                found.extend(find_placeholders(item, f"{path}[{index}]"))
        elif isinstance(value, str) and re.search(r"\b(?:FILL|TODO|TBD|PLACEHOLDER)\b", value, re.IGNORECASE):
            found.append(path)
        return found
    errors.extend(f"unresolved template placeholder at {path}" for path in find_placeholders(profile))
    return errors


def get_profile(task_id: str) -> tuple[dict, str]:
    path = profile_path(task_id)
    raw = path.read_bytes()
    profile = json.loads(raw.decode("utf-8"))
    errors = validate_profile(profile)
    if profile.get("task_id") != task_id:
        errors.append(f"profile task_id does not match requested task {task_id}")
    if errors:
        raise WorkflowError("invalid task profile:\n- " + "\n- ".join(errors))
    return profile, sha256_bytes(raw)


def get_run_profile(directory: Path) -> dict:
    manifest = read_json(directory / "run.json")
    snapshot = directory / manifest.get("profile_snapshot", "task-profile.json")
    raw = snapshot.read_bytes()
    if sha256_bytes(raw) != manifest.get("profile_sha256"):
        raise WorkflowError("run's frozen task profile was modified")
    profile = json.loads(raw.decode("utf-8"))
    errors = validate_profile(profile)
    if profile.get("task_id") != directory.parent.name:
        errors.append(f"frozen profile task_id does not match run directory {directory.parent.name}")
    if errors:
        raise WorkflowError("invalid frozen task profile:\n- " + "\n- ".join(errors))
    return profile


def run_dir(task_id: str, run_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", run_id):
        raise WorkflowError("run id must be 3-64 lowercase letters/digits/dot/underscore/hyphen")
    return RUN_ROOT / task_id / run_id


def load_events(directory: Path) -> list[dict]:
    path = directory / "events.jsonl"
    if not path.is_file():
        return []
    events = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"invalid event JSON at {path}:{line_number}: {exc}") from exc
        events.append(event)
    return events


def current_state(directory: Path) -> dict:
    events = load_events(directory)
    if not events:
        raise WorkflowError(f"run has no events: {directory}")
    state = "<new>"
    for index, event in enumerate(events):
        if index == 0:
            if event.get("from") != "<new>" or event.get("to") != "planned":
                raise WorkflowError("first event must transition from <new> to planned")
            state = "planned"
            continue
        if event.get("from") != state:
            raise WorkflowError(f"broken event chain at {event.get('at')}: expected from={state}")
        if event.get("to") not in STATES or event.get("to") not in TRANSITIONS.get(state, set()):
            raise WorkflowError(f"invalid event transition {state} -> {event.get('to')}")
        state = event["to"]
    return {"state": state, "events": events}


def append_event(directory: Path, from_state: str, to_state: str, evidence: dict) -> None:
    events = load_events(directory)
    if not events and from_state != "<new>":
        raise WorkflowError("first event must use from=<new>")
    if events:
        actual = current_state(directory)["state"]
        if actual != from_state:
            raise WorkflowError(f"stale transition: current state is {actual}, not {from_state}")
        if to_state not in TRANSITIONS.get(from_state, set()):
            raise WorkflowError(f"transition not allowed: {from_state} -> {to_state}")
    elif to_state != "planned":
        raise WorkflowError("a new run must begin in planned state")
    event = {"from": from_state, "to": to_state, "at": utc_now(), "evidence": evidence}
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json(directory / "state.json", {
        "state": to_state,
        "updated_at": event["at"],
        "event_count": len(events) + 1,
        "evidence": evidence,
    })


def inspect_global_submissions() -> list[tuple[str, dict]]:
    result = []
    if not RUN_ROOT.exists():
        return result
    for events_path in RUN_ROOT.glob("*/*/events.jsonl"):
        directory = events_path.parent
        task_id = directory.parent.name
        events = load_events(directory)
        latest_state = current_state(directory)["state"] if events else ""
        for event in events:
            evidence = event.get("evidence", {})
            completed_attempt = event.get("to") in {"submitted", "record_confirmed", "evaluating", "completed", "promoted"}
            unresolved_arm = event.get("to") == "upload_armed" and latest_state != "upload_aborted"
            if evidence.get("upload_attempted") or completed_attempt or unresolved_arm:
                result.append((task_id, event))
    return result


def ledger_has_package(profile: dict, package_hash: str) -> bool:
    ledger_spec = profile["result_ledger"]
    ledger_path = ROOT / ledger_spec["path"]
    if ledger_spec["format"] == "generic-v1":
        return any(row.get("artifact_sha256") == package_hash for row in load_jsonl(ledger_path))
    if ledger_spec["format"] == "task78-v1":
        module = task78_module(profile)
        return any(row.get("local_archive_sha256") == package_hash
                   for row in module.load_ledger(ledger_path))
    raise WorkflowError(f"unsupported ledger format: {ledger_spec['format']}")


def validate_transition(task_id: str, directory: Path, from_state: str, to_state: str,
                        evidence: dict) -> None:
    profile = get_run_profile(directory)
    if to_state not in TRANSITIONS.get(from_state, set()):
        raise WorkflowError(f"transition not allowed: {from_state} -> {to_state}")
    if to_state == "tool_unavailable":
        if evidence.get("service_state") not in {"absent", "configured_unavailable"}:
            raise WorkflowError("tool_unavailable needs service_state absent/configured_unavailable")
        if evidence.get("active_registry_checked") is not True:
            raise WorkflowError("tool_unavailable requires a live registry check")
    elif to_state == "prepared":
        if evidence.get("service_state") != "callable" or not evidence.get("tool_name"):
            raise WorkflowError("prepared requires a callable registered operation and its exact name")
        baseline_hashes = evidence.get("baseline_hashes")
        if not isinstance(baseline_hashes, dict) or set(baseline_hashes) != set(profile["targets"]):
            raise WorkflowError("prepared requires a frozen baseline-input hash for every profile target")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in baseline_hashes.values()):
            raise WorkflowError("every per-target baseline input must be an exact SHA-256")
        baseline = profile["baseline"]
        baseline_mode = baseline.get("mode", "optimize_existing_sources")
        if baseline_mode == "generate_from_official_reference":
            reference = baseline["reference_source"]
            reference_hash = reference["sha256"]
            expected = {target: reference_hash for target in profile["targets"]}
            if evidence.get("baseline_kind") != "official_reference" or baseline_hashes != expected:
                raise WorkflowError("initial generation must freeze the official reference hash as every target's baseline input")
        elif evidence.get("baseline_kind", "candidate_source") != "candidate_source":
            raise WorkflowError("existing-source optimization must identify its baseline inputs as candidate_source")
        forecast_status = evidence.get("forecast_status")
        if forecast_status == "pending":
            plan_hash = evidence.get("generation_plan_sha256")
            if (not isinstance(evidence.get("pre_generation_hypothesis"), str)
                    or not evidence["pre_generation_hypothesis"].strip()
                    or not isinstance(plan_hash, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", plan_hash)):
                raise WorkflowError(
                    "diagnostic generation requires a frozen structural hypothesis and plan SHA-256"
                )
        elif forecast_status != "pass":
            raise WorkflowError("prepared forecast_status must be pass or pending for diagnostic generation")
        if not evidence.get("frozen_task_goal") or not evidence.get("goal_source"):
            raise WorkflowError("prepared requires the task goal and its official evidence source")
        if from_state == "prepared":
            if (evidence.get("activity") != "generation_started"
                    or evidence.get("source_generation_started") is not True):
                raise WorkflowError(
                    "prepared self-checkpoint is reserved for a durable generation-start marker"
                )
        elif evidence.get("source_generation_started") is True:
            raise WorkflowError("the first prepared checkpoint cannot claim generation already started")
    elif to_state == "generated":
        source_hashes = evidence.get("source_hashes")
        members = profile["package"].get("root_members", [])
        if not evidence.get("request_sha256") or not isinstance(source_hashes, dict):
            raise WorkflowError("generated requires request and returned-source hashes")
        if not evidence.get("service_job_id"):
            receipt_hash = evidence.get("service_call_receipt_sha256")
            receipt_hashes = evidence.get("service_call_receipts_sha256")
            receipt_bundle_valid = (
                isinstance(receipt_hashes, dict)
                and set(receipt_hashes) == set(members)
                and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                        for value in receipt_hashes.values())
            )
            if ((not isinstance(receipt_hash, str)
                 or not re.fullmatch(r"[0-9a-f]{64}", receipt_hash))
                    and not receipt_bundle_valid):
                raise WorkflowError(
                    "generated requires a service job id, saved synchronous receipt SHA-256, or complete per-member receipt hashes"
                )
        if set(source_hashes) != set(members):
            raise WorkflowError("generated source hashes must cover exactly the adapter's package root_members")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in source_hashes.values()):
            raise WorkflowError("every returned package source must have an exact lowercase SHA-256")
    elif to_state == "locally_validated":
        if evidence.get("contract") != "pass" or evidence.get("local_correctness") != "pass":
            raise WorkflowError("locally_validated requires contract and local semantic gates to pass")
        if not evidence.get("reports"):
            raise WorkflowError("locally_validated requires saved report paths/hashes")
    elif to_state == "reviewed":
        reviewers = evidence.get("reviewers")
        if not isinstance(reviewers, list) or len(reviewers) < 2:
            raise WorkflowError("reviewed requires two independent reviewer receipts")
        ids = [item.get("invocation_id") for item in reviewers if isinstance(item, dict)]
        if len(ids) < 2 or not ids[0] or not ids[1] or ids[0] == ids[1]:
            raise WorkflowError("reviewer receipts need distinct platform invocation ids")
        if any(item.get("status") != "pass" for item in reviewers[:2]):
            raise WorkflowError("both independent reviews must pass")
    elif to_state == "target_validated":
        if evidence.get("correctness") != "pass" or set(evidence.get("targets_passed", [])) != set(profile["targets"]):
            raise WorkflowError("target_validated requires complete target correctness coverage")
        if not evidence.get("raw_target_evidence_hash"):
            raise WorkflowError("target_validated requires preserved raw target evidence")
    elif to_state == "measured":
        if evidence.get("score_source") == "official-platform":
            if set(evidence.get("target_results", {})) != set(profile["targets"]):
                raise WorkflowError("official measurement requires every profile target result")
            if "official_aggregate" not in evidence:
                raise WorkflowError("official measurement requires the platform aggregate value")
        elif evidence.get("case_coverage") != "complete" or evidence.get("minimum_samples_per_case", 0) < 5:
            raise WorkflowError("trusted measurement requires complete cases and at least five samples/case")
        elif evidence.get("benchmark_method") != "trusted-target-runner":
            raise WorkflowError("measured requires official platform scores or a trusted complete target runner")
    elif to_state == "package_ready":
        if evidence.get("hurdle") != "pass" or not evidence.get("package_sha256"):
            raise WorkflowError("package_ready requires a passing release hurdle and package SHA-256")
        artifact_path = (ROOT / str(evidence.get("artifact_path", ""))).resolve()
        if (not evidence.get("artifact_path") or not artifact_path.is_file()
                or not str(artifact_path).startswith(str(ROOT) + os.sep)):
            raise WorkflowError("package_ready requires the exact ZIP artifact path inside the repository")
        artifact_bytes = artifact_path.read_bytes()
        if sha256_bytes(artifact_bytes) != evidence.get("package_sha256"):
            raise WorkflowError("package_ready archive bytes do not match the frozen package SHA-256")
        try:
            with zipfile.ZipFile(artifact_path) as archive:
                names = archive.namelist()
                expected_members = profile["package"]["root_members"]
                if set(names) != set(expected_members) or len(names) != len(expected_members):
                    raise WorkflowError("package ZIP does not contain exactly the adapter's root_members")
                packaged_hashes = {name: sha256_bytes(archive.read(name)) for name in expected_members}
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            raise WorkflowError(f"cannot verify package-ready ZIP: {exc}") from exc
        generated = next((event.get("evidence", {}) for event in reversed(current_state(directory)["events"])
                          if event.get("to") == "generated"), None)
        if generated is None or packaged_hashes != generated.get("source_hashes"):
            raise WorkflowError("package ZIP source bytes do not match the exact KernelGen outputs")
        if from_state == "reviewed":
            if evidence.get("target_validation_mode") != "official-flagos-evaluation":
                raise WorkflowError("reviewed candidates need a trusted target preflight or the session-authorized official evaluation path")
            if not SESSION_AUTH.is_file() or not read_json(SESSION_AUTH).get("official_evaluation_as_target_validation"):
                raise WorkflowError("official FlagOS evaluation as target validation is not authorized for this run")
    elif to_state == "upload_armed":
        authorize_upload(task_id, evidence)
        package_hash = evidence.get("package_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", str(package_hash or "")):
            raise WorkflowError("upload_armed requires the exact 64-character package SHA-256")
        used = [event for _, event in inspect_global_submissions()
                if event.get("evidence", {}).get("package_sha256") == package_hash]
        if used or ledger_has_package(profile, package_hash):
            raise WorkflowError("this exact package hash already has an upload attempt/result; inspect its FlagOS record")
        last = max((event.get("at", "") for _, event in inspect_global_submissions()), default="")
        if last:
            try:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last.replace("Z", "+00:00"))).total_seconds()
            except ValueError:
                raise WorkflowError("cannot verify the global submission interval from prior events")
            if elapsed <= 120:
                raise WorkflowError(f"FlagOS rate limit requires >120 seconds; last upload was {elapsed:.1f}s ago")
    elif to_state in {"submitted", "record_confirmed"}:
        if not evidence.get("submission_id"):
            raise WorkflowError(f"{to_state} requires a verified FlagOS submission id")
        if not evidence.get("package_sha256"):
            raise WorkflowError(f"{to_state} requires the submitted package SHA-256")
        if to_state == "submitted" and evidence.get("upload_attempted") is not True:
            raise WorkflowError("submitted requires a recorded single Chrome upload attempt")
        armed = next((event.get("evidence", {}) for event in reversed(load_events(directory))
                      if event.get("to") == "upload_armed"), None)
        if armed is None or armed.get("package_sha256") != evidence.get("package_sha256"):
            raise WorkflowError("submission record package does not match the armed Chrome upload")
    elif to_state == "upload_aborted":
        if evidence.get("records_checked") is not True or evidence.get("matching_record_found") is not False:
            raise WorkflowError("upload_aborted requires a fresh Chrome record check proving no matching record")
        if evidence.get("upload_attempted") is not False or not evidence.get("package_sha256"):
            raise WorkflowError("upload_aborted requires proof that no click was sent and the package hash")
    elif to_state == "evaluating":
        if not evidence.get("submission_id") or evidence.get("status") not in {"queued", "running"}:
            raise WorkflowError("evaluating requires the confirmed record in a live nonterminal state")
    elif to_state == "completed":
        if evidence.get("terminal_status") != "completed":
            raise WorkflowError("completed requires a terminal completed FlagOS record")
        if set(evidence.get("target_results", {})) != set(profile["targets"]):
            raise WorkflowError("completed requires a result entry for every profile target")
        normalize_targets(profile, evidence)
        if evidence.get("record_identity_verified") is not True:
            raise WorkflowError("completed requires matching task, batch, team scope, and package identity")
        if not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("raw_result_sha256", ""))):
            raise WorkflowError("completed requires a hash of the saved raw FlagOS result")
        raw_path = (directory / str(evidence.get("raw_result_path", ""))).resolve()
        if not raw_path.is_file() or not str(raw_path).startswith(str(directory.resolve()) + os.sep):
            raise WorkflowError("completed requires a saved raw result file inside this run")
        if sha256_bytes(raw_path.read_bytes()) != evidence["raw_result_sha256"]:
            raise WorkflowError("saved raw FlagOS result hash does not match the completion receipt")
        armed = next((event.get("evidence", {}) for event in reversed(load_events(directory))
                      if event.get("to") == "upload_armed"), None)
        if (armed is None or armed.get("package_sha256") != evidence.get("package_sha256")
                or evidence.get("submission_id") is None):
            raise WorkflowError("completed FlagOS result is not bound to the exact armed package")
        confirmed = next((event.get("evidence", {}) for event in reversed(load_events(directory))
                          if event.get("to") in {"record_confirmed", "submitted"}), None)
        if confirmed is not None and confirmed.get("submission_id") != evidence.get("submission_id"):
            raise WorkflowError("terminal result belongs to a different FlagOS record")
    elif to_state == "promoted":
        if evidence.get("release_goal") != "reached" or evidence.get("all_targets_correct") is not True:
            raise WorkflowError("promotion requires the per-task goal and all-target correctness")


def authorize_upload(task_id: str, evidence: dict) -> None:
    if not SESSION_AUTH.is_file():
        raise WorkflowError("session authorization is missing; no FlagOS upload is allowed")
    auth = read_json(SESSION_AUTH)
    if auth.get("competition") != "flagos-s2" or auth.get("scope") != "all-tasks-current-team-current-visible-daily-quota":
        raise WorkflowError("session authorization does not cover this FlagOS S2 task")
    if auth.get("browser") != "chrome" or evidence.get("browser") != "chrome":
        raise WorkflowError("upload must use Chrome")
    required = ("task_id", "batch_id", "team_ref", "visible_quota_remaining", "artifact_path",
                "quota_checked_at", "latest_record_checked", "duplicate_checked",
                "target_validation_mode")
    missing = [name for name in required if evidence.get(name) is None]
    if missing:
        raise WorkflowError("upload preflight missing current Chrome evidence: " + ", ".join(missing))
    if evidence["task_id"] != task_id:
        raise WorkflowError("Chrome currently selected a different task")
    if not evidence.get("batch_id") or not evidence.get("team_ref"):
        raise WorkflowError("selected batch and currently logged-in team must be identified in Chrome")
    try:
        visible_quota = int(evidence["visible_quota_remaining"])
    except (TypeError, ValueError) as exc:
        raise WorkflowError("visible_quota_remaining must be an integer read from Chrome") from exc
    if visible_quota <= 0:
        raise WorkflowError("current visible daily quota is exhausted")
    artifact_path = (ROOT / evidence["artifact_path"]).resolve()
    if not artifact_path.is_file() or not str(artifact_path).startswith(str(ROOT) + os.sep):
        raise WorkflowError("the package checked in Chrome must exist inside this repository")
    if sha256_bytes(artifact_path.read_bytes()) != evidence.get("package_sha256"):
        raise WorkflowError("Chrome preflight package hash differs from the local archive bytes")
    if evidence["latest_record_checked"] is not True or evidence["duplicate_checked"] is not True:
        raise WorkflowError("must inspect latest record and duplicate package history before upload")
    if evidence["target_validation_mode"] not in {"trusted-preflight", "official-flagos-evaluation"}:
        raise WorkflowError("target validation must be a trusted preflight or the authorized official FlagOS evaluation")
    if evidence["target_validation_mode"] == "official-flagos-evaluation" and not auth.get("official_evaluation_as_target_validation"):
        raise WorkflowError("official FlagOS evaluation as target validation is not authorized")
    if evidence.get("team_matches_session") is not True:
        raise WorkflowError("currently logged-in team was not confirmed")
    try:
        checked = datetime.fromisoformat(evidence["quota_checked_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise WorkflowError("quota_checked_at must be an ISO timestamp") from exc
    if (datetime.now(timezone.utc) - checked.astimezone(timezone.utc)).total_seconds() > 300:
        raise WorkflowError("quota snapshot is older than five minutes; refresh it in Chrome")
    if (datetime.now(timezone.utc) - checked.astimezone(timezone.utc)).total_seconds() < -60:
        raise WorkflowError("quota snapshot timestamp is in the future")


def cmd_validate(args) -> None:
    profile, digest = get_profile(args.task_id)
    print(json.dumps({"task_id": profile["task_id"], "profile_sha256": digest,
                      "operator": profile["operator"], "target_count": len(profile["targets"]),
                      "source_files": profile["source_files"], "status": "valid"}, ensure_ascii=False, indent=2))


def cmd_new(args) -> None:
    profile, digest = get_profile(args.task_id)
    directory = run_dir(args.task_id, args.run_id)
    if directory.exists():
        raise WorkflowError(f"run already exists: {directory}")
    directory.mkdir(parents=True)
    shutil = __import__("shutil")
    shutil.copyfile(profile_path(args.task_id), directory / "task-profile.json")
    git_head = "unknown"
    git_branch = "unknown"
    try:
        import subprocess
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    manifest = {
        "schema_version": 1,
        "task_id": args.task_id,
        "run_id": args.run_id,
        "created_at": utc_now(),
        "profile_sha256": digest,
        "repository_revision": git_head,
        "repository_branch": git_branch,
        "parent_run_id": args.parent_run_id,
        "profile_snapshot": "task-profile.json",
        "events": "events.jsonl",
        "resume_rule": "inspect Chrome records before any action from upload_armed or later",
    }
    write_json(directory / "run.json", manifest)
    append_event(directory, "<new>", "planned", {"reason": args.reason})
    print(f"created {args.task_id}/{args.run_id}: planned; checkpoint root {directory}")


def cmd_checkpoint(args) -> None:
    directory = run_dir(args.task_id, args.run_id)
    if not directory.is_dir():
        raise WorkflowError(f"unknown run: {args.task_id}/{args.run_id}")
    get_run_profile(directory)
    state = current_state(directory)["state"]
    evidence = read_json(Path(args.evidence).resolve()) if args.evidence else {"note": args.note}
    validate_transition(args.task_id, directory, state, args.to, evidence)
    append_event(directory, state, args.to, evidence)
    print(f"checkpointed {args.task_id}/{args.run_id}: {state} -> {args.to}")


def cmd_resume(args) -> None:
    directory = run_dir(args.task_id, args.run_id)
    manifest = read_json(directory / "run.json")
    profile = get_run_profile(directory)
    state_data = current_state(directory)
    latest = state_data["events"][-1]
    state = state_data["state"]
    actions = {
        "planned": "check the live KernelGen registry; do not edit source before callable",
        "tool_unavailable": "resume only after the required operation appears in the live registry",
        "prepared": "call KernelGen with the frozen request and hypothesis; a pending forecast keeps the run diagnostic-only until the release hurdle passes",
        "generated": "run the adapter local contract and semantic gates",
        "locally_validated": "obtain two distinct fresh source-review receipts",
        "reviewed": "run trusted target preflight, or use the session-authorized official FlagOS evaluation path",
        "target_validated": "measure the exact declared workload",
        "measured": "apply the task release hurdle and prepare the immutable ZIP",
        "package_ready": "refresh task, batch, team, quota, duplicate and package checks in Chrome",
        "upload_armed": "inspect Chrome submission records first; never blindly click/retry",
        "upload_aborted": "a fresh record check proved no submission; revalidate package and quota before reconsidering",
        "submitted": "confirm the matching FlagOS record and poll that record",
        "record_confirmed": "wait on the confirmed record; do not create another submission",
        "evaluating": "resume polling the same FlagOS record",
        "completed": "turn the terminal record into target-correctness and score evidence",
        "promoted": "task goal reached; stop this task's iteration",
        "rejected": "preserve the attempt and create a child run only for a justified repair",
        "inconclusive": "resolve the named evidence gap before continuing",
    }[state]
    if state == "prepared" and latest.get("evidence", {}).get("source_generation_started") is True:
        actions = (
            "resume from the saved generation manifest; keep completed target responses and invoke only targets without a saved response"
        )
    print(json.dumps({"task_id": args.task_id, "run_id": args.run_id, "state": state,
                      "updated_at": latest.get("at"), "next_action": actions,
                      "repository_revision": manifest.get("repository_revision"),
                      "profile_sha256": manifest.get("profile_sha256"),
                      "operator": profile.get("operator"),
                      "event_count": len(state_data["events"]),
                      "evidence": latest.get("evidence", {})}, ensure_ascii=False, indent=2))


def append_line(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def persist_result_row(path: Path, rows: list[dict], row: dict, *,
                       record_id: str, package_hash: str, package_key: str,
                       identity_fields: tuple[str, ...]) -> bool:
    """Append once, or accept an identical row after an interrupted resume."""
    same_record = [item for item in rows if item.get("record_id") == record_id]
    if len(same_record) > 1:
        raise WorkflowError(f"duplicate result record id already exists: {record_id}")
    if same_record:
        prior = same_record[0]
        mismatches = [field for field in identity_fields if prior.get(field) != row.get(field)]
        if mismatches:
            raise WorkflowError(
                f"record {record_id} already exists with conflicting result fields: "
                + ", ".join(mismatches)
            )
        return False
    if any(item.get(package_key) == package_hash for item in rows):
        raise WorkflowError("this exact package is already represented under another result record")
    append_line(path, row)
    return True


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"invalid ledger JSON at {path}:{line_number}: {exc}") from exc
    return rows


def task78_module(profile: dict):
    validator = ROOT / profile["result_ledger"].get("validator", "")
    spec = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location(
        "_flagos_task78_results", validator
    )
    if spec is None or spec.loader is None:
        raise WorkflowError(f"cannot load Task 78 ledger adapter: {validator}")
    module = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_targets(profile: dict, evidence: dict) -> tuple[dict, int]:
    results = evidence.get("target_results")
    targets = set(profile["targets"])
    if not isinstance(results, dict) or set(results) != targets:
        raise WorkflowError("official result needs exactly the task profile's target ids")
    normalized = {}
    passed = 0
    for target in profile["targets"]:
        item = results[target]
        if not isinstance(item, dict):
            raise WorkflowError(f"{target}: official target result must be an object")
        status = item.get("status")
        score = item.get("speedup")
        if (status == "pass" and isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(score) and score > 0):
            passed += 1
        elif status != "fail" or score is not None:
            raise WorkflowError(f"{target}: pass needs positive score; fail needs null speedup")
        normalized[target] = item
    aggregate = evidence.get("official_aggregate")
    if (passed == len(targets) and (not isinstance(aggregate, (int, float))
                                    or isinstance(aggregate, bool) or not math.isfinite(aggregate)
                                    or aggregate <= 0)):
        raise WorkflowError("all targets passed but official aggregate is missing")
    if passed != len(targets) and aggregate is not None:
        raise WorkflowError("partial official result cannot have an aggregate")
    return normalized, passed


def zip_source_hashes(profile: dict, archive_path: Path) -> dict[str, dict[str, str]]:
    expected = profile["package"].get("root_members")
    if not isinstance(expected, list) or not expected:
        raise WorkflowError("task package profile must list exact root_members")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if set(names) != set(expected) or len(names) != len(expected):
                raise WorkflowError(f"package ZIP must contain exactly these root members: {expected}")
            hashes = {name: sha256_bytes(archive.read(name)) for name in expected}
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise WorkflowError(f"cannot verify package ZIP contents: {exc}") from exc
    mapping = profile.get("target_source_map")
    if mapping:
        return {target: {"member": mapping[target], "sha256": hashes[mapping[target]]}
                for target in profile["targets"]}
    if len(expected) != 1:
        raise WorkflowError("multi-source task profile must define target_source_map")
    member = expected[0]
    return {target: {"member": member, "sha256": hashes[member]} for target in profile["targets"]}


def cmd_record_result(args) -> None:
    directory = run_dir(args.task_id, args.run_id)
    profile = get_run_profile(directory)
    state = current_state(directory)
    if state["state"] != "completed":
        raise WorkflowError("append official results only from a terminal completed run")
    evidence = state["events"][-1].get("evidence", {})
    targets, passed = normalize_targets(profile, evidence)
    if evidence.get("record_identity_verified") is not True:
        raise WorkflowError("official FlagOS result identity has not been confirmed")
    package_path = (ROOT / evidence.get("artifact_path", "")).resolve()
    if not package_path.is_file() or not str(package_path).startswith(str(ROOT) + os.sep):
        raise WorkflowError("the exact local package artifact is missing or outside this repository")
    archive_hash = sha256_bytes(package_path.read_bytes())
    if archive_hash != evidence.get("package_sha256"):
        raise WorkflowError("local package bytes do not match the recorded upload hash")
    ledger_spec = profile["result_ledger"]
    ledger_path = ROOT / ledger_spec["path"]
    raw_digest = evidence["raw_result_sha256"]
    common_targets = {}
    if ledger_spec["format"] == "generic-v1":
        source_hashes = zip_source_hashes(profile, package_path)
        for target, item in targets.items():
            expected_source = source_hashes[target]
            if item.get("source_sha256") and item["source_sha256"] != expected_source["sha256"]:
                raise WorkflowError(f"{target}: official source hash does not match the uploaded package")
            common_targets[target] = {
                "status": item["status"],
                "speedup": item.get("speedup"),
                "source_sha256": expected_source["sha256"],
                "source_member": expected_source["member"],
                "failure": item.get("failure"),
            }
        rows = load_jsonl(ledger_path)
        record_id = str(evidence.get("submission_id", ""))
        if not record_id:
            raise WorkflowError("missing FlagOS record id in result ledger")
        row = {
            "schema_version": 1,
            "record_id": record_id,
            "task_id": args.task_id,
            "version": evidence.get("version", args.run_id),
            "submitted_at": evidence.get("submitted_at"),
            "batch_id": evidence.get("batch_id"),
            "submission_id": record_id,
            "artifact": str(package_path.relative_to(ROOT)),
            "artifact_sha256": archive_hash,
            "source_commit": evidence.get("source_commit"),
            "source_sha256": next(iter(source_hashes.values()))["sha256"] if len(source_hashes) == 1 else None,
            "provenance": "confirmed FlagOS result; exact local package hash verified",
            "raw_result_sha256": raw_digest,
            "targets": common_targets,
            "pass_count": passed,
            "aggregate_speedup": evidence.get("official_aggregate"),
            "notes": evidence.get("notes", ""),
        }
        persist_result_row(
            ledger_path, rows, row, record_id=record_id, package_hash=archive_hash,
            package_key="artifact_sha256",
            identity_fields=("record_id", "task_id", "artifact_sha256", "raw_result_sha256",
                             "targets", "pass_count", "aggregate_speedup"),
        )
        render_generic_ledger(profile, load_jsonl(ledger_path))
    elif ledger_spec["format"] == "task78-v1":
        module = task78_module(profile)
        rows = module.load_ledger(ledger_path)
        record_id = str(evidence.get("submission_id", ""))
        if not record_id:
            raise WorkflowError("missing FlagOS record id in Task 78 ledger")
        target_records = {}
        for target, item in targets.items():
            source_hash, member = module.target_source_hash(
                {"artifact": package_path.name, "targets": {target: item}}, target, package_path.parent
            )
            if item.get("source_sha256") and item["source_sha256"] != source_hash:
                raise WorkflowError(f"{target}: official source hash does not match the uploaded Task 78 package")
            target_records[target] = {
                "status": item["status"],
                "speedup": item.get("speedup"),
                "source_sha256": source_hash,
                "source_member": member,
            }
        row = {
            "record_id": record_id,
            "task": 78,
            "batch": int(re.sub(r"\D", "", str(evidence.get("batch_id", ""))) or 0),
            "submitted_at": evidence.get("submitted_at"),
            "version": evidence.get("version", args.run_id),
            "artifact": package_path.name,
            "local_archive_sha256": archive_hash,
            "raw_result_sha256": raw_digest,
            "archive_identity": "local bytes verified against upload record; platform upload digest not exposed",
            "status": "completed",
            "targets": target_records,
            "pass_count": passed,
            "aggregate_speedup": evidence.get("official_aggregate"),
            "notes": "Official FlagOS result recorded by the shared FlagOS S2 workflow.",
        }
        prior = next((item for item in rows if item.get("record_id") == record_id), None)
        validation_rows = rows if prior is not None else rows + [row]
        errors = module.validate_ledger(validation_rows, package_path.parent)
        if errors:
            raise WorkflowError("Task 78 ledger validation failed:\n- " + "\n- ".join(errors))
        persist_result_row(
            ledger_path, rows, row, record_id=record_id, package_hash=archive_hash,
            package_key="local_archive_sha256",
            identity_fields=("record_id", "task", "local_archive_sha256", "raw_result_sha256",
                             "targets", "pass_count", "aggregate_speedup"),
        )
        ledger_rows = module.load_ledger(ledger_path)
        Path(ROOT / ledger_spec["report"]).write_text(
            module.render(ledger_rows, package_path.parent), encoding="utf-8"
        )
    else:
        raise WorkflowError(f"unsupported ledger format: {ledger_spec['format']}")
    write_json(directory / "ledger-row.json", row)
    print(f"appended FlagOS record {row['record_id']} to {ledger_path}")


def render_generic_ledger(profile: dict, rows: list[dict]) -> None:
    path = ROOT / profile["result_ledger"].get("report", "")
    labels = profile.get("target_labels", {})
    lines = [f"# {profile['task_id'].upper()} official result ledger", "",
             f"Records: {len(rows)}. Aggregate values are copied from the official platform; "
             "partial target results never receive an aggregate.", "",
             "| Record | Version | Submitted | Pass | Official aggregate |", "| --- | --- | --- | ---: | ---: |"]
    for row in rows:
        aggregate = row.get("aggregate_speedup")
        aggregate_text = f"{aggregate:.2f}x" if isinstance(aggregate, (int, float)) else "—"
        lines.append(f"| {row.get('record_id')} | {row.get('version')} | {row.get('submitted_at')} "
                     f"| {row.get('pass_count')}/{len(profile['targets'])} | {aggregate_text} |")
    lines += ["", "## Per-target best observations", "", "| Target | Best score | Record |", "| --- | ---: | --- |"]
    for target in profile["targets"]:
        candidates = [row for row in rows if row.get("targets", {}).get(target, {}).get("status") == "pass"]
        best = max(candidates, key=lambda row: row["targets"][target]["speedup"], default=None)
        score = f"{best['targets'][target]['speedup']:.2f}x" if best else "—"
        lines.append(f"| {labels.get(target, target)} | {score} | {best.get('record_id') if best else '—'} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_list(args) -> None:
    inventory = read_json(TASK_INVENTORY) if TASK_INVENTORY.is_file() else None
    selected_task_ids = args.tasks or []
    if len(selected_task_ids) != len(set(selected_task_ids)):
        raise WorkflowError("--tasks must not contain duplicate task ids")
    if selected_task_ids and (
            inventory is None
            or inventory.get("coverage", {}).get("complete") is not True
            or inventory.get("coverage", {}).get("captured_from_chrome") is not True):
        raise WorkflowError("explicit task selection requires a complete current task inventory captured from Chrome")
    contract_document = read_json(TASK_CONTRACT_EVIDENCE) if TASK_CONTRACT_EVIDENCE.is_file() else None
    contract_rows = {}
    contract_document_errors = []
    if contract_document is not None:
        if contract_document.get("schema_version") != 1:
            contract_document_errors.append("task contract evidence schema_version must be 1")
        if contract_document.get("competition") != "flagos-s2":
            contract_document_errors.append("task contract evidence competition must be flagos-s2")
        captured_tasks = contract_document.get("tasks")
        if not isinstance(captured_tasks, list):
            contract_document_errors.append("task contract evidence tasks must be a list")
        else:
            for item in captured_tasks:
                if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
                    contract_document_errors.append("task contract evidence contains a row without task_id")
                    continue
                if item["task_id"] in contract_rows:
                    contract_document_errors.append(f"duplicate task contract evidence for {item['task_id']}")
                    continue
                contract_rows[item["task_id"]] = item

    runtime_check = read_json(KERNELGEN_RUNTIME_CHECK) if KERNELGEN_RUNTIME_CHECK.is_file() else None
    kernelgen_status, missing_operations = assess_kernelgen_runtime(runtime_check)
    tasks = {}
    if inventory is not None:
        validate_inventory(inventory)
        for item in inventory["tasks"]:
            tasks[item["task_id"]] = {
                "task_id": item["task_id"],
                "task_name": item["task_name"],
                "batch_id": item["batch_id"],
                "availability": item["availability"],
                "detail_url": item.get("detail_url"),
                "inventory_source": inventory["source_url"],
            }

    for path in sorted(PROFILE_DIR.glob("task[0-9]*.json")):
        profile = read_json(path)
        task_id = path.stem
        tasks.setdefault(task_id, {
            "task_id": task_id,
            "task_name": profile.get("task_name"),
            "availability": "unresolved",
        })

    rows = []
    for task_id, row in sorted(tasks.items(), key=lambda item: task_sort_key(item[0])):
        contract = contract_rows.get(task_id)
        if contract is not None:
            row["contract_evidence"] = {
                "status": contract.get("status", "unknown"),
                "capture_type": contract.get("capture_type"),
                "operator_path": contract.get("operator_path"),
                "missing_for_adapter": contract.get("missing_for_adapter", []),
                "profile": contract.get("profile"),
            }
        profile_file = PROFILE_DIR / f"{task_id}.json"
        if profile_file.is_file():
            profile = read_json(profile_file)
            errors = validate_profile(profile)
            if profile.get("task_id") != task_id:
                errors.append(f"profile task_id does not match filename/id {task_id}")
            row.update({
                "profile": str(profile_file.relative_to(ROOT)),
                "profile_valid": not errors,
                "profile_errors": errors,
                "readiness": "ready" if not errors else "invalid_profile",
            })
        else:
            row.update({
                "profile": None,
                "profile_valid": False,
                "profile_errors": ["task-specific contract/profile is missing"],
                "readiness": "needs_task_adapter",
            })
        if row["profile_valid"]:
            row["contract_next_action"] = "use_validated_task_adapter"
        elif contract is None:
            row["contract_next_action"] = "capture_official_task_contract_in_chrome"
        else:
            row["contract_next_action"] = "resolve_missing_adapter_evidence_without_inference"
        latest = latest_run_summary(task_id)
        row["latest_run"] = latest
        latest_state = latest.get("state") if latest else None
        if latest_state in {"upload_armed", "submitted", "record_confirmed", "evaluating"}:
            row["queue_priority"] = "resume_before_new_upload_or_iteration"
            row["execution_readiness"] = "resume_submission_checkpoint"
        elif latest_state == "completed":
            row["queue_priority"] = "record_terminal_result"
            row["execution_readiness"] = "record_terminal_result"
        elif row.get("availability") == "open" and row["readiness"] != "ready":
            row["queue_priority"] = "onboard_task_contract"
            row["execution_readiness"] = "needs_task_adapter"
        elif row.get("availability") == "locked":
            row["queue_priority"] = "wait_for_batch_to_open"
            row["execution_readiness"] = "not_open"
        elif row.get("availability") != "open":
            row["queue_priority"] = "refresh_open_task_status_in_chrome"
            row["execution_readiness"] = "open_status_unconfirmed"
        elif not row["profile_valid"]:
            row["queue_priority"] = "onboard_task_contract"
            row["execution_readiness"] = "needs_task_adapter"
        elif latest_state == "inconclusive":
            row["queue_priority"] = "resolve_inconclusive_evidence"
            row["execution_readiness"] = "blocked_on_recorded_evidence_gap"
        elif latest_state == "promoted":
            row["queue_priority"] = "goal_reached"
            row["execution_readiness"] = "goal_reached"
        elif latest_state in {None, "planned", "prepared", "tool_unavailable"}:
            if kernelgen_status["state"] == "unchecked":
                row["queue_priority"] = "check_kernelgen_registry"
                row["execution_readiness"] = "kernelgen_unchecked"
            elif missing_operations:
                row["queue_priority"] = (
                    "resume_after_kernelgen_is_live" if latest_state == "tool_unavailable"
                    else "wait_for_kernelgen"
                )
                row["execution_readiness"] = "waiting_for_kernelgen"
            elif latest_state == "tool_unavailable":
                row["queue_priority"] = "resume_after_kernelgen_is_live"
                row["execution_readiness"] = "ready_to_resume"
            elif latest_state is None:
                row["queue_priority"] = "start_first_iteration"
                row["execution_readiness"] = "ready_to_generate"
            else:
                row["queue_priority"] = "resume_or_repair"
                row["execution_readiness"] = "ready_to_resume"
        elif latest:
            row["queue_priority"] = "resume_or_repair"
            row["execution_readiness"] = "ready_to_resume"
        else:
            row["queue_priority"] = "start_first_iteration"
            row["execution_readiness"] = "ready_to_generate"
        row["selected_for_iteration"] = task_id in selected_task_ids
        rows.append(row)

    rows.sort(key=queue_sort_key)
    known_open = {row["task_id"] for row in rows if row.get("availability") == "open"}
    invalid_selection = [task_id for task_id in selected_task_ids if task_id not in known_open]
    if invalid_selection:
        raise WorkflowError("selected task ids are not marked open in the current Chrome inventory: "
                            + ", ".join(invalid_selection))

    discovery = {
        "inventory_path": str(TASK_INVENTORY.relative_to(ROOT)),
        "inventory_present": inventory is not None,
        "observed_at": inventory.get("observed_at") if inventory else None,
        "source_url": inventory.get("source_url") if inventory else None,
        "open_task_count_from_chrome": inventory.get("coverage", {}).get("open_task_count") if inventory else None,
        "open_tasks_listed": sum(1 for row in rows if row.get("availability") == "open"),
        "inventory_complete": inventory.get("coverage", {}).get("complete") if inventory else False,
    }
    open_rows = [row for row in rows if row.get("availability") == "open"]
    ready_open = [row for row in open_rows if row.get("profile_valid")]
    captured_open = [row for row in open_rows if row.get("contract_evidence")]
    open_queue = [row for row in rows if row.get("availability") == "open"]
    selected_queue = [row for row in open_queue if row["selected_for_iteration"]]
    campaign = {
        "lifecycle": "shared-task-neutral",
        "scope": "only task ids explicitly selected by the user are eligible for iteration",
        "open_task_count": len(open_rows),
        "open_tasks_with_contract_evidence": len(captured_open),
        "open_tasks_without_contract_evidence": len(open_rows) - len(captured_open),
        "open_tasks_with_valid_adapters": len(ready_open),
        "open_tasks_waiting_for_kernelgen": sum(
            row.get("execution_readiness") == "waiting_for_kernelgen" for row in open_rows
        ),
        "kernelgen": kernelgen_status,
        "selected_task_ids": selected_task_ids,
        "selected_task_count": len(selected_queue),
        "selection_required": not selected_task_ids,
        "open_task_overview_order": [row["task_id"] for row in open_queue],
        "queue_order": [row["task_id"] for row in selected_queue],
        "next_task": ({
            "task_id": selected_queue[0]["task_id"],
            "task_name": selected_queue[0].get("task_name"),
            "priority": selected_queue[0]["queue_priority"],
            "next_action": selected_queue[0]["execution_readiness"],
        } if selected_queue else None),
        "contract_evidence_errors": contract_document_errors,
    }
    print(json.dumps({"campaign": campaign, "discovery": discovery, "tasks": rows}, ensure_ascii=False, indent=2))


def task_sort_key(task_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"task([0-9]+)", task_id)
    return (int(match.group(1)) if match else sys.maxsize, task_id)


def queue_sort_key(row: dict) -> tuple[int, int, str]:
    """Order the campaign by recovery and readiness, then stable task id."""
    priority = QUEUE_PRIORITY_ORDER.get(row.get("queue_priority"), sys.maxsize)
    task_number, task_id = task_sort_key(str(row.get("task_id", "")))
    return priority, task_number, task_id


def assess_kernelgen_runtime(runtime_check: dict | None) -> tuple[dict, list[str]]:
    """Fail closed unless a fresh active registry snapshot proves every operation."""
    if not isinstance(runtime_check, dict):
        return ({"state": "unchecked", "source_generation_allowed": False},
                sorted(REQUIRED_KERNELGEN_OPERATIONS))
    checked_at = runtime_check.get("checked_at")
    timestamp = None
    try:
        if isinstance(checked_at, str):
            timestamp = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError:
        timestamp = None
    age_seconds = (
        (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()
        if timestamp is not None and timestamp.tzinfo is not None else None
    )
    fresh = age_seconds is not None and -60 <= age_seconds <= 300
    visible = runtime_check.get("visible_tool_registry")
    visible_tools = (
        set(visible) if isinstance(visible, list) and all(isinstance(x, str) for x in visible)
        else set()
    )
    missing_operations = sorted(REQUIRED_KERNELGEN_OPERATIONS - visible_tools)
    required_snapshot = runtime_check.get("required_operations")
    exact_contract = (
        isinstance(required_snapshot, list)
        and all(isinstance(x, str) for x in required_snapshot)
        and set(required_snapshot) == REQUIRED_KERNELGEN_OPERATIONS
    )
    registry_checked = runtime_check.get("active_registry_checked") is True
    if not fresh or not registry_checked or not exact_contract:
        return ({
            "state": "unchecked",
            "checked_at": checked_at,
            "missing_operations": missing_operations,
            "source_generation_allowed": False,
        }, missing_operations or sorted(REQUIRED_KERNELGEN_OPERATIONS))
    return ({
        "state": "available" if not missing_operations else "operation_unavailable",
        "checked_at": checked_at,
        "missing_operations": missing_operations,
        "source_generation_allowed": not missing_operations,
    }, missing_operations)


def validate_inventory(inventory: dict) -> None:
    if not isinstance(inventory, dict):
        raise WorkflowError("Chrome task inventory must be a JSON object")
    errors = []
    if inventory.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if inventory.get("competition") != "flagos-s2":
        errors.append("competition must be flagos-s2")
    if not isinstance(inventory.get("source_url"), str) or not inventory["source_url"].startswith("https://flagos.io/"):
        errors.append("source_url must be the official FlagOS HTTPS competition page")
    if not isinstance(inventory.get("observed_at"), str) or not inventory["observed_at"].endswith("Z"):
        errors.append("observed_at must be an ISO UTC timestamp ending in Z")
    coverage = inventory.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("scope") != "all-currently-open-tasks":
        errors.append("coverage.scope must be all-currently-open-tasks")
    elif (not isinstance(coverage.get("open_task_count"), int)
          or isinstance(coverage.get("open_task_count"), bool)
          or coverage["open_task_count"] < 0
          or not isinstance(coverage.get("complete"), bool)):
        errors.append("coverage must include a nonnegative open_task_count and boolean complete")
    tasks = inventory.get("tasks")
    if not isinstance(tasks, list):
        errors.append("tasks must be a list of exact task cards captured from Chrome")
        tasks = []
    seen = set()
    open_count = 0
    for index, item in enumerate(tasks):
        prefix = f"tasks[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not re.fullmatch(r"task[0-9]+", task_id):
            errors.append(f"{prefix}.task_id must be an exact id like task60")
        elif task_id in seen:
            errors.append(f"duplicate task id in Chrome inventory: {task_id}")
        else:
            seen.add(task_id)
        for field in ("task_name", "batch_id"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"{prefix}.{field} must be captured from the task card")
        if item.get("availability") not in {"open", "locked"}:
            errors.append(f"{prefix}.availability must be open or locked")
        if item.get("availability") == "open":
            open_count += 1
        if item.get("detail_url") is not None and not (
            isinstance(item["detail_url"], str) and item["detail_url"].startswith("https://flagos.io/")
        ):
            errors.append(f"{prefix}.detail_url must be an official HTTPS URL when present")
    if isinstance(coverage, dict) and coverage.get("complete") is True:
        if open_count != coverage.get("open_task_count"):
            errors.append("complete inventory must contain every currently open task card")
        if coverage.get("captured_from_chrome") is not True:
            errors.append("complete inventory must carry captured_from_chrome=true")
    if errors:
        raise WorkflowError("invalid Chrome task inventory:\n- " + "\n- ".join(errors))


def latest_run_summary(task_id: str) -> dict | None:
    task_root = RUN_ROOT / task_id
    if not task_root.is_dir():
        return None
    candidates = []
    for manifest_path in task_root.glob("*/run.json"):
        try:
            manifest = read_json(manifest_path)
            state = current_state(manifest_path.parent)["state"]
            candidates.append((str(manifest.get("created_at", "")), {
                "run_id": manifest.get("run_id", manifest_path.parent.name),
                "state": state,
                "updated_at": read_json(manifest_path.parent / "state.json").get("updated_at"),
                "repository_revision": manifest.get("repository_revision"),
            }))
        except (WorkflowError, OSError, json.JSONDecodeError):
            candidates.append((str(manifest_path.stat().st_mtime_ns), {
                "run_id": manifest_path.parent.name,
                "state": "checkpoint_invalid",
                "updated_at": None,
                "repository_revision": None,
            }))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def cmd_import_inventory(args) -> None:
    inventory = read_json(Path(args.inventory).resolve())
    validate_inventory(inventory)
    write_json(TASK_INVENTORY, inventory)
    print(json.dumps({
        "inventory_path": str(TASK_INVENTORY.relative_to(ROOT)),
        "observed_at": inventory["observed_at"],
        "listed_tasks": len(inventory["tasks"]),
        "open_tasks": sum(item["availability"] == "open" for item in inventory["tasks"]),
        "complete": inventory["coverage"]["complete"],
    }, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subs = root.add_subparsers(dest="command", required=True)
    validate = subs.add_parser("validate-profile")
    validate.add_argument("task_id")
    validate.set_defaults(func=cmd_validate)
    create = subs.add_parser("new-run")
    create.add_argument("task_id")
    create.add_argument("run_id")
    create.add_argument("--parent-run-id")
    create.add_argument("--reason", required=True)
    create.set_defaults(func=cmd_new)
    checkpoint = subs.add_parser("checkpoint")
    checkpoint.add_argument("task_id")
    checkpoint.add_argument("run_id")
    checkpoint.add_argument("--to", required=True, choices=sorted(STATES - {"planned"}))
    checkpoint.add_argument("--evidence")
    checkpoint.add_argument("--note")
    checkpoint.set_defaults(func=cmd_checkpoint)
    resume = subs.add_parser("resume")
    resume.add_argument("task_id")
    resume.add_argument("run_id")
    resume.set_defaults(func=cmd_resume)
    record = subs.add_parser("record-result")
    record.add_argument("task_id")
    record.add_argument("run_id")
    record.set_defaults(func=cmd_record_result)
    ls = subs.add_parser("list-tasks")
    ls.add_argument("--tasks", nargs="+", metavar="TASK_ID",
                    help="explicitly select task ids for iteration; without this, show overview only")
    ls.set_defaults(func=cmd_list)
    inventory = subs.add_parser("import-inventory")
    inventory.add_argument("inventory", help="Chrome-captured JSON task inventory")
    inventory.set_defaults(func=cmd_import_inventory)
    return root


def main() -> int:
    try:
        args = parser().parse_args()
        args.func(args)
        return 0
    except WorkflowError as exc:
        print(f"workflow error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
