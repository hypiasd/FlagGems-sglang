"""Evidence discipline for the competition-PR channel.

Adapted from the KernelGen workflow's reliability gates
(``.agents/skills/kernelgen-flagos/references/``).  The MCP tools themselves
are not reusable here -- their chip knowledge base covers Huawei only and they
are not registered in this session -- but the *evidence contract* is:

* capability is ``absent`` / ``configured_unavailable`` / ``callable``, and a
  config file, a reachable host or a valid token is never proof of the last
  one; only the live tool registry is;
* a candidate/target pair carries exactly one evidence state and the states
  are not interchangeable -- "AI reviewed" is not a correctness state;
* promotion needs target evidence, not interpretation; zero, partial or
  unbound case coverage is ``inconclusive``, never a pass;
* performance labels never collapse into a score: a local parser cannot
  authenticate a run, so ``official_score_computed`` is always false and the
  official aggregate is recorded separately;
* platform records are append-only, and an ``evaluating`` record is
  provisional -- its per-target values may be revised.

Nothing here reads credentials, tensors or hidden tests, and nothing here
executes Triton.
"""

from __future__ import annotations

import statistics

# --------------------------------------------------------------------------
# Capability: absent / configured_unavailable / callable
# --------------------------------------------------------------------------

ABSENT = "absent"
CONFIGURED_UNAVAILABLE = "configured_unavailable"
CALLABLE = "callable"
CAPABILITY_STATES = (ABSENT, CONFIGURED_UNAVAILABLE, CALLABLE)


def classify_capability(
    *,
    in_tool_registry: bool,
    config_present: bool = False,
    reachable: bool = False,
) -> str:
    """Classify a required service or tool without reading its credentials.

    ``reachable`` records a TCP connect, an HTTP handshake or a working token.
    It is accepted so the observation is not lost, but it can only ever yield
    ``configured_unavailable``: the skill's rule is that a tool counts as
    available only when the current agent can call it from its registry.
    """
    if in_tool_registry:
        return CALLABLE
    if config_present or reachable:
        return CONFIGURED_UNAVAILABLE
    return ABSENT


# --------------------------------------------------------------------------
# Evidence states
# --------------------------------------------------------------------------

EVIDENCE_STATES = {
    "prepared": "contract, baseline, target and experiment are recorded",
    "tool_unavailable": "config may exist, the required tool is not callable",
    "generated": "a source artifact was returned and saved; no claim follows",
    "locally_validated": "local syntax/contract/semantic tests passed",
    "target_validated": "exact source compiled and passed a non-empty "
    "correctness suite on the named target",
    "measured": "target timings plus baseline on the same case set, repeated",
    "target_reported": "a manifest claims target compile/correctness, not "
    "directly verified",
    "performance_reported_complete": "repeated timings for the declared case "
    "set, run not authenticated",
    "performance_reported_subset": "timings for a diagnostic subset only",
    "partial_measurement": "timings for part of the case set only",
    "reported_only": "a normalized claim without a preserved raw result or "
    "invocation reference",
    "submission_candidate": "local/review gates passed; missing target "
    "evidence is explicit",
    "submitted": "the identified package was uploaded once, record pending",
    "officially_evaluated": "terminal platform record captured",
    "rejected": "a concrete contract/source/correctness/perf gate failed",
    "inconclusive": "required evidence is absent, ambiguous or too noisy",
}

# States that cannot carry target execution evidence.  Interpretation alone
# must never move a candidate out of these into a target state.
NO_TARGET_EVIDENCE_FROM = ("tool_unavailable", "generated", "reported_only")

TARGET_STATES = ("target_validated", "measured")

# Reachability of the state machine.  Guards below decide whether a transition
# is allowed, this table decides whether it is even representable.
ALLOWED = {
    "prepared": (
        "tool_unavailable",
        "generated",
        "submission_candidate",
        "rejected",
        "inconclusive",
    ),
    "tool_unavailable": ("prepared", "inconclusive", "rejected"),
    "generated": ("locally_validated", "rejected", "inconclusive"),
    "locally_validated": (
        "target_reported",
        "reported_only",
        "target_validated",
        "measured",
        "submission_candidate",
        "rejected",
        "inconclusive",
    ),
    "target_reported": (
        "target_validated",
        "performance_reported_complete",
        "performance_reported_subset",
        "reported_only",
        "rejected",
        "inconclusive",
    ),
    "reported_only": ("rejected", "inconclusive"),
    "performance_reported_complete": (
        "target_validated",
        "measured",
        "submission_candidate",
        "rejected",
        "inconclusive",
    ),
    "performance_reported_subset": (
        "submission_candidate",
        "rejected",
        "inconclusive",
    ),
    "partial_measurement": (
        "submission_candidate",
        "rejected",
        "inconclusive",
    ),
    "target_validated": (
        "measured",
        "submission_candidate",
        "rejected",
        "inconclusive",
    ),
    "measured": ("submission_candidate", "rejected", "inconclusive"),
    "submission_candidate": ("submitted", "rejected", "inconclusive"),
    "submitted": ("officially_evaluated", "rejected", "inconclusive"),
    "officially_evaluated": ("rejected", "inconclusive"),
    "rejected": ("prepared", "inconclusive"),
    "inconclusive": (
        "prepared",
        "generated",
        "locally_validated",
        "target_validated",
        "measured",
        "submission_candidate",
        "rejected",
    ),
}

MIN_SAMPLES = 5
DEFAULT_MAX_RELATIVE_NOISE = 0.10

# Never set true by this module: a local parser cannot authenticate a run.
OFFICIAL_SCORE_COMPUTED = False


class EvidenceError(ValueError):
    """An illegal state transition or a rejected evidence bundle."""


def _identity_gaps(evidence: dict) -> list[str]:
    target = evidence.get("target") or {}
    return [
        key
        for key in ("backend", "device", "compiler", "runtime")
        if not str(target.get(key, "")).strip()
    ]


def _provenance_gaps(evidence: dict) -> list[str]:
    provenance = evidence.get("provenance") or {}
    gaps = []
    if not str(provenance.get("kind", "")).strip():
        gaps.append("provenance.kind")
    if not str(provenance.get("invocation_id", "")).strip():
        gaps.append("provenance.invocation_id")
    if not str(provenance.get("raw_result_sha256", "")).strip():
        gaps.append("provenance.raw_result_sha256")
    return gaps


def correctness_status(evidence: dict) -> tuple[str, str]:
    """Return ``(verdict, detail)`` for the correctness block.

    ``verdict`` is ``pass``, ``rejected`` or ``inconclusive``.  Zero, partial
    or unbound coverage is inconclusive; only a failed case rejects.
    """
    correctness = evidence.get("correctness")
    if not isinstance(correctness, dict):
        return "inconclusive", "no correctness record"
    cases = correctness.get("cases") or []
    if not cases:
        return "inconclusive", "correctness suite is empty"
    contract = correctness.get("case_contract") or {}
    required = contract.get("required_case_ids") or []
    if not required:
        return "inconclusive", "no declared required correctness case set"
    ids = [str(case.get("case_id", "")) for case in cases]
    if any(not case_id for case_id in ids):
        return "inconclusive", "a correctness case has no case_id"
    if len(set(ids)) != len(ids):
        return "inconclusive", "duplicate correctness case_ids"
    missing = sorted(set(map(str, required)) - set(ids))
    extra = sorted(set(ids) - set(map(str, required)))
    if missing or extra:
        return (
            "inconclusive",
            f"case coverage is partial or unbound (missing={missing}, "
            f"extra={extra})",
        )
    unsigned = [
        case.get("case_id")
        for case in cases
        if not str(case.get("input_signature", "")).strip()
    ]
    if unsigned:
        return (
            "inconclusive",
            f"case(s) without an input signature: {unsigned}",
        )
    failed = [
        case.get("case_id") for case in cases if case.get("status") != "passed"
    ]
    if failed:
        return "rejected", f"correctness failed on {failed}"
    total = correctness.get("total_cases")
    if total is not None and total != len(cases):
        return (
            "inconclusive",
            f"total_cases={total} does not match {len(cases)} listed cases",
        )
    return "pass", f"{len(cases)} correctness case(s) passed"


def benchmark_status(
    evidence: dict, max_relative_noise: float = DEFAULT_MAX_RELATIVE_NOISE
) -> tuple[str, dict]:
    """Return ``(coverage, detail)`` for the benchmark block.

    ``coverage`` is ``complete``, ``subset``, ``incomplete`` or ``rejected``.
    Repeated samples are preserved, never trimmed, so the robust median is a
    summary rather than permission to delete an outlier.
    """
    correctness_verdict, _ = correctness_status(evidence)
    if correctness_verdict != "pass":
        return "rejected", {"reason": "benchmark without passing correctness"}
    benchmark = evidence.get("benchmark")
    if not isinstance(benchmark, dict):
        return "incomplete", {"reason": "no benchmark record"}
    if not benchmark.get("warmup_runs"):
        return "incomplete", {"reason": "no warm-up recorded"}
    cases = benchmark.get("cases") or []
    if not cases:
        return "incomplete", {"reason": "no benchmark cases"}
    contract = benchmark.get("case_contract") or {}
    required = [str(item) for item in contract.get("required_case_ids") or []]
    if not required:
        return "incomplete", {"reason": "benchmark case set not declared"}
    if not str(contract.get("score_rule", "")).strip():
        return "incomplete", {"reason": "no declared score rule"}
    passed = {
        str(case.get("case_id"))
        for case in (evidence.get("correctness") or {}).get("cases") or []
        if case.get("status") == "passed"
    }
    signatures = {
        str(case.get("case_id")): str(case.get("input_signature"))
        for case in (evidence.get("correctness") or {}).get("cases") or []
    }
    per_case, ratios = {}, []
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if case_id not in passed:
            return "rejected", {
                "reason": f"benchmark case {case_id!r} did not pass correctness"
            }
        if str(case.get("input_signature", "")) != signatures.get(case_id):
            return "rejected", {
                "reason": f"benchmark case {case_id!r} signature differs from "
                "the correctness case"
            }
        baseline = case.get("baseline_ms") or []
        candidate = case.get("candidate_ms") or []
        if len(baseline) < MIN_SAMPLES or len(candidate) < MIN_SAMPLES:
            return "incomplete", {
                "reason": f"case {case_id!r} has {len(baseline)} baseline / "
                f"{len(candidate)} candidate samples, {MIN_SAMPLES} required",
            }
        noise = max(
            _relative_noise(values) for values in (baseline, candidate)
        )
        if noise > max_relative_noise:
            return "rejected", {
                "reason": f"case {case_id!r} relative noise {noise:.3f} "
                f"exceeds {max_relative_noise}"
            }
        median_baseline = statistics.median(baseline)
        median_candidate = statistics.median(candidate)
        ratio = median_baseline / median_candidate
        ratios.append(ratio)
        per_case[case_id] = {
            "baseline_ms": median_baseline,
            "candidate_ms": median_candidate,
            "speedup": ratio,
            "relative_noise": noise,
        }
    detail = {"per_case": per_case, "geometric_mean": _geometric_mean(ratios)}
    missing = sorted(set(required) - set(per_case))
    extra = sorted(set(per_case) - set(required))
    if missing or extra:
        detail["missing_case_ids"] = missing
        detail["extra_case_ids"] = extra
        return "subset", detail
    return "complete", detail


def _relative_noise(values: list[float]) -> float:
    median = statistics.median(values)
    if median == 0:
        return float("inf")
    deviations = [abs(value - median) / abs(median) for value in values]
    return max(deviations)


def _geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    product = 1.0
    for value in values:
        product *= value
    return product ** (1.0 / len(values))


# --------------------------------------------------------------------------
# Promotion
# --------------------------------------------------------------------------


def promote(
    current: str,
    target: str,
    evidence: dict | None = None,
    *,
    max_relative_noise: float = DEFAULT_MAX_RELATIVE_NOISE,
) -> dict:
    """Return the transition receipt, or raise :class:`EvidenceError`.

    Messaging convention: a guard that fails because evidence is missing or
    partial raises ``inconclusive: ...``; a guard that fails on a concrete
    failure raises ``rejected: ...``.
    """
    for state in (current, target):
        if state not in EVIDENCE_STATES:
            raise EvidenceError(f"unknown evidence state {state!r}")
    if current == target:
        raise EvidenceError(f"{current} -> {target} is not a transition")
    if target in TARGET_STATES and current in NO_TARGET_EVIDENCE_FROM:
        raise EvidenceError(
            f"{current} cannot become {target}: no target execution evidence "
            "can exist in that state"
        )
    if target not in TARGET_STATES and target not in ALLOWED[current]:
        raise EvidenceError(f"{current} -> {target} is not a legal transition")

    if target in TARGET_STATES:
        if not isinstance(evidence, dict) or not evidence:
            raise EvidenceError(
                f"inconclusive: {target} requires a target-evidence bundle"
            )
        gaps = _identity_gaps(evidence) + _provenance_gaps(evidence)
        if gaps:
            raise EvidenceError(
                f"inconclusive: target evidence is missing {sorted(gaps)}"
            )
        compile_block = evidence.get("compile") or {}
        if compile_block.get("success") is not True:
            raise EvidenceError(
                "rejected: target compilation is not reported successful"
            )
        verdict, detail = correctness_status(evidence)
        if verdict == "rejected":
            raise EvidenceError(f"rejected: {detail}")
        if verdict != "pass":
            raise EvidenceError(f"inconclusive: {detail}")
        receipt = {"state": target, "correctness": detail}

    if target == "measured":
        coverage, detail = benchmark_status(evidence, max_relative_noise)
        if coverage == "rejected":
            raise EvidenceError(f"rejected: {detail.get('reason')}")
        if coverage != "complete":
            raise EvidenceError(
                f"inconclusive: benchmark coverage is {coverage} "
                f"({detail.get('reason') or detail.get('missing_case_ids')})"
            )
        receipt["benchmark"] = detail
        receipt["official_score_computed"] = OFFICIAL_SCORE_COMPUTED
        receipt["note"] = (
            "consistency of the supplied records only; the workflow, not this "
            "parser, may promote on a directly inspected trusted result"
        )
        return receipt

    if target == "target_validated":
        receipt["note"] = (
            "compilation and correctness are reported, not authenticated"
        )
        return receipt
    return {"state": target, "note": f"{current} -> {target}"}


def performance_label(
    *,
    coverage: str,
    raw_result_preserved: bool,
    invocation_id: str | None,
) -> dict:
    """Label a performance claim without ever inferring an official score."""
    if not raw_result_preserved or not str(invocation_id or "").strip():
        label = "reported_only"
        note = (
            "no preserved raw result or invocation reference: this is a claim"
        )
    elif coverage == "complete":
        label = "performance_reported_complete"
        note = "declared case set covered; the run is still unauthenticated"
    elif coverage == "subset":
        label = "performance_reported_subset"
        note = "diagnostic subset only: never a task-level speedup"
    else:
        label = "performance_reported_incomplete"
        note = "timing coverage is incomplete"
    return {
        "label": label,
        "coverage": coverage,
        "official_score_computed": OFFICIAL_SCORE_COMPUTED,
        "note": note,
    }


# --------------------------------------------------------------------------
# Platform submission lifecycle
# --------------------------------------------------------------------------

LIFECYCLE = (
    "prepared",
    "submitted",
    "record_confirmed",
    "evaluating",
    "completed",
)
STATUS_TO_LIFECYCLE = {
    "uploaded": "submitted",
    "submitted": "submitted",
    "confirmed": "record_confirmed",
    "record_confirmed": "record_confirmed",
    "evaluating": "evaluating",
    "completed": "completed",
}
TERMINAL_TARGET_STATUSES = ("pass", "failed", "fail", "error", "rejected")


def submission_lifecycle(record: dict) -> dict:
    """Derive the resumable platform lifecycle of one official record.

    Uploading is not correctness or performance evidence, and an
    ``evaluating`` record is provisional: per-target values may be revised and
    the aggregate must stay null until the record is terminal.
    """
    status = str(record.get("status", "")).lower()
    stage = STATUS_TO_LIFECYCLE.get(status)
    if stage is None:
        raise EvidenceError(
            f"unknown platform record status {record.get('status')!r}"
        )
    targets = record.get("targets") or {}
    pending = sorted(
        name
        for name, value in targets.items()
        if (value or {}).get("status") not in TERMINAL_TARGET_STATUSES
    )
    errors = []
    if stage != "completed" and record.get("aggregate_speedup") is not None:
        errors.append(
            "aggregate_speedup must stay null while the record is provisional"
        )
    if stage == "completed" and pending:
        errors.append(
            f"completed record still has non-terminal targets: {pending}"
        )
    if stage != "completed" and not pending and targets:
        errors.append(
            "all targets are terminal but the record is not completed"
        )
    return {
        "stage": stage,
        "provisional": stage != "completed",
        "pending_targets": pending,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# Run record discipline
# --------------------------------------------------------------------------

RUN_RECORD_FIELDS = (
    "schema_version",
    "run_id",
    "parent_run_id",
    "operator",
    "repository_revision",
    "mode",
    "service_state",
    "tool_name",
    "request_path_or_hash",
    "service_job_id",
    "started_at",
    "finished_at",
    "contract_hash",
    "baseline_source_hash",
    "candidate_source_hash",
    "target_backend",
    "device",
    "compiler_version",
    "runtime_version",
    "structural_hypothesis",
    "structural_diff",
    "local_test_command",
    "local_correctness",
    "blind_reviewer_agent_id",
    "blind_review_raw_hash",
    "reconciliation_reviewer_agent_id",
    "reconciliation_review_raw_hash",
    "review_input_source_hashes",
    "review_provenance",
    "target_compile",
    "target_correctness",
    "target_case_signatures",
    "benchmark_method",
    "benchmark_warmups",
    "benchmark_raw_samples",
    "benchmark_case_set_id",
    "benchmark_required_case_ids",
    "target_raw_result_hash",
    "target_provider_invocation_id",
    "per_case_median_speedups",
    "aggregate_metric",
    "measurement_confidence",
    "evidence_state",
    "decision",
    "reason",
)

# The subset a record cannot be read without.  The rest are conditional on the
# mode: a local experiment has no target block, a packaged candidate has.
RUN_RECORD_REQUIRED = (
    "schema_version",
    "run_id",
    "operator",
    "mode",
    "service_state",
    "started_at",
    "contract_hash",
    "candidate_source_hash",
    "local_correctness",
    "evidence_state",
    "decision",
    "reason",
)
RUN_RECORD_MODES = ("generate", "optimize", "specialize", "autotune")
RUN_RECORD_DECISIONS = (
    "reject",
    "keep_candidate",
    "package_experiment",
    "promote_best",
)
MEASUREMENT_CONFIDENCE = (
    "stable",
    "noisy",
    "single_observation",
    "unknown",
)


def run_record_errors(record: dict) -> dict:
    """Check one append-only attempt record against the field discipline."""
    errors, missing = [], []
    for key in RUN_RECORD_REQUIRED:
        if record.get(key) in (None, "", [], {}):
            missing.append(key)
    unknown = sorted(set(record) - set(RUN_RECORD_FIELDS))
    if unknown:
        errors.append(f"unknown field(s): {unknown}")
    run_id = record.get("run_id")
    if run_id and record.get("parent_run_id") == run_id:
        errors.append(
            "a repair or retry gets a new run_id linked by parent_run_id; a "
            "record must not point at itself"
        )
    if record.get("mode") not in RUN_RECORD_MODES:
        errors.append(f"mode must be one of {list(RUN_RECORD_MODES)}")
    if record.get("service_state") not in CAPABILITY_STATES:
        errors.append(
            f"service_state must be one of {list(CAPABILITY_STATES)}"
        )
    if record.get("decision") not in RUN_RECORD_DECISIONS:
        errors.append(f"decision must be one of {list(RUN_RECORD_DECISIONS)}")
    if record.get("evidence_state") not in EVIDENCE_STATES:
        errors.append(
            f"evidence_state must be one of {sorted(EVIDENCE_STATES)}"
        )
    if record.get("service_state") == CALLABLE and not record.get("tool_name"):
        errors.append("a callable service must name the exact registered tool")
    if record.get("service_state") in (
        ABSENT,
        CONFIGURED_UNAVAILABLE,
    ) and record.get("tool_name"):
        errors.append(
            "tool_name recorded while the service is not callable; a config "
            "file is not evidence that a tool ran"
        )
    confidence = record.get("measurement_confidence")
    if confidence is not None and confidence not in MEASUREMENT_CONFIDENCE:
        errors.append(
            f"measurement_confidence must be one of "
            f"{list(MEASUREMENT_CONFIDENCE)}"
        )
    return {
        "errors": errors,
        "missing_required": missing,
        "missing_optional": [
            key for key in RUN_RECORD_FIELDS if key not in record
        ],
        "fields": len(RUN_RECORD_FIELDS),
    }


# --------------------------------------------------------------------------
# Two-stage independent review receipts
# --------------------------------------------------------------------------

REVIEW_PROVENANCE = ("platform-recorded", "unverified_claim")


def review_receipt_errors(stage_a: dict, stage_b: dict) -> dict:
    """Check that two-stage review receipts are consistent, not that they ran.

    The gate can check shape, source hashes, that the two recorded agent IDs
    differ, and that every Stage A finding has a disposition.  It cannot prove
    the reviewers were spawned, what they read or that their reasoning is
    correct, so an unverified receipt is labelled instead of trusted.

    ``errors`` are contradictions in the recorded evidence; ``gaps`` are
    things the receipts do not establish, which stay ``unavailable`` rather
    than becoming a failure.
    """
    errors, gaps, notes = [], [], []
    staged = ((stage_a, "A"), (stage_b, "B"))
    for receipt, marker in staged:
        if receipt.get("stage") != marker:
            errors.append(f"receipt for stage {marker} declares another stage")
        if receipt.get("review_provenance") not in REVIEW_PROVENANCE:
            errors.append(
                f"stage {marker} review_provenance must be one of "
                f"{list(REVIEW_PROVENANCE)}"
            )
        for key in ("candidate_source_hash", "baseline_source_hash"):
            if not str(receipt.get(key, "")).strip():
                gaps.append(f"stage {marker} is missing {key}")
    agent_a = str(stage_a.get("reviewer_agent_id", "")).strip()
    agent_b = str(stage_b.get("reviewer_agent_id", "")).strip()
    if not agent_a or not agent_b:
        gaps.append(
            "reviewer agent IDs are missing: label the review an unverified "
            "claim instead"
        )
    elif agent_a == agent_b:
        errors.append("stage B reviewer must differ from the stage A reviewer")
    for receipt, marker in staged:
        if receipt.get("review_provenance") == "unverified_claim":
            notes.append(f"stage {marker} is an unverified review claim")
    for receipt in (stage_b,):
        if receipt.get("candidate_source_hash") != stage_a.get(
            "candidate_source_hash"
        ):
            errors.append("the two stages reviewed different candidate bytes")
        if receipt.get("baseline_source_hash") != stage_a.get(
            "baseline_source_hash"
        ):
            errors.append("the two stages reviewed different baseline bytes")
    coverage = stage_a.get("coverage") or {}
    for target, entry in coverage.items():
        if not str((entry or {}).get("trace", "")).strip():
            errors.append(
                f"stage A coverage for {target} needs a source-referenced "
                "trace, not a status label"
            )
    if not coverage:
        gaps.append("stage A declares no per-target coverage")
    findings = stage_a.get("findings") or []
    dispositions = stage_b.get("dispositions") or {}
    finding_ids = [str(item.get("id", "")) for item in findings]
    for finding_id in finding_ids:
        if not finding_id:
            errors.append("a stage A finding has no id")
        elif finding_id not in dispositions:
            errors.append(
                f"stage A finding {finding_id} has no stage B disposition"
            )
    for finding_id in sorted(set(dispositions) - set(finding_ids)):
        errors.append(
            f"stage B dispositions unknown stage A finding {finding_id}"
        )
    if not findings:
        notes.append(
            "an empty finding list means only that this review found nothing; "
            "it cannot clear an unverified target risk"
        )
    return {
        "errors": errors,
        "gaps": gaps,
        "notes": notes,
        "verified": not errors and not gaps,
        "authenticated": False,
    }


# --------------------------------------------------------------------------
# Report: what we are entitled to claim, per target
# --------------------------------------------------------------------------


def derive_report(
    *,
    task_id: str,
    adaptation_id: str,
    profile_targets: list[str],
    release: dict,
    records: list[dict],
    package_sha256: str,
    routes: list[dict] | None = None,
    local_gates: dict | None = None,
    reviews: list[dict] | None = None,
) -> dict:
    """Derive the evidence report for one packaged adaptation.

    Every target starts at ``inconclusive`` and can only leave it through a
    terminal official record whose per-target source hash matches the packaged
    source.  A provisional (``evaluating``) record never raises a target above
    ``inconclusive``; it is quoted separately as a provisional observation.
    """
    capability = {}
    for route in routes or []:
        capability[route["name"]] = classify_capability(
            in_tool_registry=bool(route.get("in_tool_registry")),
            config_present=bool(route.get("config_present")),
            reachable=bool(route.get("reachable")),
        )
    target_sources = {
        name: str((block or {}).get("source_sha256", ""))
        for name, block in (release.get("targets") or {}).items()
    }
    lifecycle, provisional_records, superseded_records = [], [], []
    terminal_ids = {
        record.get("record_id")
        for record in records
        if record.get("status") == "completed"
    }
    for record in records:
        derived = submission_lifecycle(record)
        lifecycle.append(
            {
                "record_id": record.get("record_id"),
                "stage": derived["stage"],
                "provisional": derived["provisional"],
                "errors": derived["errors"],
            }
        )
        if not derived["provisional"]:
            continue
        observation = {
            "record_id": record.get("record_id"),
            "submitted_at": record.get("submitted_at"),
            "stage": derived["stage"],
            "values": {
                name: (value or {}).get("speedup")
                for name, value in (record.get("targets") or {}).items()
                if (value or {}).get("speedup") is not None
            },
        }
        # A later revision of the same submission supersedes the interim row.
        # The interim values are preserved either way, never overwritten.
        if observation["record_id"] in terminal_ids:
            observation["superseded_by"] = observation["record_id"]
            superseded_records.append(observation)
        else:
            provisional_records.append(observation)

    targets, best_per_target = {}, {}
    for name in profile_targets:
        verdict = {
            "state": "inconclusive",
            "source_sha256": target_sources.get(name),
            "basis": "no target execution evidence for this source",
            "performance": performance_label(
                coverage="incomplete",
                raw_result_preserved=False,
                invocation_id=None,
            ),
        }
        for record in records:
            block = (record.get("targets") or {}).get(name) or {}
            if record.get("status") != "completed":
                continue
            if block.get("status") != "pass":
                continue
            if block.get("source_sha256") != target_sources.get(name):
                continue
            terminal = all(
                ((record.get("targets") or {}).get(other) or {}).get("status")
                in TERMINAL_TARGET_STATUSES
                for other in profile_targets
            )
            coverage = "complete" if terminal else "subset"
            candidate = {
                "state": "target_validated",
                "source_sha256": block.get("source_sha256"),
                "basis": "terminal official record "
                f"{record.get('record_id')} ({record.get('submitted_at')})",
                "performance": {
                    **performance_label(
                        coverage=coverage,
                        raw_result_preserved=True,
                        invocation_id=record.get("record_id"),
                    ),
                    "official_speedup": block.get("speedup"),
                },
                "record_id": record.get("record_id"),
            }
            previous = best_per_target.get(name)
            if previous is None or (
                block.get("speedup") is not None
                and block["speedup"]
                > (previous.get("official_speedup") or float("-inf"))
            ):
                best_per_target[name] = {
                    "record_id": record.get("record_id"),
                    "official_speedup": block.get("speedup"),
                }
                verdict = candidate
        targets[name] = verdict

    completed = [
        record for record in records if record.get("status") == "completed"
    ]
    valid_aggregates = [
        (record["aggregate_speedup"], record.get("record_id"))
        for record in completed
        if record.get("aggregate_speedup") is not None
    ]
    best_aggregate = (
        max(valid_aggregates) if valid_aggregates else (None, None)
    )

    full_correctness = (release.get("full_correctness") or {}).get("status")
    gates_summary = local_gates or {}
    missing = [
        name
        for name, block in targets.items()
        if block["state"] == "inconclusive"
    ]
    if gates_summary.get("ready_for_pr") and missing:
        package_state = "submission_candidate"
    elif gates_summary.get("ready_for_pr"):
        package_state = "target_validated"
    else:
        package_state = "prepared"

    receipts = reviews or []
    if len(receipts) >= 2:
        review = review_receipt_errors(receipts[0], receipts[1])
    else:
        review = {
            "errors": [],
            "gaps": [
                f"only {len(receipts)} review receipt(s) recorded; the "
                "protocol needs two stages with distinct reviewers"
            ],
            "notes": [],
            "verified": False,
            "authenticated": False,
        }

    notes = []
    if provisional_records:
        notes.append(
            "provisional records are quoted, never aggregated: the platform "
            "revises interim per-target values"
        )
    if superseded_records:
        notes.append(
            f"{len(superseded_records)} interim observation(s) were superseded "
            "by the terminal revision of the same submission; the interim "
            "values are preserved, not deleted"
        )
    if full_correctness:
        notes.append(f"full_correctness status: {full_correctness}")
    if package_sha256 and not records:
        notes.append("the package has no official record yet")

    return {
        "task_id": task_id,
        "adaptation_id": adaptation_id,
        "package_sha256": package_sha256,
        "capability": capability,
        "package_state": package_state,
        "targets": targets,
        "best_per_target": best_per_target,
        "official": {
            "records": len(records),
            "completed_records": len(completed),
            "best_aggregate_speedup": best_aggregate[0],
            "best_aggregate_record_id": best_aggregate[1],
            "provisional_records": provisional_records,
            "superseded_records": superseded_records,
            "lifecycle": lifecycle,
            "official_score_computed": OFFICIAL_SCORE_COMPUTED,
        },
        "review": review,
        "local_gates": {
            "ready_for_pr": bool(gates_summary.get("ready_for_pr")),
            "failed": list(gates_summary.get("failed") or []),
            "not_run_here": list(gates_summary.get("not_run_here") or []),
        },
        "missing_target_evidence": missing,
        "notes": notes,
    }


def run_record_template(**values) -> dict:
    """A null-filled run record skeleton; never invents a value."""
    record = {field: None for field in RUN_RECORD_FIELDS}
    unknown = sorted(set(values) - set(RUN_RECORD_FIELDS))
    if unknown:
        raise EvidenceError(f"unknown run-record field(s): {unknown}")
    record.update(values)
    return record
