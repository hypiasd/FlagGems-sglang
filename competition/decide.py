"""Turn official results into the next decision.

Reading results is not the same as adjusting to them.  Three things were
missing on Task 103 and all three are mechanical:

1.  **The aggregate rule.**  The platform's aggregate is the arithmetic mean
    of the per-target speedups (verified against every eligible Task 103
    record, residual < 5e-3).  That makes the marginal value of one target
    ``delta / target_count``, so priority must be ordered by absolute gain,
    not by relative improvement.  A chip at 0.89x with 5x of headroom is
    worth far more than a chip at 18.46x with 3x of headroom.
2.  **An existence proof per target.**  The best value a target has ever
    produced, and which source bytes produced it, are already in the ledger
    and in the local adaptation snapshots; nothing was joining them.
3.  **A carry-forward decision.**  The next package should reuse, per chip,
    the bytes that produced that chip's best eligible value, unless there is
    a new hypothesis for that chip.  Instead the package was rebuilt by hand
    and regressions went unnoticed.

Everything here is derived from recorded observations.  Forecasts are
labelled as upper bounds, never as expectations.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from competition.adaptation import ledger
from competition.experiments.store import digest


def aggregate_model(records, tolerance=5e-3):
    """Check whether the official aggregate is the mean of the per-target values.

    The rule is verified rather than assumed: if a record disagrees, the
    marginal-gain arithmetic below must not be trusted.
    """
    checked, residuals = [], []
    for row in records:
        if row.get("status") != "completed" or row.get("excluded"):
            continue
        aggregate = row.get("aggregate_speedup")
        values = [
            item.get("speedup")
            for item in (row.get("targets") or {}).values()
            if isinstance(item.get("speedup"), (int, float))
            and not isinstance(item.get("speedup"), bool)
        ]
        if aggregate is None or not values:
            continue
        residual = abs(aggregate - statistics.fmean(values))
        residuals.append(residual)
        checked.append(
            {
                "record_id": row.get("record_id"),
                "official": aggregate,
                "target_mean": statistics.fmean(values),
                "targets": len(values),
                "residual": residual,
            }
        )
    matched = bool(checked) and all(
        item["residual"] <= tolerance for item in checked
    )
    return {
        "rule": (
            "arithmetic_mean_of_target_speedups" if matched else "unverified"
        ),
        "verified_on": [item["record_id"] for item in checked],
        "targets_per_record": checked[-1]["targets"] if checked else None,
        "max_residual": max(residuals) if residuals else None,
        "tolerance": tolerance,
        "marginal_rule": (
            "raising one target by delta adds delta / targets_per_record to the aggregate"
            if matched
            else "unknown: do not rank by predicted aggregate gain"
        ),
    }


def source_lookup(task, operator, local):
    """Map a packaged source digest back to the local file that produced it."""
    lookup = {}
    root = Path(local) / "adaptations"
    for directory in sorted(root.glob("adapt-*")):
        manifest = directory / "adaptation.json"
        if not manifest.is_file():
            continue
        try:
            meta = json.loads(manifest.read_text())
        except (ValueError, OSError):
            continue
        if meta.get("task_id") != task:
            continue
        source = directory / "source"
        if not source.is_dir():
            continue
        for path in sorted(source.glob("*.py")):
            source_digest = digest(path)
            lookup.setdefault(source_digest, []).append(
                {
                    "adaptation_id": directory.name,
                    "file": path.name,
                    "path": str(path),
                }
            )
    return lookup


def _provisional(records, target):
    """Best value seen in a record that is not terminal yet (evidence only)."""
    best = None
    for row in records:
        if row.get("status") == "completed":
            continue
        item = (row.get("targets") or {}).get(target) or {}
        value = item.get("speedup")
        if item.get("status") != "pass" or not isinstance(value, (int, float)):
            continue
        if best is None or value > best["speedup"]:
            best = {
                "speedup": value,
                "record_id": row.get("record_id"),
                "status": row.get("status"),
            }
    return best


def diagnose(task, local, records=None):
    """Per-target diagnosis, priority order and the carry-forward plan.

    The reference a target is measured against is the largest of the anchors
    that exist for it, because each anchor alone is blind:

    * ``best_eligible_ever`` sees only values this chip already reached, so a
      chip that has always been slow looks like it has no headroom;
    * ``provisional_observation`` is an existence proof but may be revised;
    * ``peer_median`` is an observed sibling value in the same package, so it
      exposes a chip that is far below what the same evaluation gives the
      others;
    * ``baseline_parity`` (1.0) is the floor below which the submission is
      slower than the reference implementation.

    Headroom is therefore an upper bound against an observed quantity, never a
    forecast of what a new implementation would score.
    """
    contract = ledger.profile(task)
    targets = list(contract["targets"])
    rows = records if records is not None else ledger.rows(task)
    collapsed = {row["record_id"]: row for row in rows}
    order = sorted(collapsed.values(), key=lambda row: row["submitted_at"])
    table = ledger.target_ledger(task, records=rows)["targets"]
    model = aggregate_model(order)
    count = len(targets)
    lookup = source_lookup(task, contract["operator"], local)

    observed = [
        block["latest"]["speedup"]
        for block in table.values()
        if block["latest"]
        and isinstance(block["latest"]["speedup"], (int, float))
    ]
    peer_median = statistics.median(observed) if observed else None
    current_mean = statistics.fmean(observed) if observed else None

    diagnosis, plan = {}, {}
    for target in targets:
        block = table[target]
        best_eligible = block["best_eligible"]
        best_observed = block["best_observed"]
        provisional = _provisional(order, target)
        value = block["latest"]["speedup"] if block["latest"] else None
        anchors = {"baseline_parity": 1.0}
        if best_observed:
            anchors["best_eligible_ever"] = best_observed["speedup"]
        if provisional:
            anchors["provisional_observation"] = provisional["speedup"]
        if peer_median is not None:
            anchors["peer_median"] = peer_median
        reference_kind = max(anchors, key=lambda name: anchors[name])
        reference = anchors[reference_kind]
        headroom = None
        if value is not None:
            headroom = max(reference - value, 0.0)
        gain = None
        if headroom is not None and model["rule"] != "unverified" and count:
            gain = headroom / count
        source = best_eligible or best_observed
        files = lookup.get((source or {}).get("source_sha256") or "", [])
        implementation = files[0]["file"] if files else None
        ratio = None
        if value is not None and reference > 0:
            ratio = (reference - value) / reference
        delta = block["delta_vs_best_eligible"]
        if value is None:
            action = "blocked_no_eligible_result"
        elif value < 1.0:
            action = "rework_below_baseline"
        elif ratio is not None and ratio > 0.10:
            action = "rework_behind_reference"
        elif peer_median is not None and value < peer_median / 2:
            action = "rework_under_served"
        elif delta and delta["verdict"] == "below_best_same_source":
            action = "hold_re_measure"
        else:
            action = "keep"
        diagnosis[target] = {
            "value": value,
            "action": action,
            "best_eligible": (
                best_eligible["speedup"] if best_eligible else None
            ),
            "best_observed": (
                best_observed["speedup"] if best_observed else None
            ),
            "provisional_observation": provisional,
            "anchors": anchors,
            "reference": reference,
            "reference_kind": reference_kind,
            "headroom": headroom,
            "headroom_ratio": ratio,
            "marginal_aggregate_gain_upper_bound": gain,
            "peer_median": peer_median,
            "below_baseline_parity": value is not None and value < 1.0,
            "source_versions": block["source_versions"],
            "delta_vs_best_eligible": delta,
            "implementation_file": implementation,
        }
        plan[target] = {
            "carry_forward_file": implementation,
            "carry_forward_sha256": (source or {}).get("source_sha256"),
            "from_adaptation": files[0]["adaptation_id"] if files else None,
            "evidence_record_id": (source or {}).get("record_id"),
            "value": source["speedup"] if source else None,
            "action": action,
        }

    def rank_key(block):
        gain = block["marginal_aggregate_gain_upper_bound"] or 0.0
        penalty = 1.0 if block["below_baseline_parity"] else 0.0
        value = block["value"] if block["value"] is not None else 0.0
        return (-penalty, -gain, value)

    ranked = sorted(diagnosis.items(), key=lambda item: rank_key(item[1]))
    rework = [
        name
        for name, block in ranked
        if block["action"] not in ("keep", "hold_re_measure")
    ]
    planned = sum(
        diagnosis[name]["marginal_aggregate_gain_upper_bound"] or 0.0
        for name in rework
    )
    return {
        "task_id": task,
        "operator": contract["operator"],
        "aggregate_model": model,
        "current": {
            "per_target_mean": current_mean,
            "latest_eligible_record": (
                order[-1]["record_id"] if order else None
            ),
            "observed_targets": len(observed),
            "peer_median": peer_median,
        },
        "targets": diagnosis,
        "priority": [name for name, _ in ranked],
        "rework_targets": rework,
        "rework_aggregate_gain_upper_bound": planned,
        "next_package": plan,
        "reuse_rule": (
            "carry_forward_file is the file whose bytes produced that target's "
            "best observation; a target may only be replaced when a concrete "
            "new hypothesis names the chip it changes"
        ),
        "limitation": (
            "marginal gains are upper bounds evaluated one target at a time; "
            "the platform scores one package for all targets at once, and "
            "provisional observations may be revised"
        ),
    }


def _names_chip(text, target, filename):
    """Cheap text check that a hypothesis names the chip it replaces."""
    haystack = str(text or "")
    return target in haystack or (filename or "") in haystack


def enforce(task, local, release, source_dir, records=None):
    """Block a package that drops a chip off its carry-forward source silently.

    The plan says which bytes produced each chip's best value.  Replacing them
    is allowed, but only when the release declares the chip and its hypothesis
    names it.  Without this the package is rebuilt by hand and a regression
    looks exactly like an improvement until the platform scores it.
    """
    source_dir = Path(source_dir)
    plan = diagnose(task, local, records=records)["next_package"]
    targets = release.get("targets") or {}
    declared = set(release.get("changed_targets") or [])
    errors, carried, deviated, undeclared, no_plan = [], [], [], [], []
    for target, entry in sorted(plan.items()):
        expected = entry.get("carry_forward_sha256")
        item = targets.get(target) or {}
        filename = item.get("source")
        if not expected or not filename:
            no_plan.append(target)
            continue
        path = source_dir / filename
        actual = digest(path) if path.is_file() else None
        if actual == expected:
            carried.append(target)
            continue
        deviated.append(target)
        marked = target in declared or item.get("replaces_carry_forward") is True
        if not marked:
            undeclared.append(target)
            errors.append(
                f"{target}: source {filename} is not the bytes that produced "
                f"its best value ({expected[:8]}), and the release does not "
                "declare the change; add the target to changed_targets or set "
                "replaces_carry_forward"
            )
            continue
        prose = " ".join(
            str(item.get(key) or "")
            for key in ("structural_change", "expected_mechanism")
        )
        if not _names_chip(prose, target, filename):
            errors.append(
                f"{target}: the change is declared but the hypothesis at "
                "structural_change/expected_mechanism does not name this chip"
            )
    missing_forecast = sorted(
        target
        for target in plan
        if not isinstance(
            (targets.get(target) or {}).get("expected_speedup"), (int, float)
        )
    )
    return {
        "passed": not errors,
        "errors": errors,
        "declared_changes": sorted(declared),
        "carried": carried,
        "deviated": deviated,
        "undeclared_deviations": undeclared,
        "no_carry_forward_plan": no_plan,
        "plan_basis": {
            target: {
                "value": entry.get("value"),
                "evidence_record_id": entry.get("evidence_record_id"),
                "file": entry.get("carry_forward_file"),
            }
            for target, entry in sorted(plan.items())
        },
        "forecast_missing": missing_forecast,
        "note": (
            "the chip-name check is a text check; it does not verify that the "
            "hypothesis is correct"
        ),
    }
