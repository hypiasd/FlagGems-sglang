#!/usr/bin/env python3
"""Shared validation for two-stage independent kernel review receipts.

This validates receipt consistency and hash linkage only. It cannot authenticate
agent invocations or prove that a reviewer read or correctly reasoned about the
source; callers must retain raw platform outputs and invocation records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_finding(item: object, target_ids: set[str]) -> bool:
    if not isinstance(item, dict):
        return False
    return (
        isinstance(item.get("id"), str) and bool(item["id"].strip())
        and item.get("target_id") in target_ids
        and item.get("severity") in {"blocker", "residual", "note"}
        and isinstance(item.get("location"), str) and bool(item["location"].strip())
        and isinstance(item.get("mechanism"), str) and bool(item["mechanism"].strip())
        and isinstance(item.get("activation"), str) and bool(item["activation"].strip())
        and item.get("confidence") in {"high", "medium", "low"}
        and isinstance(item.get("evidence"), list) and bool(item["evidence"])
    )


def _read_receipt(path: Path | None) -> tuple[dict | None, str | None, str | None]:
    if path is None:
        return None, None, "receipt path was not supplied"
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return None, None, f"could not read receipt: {exc}"
    if not isinstance(value, dict):
        return None, None, "receipt root must be a JSON object"
    return value, hashlib.sha256(raw).hexdigest(), None


def _static_findings(report: dict) -> list[dict]:
    result = []
    target_reports = report.get("targets", report.get("backends", {}))
    for target_id, target_report in target_reports.items():
        if not isinstance(target_report, dict):
            continue
        for finding in target_report.get("findings", []):
            if not isinstance(finding, dict):
                continue
            identity = {"target_id": target_id, **finding}
            result.append({
                "id": "static:" + canonical_json_sha256(identity),
                "target_id": target_id,
                "severity": finding.get("severity"),
                "kind": finding.get("kind"),
                "location": f"{target_id}:{finding.get('line', '?')}",
            })
    return result


def validate(
    blind_path: Path | None,
    reconciliation_path: Path | None,
    static_report: dict,
    required: bool,
    candidate_hashes: dict[str, str],
    baseline_hashes: dict[str, str],
    candidate_id: str,
    baseline_id: str,
) -> dict:
    """Validate the blind audit and independent reconciliation receipts."""
    if not required and blind_path is None and reconciliation_path is None:
        return {"required": False, "present": False, "passed": True,
                "residuals": [], "error": None}
    blind, blind_digest, blind_error = _read_receipt(blind_path)
    final, _, final_error = _read_receipt(reconciliation_path)
    if blind_error or final_error:
        return {"required": required, "present": bool(blind or final), "passed": False,
                "residuals": [], "error": blind_error or final_error}

    target_ids = set(candidate_hashes)
    blind_findings = blind.get("findings")
    coverage = blind.get("backend_coverage")
    valid_coverage = (
        isinstance(coverage, dict)
        and set(coverage) == target_ids
        and all(isinstance(item, dict)
                and item.get("status") == "complete"
                and isinstance(item.get("checks_run"), list)
                and bool(item["checks_run"])
                and all(isinstance(check, str) and check.strip()
                        for check in item["checks_run"])
                and isinstance(item.get("analysis_summary"), list)
                and bool(item["analysis_summary"])
                and all(isinstance(note, str) and note.strip()
                        for note in item["analysis_summary"])
                and isinstance(item.get("evidence"), list)
                and bool(item["evidence"])
                and all(isinstance(ref, str) and ref.strip()
                        for ref in item["evidence"])
                for item in coverage.values())
    )
    valid_blind_findings = (
        isinstance(blind_findings, list)
        and all(_valid_finding(item, target_ids) for item in blind_findings)
        and len({item["id"] for item in blind_findings}) == len(blind_findings)
    )
    blind_hashes_match = (
        blind.get("reviewed_source_sha256") == candidate_hashes
        and blind.get("reviewed_baseline_sha256") == baseline_hashes
    )
    blind_identity = blind.get("reviewer_agent_id")
    blind_valid = (
        blind.get("protocol_version") == 3
        and blind.get("review_mode") == "blind-independent"
        and blind.get("candidate") == candidate_id
        and blind.get("baseline") == baseline_id
        and isinstance(blind_identity, str) and bool(blind_identity.strip())
        and isinstance(blind.get("reviewer_name"), str) and bool(blind["reviewer_name"].strip())
        and blind.get("verdict") == "pass"
        and valid_coverage and valid_blind_findings and blind_hashes_match
        and not any(item.get("severity") == "blocker" for item in blind_findings or [])
    )

    static_findings = _static_findings(static_report)
    required_dispositions = {f"blind:{item['id']}" for item in (blind_findings or [])}
    required_dispositions.update(item["id"] for item in static_findings)
    dispositions = final.get("dispositions")
    valid_dispositions = (
        isinstance(dispositions, list)
        and all(isinstance(item, dict)
                and isinstance(item.get("finding_id"), str)
                and item.get("decision") in {"duplicate", "non_issue", "residual", "unresolved"}
                and isinstance(item.get("rationale"), str)
                and bool(item["rationale"].strip())
                for item in dispositions)
        and len({item["finding_id"] for item in dispositions}) == len(dispositions)
        and {item["finding_id"] for item in dispositions} == required_dispositions
    )
    additional = final.get("additional_findings")
    valid_additional = (
        isinstance(additional, list)
        and all(_valid_finding(item, target_ids)
                and item.get("disposition") in {"residual", "unresolved"}
                and isinstance(item.get("rationale"), str)
                and bool(item["rationale"].strip())
                for item in additional)
        and len({item["id"] for item in additional}) == len(additional)
    )
    final_hashes_match = (
        final.get("reviewed_source_sha256") == candidate_hashes
        and final.get("reviewed_baseline_sha256") == baseline_hashes
    )
    final_identity = final.get("reviewer_agent_id")
    final_valid = (
        final.get("protocol_version") == 3
        and final.get("review_mode") == "independent-reconciliation"
        and final.get("candidate") == candidate_id
        and final.get("baseline") == baseline_id
        and isinstance(final_identity, str) and bool(final_identity.strip())
        and final_identity != blind_identity
        and final.get("blind_receipt_sha256") == blind_digest
        and final.get("static_report_sha256") == canonical_json_sha256(static_report)
        and final.get("verdict") == "pass"
        and final_hashes_match and valid_dispositions and valid_additional
        and not any(item.get("decision") == "unresolved" for item in dispositions or [])
        and not any(item.get("severity") == "blocker"
                    or item.get("disposition") == "unresolved"
                    for item in additional or [])
    )
    residuals = [
        {"finding_id": item["finding_id"], "rationale": item["rationale"]}
        for item in dispositions or [] if item.get("decision") == "residual"
    ] + [
        {"finding_id": item["id"], "rationale": item["rationale"]}
        for item in additional or [] if item.get("disposition") == "residual"
    ]
    passed = blind_valid and final_valid
    return {
        "required": required,
        "present": bool(blind and final),
        "passed": bool(passed),
        "protocol_version": 3,
        "blind_receipt_sha256": blind_digest,
        "static_report_sha256": canonical_json_sha256(static_report),
        "blind_review_valid": blind_valid,
        "reconciliation_valid": final_valid,
        "reviewer_ids_distinct": bool(blind_identity and final_identity and blind_identity != final_identity),
        "source_hashes_match": blind_hashes_match and final_hashes_match,
        "static_finding_count": len(static_findings),
        "blind_finding_count": len(blind_findings) if isinstance(blind_findings, list) else None,
        "residuals": residuals,
        "error": None if passed else "two-stage review receipts are incomplete, mismatched, blocked, or unresolved",
    }
