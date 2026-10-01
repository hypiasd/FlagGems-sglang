"""Render the upstream PR description for a competition bundle.

Follows `.github/PULL_REQUEST_TEMPLATE.md` plus the competition-flavored
requirements from CONTRIBUTING section 9: state that it is a competition
submission, name the op, the tier, the target hardware and any calling
convention constraints.  Performance numbers are copied from the submission
system, never invented.
"""

from __future__ import annotations

TARGET_LABELS = {
    "iluvatar": ("天数智芯 Tianshu", "_iluvatar"),
    "metax": ("沐曦 MetaX", "_metax"),
    "hygon": ("海光 Hygon", "_hygon"),
    "kunlunxin": ("昆仑芯 Kunlunxin", "_kunlunxin"),
    "ascend": ("华为昇腾 Ascend", "_ascend"),
    "intl_a": ("国际通用芯片 A (generic tier)", None),
    "intl_b": ("国际通用芯片 B (generic tier)", None),
}


def _tiers(bundle: dict) -> str:
    rows = [
        f"- `{entry['path']}` ({entry['tier']})" for entry in bundle["files"]
    ]
    return "\n".join(rows)


def _hardware(bundle: dict) -> str:
    vendors = []
    for entry in bundle["files"]:
        suffix = entry.get("suffix")
        if suffix is None:
            continue
        label = TARGET_LABELS.get(suffix, (suffix, None))[0]
        if label not in vendors:
            vendors.append(label)
    generic = any(entry["tier"] == "generic" for entry in bundle["files"])
    if not vendors:
        return "generic tier only" if generic else "none"
    text = ", ".join(vendors)
    if generic:
        text += "; the generic tier covers the remaining competition backends"
    return text


def _accuracy(bundle: dict) -> str:
    official = bundle.get("official", {})
    if not official.get("available"):
        return (
            "> Accuracy results are collected centrally by the competition harness.\n"
            f"> No official observation is bound to this bundle yet: {official.get('reason', 'unknown')}\n"
        )
    record = official["record"]
    sha = bundle["provenance"].get("platform_package_sha256")
    lines = [
        "Reference implementation: official task reference, evaluated by "
        "the competition harness.",
        f"Platform record: {record.get('submitted_at')} "
        f"({record.get('status')}), package SHA-256 `{sha}`.",
        "",
        "| Target | Status | Speedup |",
        "|---|---|---|",
    ]
    for target, item in (record.get("targets") or {}).items():
        label = TARGET_LABELS.get(target, (target, None))[0]
        speed = item.get("speedup")
        lines.append(
            f"| {label} | {item.get('status')} | {speed if speed is not None else '-'} |"
        )
    lines.append("")
    lines.append(
        f"Passed targets: **{record.get('pass_count')}/{len(record.get('targets') or {})}**; platform aggregate: "
        f"**{record.get('aggregate_speedup') if record.get('aggregate_speedup') is not None else 'pending'}**."
    )
    if official.get("revisions"):
        lines.append(
            f"(Record revised {official['revisions']} time(s) while the platform finalised it.)"
        )
    return "\n".join(lines) + "\n"


def render(bundle: dict, extra_constraints: list[str] | None = None) -> str:
    prov = bundle["provenance"]
    op = bundle["op"]
    constraints = list(extra_constraints or [])
    constraints.append(
        'Every module exports `__all__ = ["%s"]`; the dispatcher discovers tiers by walking the '
        "`ops/` packages, so no `ops/__init__.py` re-export is used." % op
    )
    for entry in bundle["files"]:
        awarded_note = entry.get("note")
        if awarded_note:
            constraints.append(awarded_note)
    renames = bundle.get("renames") or {}
    if renames:
        pairs = ", ".join(
            f"`{old}` -> `{new}`" for old, new in sorted(renames.items())
        )
        constraints.append(
            f"Identifier renames applied to satisfy flake8 (E741 among others): {pairs}. "
            "Token-level only; strings and comments untouched."
        )
    reflowed = (bundle.get("formatting") or {}).get(
        "reflowed_text_lines"
    ) or {}
    if reflowed:
        where = ", ".join(
            f"{path} lines {lines}" for path, lines in sorted(reflowed.items())
        )
        constraints.append(
            f"Over-long comment/module-docstring lines wrapped to fit flake8's 120-column limit: {where}."
        )
    body = f"""### PR Category

Operator

### Type of Change

New Feature

### Description

This is a **FlagOS x SGLang competition, Track 1** submission for `{op}`.

- Implementation files:
{_tiers(bundle)}
- Tier(s): {', '.join(sorted({entry['tier'] for entry in bundle['files']}))}.
- Target hardware: {_hardware(bundle)}.
- Calling-convention and dtype constraints the harness needs to know:
{chr(10).join('  - ' + item for item in constraints)}

Provenance: adaptation `{bundle['adaptation_id']}` of experiment snapshot bound to submission package
SHA-256 `{prov.get('platform_package_sha256')}`, uploaded to the competition platform at
`{prov.get('submitted_at')}`{f" (submission id {prov['submission_id']})" if prov.get('submission_id') else ""}.
The platform package's own evidence: {prov.get('submission_evidence') or 'not recorded'}.

Per the competition contribution rules in `docs/CONTRIBUTING.md` section 9, this PR ships the operator
sources only. No `tests/test_{op}.py`, no `benchmark/test_{op}.py`, no `benchmark/attri_util.py` shapes,
no `docs/tasks/{op}.md` and no `conf/operators.yaml` entry: the maintainers validate against a held-out
harness. The sources are the awarded ones, re-homed into the dispatcher tree with only the Apache 2.0
header, `__all__`, and formatter changes applied; the implementation AST is unchanged.

### Issue

FlagOS x SGLang competition, Track 1, task `{bundle['task_id']}`: `{op}`.

### Accuracy Tests

{_accuracy(bundle)}
### Speed Tests and Profiling

Speedup numbers are produced by the submission system; the table above is copied verbatim from the
platform's own record for this package rather than measured locally.

### Progress

- [x] Op file lives in the correct dispatcher tier folder.
- [x] Apache 2.0 header + `__all__` present.
- [ ] `import flaggems_sglang; flaggems_sglang.all_registered_ops()` shows the op locally.
- [ ] `pre-commit run --all-files` passes.
- [x] PR description tags this as a competition submission and names the target hardware.
"""
    return body
