"""Platform failures, recorded so the same bytes never fail the same way twice.

A failed submission is a paid-for fact: it costs a slot out of the daily quota
and a long feedback round trip.  Letting it live only in a chat transcript means
the next round re-derives it from scratch -- which already happened here: Task
112's Kunlunxin failure (`TritonXPUUnrollControl` / `uni_sram`) and Task 111's
`_PLANS` safety rejection were both hit again on Task 111 because nothing
local remembered them.

Layout, one JSON object per line at ``competition/<task>/platform-failures.jsonl``::

    {"task": "task111", "target": "*", "kind": "code-safety",
     "submitted": "2026-10-02T07:04+08:00", "members": {"<file>": "<sha256>"},
     "message": "Code safety validation failed: ...", "evidence": "..."}

``kind`` is free text but the useful values are ``code-safety`` (static
rejection, retryable by editing), ``compile`` (a target's compiler refused the
kernel) and ``runtime``.  ``targets`` may be a list; ``"*"`` means every target.
"""

from __future__ import annotations

import json
from pathlib import Path

from competition.experiments.store import ROOT, now


def path_for(task: str) -> Path:
    return ROOT / "competition" / task / "platform-failures.jsonl"


def record(task: str, entry: dict) -> dict:
    """Append one failure.  ``members`` maps member name to its sha256."""
    record_ = {
        "task": task,
        "submitted": entry.get("submitted") or now(),
        "kind": entry.get("kind", "unknown"),
        "targets": entry.get("targets", ["*"]),
        "members": dict(entry.get("members") or {}),
        "message": entry.get("message", ""),
        "evidence": entry.get("evidence", ""),
    }
    target = path_for(task)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(record_, ensure_ascii=False, sort_keys=True) + "\n"
        )
    return record_


def load(task: str) -> list[dict]:
    target = path_for(task)
    if not target.is_file():
        return []
    entries = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def exact_repeat(
    task: str, routing: dict[str, str], hashes: dict[str, str]
) -> list[str]:
    """Blocking errors for shipping a target bytes that already failed on it.

    ``routing`` maps a target to the member file it will run; ``hashes`` maps a
    member to its sha256.  A record blocks only when a target it failed on is
    *still routed* to the same bytes -- which is what makes a dedicated file
    unblock a target, and what stops a resubmission that changes nothing.  A
    target that now routes elsewhere is free to be retried.
    """
    if not routing or not hashes:
        return []
    blocked = []
    for entry in load(task):
        recorded = entry.get("members") or {}
        if not recorded:
            continue
        for target in entry.get("targets") or ["*"]:
            if target == "*":
                continue
            member = routing.get(target)
            if member is None:
                continue
            if recorded.get(member) and recorded[member] == hashes.get(member):
                blocked.append(
                    f"{target} is still routed to {member} at the same bytes that "
                    f"failed ({entry.get('kind')}, {entry.get('submitted')}): "
                    f"{entry.get('message', '')[:200]}"
                )
    return blocked
