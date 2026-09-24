---
name: flagos-s2-autopilot
description: Run or resume the shared KernelGen optimization workflow for any FlagOS Season 2 task, with task-specific contracts and checkpoints.
---

# FlagOS S2 automatic iteration

Use this skill for Task 60, Task 78, and other FlagOS S2 tasks in this
repository. Read [`competition/WORKFLOW.md`](../../../competition/WORKFLOW.md)
and the selected task's profile and adapter before generating source.

## Run protocol

1. Run `python3 competition/flagos_s2_workflow.py list-tasks`; resolve or add
   the exact task profile. A missing task-specific contract, case set, score
   rule, baseline, or package layout blocks generation.
2. Read the last run checkpoint for that task. Resume it without repeating
   work. After an interruption at `upload_armed` or later, inspect records in
   Chrome first; never retry a click or archive blindly.
3. Confirm the exact KernelGen operation in this task's live tool registry.
   If absent, record `tool_unavailable` and stop source generation. A local
   config file or server handshake is not enough.
4. Freeze task-specific per-target baseline hashes and forecast before the
   generation call. Generate one immutable attempt. Run its contract and
   semantic checks, two distinct fresh source reviews, and deterministic scan.
   Repairs receive a new run ID.
5. Prefer trusted complete target preflight. The current user authorization
   permits a fully gated FlagOS S2 package to use the official FlagOS
   evaluation in Chrome as the target oracle when no trusted runner exists.
   Keep it unvalidated until the terminal record returns every target.
6. Continue per task while its frozen goal is unmet and the current visible
   daily quota remains. Before each upload, recheck selected task and batch,
   current team, quota, duplicate history, latest record, and exact package
   hash in Chrome. Upload once, confirm its record, and poll that record.
7. Append results to that task's ledger. Promote only when all targets pass
   and its profile hurdle/goal is reached. Otherwise begin a child attempt if
   quota remains. Stop on a failed hard gate, exhausted quota, ambiguous
   account/record state, or missing runtime capability.

The checkpoint manager is evidence validation and resume support. Chrome,
KernelGen, reviewers, local validators, and FlagOS remain distinct oracles;
never synthesize their evidence or claim one proves another.
