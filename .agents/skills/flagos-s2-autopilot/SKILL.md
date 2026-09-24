---
name: flagos-s2-autopilot
description: Run or resume the shared KernelGen optimization workflow for any FlagOS Season 2 task, with task-specific contracts and checkpoints.
---

# FlagOS S2 automatic iteration

Use this skill for Task 60, Task 78, and other FlagOS S2 tasks in this
repository. Read [`competition/WORKFLOW.md`](../../../competition/WORKFLOW.md)
and the selected task's profile and adapter before generating source.

## Run protocol

1. In Chrome, refresh the complete list of tasks currently marked open and
   capture the exact task ids, titles, batch ids, and detail links. Import the
   complete snapshot with `import-inventory`, then run `list-tasks`. The shared
   queue covers every open task in the snapshot, not only Task 60 or Task 78.
   A task absent from the snapshot is not assumed to be open.
2. Process each open task independently. If its adapter is missing, read that
   task's official page and onboard its profile, semantic validator, package
   contract, score hurdle, and result ledger without borrowing another task's
   values. A gap blocks that task only; continue with other open tasks whose
   profiles are complete.
3. Read the last run checkpoint for the selected task. Resume it without
   repeating work. Resolve `upload_armed` and later checkpoints before any new
   upload; inspect records in Chrome first and never retry a click or archive
   blindly.
4. Confirm the exact KernelGen operation in the live tool registry.
   If absent, record `tool_unavailable` and stop source generation. A local
   config file or server handshake is not enough.
5. Freeze task-specific per-target baseline hashes and forecast before the
   generation call. Generate one immutable attempt. Run its contract and
   semantic checks, two distinct fresh source reviews, and deterministic scan.
   Repairs receive a new run ID.
6. Prefer trusted complete target preflight. The current user authorization
   permits a fully gated FlagOS S2 package to use the official FlagOS
   evaluation in Chrome as the target oracle when no trusted runner exists.
   Keep it unvalidated until the terminal record returns every target.
7. Continue each task while its frozen goal is unmet and the current visible
   daily quota remains. Before each upload, recheck selected task and batch,
   current team, quota, duplicate history, latest record, and exact package
   hash in Chrome. Upload once, confirm its record, and poll that record.
8. Append results to that task's ledger. Promote only when all targets pass
   and its profile hurdle/goal is reached. Otherwise begin a child attempt if
   quota remains, then return to the campaign queue. Stop the campaign only
   when every currently open task is promoted or explicitly blocked, the
   shared quota is exhausted, or a campaign-wide capability such as KernelGen
   is unavailable. System or user interruptions can still happen; resume from
   the queue and each task's own checkpoint.

FlagOS currently requires a ZIP of at most 10 MB containing UTF-8 `.py` files.
The generic file is `[Kernel Name].py`; a chip-specific override is
`[Kernel Name]_[chip-id].py`. The current submission page documents `_iluvatar`,
`_metax`, `_enflame`, `_hygon`, `_kunlunxin`, and `_ascend`; international chips
A/B use the generic file unless that task's official page says otherwise.
Include only overrides for chips supported by that task, and freeze the exact
ZIP member list in its adapter. The batch overview and a task's submission page
may show different quota values. Reconcile them immediately before upload and
stop if the selected task's remaining quota cannot be established.

The checkpoint manager is evidence validation and resume support. Chrome,
KernelGen, reviewers, local validators, and FlagOS remain distinct oracles;
never synthesize their evidence or claim one proves another.
