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
   complete snapshot with `import-inventory`, then run `list-tasks` for the
   overview. Iterate only task ids the user explicitly named. If no task was
   named, do not select a task from the overview or begin iteration. Pass the
   user's exact selection to `list-tasks --tasks taskNN ...`; an open card does
   not by itself authorize iteration. A task absent from the snapshot is not
   assumed to be open.
2. Follow `campaign.queue_order`, which contains only explicitly selected
   tasks. Resolve any `upload_armed` or later submission checkpoint in Chrome
   before starting new work; record already-completed terminal results before
   creating another candidate. Never retry a click or archive blindly.
3. Onboard the open task selected by the queue when its adapter is missing.
   Read that task's official page and finish its own profile, semantic
   validator, package contract, score hurdle, and result ledger without
   borrowing another task's values. If evidence remains incomplete, leave that
   task marked `needs_task_adapter` and return to the selected-task queue; do
   not hold up another selected task with a ready adapter.
4. Before each source-generation run, confirm the exact required KernelGen
   operation in the live tool registry. If absent, record `tool_unavailable`
   for the selected task and pause its generation. Do not create checkpoints or
   candidates for unselected tasks. A local config file or server handshake is
   not enough.
5. For the selected ready open task, freeze per-target baseline inputs and
   forecast. An established adapter hashes its selected candidate source; a
   first-time adapter hashes the exact official reference source used to seed
   `generate_kernel`. Generate one immutable attempt and record a hash for each
   exact ZIP root member. Run its contract and semantic checks, two distinct
   fresh source reviews, and deterministic scan. Before `package_ready`, prove
   the ZIP members and bytes match those recorded KernelGen outputs. Repairs
   receive a new run ID. Return to `list-tasks` after each checkpoint transition
   so another selected task can proceed.
6. Prefer trusted complete target preflight. The current user authorization
   permits a fully gated FlagOS S2 package to use the official FlagOS
   evaluation in Chrome as the target oracle when no trusted runner exists.
   Keep it unvalidated until the terminal record returns every target.
7. Continue each task while its frozen goal is unmet and the current visible
   daily quota remains. Before each upload, recheck selected task and batch,
   current team, quota, duplicate history, latest record, and exact package
   hash in Chrome. Upload once, confirm its record, and poll that record.
8. Append results to that task's ledger. Promote only when all targets pass
   and its own profile hurdle/goal is reached. Otherwise create a child attempt
   if quota remains, then return to the selected-task queue. Finish when every
   selected task is promoted or explicitly blocked, or the shared quota is
   exhausted. A missing campaign-wide capability pauses source generation for
   selected tasks. System or user interruptions can still happen; resume from
   the selected queue and each task's own checkpoint.

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
