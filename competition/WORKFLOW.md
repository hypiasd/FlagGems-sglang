# FlagOS S2 task-neutral KernelGen workflow

Every FlagOS Season 2 task uses this same shared lifecycle, including tasks
other than Task 60 and Task 78. A task adapter supplies its operator contract,
source set, target matrix, baseline selector, local gate, score hurdle, package
contents, and result-ledger format. The shared layer owns discovery inventory,
campaign scheduling, run identity, durable checkpoints, evidence requirements,
Chrome preflight, single-upload handling, resume behavior, quota/rate-limit
checks, and result capture. Each task has its own candidate runs and result
ledger; one task's contract, score, or terminal result is never reused for
another task.

## Campaign-wide discovery and queue

The inventory covers every task FlagOS currently marks open for the logged-in
competition session, but visibility is not iteration authorization. The user
chooses which task ids may be iterated. Refresh the exact open-task cards in
Chrome as batches change. Capture each card's exact task id, title, batch,
availability, and official detail URL in an inventory JSON, then import it:

```sh
python3 competition/flagos_s2_workflow.py import-inventory /path/to/chrome-task-inventory.json
python3 competition/flagos_s2_workflow.py list-tasks
python3 competition/flagos_s2_workflow.py list-tasks --tasks task78 task91
```

`list-tasks` merges the Chrome inventory, every local task adapter, and the
latest checkpoint for each task. It sorts by actionable priority: recover an
in-flight submission, record a terminal result, resume work already past
generation, start a runnable task, check KernelGen, onboard an incomplete task,
resolve a task-local evidence gap, then wait on unavailable shared capability.
This keeps each task independent: a missing contract for one operator does not
hold up another task whose adapter is ready. Its campaign summary exposes
`open_task_overview_order`; each row separates profile `readiness` from runtime
`execution_readiness`. `queue_order` and `next_task` stay empty unless task ids
are explicitly supplied with `--tasks`. Only those selected ids enter the
iteration queue; all other tasks remain read-only overview rows.
A task absent from a complete Chrome inventory is not assumed open. A locked
task waits for its batch; an open task without a valid profile is marked for
task-specific onboarding. It enters the iteration queue only after the user
selects it.

The inventory must carry `schema_version: 1`, `competition: "flagos-s2"`, the
official `source_url`, an ISO UTC `observed_at`, and
`coverage: {"scope":"all-currently-open-tasks", "open_task_count":N,
"complete":true, "captured_from_chrome":true}`. Each task row has exact
`task_id` (`taskNN`), `task_name`, `batch_id`, `availability` (`open` or
`locked`), and may include `detail_url`. A complete snapshot is accepted only
when its open row count equals the count visibly reported by FlagOS. Keep
team identity, cookies, credentials, and quota out of the inventory; those are
refreshed in Chrome immediately before a submission.

The agent applies the same candidate loop independently to each user-selected
task whose adapter is valid and whose goal is unmet. If the user has not named
tasks, the workflow only reports the open-task overview. When the competition
opens a new batch, refresh/import the Chrome inventory; newly open tasks still
need explicit user selection before iteration. The shared lifecycle does not
flatten task-specific semantic contracts, chip targets, scoring formulas, or
release hurdles into one generic kernel.

## Task adapters

- [Task 60: `clamp_position`](task60/kernelgen/README.md), profile
  [`task60.json`](workflow/tasks/task60.json).
- [Task 78: `concat_and_cast_mha_k`](task78/kernelgen/README.md), profile
  [`task78.json`](workflow/tasks/task78.json).
- Every other open task: copy [`template.json`](workflow/tasks/template.json) to
  `workflow/tasks/taskNN.json`. Fill it from that task's official FlagOS page
  and add the exact official reference source, task-local contract validator,
  and append-only result ledger. The template starts in
  `generate_from_official_reference` mode because a newly onboarded task has
  no candidate source to optimize yet. The reference file is hash-pinned; the
  generated candidate paths may be absent until KernelGen returns them. Later
  runs can switch to `optimize_existing_sources` after an evaluated source is
  selected. The unresolved `FILL`/`TODO` markers are rejected by profile
  validation; a task does not inherit Task 60 or Task 78 semantics by default.
  Missing or unknown contract fields block candidate generation; another task's
  case IDs, chip routing, aggregate formula, batch, or release hurdle must not
  be inferred. Candidate source basenames must match ZIP root members exactly;
  a multi-file `generic-v1` adapter must map every target to its source member.

The optional ignored runtime evidence file
`competition/.autopilot/task-contract-evidence.json` records what Chrome showed
for each current task and lists the missing adapter evidence. `list-tasks`
surfaces this alongside profile readiness. A captured signature or reference
formula is only onboarding evidence: it does not make a task adapter valid or
authorize generation. For example, the current Batch 6 snapshot contains 17
open tasks (Task76–Task92); Task78 has a valid adapter, while the other task
pages have partial contract captures and still need their exact case matrix,
target routing, local semantic gate, source/baseline selection, and frozen
release hurdle. These gaps block those tasks independently; the overview keeps
them visible without selecting them for iteration.

## Start and resume

From the project root:

```sh
python3 competition/flagos_s2_workflow.py list-tasks
python3 competition/flagos_s2_workflow.py list-tasks --tasks task78
python3 competition/flagos_s2_workflow.py validate-profile taskNN
python3 competition/flagos_s2_workflow.py new-run taskNN auto-YYYYMMDD-01 \
  --reason "start the next FlagOS S2 task iteration"
python3 competition/flagos_s2_workflow.py resume taskNN RUN_ID
```

Replace `taskNN` with an open task that the user explicitly selected with
`list-tasks --tasks`; `campaign.next_task` and `queue_order` only select among
those named task ids.
Use that task's own run ID, profile, evidence, and ledger. The queue only
contains user-selected tasks; the open-task overview is informational and does
not authorize iteration.

Each run snapshots the profile and repository revision under the ignored
`competition/.autopilot/runs/<task>/<run-id>/`. `events.jsonl` is append-only;
`state.json` is a convenience checkpoint. Event writes are flushed before the
materialized state is updated. Result-ledger append is idempotent for the same
confirmed record, package hash, and result, so an interruption during report
rendering can resume without duplicating a row. Every stage records the
evidence needed to resume. A repair or another iteration receives a new run
ID and `--parent-run-id`; an existing candidate or historical ZIP is never edited.

Before generating any source, check the active KernelGen tool registry for
`generate_kernel`, `optimize_kernel`, and `specialize_kernel`. The queue accepts
only a complete active-registry snapshot no more than five minutes old; a local
MCP config or successful server handshake is not enough. If any required
operation is absent, checkpoint `tool_unavailable` and stop source generation.
Resume that task only after the exact operation appears in a fresh live
registry. This blocks generation for selected tasks that need KernelGen, while
already-generated runs and read-only Chrome capture may continue independently.
Unselected tasks are not checkpointed or modified. `list-tasks` reports the
latest registry observation but does not replace the live check immediately
before generation.

For an initial task generation, `prepared.baseline_hashes` pins the official
reference source hash for each target and identifies `baseline_kind` as
`official_reference`. For an optimization run, the same field pins the selected
candidate source hash per target and uses `baseline_kind: "candidate_source"`.
The `generated` checkpoint binds a SHA-256 to every exact ZIP root member. At
`package_ready`, the workflow opens the ZIP, checks its exact member list and
hash, and rejects any archive whose source bytes differ from KernelGen's
recorded output.

## Shared lifecycle

```text
planned → prepared → generated → locally_validated → reviewed
  → target_validated → measured → package_ready → upload_armed
  → submitted → record_confirmed → evaluating → completed
  → target_validated → measured → promoted
```

When no trusted target runner exists, the current user-authorized path allows a
candidate that passes its task-specific forecast, local contract/semantic gates,
independent reviews, and deterministic checks to enter `package_ready` for an
official FlagOS evaluation. That package remains an experiment until the
terminal FlagOS record provides all target outcomes. Only an 8/8 record can
validate all targets or yield a task-level aggregate. A failed/incomplete target
stops promotion and creates a repair run if the evidence supports one.

The complete candidate sequence is:

1. Read the exact task page in Chrome. Freeze public signature, reference
   semantics, official case IDs, targets, batch, package rules, official score
   field, and the per-task goal. Refresh each task's latest valid result and
   per-target source champions from its own ledger.
2. Check the live KernelGen operation. For an existing candidate, snapshot its
   per-target source bytes and hashes. For a task's first kernel, snapshot the
   exact official reference source hash as the generation seed. Register a
   falsifiable structural hypothesis and evidence-based expected and
   conservative score deltas before generation.
3. Generate an isolated candidate with KernelGen. Preserve request, raw
   response, invocation ID, returned sources, and hashes.
4. Run task-local syntax, contract, and semantic validation. Then run a fresh
   blind source review, a deterministic risk scan, and a second distinct review
   that reconciles the scan. Any edit starts a new run and review cycle.
5. Use a trusted complete target runner when available. Otherwise, the current
   explicit FlagOS S2 authorization permits official online evaluation to be
   the target oracle after the gates above; do not call the candidate
   target-validated until the record completes.
6. Freeze an exact package and SHA-256. Immediately before each upload, use
   Chrome to verify the selected task and batch, current team, visible daily
   quota, latest record, duplicate archive history, and package hash. Submit
   once, confirm the record identity, and wait on that record.
7. Append terminal per-target results and the official aggregate to the task's
   own ledger. Compare against the frozen task goal. Promote and stop when it is
   reached; otherwise, if visible quota remains, create a child run and repeat.

## Upload recovery and stop conditions

FlagOS's submission page requires a ZIP no larger than 10 MB containing UTF-8
`.py` files. Include the task's generic `[Kernel Name].py`; a chip override uses
`[Kernel Name]_[chip-id].py`. The currently documented chip suffixes are
`_iluvatar`, `_metax`, `_enflame`, `_hygon`, `_kunlunxin`, and `_ascend`.
International chips A/B use the generic file unless that task's official page
specifies otherwise. For each task, include overrides only for chips listed on
that task's own official page, and keep the exact package member list in its
adapter.

The batch overview and task submission page can expose different quota values
(for example, an overview count alongside `0/0` on an unbound task page). Before
upload, reconcile the selected task's submission-page quota with the current
team overview. Any unresolved discrepancy is a named preflight blocker; do not
infer that one screen's quota applies to another task.

`upload_armed` is a durable one-shot checkpoint. If the task is interrupted
there or later, first inspect the submission records in Chrome. Never click
Submit again or replay an archive based only on a timeout. If a fresh record
check proves no record exists and the click was never sent, record
`upload_aborted`; only then may the exact package be reconsidered after fresh
preflight. A confirmed upload is polled by record ID, never by resubmitting.

The shared gate requires the current visible quota to be positive and checked
within five minutes, respects FlagOS's strictly-more-than-120-second submission
interval, and blocks an archive hash already represented in any task run or
official ledger. When account/team, selected task/batch, quota, upload identity,
record status, or target result cannot be established, preserve the checkpoint
and stop at that named gap. All browser operations use Chrome.

Unknown task profiles, absent MCP tools, failed review/contract/hurdle gates,
and partial official results are explicit per-task stop states. System or user
interruptions can still happen; resume reads the latest checkpoint and continues
without repeating completed work or a submission. A pause on one task does not
silently mark other tasks complete; the campaign queue keeps each task's
readiness, blocker, and checkpoint visible.
