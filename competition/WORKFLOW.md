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

The target scope is every task FlagOS currently marks open for the logged-in
competition session. Refresh the exact open-task cards in Chrome as batches
change. Capture each card's exact task id, title, batch, availability, and
official detail URL in an inventory JSON, then import it:

```sh
python3 competition/flagos_s2_workflow.py import-inventory /path/to/chrome-task-inventory.json
python3 competition/flagos_s2_workflow.py list-tasks
```

`list-tasks` merges the Chrome inventory, every local task adapter, and the
latest checkpoint for each task. It prioritizes unsafe-to-repeat submission
checkpoints first, then missing task contracts, then ready task iterations. A
task absent from a complete Chrome inventory is not assumed open. A locked task
waits for its batch; an open task without a valid profile is queued for
onboarding and cannot generate or upload a candidate.

The inventory must carry `schema_version: 1`, `competition: "flagos-s2"`, the
official `source_url`, an ISO UTC `observed_at`, and
`coverage: {"scope":"all-currently-open-tasks", "open_task_count":N,
"complete":true, "captured_from_chrome":true}`. Each task row has exact
`task_id` (`taskNN`), `task_name`, `batch_id`, `availability` (`open` or
`locked`), and may include `detail_url`. A complete snapshot is accepted only
when its open row count equals the count visibly reported by FlagOS. Keep
team identity, cookies, credentials, and quota out of the inventory; those are
refreshed in Chrome immediately before a submission.

The agent applies the same candidate loop independently to every open task
whose adapter is valid and whose goal is unmet. When the competition opens a
new batch, refresh/import the Chrome inventory and continue with the newly open
tasks. This is campaign scheduling across tasks; it does not flatten their
different semantic contracts, chip targets, scoring formulas, or release
hurdles into one generic kernel.

## Task adapters

- [Task 60: `clamp_position`](task60/kernelgen/README.md), profile
  [`task60.json`](workflow/tasks/task60.json).
- [Task 78: `concat_and_cast_mha_k`](task78/kernelgen/README.md), profile
  [`task78.json`](workflow/tasks/task78.json).
- Future task: copy [`template.json`](workflow/tasks/template.json) to
  `workflow/tasks/taskNN.json`. Fill it from that task's official FlagOS page
  and add the task-local contract validator and append-only result ledger. The
  unresolved `FILL`/`TODO` markers are rejected by profile validation; a task
  does not inherit Task 60 or Task 78 semantics by default.
  Missing or unknown contract fields block candidate generation; another task's
  case IDs, chip routing, aggregate formula, batch, or release hurdle must not
  be inferred.

## Start and resume

From the project root:

```sh
python3 competition/flagos_s2_workflow.py list-tasks
python3 competition/flagos_s2_workflow.py validate-profile task60
python3 competition/flagos_s2_workflow.py validate-profile task78
python3 competition/flagos_s2_workflow.py new-run task78 auto-20260924-01 \
  --reason "continue the current FlagOS S2 optimization"
python3 competition/flagos_s2_workflow.py resume task78 auto-20260924-01
```

Each run snapshots the profile and repository revision under the ignored
`competition/.autopilot/runs/<task>/<run-id>/`. `events.jsonl` is append-only;
`state.json` is a convenience checkpoint. Event writes are flushed before the
materialized state is updated. Result-ledger append is idempotent for the same
confirmed record, package hash, and result, so an interruption during report
rendering can resume without duplicating a row. Every stage records the
evidence needed to resume. A repair or another iteration receives a new run
ID and `--parent-run-id`; an existing candidate or historical ZIP is never edited.

Before generating any source, check the active KernelGen tool registry. A local
MCP config or a successful server handshake is not enough. If the requested
operation is absent, checkpoint `tool_unavailable` and stop source generation.
Resume that run only after the exact operation appears in the live registry.

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
2. Check the live KernelGen operation. Snapshot baseline bytes and hashes;
   register a falsifiable structural hypothesis and evidence-based expected and
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
