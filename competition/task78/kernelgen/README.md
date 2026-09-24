# Task 78 KernelGen adapter

The reusable KernelGen workflow v6 is defined in
`.agents/skills/kernelgen-flagos/references/reliability-gates.md`. This file
adds only Task 78's operator contract, backend matrix, submission policy, and
observed target constraints. The general workflow's evidence requirements
remain authoritative.

The first reviewer capability holdout results—including the v22 Hygon miss—are
recorded in [`reviewer-holdout-experiments.md`](reviewer-holdout-experiments.md).

## End-to-end workflow and run states

Treat one candidate run as an immutable record. Repairs get a new run ID and point to the parent attempt. The seven files under `source/` are the implementation set; the shared generic file is evaluated separately on International A and B, for eight target results total. The [optimization manifest template](./optimization-manifest.example.json) binds the forecast, each backend hypothesis, and exact source hashes.

| State | Required evidence | Next action |
| --- | --- | --- |
| `tool_unavailable` | The required operation is not callable in the active task. | Stop generation; retain only read-only diagnosis and the experiment plan. |
| `prepared` | Contract, per-chip baseline snapshots, source method, and pre-registered forecast. | Continue only if the forecast gate passes and the required operation is callable. |
| `generated` | Raw KernelGen response, returned source, target, invocation/job ID when available, and hashes. | Inspect source, then run local checks. |
| `locally_validated` | Syntax, contract, CPU semantic checks, independent source reviews, and deterministic candidate gate pass. | Run the trusted preflight for every target. |
| `target_validated` | Exact candidate compiled and passed the full declared correctness/configuration matrix on that target. | Measure the complete target workload using the same method as the baseline. |
| `measured` | Repeated raw timings for the full required case set, exact source hashes, and score calculation. | Apply the release hurdle; package only if it passes. |
| `arc_candidate` | Exact package hash, all required gates, and a recorded decision to prepare official evaluation. | Verify Task 78 Batch 6, team, quota, and duplicate history before one upload. |
| `arc_submitted` / `arc_completed` | Confirmed official record, then its terminal per-target results. | Append the result and recalculate per-chip champions. |
| `rejected` / `inconclusive` | A named gate failed, or its required evidence is missing or mismatched. | Preserve the run; repair under a new run ID or stop. |

A source review, CPU model, HTTP probe, config file, compile-only check, or partial benchmark cannot move a target to `target_validated` or `measured`. The run is ready only when every required target has the evidence needed for that next state.

### Start or resume

Run these checks from the repository root before creating a candidate:

```sh
python3 competition/task78/kernelgen/task78_results.py verify
python3 competition/task78/kernelgen/task78_results.py summary
```

Then use [`kernelgen-mcp-setup.md`](../../../.agents/skills/kernelgen-flagos/kernelgen-mcp-setup.md) for the active client's setup and verify the required operation in the current task's live tool registry. For Codex, the checkout's `.mcp.json` is not the server registration source. If the operation is absent, stop before creating candidate source.

After the tool is callable, create a unique run from the append-only result ledger and copy the forecast template:

```sh
python3 competition/task78/kernelgen/prepare_mixed_candidate.py <run-id>
cp competition/task78/kernelgen/optimization-manifest.example.json \
  competition/task78/kernelgen/candidates/<run-id>/optimization-manifest.json
```

`prepare_mixed_candidate.py` snapshots the best-observed source for each chip into `baseline/` and seeds `source/` with those same bytes. This is a baseline, not generated output. The shared generic baseline is selected using both International A and B observations. Keep `baseline-selections.json` unchanged.

Before the first source edit or KernelGen call, fill the eight target forecasts in `optimization-manifest.json` with evidence-backed expected and lower-bound deltas, then run:

```sh
python3 competition/task78/kernelgen/submission_hurdle.py \
  competition/task78/kernelgen/candidates/<run-id>/optimization-manifest.json \
  --json competition/task78/kernelgen/candidates/<run-id>/hurdle-report.json
```

The forecast must clear the larger of `1.50x` and 5% above the current champion composite; its lower-bound mean must preserve that composite, and no target lower bound may be below -5%. A failed or unsupported forecast ends the run before generation. Record `source_method` accurately; non-KernelGen authorship requires explicit user authorization for that run.

After generation, bind each backend hypothesis to the actual `source/` and `baseline/` SHA-256 values. Run local checks, Stage A review, deterministic scan, Stage B review, and `run_candidate_gate.py` in that order. Any source change starts a new review and gate cycle under a new attempt ID.

A trusted target runner must compile and execute the exact wrapper, every declared autotune configuration, every official and required boundary correctness case, live device-limit checks, and the full benchmark set. If no such runner is callable, record `inconclusive`; a diagnostic Arc upload is a separate experiment and requires an explicit user choice. Do not package or describe it as a normal performance release.

Only after target validation, full measurement, and the release hurdle pass, validate a ZIP containing exactly the seven reviewed files. Before submission, confirm the archive hash and Task 78 Batch 6 context, upload once, verify the new record, and wait on that record. After completion, append the official outcome to `results.jsonl`, then run `task78_results.py verify`, `render`, and `summary`. A later candidate takes its baselines from that updated ledger; do not overwrite root sources or historical packages.

## Workspace and submission boundary

Candidate artifacts live under:

```text
competition/task78/kernelgen/candidates/<run-id>/
```

Each run has immutable `baseline/` and editable `source/` subdirectories. The
`prepare_mixed_candidate.py` helper seeds them from independent per-chip
best-observed sources. For the shared generic file, choose the source by the
median of its International A and B results jointly; the two targets still
remain separate in reporting.

Keep one immutable run directory per generation/repair attempt, with the
prompt, raw KernelGen response, returned source, hashes, test reports, both
reviewers' raw outputs/receipts, and raw target evidence. Never overwrite the root submission files or a
numbered historical version during exploration. The ignored candidate area is
not a submission package.

The final archive contains exactly seven `concat_and_cast_mha_k*.py` files at
ZIP root. The generic implementation is evaluated twice—International A and
International B—so the evaluation matrix contains eight target results.
Package preparation, submission, official evaluation, and per-chip promotion
are separate events. Submission authorization is scope-limited: the current
standing authorization applies only to Task 78, Batch 6, in the currently
logged-in team and within its currently visible daily quota. It does not
authorize submissions to another task/batch or future competition. Before
each upload, re-check the selected task/batch, team, quota, archive identity,
and latest submission record in Chrome; click submit only once, then verify a
new record before waiting. Respect FlagOS's minimum interval (strictly more
than two minutes between submissions). If identity, upload status, quota, or
the result is ambiguous, stop and inspect records instead of retrying blindly.
Never replay the same archive to investigate a noisy score.

## Operator contract

Public entry:

```python
def concat_and_cast_mha_k(k, k_nope, k_rope):
    ...
```

For each token `t` and head `h`:

```text
k[t, h, :nope_dim] = k_nope[t, h, :]
k[t, h, nope_dim:] = k_rope[t, 0, :]
```

Preserve the reference's cat-then-cast dtype promotion, output shape/device,
arbitrary supported source strides, empty/tail behavior, and input immutability.
Use Triton/Triton-TLE for the kernel path; no native/PyTorch fallback,
`try/except` fallback, or device-based bypass.

For a new Task 78 version, every one of the seven implementation files must
have a documented structural optimization, not just launch-constant tuning.
The generic file's results must still be recorded separately for International
A and B. A shared code change is not proof of shared target compatibility.

## Workflow for each version

1. **Freeze evidence.** Verify `results.jsonl`, then record each chip's latest
   score, best-observed score/source, same-source variance flag, contract and
   test revision, and exact known failures. Source selection is per-chip, not
   by a single whole-package version. Preserve the full aggregate champion
   separately. Keep all source baselines immutable.
2. **Plan one coherent batch.** For every implementation file, state the
   structural change, observed bottleneck, expected score effect, and the
   shapes/failure conditions that could disprove it. Estimate each target's
   expected and lower-bound score delta, confidence, and evidence basis; never
   invent percentages to satisfy the gate. Task 78's scarce-submission hurdle
   is a forecasted score of **at least 1.50× and at least 5% above the
   per-chip champion composite** (whichever is higher), using the official
   eight-target arithmetic mean. The lower-bound aggregate must not regress,
   and no target lower bound may be worse than -5%. The 1.50× floor reflects
   the stated competition goal; the 5% floor prevents tiny releases even when
   the baseline changes. Use
   `submission_hurdle.py <optimization-manifest.json>` to calculate this.
   If official workload coverage or evidence is too weak to defend that
   forecast, keep working or preserve the quota; a tiny/uncertain gain is not
   a new submission version.
3. **Check runtime capability and source method.** Confirm the needed KernelGen
   operation is actually callable in the current agent tool registry. A
   project `.mcp.json` entry is configuration evidence only. If KernelGen is
   absent, do not silently switch methods: use a non-KernelGen authoring method
   only when the user has explicitly authorized that method for this run, and
   label every source/report `agent-authored` (never KernelGen output).
4. **Generate in isolation.** Use the authorized source method to create each
   target-specific source from its own archived per-chip champion. Save the
   request or hypothesis, source method, exact champion archive/member/hash,
   candidate source, and hashes. Do not let a generator's `success` flag stand
   in for tests.
5. **Local semantic checks.** Run syntax/public-entry checks and the isolated
   CPU semantic suite, including visible autotune configurations and
   boundary/stride/empty/wide-dimension cases. These may run before review;
   they are not target evidence.
6. **Two-agent source review.** Follow `subagent_review_prompt.md`: a fresh
   Stage A agent reviews exact candidate and baseline hashes without seeing
   the static report; only after its raw response is saved does a different
   Stage B agent receive the static report and reconcile every finding. Never
   synthesize agent outputs or IDs. The gate validates receipt consistency but
   cannot prove the agents ran or that their reasoning is right. A blocker or
   unresolved finding means repair and a complete new review cycle.
7. **Deterministic gate.** Run the compiler-risk scan after Stage A, then run
   `run_candidate_gate.py` with both receipts, the exact per-chip baseline,
   and `--require-structural-delta`. The manifest must bind every candidate
   and baseline source hash to its falsifiable hypothesis. The AST-shape check
   rejects unchanged and constants/comments/names-only rewrites; it cannot
   decide whether a structurally different kernel is faster. The source
   reviewers and performance hurdle remain necessary. This gate cannot certify
   a vendor compiler, target runtime, or speedup.
8. **Target evidence and decision.** A candidate is not submission-ready until
   a trusted live runner has preflighted every backend: invoked the real public
   entrypoint/config path, compiled and exercised all declared configurations,
   queried live device limits and checked the full case/config launch matrix,
   then differentially checked every required correctness case. Bind the
   report to exact source/baseline hashes, compiler/runtime, the complete
   adapter-declared case/config sets, all pass counts, queried device limits,
   timing method, invocation/job ID, and unmodified raw result. A self-authored
   manifest is a claim; a case subset is diagnostic only. If no trusted target
   runner covers a chip, the candidate stays inconclusive and is not
   submission-ready. Only an explicit user choice may turn it into a clearly
   labeled diagnostic Arc experiment; that does not establish correctness or
   performance and does not replace the best source.
9. **Package integrity.** Validate the final ZIP against the reviewed candidate
   directory with `validate_package.py`. It must contain exactly the seven
   root-level operator files, with byte-identical contents and recorded hashes.
   Any post-review source change or repackaged mismatch invalidates approval.
10. **Submit, wait, and ledger.** In Chrome, verify the exact Task 78 Batch 6
    submission context, team, remaining quota, package filename/hash, and that
    this exact archive has not already been submitted. Under the scoped current
    authorization, submit a gate-passing package once; record the official
    submission ID/time immediately. Wait on that record until terminal state,
    reopening the record rather than clicking submit again after UI ambiguity.
    Append all eight outcomes and failures to `results.jsonl`, then regenerate
    `results.md`. Keep a separate best-observed score/source per chip, the best
    valid 8/8 aggregate, and anomaly flags. Do not drop or silently overwrite a
    single-run maximum; a failed chip invalidates only the all-chip aggregate,
    not the other chips' passing observations.

## Current target constraint

Arc v24 failed on Enflame Case 1 because the launch requested `grid.x = 131072`
while that hardware path reported a limit of `65535`. Any future Enflame design
must justify and bound its grid mapping for the actual row count (for example,
a persistent/grid-stride scheme); do not generalize this Enflame limit to other
chips without evidence. The failure is a concrete target regression, not a
reason to add an unverified global rule.

## Local gate commands

Run from the repository root:

```sh
python3 -m py_compile competition/task78/kernelgen/candidates/<run-id>/*.py
python3 competition/task78/validate_cpu.py \
  --source-dir competition/task78/kernelgen/candidates/<run-id>/source \
  --all --autotune-sweep
# First: save the output of a fresh Stage A blind sub-agent to blind-review.raw.txt/json.
# Only then run the scanner and give its report to a different Stage B sub-agent.
python3 competition/task78/kernelgen/review_candidate.py \
  competition/task78/kernelgen/candidates/<run-id>/source \
  --json competition/task78/kernelgen/candidates/<run-id>/static-review.json
# Save the Stage B raw response and normalized receipt as reconciliation-review.raw.txt/json.
python3 competition/task78/kernelgen/run_candidate_gate.py \
  competition/task78/kernelgen/candidates/<run-id>/source \
  --baseline-source-dir competition/task78/kernelgen/candidates/<run-id>/baseline \
  --require-structural-delta \
  --structural-manifest competition/task78/kernelgen/candidates/<run-id>/optimization-manifest.json \
  --require-review \
  --blind-review-json competition/task78/kernelgen/candidates/<run-id>/blind-review.json \
  --review-json competition/task78/kernelgen/candidates/<run-id>/reconciliation-review.json \
  --compiler-review-json competition/task78/kernelgen/candidates/<run-id>/static-review.json \
  --json competition/task78/kernelgen/candidates/<run-id>/local-gate.json
# After the trusted target runner has returned raw results, run the schema-v2
# target evidence gate separately for all eight target profiles. The generic
# source uses distinct manifests for International A and International B.
python3 .agents/skills/kernelgen-flagos/scripts/kernelgen_gate.py \
  <saved-kernelgen-response.json> --phase target \
  --public-symbol concat_and_cast_mha_k \
  --target-evidence <target-evidence-v2.json> \
  --candidate-source <exact-candidate-source.py> \
  --baseline-source <exact-baseline-source.py> \
  --raw-target-result <unmodified-live-result.json>
python3 competition/task78/kernelgen/validate_package.py \
  competition/task78/kernelgen/candidates/<run-id>/source \
  competition/task78/kernelgen/candidates/<run-id>/submission.zip
```

Use `--autotune-sweep-full` for a release audit when the candidate has
autotuning. The CPU model checks source-level semantics and memory coverage;
it does not compile Triton or predict any chip's score.

## Score interpretation

Keep the official raw result for every submission and a separate best-valid
score for each chip. Record a run-level aggregate only when the competition
marks every required target correct and supplies a valid score. When repeated
measurements are available, retain raw timings, compare medians, and flag
high-variance results; never silently discard an outlier or replace the
historical best with a single noisy run.
