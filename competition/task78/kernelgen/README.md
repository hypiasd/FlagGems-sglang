# Task 78 KernelGen adapter

The reusable KernelGen workflow v5 is defined in
`.agents/skills/kernelgen-flagos/references/reliability-gates.md`. This file
adds only Task 78's operator contract, backend matrix, submission policy, and
observed target constraints. The general workflow's evidence requirements
remain authoritative.

The first reviewer capability holdout results—including the v22 Hygon miss—are
recorded in [`reviewer-holdout-experiments.md`](reviewer-holdout-experiments.md).

## Workspace and submission boundary

Candidate artifacts live under:

```text
competition/task78/kernelgen/candidates/<run-id>/
```

Keep one immutable run directory per generation/repair attempt, with the
prompt, raw KernelGen response, returned source, hashes, test reports, both
reviewers' raw outputs/receipts, and raw target evidence. Never overwrite the root submission files or a
numbered historical version during exploration. The ignored candidate area is
not a submission package.

The final archive contains exactly seven `concat_and_cast_mha_k*.py` files at
ZIP root. The generic implementation is evaluated twice—International A and
International B—so the evaluation matrix contains eight target results.
Package preparation, user submission, official evaluation, and promotion of a
new best are separate events. Do not submit to Arc on the user's behalf.

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

1. **Freeze evidence.** Record the baseline version and source hashes, contract
   and test revision, each chip's latest/best score, score validity, and exact
   failures from the preceding Arc run. Keep the known-good source immutable.
2. **Plan one coherent batch.** For every implementation file, state the
   structural change, observed bottleneck, expected score effect, and the
   shapes/failure conditions that could disprove it. Batch meaningful changes
   so a scarce Arc submission tests a real hypothesis, not a one-line tweak.
3. **Check runtime capability.** Confirm the needed KernelGen operation is
   actually callable in the current agent tool registry. A project `.mcp.json`
   entry is configuration evidence only. If the tool is absent, preserve the
   diagnosis and ready-to-run request, but do not hand-write a candidate or
   call it KernelGen output.
4. **Generate in isolation.** Ask KernelGen for each target-specific source
   using the exact contract and structural objective. Save every response and
   hash. Do not let a response's `success` flag stand in for tests.
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
   `run_candidate_gate.py` with both receipts and the exact baseline directory.
   This gate can reject semantic/source risks; it cannot certify a vendor
   compiler, target runtime, or speedup.
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
10. **Arc and ledger.** After the user submits, append the official result for
   each of the eight evaluations, including failures and exact diagnostics.
   Update the per-chip best table only from valid completed results, retaining
   anomalous single-run values with an instability note. A single chip failure
   means the submission has no valid all-chip aggregate.

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
  --source-dir competition/task78/kernelgen/candidates/<run-id> \
  --all --autotune-sweep
# First: save the output of a fresh Stage A blind sub-agent to blind-review.raw.txt/json.
# Only then run the scanner and give its report to a different Stage B sub-agent.
python3 competition/task78/kernelgen/review_candidate.py \
  competition/task78/kernelgen/candidates/<run-id> \
  --json competition/task78/kernelgen/candidates/<run-id>/static-review.json
# Save the Stage B raw response and normalized receipt as reconciliation-review.raw.txt/json.
python3 competition/task78/kernelgen/run_candidate_gate.py \
  competition/task78/kernelgen/candidates/<run-id> \
  --baseline-source-dir competition/task78 \
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
  competition/task78/kernelgen/candidates/<run-id> \
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
