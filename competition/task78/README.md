# Task 78: `concat_and_cast_mha_k`

Latest official run: v25, submitted on 2026-09-24, completed 8/8 at 1.12×.
The team's best valid all-chip aggregate remains v12 at 1.27×. Per-chip best
implementations are distributed across different submissions; see the
[append-only official result ledger](results.md), which records every Task 78
Batch 6 result and derives each chip's best-observed source independently.
The root files remain the historical v19 source set, **not** a synchronized
copy of the per-chip champions. Do not use them as the next candidate baseline.

The current best-observed per-chip source composite has an arithmetic mean of
about 1.38×. To respect the limited submission quota and the stated 1.50× goal,
the Task 78 release forecast must reach at least the larger of 1.50× and 105%
of that composite mean. Its conservative aggregate must not regress, and no
chip's lower-bound forecast may be worse than -5%. This is an eligibility
forecast, not a claim that unmeasured code will achieve that score.

The ledger also flags large score spreads for byte-identical per-chip sources.
These are retained as observations, not discarded as “bad runs”; a flagged
champion remains provisional until repeated official evaluation resolves the
variance.

v24 targeted the sub-1× chips from v19. It recovered two previously failing
targets, but Enflame failed because its launch grid exceeded the reported
hardware limit; several surviving chip scores regressed. The exact per-chip
results and failure are recorded in the v24 validation record.

While v24 was reviewed, the CPU validator gained twelve `wide-dim-*` cases
because single-tile row coverage silently dropped every column past its tile
cap: v19 covers them (201/201), the un-repaired draft failed 13 of 28
backend-case pairs in a direct probe, and the packaged revision passes. The
suite is now 201 cases per backend.

Historical v22 candidate: `kernelgen/candidates/task78-v22-20260919-180120/`.
Submission package: `flagos-task78-v22.zip`, containing the seven operator
files in this directory. All files expose `concat_and_cast_mha_k(k, k_nope, k_rope)`.

v18 finished with 6/8 (Enflame/Kunlunxin failed, Ascend 0.02x); the last complete run was v16 at
1.18x. The best complete recorded run remains v12 at 1.27x.

v19 uses fused output stores on GPU paths, one-dimensional serial-head
RoPE reuse on Enflame/Kunlunxin, and persistent token blocks with hoisted
RoPE loads on Ascend. Ascend caps the entire grid at 32 programs.
All variants preserve cat-then-cast dtype promotion and source strides.
Every nonempty shape enters the new structure. Before submission, performance
was unmeasured.

v19 completed with 8/8 and an average of 1.25x: Iluvatar 2.36x, MetaX 1.45x,
Enflame 0.25x, Hygon 2.45x, Kunlunxin 0.27x, Ascend 0.14x, and International
A/B 1.60/1.51x. It restores both v18 failures and improves Iluvatar, MetaX and
Hygon, but remains 0.02x below the complete v12 result of 1.27x. The v19
device result supersedes the unmeasured wording above.

Run `python3 competition/task78/validate_cpu.py --all` from the repository
root for direct-source CPU semantic checks. To validate an isolated candidate,
pass `--source-dir`; the reusable pre-Arc gate is
`python3 competition/task78/kernelgen/run_candidate_gate.py <candidate-dir>`.
These checks do not compile Triton or prove device performance. The local [validation record](v19/validation.md)
contains the tested source hashes and results.

For new candidates, follow the shared KernelGen/FlagOS workflow and the Task 78
adapter in `kernelgen/README.md`. A missing KernelGen tool must be recorded as
unavailable; code authored by another method must not be mislabeled as
KernelGen output. Deterministic scans and independent source review are
defect-finding gates, not substitutes for target execution. Keep package
preparation, authorized submission, official evaluation, and per-chip source
promotion as separate decisions; append every official result and all eight
target statuses to `results.jsonl`.

Historical version directories and ZIPs remain reproducible references.
The user uploads candidates to FlagOS for real eight-chip evaluation.

v22 originally passed the older local semantic gate and was submitted, but the
Arc result was 6/8: Enflame failed on `num_warps=12`, Hygon failed in the
suffix cast/broadcast lowering, and the six surviving chip scores were
1.64/1.07/0.29/0.25/1.39/1.36x. The stricter compiler-risk regression gate
now rejects v22 before packaging and records both failure patterns. Do not
treat v22 as a valid optimization baseline.

## KernelGen candidate workflow

The reusable KernelGen workflow is documented in
`.agents/skills/kernelgen-flagos/references/reliability-gates.md`. Task 78
adapter rules are in [`kernelgen/README.md`](kernelgen/README.md), with
invocation guidance in `.agents/skills/task78-kernelgen/`. KernelGen outputs
remain isolated under `kernelgen/candidates/` until they pass local checks and
are deliberately prepared for evaluation.
