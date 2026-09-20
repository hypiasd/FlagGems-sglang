# Task 78: `concat_and_cast_mha_k`

Current candidate: [v24 validation record](v24/validation.md) and
[`flagos-task78-v24.zip`](flagos-task78-v24.zip) — locally gated through all
five Stages including the adversarial read-only review, **not yet submitted to
Arc**. The root submission files in this directory remain the Arc-proven **v19**
set (8/8, avg 1.25x), so upload the ZIP rather than the root files.
`flagos-task78-v23.zip` (5/8) is superseded.

v24 targets the three sub-1x chips of that v19 run (Ascend 0.14x, Enflame
0.25x, Kunlunxin 0.27x) with portable structural changes and no repeat of the
v22/v23 failure classes. Two adversarial review rounds were needed: the first
receipt returned `fail` on Ascend and Hygon with four blockers — Ascend
multiplied row indices in int32 before widening them and unrolled
`tl.static_range` over a shape-derived trip count; Hygon had dropped the 2-D
tile-product cap and used a broadcast-address 2-D rope load plus 2-D cast in the
suffix-lowering area that failed on Hygon in v22. All four were repaired (Ascend
now widens first and walks row blocks with the grid-stride loop v19 was accepted
with; Hygon uses a fully 1-D row form) and the second receipt adjudicated them
closed with no new blockers. Both receipts are kept in the ignored candidate
run directory. Residual risks left for Arc, including Ascend `num_warps=8` (the
first non-4 value on an Arc-tested Ascend launch) and `num_stages=2` on four
files, are listed in the v24 validation record.

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

For new candidates, promotion now requires two reviews before an Arc upload:
the deterministic compiler-risk scan in
`kernelgen/review_candidate.py`, followed by a read-only sub-agent review with
an auditable receipt. A candidate with unresolved branch-shape, implicit
broadcast, pointer/mask, or tile-bound blockers is not packaged. This reduces
avoidable target compilation failures but does not replace Arc validation.

Historical version directories and ZIPs remain reproducible references.
The user uploads candidates to FlagOS for real eight-chip evaluation.

v22 originally passed the older local semantic gate and was submitted, but the
Arc result was 6/8: Enflame failed on `num_warps=12`, Hygon failed in the
suffix cast/broadcast lowering, and the six surviving chip scores were
1.64/1.07/0.29/0.25/1.39/1.36x. The stricter compiler-risk regression gate
now rejects v22 before packaging and records both failure patterns. Do not
treat v22 as a valid optimization baseline.

## KernelGen candidate workflow

The repository-local KernelGen workflow is documented in
[`kernelgen/README.md`](kernelgen/README.md). The official project Skill is
under `.agents/skills/kernelgen-flagos/`; Task 78-specific routing and safety
rules are under `.agents/skills/task78-kernelgen/`. KernelGen outputs remain
isolated under `kernelgen/candidates/` until they pass the local validator and
are deliberately promoted into a numbered version.
