# Task 78 KernelGen workflow

The repository-wide guarded KernelGen workflow is defined by
`.agents/skills/kernelgen-flagos/references/reliability-gates.md`. This
directory adds only the Task 78 operator contract and candidate layout; it does
not redefine what counts as a valid KernelGen result.

This directory is the candidate-generation workspace for
`concat_and_cast_mha_k`. Nothing under it is a submission package. Generated
code must stay in a candidate directory until it passes the local checks and
is deliberately promoted into a version directory.

## One-time MCP setup

The official `kernelgen-flagos` Skill is vendored at
`.agents/skills/kernelgen-flagos/`. To configure the MCP service locally:

1. Copy `mcp.json.example` to the repository-root `.mcp.json`.
2. Replace the placeholder with the local KernelGen Token.
3. Keep `.mcp.json` untracked and restart the agent.

Never paste the token into chat or commit it. The root `.mcp.json` is ignored
by Git.

## Candidate layout

Use one run directory per KernelGen request:

```text
competition/task78/kernelgen/candidates/<run-id>/
├── prompt.md
├── ascend/
├── enflame/
├── hygon/
├── iluvatar/
├── kunlunxin/
├── metax/
└── generic/
```

The `generic` result is evaluated independently on International A and B. Do
not overwrite the seven root submission files during generation.

## Task contract for every request

The public entry point must remain:

```python
def concat_and_cast_mha_k(k, k_nope, k_rope):
    ...
```

The exact semantics are:

```text
k[t, h, :nope_dim] = k_nope[t, h, :]
k[t, h, nope_dim:] = k_rope[t, 0, :]
```

The implementation must use Triton or Triton-TLE only. Reject candidates that
use PyTorch/native fallback, `try/except` fallback, device checks to bypass the
kernel, a renamed public function, or a changed submission layout.

For a version candidate, every evaluated platform needs a structural change;
changing only `BLOCK_SIZE`, `num_warps`, or another launch constant is not
enough. Record the mechanism separately for each platform, including the two
international evaluations of the shared generic file.

## Recommended request

Use the official Skill with a target platform and a bounded iteration count:

```text
Use kernelgen-flagos to optimize Task 78 concat_and_cast_mha_k on <TARGET>.
Read competition/task78/concat_and_cast_mha_k_<TARGET>.py as the baseline and
write candidates only below competition/task78/kernelgen/candidates/<RUN_ID>/.

The operation is pure data movement and cast: copy k_nope into the prefix of k
and broadcast k_rope[t,0,:] into the suffix of every head. Preserve arbitrary
source strides and the output dtype conversion.

Generate three structurally different candidates. At least one must explore a
new tile/layout or TLE-Lite/TLE-Struct memory path; do not only tune launch
constants. The public function must be exactly
concat_and_cast_mha_k(k, k_nope, k_rope). Do not use native/PyTorch fallback,
try/except fallback, or device-based bypass. Run 5 iterations with a 1.5x
target and return the code, correctness report, and the structural diff.
```

If MCP is unavailable, stop and report that fact rather than silently replacing
KernelGen with hand-written code under this workflow.

## Promotion checks

Run from the repository root before promoting a candidate:

```bash
python3 -m py_compile competition/task78/kernelgen/candidates/<RUN_ID>/*.py
python3 competition/task78/validate_cpu.py --source-dir competition/task78/kernelgen/candidates/<RUN_ID> --all
python3 competition/task78/kernelgen/run_candidate_gate.py \
  competition/task78/kernelgen/candidates/<RUN_ID> \
  --require-review \
  --review-json competition/task78/kernelgen/candidates/<RUN_ID>/subagent-review.json \
  --json competition/task78/kernelgen/candidates/<RUN_ID>/local-gate.json
git diff --check
```

Before the candidate gate, run the deterministic compiler-risk scan:

```bash
python3 competition/task78/kernelgen/review_candidate.py \
  competition/task78/kernelgen/candidates/<RUN_ID> \
  --json competition/task78/kernelgen/candidates/<RUN_ID>/static-review.json
```

The candidate gate also runs this scan automatically. Passing an old
`subagent-review.json` cannot bypass a newly discovered compiler-risk rule.
The scan currently blocks runtime JIT branches, uncapped tile powers,
non-power-of-two or unproven `num_warps`, and the Hygon 1-D load-cast followed
by broadcast pattern that failed in the v22 Arc run.

Then send the candidate and `static-review.json` to a read-only sub-agent. The
sub-agent must inspect every backend and write a small receipt with this shape:

```json
{
  "review_type": "read-only-subagent",
  "reviewer": "<agent id or nickname>",
  "candidate": "<run id>",
  "reviewed_source_sha256": {"default": "..."},
  "backend_findings": {
    "default": {
      "status": "pass",
      "notes": [],
      "evidence": ["static-review:<finding-or-none>"]
    }
  },
  "blockers": []
}
```

The sub-agent is not trusted as a compiler or benchmark. Its receipt is a
mandatory review checkpoint, and `blockers` must be empty before packaging.
`status` must be one of `pass`, `fail`, or `unknown`; every backend must state
what evidence supports the status. An unverified target compiler is recorded
as evidence, not silently treated as a pass. The deterministic scan runs
inside `run_candidate_gate.py`, so both layers must agree before packaging.
The gate also compares every receipt hash with the current seven source files,
and requires `candidate` to equal the candidate directory name, so a review
cannot be reused after the candidate changes or copied between runs.
The deterministic scan catches known high-risk patterns (runtime JIT control
flow and uncapped tile powers); the sub-agent checks branch shapes, implicit
broadcasts, pointer/mask safety, and whether the structural change is real.
The target compiler/device gate remains necessary.

The candidate gate checks all seven files, the exact public entry, forbidden
fallbacks/native concatenation, Python syntax, and the full 189-case CPU
semantic suite and—when `--require-review` is used—a passing sub-agent receipt.
`validate_cpu.py --source-dir` makes the validator operate on
an isolated candidate directory instead of silently reading the root baseline.
Known autotune/cache-hint syntax is ignored only by the CPU model; target
compilation is still a separate gate.

The CPU validator checks source semantics and memory coverage only. It does
not compile Triton, validate FlagTree lowering, or predict Arc performance.
KernelGen responses with no executed correctness cases or no numeric target
benchmark remain inconclusive. The final performance gate remains an actual
Arc submission.

For regression testing against the historical failures:

```bash
python3 competition/task78/kernelgen/review_candidate.py \
  competition/task78/kernelgen/candidates/task78-v21-20260919-130000
python3 competition/task78/kernelgen/review_candidate.py \
  competition/task78/kernelgen/candidates/task78-v22-20260919-180120
python3 competition/task78/kernelgen/test_review_candidate_regressions.py
```

The v21 scan must report runtime JIT branch blockers in the Iluvatar and
MetaX files. The v22 scan must report the Enflame `num_warps=12` blockers and
the Hygon cast-before-broadcast blocker.
