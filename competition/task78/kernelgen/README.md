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
  --json competition/task78/kernelgen/candidates/<RUN_ID>/local-gate.json
git diff --check
```

The candidate gate checks all seven files, the exact public entry, forbidden
fallbacks/native concatenation, Python syntax, and the full 189-case CPU
semantic suite. `validate_cpu.py --source-dir` makes the validator operate on
an isolated candidate directory instead of silently reading the root baseline.
Known autotune/cache-hint syntax is ignored only by the CPU model; target
compilation is still a separate gate.

The CPU validator checks source semantics and memory coverage only. It does
not compile Triton, validate FlagTree lowering, or predict Arc performance.
KernelGen responses with no executed correctness cases or no numeric target
benchmark remain inconclusive. The final performance gate remains an actual
Arc submission.
