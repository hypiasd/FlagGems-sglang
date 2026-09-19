---
name: task78-kernelgen
description: Generate or optimize FlagOS Season 2 Task 78 kernels through the KernelGen workflow while preserving the competition contract and candidate isolation.
metadata:
  version: "1.0.0"
  category: gpu-kernel-optimization
  tags: [flagos, task78, kernelgen, triton, triton-tle]
---

# Task 78 KernelGen workflow

Use this skill when the user asks to use KernelGen, generate a candidate, or
optimize `concat_and_cast_mha_k` in this repository.

Before doing anything else, read:

- `competition/task78/kernelgen/README.md`
- `competition/task78/README.md`
- the current baseline source file for the requested platform

Use the official `kernelgen-flagos` Skill and its KernelGen MCP tools for code
generation, optimization, and specialization. If the MCP service is not
configured or its tools are unavailable, report that and stop this workflow;
do not silently hand-write a result while claiming it came from KernelGen.

Keep all generated files under
`competition/task78/kernelgen/candidates/<run-id>/`. Never overwrite the root
submission files or a historical version while exploring candidates.

Every candidate must preserve the exact public function
`concat_and_cast_mha_k(k, k_nope, k_rope)`, use only Triton/Triton-TLE for the
kernel path, and avoid PyTorch fallback, exception fallback, or device-based
bypass. Reject launch-constant-only changes when the request is for a new
version: each evaluated platform needs a structural optimization.

Run the repository CPU validator and syntax checks before considering a
candidate for promotion. Treat those checks as semantic evidence only; use
the actual Arc result as the performance evidence. International A and B must
be reported separately even when they share the generic source file.

For each new version, use the repository's Workflow v2 in
`competition/task78/kernelgen/README.md`: KernelGen generation per backend,
automatic deterministic hard gate, isolated semantic validation, adversarial
read-only sub-agent review for novel risks, then deliberate packaging. Do not
describe a candidate as ready merely because the CPU validator or sub-agent
review passed.

Do not place KernelGen Tokens, local MCP configuration, or benchmark credentials
in Git. The repository-local MCP example is
`competition/task78/kernelgen/mcp.json.example`.
