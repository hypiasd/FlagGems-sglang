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

Use the shared workflow v5 in
`.agents/skills/kernelgen-flagos/references/reliability-gates.md` and the
registered KernelGen MCP operation for code generation, optimization, and
specialization. A config file is not proof that a tool is live. If the tool is
unavailable, continue with read-only diagnosis and an experiment plan, but do
not generate/patch source or claim a KernelGen run.

Keep all generated files under
`competition/task78/kernelgen/candidates/<run-id>/`. Never overwrite the root
submission files or a historical version while exploring candidates.

Every candidate must preserve the exact public function
`concat_and_cast_mha_k(k, k_nope, k_rope)`, use only Triton/Triton-TLE for the
kernel path, and avoid PyTorch fallback, exception fallback, or device-based
bypass. Reject launch-constant-only changes when the request is for a new
version: each evaluated platform needs a structural optimization.

Run the repository CPU validator and syntax checks as local evidence only. A
candidate is not submission-ready until every backend passes the live target
preflight in the shared workflow: exact backend API/config invocation, every
autotune configuration compiled and exercised, device limits queried, complete
launch bounds checked, and reference-based correctness tests passed. If no
trusted per-chip runner is callable, keep the target state inconclusive; do not
spend an official Arc attempt as a substitute unless the user explicitly
chooses a diagnostic submission. Use the official Arc result as performance
evidence. International A and B must be reported separately even when they
share the generic source file.

The deterministic candidate gate is intentionally conservative: it checks portable
`triton.Config` options, autotune-to-launch parameter binding, masked pointer
construction, scalar/vector mask shapes, and coverage for every visible
autotune config. Follow the Task78 two-agent prompt: first a fresh blind source
auditor, then a different fresh reconciler after the static report exists. Keep
both raw agent outputs, platform IDs, and exact source/baseline hashes. The
gate checks receipt consistency; it does not prove the agents ran or that their
reasoning is correct. A blocker or unresolved issue fails the candidate.

For each new version, use the Task 78 adapter in
`competition/task78/kernelgen/README.md`: freeze the exact baseline and Arc
result, make a structural hypothesis for every implementation backend, generate
in isolation, run semantic/static checks, obtain the blind and reconciliation
reviews when available, and separate an evaluation package from promotion of a
new best. Do not describe a candidate as target-validated from CPU tests or AI
review alone. Record generic-source results separately for International A and
B, and keep every Arc attempt plus per-chip best scores in the results record.

Do not place KernelGen Tokens, local MCP configuration, or benchmark credentials
in Git. The repository-local MCP example is
`competition/task78/kernelgen/mcp.json.example`.
