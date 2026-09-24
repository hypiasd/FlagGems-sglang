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

Follow the **End-to-end workflow and run states** section in the KernelGen adapter as the execution order and evidence contract. Copy `competition/task78/kernelgen/optimization-manifest.example.json` into every new run; forecast evidence must be registered before the first source edit or generation call.

Use the shared workflow in
`.agents/skills/kernelgen-flagos/references/reliability-gates.md`. First
confirm whether the KernelGen MCP operation is callable; a config file is not
proof that the tool is live. If it is absent, do not silently change methods.
For client-specific setup or recovery, read
`.agents/skills/kernelgen-flagos/kernelgen-mcp-setup.md`; Codex uses its
user-level `config.toml` and does not load this checkout's `.mcp.json`.
Only use agent-authored or other non-KernelGen source when the user has
explicitly authorized that method for this run, and label it `agent-authored`
throughout the run record. Never claim that such code came from KernelGen.

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
`competition/task78/kernelgen/README.md`: freeze each backend's own best-source
baseline and latest result, make a structural hypothesis for every
implementation backend, generate in isolation, run semantic/static checks,
obtain the blind and reconciliation reviews, and separate an evaluation
package from promotion of a new best. Do not describe a candidate as
target-validated from CPU tests or AI review alone. Record generic-source
results separately for International A and B. Keep every official attempt in
`competition/task78/results.jsonl`, regenerate `results.md`, preserve all
per-chip observations and same-source variance flags, and compute an all-chip
aggregate only for an official 8/8 completion.

Browser submission and waiting are explicit lifecycle steps, not a blind
retry loop. Re-check task, batch, team, quota, package identity and duplicate
history in Chrome, submit once, confirm the new record, then wait for that
record's terminal result. Stop on an ambiguous upload or account state and
inspect the record list before any further action. Current submission
authorization is limited to Task 78 Batch 6 and its currently visible quota.

Do not place KernelGen Tokens, local MCP configuration, or benchmark credentials
in Git. The repository-local MCP example is
`competition/task78/kernelgen/mcp.json.example`.
