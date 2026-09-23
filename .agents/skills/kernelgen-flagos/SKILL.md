---
name: kernelgen-flagos
description: >
  Unified GPU kernel operator generation and optimization skill. Automatically detects the target
  repository type (FlagGems, vLLM, or general Python/Triton) and dispatches to the appropriate
  specialized sub-skill. Includes operator generation, MCP-based iterative optimization, and
  feedback submission sub-skills. Use this skill when the user wants to generate or optimize a
  GPU kernel operator, create a Triton kernel, or says things like "generate an operator",
  "create a kernel for X", "optimize triton kernel", or "/kernelgen-flagos".
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: gpu-kernel-generation
  tags: [kernelgen, triton, gpu, mcp, operator-generation, operator-optimization, flaggems, vllm, feedback]
  argument_hint: "<operator_name> [--func-type <type>]"
  user_invokable: true
  compatibility: "Python 3.8+, PyTorch with CUDA, Triton"
allowed-tools:
  - Bash
  - Bash(gh:*)
  - Bash(python:*)
  - Bash(python3:*)
  - Bash(command:*)
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - AskUserQuestion
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# kernelgen-flagos — Unified GPU Operator Generation Skill

This is a **unified entry point** that bundles generation and optimization sub-skills into one.

This entry point produces candidates; it does not automatically promote them.
Shared acceptance gates are in
[`references/reliability-gates.md`](references/reliability-gates.md), and the
run-record fields are in [`references/run-record.md`](references/run-record.md):

| Sub-skill file | Purpose |
|---|---|
| **Generation** | |
| `kernelgen-generate.md` | Generate GPU kernels for **any** Python/Triton repository |
| `kernelgen-generate-for-flaggems.md` | Specialized generation for **FlagGems** repositories |
| `kernelgen-generate-for-vllm.md` | Specialized generation for **vLLM** repositories |
| **Optimization** | |
| `kernelgen-optimize.md` | Optimize existing Triton kernels via MCP iterative optimization (general purpose) |
| `kernelgen-optimize-for-flaggems.md` | Optimize Triton operators and integrate into **FlagGems** (3 modes: built-in/external/experimental) |
| `kernelgen-optimize-for-vllm.md` | Optimize Triton operators and integrate into **vLLM** (with CustomOp registration) |
| **Platform Specialization** | |
| `kernelgen-specialize.md` | Specialize Triton operators to target platforms (e.g., GPU → Ascend NPU) via MCP `specialize_kernel` |
| `kernelgen-specialize-for-flaggems.md` | Platform specialization + **FlagGems** integration (4 modes: vendor-ops/vendor-fused/override-builtin/experimental) |
| **MCP Configuration** | |
| `kernelgen-mcp-setup.md` | Check and auto-configure the `kernelgen-server` MCP service (URL built-in, user only provides Token) |
| **Feedback** | |
| `kernelgen-submit-feedback.md` | Submit bug reports and feedback via GitHub or email |

All sub-skill files are located in the **same directory** as this `SKILL.md` file.

---

## Routing Protocol — Follow This BEFORE Doing Anything Else

### Phase 0: MCP Configuration and Runtime Check

Before any generation/optimization call, distinguish local configuration from live
tool availability. A `.mcp.json` entry does not prove that this session connected
the server or registered its tools.

Use the Glob tool to find `kernelgen-mcp-setup.md` in this skill's directory:

```
Glob: **/skills/kernelgen-flagos/kernelgen-mcp-setup.md
```

Then use the Read tool to read the matched file and **follow its instructions exactly**.

- If the required operation is visible in the current tool registry → proceed to Phase 1.
- If MCP is not configured → the setup skill will guide the user through configuration.
  Stop code generation until the tool is callable. Read-only repository diagnosis
  and experiment planning may continue without making source changes.
- If configuration exists but the required tool is not callable → do not request a
  new token, rewrite config, or assume repeated restarts will fix it. Continue only
  with read-only diagnosis and a bounded experiment plan; stop code generation until
  the tool is exposed.

### Phase 1: Detect Repository Type

Use the Glob tool to check for project identity files in the current working directory:

```
Glob: pyproject.toml
Glob: setup.py
Glob: setup.cfg
```

Then use the Read tool to read whichever file exists. Determine the **project name** from
the file contents (e.g., `name = "flag_gems"` in pyproject.toml, or `name='vllm'` in setup.py).

Also use the Glob tool to check for characteristic directory structures:

**FlagGems indicators** (match ANY):
- `src/flag_gems/` directory exists
- Project name is `flag_gems` or `flag-gems` or `FlagGems`
- `import flag_gems` appears in test files

**vLLM indicators** (match ANY):
- `vllm/` directory exists at the repo root (with `vllm/__init__.py`)
- Project name is `vllm`
- `csrc/` directory exists alongside `vllm/`

### Phase 2: Dispatch to Sub-skill

Based on the detection result, use the **Read tool** to read the appropriate sub-skill file
from this skill's directory, then **follow the instructions in that file exactly**.

**To locate the sub-skill files**: They are in the same directory as this SKILL.md. Use the
Glob tool to find the path:

```
Glob: **/skills/kernelgen-flagos/kernelgen-generate.md
```

Then use the Read tool to read the matched path.

#### Decision Table

**Generation requests** (user wants to create/generate a new operator):

| Detection Result | Action |
|---|---|
| FlagGems repository detected | Read `kernelgen-generate-for-flaggems.md` and follow it |
| vLLM repository detected | Read `kernelgen-generate-for-vllm.md` and follow it |
| Neither detected (or unknown) | Read `kernelgen-generate.md` and follow it |

**Optimization requests** (user wants to optimize an existing operator, mentions "optimize", "speedup", "improve performance"):

| Detection Result | Action |
|---|---|
| FlagGems repository detected | Read `kernelgen-optimize-for-flaggems.md` and follow it |
| vLLM repository detected | Read `kernelgen-optimize-for-vllm.md` and follow it |
| Neither detected (or unknown) | Read `kernelgen-optimize.md` and follow it |

**Specialization requests** (user wants to migrate/specialize an operator to a different platform, mentions "specialize", "migrate to Ascend/NPU", "platform migration"):

| Detection Result | Action |
|---|---|
| FlagGems repository detected | Read `kernelgen-specialize-for-flaggems.md` and follow it |
| Neither detected (or unknown) | Read `kernelgen-specialize.md` and follow it |

**Feedback requests**:

| Detection Result | Action |
|---|---|
| User reports a bug or requests feedback submission | Read `kernelgen-submit-feedback.md` and follow it |

**Important rules:**
1. **Always detect first, dispatch second.** Never skip detection.
2. **Read the entire sub-skill file** before starting execution — do not partially read it.
3. **Follow the sub-skill** for framework- and mode-specific steps. The shared
   `references/reliability-gates.md` is normative for live tool availability,
   evidence states, correctness/performance claims, and promotion; it takes
   precedence if a sub-skill conflicts with it.
4. **Do not mix sub-skills.** Once you dispatch to a sub-skill, follow it to completion.
5. If the user explicitly requests a specific sub-skill (e.g., "use the FlagGems version"),
   honor that request regardless of auto-detection results.
6. **KernelGen generation must use the live registered tool** for the selected
   operation (`generate_kernel`, `optimize_kernel`, or `specialize_kernel`). Do not
   guess a tool namespace from a config key. If the tool is not callable, do not
   create implementation code under the KernelGen workflow; read-only diagnosis
   and a ready-to-run request may still be delivered. Do not silently substitute
   hand-written code and label it KernelGen output.

### Phase 3: Feedback Handling

At **any point** during the workflow, if the user reports a bug, says something is broken,
or asks to submit feedback about the skill:

1. Use the Read tool to read `kernelgen-submit-feedback.md` from this skill's directory.
2. Follow the feedback submission workflow described in that file.
3. After feedback is submitted, ask the user if they want to continue with the operator
   generation workflow or stop.

## Reliability gates and promotion

Apply these gates after every MCP call:

1. Create or update the operator contract before generation. Record the public
   entry, layouts, dtypes, empty cases, mutation rules, target, and baseline.
2. Save the MCP response and assign an evidence state. `success=true` is not
   sufficient: a null code artifact, a missing public entry, or
   `verify_result.total_tests == 0` is not a successful candidate.
3. Parse and inspect returned source before executing it. Reject renamed
   public functions, hidden fallbacks, unsupported imports, wrapper/kernel
   name mismatches, and semantic changes.
4. Run repository semantic tests separately from target compilation and
   performance tests. Never describe a CPU model as device validation.
5. Preserve the unmodified raw target-run result, provider invocation/job ID,
   exact candidate/baseline hashes, and the adapter's complete required case
   set. A normalized manifest alone is a report claim, not proof of execution;
   a subset benchmark is diagnostic only.
6. For non-trivial kernels, use the two-stage protocol in
   `references/reviewer-protocol.md`: a fresh blind source auditor first, then
   a different fresh reviewer after static findings are available. Save both
   raw tool responses and receipts. The gate checks their consistency, not the
   truth of their reasoning or platform invocation.
7. Promote only after the candidate beats the recorded baseline under the same
   measurement method. Keep rejected and inconclusive artifacts isolated.

For a deterministic first pass over a saved response, run the candidate phase:

```bash
python3 scripts/kernelgen_gate.py response.json \
  --phase candidate --public-symbol <public_function>
```

Target reports use a separate manifest tied to exact candidate and baseline
files, plus the unmodified raw runner result:

```bash
python3 scripts/kernelgen_gate.py response.json \
  --phase target --public-symbol <public_function> \
  --target-evidence target-evidence.json \
  --candidate-source candidate.py --baseline-source baseline.py \
  --raw-target-result target-result.raw.json
```

Read [`references/reliability-gates.md`](references/reliability-gates.md) when
choosing the mode-specific gate, and use
[`references/run-record.md`](references/run-record.md) when recording attempts.

---

## Quick Reference for Users

```bash
# === Generation ===
# Generate a kernel operator (auto-detects repo type)
/kernelgen-flagos relu

# Generate with explicit function type
/kernelgen-flagos rms_norm --func-type normalization

# === Optimization ===
# Optimize an existing Triton kernel (auto-detects repo type)
# Just say "optimize the relu kernel" or "improve kernel performance"
# The skill will automatically dispatch to the right optimization sub-skill

# The skill will automatically:
# - Detect if you're in a FlagGems repo → use FlagGems-specific workflow
# - Detect if you're in a vLLM repo → use vLLM-specific workflow
# - Otherwise → use the general-purpose workflow
```

If you encounter any issues during generation, just say "submit feedback" or "report a bug"
and the skill will guide you through the feedback submission process.
