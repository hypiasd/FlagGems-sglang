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

# General Triton optimization workflow

This procedure is for optimizing an existing Triton operator outside a
framework-specific integration flow. The shared reliability policy is
[`references/reliability-gates.md`](references/reliability-gates.md); use
[`references/run-record.md`](references/run-record.md) for every attempt.
FlagGems and vLLM integrations should follow their dedicated optimization
guides, while retaining the shared evidence gates.

## 1. Preflight

1. Read the operator implementation, its public API, repository tests, and
   benchmark. Identify the actual target(s), baseline revision, score formula,
   shape distribution, dtype/layout contract, and available test hardware.
2. Check whether the required KernelGen operation is callable in the current
   tool registry. A local MCP config only proves configuration was written;
   it does not prove the server connected or exposed the tool.
3. If the tool is unavailable, record `tool_unavailable`, complete the
   read-only diagnosis and a bounded experiment proposal, and stop code
   generation. Do not silently hand-author a replacement or claim a speedup.
4. Freeze baseline source/tests and record their hashes before requesting a
   candidate. Do not modify the baseline in place.

## 2. Propose a testable change

Tie each proposed optimization to observed code or measurements. State the
structural change, expected benefit, likely regressions, target(s), and cases
that would falsify it. A request for a new design must not be satisfied by
changing only `BLOCK_SIZE`, `num_warps`, or another launch constant.

Keep the number of candidates bounded. Prefer a few distinct hypotheses over
many nearly identical prompts. If multiple backends are involved, make the
target scope explicit and retain separate source and evidence for each target;
do not infer that a shared source passes every backend.

## 3. Generate an isolated candidate

Call the currently registered KernelGen `optimize_kernel` operation with the
complete baseline source, concise operator contract, target, previous
measurements, and the structural hypothesis. Save the exact request and raw
response beside the returned code in a unique, ignored run directory.

Before running code, parse the response and source. Require a non-empty code
artifact, the exact public function, syntactically valid Python, unchanged
contract, and no hidden PyTorch/native/exception fallback. A service success
flag without source is not a candidate; code without executed tests is only
`generated`.

## 4. Validate semantics and inspect risk

Run repository tests and a contract-focused semantic suite on the isolated
candidate. Include boundary/tail, empty, dtype-promotion, stride/layout, and
mutation cases relevant to the operator. Exercise each autotune configuration
or explain why it cannot be enumerated. Preserve exact commands and counts.

Use the shared two-agent protocol in
[`references/reviewer-protocol.md`](references/reviewer-protocol.md): first a
fresh blind source audit without the static report, then a different fresh
reviewer to challenge that audit and reconcile every static/initial finding.
Save raw sub-agent outputs and platform invocation IDs. Any source change
invalidates both receipts. Do not require a fixed number of "novel" bugs and
do not call a locally generated receipt proof that an independent agent ran.
If no real independent reviewer is available, report that gap.

Local/CPU success is semantic evidence only. It does not establish that a
vendor compiler accepts the code or that the target runs it correctly.

## 5. Require target evidence for target claims

When target execution is available, preserve the unmodified raw runner output
and invocation/job ID as well as compiler/runtime identity,
compile outcome, executed test cases and input signatures, correctness counts,
source/baseline hashes, benchmark method, warm-ups, and raw repeated timings.
Use the normalized target-evidence manifest accepted by
`scripts/kernelgen_gate.py --phase target`. The gate requires the evidence to
match the exact source files and will not accept an unshaped scalar speedup or
zero-test “success”.

Use the operator's real score rule and adapter-declared complete case set.
Compare baseline and candidate on the same target/cases, retain raw samples,
and report per-case medians plus the declared aggregate. A partial case set is
diagnostic only; it is not a task-level speedup. Require at least five timing
samples per case; high variance or a gain inside measurement noise calls for a
repeat. Keep correctness and performance as separate statuses. A self-authored
normalized JSON manifest is a claim; the parser cannot authenticate execution.

If target hardware or a trustworthy remote runner is unavailable, leave the
target result `inconclusive`. A package may be prepared only as an explicitly
unvalidated user-controlled experiment when the repository adapter permits
it. Never replace the best source or call the candidate faster on local
semantic or AI-review evidence alone.

## 6. Decide and report

Keep these outcomes separate:

- `generated`: source returned;
- `locally_validated`: repository semantics and source gates passed;
- `target_validated`: direct, preserved target-run output confirms exact-source
  compilation and non-empty correctness tests;
- `measured`: direct, preserved target-run output covers the adapter's complete
  case set and compares against the frozen baseline;
- `package_experiment`: package prepared, with every remaining risk visible;
- `promote_best`: measured result improves the real score and meets correctness
  requirements for the required target set.

Do not submit, publish, or replace a best implementation automatically.
Report the structural diff, all gate results, exact target coverage, evidence
hashes, per-target measurements, and what remains unknown. Append a new run
record for every retry; never rewrite prior evidence.
