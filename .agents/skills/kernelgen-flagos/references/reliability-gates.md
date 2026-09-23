# KernelGen workflow v5: executable target preflight and evidence provenance

This is the shared workflow for generation, optimization, and platform
specialization. A repository adapter may add operator-specific constraints,
target matrices, or scoring rules, but must not weaken these evidence gates.

## Operating rules

- A local MCP config file is not proof that the server is connected. A tool is
  available only when the current agent can actually call it from its tool
  registry.
- If the required KernelGen tool is not callable, finish read-only diagnosis
  and write a concrete experiment request if useful. Do not generate or patch
  kernel source by another method while calling it KernelGen output. Resume
  code generation only after the tool is available.
- Keep the known-good source and candidate isolated. A candidate never replaces
  the best source merely because it was generated, reviewed, or packaged.
- Treat every result as evidence about an exact source hash, target, contract,
  and test/benchmark set. Do not transfer a pass from one of those to another.
- Source review and CPU semantic models find some defects; neither proves
  target compilation, runtime correctness, or performance.
- A candidate is not ready for a scarce official evaluation until every
  required target passes an executable target preflight. If no trusted target
  runner/compiler is callable, the state remains `inconclusive`; source review
  cannot substitute for missing hardware evidence.

## Evidence states

Record one state per candidate and target. States are not interchangeable:

| State | Meaning |
| --- | --- |
| `prepared` | Contract, baseline, target, and experiment are recorded. |
| `tool_unavailable` | Config may exist, but the required MCP tool is not callable. No code was generated. |
| `generated` | A source artifact was returned and saved; no correctness or speed claim follows. |
| `locally_validated` | Syntax, contract, and available local semantic tests passed. Target behavior remains unknown unless separately evidenced. |
| `target_validated` | The exact source compiled and passed a non-empty correctness suite on the named target. |
| `measured` | Target timings and the comparison baseline were collected on the same case set with repeated raw samples. |
| `target_reported` | A normalized manifest claims target compilation/correctness; execution has not been directly verified from a trusted live result. |
| `performance_reported_complete` | The manifest has repeated timings for the adapter-declared case set; a script checked consistency, but the raw run is not authenticated and no official score is inferred. |
| `partial_measurement` | Timings exist for only a subset/diagnostic case set; this is not a task-level speedup. |
| `reported_only` | A normalized JSON claim lacks a preserved raw result or invocation/job reference. |
| `arc_candidate` | Local/review gates passed and a package was prepared for a user-controlled competition run; any missing target evidence is explicit. |
| `arc_submitted` | The user submitted the identified package; the official evaluation is pending. |
| `arc_completed` | The official result, including failures and per-target scores, was recorded. |
| `rejected` | A concrete contract, source, correctness, compilation, or performance gate failed. |
| `inconclusive` | Required evidence is absent, ambiguous, mismatched, or too noisy. |

Never turn `generated`, `locally_validated`, `tool_unavailable`, or
`inconclusive` into `target_validated` or `measured` by interpretation. `AI
reviewed` is not a correctness state.

## Workflow

### 1. Freeze the contract and baseline

Record the public API, reference semantics, output shape/dtype/device/layout,
supported input layouts and dtypes, empty/tail behavior, mutation rules,
tolerances, and forbidden fallbacks. Identify the exact baseline source hash,
test/benchmark revision, target backend/device, and score aggregation rule.
For each target, capture the official workload signatures and target capability
profile (compiler/runtime revision, supported launch/config API, and live device
limits). Unknown fields remain explicit; AI must not infer target limits from
another chip or from a prior error string.
Unknown contract fields stay marked unknown; generated code cannot define them.

Read the current tests and benchmark before proposing a change. If the target
case distribution or benchmark comparison is unknown, report that gap instead
of inventing shapes or a speedup target.

### 2. Check execution capability, not just configuration

Classify the required service as `absent`, `configured_unavailable`, or
`callable`. The last state requires the requested operation to be visible in
the current tool registry. Do not read or print tokens to make this decision.

When not callable, stop only the code-generation portion: preserve the
read-only diagnosis, source-level bottlenecks, and a ready-to-run bounded MCP
request. Do not rewrite configuration, repeatedly restart, or switch to
hand-authored kernel code without the user's explicit change of method.

### 3. State a falsifiable optimization hypothesis

For each experiment, record:

- the bottleneck evidence and the exact source/target it applies to;
- the structural change (not merely launch-constant tuning when a new design
  is requested);
- the predicted gain and likely regression/failure modes;
- the cases and targets that could disprove the hypothesis.

Do not bundle unrelated speculative changes. For a multi-backend release,
record a separate structural delta and predicted effect for every backend;
shared source files still require separate target evidence.

### 4. Generate and preserve candidates

Use the registered KernelGen operation for the chosen mode. Keep the request,
raw response, returned source, target identity, and hashes in a unique
run directory outside submission paths. Use bounded requests and compare a
small number of structurally distinct candidates against the frozen baseline.
Never overwrite the baseline or a numbered historical result during search.

Reject missing artifacts, wrong public entries, syntax errors, contract drift,
hidden native/PyTorch or exception fallback, and a change that does not satisfy
the stated structural objective. A service's `success` field alone is not
evidence that tests ran.

### 5. Run local checks and the two-agent review protocol

Run syntax, repository tests, and a contract-focused semantic harness on the
isolated candidate. Prefer property-based case generation from the operator
contract: vary shapes around tile/loop boundaries discovered in the candidate,
generate legal stride/layout combinations, and record seeds and signatures.
Keep the official case set separate from generated stress cases. Record counts
and exact commands. If only a CPU model is available, label it local semantic
evidence; do not call it target compilation or device validation.

For non-trivial or target-sensitive kernels, follow
[`reviewer-protocol.md`](reviewer-protocol.md). Stage A is a fresh read-only
sub-agent invocation given source, baseline, contract, tests, and target context
but not the current deterministic findings or scanner implementation. Save its
raw output and exact source/baseline hashes. Only then run/show the static
reports and ask a *different* fresh sub-agent to challenge Stage A, reconcile
every finding, and search for issues both missed. Any source edit invalidates
both reviews. Do not require a fixed count of "novel" bugs. If real sub-agent
invocation records are unavailable, call the result an unverified review
claim; the Python gate can validate receipt consistency but cannot prove
independent reasoning or authenticate agent identity.

An empty review is only “no issue found by this review.” It cannot clear an
unverified compiler/runtime risk. A credible unresolved risk requires a repair
or target smoke evidence before the candidate is called ready for that target.

### 6. Mandatory executable preflight on every target

For every backend, a trusted live runner must execute a preflight against the
exact candidate. This is the primary detector for target-only failures; source
reviewers are not expected to predict undocumented compiler/device behavior.
The preflight must:

- invoke the real public wrapper and backend launch path, exercising argument
  binding and the target's actual autotune/config wrapper;
- construct and compile every declared launch/autotune configuration with the
  target compiler, not only the configuration that happened to win tuning;
- query device limits from the live runtime and check every generated launch
  grid for the official workload and generated boundary cases;
- compare output against the reference on the target, including every official
  correctness case and generated stride/tail/dtype cases; do not use an AI
  judgment in place of numeric comparison;
- preserve exact source/baseline hashes, target/compiler/runtime, case and
  config IDs, input signatures, device limits, logs, raw outputs, and runner
  invocation ID.

These are mechanism-level oracles, not version-specific signatures: real
entrypoint invocation catches binding/decorator interactions; actual config
construction/compilation catches backend API and lowering incompatibilities;
reference comparison catches numerical/coverage errors; live limit queries
catch invalid grid/resource requests. A failure blocks that candidate/target.
Missing target execution or incomplete config/case coverage is `inconclusive`,
not a pass. The structured parser checks consistency only; preserve and inspect
the raw result from the trusted runner because hashes alone cannot authenticate
an execution.

Every target correctness run must cover the adapter-declared correctness case
set exactly; zero, partial, or unbound coverage is `inconclusive`. Any failed
correctness case rejects that target candidate. A compile-only smoke test is
useful but does not establish correctness.

For a task-level performance claim, the task adapter must declare a versioned
required case set and the official score formula. The measured case IDs must
match that required set exactly, and candidate/baseline must use the same
target, environment, and distribution. Without a declared required set, call
the result a diagnostic subset and do not produce a task-level aggregate.
Preserve raw samples and compare robust per-case medians; aggregate only with
the operator's documented score rule.
Require at least five samples per case. If dispersion is high or the gain is
within observed noise, repeat instead of declaring a win. A single scalar
speedup without shapes, baseline identity, and measurement context is not
benchmark evidence.

If target hardware or a trustworthy remote target runner is unavailable, leave
target state `inconclusive` and do not label the candidate submission-ready.
Only if the user explicitly chooses a diagnostic submission may an adapter
prepare an `unvalidated_experiment`; it must be clearly separated from normal
optimization submissions, must not replace the best source, and must not be
described as a performance improvement.

### 7. Decide, package, and learn from the official result

Keep three decisions distinct: **candidate generated**, **package prepared for
evaluation**, and **best implementation promoted**. Promotion requires the
required target correctness and measured score for the operator's actual
targets; preparation alone does not.

Do not consume scarce evaluation attempts for a cosmetic or unmotivated
candidate. Before packaging, show the expected benefit, structural diff,
coverage by target, evidence gaps, and the exact artifact hash. Do not submit
externally unless the user has explicitly asked for that action.

After an official run, append the result rather than overwriting history:
package/source hash, timestamp, each target's status and score, exact failure
diagnostics, and aggregate score if valid. Maintain both the best observed
score per target and a stability note; retain anomalous raw values and label
them rather than silently dropping them. Turn only reproduced or directly
evidenced failures into reusable target constraints.

Evaluate source-review capability with blinded holdouts and mechanism-level
mutations. Keep those results separate from production gates and static-rule
regressions. Vary names, formatting, thresholds, and shapes so an evaluator
cannot pass by memorizing a file, line, or error string. Report detection,
misses, and false alarms. Once a holdout's diagnosis is exposed, retire it from
unbiased evaluation; do not tune a prompt or scanner on it and continue to
count it as a blind success. Regardless of reviewer scores, the target
preflight remains the acceptance oracle for target-only behavior.

## Normalized target-evidence gate

`scripts/kernelgen_gate.py --phase target` accepts a saved MCP response only
alongside a target-evidence manifest and the exact candidate/baseline source
files. See [`target-evidence.md`](target-evidence.md) for provenance and case
coverage rules. The parser checks internal consistency; unless the workflow
also preserves a raw result from a trusted live runner and verifies complete
task case coverage, report the result as claimed or subset-only—not
`target_validated` or task-level `measured`. It never makes the cross-target
promotion decision by itself.
