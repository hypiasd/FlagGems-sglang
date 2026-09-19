# KernelGen reliability gates

This reference is shared by generation, optimization, and platform-specialization
workflows. A KernelGen response is evidence only after it passes the gates below.

## Evidence states

Use these states in a run record:

- `requested`: the MCP request was accepted, but no artifact exists yet.
- `generated`: code was returned, but correctness and performance are unknown.
- `locally_validated`: repository checks and semantic tests passed.
- `target_validated`: the target compiler/device accepted the candidate and all
  target correctness tests passed.
- `measured`: target performance was measured against the same baseline.
- `promotable`: all required gates passed and the candidate may be assembled
  into a version.
- `rejected`: a deterministic gate failed.
- `inconclusive`: the service or target environment did not provide enough
  evidence; this is not a pass.

Never convert `generated` or `inconclusive` into `promotable` by interpretation.

## Gate 0 — operator contract

Before calling MCP, record the contract:

- exact public entry and argument order;
- output shape, dtype, device, contiguity, and mutation rules;
- supported layouts, strides, dtypes, empty cases, and numerical tolerance;
- target platform and the baseline source revision;
- forbidden behavior such as native fallback, `try/except` fallback, or device
  bypass;
- the performance metric and its baseline measurement.

If a contract field is unknown, mark it `unknown`; do not let a generated
candidate silently define the contract.

## Gate 1 — MCP response

For a response that claims completion, require:

- a completed status and no service error;
- a non-empty `triton_code` (or the explicitly documented code field);
- the expected public function name in the returned source;
- no missing or null verification report when verification was requested;
- no performance claim without a numeric measurement and measurement context.

For `autotune_kernel`, `verify_result.total_tests == 0` is always
`inconclusive`, even when `success == true`. For `optimize_kernel`, which is a
single-shot rewrite, the absence of a returned code artifact is a rejection,
not a successful no-op.

## Gate 2 — source contract

Parse the returned source before executing it. Reject when:

- the required public entry is missing or renamed;
- a `try`/`except` path hides compilation or device failure;
- a native/PyTorch implementation bypasses the intended kernel path;
- source and wrapper use different kernel names;
- an unsupported import, platform API, or TLE dialect is introduced;
- output allocation, strides, dtype promotion, masking, or empty handling
  differs from the contract;
- the change only adjusts launch constants when a structural change is
  required.

Static checks cannot prove performance. They only prevent obvious invalid
submissions from consuming target evaluation opportunities.

For Triton autotune candidates, the local gate must also verify that every
`triton.Config` option is supported by the target ABI, that launch kwargs do
not duplicate autotuned tile constexprs, and that host-side loop/grid counts
remain consistent with every configured tile. If the target compiler is
unavailable, unknown or non-portable options are blockers rather than assumed
supported features.

## Gate 3 — local correctness

Run the repository's tests and a source-level semantic harness. Cover the
contract boundaries, not just the happy path: mixed dtypes, tails, empty
dimensions, non-contiguous views, zero strides, storage offsets, and input
immutability where applicable.

If the local harness cannot compile the target dialect, report semantic
evidence separately from compiler evidence. Do not call a CPU model a device
test.

If the candidate has autotune configs, a single-config semantic pass is
insufficient. Exercise every visible config, at least on tail, zero-segment,
and strided cases; a config that leaves missing or duplicate writes rejects the
candidate.

## Gate 4 — target evidence

The target gate requires:

- successful target compilation;
- all target correctness tests passing;
- the same input distribution and timing method for baseline and candidate;
- a numeric score with enough repeated measurements to identify an outlier;
- source hash and environment recorded with the score.

When only one noisy measurement is available, record it as `measured` with
low confidence; do not use it to discard a stable baseline without a second
observation.

## Gate 5 — promotion

Promote only when the candidate is target-validated, has a real performance
measurement, and the intended structural change is documented. Keep rejected
and inconclusive candidates isolated so their evidence remains auditable.

KernelGen may propose a candidate, but it never decides promotion. A read-only
sub-agent review is useful before the target run, especially for public names,
masked addresses, stride arithmetic, and duplicated stores.

The sub-agent is a discovery layer, not a checklist formatter. Any credible
finding outside the deterministic rules remains unresolved and blocks
promotion until KernelGen repairs it or a target compile/correctness smoke test
resolves it. An empty deterministic report does not override a non-empty
`novel_findings` receipt.

## Mode-specific use

- `generate_kernel`: use for a new operator after Gate 0; expect the most
  human adaptation work.
- `optimize_kernel`: use for a small, well-scoped kernel; treat it as a
  single-shot rewrite unless the service returns real validation evidence.
- `specialize_kernel`: use for platform migration; inspect the entire diff for
  target-only APIs and semantic drift.
- `autotune_kernel`: use only when input specs and executable correctness tests
  are available; stop or mark `inconclusive` when attempts remain at zero.
