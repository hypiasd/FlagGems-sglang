# Independent reviewer protocol

This is a reusable, supplementary source-review protocol for generated or
optimized kernels. Task/framework adapters supply the target IDs, contract,
baseline, workload signatures, capability profile, visible tests, and static
report; they must not weaken the two-stage independence rules. The reviewers
help find source-level defects and explain evidence gaps; they do not replace
the executable target preflight in `reliability-gates.md`.

## Stage A: blind audit

Use a real read-only sub-agent in a fresh context. Provide exact candidate and
baseline bytes/hashes, operator contract, visible tests/workload signatures,
live target/compiler capability profile, and relevant prior external failures.
Do not provide current
deterministic-scan findings or scanner source. Do not ask the reviewer to find
a fixed number of "novel" bugs: that rewards speculation. Ask for a concrete
audit of dataflow, shapes/ranks, masks, pointers/strides, casts, launch mapping,
compile-time/runtime control flow, bounds, and baseline regressions. Require it
to report actual per-target coverage and unknown target-only behavior.

Save the raw tool response unchanged. The orchestrator records the real
platform agent/invocation ID in the normalized receipt; it must not be invented
or copied from a prior run. A receipt is bound to the exact candidate and
baseline hashes. Any source change invalidates it.

## Stage B: independent challenge

Only after Stage A's output is durably saved, run deterministic scans and then
spawn a *different* read-only sub-agent in a fresh context. Give it the same
source/contract bundle plus the immutable Stage A output and deterministic
reports. It must independently challenge Stage A, account for every Stage A
and static finding, and search for issues neither found. Every finding needs a
source location, failure mechanism, activation condition, evidence, and
confidence. Preserve residual uncertainty; no finding may disappear without a
reasoned disposition.

Adapters must normalize Stage A with a protocol version, stage marker,
platform-provided agent ID, candidate/baseline identity and hashes, complete
per-target coverage, structured findings, verdict, and limitations. Stage B
must bind the exact Stage A receipt and static report hashes, record a distinct
platform agent ID, account for every Stage A/static finding exactly once, and
preserve any additional findings and residuals. Target IDs and source-hash
maps are adapter-defined; the two-stage semantics are not. Do not invent a
one-shot checklist receipt for a new adapter.

For a clean Stage A verdict, each target's coverage record still needs a
specific prose trace and source references; labels such as "checked masks" are
not enough. Findings include target, severity, location, mechanism, activation
condition, confidence, and concrete evidence. The Stage B response must give a
reasoned disposition for every finding, not just copy status labels.

## What the gate can and cannot establish

The local gate can check receipt shape, source/baseline hashes, that the two
recorded agent IDs differ, and that every static/Stage A finding has a
disposition. It cannot cryptographically prove the agents were spawned, the
inputs they saw, that they actually read the source, or that their reasoning is
correct. Preserve raw platform responses and invocation records. If those are
unavailable, label the review as an unverified review claim; never say the
Python gate proved sub-agent independence.

An empty finding list is not evidence that a reviewer can detect unknown bugs.
Measure that capability separately with blinded holdout trials: use known
historical failures, remove their diagnosis/logs from reviewer inputs, then
compare the blind report to the independently curated failure facts. Include
mechanism-level variants with different names, shapes, and source structure.
Report per-trial detection, false alarms, and misses. Static-rule regression
tests prove only that the rules still catch their encoded patterns; they do not
evaluate the reviewer. Do not tune a prompt on a holdout and then reuse that
same case as an unbiased success metric. Even strong holdout performance does
not let AI review clear missing target execution evidence.
