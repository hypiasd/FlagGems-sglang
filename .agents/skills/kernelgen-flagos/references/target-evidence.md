# Normalized target-evidence manifest

Use one manifest per exact candidate-source/baseline-source pair and target.
Do not include credentials, private input tensors, or hidden-test contents.
`input_signature` records metadata only: tensor shapes, dtypes, relevant
strides/layouts, and scalar parameters needed to identify the case.

```json
{
  "schema_version": 1,
  "run_id": "operator-run-target-attempt",
  "target": {
    "backend": "backend-name",
    "device": "device-model",
    "compiler": "compiler-version",
    "runtime": "runtime-version"
  },
  "provenance": {
    "kind": "captured-live-tool-result",
    "provider": "target-runner-name",
    "invocation_id": "provider-job-or-tool-call-id",
    "raw_result_sha256": "<sha256 of separately preserved unmodified result>"
  },
  "source_sha256": "<sha256 of exact candidate source file>",
  "baseline_source_sha256": "<sha256 of exact comparison source file>",
  "compile": {
    "success": true,
    "log_ref": "local-log-or-service-job-reference"
  },
  "correctness": {
    "suite_id": "suite-name-and-revision",
    "total_cases": 2,
    "passed_cases": 2,
    "cases": [
      {
        "case_id": "tail-case",
        "input_signature": "x: shape=[33,65], dtype=fp16, strides=[65,1]",
        "status": "passed"
      },
      {
        "case_id": "strided-case",
        "input_signature": "x: shape=[32,64], dtype=bf16, strides=[128,2]",
        "status": "passed"
      }
    ]
  },
  "benchmark": {
    "method": "device-event-or-framework-benchmark-name-and-version",
    "warmup_runs": 10,
    "case_contract": {
      "case_set_id": "adapter-defined-case-set-revision",
      "required_case_ids": ["tail-case", "strided-case"],
      "score_rule": "adapter's documented aggregation formula"
    },
    "cases": [
      {
        "case_id": "tail-case",
        "input_signature": "x: shape=[33,65], dtype=fp16, strides=[65,1]",
        "baseline_ms": [0.12, 0.11, 0.12, 0.11, 0.12],
        "candidate_ms": [0.09, 0.09, 0.10, 0.09, 0.09]
      }
    ]
  }
}
```

The gate verifies that the candidate file is byte-for-byte the returned source,
both source hashes match, target/compiler identity is present, compilation is
reported successful, and every enumerated correctness case is reported passed.
It also links a provenance hash to a separately preserved raw tool result when
`--raw-target-result` is supplied. A zero-case report is inconclusive; a failed
case is rejected. These checks establish consistency of the supplied records,
not that the provider actually ran them: a hand-authored manifest and matching
hash are still a claim. Preserve the live tool/official platform result and
record its invocation/job ID; the workflow must not call a self-authored JSON
manifest an independently verified target run.

For measurement, every benchmark case must refer to a correctness-passed case
with the identical signature. Keep at least five positive raw timings for both
baseline and candidate and record at least one warm-up. The gate computes the
median of each timing list and their ratio, then a geometric mean across the
listed cases. It calculates relative median absolute deviation (MAD) and the
largest sample's relative deviation from the median. Preserve every sample;
the robust median is not permission to delete an outlier.

The adapter must declare a versioned `case_contract` with the complete required
case IDs and score rule. If the measured set differs from that exact list, the
gate labels the output `performance_reported_subset`; the aggregate is only a
diagnostic for those listed cases. With complete coverage, it labels the
normalized result `performance_reported_complete`, not `measured`: the Python
gate cannot authenticate execution, provider identity, or provenance. The
calling workflow may promote an evidence state only after directly inspecting
the raw result from a trusted live runner/official evaluation. `reported_only`
and subset results never establish a task-level score.

The computed geometric mean is a diagnostic comparison over the listed cases,
not the task's official score (`official_score_computed` is always false in this
gate). It does not override an operator's score formula or an Arc leaderboard
result. Record any official aggregate separately. Without directly observed,
preserved target output, correctness remains `target_reported`; incomplete
timing remains `performance_reported_incomplete`. Only the workflow—not this
JSON parser—can assign `target_validated` or `measured` after inspecting a
trusted live runner/official result.

Run the gate with:

```sh
python3 .agents/skills/kernelgen-flagos/scripts/kernelgen_gate.py response.json \
  --phase target --public-symbol <public_function> \
  --target-evidence target-evidence.json \
  --candidate-source candidate.py --baseline-source baseline.py \
  --raw-target-result target-result.raw.json
```

`--minimum-speedup` defaults to `1.0`; `--max-relative-noise` defaults to `0.10`.
Use an operator-specific threshold only when the benchmark protocol defines
one, and record the chosen value in the run record.
