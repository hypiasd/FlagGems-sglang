# Task 78 two-agent adversarial review protocol

This adapter uses the shared review contract in
`.agents/skills/kernelgen-flagos/references/reliability-gates.md`. The two
review stages must be separate real sub-agent invocations with fresh contexts.
Do not simulate a sub-agent by writing a receipt yourself. Keep each raw agent
response unchanged beside the normalized JSON receipt; the orchestrator adds
the platform-provided agent ID to the normalized receipt.

## Stage A — independent blind source audit

Spawn a read-only reviewer with a fresh context. Give it only:

- the seven exact candidate files and their SHA-256 hashes;
- the seven exact baseline files and their SHA-256 hashes;
- the public operator contract, visible tests, target/compiler profile, and
  relevant historical target failures;
- this Stage A section and the Stage A JSON schema below.

Do not provide the deterministic scan output, scanner findings, the scanner's
source code, or another agent's review. Do not place any of those files in the
reviewer's input bundle. The reviewer may inspect the candidate and baseline
but must not edit either, run code, or execute benchmarks.

Ask the reviewer to independently trace wrapper-to-kernel dataflow, tensor
ranks/shapes, masks, pointers, strides, casts, launch grids, compile-time versus
runtime control flow, bounds/coverage, and regressions against the baseline.
It must list what it actually traced for each backend and identify unavailable
compiler/hardware facts as unknown—not as passes. Do not require it to invent a
"novel" bug. A clean review is valid if its concrete coverage and limitations
are recorded.

Save the raw response unchanged as `blind-review.raw.txt`. Normalize it to
`blind-review.json`, using the platform's actual sub-agent ID as
`reviewer_agent_id`; never invent this ID. The JSON schema is:

```json
{
  "protocol_version": 3,
  "review_mode": "blind-independent",
  "reviewer_agent_id": "<platform-provided agent id>",
  "reviewer_name": "<model or reviewer label>",
  "candidate": "<candidate directory name>",
  "baseline": "<baseline directory name>",
  "reviewed_source_sha256": {"default": "..."},
  "reviewed_baseline_sha256": {"default": "..."},
  "backend_coverage": {
    "default": {
      "status": "complete",
      "checks_run": ["dataflow", "bounds"],
      "analysis_summary": ["Concrete trace of this backend's row/column mapping"],
      "evidence": ["concat_and_cast_mha_k.py:12-31"]
    }
  },
  "findings": [
    {
      "id": "A-1",
      "target_id": "default",
      "severity": "blocker|residual|note",
      "location": "file.py:line or exact expression",
      "mechanism": "why the source may fail or regress",
      "activation": "shape, stride, launch or target condition",
      "confidence": "high|medium|low",
      "evidence": ["concrete source/dataflow evidence"]
    }
  ],
  "verdict": "pass|blocked",
  "limitations": ["target-only behavior not established by source review"]
}
```

Use every backend key: `default`, `ascend`, `enflame`, `hygon`, `iluvatar`,
`kunlunxin`, and `metax`. Each coverage entry must summarize concrete code
paths and cite source locations; a list of generic checklist labels is not
enough. `blocker` findings require a source repair or target
evidence; do not pass them by putting them in `limitations`. Any source edit
invalidates both the receipt and its hashes, so rerun Stage A on the new
candidate bytes.

## Stage B — independent challenge and reconciliation

After Stage A is complete and its raw response/receipt are saved, run the
deterministic compiler-risk scan. Then spawn a *different* read-only reviewer
with a fresh context. Provide the same contract/source/baseline bundle, the
immutable Stage A raw response and receipt, the deterministic scan JSON, and
this Stage B schema. Require the reviewer to inspect the code independently,
challenge Stage A's reasoning, compare every Stage A finding with every static
finding, and look for additional risks missed by both. It must not silently
remove or weaken an earlier finding.

Canonical JSON means UTF-8 from `json.dumps(value, ensure_ascii=False,
sort_keys=True, separators=(",", ":"))`; the final gate computes this hash for
the scan report. `blind_receipt_sha256` is SHA-256 of the exact raw bytes of
`blind-review.json`. For a static finding, use ID
`static:<canonical-json-sha256>` where the canonical object is
`{"target_id": <target id>, ...<finding fields>}`.

Save the raw response as `reconciliation-review.raw.txt` and the normalized
receipt as `reconciliation-review.json`:

```json
{
  "protocol_version": 3,
  "review_mode": "independent-reconciliation",
  "reviewer_agent_id": "<different platform-provided agent id>",
  "reviewer_name": "<model or reviewer label>",
  "candidate": "<candidate directory name>",
  "baseline": "<baseline directory name>",
  "reviewed_source_sha256": {"default": "..."},
  "reviewed_baseline_sha256": {"default": "..."},
  "blind_receipt_sha256": "<sha256 of exact blind-review.json bytes>",
  "static_report_sha256": "<canonical-json sha256 of static-review.json>",
  "dispositions": [
    {
      "finding_id": "blind:A-1 or static:<sha256>",
      "decision": "duplicate|non_issue|residual|unresolved",
      "rationale": "source evidence for this disposition"
    }
  ],
  "additional_findings": [],
  "verdict": "pass|blocked",
  "limitations": ["target-only behavior still requires target evidence"]
}
```

There must be exactly one disposition for every Stage A and static finding.
Each item in `additional_findings` uses the Stage A finding fields plus a
`disposition` of `residual` or `unresolved` and a non-empty `rationale`.
Keep residuals visible; any unresolved or blocker finding fails the gate. A
passing receipt means the reviewers completed this source-level protocol; it
does not prove their reasoning is correct, authenticate target execution, or
replace compiler/device tests. The local gate verifies hashes/schema and that
two distinct platform IDs were recorded. It cannot itself prove the platform
spawned those agents; retain raw tool outputs and invocation records for audit.
