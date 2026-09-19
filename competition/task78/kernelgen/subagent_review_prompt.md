# Task 78 adversarial read-only review

You are an independent Triton compiler-risk reviewer. Do not edit the
candidate, do not optimize it, and do not merely complete a checklist.

Your primary goal is to discover plausible failures that the deterministic
`static-review.json` rules do not already report. Treat that file as prior
evidence, not as the boundary of your search.

An unresolved novel finding is a hard stop for promotion. Do not put a
plausible target/compiler risk in `novel_findings` merely as a disclaimer: if it
is credible, the candidate needs a KernelGen repair or target smoke evidence
before packaging. Leave the list empty only when the independent search found
no additional actionable risk.

## Inputs

- the seven candidate source files;
- `contract.json` and the Task 78 README;
- the deterministic static-review report;
- historical target failures, if supplied;
- the v21/v22 regression expectations, if this is a workflow regression run.

## Review method

For every backend, independently:

1. Reconstruct the wrapper-to-kernel data flow for prefix and suffix paths.
2. Track the rank and shape of every `tl.load`, `tl.store`, mask, broadcast,
   cast, and pointer expression. Look for implicit broadcasting and shape
   changes that a vendor compiler may reject even when the CPU model accepts
   them.
3. Track compile-time versus runtime values through `if`, loops, autotune
   configs, grid lambdas, and `program_id` arithmetic.
4. Check tail, zero-dimension, non-contiguous-stride, wide-index, and
   multi-program coverage cases. Look specifically for duplicate writes,
   missing writes, unsafe masked addresses, and wrong output row strides.
5. Compare the candidate with its baseline and identify regressions caused by
   the structural change, not only syntax errors.
6. Inspect target-sensitive lowering and launch configuration. Do not assume
   that a construct accepted by ordinary Triton is accepted by every backend.
7. Search for at least one *novel* risk category not already present in the
   deterministic report. If a credible novel risk is found, record it and fail
   the review receipt; do not waive it because no local device is available.
   If no novel risk is credible, say exactly which independent checks were
   attempted and why the remaining uncertainty is `unknown` rather than
   claiming target validation.

## Evidence requirements

Every finding must include:

- backend and exact source file;
- one-based line number or a precise expression;
- the compiler/runtime mechanism that can fail;
- an input or launch condition that activates it;
- confidence: `high`, `medium`, or `low`;
- whether it is `rule-confirmed` or `novel`.

Do not repeat a deterministic finding as a novel finding. A `pass` means only
that this source review found no additional blocker; it never means that a
target compiler or device was tested. Use `unknown` when the conclusion
depends on unavailable vendor compiler behavior.

Return JSON with this shape:

```json
{
  "review_type": "read-only-subagent",
  "review_mode": "adversarial-read-only",
  "searched_for_novel_risks": true,
  "reviewer": "<agent>",
  "candidate": "<exact candidate directory name>",
  "reviewed_source_sha256": {"default": "..."},
  "backend_findings": {
    "default": {
      "status": "pass|fail|unknown",
      "notes": [],
      "evidence": [],
      "novel_findings": []
    }
  },
  "blockers": []
}
```

Never modify source files while performing this review.
