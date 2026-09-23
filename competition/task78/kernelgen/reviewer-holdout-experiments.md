# Reviewer holdout experiments

This is a small retrospective capability check of the two-agent source-review
workflow. It is separate from deterministic scanner regression tests. It does
not establish reliable generalization: there are too few trials, v21 was a
performance hypothesis rather than a known defect, and target-failure labels
are available for only v22.

## Inputs and artifact identity

Reviewers received the candidate and matching root baseline source, contract,
and target context. Stage A did not receive scanner output, scanner source, or
the historical Arc diagnostics. A corrected Stage B run received only the v22
static report and the v22 Stage A result. No reviewer ran code or benchmarked.

The reviewed candidate directories were checked against the submitted ZIPs;
each archive contains the same seven root-level operator files byte-for-byte:

| Candidate | ZIP SHA-256 | Identity check |
| --- | --- | --- |
| v21 `task78-v21-20260919-130000` | `21b3c506323ae58d30cd8b5cd134f47a052d67be42c23022e7edbd18e5f53d2d` | all seven source files match |
| v22 `task78-v22-20260919-180120` | `0784b38e6a9af43c25c50fed2b83ca297dc28b67f2eb3e7c52bb542be8fb7eac` | all seven source files match |

The v21 candidate was not submitted, so its performance concern has no target
confirmation. v22 was officially 6/8; the known failures were Enflame's
`num_warps=12` configuration and Hygon's suffix cast/broadcast lowering. See
the [Task 78 result ledger](../README.md).

## Reviewer outcomes

| Trial | Reviewer invocation | Finding | Outcome against known target result |
| --- | --- | --- | --- |
| v21 Stage A | `01a0cf17-272d-7c11-814d-f12cefaf4061` | MetaX grid launches `2*(A+B)` jobs although parity mapping needs only `A` prefix and `B` suffix jobs; surplus programs do masked no-op work. | A plausible performance hypothesis, not a confirmed defect or score change; v21 was not submitted. |
| v22 Stage A | `01a0cf17-27a0-7f51-a228-67dfaaf0c497` | Flagged Enflame `num_warps=12` as a target-compiler risk; found no source-level Hygon defect. | Detected the Enflame failure class; missed the actual Hygon lowering failure. |
| v22 Stage B | `01a0cf1d-2c61-7620-b460-67281dec93ce` | Kept Enflame 12-warps as unverified; considered Hygon's rank-1 cast then broadcast shape-valid. Added speculative Enflame static-loop expansion and grid-axis-limit risks. | Did not identify the Hygon Arc failure. The two extra risks have no target confirmation and must not be counted as defects. |

An earlier Stage B attempt mixed v21 and v22 static findings in one prompt. It
was discarded and is not counted. The corrected Stage B run was restricted to
v22. The three outcomes above were recovered from completed agent results for
this retrospective; the original full platform envelopes/raw invocation
payloads were not preserved as repository files, so this record is not an
audit-complete invocation archive.

## Interpretation and workflow change

- For v22, Stage A detected one of the two known Arc failure classes (Enflame)
  and missed the other (Hygon). The corrected Stage B also missed Hygon. AI
  review therefore did **not** prevent this known failure.
- The static regression scanner independently recognizes both v22 failure
  patterns. That demonstrates rule coverage for those encoded patterns only;
  it is not evidence that a reviewer can discover unknown defects.
- The v21 MetaX observation is useful as a falsifiable optimization hypothesis
  but supplies no evidence about correctness or measured speed.
- Stage B can generate plausible, target-dependent risks that cannot be
  classified as true or false without execution. These remain residual
  hypotheses, not automatic blockers or confirmed bugs.

The shared workflow now requires real, distinct Stage A and Stage B
sub-agent invocations; raw outputs and platform invocation IDs must be saved
before normalization; each receipt is hash-bound to exact candidate/baseline
sources and reports; and the deterministic gate checks receipt completeness
without claiming to authenticate the agents. Capability evaluation is tracked
separately from scanner unit tests, with holdout detection, misses, and false
alarms recorded. Until more blinded trials exist, source review remains a
useful additional perspective—not a substitute for local semantic tests,
target execution, or the official Arc result.
