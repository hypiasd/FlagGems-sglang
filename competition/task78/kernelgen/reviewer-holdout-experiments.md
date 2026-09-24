# Reviewer holdout experiments

This is a retrospective capability check of the two-agent source-review
workflow through v24. It is separate from deterministic scanner regression
tests, and these exposed examples are now **retired** as blind holdouts. The
trials do not establish reliable generalization: source reviewers saw no live
compiler/device capability profile, and the service could not be invoked here.

## Inputs and artifact identity

Reviewers received the candidate and matching root baseline source, contract,
and backend names. Stage A did not receive scanner output, scanner source, or
historical FlagOS-result diagnostics. Stage B received only the corresponding Stage A
result and static report. The v23/v24 source bundles were isolated under
`/tmp` so reviewers could not read repository history. No reviewer ran code or
benchmark. The prompts did not include the official workload shape matrix or a
live device-capability query, so they could not settle target-only launch/API
behavior.

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
| v22 Stage B | `01a0cf1d-2c61-7620-b460-67281dec93ce` | Kept Enflame 12-warps as unverified; considered Hygon's rank-1 cast then broadcast shape-valid. Added speculative Enflame static-loop expansion and grid-axis-limit risks. | Did not identify the Hygon failure in the FlagOS v22 record. The two extra risks have no target confirmation and must not be counted as defects. |
| v23 Stage A | `01a0cf28-f8e2-79a1-86f0-b2f48eb6f794` | Found Iluvatar's conditional `BC=256` coverage hole when host loop counts are derived using 512; noted Ascend config support as unknown. | A separate source-level hazard, not one of the three failures reported by FlagOS. It did not identify the Enflame duplicate-`BR` runtime error or Kunlunxin numeric mismatch; Ascend `multibuffer` stayed unknown. |
| v23 Stage B | `01a0cf2e-ae3c-7082-aed5-50753129578a` | Confirmed the Iluvatar hole; found Enflame's uncapped head-grid axis against a limit stated in a source comment; treated explicit `BR`/`NRC` as a non-issue because visible autotune configs varied only `HS`. | The static heuristic flag did not explain the actual Enflame duplicate-`BR` failure. Kunlunxin mismatch remained unresolved/not found. The comment-based grid limit is not a substitute for querying the target. |
| v24 Stage A | `01a0cf28-f964-7080-8063-01f9ce26e132` | Found no source-provable defect; noted `T*H` launch-grid limits as unverified. | Did not conclude that the actual Enflame grid exceeded the device cap; workload maxima and live limits were absent. |
| v24 Stage B | `01a0cf2e-adbf-7590-945a-fa8636358ed2` | Found conditional 32-bit overflow risk in wide inner-stride offsets for default, Hygon, Iluvatar, and MetaX. | Additional source-level risk under very large strides, not the official v24 launch failure and not device-tested. |

An earlier Stage B attempt mixed v21 and v22 static findings in one prompt. It
was discarded and is not counted. The corrected Stage B run was restricted to
v22. The findings above were recovered from completed agent results; the raw
platform envelopes and temporary v23/v24 scan reports were not committed, so
this is a retrospective summary rather than an audit-complete invocation
archive. The v24 CPU model did pass all 201 cases on each of seven backends,
yet did not model the target grid-axis limit and therefore missed the official
v24 launch failure. The v23 all-backend CPU sweep was interrupted during the
Hygon backend and is not reported as a complete run.

## Interpretation and workflow change

- For v22, Stage A detected one of the two known FlagOS failure classes (Enflame)
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
- On v23, the static scanner's explicit-constexpr heuristic flagged `BR`, but
  the independent reconciler judged it non-issue from visible config keys; the
  FlagOS-reported duplicate-argument error remained unexplained. This is a concrete
  example that adding or matching a source pattern is not root-cause proof.
- On v24, both reviewers could only state that launch limits were unknown. A
  live query plus actual workload signatures is required to calculate whether
  any generated grid violates that target's limit.

The shared workflow v5 keeps source review but adds a hard, executable
per-target preflight: exercise the real wrapper/config path, compile and invoke
each configuration, query device limits, verify generated launch bounds, and
compare every required result against the reference. The target-evidence gate
requires exact case/config coverage. If no trusted target runner is available,
the candidate remains inconclusive and is not submission-ready. This is how
the process can catch the tested failure mechanisms without adding v21-v24
error-string rules. The current tool registry exposes no KernelGen target
runner, so this session could not execute that preflight; no claim is made that
the new requirement has already caught the v22/v23/v24 target failures.
