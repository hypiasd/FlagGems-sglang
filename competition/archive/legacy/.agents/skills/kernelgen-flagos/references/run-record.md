# KernelGen run record

Create one append-only record per candidate/target attempt. Keep generated
source and service responses in the run directory (or another local ignored
location); never store tokens, cookies, credentials, or host-specific secrets.
Do not rewrite a prior attempt when repairing or retrying it: assign a new
attempt ID and link it to the parent candidate.

Record these fields when applicable:

```text
schema_version
run_id
parent_run_id
operator
repository_revision
mode                         # generate | optimize | specialize | autotune
service_state                # absent | configured_unavailable | callable
tool_name                    # exact registered name; omit if not callable
request_path_or_hash
service_job_id
started_at / finished_at
contract_hash
baseline_source_hash
candidate_source_hash
target_backend / device
compiler_version / runtime_version
structural_hypothesis
structural_diff
local_test_command
local_correctness             # pass | fail | inconclusive + counts
blind_reviewer_agent_id       # platform-provided invocation identity
blind_review_raw_hash
reconciliation_reviewer_agent_id # must differ from Stage A reviewer
reconciliation_review_raw_hash
review_input_source_hashes    # candidate and baseline, per target
review_provenance             # platform-recorded | unverified_claim
target_compile                 # pass | fail | unknown
target_correctness             # pass | fail | inconclusive + suite/case IDs
target_case_signatures         # shapes, dtypes, relevant strides/layouts
benchmark_method
benchmark_warmups
benchmark_raw_samples          # baseline and candidate, per case
benchmark_case_set_id
benchmark_required_case_ids   # adapter's complete set, if defined
target_raw_result_hash
target_provider_invocation_id
per_case_median_speedups
aggregate_metric               # exact operator scoring rule, not an assumed mean
measurement_confidence        # stable | noisy | single_observation | unknown
evidence_state                 # see reliability-gates.md
decision                       # reject | keep_candidate | package_experiment | promote_best
reason
```

Keep a separate per-operator best table that points to immutable run IDs. Show
the best observed score per target and preserve each raw attempt. If a score
looks anomalous, mark it and seek an independent repeat when possible; never
delete it or silently replace it with a preferred value.
