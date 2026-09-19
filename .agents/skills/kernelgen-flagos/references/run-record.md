# KernelGen run record

Keep one run record per MCP request. Store it outside Git when it contains
generated code or service output; never store tokens, cookies, credentials, or
host-specific secrets.

The record should contain:

```text
run_id
operator
repository
mode                    # generate | optimize | specialize | autotune
target
baseline_revision       # source hash or version identifier
contract_revision       # hash of the contract/test specification
request_started_at
request_finished_at
service_status
service_job_id
evidence_state           # see reliability-gates.md
returned_code_hash
public_entry
local_correctness        # pass/fail/inconclusive + counts
target_correctness       # pass/fail/inconclusive + counts
baseline_measurements
candidate_measurements
confidence               # high/medium/low
structural_change
decision                 # promote/reject/inconclusive
rejection_reasons
```

For repeated target measurements, append one record per attempt rather than
overwriting the previous score. Keep a separate per-operator best table that
points back to the run IDs; a best score is not automatically a stable score.

