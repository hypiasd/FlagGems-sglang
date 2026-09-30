# Task 60 adapter: `clamp_position`

Task 60 uses the shared FlagOS S2 lifecycle in [`competition/WORKFLOW.md`](../../WORKFLOW.md)
and `competition/flagos_s2_workflow.py`. This file records only Task 60's
operator contract and adapter details.

## Contract

The public entry point is `clamp_position(seq_lens)`. For every element, return
`max(seq_lens - 1, 0)` while preserving input shape and dtype and leaving the
input unchanged. The submission ZIP contains exactly `clamp_position.py` at
its root. The official target matrix is Iluvatar, MetaX, Enflame, Hygon,
Kunlunxin, Ascend, International A, and International B.

The exact official correctness case IDs, batch, score field, and visible
leaderboard goal must be re-read and frozen from the Task 60 details in Chrome
before a run is prepared. Unknown case coverage or an unavailable task profile
blocks generation and submission; do not copy Task 78's shapes or score rules.

## Baseline and result history

`results.jsonl` contains a clearly labeled historical v11 import from the
project runbook, bound to source commit `16c14a1` and source SHA-256. Its
original ZIP and platform record ID were not retained. The Task 60 v5 Enflame
`9.40x` observation was confirmed invalid and is not an eligible score or
baseline. New results must preserve their official record ID, exact package
hash, per-target statuses/scores, and raw evidence digest.

## Local semantic check

Run from the repository root:

```sh
python3 competition/task60/validate_cpu.py \
  --source competition/task60/kernelgen/candidates/<run-id>/source/clamp_position.py
```

The CPU model exercises the generic Triton kernel body on CPU and separately
checks the Ascend host dispatch over empty, boundary, contiguous, and strided
inputs. It does not enter Enflame's GCU launch, compile any vendor backend,
query device limits, or measure speed. Only the official FlagOS record or a
trusted complete target runner can establish target correctness and scores.

## Candidate boundary

Freeze the source selected from the verified Task 60 ledger before generation.
Keep each run and repair isolated; never overwrite `clamp_position.py`, a
numbered release, or historical evidence during generation. The adapter uses
the shared independent-review, deterministic-scan, package-hash, Chrome
preflight, quota, duplicate-check, record-wait, and append-only result stages.
