# Task 60 official result ledger

This ledger currently has one historical v11 result imported from the grounds
project runbook. Its source commit and file hash are known; the original ZIP,
FlagOS record ID, batch, and exact submission time were not retained. Keep that
provenance limitation visible when selecting future baselines.

| Record | Version | Submitted | Pass | Official aggregate |
| --- | --- | --- | ---: | ---: |
| `legacy-task60-v11-runbook-import` | v11 | 2026-09-17 | 8/8 | 1.41x |

## Per-target best observations

| Target | Best score | Record |
| --- | ---: | --- |
| 天数智芯 | 1.85x | v11 |
| 沐曦 | 1.35x | v11 |
| 燧原 | 1.45x | v11 |
| 海光 | 1.58x | v11 |
| 昆仑芯 | 1.01x | v11 |
| 华为 | 1.18x | v11 |
| 国际通用 A | 1.42x | v11 |
| 国际通用 B | 1.44x | v11 |

Task 60 v5's Enflame `9.40x` observation was confirmed invalid and is excluded.
New records are appended by `competition/flagos_s2_workflow.py record-result`;
the score and target statuses must match the terminal FlagOS record.
