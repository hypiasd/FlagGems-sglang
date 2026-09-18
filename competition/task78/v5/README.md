# Task 78 v5 experiment

Observed v4 online baseline, 2026-09-18: 8/8, mean 0.94x.
Chip order: Iluvatar 1.84, MetaX 1.00, Enflame 0.10, Hygon 1.34,
Kunlunxin 0.39, Ascend 0.08, international A 1.37, B 1.37.

Only the Ascend file changes in v5. The generic file is identical to v4.
The Ascend kernel retains a maximum of 32 persistent programs, but uses a
2D row/column tile. Token/head division is computed per row rather than per
element. Prefix and suffix use independent contiguous column ranges within
one launch. Both loads retain source dtype promotion before destination cast.
Up to 16 rows are handled per iteration, with a target of at most 4096
elements per source tile for ordinary head dimensions.

Validation: Python syntax, ZIP integrity, and 36 CPU index/coverage/cast
cases passed, including strided inputs, mixed fp16/bf16 inputs, empty
prefix/suffix and more than 65535 rows. These are not Triton execution tests.
There is no local Ascend device. Correctness and performance of the new
device access pattern require online evaluation; retain v4 as baseline.
The v4 success does not isolate the cause of v1-v3 failures.
