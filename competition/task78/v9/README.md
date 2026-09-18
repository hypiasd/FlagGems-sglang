# Task 78 v9 experiment

v9 treats v8 as a falsified launch-parameter experiment: raising the warp
count alone regressed Iluvatar and MetaX. The generic root entry remains the
v7 implementation, while the recognized Iluvatar, MetaX, Enflame, and Hygon
suffix entries use a contiguous-input kernel that processes four adjacent
`(token, head)` rows per program when the row tile is small. The Kunlunxin
suffix is restored to the v7 one-row copy path because its v7 result was
already stable.

The Ascend suffix keeps the v5-proven row-tiled algorithm and increases its
bounded row batch from 16 to 32 while respecting the official 48-program grid
limit. All paths still fuse `expand -> cat -> cast`, use Triton only, and keep
the strided fallback unchanged.

Validation on the Apple M3: Python syntax, AST public-entry checks, CPU
reference-equivalence cases, and ZIP integrity. Triton compilation and target
chip performance require online evaluation.
