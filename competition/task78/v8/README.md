# Task 78 v8 experiment

v8 keeps the v7 generic contiguous fast path and the v5-proven Ascend
row-tile. It adds official suffix-specific launch variants for Iluvatar,
MetaX, Enflame, Hygon, and Kunlunxin. The variants share the same copy/cast
kernel and only change `num_warps`, following the corresponding FlagGems
backend heuristics.

This is a controlled launch-parameter experiment. It does not use native
Torch fallback or change the mathematical implementation. The goal is to
recover the large gap visible in v7 on MetaX, Enflame, Hygon, and Kunlunxin
while preserving the v5 Ascend correctness path.

Validation on the Apple M3: Python syntax, AST public-entry checks, CPU
reference-equivalence cases, and ZIP integrity. Triton compilation and target
chip performance require online evaluation.
