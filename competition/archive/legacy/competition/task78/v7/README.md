# Task 78 v7 experiment

v7 fixes the v6 packaging regression by restoring the v5 Ascend bounded
persistent row-tile companion. The generic kernel keeps the v6 segmented
prefix/suffix algorithm and adds a contiguous-input fast path with
compile-time row strides; strided inputs retain the previous general path.

The goal is to recover Huawei correctness while reducing address arithmetic on
the common contiguous case. The package still contains one generic kernel
body plus the known Ascend companion, not three new vendor implementations.

Validation on the Apple M3: Python syntax, AST/CPU semantic checks, and ZIP
integrity. Triton compilation and target-chip performance require online
evaluation.
