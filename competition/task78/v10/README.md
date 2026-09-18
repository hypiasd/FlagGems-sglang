# Task 78 v10 experiment

v10 is a bundled follow-up to the v9 online result (`8/8`, `1.15x`), not a
single-constant sweep. It keeps the v9 four-row contiguous fast path for
Iluvatar, MetaX, and Hygon because those changes produced large online gains.

The Ascend suffix is restored byte-for-byte to the v5-proven bounded
persistent row tile (`BM <= 16`, grid `<= 32`) after v9 reduced Huawei from
`0.22x` to `0.14x`. The Enflame suffix gets a separate regular head-tile
kernel: one program owns several adjacent heads of one token, so its
contiguous path avoids flattening rows followed by integer division/modulo and
uses one warp. The Kunlunxin suffix now uses the same conservative multi-row
contiguous copy path as v9's successful GPU-style suffixes, while retaining
the simple two-segment load/store structure.

All paths retain the exact `expand -> cat -> cast` semantics, the strided
fallback, and Triton-only core computation. Validation on the Apple M3 covers
Python syntax, AST public-entry/no-native-op checks, row/head mapping reference
cases, known Ascend hash, and ZIP integrity. Triton compilation and target
chip performance require online evaluation.
