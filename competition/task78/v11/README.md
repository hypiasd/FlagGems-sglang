# Task 78 v11 experiment

v11 follows the v10 online result (`8/8`, `1.14x`) with an adaptive head-tile
strategy. v10 improved Enflame from `0.18x` to `0.31x`, but its four-head
tile reduced Iluvatar from `2.29x` to `2.20x` and Hygon from `2.06x` to
`1.96x`; v11 therefore does not force one tile width across vendors.

The Iluvatar and Hygon suffixes use a two-head tile to reduce register and
live-pointer pressure while retaining RoPE reuse. MetaX and Enflame use an
eight-head tile for small row tiles and fall back to four or one head as the
tile grows. The generic entry and Kunlunxin suffix now use the same
token/head broadcast layout, reducing duplicate RoPE loads; the strided
fallback remains one program per `(token, head)`. Ascend adds a new bounded
persistent token loop: each program processes several tokens and head tiles,
reusing one RoPE load per head tile while retaining the v10 grid bound and
strided fallback.

All paths preserve exact `expand -> cat -> cast` semantics and Triton-only
core computation. Validation on the Apple M3 covers Python syntax, AST
public-entry/no-native-op checks, adaptive head-tile reference cases, and ZIP
integrity. Triton compilation and target-chip performance require online
evaluation.
