# Task 78: `concat_and_cast_mha_k`

The submission entry is `concat_and_cast_mha_k` in
`concat_and_cast_mha_k.py`. It fuses the reference sequence
`expand -> cat -> cast` into one Triton kernel and writes directly to the
destination tensor.

The generic implementation is a cross-chip row-tiled kernel. It keeps the
NoPE prefix and broadcast RoPE suffix as two regular load/store segments in
one Triton program, avoiding mixed masked loads and data-dependent pointer
selection. The current `v5`/`v6` Ascend companion remains available as a
separate submission experiment, but the generic path is the primary candidate
for testing whether one portable kernel is sufficient.

Real performance must be measured on the eight FlagOS target chips; this
checkout is on an Apple M3 and cannot replace those measurements.
