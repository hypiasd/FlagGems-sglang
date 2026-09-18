# Task 78: `concat_and_cast_mha_k`

The submission entry is `concat_and_cast_mha_k` in
`concat_and_cast_mha_k.py`. It fuses the reference sequence
`expand -> cat -> cast` into one Triton kernel and writes directly to the
destination tensor.

The current version is a correctness-first cross-chip baseline. Real
performance must be measured on the eight FlagOS target chips; this checkout
is on an Apple M3 and cannot replace those measurements.
