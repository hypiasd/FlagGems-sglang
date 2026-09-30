# Task 78 v12 experiment

v12 is a compatibility-and-performance candidate based on the v11 online
result (`7/8`): International B failed while International A improved to
`1.53x`, and Hygon fell to `1.57x`.

The generic, GPU suffix, and Ascend contiguous kernels now load the broadcast
RoPE segment once as a one-dimensional vector and explicitly expand that value
across the head tile. This keeps the RoPE reuse optimization while avoiding a
two-dimensional masked source load and pointer broadcast that may not compile
consistently on the two unspecified international backends. Hygon uses a
four-head tile again, combined with the new one-dimensional RoPE load, to
reduce the v11 register-pressure regression rather than submitting a bare
rollback.

All paths preserve exact `expand -> cat -> cast` semantics. The submission ZIP
contains only the seven `.py` operator files; this README is project-local
documentation and is not included in the ZIP.

Validation on the Apple M3 covers Python syntax, AST public-entry/no-native-op
checks, reference mapping for the adaptive head tiles, and ZIP integrity.
Triton compilation and target-chip performance still require online
evaluation.
