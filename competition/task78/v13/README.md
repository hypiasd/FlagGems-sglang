# Task 78 v13 experiment

v13 starts from the first valid v12 baseline: `8/8`, average `1.27x`.
The v12 per-chip result was Iluvatar `2.21x`, MetaX `1.50x`, Enflame
`0.40x`, Hygon `2.39x`, Kunlunxin `0.31x`, Ascend `0.25x`, International A
`1.59x`, and International B `1.51x`.

The candidate freezes the v12 generic, Hygon, and international-compatible
paths. Kunlunxin gets a new bounded contiguous row-batch kernel: it uses the
validated v5-style up-to-16-row schedule only for contiguous inputs and keeps
the existing strided fallback for general layouts. Enflame tests a 16-head
tile only for the smallest source blocks while retaining one warp and the
one-dimensional RoPE load. Ascend tests an 8-head tile only for source blocks
up to 128 elements, keeping its persistent grid bounded at 32 programs.

The ZIP contains only the seven `.py` operator files. Validation on the Apple
M3 covers Python syntax, AST public-entry/no-native-op checks, reference shape
mapping, and ZIP integrity. Triton compilation and target-chip performance
still require online evaluation.
