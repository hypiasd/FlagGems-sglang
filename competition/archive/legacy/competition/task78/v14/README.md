# Task 78 v14 experiment

v14 is based on the v13 online result: `8/8`, average `1.23x`. The v13
scores were Iluvatar `2.23x`, MetaX `1.50x`, Enflame `0.49x`, Hygon
`1.97x`, Kunlunxin `0.28x`, Ascend `0.33x`, International A `1.52x`, and
International B `1.50x`.

The result isolates three useful v13 changes: Enflame's 16-head small-block
tile, Ascend's 8-head small-block persistent tile, and Iluvatar's contiguous
four-row batch. v14 actively retunes each of them: Enflame widens the
16-head regime to 256-element source blocks and uses two warps for larger
blocks; Ascend lets tiny-block persistent programs cover two tokens while
keeping the hard 32-program bound; and Iluvatar tests an eight-row batch for
small blocks. It removes the v13 multi-head tail specialization that
coincided with the Hygon and international regressions, and returns
Kunlunxin's contiguous multi-head schedule to the v12 head-tile path after
its row-batch score fell from `0.31x` to `0.28x`. A new, narrow optimization
also remains: when the input has exactly one head, every head-tile wrapper
launches a one-head tile, avoiding guaranteed masked lanes without changing
the multi-head schedule.

The ZIP contains only the seven `.py` operator files. Validation on the Apple
M3 covers Python syntax, AST public-entry/no-native-op checks, row/head mapping,
and ZIP integrity. Triton compilation and target-chip performance still
require online evaluation.
