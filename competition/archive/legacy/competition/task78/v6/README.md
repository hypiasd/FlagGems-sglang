# Task 78 v6 experiment

This candidate changes only the generic `concat_and_cast_mha_k.py` kernel.
It keeps one program per `(token, head)` row, but copies the NoPE prefix and
broadcast RoPE suffix through independent contiguous load/store pairs instead
of loading both sides and selecting with `tl.where`.

The wrapper preserves the reference `cat -> cast` semantics by promoting both
source values to their common dtype before storing into the destination dtype.
The Ascend companion is unchanged from v5 and is included in the submission
package so the online comparison still has the known bounded-grid path.

Validation on the Apple M3: Python syntax, AST entry-point checks, ZIP
integrity, and CPU reference-equivalence cases. Triton compilation and target
chip performance still require FlagOS online evaluation.
