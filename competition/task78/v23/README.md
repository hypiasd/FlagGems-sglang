# Task 78 v23 — fused per-backend candidate with adversarial review

Status: locally gated and packaged; not yet submitted to Arc. The package is
[`../flagos-task78-v23.zip`](../flagos-task78-v23.zip). It contains exactly
the seven `concat_and_cast_mha_k*.py` files at ZIP root.

v23 uses the redesigned Workflow v2: KernelGen generation per backend,
deterministic compiler-risk scan, the isolated semantic suite, an adversarial
read-only sub-agent review, KernelGen repairs for every discovered blocker,
then packaging. The generated sources were kept in the ignored candidate run
directory while being repaired; the ZIP is the deliberate promotion artifact.

Structural changes include a fused full-row default/Iluvatar schedule,
Ascend's hoisted token/head tile arithmetic and canonical masked loads,
Enflame's fused column schedule with explicit `other` loads instead of
zero-tile initialization, Hygon's broadcast-before-cast suffix path,
Kunlunxin's autotune-selected `meta["HS"]` grid, and MetaX's frontend-safe
configuration and fused tile path.

The local result is not a device result. Target compiler compatibility and
performance remain unknown until Arc evaluates the ZIP.

See [validation.md](validation.md) for hashes, gate results, and the remaining
risks reported by the independent review.
