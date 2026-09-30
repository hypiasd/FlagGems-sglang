# Task 78 v17 experiment

The latest online baseline is v16: all eight targets passed, with an average
of `1.18x`. The measured scores were Iluvatar `1.86x`, MetaX `1.47x`, Enflame
`0.60x`, Hygon `1.81x`, Kunlunxin `0.30x`, Ascend `0.21x`, International A
`1.62x`, and International B `1.53x`.

v17 follows the standing constraint that every target receives a structural
kernel change. This is a candidate only; it has not been submitted online.

| Target | Source path | v17 structural change | Targeted issue |
| --- | --- | --- | --- |
| International A | `concat_and_cast_mha_k.py` | Route compact contiguous rows through separate NoPE-prefix and RoPE-suffix kernels, including exact power-of-two rows. | Make the new split mapping observable on both shared generic backends instead of only retaining v16's full-row mapping. |
| International B | `concat_and_cast_mha_k.py` | Same source change, recorded independently because A and B compile the shared file on different backends. | Check whether removing mixed prefix/suffix live state helps B's v16 `1.53x` path. |
| Hygon | `concat_and_cast_mha_k_hygon.py` | Add a two-token × head-tile kernel for compact rows ahead of the flattened two-row path. | Reduce repeated program setup while preserving the v16 non-persistent organization. |
| MetaX | `concat_and_cast_mha_k_metax.py` | Add independent wide-row prefix and suffix kernels before the full-block path. | Reduce register pressure on wide rows, where the v16 full-block specialization is costly. |
| Kunlunxin | `concat_and_cast_mha_k_kunlunxin.py` | Add a non-persistent two-token × head-tile kernel before the v16 split path. | Preserve the safe split organization while adding reuse without reintroducing the failed persistent loop. |
| Ascend | `concat_and_cast_mha_k_ascend.py` | Add a bounded-grid two-token × head-tile kernel for compact rows. | Replace the v16 flat-row mapping only where the compact shape makes pair reuse plausible; the grid remains capped. |
| Enflame | `concat_and_cast_mha_k_enflame.py` | Add a two-token × head-tile kernel before the token-persistent path. | Reduce the v16 token-loop overhead that coincided with the `0.60x` result. |
| Iluvatar | `concat_and_cast_mha_k_iluvatar.py` | Add independent wide-row prefix and suffix kernels before the compact pair path. | Bound live state for wide rows while retaining v16's compact two-token path. |

Local validation covers Python syntax, AST public-entry/no-native-op checks,
token/head tail coverage, prefix/suffix coverage, and ZIP integrity. Triton
and the eight target backends are not available in this Apple checkout, so
online compilation, correctness, and performance still need a submission.
