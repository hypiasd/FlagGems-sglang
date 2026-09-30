# Task 78 v16 experiment

v15 completed with `7/8`: Iluvatar `1.37x`, MetaX `1.51x`, Enflame
`0.48x`, Hygon `1.45x`, Kunlunxin `Failed`, Ascend `0.21x`, International A
`1.55x`, and International B `1.60x`. v16 responds to those results while
keeping the rule that every target chip must receive a structural change.

| Target | Source path | v16 structural change | Motivation / risk |
| --- | --- | --- | --- |
| International A | `concat_and_cast_mha_k.py` | Add an unmasked full power-of-two contiguous kernel; retain the v15 two-token mapping as the masked fallback. | Preserve A's gain while removing masks on exact blocks; unknown compiler risk. |
| International B | `concat_and_cast_mha_k.py` | Same full-block kernel is tracked separately for B; its v15 gain is not assumed to transfer automatically. | B improved in v15, but full-block code may expose backend differences. |
| Hygon | `concat_and_cast_mha_k_hygon.py` | Replace v15's persistent main path with a new flattened two-row kernel. | Targets the v15 `1.45x` regression without merely restoring the old kernel. |
| MetaX | `concat_and_cast_mha_k_metax.py` | Add a two-token unmasked kernel for even token counts and complete source blocks. | Builds on v15's full-block path with a new token mapping; odd/tail shapes use fallback. |
| Kunlunxin | `concat_and_cast_mha_k_kunlunxin.py` | Replace the failing persistent RoPE path with separate contiguous NoPE-prefix and RoPE-suffix kernels. | Removes the v15 Failed loop structure; two launches may cost latency. |
| Ascend | `concat_and_cast_mha_k_ascend.py` | Add a bounded flat contiguous row-tile kernel for small/medium blocks. | Avoids v15 token-level RoPE reuse that fell to `0.21x`; bounded grid remains. |
| Enflame | `concat_and_cast_mha_k_enflame.py` | Add a token-persistent kernel that loads RoPE once before walking head tiles. | New token-level reuse for compact rows; register pressure may offset reuse. |
| Iluvatar | `concat_and_cast_mha_k_iluvatar.py` | Replace v15's single-token persistent path with a two-token/head-tile kernel. | Targets the v15 `1.37x` regression while retaining explicit RoPE reuse. |

The ZIP contains only the seven `.py` operator files. Local validation covers
Python syntax, AST public-entry/no-native-op checks, token/head mapping, split
prefix/suffix coverage, and ZIP integrity. There is no target Triton backend in
this checkout, so online compilation, correctness and performance remain
unverified. v16 is not submitted online yet.
