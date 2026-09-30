# Task 78 v15 experiment

v15 is a structural-optimization candidate created under the new admission
rule: every target chip must receive a new data-flow, mapping, memory-access,
or kernel-organization change. Parameter-only changes and simple rollback are
not counted.

| Target | Source path | New structural change | Main risk |
| --- | --- | --- | --- |
| International A | `concat_and_cast_mha_k.py` | Two adjacent tokens are handled by one contiguous program, with independent RoPE loads and a 3-D token/head/column mapping. | Extra live dimensions can hurt small backends or tail masks. |
| International B | `concat_and_cast_mha_k.py` | Same token-pair kernel as A; the shared path is recorded separately because the two unknown backends may compile the mapping differently. | The v13 generic path already regressed B; compatibility must be rechecked. |
| Hygon | `concat_and_cast_mha_k_hygon.py` | A persistent token loop replaces the 2-D token/head launch for medium contiguous rows. | Persistent looping may reduce occupancy when token count is small. |
| MetaX | `concat_and_cast_mha_k_metax.py` | Power-of-two full rows use an unmasked contiguous kernel; masked fallback remains for tails and irregular dimensions. | Unmasked path requires exact full blocks and may increase register use. |
| Kunlunxin | `concat_and_cast_mha_k_kunlunxin.py` | A token-persistent kernel loads one RoPE vector once and reuses it across head tiles. | Reuse can increase live state; XPU loop lowering must remain compatible. |
| Ascend | `concat_and_cast_mha_k_ascend.py` | Tiny contiguous rows use a token-persistent kernel with one RoPE load outside the head loop. | Persistent token/head nesting may conflict with UB pressure. |
| Enflame | `concat_and_cast_mha_k_enflame.py` | Wide contiguous rows are split into independent NoPE-prefix and RoPE-suffix kernels, reducing per-kernel live state. | Two launches may outweigh register savings on short workloads. |
| Iluvatar | `concat_and_cast_mha_k_iluvatar.py` | Small contiguous rows use a token-persistent kernel that reuses one RoPE load across all head tiles. | The new persistent loop may lose to the existing row-batch path. |

The ZIP contains only the seven `.py` operator files. Local validation covers
Python syntax, AST public-entry/no-native-op checks, shape mapping and ZIP
integrity. This checkout has no target Triton backends, so compilation,
correctness and performance still require online evaluation. v15 is not
submitted online yet.
