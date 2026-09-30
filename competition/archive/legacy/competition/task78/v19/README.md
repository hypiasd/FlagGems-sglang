# Task 78 v19 — fused output stores and RoPE reuse

Status: submitted and completed with 8/8. Source baseline: v18 at `3749eff`;
v18 result documentation: `954ca8d`.

v18 finished with 6/8: Iluvatar 2.18x, MetaX 1.25x, Enflame Failed,
Hygon 2.12x, Kunlunxin Failed, Ascend 0.02x, International A/B 1.57/1.54x.
Enflame is a NEW failure relative to v17's 0.47x; Kunlunxin failed in both.
No traceback establishes whether either failure is compilation, runtime,
comparison, or timeout. The suspected causes below are not diagnoses.

## Observed v19 result

The official FlagOS record completed v19 at 09-18 22:57 with Iluvatar `2.36x`, MetaX `1.45x`,
Enflame `0.25x`, Hygon `2.45x`, Kunlunxin `0.27x`, Ascend `0.14x`,
International A `1.60x`, and International B `1.51x`. All 8/8 chips passed;
the aggregate was `1.25x`. Compared with v18, Iluvatar, MetaX, Hygon,
Ascend and both generic chips changed by `+0.18/+0.20/+0.33/+0.12/+0.03/-0.03`;
Enflame and Kunlunxin became measurable after their v18 failures. Compared
with the complete v12 result of `1.27x`, v19 is still `0.02x` lower.

The strongest positive evidence is that the new structures restore 8/8 and
raise Iluvatar, MetaX and Hygon. The remaining performance bottlenecks are
Enflame `0.25x`, Ascend `0.14x`, and Kunlunxin `0.27x`; their passing scores
are still below v16's `0.60/0.21/0.30x` in the same chip order for Enflame,
Ascend and Kunlunxin. This is a device observation, not a compiler-root-cause
traceback.

## Per-chip structural changes

Paths are relative to `competition/task78/`. Each nonempty call executes
the named v19 kernel, including noncontiguous inputs; no old-version dispatch
or unobserved large-shape-only gate is used. A shared implementation is
explicitly recorded for each affected chip.

| Chip | File / kernel | New structure versus v18 | Intended benefit | Risk and local evidence |
| --- | --- | --- | --- | --- |
| International A | `concat_and_cast_mha_k.py::_concat_output_tile_v19` | Output-column tile loads both sources with disjoint masks, promotes values, selects them and issues a single contiguous store per head tile. | Removes job-type branch, per-element division and separated output stores; keeps RoPE reuse across heads. | Padded output width and predicated loads may cost registers/occupancy; 189 source-model cases pass, hardware untested. |
| International B | Same file/kernel | Same output-oriented mapping replaces B's dense-prefix/broadcast-suffix job split. | Fewer programs for typical wide-prefix rows; adjacent output segments use one store. | B's compiler is untested independently; A cannot certify B; same 189 model cases. |
| Iluvatar | `concat_and_cast_mha_k_iluvatar.py::_concat_output_tile_v19` | Fuses the two source job classes into head/output-column tiles and one output store. | Removes repeated row reconstruction and scheduling of separate segments. | v18 already reached 2.18x; larger live tile can regress it; 189 model cases. |
| MetaX | `concat_and_cast_mha_k_metax.py::_concat_output_tile_v19` | Replaces hybrid jobs with fused output writes; tile axes are heads and output columns. | Removes program-level branch and separated destination writes. | At wide rows the 4096-element tile cap reduces head reuse from 8 to 4; 189 model cases, no performance claim. |
| Hygon | `concat_and_cast_mha_k_hygon.py::_concat_output_tile_v19` | Removes serial two-token processing and two separate stores per token; one token/head tile writes both segments together. | Removes token-tail control flow and repeated segment stores/column jobs. | Gives up two-token program amortization; 189 model cases. |
| Enflame | `concat_and_cast_mha_k_enflame.py::_concat_serial_heads_v19` | One-dimensional RoPE vector loaded once, reused across up to four statically unrolled heads; bounded NoPE column loop. | Avoids element division, job-type branches and head broadcasting; reduces repeated suffix loads. | Serial heads can reduce parallelism and backend unrolling remains untested; 189 model cases. |
| Kunlunxin | `concat_and_cast_mha_k_kunlunxin.py::_concat_serial_heads_v19` | Same one-dimensional organization with up to two serial heads; prefix and suffix remain independent typed loads/stores in one launch. | Reuses suffix registers while avoiding v18 dynamic segment jobs and v17 three-dimensional broadcasting. | Fused multiple loads/stores and static unrolling still need XPU validation; 189 model cases. |
| Ascend | `concat_and_cast_mha_k_ascend.py::_concat_token_blocks_v19` | Persistent work tiles cover up to four tokens and four serial heads. A [tokens, columns] RoPE tile is loaded outside the head loop. | Amortizes persistent scheduling and reuses suffix values; eliminates per-element row division. | Scalar work-tile division remains; strided token rows and unrolling may lower poorly; 189 model cases, total grid <=32. |

## What the source changes can and cannot establish

For the illustrative shape T=128, H=128, DN=512, DR=64 (not an official
test-shape claim), default/Iluvatar programs fall from 12288 to 4096.
MetaX programs fall from 10240 to 4096, but suffix head reuse decreases.
Enflame and Kunlunxin load each token's suffix once per four/two heads
instead of once per head. Ascend reduces persistent work tiles from 9216
segment jobs to 1024 token/head tiles, with more work inside each tile.
These counts follow from source mapping. They do not predict elapsed time
or prove device-memory traffic after cache effects/compiler lowering.

The GPU design trades independent segment widths for a padded output width.
For DN=512 and DR=64, 576 of 1024 column lanes are active. This is a real
occupancy risk; one store and fewer programs do not guarantee improvement.
The 4096-element head/output tile cap limits, but does not measure, register use.

The Enflame/XPU organization is not a restoration of the earlier two-kernel
copy path: it fuses launches and explicitly reuses a suffix vector across
serial heads without creating a multidimensional load/store tensor.
Ascend's axes are tokens/columns rather than the earlier heads/columns or
flat row batches; suffix values remain live across the unrolled head loop.

## Semantics and boundaries

- Exactly the public entry `concat_and_cast_mha_k(k, k_nope, k_rope)`.
- Contiguous newly allocated output; inputs are not modified.
- Source strides and view storage offsets are respected.
- Each source is converted to `torch.promote_types`' common dtype before
  destination conversion, matching cat-then-cast semantics.
- Zero output sizes launch nothing; zero-length segments and partial tiles
  are masked. Each nonempty call launches once.
- Native-width indices for ordinary shapes; `WIDE` promotes program/column
  indices before address multiplication for large relative offsets.
  CPU tests use int64 internally and cannot establish actual 32-bit lowering.
- Torch is used for allocation, dtype promotion and tensor metadata only.
- No device-dependent score prediction or exception fallback. The v19 package
  was submitted by the user; the result above is from the FlagOS result page.

## Validation and handoff

See [validation.md](validation.md) for exact tested source hashes and limits.
The local environment has no target Triton compiler or competition device.
The checks establish source-level indexing and packaging only.

Submission: `competition/task78/flagos-task78-v19.zip`, seven operator files
at ZIP root. Historical ZIPs remain untouched.
