# Task 78 v18 — single-launch segment scheduling

Status: submitted to FlagOS and completed with 6/8 chips. Base: v17 commit
`ec5b273`. All nonempty inputs in each of the seven submission files execute
a v18 kernel, including strided inputs. No shape gate restricts the new
implementation to an unobserved wide-row subset.

## Observed v17 result

Read from the logged-in [FlagOS Task78 submission page](https://flagos.io/race-detail-season2?id=782kzq4m&lang=cn)
on 2026-09-18. Submission time: 21:19 Beijing time. Status: completed.

| Target | v16 | v17 | Difference |
| --- | ---: | ---: | ---: |
| Iluvatar | 1.86 | 1.86 | 0.00 |
| MetaX | 1.47 | 1.49 | +0.02 |
| Enflame | 0.60 | 0.47 | -0.13 |
| Hygon | 1.81 | 2.50 | +0.69 |
| Kunlunxin | 0.30 | Failed | correctness regression |
| Ascend | 0.21 | 0.20 | -0.01 |
| International A | 1.62 | 0.85 | -0.77 |
| International B | 1.53 | 1.08 | -0.45 |
| Passed / average | 8/8; 1.18x | 7/8; unavailable | invalid aggregate |

The detail page showed a leading score of 1.90x and our best 1.27x. These are
observations at this check, not permanent leaderboard facts.

## Observed v18 result

The completed v18 submission returned: Iluvatar `2.18x`, MetaX `1.25x`,
Enflame `Failed`, Hygon `2.12x`, Kunlunxin `Failed`, Ascend `0.02x`,
International A `1.57x`, and International B `1.54x`. Only 6/8 chips passed,
so the aggregate was invalid. The result page provided no failing-case
traceback. The main structural suspect is the segment-job/persistent-grid
organization on Ascend, Enflame, and Kunlunxin; this is a hypothesis, not a
confirmed compiler root cause.

The table does not reveal failing case shapes, latencies or a traceback.
In particular, Kunlunxin's 3-D path is a suspect, not an established root
cause. The generic split added one launch and removed head reuse; these
are plausible contributors to the A/B regression, not isolated measurements.
v17's Ascend grid cap only applied to one dimension:
`min(32, ceil(T/2)) * ceil(H/4)` could greatly exceed 32 total programs.

## Per-target structural changes and risks

Paths below are relative to `competition/task78/`. A/B deliberately share
one file but are tracked separately. All rows have the same local evidence:
189 direct-source CPU-model cases per file, exact numerical comparisons,
bounds/unique-write/input-immutability checks, and one launch per nonempty case.
Device results exist, but the local CPU-model checks below do not validate
target-device compilation, correctness, or performance.

| Target | Code / entry | Structural change from v17 | Expected benefit | Main risk | Evidence |
| --- | --- | --- | --- | --- | --- |
| International A | `concat_and_cast_mha_k.py::_concat_hybrid_jobs_v18` | Dense 1-D NoPE jobs and 2-D broadcast RoPE jobs share one launch, including irregular rows and strides. | Remove the second compact-row launch, fill tiles across rows, reuse suffix vectors across heads. | Uniform branch overhead, gapped stores and division for non-power-of-two dimensions. | 189 CPU-model cases; 181 nonempty launches hit the new kernel. |
| International B | Same file/entry | Same scheduling change runs on all nonempty B inputs; no vendor identity is guessed. | Recover launch amortization while limiting vectors to at most 2 dimensions. | B's compiler may lower the uniform branches differently; A's result cannot certify B. | Same 189 source-model cases; B compilation/performance untested. |
| Iluvatar | `concat_and_cast_mha_k_iluvatar.py::_concat_hybrid_jobs_v18` | Replace token-pair/head loops and wide-only split with dense prefix + broadcast suffix job mapping at all sizes. | Parallelize NoPE across token/head boundaries and bound RoPE column tiles. | Some small shapes have more programs than the old token-pair path. | 189 CPU-model cases; no untested width gate. |
| MetaX | `concat_and_cast_mha_k_metax.py::_concat_hybrid_jobs_v18` | Replace full-block 3-D pairs/wide-only split with dense prefix + 2-D suffix jobs. | Bound per-program data footprint independently of entire source row width. | Extra scheduling/branch work may outweigh reduced live state. | 189 CPU-model cases, including 512/513 and 1024/1025 boundaries. |
| Enflame | `concat_and_cast_mha_k_enflame.py::_concat_segment_jobs_v18` | Pack both segments into separate 1-D jobs within a single launch; remove token/head 3-D broadcasting. | Regular bounded vectors and full tiles across short rows; no token-persistent nesting. | Integer division and explicit repeated suffix loads across heads can cost throughput. | 189 CPU-model cases; one new kernel per nonempty call. |
| Kunlunxin | `concat_and_cast_mha_k_kunlunxin.py::_concat_segment_jobs_v18` | Combine independent 1-D prefix/suffix jobs in one launch; no persistent loop or 3-D tensor. | Retain simple copy/cast operations while avoiding v16's two launches and v17's broadcast shape. | Program-level branching/64-bit address arithmetic still requires XPU compilation; failure cause remains unknown. | 189 CPU-model cases; all active addresses bounded and each output written once. |
| Ascend | `concat_and_cast_mha_k_ascend.py::_concat_segment_jobs_v18` | One-dimensional persistent grid traverses packed segment jobs, capped at 32 total programs. | Fix total-grid growth and bound temporary storage to one segment block. | Persistent branch lowering and repeated RoPE reads may limit speed. | 189 CPU-model cases; observed maximum grid product 32; multi-iteration cases exercised. |
| Hygon | `concat_and_cast_mha_k_hygon.py::_concat_serial_pair_v18` | Statically process two tokens in sequence using 2-D head/column tiles; split wide columns into bounded subtiles. | Keep two-token program amortization while reducing source-level simultaneous tensor rank and wide-row tile size. | Compiler may overlap live ranges; serial processing may regress v17's 2.50x. | 189 CPU-model cases; odd tokens, head tails and multi-column tiles exercised. |

These are optimization hypotheses. Structural novelty and local correctness
do not guarantee positive performance changes on every chip.

## Semantics and implementation

For dense prefix index `i`, `row=i//DN`, `col=i%DN`, output address is
`row*(DN+DR)+col`. For suffix index `i`, `row=i//DR`, `col=i%DR`,
source token is `row//H` and output address is `row*(DN+DR)+DN+col`.
Separate compile-time guards handle zero-length segments before division.
The hybrid suffix instead loads a single RoPE column vector and broadcasts
only on the 2-D store. Prefix/suffix programs write disjoint element ranges.

Strided inputs use their supplied strides and base pointers. All variants
first cast to `torch.promote_types(k_nope.dtype, k_rope.dtype)`, then let the
output store cast to `k.dtype`, preserving the reference cat-then-cast order.
Output allocation is fresh and contiguous; `k` supplies shape/dtype/device
and is not read for values or modified. No native Torch computational fallback,
input-value dispatch, cached output, environment probe or timing probe is used.

Old experimental kernel definitions were removed from the live submission
files; historical ZIPs and Git commits retain their sources.

## Reproducible local validation

```sh
python3 competition/task78/validate_cpu.py --all
python3 -m py_compile competition/task78/concat_and_cast_mha_k*.py
git diff --check
unzip -t competition/task78/flagos-task78-v18.zip
```

The CPU model executes the actual wrappers and Python kernel bodies with a
restricted emulation of arange/load/store/program IDs. It does not load
Triton, generate device IR, or measure accelerator time. It checks all 27
source/output fp16/bf16/fp32 combinations, arbitrary positive/zero strides,
nonzero storage offsets, empty sources, partial blocks and bounded-grid loops.
It also tests itself against deliberately bad addresses, duplicate/missing
writes and input mutations. Numerical equality uses zero tolerance and equal
NaNs; signed-zero bits and NaN payloads are not certified.

Result: 189/189 per file, 1,323/1,323 total. Each backend had 181 nonempty
single-launch cases and eight empty cases with no launch. See
[validation results](validation.md).

Neither this checkout's tests nor upstream master `d901fe8` contained a
Task78/concat_and_cast_mha_k test file. These are self-built validation cases,
not the competition's official or hidden tests. Target Triton compilation,
device exactness and speed still require platform testing.

The submission ZIP contains only the seven operator files, byte-for-byte
equal to the reviewed working-tree sources. This README and the CPU validator
are deliberately excluded from the ZIP. The user performs the online upload.
