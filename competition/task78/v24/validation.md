# v24 validation record

This record separates local evidence from the subsequent official Arc result.
The local gates below do not imply target compilation, correctness, or
performance; Arc evaluated the packaged source on 2026-09-20.

## Gate results

| Gate | Result |
| --- | --- |
| Python syntax and public entry | PASS, 7/7 |
| Deterministic compiler-risk scan | PASS, 0 blockers, 0 cache-hint warnings |
| CPU semantic suite | PASS, 7/7 backends and 0/1407 case-executions (201 cases per backend, autotune sweep on) |
| Adversarial read-only review (second round) | PASS, 0 blockers, 0 novel findings; all four first-round blockers adjudicated closed |
| ZIP integrity and file count | PASS, exactly 7 files, contents hash-identical to the reviewed sources |

The CPU model checks mixed dtype promotion, odd tails, empty outputs and
segments, arbitrary strided/offset/broadcast views, active pointer bounds,
exactly-once output coverage, and input immutability. It is not a Triton
compiler, target-device test, or performance benchmark.

## Source hashes

| File | SHA-256 |
| --- | --- |
| concat_and_cast_mha_k.py | 938926df4770d76154dd5c4b57756d900ce8930cb7cc09e14a6e94901b472312 |
| concat_and_cast_mha_k_ascend.py | 77927979ae69abde7efe0042c4e8c27a9ecec82a4d9af08ad119cee430cd072f |
| concat_and_cast_mha_k_enflame.py | fa649bbffebf9174c42f573695b890d3cbfde9107f7227980df122825da9887e |
| concat_and_cast_mha_k_hygon.py | 949e39cc4d6dc997857a1c6576942b4f0c04e1fd4ad0376cd304389a07f1e7fa |
| concat_and_cast_mha_k_iluvatar.py | 22acbfa65bede39b79b7602788d093ae85d2da62d6f4250fee2d47480f4c5417 |
| concat_and_cast_mha_k_kunlunxin.py | 2f7f8a9b82eec6cd5eca85ea4a5a43b29f3612e3b36738938ba5227e79b4dad1 |
| concat_and_cast_mha_k_metax.py | 8641369bae965911cfdceec1122df4a7c1e0bb58afaa45beca75dcd4bd0afad9 |

The recorded v24 result predates workflow v4 and used the former one-receipt
review protocol. It also predates the v5 mandatory target preflight. Its
archived `local-gate.json` and review artifacts remain the historical record;
they do not satisfy the current review or target-evidence gate. Re-running v24
through today's gate requires fresh blind and reconciliation agents, plus a
trusted target run bound to current source/baseline hashes. The Arc outcome
below is unchanged.

Reproduce the non-review portions from the project root:

```sh
python3 competition/task78/kernelgen/review_candidate.py \
  competition/task78/kernelgen/candidates/task78-v24-20260920-000454 \
unzip -t competition/task78/flagos-task78-v24.zip
```

## Repair log for this candidate

1. **Column coverage.** The first draft covered each row with a single capped
   tile and no column loop, so every column past the 4096 cap was silently
   dropped. A direct probe with the repository CPU model showed the v19 baseline
   passing 28/28 backend-case pairs while the draft failed 13/28 (for example
   `missing=12` columns at `dn=4097` and `missing=24576` at `dn=dr=4096`). All
   seven files received the compile-time column loops the v19 baseline already
   used.
2. **Validator gap.** `validate_cpu.py` gained twelve `wide-dim-*` cases
   (NoPE/RoPE dimensions at and past the 4096 cap, contiguous and strided
   layouts), taking the suite from 189 to 201 cases per backend; the v19
   baseline was re-verified at 201/201 against it.
3. **First adversarial round (fail).** Ascend: int32 row arithmetic with the
   mask computed before widening, and `tl.static_range` unrolled over a
   shape-derived trip count. Hygon: uncapped 2-D tile product (up to 16x4096
   elements) and a broadcast-address 2-D rope load plus 2-D cast inside the
   lowering area that failed on Hygon in v22.
4. **Repair.** Ascend now widens the block index before multiplying and walks
   row blocks with the grid-stride loop v19 was accepted with; Hygon uses the
   1-D row form with no broadcast anywhere; all seven files launch with a
   1-tuple grid. Static scan 0 blockers, CPU 201/201 on all seven backends.
5. **Second adversarial round (pass).** All four blockers confirmed closed and
   re-adjudicated; no new blocker found. The first-round receipt is kept beside
   the shipped one as `subagent-review-first-round.json`, and the rejected gate
   run as `local-gate-first-round.json`, in the candidate directory.

## Review notes before Arc

None of these was judged a blocker by the second review; they are recorded so
an Arc failure can be triaged quickly.

- **Ascend `num_warps=8`** (line 82) is the first non-4 `num_warps` on an
  Arc-tested Ascend launch (v19–v23 all used 4). If Ascend reports a launch or
  UB rejection, reverting to 4 is the direct counterfactual. A loud failure, not
  a silent numeric one.
- **`num_stages=2`** on the generic, Enflame, Iluvatar and Kunlunxin files, where
  v19 used 1. Unverified vendor pipelining on a loop body; low confidence.
- **`WIDE`-only index widths.** Under `span >= 2**31` the column/rope terms stay
  int32 in most files, inherited from the v19 root generic. This needs an
  ~8 GiB extreme-stride view to activate and is compile-time pruned (the `WIDE`
  branch is constexpr) for every shape this competition can build.
- **Ascend loop-variable rebinding.** The grid-stride induction variable is
  rebound (`block = block.to(tl.int64)`) rather than copied to a new name as in
  v19; the branch is constexpr-pruned unless `WIDE` is true, so it cannot reach
  an Arc shape.

## Arc result (2026-09-20 21:31)

The package completed with 7/8 targets. Arc reported no aggregate speedup,
because Enflame failed. The per-target results were:

| Target | Result |
| --- | ---: |
| 天数智芯 / Iluvatar | 1.74× |
| 沐曦 / MetaX | 0.89× |
| 燧原 / Enflame | Failed |
| 海光 / Hygon | 1.26× |
| 昆仑芯 / Kunlunxin | 0.31× |
| 华为 / Ascend | 0.21× |
| International A | 1.57× |
| International B | 1.71× |

Enflame Case 1 failed at launch: `grid.x` required 131072, while the reported
hardware limit was 65535. This is the confirmed v24 blocker. v24 recovered
Kunlunxin and Ascend from earlier failed submissions, but both are far below
1×; Iluvatar, MetaX, Hygon, and International A/B also scored below their v23
results. This run therefore does not replace the best valid aggregate (v12,
1.27×) or the known complete v19 reference (8/8, 1.25×).
