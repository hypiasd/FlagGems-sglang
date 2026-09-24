# v23 validation record

The final seven files passed syntax/public-entry checks, the deterministic
compiler-risk scan, and the restricted CPU semantic model: 189 cases per
backend, 1323 total. The CPU model checks mixed dtype promotion, odd tails,
empty outputs and segments, arbitrary strided/offset/broadcast views, active
pointer bounds, exactly-once output coverage, and input immutability. It is not
a Triton compiler, target-device test, or performance benchmark.

The final gate result was:

| Gate | Result |
| --- | --- |
| Python syntax and public entry | PASS, 7/7 |
| Deterministic compiler-risk scan | PASS, 0 blockers |
| CPU semantic suite | PASS, 7/7 and 0/1323 failures |
| Adversarial read-only sub-agent receipt | PASS, 0 unresolved blockers |
| ZIP integrity and file count | PASS, exactly 7 files |

The sub-agent left three medium-confidence risks for target-runner inspection rather
than silently treating them as solved: dynamic `range` lowering on Ascend,
implicit scalar-mask broadcasting on Enflame, and possible narrow intermediate
column arithmetic on Iluvatar's WIDE path. These are not known failures and
there is no target compiler or device available locally.

| File | SHA-256 |
| --- | --- |
| concat_and_cast_mha_k.py | d7ba238a7b16ee2e36ba4b6256faa7dc67227f7556fdb6f7e3476368ae646f1c |
| concat_and_cast_mha_k_ascend.py | d4dd62430e9ae71f2d84149503d0f5875c8b844688d0a933936af8353e1393fc |
| concat_and_cast_mha_k_enflame.py | d4411cbb7e638c82a3bd9407b972879771be16dbedf1532e6ceabec4a3412224 |
| concat_and_cast_mha_k_hygon.py | eac663c9535eaa0306ad4d8f216a9850e9e1cf77dad7ac44c4bc9d7783d323f5 |
| concat_and_cast_mha_k_iluvatar.py | d9547f19741bb5c750fb312f6bab866baf1852590328e76daed92ece5433b298 |
| concat_and_cast_mha_k_kunlunxin.py | c635dbb0be1aac00810d6bb03259bc862f3fdf638f26b561f3eb0f3841471ad3 |
| concat_and_cast_mha_k_metax.py | 60052fa765ac10922b5989122c18e25bb0fd5e7ead47425585c808722dc707d2 |

Reproduce from the project root:

```sh
python3 competition/task78/kernelgen/run_candidate_gate.py \
  competition/task78/kernelgen/candidates/task78-v23-20260919-195717 \
  --require-review \
  --review-json competition/task78/kernelgen/candidates/task78-v23-20260919-195717/subagent-review.json
unzip -t competition/task78/flagos-task78-v23.zip
```

## Official FlagOS result

FlagOS completed v23 at `09-19 20:41` with `5/8` passing chips. The surviving
scores were Iluvatar `2.19x`, MetaX `1.10x`, Hygon `2.28x`, International A
`1.64x`, and International B `1.75x`. Because three chips failed, FlagOS showed
no average acceleration for this submission.

The failure details were:

| Chip | FlagOS failure |
| --- | --- |
| Enflame | `JITFunction.run() got multiple values for keyword argument 'BR'` in every reported case. |
| Kunlunxin | Numerical mismatch; roughly 76%–82% of elements mismatched in the displayed cases, with very large absolute/relative errors. |
| Ascend | `Config.__init__() got an unexpected keyword argument 'multibuffer'` in every reported case. |

The results confirm that the local CPU model and static scan are useful
pre-submit filters but do not model the target Triton runtime, vendor
configuration schema, or device execution semantics. FlagOS remains the source
of target compilation, correctness, and speedup evidence.

## Post-v23 workflow audit

The local record above reflects the gate that existed before the FlagOS run. After
the reported failures, the gate was strengthened. Re-running v23 under the new
workflow rejects it before packaging because it contains:

- Ascend `triton.Config(..., multibuffer=True)`, outside the portable config
  ABI allowlist;
- Enflame explicit `BR`/`NRC` tile kwargs on an autotuned launch, plus masked
  negative pointer arithmetic and scalar-mask broadcasting;
- Iluvatar autotune `BC` values that vary while the host loop counts stay fixed
  at a divisor of 512;
- Kunlunxin masked negative pointer arithmetic and scalar-mask broadcasting.

The new CPU validator also sweeps every visible autotune config. On the v23
Iluvatar source this independently reports missing output coverage for configs
4 and 5 (the `BC=256` variants), reproducing a concrete failure in the local semantic model.
This does not retroactively change the submitted score; it closes the gap for
future candidates.
