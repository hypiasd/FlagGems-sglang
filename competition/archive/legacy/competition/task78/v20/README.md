# Task 78 v20 — KernelGen structural draft

Status: generated from each v19 backend by KernelGen; not submitted and not hardware-validated.

This directory is an isolated draft. The root submission files remain v19. Each file keeps the public entry concat_and_cast_mha_k(k, k_nope, k_rope) and introduces a structural change from v19:

| Backend | v20 structure | KernelGen target |
| --- | --- | --- |
| International A/B | Flat output-index mapping with coalesced stores and autotuned blocks | NVIDIA |
| Ascend | Flat persistent 1-D traversal with bounded grid and safe masked offsets | Huawei |
| Enflame | 2-D token/head tile with one RoPE load reused across heads | Enflame |
| Hygon | Flat output-index mapping with autotuned streaming blocks | Hygon |
| Iluvatar | Flat output-index mapping with coalesced stores and fixed block | Iluvatar/Tianshu |
| Kunlunxin | Flat coalesced mapping with clamped inactive RoPE offsets | Kunlun |
| MetaX | Flat output-index mapping with autotuned streaming blocks | MetaX/Muxi |

Local checks performed before assembling this draft: Python syntax, public symbol, fallback scan, and comparison against the v19 source. The repository CPU validator still needs a candidate-path adapter because some generated files use backend-specific Triton keyword arguments. No speedup is recorded: optimize_kernel returns rewritten code but does not execute benchmark. The next gate is candidate-specific semantic validation followed by one official FlagOS submission through Chrome.
