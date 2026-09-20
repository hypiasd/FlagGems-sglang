# Task 78 v24 — coverage-repaired per-backend candidate, locally gated

Status: locally gated (Stages 3–5) and packaged; **not yet submitted to Arc**.
The package is [`../flagos-task78-v24.zip`](../flagos-task78-v24.zip), holding
exactly the seven `concat_and_cast_mha_k*.py` files at ZIP root. The root
submission files in `competition/task78/` remain the Arc-proven **v19** set
(8/8, avg 1.25x) until a new candidate is accepted; the ZIP is the artifact to
upload.

Goal: raise the three sub-1x chips of v19 (Ascend 0.14x, Enflame 0.25x,
Kunlunxin 0.27x) with portable structural changes, keep the strong chips
(Iluvatar 2.36x, Hygon 2.45x, MetaX 1.45x, International A/B 1.60/1.51x) at
least as good, and avoid the v22/v23 failure classes: autotuned duplicate tile
constexpr kwargs, non-portable `triton.Config` options, masked negative pointer
arithmetic, scalar/vector mask mixing, and `num_warps=12`.

Structural changes per backend:

- International A/B (generic), Iluvatar, MetaX: flat `token*head` row index with
  one contiguous row per program, contiguous fused store, capped power-of-two
  tiles and a compile-time column loop.
- Hygon: the same 1-D row form, chosen so that no broadcast and no 2-D tile
  reaches a load, a cast or a store on the backend whose suffix lowering failed
  in v22.
- Enflame, Kunlunxin: one program per `(token, head)` for full parallelism, with
  separate capped NoPE and RoPE column loops of 1-D masks and non-negative
  offsets.
- Ascend: row blocks walked by a 32-program grid-stride loop, with capped NoPE
  (512) and RoPE (4096) column loops and a bounded tile product.

Generation came from KernelGen `optimize_kernel`; two documented agent-applied
mechanical passes (in the candidate's `prompt.md`, gate-hygiene items 1–6) were
required to satisfy the deterministic scan and the CPU semantic suite. The
Stage 5 adversarial read-only review ran twice: the first receipt returned
`fail` on Ascend and Hygon with four blockers, and all four were closed before
this package was built.

See [validation.md](validation.md) for hashes, the full gate table, the repair
log and the residual risks left for Arc.
