"""Task 78 v19: Iluvatar, output-aligned fused stores with head reuse."""
import torch
import triton
import triton.language as tl


@triton.jit
def _concat_output_tile_v19(
    out, nope, rope,
    T: tl.constexpr, H: tl.constexpr, DN: tl.constexpr, DR: tl.constexpr,
    NS0: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr,
    RS0: tl.constexpr, RS2: tl.constexpr,
    COMMON: tl.constexpr, WIDE: tl.constexpr,
    BH: tl.constexpr, BC: tl.constexpr,
):
    token = tl.program_id(0)
    if WIDE:
        token = token.to(tl.int64)
    head_job = tl.program_id(1)
    col_job = tl.program_id(2)
    if WIDE:
        head_job = head_job.to(tl.int64)
        col_job = col_job.to(tl.int64)
    heads = head_job * BH + tl.arange(0, BH)
    cols = col_job * BC + tl.arange(0, BC)
    valid = (heads[:, None] < H) & (cols[None, :] < DN + DR)
    # Clamp negative inactive suffix offsets; merge values after promotion.
    nc = cols
    rc = tl.maximum(cols - DN, 0)
    prefix = tl.load(
        nope + token * NS0 + heads[:, None] * NS1 + nc[None, :] * NS2,
        mask=(heads[:, None] < H) & (cols[None, :] < DN), other=0,
    ).to(COMMON)
    suffix = tl.load(
        rope + token * RS0 + rc * RS2,
        mask=(cols >= DN) & (cols < DN + DR), other=0,
    ).to(COMMON)
    value = tl.where(cols[None, :] < DN, prefix, suffix[None, :])
    # Adjacent prefix and suffix lanes share one store instruction.
    tl.store(out + (token * H + heads[:, None]) * (DN + DR) + cols[None, :],
             value, mask=valid)


def concat_and_cast_mha_k(k, k_nope, k_rope):
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device,
                      memory_format=torch.contiguous_format)
    tokens, heads, total_dim = k.shape
    if tokens == 0 or heads == 0 or total_dim == 0:
        return out
    dn, dr = k_nope.shape[2], k_rope.shape[2]
    common = getattr(tl, str(torch.promote_types(k_nope.dtype, k_rope.dtype)).split(".")[-1])
    # Use native-width indices on ordinary tensors, widen before multiplication
    # when any relative address needs more than signed 32 bits.
    ns, rs = k_nope.stride(), k_rope.stride()
    span = max(tokens * heads * total_dim,
               (tokens - 1) * ns[0] + (heads - 1) * ns[1] + max(dn - 1, 0) * ns[2] + 1,
               (tokens - 1) * rs[0] + max(dr - 1, 0) * rs[2] + 1)
    wide = span >= 2**31
    bh = min(4, triton.next_power_of_2(heads))
    bc = min(1024, triton.next_power_of_2(total_dim))
    bh = min(bh, max(1, 4096 // bc))
    _concat_output_tile_v19[(tokens, triton.cdiv(heads, bh),
                             triton.cdiv(total_dim, bc))](
        out, k_nope, k_rope, tokens, heads, dn, dr,
        *ns, rs[0], rs[2],
        COMMON=common, WIDE=wide, BH=bh, BC=bc,
        num_warps=4, num_stages=1,
    )
    return out


__all__ = ["concat_and_cast_mha_k"]
