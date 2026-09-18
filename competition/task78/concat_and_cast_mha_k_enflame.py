"""Task 78 v19: Enflame, serial heads sharing one 1-D RoPE vector."""
import torch
import triton
import triton.language as tl


@triton.jit
def _concat_serial_heads_v19(
    out, nope, rope,
    T: tl.constexpr, H: tl.constexpr, DN: tl.constexpr, DR: tl.constexpr,
    NS0: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr,
    RS0: tl.constexpr, RS2: tl.constexpr,
    COMMON: tl.constexpr, WIDE: tl.constexpr,
    HS: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
):
    token = tl.program_id(0)
    head_job = tl.program_id(1)
    if WIDE:
        token = token.to(tl.int64)
        head_job = head_job.to(tl.int64)
    first_head = head_job * HS
    if DR > 0:
        rc = tl.arange(0, BR)
        if WIDE:
            rc = rc.to(tl.int64)
        rope_value = tl.load(rope + token * RS0 + rc * RS2,
                             mask=rc < DR, other=0).to(COMMON)
    # Only 1-D vectors reach load/store: no head broadcast, job-type branch,
    # element division/modulo, or persistent loop. Reuse the suffix registers.
    for h in tl.static_range(0, HS):
        head = first_head + h
        dst = (token * H + head) * (DN + DR)
        if DN > 0:
            for first_col in range(0, DN, BN):
                nc = first_col + tl.arange(0, BN)
                if WIDE:
                    nc = nc.to(tl.int64)
                valid = (head < H) & (nc < DN)
                value = tl.load(nope + token * NS0 + head * NS1 + nc * NS2,
                                mask=valid, other=0).to(COMMON)
                tl.store(out + dst + nc, value, mask=valid)
        if DR > 0:
            tl.store(out + dst + DN + rc, rope_value,
                     mask=(head < H) & (rc < DR))


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
    hs = min(4, heads)
    bn = min(512, triton.next_power_of_2(max(1, dn)))
    br = triton.next_power_of_2(max(1, dr))
    _concat_serial_heads_v19[(tokens, triton.cdiv(heads, hs))](
        out, k_nope, k_rope, tokens, heads, dn, dr,
        *ns, rs[0], rs[2],
        COMMON=common, WIDE=wide, HS=hs, BN=bn, BR=br,
        num_warps=1, num_stages=1,
    )
    return out


__all__ = ["concat_and_cast_mha_k"]
