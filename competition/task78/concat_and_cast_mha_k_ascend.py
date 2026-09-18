"""Task 78 v19: Ascend token blocks, serial heads and hoisted RoPE loads."""
import torch
import triton
import triton.language as tl


@triton.jit
def _concat_token_blocks_v19(
    out, nope, rope,
    T: tl.constexpr, H: tl.constexpr, DN: tl.constexpr, DR: tl.constexpr,
    NS0: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr,
    RS0: tl.constexpr, RS2: tl.constexpr,
    COMMON: tl.constexpr, WIDE: tl.constexpr,
    BT: tl.constexpr, HS: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
    HEAD_JOBS: tl.constexpr, JOBS: tl.constexpr,
):
    for job in range(tl.program_id(0), JOBS, tl.num_programs(0)):
        # Division is scalar per work tile; no element-wise row reconstruction.
        if WIDE:
            work = job.to(tl.int64)
        else:
            work = job
        first_token = (work // HEAD_JOBS) * BT
        first_head = (work % HEAD_JOBS) * HS
        tokens = first_token + tl.arange(0, BT)
        if DR > 0:
            rc = tl.arange(0, BR)
            if WIDE:
                rc = rc.to(tl.int64)
            rope_mask = (tokens[:, None] < T) & (rc[None, :] < DR)
            rope_value = tl.load(
                rope + tokens[:, None] * RS0 + rc[None, :] * RS2,
                mask=rope_mask, other=0,
            ).to(COMMON)
        # A tile's suffix is loaded once and reused for every serial head.
        # Tensor axes are [tokens, columns], never [tokens, heads, columns].
        for h in tl.static_range(0, HS):
            head = first_head + h
            rows = tokens * H + head
            if DN > 0:
                for first_col in range(0, DN, BN):
                    nc = first_col + tl.arange(0, BN)
                    if WIDE:
                        nc = nc.to(tl.int64)
                    valid = ((tokens[:, None] < T) & (head < H)
                             & (nc[None, :] < DN))
                    value = tl.load(
                        nope + tokens[:, None] * NS0 + head * NS1
                        + nc[None, :] * NS2, mask=valid, other=0,
                    ).to(COMMON)
                    tl.store(out + rows[:, None] * (DN + DR) + nc[None, :],
                             value, mask=valid)
            if DR > 0:
                tl.store(out + rows[:, None] * (DN + DR) + DN + rc[None, :],
                         rope_value, mask=rope_mask & (head < H))


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
    bn = min(512, triton.next_power_of_2(max(1, dn)))
    br = triton.next_power_of_2(max(1, dr))
    bt = min(4, triton.next_power_of_2(tokens),
             max(1, 4096 // max(bn, br)))
    hs = min(4, heads)
    head_jobs = triton.cdiv(heads, hs)
    jobs = triton.cdiv(tokens, bt) * head_jobs
    _concat_token_blocks_v19[(min(32, jobs),)](
        out, k_nope, k_rope, tokens, heads, dn, dr,
        *ns, rs[0], rs[2],
        COMMON=common, WIDE=wide, BT=bt, HS=hs, BN=bn, BR=br,
        HEAD_JOBS=head_jobs, JOBS=jobs, num_warps=4, num_stages=1,
    )
    return out


__all__ = ["concat_and_cast_mha_k"]
