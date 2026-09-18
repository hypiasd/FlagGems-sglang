"""Ascend row-tiled copy/cast candidate with a bounded persistent grid."""
import torch
import triton
import triton.language as tl


@triton.jit
def _concat_rows(
    out, nope, rope, ROWS: tl.constexpr, H: tl.constexpr,
    DN: tl.constexpr, DR: tl.constexpr,
    NS0: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr,
    RS0: tl.constexpr, RS2: tl.constexpr,
    COMMON: tl.constexpr, BM: tl.constexpr,
    BN: tl.constexpr, BR: tl.constexpr,
):
    for first in range(tl.program_id(0) * BM, ROWS,
                       tl.num_programs(0) * BM):
        rows = first + tl.arange(0, BM)
        token = rows // H
        head = rows % H
        dst = rows * (DN + DR)
        if DN > 0:
            cols = tl.arange(0, BN)
            mask = (rows[:, None] < ROWS) & (cols[None, :] < DN)
            value = tl.load(
                nope + token[:, None] * NS0 + head[:, None] * NS1
                + cols[None, :] * NS2, mask, other=0,
            ).to(COMMON)
            tl.store(out + dst[:, None] + cols[None, :], value, mask)
        if DR > 0:
            cols = tl.arange(0, BR)
            mask = (rows[:, None] < ROWS) & (cols[None, :] < DR)
            value = tl.load(
                rope + token[:, None] * RS0 + cols[None, :] * RS2,
                mask, other=0,
            ).to(COMMON)
            tl.store(out + dst[:, None] + DN + cols[None, :], value, mask)


@triton.jit
def _concat_tokens_contiguous(
    out, nope, rope, TOKENS: tl.constexpr, H: tl.constexpr,
    DN: tl.constexpr, DR: tl.constexpr,
    COMMON: tl.constexpr, HEAD_TILE: tl.constexpr,
    BN: tl.constexpr, BR: tl.constexpr,
):
    """Persistent token/head-tile path with one RoPE load per head tile."""
    for token in range(tl.program_id(0), TOKENS, tl.num_programs(0)):
        for first_head in range(0, H, HEAD_TILE):
            heads = first_head + tl.arange(0, HEAD_TILE)
            head_mask = heads < H
            rows = token * H + heads
            dst = rows * (DN + DR)
            if DN > 0:
                cols = tl.arange(0, BN)
                mask = head_mask[:, None] & (cols[None, :] < DN)
                value = tl.load(
                    nope + rows[:, None] * DN + cols[None, :],
                    mask, other=0,
                ).to(COMMON)
                tl.store(out + dst[:, None] + cols[None, :], value, mask)
            if DR > 0:
                cols = tl.arange(0, BR)
                mask = head_mask[:, None] & (cols[None, :] < DR)
                value = tl.load(
                    rope + token * DR + cols[None, :],
                    mask, other=0,
                ).to(COMMON)
                tl.store(out + dst[:, None] + DN + cols[None, :], value, mask)


def concat_and_cast_mha_k(k, k_nope, k_rope):
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if k.numel() == 0:
        return out
    tokens = k.shape[0]
    rows = k.shape[0] * k.shape[1]
    dn, dr = k_nope.shape[2], k_rope.shape[2]
    bn, br = triton.next_power_of_2(max(1, dn)), triton.next_power_of_2(max(1, dr))
    # Bound temporary tile size while amortizing per-row address arithmetic.
    bm = max(1, min(16, 4096 // max(bn, br)))
    common = getattr(tl, str(torch.promote_types(k_nope.dtype, k_rope.dtype)).split('.')[-1])
    grid = (min(32, triton.cdiv(rows, bm)),)
    if k_nope.is_contiguous() and k_rope.is_contiguous():
        _concat_tokens_contiguous[(min(32, tokens),)](
            out, k_nope, k_rope, tokens, k.shape[1], dn, dr,
            COMMON=common, HEAD_TILE=4, BN=bn, BR=br, num_warps=4,
        )
    else:
        _concat_rows[grid](
            out, k_nope, k_rope, rows, k.shape[1], dn, dr,
            *k_nope.stride(), k_rope.stride(0), k_rope.stride(2),
            COMMON=common, BM=bm, BN=bn, BR=br, num_warps=4,
        )
    return out


__all__ = ['concat_and_cast_mha_k']
