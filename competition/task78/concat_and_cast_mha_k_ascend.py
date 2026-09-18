"""Ascend row-tiled copy/cast candidate with a bounded persistent grid."""
import torch
import triton
import triton.language as tl


@triton.jit
def _concat_token_pair_contiguous_v17(
    out, nope, rope, TOKENS: tl.constexpr, H: tl.constexpr,
    DN: tl.constexpr, DR: tl.constexpr,
    COMMON: tl.constexpr, HEAD_TILE: tl.constexpr,
    BN: tl.constexpr, BR: tl.constexpr,
):
    """Bounded two-token/head-tile path for compact Ascend rows."""
    for pair in range(
        tl.program_id(0), (TOKENS + 1) // 2, tl.num_programs(0)
    ):
        tokens = pair * 2 + tl.arange(0, 2)
        token_mask = tokens < TOKENS
        heads = tl.program_id(1) * HEAD_TILE + tl.arange(0, HEAD_TILE)
        head_mask = heads < H
        rows = tokens[:, None] * H + heads[None, :]
        dst = out + rows[:, :, None] * (DN + DR)

        if DN > 0:
            cols = tl.arange(0, BN)
            mask = token_mask[:, None, None] & head_mask[None, :, None]
            mask = mask & (cols[None, None, :] < DN)
            value = tl.load(
                nope + rows[:, :, None] * DN + cols[None, None, :],
                mask=mask,
                other=0,
            ).to(COMMON)
            tl.store(dst + cols[None, None, :], value, mask=mask)

        if DR > 0:
            cols = tl.arange(0, BR)
            rope_mask = token_mask[:, None] & (cols[None, :] < DR)
            value = tl.load(
                rope + tokens[:, None] * DR + cols[None, :],
                mask=rope_mask,
                other=0,
            ).to(COMMON)
            mask = token_mask[:, None, None] & head_mask[None, :, None]
            mask = mask & (cols[None, None, :] < DR)
            tl.store(
                dst + DN + cols[None, None, :],
                value[:, None, :],
                mask=mask,
            )


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
def _concat_rows_contiguous_v16(
    out, nope, rope, ROWS: tl.constexpr, H: tl.constexpr,
    DN: tl.constexpr, DR: tl.constexpr,
    COMMON: tl.constexpr, BM: tl.constexpr,
    BN: tl.constexpr, BR: tl.constexpr,
):
    """Bounded contiguous row tiles without a persistent token loop."""
    for first in range(tl.program_id(0) * BM, ROWS,
                       tl.num_programs(0) * BM):
        rows = first + tl.arange(0, BM)
        token = rows // H
        dst = rows * (DN + DR)
        row_mask = rows < ROWS
        if DN > 0:
            cols = tl.arange(0, BN)
            mask = row_mask[:, None] & (cols[None, :] < DN)
            value = tl.load(
                nope + rows[:, None] * DN + cols[None, :],
                mask,
                other=0,
            ).to(COMMON)
            tl.store(out + dst[:, None] + cols[None, :], value, mask)
        if DR > 0:
            cols = tl.arange(0, BR)
            mask = row_mask[:, None] & (cols[None, :] < DR)
            value = tl.load(
                rope + token[:, None] * DR + cols[None, :],
                mask,
                other=0,
            ).to(COMMON)
            tl.store(out + dst[:, None] + DN + cols[None, :], value, mask)


@triton.jit
def _concat_tokens_reuse_rope(
    out, nope, rope, TOKENS: tl.constexpr, H: tl.constexpr,
    DN: tl.constexpr, DR: tl.constexpr,
    COMMON: tl.constexpr, HEAD_TILE: tl.constexpr,
    BN: tl.constexpr, BR: tl.constexpr,
):
    """Persistent Ascend path that loads each token's RoPE once."""
    for token in range(tl.program_id(0), TOKENS, tl.num_programs(0)):
        rope_cols = tl.arange(0, BR)
        rope_mask = rope_cols < DR
        rope_value = tl.load(
            rope + token * DR + rope_cols,
            mask=rope_mask,
            other=0,
        ).to(COMMON)
        for first_head in range(0, H, HEAD_TILE):
            heads = first_head + tl.arange(0, HEAD_TILE)
            head_mask = heads < H
            rows = token * H + heads
            dst = out + rows[:, None] * (DN + DR)
            if DN > 0:
                cols = tl.arange(0, BN)
                mask = head_mask[:, None] & (cols[None, :] < DN)
                value = tl.load(
                    nope + rows[:, None] * DN + cols[None, :],
                    mask,
                    other=0,
                ).to(COMMON)
                tl.store(dst + cols[None, :], value, mask)
            if DR > 0:
                tl.store(
                    dst + DN + rope_cols[None, :],
                    rope_value[None, :],
                    head_mask[:, None] & rope_mask[None, :],
                )


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
                rope_mask = cols < DR
                value = tl.load(
                    rope + token * DR + cols,
                    rope_mask, other=0,
                ).to(COMMON)
                tl.store(
                    out + dst[:, None] + DN + cols[None, :],
                    value[None, :],
                    head_mask[:, None] & rope_mask[None, :],
                )


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
        # Small source tiles can afford a wider head tile and amortize the
        # persistent token-loop overhead. v16's tiny/medium path uses bounded
        # flat row tiles; larger rows retain the token loop and 32-program
        # bound.
        head_tile = min(8 if max(bn, br) <= 128 else 4, k.shape[1])
        token_span = 2 if max(bn, br) <= 128 else 1
        token_programs = min(32, max(1, triton.cdiv(tokens, token_span)))
        if max(bn, br) <= 128 and tokens >= 2:
            _concat_token_pair_contiguous_v17[
                (min(32, triton.cdiv(tokens, 2)), triton.cdiv(k.shape[1], 4))
            ](
                out,
                k_nope,
                k_rope,
                tokens,
                k.shape[1],
                dn,
                dr,
                COMMON=common,
                HEAD_TILE=4,
                BN=bn,
                BR=br,
                num_warps=1,
                num_stages=1,
            )
        elif max(bn, br) <= 256:
            _concat_rows_contiguous_v16[grid](
                out, k_nope, k_rope, rows, k.shape[1], dn, dr,
                COMMON=common, BM=bm, BN=bn, BR=br, num_warps=4,
            )
        else:
            _concat_tokens_contiguous[(token_programs,)](
                out, k_nope, k_rope, tokens, k.shape[1], dn, dr,
                COMMON=common, HEAD_TILE=head_tile, BN=bn, BR=br, num_warps=4,
            )
    else:
        _concat_rows[grid](
            out, k_nope, k_rope, rows, k.shape[1], dn, dr,
            *k_nope.stride(), k_rope.stride(0), k_rope.stride(2),
            COMMON=common, BM=bm, BN=bn, BR=br, num_warps=4,
        )
    return out


__all__ = ['concat_and_cast_mha_k']
