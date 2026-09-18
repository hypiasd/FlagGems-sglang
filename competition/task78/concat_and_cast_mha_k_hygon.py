"""Task 78 v18: Hygon serial token pair with 2-D head/column tiles."""
import torch
import triton
import triton.language as tl


@triton.jit
def _concat_serial_pair_v18(
    out, nope, rope,
    T: tl.constexpr, H: tl.constexpr, DN: tl.constexpr, DR: tl.constexpr,
    NS0: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr,
    RS0: tl.constexpr, RS2: tl.constexpr, COMMON: tl.constexpr,
    HEAD_TILE: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
    COL_TILE: tl.constexpr,
):
    first_token = tl.program_id(0).to(tl.int64) * 2
    heads = tl.program_id(1) * HEAD_TILE + tl.arange(0, HEAD_TILE)
    first_col = tl.program_id(2) * COL_TILE
    # Unroll in program order instead of materializing [2, heads, columns].
    # Each token's RoPE vector is loaded in 1-D and broadcast only on store.
    for offset in tl.static_range(0, 2):
        token = first_token + offset
        if token < T:
            rows = token * H + heads
            if DN > 0:
                cols = first_col + tl.arange(0, BN)
                valid = (heads[:, None] < H) & (cols[None, :] < DN)
                value = tl.load(
                    nope + token * NS0 + heads[:, None] * NS1
                    + cols[None, :] * NS2,
                    mask=valid, other=0,
                ).to(COMMON)
                tl.store(
                    out + rows[:, None] * (DN + DR) + cols[None, :],
                    value, mask=valid,
                )
            if DR > 0:
                cols = first_col + tl.arange(0, BR)
                value = tl.load(
                    rope + token * RS0 + cols * RS2,
                    mask=cols < DR, other=0,
                ).to(COMMON)
                tl.store(
                    out + rows[:, None] * (DN + DR) + DN + cols[None, :],
                    value[None, :],
                    mask=(heads[:, None] < H) & (cols[None, :] < DR),
                )


def concat_and_cast_mha_k(k, k_nope, k_rope):
    out = torch.empty(
        k.shape, dtype=k.dtype, device=k.device,
        memory_format=torch.contiguous_format,
    )
    tokens, heads, total_dim = k.shape
    if tokens == 0 or heads == 0 or total_dim == 0:
        return out
    dn, dr = k_nope.shape[2], k_rope.shape[2]
    common = getattr(
        tl, str(torch.promote_types(k_nope.dtype, k_rope.dtype)).split(".")[-1],
    )
    head_tile = min(4, triton.next_power_of_2(heads))
    col_tile = 256
    bn = min(col_tile, triton.next_power_of_2(max(1, dn)))
    br = min(col_tile, triton.next_power_of_2(max(1, dr)))
    _concat_serial_pair_v18[
        (triton.cdiv(tokens, 2), triton.cdiv(heads, head_tile),
         triton.cdiv(max(dn, dr), col_tile))
    ](
        out, k_nope, k_rope, tokens, heads, dn, dr,
        *k_nope.stride(), k_rope.stride(0), k_rope.stride(2),
        COMMON=common, HEAD_TILE=head_tile, BN=bn, BR=br, COL_TILE=col_tile,
        num_warps=1 if max(bn, br) <= 128 else 2, num_stages=1,
    )
    return out


__all__ = ["concat_and_cast_mha_k"]
