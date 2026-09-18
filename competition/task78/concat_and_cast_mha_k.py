"""Triton implementation for Task 78: concat_and_cast_mha_k.

The reference implementation materializes an expanded RoPE tensor, performs a
concatenation, and finally casts into the destination cache dtype.  This
kernel writes the destination directly: each program owns one ``(token,
head)`` row and copies the NoPE prefix followed by the broadcast RoPE suffix.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _concat_and_cast_mha_k_kernel(
    out_ptr,
    nope_ptr,
    rope_ptr,
    out_s0,
    out_s1,
    out_s2,
    nope_s0,
    nope_s1,
    nope_s2,
    rope_s0,
    rope_s2,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    COMMON: tl.constexpr,
):
    """Copy one token/head row through two regular contiguous segments.

    Keeping the prefix and suffix as independent load/store pairs avoids the
    mixed masked loads and data-dependent pointer selection in the baseline.
    The source values are promoted before the final store so this still has
    the reference ``cat -> to(k.dtype)`` dtype semantics when the two inputs
    have different dtypes.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    out_row = out_ptr + token * out_s0 + head * out_s1

    if NOPE_DIM > 0:
        nope_cols = tl.arange(0, BLOCK_NOPE)
        nope_mask = nope_cols < NOPE_DIM
        nope_value = tl.load(
            nope_ptr
            + token * nope_s0
            + head * nope_s1
            + nope_cols * nope_s2,
            mask=nope_mask,
            other=0,
        ).to(COMMON)
        tl.store(
            out_row + nope_cols * out_s2,
            nope_value,
            mask=nope_mask,
        )

    if ROPE_DIM > 0:
        rope_cols = tl.arange(0, BLOCK_ROPE)
        rope_mask = rope_cols < ROPE_DIM
        rope_value = tl.load(
            rope_ptr + token * rope_s0 + rope_cols * rope_s2,
            mask=rope_mask,
            other=0,
        ).to(COMMON)
        tl.store(
            out_row + (NOPE_DIM + rope_cols) * out_s2,
            rope_value,
            mask=rope_mask,
        )


def concat_and_cast_mha_k(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> torch.Tensor:
    """Build ``k`` from per-head NoPE and single-head broadcast RoPE data."""
    tokens, heads, total_dim = k.shape
    # ``torch.cat(...).to(...)`` in the reference produces a contiguous
    # result.  Do not inherit a vendor-specific/non-contiguous layout from
    # the shape-and-dtype carrier ``k``: NPU exact comparison checks layout
    # independently of strides.
    out = torch.empty(
        k.shape,
        dtype=k.dtype,
        device=k.device,
        memory_format=torch.contiguous_format,
    )
    if tokens == 0 or heads == 0 or total_dim == 0:
        return out

    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    block_nope = triton.next_power_of_2(max(1, nope_dim))
    block_rope = triton.next_power_of_2(max(1, rope_dim))
    max_block = max(block_nope, block_rope)

    # Keep one program per row. This is a pure copy/cast kernel, so larger
    # rows benefit from more warps while small cache rows avoid idle lanes.
    if max_block <= 128:
        num_warps = 1
    elif max_block <= 1024:
        num_warps = 2
    else:
        num_warps = 4

    common = getattr(
        tl,
        str(torch.promote_types(k_nope.dtype, k_rope.dtype)).split(".")[-1],
    )

    _concat_and_cast_mha_k_kernel[(tokens, heads)](
        out,
        k_nope,
        k_rope,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        k_nope.stride(0),
        k_nope.stride(1),
        k_nope.stride(2),
        k_rope.stride(0),
        k_rope.stride(2),
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_NOPE=block_nope,
        BLOCK_ROPE=block_rope,
        COMMON=common,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


__all__ = ["concat_and_cast_mha_k"]
