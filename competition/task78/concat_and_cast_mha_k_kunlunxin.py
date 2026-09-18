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
def _concat_rows_contiguous(
    out, nope, rope, ROWS, H,
    DN: tl.constexpr, DR: tl.constexpr,
    COMMON: tl.constexpr, BM: tl.constexpr,
    BN: tl.constexpr, BR: tl.constexpr,
):
    """Bounded row-batch path for contiguous XPU inputs.

    A row batch keeps the XPU launch simple and amortizes token/head address
    arithmetic across several rows. It mirrors the previously validated v5
    schedule, but is now restricted to the contiguous case so the strided
    fallback can retain its general pointer semantics.
    """
    for first in range(
        tl.program_id(0) * BM,
        ROWS,
        tl.num_programs(0) * BM,
    ):
        rows = first + tl.arange(0, BM)
        token = rows // H
        head = rows % H
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
def _concat_and_cast_mha_k_contiguous_kernel(
    out_ptr,
    nope_ptr,
    rope_ptr,
    HEADS,
    HEADS_PER_PROGRAM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    COMMON: tl.constexpr,
):
    """Contiguous path reusing the broadcast RoPE load across a head tile."""
    token = tl.program_id(0)
    heads = tl.program_id(1) * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    row_mask = heads < HEADS
    rows = token * HEADS + heads
    out_row = out_ptr + rows[:, None] * (NOPE_DIM + ROPE_DIM)
    nope_row = nope_ptr + rows[:, None] * NOPE_DIM
    rope_row = rope_ptr + token * ROPE_DIM

    if NOPE_DIM > 0:
        nope_cols = tl.arange(0, BLOCK_NOPE)
        nope_mask = row_mask[:, None] & (nope_cols[None, :] < NOPE_DIM)
        nope_value = tl.load(
            nope_row + nope_cols,
            mask=nope_mask,
            other=0,
        ).to(COMMON)
        tl.store(out_row + nope_cols, nope_value, mask=nope_mask)

    if ROPE_DIM > 0:
        rope_cols = tl.arange(0, BLOCK_ROPE)
        rope_mask = rope_cols < ROPE_DIM
        rope_value = tl.load(
            rope_row + rope_cols,
            mask=rope_mask,
            other=0,
        ).to(COMMON)
        tl.store(
            out_row + NOPE_DIM + rope_cols[None, :],
            rope_value[None, :],
            mask=row_mask[:, None] & rope_mask[None, :],
        )


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
    rows = tokens * heads
    row_batch = max(1, min(16, 4096 // max_block))
    row_grid = (min(32, triton.cdiv(rows, row_batch)),)

    # Keep the strided fallback one program per row. The contiguous path above
    # uses a bounded row batch; larger rows still benefit from more warps while
    # small cache rows avoid idle lanes.
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

    if k_nope.is_contiguous() and k_rope.is_contiguous():
        _concat_rows_contiguous[row_grid](
            out,
            k_nope,
            k_rope,
            rows,
            heads,
            DN=nope_dim,
            DR=rope_dim,
            COMMON=common,
            BM=row_batch,
            BN=block_nope,
            BR=block_rope,
            num_warps=4,
            num_stages=1,
        )
    else:
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
