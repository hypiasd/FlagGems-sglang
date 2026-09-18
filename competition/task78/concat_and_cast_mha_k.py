"""Triton implementation for Task 78: concat_and_cast_mha_k.

The reference implementation materializes an expanded RoPE tensor, performs a
concatenation, and finally casts into the destination cache dtype.  This
kernel writes the destination directly: each program owns one ``(token,
head)`` row and copies the NoPE prefix followed by the broadcast RoPE suffix.
"""

import torch
import triton
import triton.language as tl


HEADS_PER_PROGRAM = 4


@triton.jit
def _concat_and_cast_mha_k_full_contiguous_kernel(
    out_ptr,
    nope_ptr,
    rope_ptr,
    HEADS,
    HEADS_PER_PROGRAM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    COMMON: tl.constexpr,
):
    """Write complete power-of-two source blocks without column masks."""
    token = tl.program_id(0)
    heads = tl.program_id(1) * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    rows = token * HEADS + heads
    dst = out_ptr + rows[:, None] * (NOPE_DIM + ROPE_DIM)
    nope_cols = tl.arange(0, NOPE_DIM)
    rope_cols = tl.arange(0, ROPE_DIM)
    nope_value = tl.load(
        nope_ptr + rows[:, None] * NOPE_DIM + nope_cols[None, :]
    ).to(COMMON)
    tl.store(dst + nope_cols[None, :], nope_value)
    rope_value = tl.load(rope_ptr + token * ROPE_DIM + rope_cols).to(COMMON)
    tl.store(dst + NOPE_DIM + rope_cols[None, :], rope_value[None, :])


@triton.jit
def _concat_and_cast_mha_k_token_pair_contiguous_kernel(
    out_ptr,
    nope_ptr,
    rope_ptr,
    TOKENS,
    HEADS,
    HEADS_PER_PROGRAM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    COMMON: tl.constexpr,
):
    """Process two adjacent tokens and one head tile per program.

    The ordinary contiguous kernel amortizes the launch over a head tile. For
    very short rows, the two-token mapping also amortizes program setup while
    keeping each token's RoPE vector separate and exact.
    """
    token_pair = tl.program_id(0) * 2 + tl.arange(0, 2)
    token_mask = token_pair < TOKENS
    heads = tl.program_id(1) * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    head_mask = heads < HEADS
    rows = token_pair[:, None] * HEADS + heads[None, :]
    dst = out_ptr + rows[:, :, None] * (NOPE_DIM + ROPE_DIM)

    if NOPE_DIM > 0:
        cols = tl.arange(0, BLOCK_NOPE)
        mask = (
            token_mask[:, None, None]
            & head_mask[None, :, None]
            & (cols[None, None, :] < NOPE_DIM)
        )
        value = tl.load(
            nope_ptr + rows[:, :, None] * NOPE_DIM + cols[None, None, :],
            mask=mask,
            other=0,
        ).to(COMMON)
        tl.store(dst + cols[None, None, :], value, mask=mask)

    if ROPE_DIM > 0:
        cols = tl.arange(0, BLOCK_ROPE)
        mask = (
            token_mask[:, None, None]
            & head_mask[None, :, None]
            & (cols[None, None, :] < ROPE_DIM)
        )
        value = tl.load(
            rope_ptr + token_pair[:, None] * ROPE_DIM + cols[None, :],
            mask=token_mask[:, None] & (cols[None, :] < ROPE_DIM),
            other=0,
        ).to(COMMON)
        tl.store(dst + NOPE_DIM + cols[None, None, :], value[:, None, :], mask=mask)


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

    # Reuse the broadcast RoPE vector across a small head tile. Larger rows
    # fall back to one head to keep register pressure bounded.
    # Avoid launching a partially masked head tile when the test shape has
    # fewer heads than the preferred tile width.
    preferred_heads = HEADS_PER_PROGRAM if max_block <= 512 else 1
    heads_per_program = 1 if heads == 1 else preferred_heads

    # Keep the strided fallback one program per row. The contiguous path above
    # uses head tiles; larger rows still benefit from more warps while small
    # cache rows avoid idle lanes.
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
        full_blocks = (
            nope_dim > 0
            and rope_dim > 0
            and nope_dim == block_nope
            and rope_dim == block_rope
            and heads % heads_per_program == 0
        )
        if full_blocks:
            _concat_and_cast_mha_k_full_contiguous_kernel[
                (tokens, triton.cdiv(heads, heads_per_program))
            ](
                out,
                k_nope,
                k_rope,
                HEADS=heads,
                HEADS_PER_PROGRAM=heads_per_program,
                NOPE_DIM=nope_dim,
                ROPE_DIM=rope_dim,
                COMMON=common,
                num_warps=num_warps,
                num_stages=1,
            )
        elif max_block <= 128 and tokens >= 2:
            _concat_and_cast_mha_k_token_pair_contiguous_kernel[
                (triton.cdiv(tokens, 2), triton.cdiv(heads, heads_per_program))
            ](
                out,
                k_nope,
                k_rope,
                tokens,
                heads,
                HEADS_PER_PROGRAM=heads_per_program,
                NOPE_DIM=nope_dim,
                ROPE_DIM=rope_dim,
                BLOCK_NOPE=block_nope,
                BLOCK_ROPE=block_rope,
                COMMON=common,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            _concat_and_cast_mha_k_contiguous_kernel[(tokens, triton.cdiv(heads, heads_per_program))](
                out,
                k_nope,
                k_rope,
                HEADS=heads,
                HEADS_PER_PROGRAM=heads_per_program,
                NOPE_DIM=nope_dim,
                ROPE_DIM=rope_dim,
                BLOCK_NOPE=block_nope,
                BLOCK_ROPE=block_rope,
                COMMON=common,
                num_warps=num_warps,
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
