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
    n_heads,
    nope_dim,
    total_dim,
    out_s0,
    out_s1,
    out_s2,
    nope_s0,
    nope_s1,
    nope_s2,
    rope_s0,
    rope_s2,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < total_dim

    out = out_ptr + token * out_s0 + head * out_s1 + cols * out_s2
    is_nope = cols < nope_dim
    nope = nope_ptr + token * nope_s0 + head * nope_s1 + cols * nope_s2
    rope_cols = cols - nope_dim
    rope = rope_ptr + token * rope_s0 + rope_cols * rope_s2

    # Loading both sides under masks keeps the kernel valid for arbitrary
    # prefix/suffix sizes while the select ensures only the selected load is
    # observable. The store pointer has the destination dtype, so Triton
    # performs the required low-precision cache cast at the final write.
    nope_value = tl.load(nope, mask=mask & is_nope, other=0.0)
    rope_value = tl.load(rope, mask=mask & ~is_nope, other=0.0)
    value = tl.where(is_nope, nope_value, rope_value)
    tl.store(out, value, mask=mask)


def concat_and_cast_mha_k(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> torch.Tensor:
    """Build ``k`` from per-head NoPE and single-head broadcast RoPE data."""
    tokens, heads, total_dim = k.shape
    nope_dim = k_nope.shape[-1]
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

    block = triton.next_power_of_2(total_dim)
    # Keep one program per row. This is a pure copy/cast kernel, so larger
    # rows benefit from more warps while small cache rows avoid idle lanes.
    if block <= 128:
        num_warps = 1
    elif block <= 512:
        num_warps = 2
    else:
        num_warps = 4

    _concat_and_cast_mha_k_kernel[(tokens, heads)](
        out,
        k_nope,
        k_rope,
        heads,
        nope_dim,
        total_dim,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        k_nope.stride(0),
        k_nope.stride(1),
        k_nope.stride(2),
        k_rope.stride(0),
        k_rope.stride(2),
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


__all__ = ["concat_and_cast_mha_k"]
