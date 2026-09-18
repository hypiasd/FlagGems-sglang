"""Ascend implementation for Task 78: concat_and_cast_mha_k.

The generic kernel uses a select between two masked loads.  That is compact,
but some Ascend Triton backends are less tolerant of the inactive load's
negative pointer arithmetic and of the mixed select lowering.  Keep the same
pure-Triton implementation while making each copy path explicit: one kernel
copies the per-head NoPE prefix and a second kernel copies the broadcast RoPE
suffix.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_nope_prefix_kernel(
    out_ptr,
    nope_ptr,
    nope_dim,
    out_s0,
    out_s1,
    out_s2,
    nope_s0,
    nope_s1,
    nope_s2,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < nope_dim

    out = out_ptr + token * out_s0 + head * out_s1 + cols * out_s2
    nope = nope_ptr + token * nope_s0 + head * nope_s1 + cols * nope_s2
    value = tl.load(nope, mask=mask, other=0.0)
    tl.store(out, value, mask=mask)


@triton.jit
def _copy_rope_suffix_kernel(
    out_ptr,
    rope_ptr,
    nope_dim,
    rope_dim,
    out_s0,
    out_s1,
    out_s2,
    rope_s0,
    rope_s2,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < rope_dim

    out = out_ptr + token * out_s0 + head * out_s1 + (nope_dim + cols) * out_s2
    # k_rope has one head and is broadcast over the destination heads.
    rope = rope_ptr + token * rope_s0 + cols * rope_s2
    value = tl.load(rope, mask=mask, other=0.0)
    tl.store(out, value, mask=mask)


def _num_warps(block: int) -> int:
    if block <= 128:
        return 1
    if block <= 512:
        return 2
    return 4


def concat_and_cast_mha_k(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> torch.Tensor:
    """Build ``k`` using two explicit Triton copy/cast paths."""
    tokens, heads, _ = k.shape
    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    out = torch.empty(
        k.shape,
        dtype=k.dtype,
        device=k.device,
        memory_format=torch.contiguous_format,
    )
    if tokens == 0 or heads == 0:
        return out

    if nope_dim:
        nope_block = triton.next_power_of_2(nope_dim)
        _copy_nope_prefix_kernel[(tokens, heads)](
            out,
            k_nope,
            nope_dim,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            k_nope.stride(0),
            k_nope.stride(1),
            k_nope.stride(2),
            BLOCK=nope_block,
            num_warps=_num_warps(nope_block),
            num_stages=1,
        )

    if rope_dim:
        rope_block = triton.next_power_of_2(rope_dim)
        _copy_rope_suffix_kernel[(tokens, heads)](
            out,
            k_rope,
            nope_dim,
            rope_dim,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            k_rope.stride(0),
            k_rope.stride(2),
            BLOCK=rope_block,
            num_warps=_num_warps(rope_block),
            num_stages=1,
        )

    return out


__all__ = ["concat_and_cast_mha_k"]
