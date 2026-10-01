"""Task 112 ``dcp_lse_combine``: N-way LSE-weighted partial-output combine.

Status: **starting point, not an optimization, and untested on this machine**
(there is no torch/triton in the local interpreter and no reachable device).
It reproduces the official reference semantics with a baseline-shaped layout
-- one program per ``(batch, head)``, ``N`` and ``HEAD_DIM`` as constexpr, a
first pass for the max and a second for the weighted sum -- so that the
optimization loop has a correct-by-construction seed to measure against.

Deliberate choices versus the upstream baseline:

* input strides are honoured instead of forcing ``contiguous()``, so no extra
  copy is paid for a strided ``recv_output`` view;
* ``HEAD_DIM`` is served mask-free when it is already a power of two (64, 128,
  512 in the published shapes) and masked otherwise;
* ``is_lse_base_on_e`` is a constexpr, so the exp2/log2 variant is a separate
  compiled kernel rather than a runtime branch.

No try/except and no device or dtype branching: the task's anti-cheat rule
forbids any path that could fall back to native torch compute.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["dcp_lse_combine"]


@triton.jit
def _dcp_lse_combine_kernel(
    recv_output_ptr,
    recv_lse_ptr,
    out_ptr,
    out_lse_ptr,
    o_stride_n,
    o_stride_b,
    o_stride_h,
    o_stride_d,
    l_stride_n,
    l_stride_b,
    l_stride_h,
    r_stride_b,
    r_stride_h,
    r_stride_d,
    lr_stride_b,
    lr_stride_h,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_BLOCK: tl.constexpr,
    MASK_D: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D_BLOCK)
    lse_base = batch * l_stride_b + head * l_stride_h

    # Pass 1: the shard max, with NaN / +inf shards sanitized to -inf.
    lse_max = float("-inf")
    for i in tl.static_range(N):
        value = tl.load(recv_lse_ptr + lse_base + i * l_stride_n)
        value = value.to(tl.float32)
        value = tl.where(
            (value != value) | (value == float("inf")), -float("inf"), value
        )
        lse_max = tl.maximum(lse_max, value)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)

    # Pass 2: weights and the weighted sum, accumulated in fp32.
    weight_sum = 0.0
    acc = tl.zeros([D_BLOCK], dtype=tl.float32)
    for i in tl.static_range(N):
        value = tl.load(recv_lse_ptr + lse_base + i * l_stride_n)
        value = value.to(tl.float32)
        value = tl.where(
            (value != value) | (value == float("inf")), -float("inf"), value
        )
        centered = value - lse_max
        weight = tl.exp(centered) if IS_BASE_E else tl.exp2(centered)
        weight_sum += weight
        offsets = (
            i * o_stride_n
            + batch * o_stride_b
            + head * o_stride_h
            + d * o_stride_d
        )
        if MASK_D:
            partial = tl.load(
                recv_output_ptr + offsets, mask=d < HEAD_DIM, other=0.0
            )
        else:
            partial = tl.load(recv_output_ptr + offsets)
        acc += weight * partial.to(tl.float32)

    acc = acc / weight_sum
    result_offsets = batch * r_stride_b + head * r_stride_h + d * r_stride_d
    if MASK_D:
        tl.store(
            out_ptr + result_offsets,
            acc.to(out_ptr.dtype.element_ty),
            mask=d < HEAD_DIM,
        )
    else:
        tl.store(out_ptr + result_offsets, acc.to(out_ptr.dtype.element_ty))

    if RETURN_LSE:
        combined = tl.log(weight_sum) if IS_BASE_E else tl.log2(weight_sum)
        tl.store(
            out_lse_ptr + batch * lr_stride_b + head * lr_stride_h,
            (combined + lse_max).to(out_lse_ptr.dtype.element_ty),
        )


def dcp_lse_combine(recv_output, recv_lse, is_lse_base_on_e, return_lse):
    """Combine ``N`` partial attention outputs weighted by their LSE values."""
    n, batch, head, dim = recv_output.shape
    out = torch.empty(
        (batch, head, dim), device=recv_output.device, dtype=recv_output.dtype
    )
    if return_lse:
        out_lse = torch.empty(
            (batch, head), device=recv_lse.device, dtype=recv_lse.dtype
        )
        lr_stride_b, lr_stride_h = out_lse.stride(0), out_lse.stride(1)
    else:
        # Unused by the kernel when RETURN_LSE is False, so no allocation is
        # paid for it; the pointer only has to be a valid tensor argument.
        out_lse = recv_lse
        lr_stride_b = lr_stride_h = 0
    block = triton.next_power_of_2(dim)
    _dcp_lse_combine_kernel[(batch, head)](
        recv_output,
        recv_lse,
        out,
        out_lse,
        recv_output.stride(0),
        recv_output.stride(1),
        recv_output.stride(2),
        recv_output.stride(3),
        recv_lse.stride(0),
        recv_lse.stride(1),
        recv_lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lr_stride_b,
        lr_stride_h,
        N=n,
        HEAD_DIM=dim,
        D_BLOCK=block,
        MASK_D=dim != block,
        IS_BASE_E=bool(is_lse_base_on_e),
        RETURN_LSE=bool(return_lse),
    )
    if not return_lse:
        return out, None
    return out, out_lse
