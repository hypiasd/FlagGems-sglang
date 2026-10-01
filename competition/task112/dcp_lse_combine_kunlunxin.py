"""Task 112 ``dcp_lse_combine``, Kunlunxin-specific module (revision 3).

Revision 3 is driven by the platform's **actual compiler error**, which the
submission page exposes when the Kunlunxin ``Failed`` cell is clicked (13/13
official cases, newest record 2026-10-01 23:04)::

    RuntimeError: PassManager::run failed: loc("<path>/dcp_lse_combine.py":45:0)
      Pipeline failed while executing [TritonXPUUnrollControl on 'builtin.module']
    OutOfResources: out of resource: uni_sram
      Required: 0, Hardware limit: 0. Reducing block sizes or `num_stages` may help.

The observed failure is at compile time in ``TritonXPUUnrollControl`` across
these 13 cases; this does not establish shape independence or SRAM exhaustion.
A separately inspected FlagTree 0.7.0+xpu3.6 backend wraps any exception from
``pm.run(mod, 'make_ttxir')`` as ``OutOfResources(0, 0, "uni_sram ...")``.
That build has not been verified as the competition's compiler version.
Revisions 1 and 2 retained ``tl.static_range`` in both passes.  Testing ordinary
loops is therefore a hypothesis about compiler compatibility, not a diagnosed
root cause or a locally verified IR transformation.

Revision 3 changes one thing: both shard loops become plain ``range(N)`` loops
(``N`` remains constexpr).  Accelerator compilation remains to be evaluated.
Everything else is retained from revision 2 (flat 1-D grid, ``[1]`` block LSE
accesses, explicit ``tl.broadcast_to``, one ``D_BLOCK`` accumulator, no
``tl.dot``), and the six targets that pass keep the exact generic bytes that
produced their scores.

History, kept because both falsifications are evidence:

* revision 1 (flat grid + shard-0 max seed + explicit fp32 casts) failed at
  23:01 (``adapt-e29585109e6e``);
* revision 2 (all-block accesses, explicit broadcasts) failed at 23:04
  (``adapt-5218f1763fce``) -- and that run is what produced the error above.

Falsifier: Kunlunxin fails again, or scores below 0.1.  If it fails, the next
lever named by the error itself is block size / ``num_stages``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["dcp_lse_combine"]


@triton.jit
def _dcp_lse_combine_kunlunxin_kernel(
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
    H: tl.constexpr,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_BLOCK: tl.constexpr,
    MASK_D: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    position = tl.program_id(0)
    batch = position // H
    head = position % H
    one = tl.arange(0, 1)
    d = tl.arange(0, D_BLOCK)
    lse_base = batch * l_stride_b + head * l_stride_h + one

    # Pass 1: the shard max over a [1] block; NaN / +inf become -inf.
    lse_max = tl.full([1], float("-inf"), dtype=tl.float32)
    for i in range(N):
        value = tl.load(recv_lse_ptr + lse_base + i * l_stride_n).to(
            tl.float32
        )
        value = tl.where(
            (value != value) | (value == float("inf")),
            -float("inf"),
            value,
        )
        lse_max = tl.maximum(lse_max, value)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)

    # Pass 2: weights and the weighted sum, accumulated in fp32.
    weight_sum = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D_BLOCK], dtype=tl.float32)
    for i in range(N):
        value = tl.load(recv_lse_ptr + lse_base + i * l_stride_n).to(
            tl.float32
        )
        value = tl.where(
            (value != value) | (value == float("inf")),
            -float("inf"),
            value,
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
        acc += tl.broadcast_to(weight, [D_BLOCK]) * partial.to(tl.float32)

    acc = acc / tl.broadcast_to(weight_sum, [D_BLOCK])
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
            out_lse_ptr + batch * lr_stride_b + head * lr_stride_h + one,
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
        # Unused by the kernel when RETURN_LSE is False.
        out_lse = recv_lse
        lr_stride_b = lr_stride_h = 0
    block = triton.next_power_of_2(dim)
    _dcp_lse_combine_kunlunxin_kernel[(batch * head,)](
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
        H=head,
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
