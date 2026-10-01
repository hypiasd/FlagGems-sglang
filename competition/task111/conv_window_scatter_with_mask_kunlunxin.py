"""Kunlunxin-dedicated Task 111 ``conv_window_scatter_with_mask``.

Why this file exists
--------------------
The generic module is routed to every target and fails to compile on Kunlunxin
(XPU):

    PassManager::run failed: loc("conv_window_scatter_with_mask.py":37:0)
    Pipeline failed while executing [TritonXPUUnrollControl on 'builtin.module']
    OutOfResources: out of resource: uni_sram  Required: 0, Hardware limit: 0
    Reducing block sizes or `num_stages` may help.

Identical 13/13-case failure on Task 112's Kunlunxin runs.  The `uni_sram` /
`Required: 0 / Hardware limit: 0` wrapping is not evidence of real SRAM
exhaustion -- it is how this build reports any exception raised inside
``make_ttxir`` -- so this file does not chase a resource limit.  It changes the
one construct that is most likely to break an unroll-control pass.

The single structural change
---------------------------
The generic kernel resolves ``slot -> request`` with a **Python-level loop over
``range(REQUESTS)``**, which unrolls into 2*REQUESTS scalar loads plus a chain of
scalar ``tl.where`` selections.  Here the request table is loaded as one block
and reduced instead: same semantics ("highest-index valid request wins"), no
unrolled scalar chain, and the loop-carried dependency disappears.

Compatibility hypothesis and its falsifier, kept separate from performance:
  * hypothesis: the unrolled scalar chain is what trips ``TritonXPUUnrollControl``;
  * falsifier: if Kunlunxin still reports the same pass failure with this file,
    the scan is not the trigger and the next test must bisect the remaining
    constructs (integer ``//``/``%`` by the non-power-of-two ``WINDOW``, the
    masked load/store, the ``tl.where`` merge).

Nothing else is changed: the schedule, the addressing, the masks and the
``num_stages`` hint are the generic ones.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["conv_window_scatter_with_mask"]

_ROW_CAP = 1024


@triton.jit
def _conv_window_scatter_kernel(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    out_ptr,
    dst_s0: tl.constexpr,
    dst_s1: tl.constexpr,
    dst_s2: tl.constexpr,
    dst_s3: tl.constexpr,
    src_s0: tl.constexpr,
    src_s1: tl.constexpr,
    src_s2: tl.constexpr,
    src_s3: tl.constexpr,
    src_s4: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    out_s3: tl.constexpr,
    CACHE: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    N_CHUNK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    REQUESTS: tl.constexpr,
    REQ_BLOCK: tl.constexpr,
):
    position = tl.program_id(0)
    layer = position // (CACHE * N_CHUNK)
    rest = position % (CACHE * N_CHUNK)
    slot = rest // N_CHUNK
    chunk = rest % N_CHUNK

    # Reverse mapping, resolved in-kernel as one block load plus a reduction:
    # the highest-index valid request that targets this slot.
    req = tl.arange(0, REQ_BLOCK)
    req_mask = req < REQUESTS
    targets = tl.load(dst_idx_ptr + req, mask=req_mask, other=-1)
    candidates = tl.load(step_idx_ptr + req, mask=req_mask, other=-1)
    valid = (targets == slot) & (candidates >= 0)
    source = tl.max(tl.where(valid, req, -1), axis=0)
    # Exactly one entry satisfies `req == source` and its candidate is >= 0, so
    # a second maximum (with -1 as the neutral element) extracts `step` without
    # a gather; when nothing matched, `source == -1` fires nowhere and step stays
    # -1, which is unused because `hit` is false.
    step = tl.max(tl.where(req == source, candidates, -1), axis=0)
    hit = source >= 0

    row = chunk * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    in_row = row < DIM * WINDOW
    dim_index = row // WINDOW
    window_index = row % WINDOW

    dst_offset = (
        layer * dst_s0 + slot * dst_s1 + dim_index * dst_s2 + window_index * dst_s3
    )
    # Slots that resolve to a request overwrite their destination value, so
    # reading it first would be a dead load.
    value = tl.load(dst_ptr + dst_offset, mask=in_row & (~hit), other=0.0)

    src_offset = (
        layer * src_s0
        + source * src_s1
        + step * src_s2
        + dim_index * src_s3
        + window_index * src_s4
    )
    gathered = tl.load(src_ptr + src_offset, mask=in_row & hit, other=0.0)
    value = tl.where(hit, gathered, value)

    out_offset = (
        layer * out_s0 + slot * out_s1 + dim_index * out_s2 + window_index * out_s3
    )
    tl.store(out_ptr + out_offset, value, mask=in_row)


def conv_window_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    """Masked gather-scatter over an overlapping conv-window view, one launch."""
    layers, cache, dim, window = dst.shape
    requests = dst_indices_raw.shape[0]
    row_length = dim * window
    row_block = min(triton.next_power_of_2(max(row_length, 1)), _ROW_CAP)
    chunks = triton.cdiv(row_length, row_block)
    req_block = max(triton.next_power_of_2(max(requests, 1)), 2)
    out = torch.empty_like(dst)

    _conv_window_scatter_kernel[(layers * cache * chunks,)](
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        out,
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        dst.stride(3),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        src.stride(3),
        src.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        cache,
        dim,
        window,
        chunks,
        row_block,
        requests,
        req_block,
        num_warps=4,
        num_stages=1,
    )
    return out
