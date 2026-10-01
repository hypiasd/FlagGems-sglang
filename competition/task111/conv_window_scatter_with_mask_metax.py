"""Metax-dedicated Task 111 ``conv_window_scatter_with_mask``.

Why this file exists
--------------------
The generic member is correct on this target but slow: the official table read
back on 2026-10-02 07:51 (+08:00) gives 沐曦 **0.56x** on the same bytes
that score 5.14x / 6.16x / 0.17x / 7.37x on other targets (submission
``task111-2026-10-02T07:41:56+08:00-53db409aa45f``).  A 5-30x gap between
targets on identical source is a backend artefact, not an algorithmic one, so
this file changes exactly one construct.

The single change
-----------------
``range(REQUESTS)`` becomes ``tl.static_range(REQUESTS)``.  Both bounds are the
same ``tl.constexpr`` launch parameter, and ``static_range`` only decides
*when* the iterations are laid out: the frontend emits the body once per
iteration as straight-line code instead of an ``scf.for``.  The arithmetic, the
memory operations, the masks and the "last valid request wins" resolution are
untouched, so **this change cannot alter a single output element** -- it is
safe to make on a target that already passes.

Why it is the right first lever: the scan issues ``2 * REQUESTS`` scalar loads
of ``dst_indices_raw``/``step_indices_raw`` with a loop-carried ``tl.where``
chain.  On a backend that hides global-memory latency with wide vector accesses
rather than with many concurrent warps, that dependent scalar chain is the
dominant cost, and unrolling it exposes the loads to the scheduler.

Falsifier
---------
If this target reports the same speed as the generic member, the scan loop is
not the bottleneck here and the next candidate is the block-reduction lookup
(one vector load of the request table plus ``tl.max`` over the request axis),
which removes the scalar chain entirely.  Because the change is
semantics-preserving, a correctness failure here would contradict the
equivalence claim and not a tuning choice.
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
):
    position = tl.program_id(0)
    layer = position // (CACHE * N_CHUNK)
    rest = position % (CACHE * N_CHUNK)
    slot = rest // N_CHUNK
    chunk = rest % N_CHUNK

    # Reverse mapping, resolved in-kernel: the highest-index valid request that
    # targets this slot.  ``REQUESTS`` is the request count, not a block size.
    source = -1
    step = 0
    for i in tl.static_range(REQUESTS):
        target = tl.load(dst_idx_ptr + i)
        candidate = tl.load(step_idx_ptr + i)
        match = (target == slot) & (candidate >= 0)
        source = tl.where(match, i, source)
        step = tl.where(match, candidate, step)
    hit = source >= 0

    row = chunk * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    in_row = row < DIM * WINDOW
    dim_index = row // WINDOW
    window_index = row % WINDOW

    dst_offset = (
        layer * dst_s0 + slot * dst_s1 + dim_index * dst_s2 + window_index * dst_s3
    )
    # Slots that resolve to a request overwrite their destination value, so
    # reading it first would be a dead load.  Dropping it removes the
    # `requests / cache` share of the destination read stream.
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
    """Masked gather-scatter over an overlapping conv-window view, one launch.

    The schedule is the measured best one (one program per (layer, slot,
    row-chunk), reverse lookup by scanning the request table): a 54x smaller
    grid measured 3-4x *slower* on T4, because each program then carries tens of
    thousands of elements and the device runs out of parallelism.

    What the launch passes: five pointers plus the thirteen real strides of
    ``dst``/``src``/``out`` and six shape-derived constants, all as
    ``tl.constexpr``.  Triton keys its compiled-kernel cache on those constants,
    so a new stride layout costs one compilation and every later call with the
    same layout reuses it -- without this module holding any state of its own,
    which a cached plan would require and which the platform forbids.
    """
    layers, cache, dim, window = dst.shape
    requests = dst_indices_raw.shape[0]
    row_length = dim * window
    row_block = min(triton.next_power_of_2(max(row_length, 1)), _ROW_CAP)
    chunks = triton.cdiv(row_length, row_block)
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
    )
    return out
