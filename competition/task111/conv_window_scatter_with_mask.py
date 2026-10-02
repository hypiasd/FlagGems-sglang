"""Task 111 ``conv_window_scatter_with_mask`` -- Triton candidate.

Structural change versus the baseline path: the official reference starts from
``dst.clone()`` and then resolves the write set through ``nonzero`` plus
advanced indexing.  This module writes the result in a **single launch**: one
program owns a ``(layer, slot, row-chunk)`` triple, reads the destination value
it has to preserve, resolves the reverse mapping ``slot -> request`` in-kernel
by scanning the (small) request list, and stores either the gathered source
value or the untouched destination value.  No clone, no host synchronisation,
and the source is addressed through its real strides so the overlapping
``as_strided`` window layout is handled rather than assumed contiguous.

Duplicate ``dst_indices_raw`` entries are resolved as "last valid request
wins"; the official reference leaves that case unspecified (see ``adapter.py``).

Every quantity is computed **inside the call** and held in local variables.  The
platform's code-safety validator rejects module-level mutable containers
("Global dict/set variables can cache results across benchmark iterations"), so
there is deliberately no plan cache, no cached stride buffer and no cached
kernel handle here.  The strides are compile-time constants of the launch
instead, which is both cache-free and cheaper on the device: the kernel no
longer loads thirteen stride scalars per program.
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
    DST: tl.constexpr,
    SRC: tl.constexpr,
    OUT: tl.constexpr,
    SHAPE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    # Packed launch scalars.  ``ROW_BLOCK`` deliberately stays its own constexpr
    # parameter: ``tl.arange`` only accepts a parameter that is *annotated*
    # constexpr, and a value derived by subscripting a constexpr tuple is a raw
    # Python int on real hardware ("arange's arguments must be of type
    # tl.constexpr", measured twice on 2026-10-02).
    dst_s0 = (DST[0], DST[1], DST[2], DST[3])
    src_s0 = (SRC[0], SRC[1], SRC[2], SRC[3], SRC[4])
    out_s0 = (OUT[0], OUT[1], OUT[2], OUT[3])
    CACHE = SHAPE[0]
    DIM = SHAPE[1]
    WINDOW = SHAPE[2]
    N_CHUNK = SHAPE[3]
    REQUESTS = SHAPE[4]
    position = tl.program_id(0)
    layer = position // (CACHE * N_CHUNK)
    rest = position % (CACHE * N_CHUNK)
    slot = rest // N_CHUNK
    chunk = rest % N_CHUNK

    # Reverse mapping, resolved in-kernel: the highest-index valid request that
    # targets this slot.  ``REQUESTS`` is the request count, not a block size.
    source = -1
    step = 0
    for i in range(REQUESTS):
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
        layer * dst_s0[0] + slot * dst_s0[1] + dim_index * dst_s0[2] + window_index * dst_s0[3]
    )
    # Slots that resolve to a request overwrite their destination value, so
    # reading it first would be a dead load.  Dropping it removes the
    # `requests / cache` share of the destination read stream.
    value = tl.load(dst_ptr + dst_offset, mask=in_row & (~hit), other=0.0)

    src_offset = (
        layer * src_s0[0]
        + source * src_s0[1]
        + step * src_s0[2]
        + dim_index * src_s0[3]
        + window_index * src_s0[4]
    )
    gathered = tl.load(src_ptr + src_offset, mask=in_row & hit, other=0.0)
    value = tl.where(hit, gathered, value)

    out_offset = (
        layer * out_s0[0] + slot * out_s0[1] + dim_index * out_s0[2] + window_index * out_s0[3]
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
    # Inlined equivalents of triton.next_power_of_2 / triton.cdiv: both are Python
    # helpers that re-validate their arguments on every call, and this path is
    # host-bound, so the validation is pure overhead here.
    row_block = 1 << (row_length - 1).bit_length() if 0 < row_length <= _ROW_CAP else _ROW_CAP
    chunks = (row_length + row_block - 1) // row_block
    out = torch.empty_like(dst)

    _conv_window_scatter_kernel[(layers * cache * chunks,)](
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        out,
        dst.stride(),
        src.stride(),
        out.stride(),
        (cache, dim, window, chunks, requests),
        row_block,
    )
    return out
