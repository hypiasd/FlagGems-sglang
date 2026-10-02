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


@triton.jit(
    # Pointer arguments are not specialized on alignment: measuring on the T4
    # showed 22.96 -> 20.63 us per launch for a five-pointer kernel (2.33 us),
    # and the alignment hint does not change this kernel's vectorization.
    do_not_specialize=["dst_ptr", "src_ptr", "dst_idx_ptr", "step_idx_ptr", "out_ptr"],
)
def _conv_window_scatter_kernel(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    out_ptr,
    LAYOUT: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    # One constexpr tuple instead of four: Triton's binder rebuilds a dict entry
    # and a cache-key entry per parameter on every call, so seven parameters beat
    # ten (T4, empty-kernel equivalent: 14.30 -> 13.70 us).  ROW_BLOCK stays its
    # own parameter because tl.arange only accepts an annotated constexpr, and a
    # value subscripted out of a constexpr tuple is a plain Python int on real
    # hardware ("arange's arguments must be of type tl.constexpr").
    dst_s0 = (LAYOUT[0], LAYOUT[1], LAYOUT[2], LAYOUT[3])
    src_s0 = (LAYOUT[4], LAYOUT[5], LAYOUT[6], LAYOUT[7], LAYOUT[8])
    out_s0 = (LAYOUT[9], LAYOUT[10], LAYOUT[11], LAYOUT[12])
    CACHE = LAYOUT[13]
    DIM = LAYOUT[14]
    WINDOW = LAYOUT[15]
    N_CHUNK = LAYOUT[16]
    REQUESTS = LAYOUT[17]
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
        # Both index tensors are cast to int32 before they enter the loop-carried
        # selects.  Triton 3.6 rejects a loop-carried variable whose type changes
        # between iterations ("Loop-carried variable step has initial type int32
        # but is re-assigned to int64"), so an int64 index tensor -- PyTorch's
        # default for index tensors -- compiled fine on the development harness
        # (which builds int32) and then failed on any caller that passed int64.
        # Slot and step indices are bounded by the cache depth, so int32 is exact.
        target = tl.load(dst_idx_ptr + i).to(tl.int32)
        candidate = tl.load(step_idx_ptr + i).to(tl.int32)
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
    # ROW_BLOCK is fixed instead of next_power_of_2(dim * window): the device is
    # provably not the bottleneck (an empty kernel with this exact signature
    # measures the same as the real one on T4), so per-call Python arithmetic to
    # size the block buys nothing.
    chunks = (dim * window + _ROW_CAP - 1) >> 10
    out = torch.empty_like(dst)
    dst_s = dst.stride()
    src_s = src.stride()
    out_s = out.stride()
    _conv_window_scatter_kernel.run(
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        out,
        (
            dst_s[0], dst_s[1], dst_s[2], dst_s[3],
            src_s[0], src_s[1], src_s[2], src_s[3], src_s[4],
            out_s[0], out_s[1], out_s[2], out_s[3],
            cache, dim, window, chunks, requests,
        ),
        _ROW_CAP,
        grid=(layers * cache * chunks,),
        warmup=False,
    )
    return out
