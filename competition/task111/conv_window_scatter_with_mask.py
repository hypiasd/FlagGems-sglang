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
    # These five are integer addresses, not tensors (see the body), so the list
    # also keeps Triton from specializing them on divisibility by 16.
    do_not_specialize=[
        "dst_addr",
        "src_addr",
        "dst_idx_addr",
        "step_idx_addr",
        "out_addr",
    ],
)
def _conv_window_scatter_kernel(
    dst_addr,
    src_addr,
    dst_idx_addr,
    step_idx_addr,
    out_addr,
    LAYOUT: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    IDX64: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # One constexpr tuple instead of four: Triton's binder rebuilds a dict entry
    # and a cache-key entry per parameter on every call, so few parameters beat
    # many (T4, empty-kernel equivalent: 14.30 -> 13.70 us).  ROW_BLOCK stays its
    # own parameter because tl.arange only accepts an annotated constexpr, and a
    # value subscripted out of a constexpr tuple is a plain Python int on real
    # hardware ("arange's arguments must be of type tl.constexpr").
    #
    # All five pointers arrive as raw integer addresses rather than tensors.
    # Handing Triton an integer skips its per-argument bookkeeping -- the params
    # entry, the alignment specialization and the per-argument launch metadata --
    # and, most importantly, stops the launch path from holding a reference to the
    # output tensor that this call just allocated.  Together that is 0.60 us per
    # launch on the T4 (21.00 -> 20.40 us measured in an interleaved A/B), which
    # is the largest win any rewrite of this wrapper has produced.  The tensors
    # stay alive in the caller's locals for the whole launch, so the addresses
    # stay valid.  Because the element and index types can no longer be inferred
    # they are stated here: IDX64 because the task declares int32 indices but a
    # caller may still pass int64 (PyTorch's default index dtype), DTYPE for the
    # data itself.
    #
    # Both selections happen in Python while tracing.  The ``if``/``else`` form is
    # rejected for the pointer width -- Triton emits both branches and then fails
    # on the mismatched pointer types -- so IDX64 uses a conditional expression.
    index_ty = tl.int64 if IDX64 else tl.int32
    if DTYPE == 0:
        elem_ty = tl.float32
    elif DTYPE == 1:
        elem_ty = tl.float16
    else:
        elem_ty = tl.bfloat16
    out_ptr = tl.cast(out_addr, tl.pointer_type(elem_ty))
    dst_ptr = tl.cast(dst_addr, tl.pointer_type(elem_ty))
    src_ptr = tl.cast(src_addr, tl.pointer_type(elem_ty))
    dst_idx_ptr = tl.cast(dst_idx_addr, tl.pointer_type(index_ty))
    step_idx_ptr = tl.cast(step_idx_addr, tl.pointer_type(index_ty))
    # ``dst.shape`` travels whole: unpacking it on the host and re-packing four
    # scalars measured 0.66 us per call, more than either selection above.
    dst_s0 = (LAYOUT[0], LAYOUT[1], LAYOUT[2], LAYOUT[3])
    src_s0 = (LAYOUT[4], LAYOUT[5], LAYOUT[6], LAYOUT[7], LAYOUT[8])
    out_s0 = (LAYOUT[9], LAYOUT[10], LAYOUT[11], LAYOUT[12])
    SHAPE = LAYOUT[13]
    CACHE = SHAPE[1]
    DIM = SHAPE[2]
    WINDOW = SHAPE[3]
    N_CHUNK = LAYOUT[14]
    REQUESTS = LAYOUT[15]
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

    The launch passes five **addresses** -- ``dst``, ``src``, both index tensors
    and the freshly allocated output -- plus one constexpr tuple holding the
    thirteen real strides of ``dst``/``src``/``out``, ``dst.shape`` and two
    shape-derived constants, plus the pointer-width and element-type codes.  All
    five tensors stay referenced by the caller's locals for the whole launch, so
    the addresses remain valid; passing them as integers instead of tensors is
    what avoids the launch path holding the output allocation.  Triton keys its
    compiled-kernel cache on the constexpr tuple, so a new stride layout costs one
    compilation and every later call with the same layout reuses it -- without
    this module holding any state of its own, which a cached plan would require
    and which the platform forbids.
    """
    shape = dst.shape
    cache, dim, window = shape[1], shape[2], shape[3]
    # ROW_BLOCK is fixed instead of next_power_of_2(dim * window): the device is
    # provably not the bottleneck (an empty kernel with this exact signature
    # measures the same as the real one on T4), so per-call Python arithmetic to
    # size the block buys nothing.
    chunks = (dim * window + _ROW_CAP - 1) >> 10
    out = torch.empty_like(dst)
    _conv_window_scatter_kernel.run(
        dst.data_ptr(),
        src.data_ptr(),
        dst_indices_raw.data_ptr(),
        step_indices_raw.data_ptr(),
        out.data_ptr(),
        dst.stride() + src.stride() + out.stride() + (shape, chunks, dst_indices_raw.shape[0]),
        _ROW_CAP,
        int(dst_indices_raw.dtype == torch.int64),
        1 if dst.dtype == torch.float16 else (2 if dst.dtype == torch.bfloat16 else 0),
        grid=(shape[0] * cache * chunks,),
        warmup=False,
    )
    return out
