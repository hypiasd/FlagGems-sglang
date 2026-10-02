"""Task 111 ``conv_window_scatter_with_mask`` -- Kunlunxin member.

Isolating the one variable that separates 0.02x from 7.17x
---------------------------------------------------------
Three bespoke device-side designs have now been falsified on this backend:

* a body loop -> ``TritonXPUUnrollControl`` fails the pipeline;
* 2-D ``tl.arange(0, N)[:, None]`` -> ``TritonXPULegalize`` fails the pipeline
  (``tt.make_range`` size mismatch), and the failure text named that construct
  at the exact line;
* the reverse-lookup body (1-D blocks, loop-free, no 0-d scalar load indexing
  memory, addresses affine in ``program_id``) *compiles* and returns a wrong
  result -- 82.3% of Case 1 and 83.2% of Case 3 mismatched.

Meanwhile one member has always compiled and always been correct on this
backend: the generic module.  It scored 0.02x here while the *same idea* scores
7.17x on intl_a, and the only structural difference between the two is how the
pointer arguments reach the kernel:

* the generic (R10) passes five ``data_ptr()`` integers and rebuilds them inside
  the kernel with ``tl.cast(addr, tl.pointer_type(elem_ty))``;
* this member passes the tensors themselves, so the backend sees typed,
  specialised pointers from the launcher.

An integer that is cast to a pointer inside the body carries no alignment or
contiguity information, which is precisely the kind of fact a code generator
uses to pick vectorised, coalesced accesses over scalar ones.  A 50x gap is
what that looks like when it goes the wrong way.

So this member is the intl_a/hygon byte-for-byte body: the packed-constexpr,
loop-scanning generic with real tensor pointer arguments.  It is the same
correctness evidence as the two targets already shipping it (10/10 CPU
semantic cases), and it changes exactly one variable against the 0.02x member.

Falsifier: if this reports 0.02x as well, the ``tl.cast`` hypothesis is wrong
and the gap is the request-scan loop itself; if it reports anything near the
other targets, the generic module's pointer passing is what cost this target
50x all along.
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
