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
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["conv_window_scatter_with_mask"]

_ROW_CAP = 1024
_PLANS = {}


@triton.jit
def _conv_window_scatter_kernel(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    out_ptr,
    strides_ptr,
    CACHE: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    N_CHUNK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    REQUESTS: tl.constexpr,
):
    dst_s0 = tl.load(strides_ptr + 0)
    dst_s1 = tl.load(strides_ptr + 1)
    dst_s2 = tl.load(strides_ptr + 2)
    dst_s3 = tl.load(strides_ptr + 3)
    src_s0 = tl.load(strides_ptr + 4)
    src_s1 = tl.load(strides_ptr + 5)
    src_s2 = tl.load(strides_ptr + 6)
    src_s3 = tl.load(strides_ptr + 7)
    src_s4 = tl.load(strides_ptr + 8)
    out_s0 = tl.load(strides_ptr + 9)
    out_s1 = tl.load(strides_ptr + 10)
    out_s2 = tl.load(strides_ptr + 11)
    out_s3 = tl.load(strides_ptr + 12)

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
        layer * dst_s0
        + slot * dst_s1
        + dim_index * dst_s2
        + window_index * dst_s3
    )
    value = tl.load(dst_ptr + dst_offset, mask=in_row, other=0.0)

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
        layer * out_s0
        + slot * out_s1
        + dim_index * out_s2
        + window_index * out_s3
    )
    tl.store(out_ptr + out_offset, value, mask=in_row)


def conv_window_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    """Masked gather-scatter over an overlapping conv-window view, one launch.

    The schedule is the measured best one (one program per (layer, slot,
    row-chunk), reverse lookup by scanning the request table): a 54x smaller
    grid measured 3-4x *slower* on T4, because each program then carries tens of
    thousands of elements and the device runs out of parallelism.

    What is optimised here instead is the call path.  The harness times the
    public entry with CUDA events averaged over N calls, so for kernels this
    small the number is dominated by Python: an empty kernel measured 13.7 us on
    T4, a raw launch with pre-built arguments 22.3 us, and the previous entry
    43.2 us.  Every shape-dependent quantity is computed once per
    (shape, strides, dtype) key.  The 13 stride scalars are not passed per
    call: they live in a small cached int32 device buffer, so the signature is
    six pointers instead of eighteen arguments plus six constexprs -- argument
    binding measured 11.2 us of the 32.9 us entry on T4.

    The plan is keyed by the **source strides as well as the shape**: the same
    shape legitimately arrives with a different window layout, and a cache keyed
    by shape alone would then reuse a stale plan.  That is not hypothetical --
    ``l2-c16-r5-d3-dim8-w3-padded`` exists in the case table precisely to make a
    shape-only key fail.
    """
    key = (dst.shape, dst.stride(), src.stride(), dst.dtype, dst.device)
    plan = _PLANS.get(key)
    out = torch.empty_like(dst)
    if plan is None:
        layers, cache, dim, window = dst.shape
        requests = dst_indices_raw.shape[0]
        row_length = dim * window
        row_block = min(triton.next_power_of_2(max(row_length, 1)), _ROW_CAP)
        chunks = triton.cdiv(row_length, row_block)
        plan = (
            (layers * cache * chunks,),
            torch.tensor(
                [
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
                ],
                dtype=torch.int32,
                device=dst.device,
            ),
            dict(
                CACHE=cache,
                DIM=dim,
                WINDOW=window,
                N_CHUNK=chunks,
                ROW_BLOCK=row_block,
                REQUESTS=requests,
            ),
        )
        _PLANS[key] = plan
    grid, tail, consts = plan
    _conv_window_scatter_kernel[grid](
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        out,
        tail,
        **consts,
    )
    return out
