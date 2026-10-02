"""Iluvatar-dedicated Task 111 ``conv_window_scatter_with_mask``: packed constexprs.

What was measured, and where the ceiling is
-------------------------------------------
On the T4 ladder the candidate is host-dispatch bound, not device bound:

| case group | reference | public launch path | direct ``kernel.run`` |
|---|---:|---:|---:|
| 8 small dev cases | ~219 us | 28.0 us = 7.7x | 17.0 us = 13.8x |
| all-invalid (pure copy, ref is ``clone``) | 53.5 us | 28.7 us = 1.87x | 18.6 us = 3.01x |
| big 2.1M elements | 267 us | 78.9 us = 3.38x | 76.7 us = 3.48x |

The ~11 us difference is ``JITFunction.run``'s per-call bookkeeping.  Closing that
gap entirely is what the (rejected) plan cache did; a mean of 10x over the ten
dev cases is therefore not reachable from the public launch path, because the
two non-small cases stay at 1.87x/3.38x no matter how fast the host gets.

The single change here is the *legal* part of that gap
-----------------------------------------------------
``triton/runtime/jit.py`` (FlagTree 0.6.1) generates the argument binder as one
native function::

    def dynamic_func(<every parameter>, **options):
        params = {'<name>': <name>, ...}          # one dict entry per parameter
        specialization = [...]                     # one list entry per parameter
        return params, specialization, options

and ``compute_cache_key`` then wraps ``tuple(specialization)`` into the key it
looks up.  Every entry is rebuilt on **every call**, so the per-call host cost
scales with the *number of parameters*.  This file passes the nineteen
launch-time scalars as four ``tl.constexpr`` tuples instead of nineteen
parameters -- 9 parameters instead of 24 -- and ``constexpr.__getitem__``
resolves them at compile time, so the generated device code is unchanged.

What this does NOT do: it holds no state between calls and caches nothing.  The
compiled kernel still lives only in Triton's own ``device_caches``, reached
through the public ``kernel[grid](...)`` path exactly as before.

Why only this chip
------------------
The change is host-side and backend-independent, but it is unverified on real
hardware: no device was reachable when it was written (the T4 tunnel answered
TCP and then closed during key exchange from 08:02 to 08:06 +08:00).  Routing it
to one stable chip makes the outcome decisive and bounded -- if this backend
rejects a tuple ``tl.constexpr``, only this target is lost.

Falsifier
---------
This target scored 5.14x on the same shapes with nineteen scalar parameters
(record ``task111-2026-10-02T07:41:56+08:00-53db409aa45f``).  If it reports the
same number here, per-parameter binder cost is not a measurable part of the
official measurement and the packing is not worth keeping.
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
):
    # The nineteen launch-time scalars arrive as four constexpr tuples instead of
    # nineteen separate parameters.  ``constexpr.__getitem__`` resolves these at
    # compile time, so the generated kernel is identical; what changes is the
    # host side: Triton's binder builds one dict entry and one cache-key entry
    # per *parameter*, so nine parameters instead of twenty-four is nine dict
    # entries and nine key entries instead of twenty-four, on every call.
    dst_s0 = (DST[0], DST[1], DST[2], DST[3])
    src_s0 = (SRC[0], SRC[1], SRC[2], SRC[3], SRC[4])
    out_s0 = (OUT[0], OUT[1], OUT[2], OUT[3])
    CACHE = SHAPE[0]
    DIM = SHAPE[1]
    WINDOW = SHAPE[2]
    N_CHUNK = SHAPE[3]
    ROW_BLOCK = SHAPE[4]
    REQUESTS = SHAPE[5]
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
    row_block = min(triton.next_power_of_2(max(row_length, 1)), _ROW_CAP)
    chunks = triton.cdiv(row_length, row_block)
    out = torch.empty_like(dst)

    _conv_window_scatter_kernel[(layers * cache * chunks,)](
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        out,
        (dst.stride(0), dst.stride(1), dst.stride(2), dst.stride(3)),
        (
            src.stride(0),
            src.stride(1),
            src.stride(2),
            src.stride(3),
            src.stride(4),
        ),
        (out.stride(0), out.stride(1), out.stride(2), out.stride(3)),
        (cache, dim, window, chunks, row_block, requests),
        num_warps=4,
        num_stages=1,
    )
    return out
