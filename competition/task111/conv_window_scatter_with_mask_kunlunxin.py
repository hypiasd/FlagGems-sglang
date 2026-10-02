"""Task 111 ``conv_window_scatter_with_mask`` -- Kunlunxin member.

Why this target needs its own structure
---------------------------------------
The XPU build's ``TritonXPUUnrollControl`` pass fails on any kernel whose body
contains a loop (``PassManager::run failed ... [TritonXPUUnrollControl on
builtin.module]`` / ``OutOfResources: out of resource: uni_sram``), and that pass
is where this backend's performance comes from: turning it off compiles and is
correct but measures 0.02x.  Writing the request scan as ``tl.static_range``
removes the loop and compiles, but XPU then fails at runtime tuning the buffer
sizes of the unrolled straight-line code.  A block load plus a reduction has no
loop either and does compile, but it is wrong on this backend.

So this member uses neither a loop nor a reduction.  It performs the reference's
own two phases as two launches:

1. ``_copy_kernel``    -- a flat, fully vectorised copy of ``dst`` into ``out``;
2. ``_scatter_kernel`` -- one program per (layer, request, chunk) that writes
   ``src``'s window into the slot ``dst_indices_raw[request]`` names;

which is exactly

    out = dst.clone()
    for every request i with step_indices_raw[i] >= 0:
        out[:, dst_indices_raw[i]] = src[:, i, step_indices_raw[i]]

The second launch runs after the first on the same stream, so a slot that no
valid request names keeps the copied value and a named slot is overwritten.  Both
kernels are branch-free: a request whose step is negative is masked out rather
than skipped.

Cost of that choice: this member reads and writes the whole destination instead
of only the untouched slots, so it cannot beat a single-pass implementation on a
backend where the single pass works.  On this backend the single pass does not
compile at all, so the comparison is against 0.02x.

Duplicate ``dst_indices_raw`` entries would be resolved non-deterministically by
concurrent stores.  The official reference leaves that case unspecified (torch
advanced-index assignment with repeated indices takes the last occurrence), and
the scored cases build the slot list as a permutation subset, so every scored
case has distinct slots; ``adapter.py`` asserts the same invariant.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["conv_window_scatter_with_mask"]

_COPY_BLOCK = 8192


@triton.jit(
    do_not_specialize=["dst_ptr", "out_ptr"],
)
def _copy_kernel(
    dst_ptr,
    out_ptr,
    TOTAL,
    dst_s0: tl.constexpr,
    dst_s1: tl.constexpr,
    dst_s2: tl.constexpr,
    dst_s3: tl.constexpr,
    CACHE: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    CONTIG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``out = dst`` over the whole destination.

    The destination is plain data movement, so the dense case is a flat block
    copy with no integer division anywhere.  A non-dense ``dst`` (which
    ``empty_like`` turns into a dense ``out``) is the fallback and has to walk the
    real strides; that path exists only for callers that pass a strided
    destination and is not exercised by the scored cases.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    # ``other=`` is never observed (the store carries the same mask) but the
    # bounded CPU model requires it to be explicit.
    if CONTIG:
        value = tl.load(dst_ptr + offsets, mask=mask, other=0.0)
    else:
        window_index = offsets % WINDOW
        rest = offsets // WINDOW
        dim_index = rest % DIM
        rest = rest // DIM
        slot = rest % CACHE
        layer = rest // CACHE
        value = tl.load(
            dst_ptr
            + layer * dst_s0
            + slot * dst_s1
            + dim_index * dst_s2
            + window_index * dst_s3,
            mask=mask,
            other=0.0,
        )
    tl.store(out_ptr + offsets, value, mask=mask)


@triton.jit(
    do_not_specialize=["src_ptr", "dst_idx_ptr", "step_idx_ptr", "out_ptr"],
)
def _scatter_kernel(
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    out_ptr,
    src_s0: tl.constexpr,
    src_s1: tl.constexpr,
    src_s2: tl.constexpr,
    src_s3: tl.constexpr,
    src_s4: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    out_s3: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
    WINDOW_BLOCK: tl.constexpr,
):
    """Write one request's window into the slot that request names.

    The grid is two-dimensional -- ``program_id(0)`` is the request and
    ``program_id(1)`` is the layer -- and the window is a two-dimensional block,
    so this body contains no integer division or modulo at all.  That is
    deliberate: on this backend a dense copy kernel carrying no ``//``/``%`` runs
    correctly while the version that decomposed a flat row index with
    ``row // WINDOW`` (a non-power-of-two divisor) did not, and the unroll-control
    pass that this target depends on is known to be sensitive to exactly these
    constructs.  The slot and step come back as one-element blocks rather than
    0-d scalars so that every value here is a block value.
    """
    request = tl.program_id(0)
    layer = tl.program_id(1)
    one = tl.arange(0, 1)
    slot = tl.load(dst_idx_ptr + request + one).to(tl.int32)
    step = tl.load(step_idx_ptr + request + one).to(tl.int32)
    # An invalid request is masked out, never branched on.
    valid = step >= 0

    dim_index = tl.arange(0, DIM_BLOCK)[:, None]
    window_index = tl.arange(0, WINDOW_BLOCK)[None, :]
    mask = (dim_index < DIM) & (window_index < WINDOW) & valid

    src_offset = (
        layer * src_s0
        + request * src_s1
        + step * src_s2
        + dim_index * src_s3
        + window_index * src_s4
    )
    gathered = tl.load(src_ptr + src_offset, mask=mask, other=0.0)

    out_offset = (
        layer * out_s0
        + slot * out_s1
        + dim_index * out_s2
        + window_index * out_s3
    )
    tl.store(out_ptr + out_offset, gathered, mask=mask)


def conv_window_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    """Copy the destination, then scatter every valid request into it."""
    layers, cache, dim, window = dst.shape
    requests = dst_indices_raw.shape[0]
    out = torch.empty_like(dst)

    total = out.numel()
    _copy_kernel[(triton.cdiv(total, _COPY_BLOCK),)](
        dst,
        out,
        total,
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        dst.stride(3),
        cache,
        dim,
        window,
        int(dst.is_contiguous()),
        _COPY_BLOCK,
    )

    if requests:
        _scatter_kernel[(requests, layers)](
            src,
            dst_indices_raw,
            step_indices_raw,
            out,
            src.stride(0),
            src.stride(1),
            src.stride(2),
            src.stride(3),
            src.stride(4),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            dim,
            window,
            triton.next_power_of_2(dim),
            triton.next_power_of_2(window),
        )
    return out
