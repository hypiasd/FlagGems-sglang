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

import os
import sys

import torch
import triton
import triton.language as tl

__all__ = ["conv_window_scatter_with_mask"]

_ROW_CAP = 1024
_PLANS = {}


def _stream_getter():
    """``() -> stream`` using the same driver accessor as ``JITFunction.run``.

    Read out of ``sys.modules`` rather than by importing or by attribute access
    on ``triton``: ``triton.runtime.jit`` does ``from .driver import driver``, so
    a running Triton always has it loaded, while a substitute module (the CPU
    semantic model used to validate this file) raises on unknown attributes.
    Returning ``None`` disables the direct launch.
    """
    module = sys.modules.get("triton.runtime.driver")
    # The module exposes ``driver`` (a DriverConfig) whose ``active`` property is
    # the backend driver that ``JITFunction.run`` uses; some versions put
    # ``active`` on the module itself.
    active = getattr(getattr(module, "driver", None), "active", None)
    if active is None:
        active = getattr(module, "active", None)
    if not callable(getattr(active, "get_current_stream", None)):
        return None
    return lambda: active.get_current_stream(active.get_current_device())


def _runtime_hooks():
    """``(enter, exit)`` launch hooks, exactly as ``JITFunction.run`` reads them.

    ``None`` means "could not be determined", which keeps the public path rather
    than risk skipping a hook the runner installed.
    """
    knobs = sys.modules.get("triton.knobs")
    runtime = getattr(knobs, "runtime", None)
    if runtime is None:
        return None
    return (
        getattr(runtime, "launch_enter_hook", None),
        getattr(runtime, "launch_exit_hook", None),
    )


def _distributed_prefix():
    """FlagTree's optional ``dist_param`` prefix for ``kernel.run``.

    FlagTree 0.6.1 inserts two distributed-runtime pointers before the bound
    arguments **only** when ``FLAGTREE_LITE_DIST`` opts in and the context is in
    lite mode (``triton/runtime/jit.py``); with the variable unset the prefix is
    empty, which is why the same call form works on every wheel.  Mirrored here
    so a lite-mode launch is byte-identical to the platform's own.
    """
    action = os.environ.get("FLAGTREE_LITE_DIST", "").strip().upper()
    if action not in ("1", "ON", "TRUE"):
        return ()
    module = sys.modules.get("triton.runtime._distributed")
    context_type = getattr(module, "DistributedRtContext", None)
    if context_type is None:
        return ()
    context = context_type()
    if context.is_lite_mode:
        return (context.comm_ptr, context.mem_ptr)
    return ()


def _fast_launcher(compiled, grid, hooks, get_stream):
    """Direct launch on the compiled kernel, or ``None`` to keep the public path.

    ``JITFunction.run`` redoes argument binding, cache-key construction, kernel
    lookup and global-value invalidation on **every** call; measured on T4 that
    is 19.9 us against 8.4 us for the equivalent direct launch.  The direct call
    mirrors ``JITFunction.run`` exactly -- same argument list, same grid
    canonicalisation, same hooks -- and only the redundant bookkeeping is
    skipped.

    Nothing here is discovered by exception handling (the task rejects any
    ``try`` in this file): every capability is probed with ``getattr``, and if
    the running Triton does not expose the launch internals the plan falls back
    to the public ``kernel[grid](...)`` path permanently.
    """
    function = getattr(compiled, "function", None)
    packed = getattr(compiled, "packed_metadata", None)
    metadata = getattr(compiled, "launch_metadata", None)
    run = getattr(compiled, "run", None)
    if (
        function is None
        or packed is None
        or run is None
        or hooks is None
        or not callable(metadata)
    ):
        return None
    if grid is None or len(grid) < 1:
        return None
    return (
        run,
        grid[0],
        grid[1] if len(grid) > 1 else 1,
        grid[2] if len(grid) > 2 else 1,
        function,
        packed,
        metadata,
        hooks[0],
        hooks[1],
        _distributed_prefix(),
        get_stream,
    )


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
    small the number is dominated by Python.  Measured on T4 at the smallest
    case (medians of 7, same protocol): an empty one-argument kernel 12.3 us; a
    bare launch of this kernel with a preallocated output 19.3 us; a fresh
    ``torch.empty_like`` interleaved with launches adds ~6 us (it costs only
    2.9 us in isolation -- the allocator has to account for blocks still in use
    by queued kernels); the previous entry 43.2 us.  Every shape-dependent
    quantity is computed once per (shape, strides, dtype) key.  The 13 stride
    scalars are not passed per call: they live in a small cached int32 device
    buffer, so the signature is six pointers instead of eighteen arguments plus
    six constexprs -- argument binding measured 11.2 us of the 32.9 us entry.
    Binding a *fresh* tensor object is itself free: cycling a preallocated pool
    of eight outputs measured 18.6 us against 19.3 us for one fixed tensor.

    The plan is keyed by the **source strides as well as the shape**: the same
    shape legitimately arrives with a different window layout, and a cache keyed
    by shape alone would then reuse a stale plan.  That is not hypothetical --
    ``l2-c16-r5-d3-dim8-w3-padded`` exists in the case table precisely to make a
    shape-only key fail.

    The remaining constexprs are passed **positionally** from a tuple built with
    the plan.  On the same kernel, grid, tensors and values that measured 25.4 us
    against 27.4 us with ``**consts`` (order-controlled, interleaved trials), so
    keyword expansion plus signature binding costs about 2 us per call.  Caching
    the ``kernel[grid]`` launcher object changed nothing (24.5 vs 24.6 us).

    Finally, the first call for a plan captures the compiled kernel that
    ``JITFunction.run`` returns and, if that Triton exposes the launch internals,
    the plan stores a **direct launch**: identical arguments, grid and hooks, but
    without the per-call binding, cache-key construction, kernel lookup and
    global-value invalidation.  Measured on T4: 8.4 us against 19.9 us for the
    public launch of the very same kernel.  If any capability is missing (probed
    with ``getattr``, never with an exception) the plan stays on the public path.
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
        plan = {
            "grid": (layers * cache * chunks,),
            "tail": torch.tensor(
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
            # Order must match the kernel signature, but positionally.
            "consts": (cache, dim, window, chunks, row_block, requests),
            "hooks": _runtime_hooks(),
            "get_stream": _stream_getter(),
            "launch": None,  # None = undecided, False = public path, tuple = fast
        }
        _PLANS[key] = plan

    grid, tail, consts = plan["grid"], plan["tail"], plan["consts"]
    args = (dst, src, dst_indices_raw, step_indices_raw, out, tail, *consts)
    launch = plan["launch"]
    if launch is False:
        _conv_window_scatter_kernel[grid](*args)
        return out
    if launch is None:
        # First call: take the public path -- which also compiles -- and keep the
        # compiled kernel it returns for the direct launch from here on.
        compiled = _conv_window_scatter_kernel[grid](*args)
        get_stream = plan["get_stream"]
        launch = None
        if get_stream is not None:
            launch = _fast_launcher(compiled, grid, plan["hooks"], get_stream)
        plan["launch"] = launch if launch is not None else False
        return out
    (
        run,
        grid_x,
        grid_y,
        grid_z,
        function,
        packed,
        launch_metadata,
        enter_hook,
        exit_hook,
        prefix,
        get_stream,
    ) = launch
    stream = get_stream()
    run(
        grid_x,
        grid_y,
        grid_z,
        stream,
        function,
        packed,
        launch_metadata(grid, stream, *args),
        enter_hook,
        exit_hook,
        *prefix,
        *args,
    )
    return out
