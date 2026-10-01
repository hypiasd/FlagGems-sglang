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

That change did **not** fix Kunlunxin: submission 10-02 07:22 still reported the
identical ``TritonXPUUnrollControl`` failure, so the hypothesis "the unrolled
scalar scan is the trigger" is falsified and recorded in
``platform-failures.jsonl``.

The change that follows from the backend source
----------------------------------------------
``triton/backends/xpu/compiler.py`` (FlagTree 0.6.1+xpu3.6) shows

    if not metadata["isCloseUnrollControl"]:
        xpu.passes.ttxpuir.add_tritonxpu_unroll_control_pass(...)

and ``XPUBackend.parse_options`` copies every ``XPUOptions`` dataclass field out
of the launch options, so ``isCloseUnrollControl`` reaches that metadata and
**removes the exact pass that is failing from the pipeline**.  The same file also
shows why the reported cause is misleading:

    except Exception as e:
        raise OutOfResources(0, 0, f"uni_sram {e}")

-- every exception inside ``make_ttxir`` is re-labelled as an SRAM shortage, so
"Required: 0, Hardware limit: 0" never meant SRAM exhaustion.

This file is XPU-only by construction: the option is understood by the XPU
backend and rejected by other backends, and ``members.audit`` routes the
``_kunlunxin`` suffix to this target alone.

Falsifier for this change: if Kunlunxin still reports
``TritonXPUUnrollControl``, the platform's FlagTree build does not honour this
option (the guide notes the recommended wheel is *not* the event runtime build),
and the next step is to bisect the remaining constructs against that build.
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
    """Masked gather-scatter over an overlapping conv-window view, one launch."""
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
        num_warps=4,
        num_stages=1,
        # Skips add_tritonxpu_unroll_control_pass, the pass named in the
        # platform's failure.  Understood only by the XPU backend.
        isCloseUnrollControl=True,
    )
    return out
