# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified implementation for FlagOS Track 1 Task 60.

The submission runner loads one public operator from the archive. Keep the
vendor paths in this file and dispatch from the tensor device so a later
vendor-specific definition cannot shadow an earlier one.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_position_contiguous_kernel(
    seq_lens_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    seq_lens = tl.load(
        seq_lens_ptr + offsets,
        mask=mask,
        other=0,
    )
    # Spell out the clamp instead of using tl.maximum: the latter currently
    # fails lowering for integer blocks on some Triton-TLE backends.
    positions = tl.where(seq_lens > 0, seq_lens - 1, 0)
    tl.store(out_ptr + offsets, positions, mask=mask)


@triton.jit
def _clamp_position_strided_kernel(
    seq_lens_ptr,
    out_ptr,
    n_elements,
    input_stride,
    output_stride,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    seq_lens = tl.load(
        seq_lens_ptr + offsets * input_stride,
        mask=mask,
        other=0,
    )
    positions = tl.where(seq_lens > 0, seq_lens - 1, 0)
    tl.store(out_ptr + offsets * output_stride, positions, mask=mask)


@triton.jit
def _clamp_position_i32_kernel(
    seq_lens_ptr,
    out_ptr,
    n_elements,
    input_stride,
    output_stride,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    seq_lens = tl.load(
        seq_lens_ptr + offsets * input_stride,
        mask=mask,
        other=0,
    )
    positions = tl.where(seq_lens > 0, seq_lens - 1, 0)
    tl.store(out_ptr + offsets * output_stride, positions, mask=mask)


def clamp_position(seq_lens: torch.Tensor) -> torch.Tensor:
    """Return ``(seq_lens - 1).clamp(min=0)`` with the same dtype and shape."""
    device_type = seq_lens.device.type
    vendor = os.environ.get("DNN_VENDOR", "").lower()

    # Torch-FL exposes Enflame as ``gcu``. Its Triton lowering is much faster
    # when int64 sequence lengths are bridged through an int32 work buffer.
    # Keep this dispatch tied to the actual tensor device; vendor-only
    # dispatch is intentionally avoided because the historical Enflame score
    # used for comparison was identified as invalid.
    if device_type == "gcu":
        original_dtype = seq_lens.dtype
        work = (
            seq_lens.to(torch.int32)
            if original_dtype == torch.int64
            else seq_lens
        )
        out_work = torch.empty_like(
            work,
            memory_format=torch.contiguous_format,
        )
        n_elements = work.numel()
        if n_elements == 0:
            return (
                out_work.to(original_dtype)
                if original_dtype == torch.int64
                else out_work
            )

        block = min(1024, max(32, triton.next_power_of_2(n_elements)))
        _clamp_position_i32_kernel[(triton.cdiv(n_elements, block),)](
            work,
            out_work,
            n_elements,
            work.stride(0),
            out_work.stride(0),
            BLOCK=block,
            num_warps=1,
            num_stages=1,
        )
        return (
            out_work.to(original_dtype)
            if original_dtype == torch.int64
            else out_work
        )

    # Torch-NPU exposes Ascend as ``npu`` in the normal runtime. Keep the
    # private-use spelling as a compatibility fallback for older adapters.
    if vendor == "ascend" or device_type == "npu" or (
        device_type == "privateuseone" and hasattr(torch, "npu")
    ):
        # The subtraction already creates a fresh tensor. Clamping that
        # temporary in place avoids a second allocation on the NPU path.
        return (seq_lens - 1).clamp_min_(0)

    n_elements = seq_lens.numel()
    out = torch.empty_like(seq_lens, memory_format=torch.contiguous_format)
    if n_elements == 0:
        return out

    # Match the official pointwise backend limits only when the runner tells
    # us the vendor explicitly. Keep the v5/v10 fallback for other runners,
    # because several CUDA-compatible devices share the same torch device
    # type and cannot be identified safely from ``device.type`` alone.
    if vendor in ("metax", "hygon"):
        max_block = 2048
    elif vendor == "tsingmicro":
        max_block = 4096
    else:
        max_block = 1024

    block = min(max_block, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, block),)

    # Small pointwise tiles have no reduction or inter-warp communication.
    # Try one warp through 256 elements; retain the large-tile configuration
    # so the online comparison isolates the small-input launch policy.
    if vendor == "tsingmicro" or block <= 256:
        num_warps = 1
    else:
        num_warps = 4

    if seq_lens.is_contiguous():
        _clamp_position_contiguous_kernel[grid](
            seq_lens,
            out,
            n_elements,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _clamp_position_strided_kernel[grid](
            seq_lens,
            out,
            n_elements,
            seq_lens.stride(0),
            out.stride(0),
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )
    return out


__all__ = ["clamp_position"]
