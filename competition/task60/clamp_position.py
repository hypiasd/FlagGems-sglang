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

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_position_kernel(
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
    # Spell out the clamp instead of using tl.maximum: the latter currently
    # fails lowering for integer blocks on some Triton-TLE backends.
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

    # Torch-FL exposes Enflame as ``gcu``. Its Triton lowering is much faster
    # when int64 sequence lengths are bridged through an int32 work buffer.
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
    if device_type == "npu" or (
        device_type == "privateuseone" and hasattr(torch, "npu")
    ):
        return torch.clamp_min(seq_lens - 1, 0)

    n_elements = seq_lens.numel()
    out = torch.empty_like(seq_lens, memory_format=torch.contiguous_format)
    if n_elements == 0:
        return out

    block = min(1024, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, block),)

    if block <= 32:
        num_warps = 1
    elif block <= 128:
        num_warps = 2
    else:
        num_warps = 4

    _clamp_position_kernel[grid](
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
