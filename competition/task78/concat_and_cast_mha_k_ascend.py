"""Task 78 v18: Ascend bounded total grid, dense segment jobs in one launch.

NoPE and RoPE jobs never mix source pointer types or use 3-D broadcasts.
PyTorch only allocates output and determines the reference promotion dtype.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _concat_segment_jobs_v18(
    out, nope, rope,
    T: tl.constexpr, H: tl.constexpr, DN: tl.constexpr, DR: tl.constexpr,
    NS0: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr,
    RS0: tl.constexpr, RS2: tl.constexpr,
    COMMON: tl.constexpr, CONTIGUOUS: tl.constexpr,
    BLOCK: tl.constexpr, PREFIX_JOBS: tl.constexpr, TOTAL_JOBS: tl.constexpr,
):
    """Copy actual elements packed across rows; prefix/suffix stores are disjoint."""
    for job in range(tl.program_id(0), TOTAL_JOBS, tl.num_programs(0)):
        if DN > 0:
            if job < PREFIX_JOBS:
                offsets = job.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                row = offsets // DN
                col = offsets % DN
                valid = offsets < T * H * DN
                if CONTIGUOUS:
                    src = offsets
                else:
                    src = (row // H) * NS0 + (row % H) * NS1 + col * NS2
                value = tl.load(nope + src, mask=valid, other=0).to(COMMON)
                tl.store(out + row * (DN + DR) + col, value, mask=valid)
        if DR > 0:
            if job >= PREFIX_JOBS:
                offsets = (job.to(tl.int64) - PREFIX_JOBS) * BLOCK + tl.arange(0, BLOCK)
                row = offsets // DR
                col = offsets % DR
                valid = offsets < T * H * DR
                if CONTIGUOUS:
                    src = (row // H) * DR + col
                else:
                    src = (row // H) * RS0 + col * RS2
                value = tl.load(rope + src, mask=valid, other=0).to(COMMON)
                tl.store(out + row * (DN + DR) + DN + col, value, mask=valid)


def concat_and_cast_mha_k(k, k_nope, k_rope):
    out = torch.empty(
        k.shape, dtype=k.dtype, device=k.device,
        memory_format=torch.contiguous_format,
    )
    tokens, heads, total_dim = k.shape
    if tokens == 0 or heads == 0 or total_dim == 0:
        return out
    dn, dr = k_nope.shape[2], k_rope.shape[2]
    common = getattr(
        tl, str(torch.promote_types(k_nope.dtype, k_rope.dtype)).split(".")[-1],
    )
    block = 1024
    prefix_jobs = triton.cdiv(tokens * heads * dn, block)
    jobs = prefix_jobs + triton.cdiv(tokens * heads * dr, block)
    _concat_segment_jobs_v18[(min(32, jobs),)](
        out, k_nope, k_rope, tokens, heads, dn, dr,
        *k_nope.stride(), k_rope.stride(0), k_rope.stride(2),
        COMMON=common,
        CONTIGUOUS=k_nope.is_contiguous() and k_rope.is_contiguous(),
        BLOCK=block, PREFIX_JOBS=prefix_jobs, TOTAL_JOBS=jobs,
        num_warps=4, num_stages=1,
    )
    return out


__all__ = ["concat_and_cast_mha_k"]
