"""Task 78 v18: international A/B, dense NoPE + broadcast RoPE in one launch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _concat_hybrid_jobs_v18(
    out, nope, rope,
    T: tl.constexpr, H: tl.constexpr, DN: tl.constexpr, DR: tl.constexpr,
    NS0: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr,
    RS0: tl.constexpr, RS2: tl.constexpr,
    COMMON: tl.constexpr, CONTIGUOUS: tl.constexpr,
    BLOCK: tl.constexpr, PREFIX_JOBS: tl.constexpr,
    HEAD_TILE: tl.constexpr, ROPE_BLOCK: tl.constexpr,
    HEAD_JOBS: tl.constexpr, COL_JOBS: tl.constexpr,
):
    """Uniform program branches allow different work units for the two sources."""
    job = tl.program_id(0).to(tl.int64)
    if DN > 0:
        if job < PREFIX_JOBS:
            offsets = job * BLOCK + tl.arange(0, BLOCK)
            row, col = offsets // DN, offsets % DN
            valid = offsets < T * H * DN
            if CONTIGUOUS:
                src = offsets
            else:
                src = (row // H) * NS0 + (row % H) * NS1 + col * NS2
            value = tl.load(nope + src, mask=valid, other=0).to(COMMON)
            tl.store(out + row * (DN + DR) + col, value, mask=valid)
    if DR > 0:
        if job >= PREFIX_JOBS:
            suffix_job = job - PREFIX_JOBS
            token = suffix_job // (HEAD_JOBS * COL_JOBS)
            head_job = (suffix_job // COL_JOBS) % HEAD_JOBS
            col_job = suffix_job % COL_JOBS
            heads = head_job * HEAD_TILE + tl.arange(0, HEAD_TILE)
            cols = col_job * ROPE_BLOCK + tl.arange(0, ROPE_BLOCK)
            # One source vector per token/head tile; only the store broadcasts.
            value = tl.load(
                rope + token * RS0 + cols * RS2,
                mask=cols < DR, other=0,
            ).to(COMMON)
            rows = token * H + heads
            valid = (heads[:, None] < H) & (cols[None, :] < DR)
            tl.store(
                out + rows[:, None] * (DN + DR) + DN + cols[None, :],
                value[None, :], mask=valid,
            )


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
    head_tile = min(4, triton.next_power_of_2(heads))
    rope_block = min(128, triton.next_power_of_2(max(1, dr)))
    prefix_jobs = triton.cdiv(tokens * heads * dn, block)
    head_jobs = triton.cdiv(heads, head_tile)
    col_jobs = triton.cdiv(dr, rope_block)
    jobs = prefix_jobs + tokens * head_jobs * col_jobs
    _concat_hybrid_jobs_v18[(jobs,)](
        out, k_nope, k_rope, tokens, heads, dn, dr,
        *k_nope.stride(), k_rope.stride(0), k_rope.stride(2),
        COMMON=common,
        CONTIGUOUS=k_nope.is_contiguous() and k_rope.is_contiguous(),
        BLOCK=block, PREFIX_JOBS=prefix_jobs,
        HEAD_TILE=head_tile, ROPE_BLOCK=rope_block,
        HEAD_JOBS=head_jobs, COL_JOBS=col_jobs,
        num_warps=4, num_stages=1,
    )
    return out


__all__ = ["concat_and_cast_mha_k"]
