"""Bounded-grid Ascend experiment; not yet validated on target hardware."""
import torch
import triton
import triton.language as tl


@triton.jit
def _concat_persistent(out, nope, rope, N: tl.constexpr, H: tl.constexpr,
                       D: tl.constexpr, DN: tl.constexpr,
                       NS0: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr,
                       RS0: tl.constexpr, RS2: tl.constexpr,
                       COMMON: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(tl.program_id(0) * BLOCK, N,
                       tl.num_programs(0) * BLOCK):
        i = start + tl.arange(0, BLOCK)
        row = i // D
        col = i % D
        token = row // H
        head = row % H
        prefix = col < DN
        nc = tl.minimum(col, tl.maximum(DN - 1, 0))
        rc = tl.maximum(col - DN, 0)
        a = tl.load(nope + token * NS0 + head * NS1 + nc * NS2,
                    (i < N) & prefix, other=0).to(COMMON)
        b = tl.load(rope + token * RS0 + rc * RS2,
                    (i < N) & ~prefix, other=0).to(COMMON)
        tl.store(out + i, tl.where(prefix, a, b), i < N)


def concat_and_cast_mha_k(k, k_nope, k_rope):
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    n = k.numel()
    if n == 0:
        return out
    promoted = torch.promote_types(k_nope.dtype, k_rope.dtype)
    common = getattr(tl, str(promoted).split('.')[-1])
    block = 1024
    _concat_persistent[(min(32, triton.cdiv(n, block)),)](
        out, k_nope, k_rope, n, k.shape[1], k.shape[2],
        k_nope.shape[2], *k_nope.stride(),
        k_rope.stride(0), k_rope.stride(2),
        COMMON=common, BLOCK=block, num_warps=4,
    )
    return out


__all__ = ['concat_and_cast_mha_k']
