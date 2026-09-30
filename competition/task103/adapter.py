"""Task 103 reference and development shapes; not an official case table."""
from competition.experiments.checking import check_output


def cases():
    return [{"id": "tiny", "shape": [1, 64, 1, 2, 32, 32, 16]},
            {"id": "main", "shape": [8, 1024, 8, 32, 128, 128, 64]},
            {"id": "bt128", "shape": [4, 1024, 4, 16, 128, 128, 128]}]


def quick_ids():
    return ["tiny", "main", "bt128"]


def inputs(case, device, seed):
    import torch
    b, t, hg, h, k, v, bt = case["shape"]
    gen = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.randn(*shape, generator=gen).to(device)
    return (rand(b, t, hg, k).bfloat16(), rand(b, t, h, v).bfloat16(),
            torch.rand(b, t, h, generator=gen).to(device),
            -torch.rand(b, t, h, generator=gen).to(device) * 0.5,
            (rand(b, t, h, bt) * 0.05).bfloat16(), None)


def reference(k, v, beta, g_cumsum, A, cu_seqlens):
    import torch
    assert cu_seqlens is None
    b, t, hg, width = k.shape
    h, bt = v.shape[2], A.shape[-1]
    w = k.new_empty(b, t, h, width)
    u = torch.empty_like(v)
    for c in range(t // bt):
        rows = slice(c * bt, (c + 1) * bt)
        ac = A[:, rows].permute(0, 2, 1, 3).float()
        bb = beta[:, rows].permute(0, 2, 1).float()
        bg = torch.exp(g_cumsum[:, rows].permute(0, 2, 1).float())
        vb = v[:, rows].permute(0, 2, 1, 3).float() * bb[..., None]
        u[:, rows] = (ac @ vb.to(v.dtype).float()).permute(0, 2, 1, 3).to(v.dtype)
        kh = k[:, rows].permute(0, 2, 1, 3).float().repeat_interleave(h // hg, dim=1)
        kb = kh * bb[..., None] * bg[..., None]
        w[:, rows] = (ac @ kb.to(k.dtype).float()).permute(0, 2, 1, 3).to(k.dtype)
    return w, u


def check(got, want):
    check_output(got, want, atol=0.015, rtol=0.015)


def metadata(case):
    return {**case, "coverage": "development-assumptions"}
