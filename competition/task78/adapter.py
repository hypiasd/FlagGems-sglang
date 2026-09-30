"""Task 78 exact semantics and its complete development regression matrix."""
from competition.experiments.checking import check_output, mirror


def cases():
    from competition.task78.validate_cpu import cases as matrix
    return [{"id": c.name, "spec": c} for c in matrix()]


def quick_ids():
    return ["tiles-1-1-7-9", "grid-loop-65-513", "wide-dim-4096-4096-contiguous"]


def inputs(case, device, seed):
    import torch
    from competition.task78.validate_cpu import make_tensor
    c = case["spec"]
    t, h, dn, dr = c.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    nd, rd, kd = c.dtypes
    return tuple(mirror(make_tensor(shape, dtype, layout, gen), device) for shape, dtype, layout in
                 [((t, h, dn + dr), kd, c.layouts[0]), ((t, h, dn), nd, c.layouts[1]), ((t, 1, dr), rd, c.layouts[2])])


def reference(k, nope, rope):
    import torch
    return torch.cat([nope, rope.expand(-1, k.shape[1], -1)], dim=-1).to(k.dtype)


def check(got, want):
    check_output(got, want)
    if not got.is_contiguous():
        raise AssertionError("Task 78 output must be contiguous")


def metadata(case):
    c = case["spec"]
    return {"shape": list(c.shape), "dtypes": [str(d) for d in c.dtypes], "layouts": list(c.layouts)}
