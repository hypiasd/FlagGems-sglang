"""Task 60 semantics; integer boundaries, layouts and tile tails."""
from competition.experiments.checking import check_output, mirror


def cases():
    return [{"id": f"{dtype}-{size}-{layout}", "dtype": dtype, "size": size, "layout": layout}
            for dtype in ("int32", "int64", "float32")
            for size in (0, 1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025)
            for layout in ("contiguous", "strided")]


def quick_ids():
    return ["int32-1-contiguous", "int32-257-contiguous", "int64-1025-strided"]


def inputs(case, device, seed):
    import torch
    from competition.task60.validate_cpu import make_input
    values = [i % 31 - 15 for i in range(case["size"])]
    return (mirror(make_input(values, getattr(torch, case["dtype"]), case["layout"] == "strided"), device),)


def reference(seq_lens):
    return (seq_lens - 1).clamp_min(0).contiguous()


def check(got, want):
    check_output(got, want)


def metadata(case):
    return case
