"""Task 78 semantic suite, using the shared CPU pointer model."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch
from dataclasses import dataclass
import torch
from competition.experiments.cpu_model import *
BACKENDS = ("default", "ascend", "enflame", "hygon", "iluvatar", "kunlunxin", "metax")
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, int, int, int]  # tokens, heads, NoPE dim, RoPE dim
    dtypes: tuple = (torch.float32, torch.float32, torch.float32)  # NoPE, RoPE, k
    layouts: tuple = ("contiguous", "contiguous", "contiguous")  # k, NoPE, RoPE


def make_tensor(shape, dtype, layout, generator):
    t, h, d = (max(1, n) for n in shape)
    strides = {
        "contiguous": (h * d, d, 1),
        "offset": (h * d, d, 1),
        "token": (2 * h * d, d, 1),
        "head": (3 * h * d, 3 * d, 1),
        "column": (2 * h * d, 2 * d, 2),
        "all": (2 * (3 * h + 1) * (2 * d + 1), 3 * (2 * d + 1), 2),
        "transpose": (1, t * d, t),
        "broadcast_token": (0, d, 1),
        "broadcast_head": (d, 0, 1),
    }[layout]
    offset = 0 if layout == "contiguous" else 7
    extent = 0 if 0 in shape else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    length = offset + extent + (0 if layout == "contiguous" else 7)
    data = torch.randn(length, generator=generator, dtype=torch.float32) * 17
    # Include rounding boundaries, signed zero, tiny/large values and nonfinites.
    special = torch.tensor([0.0, -0.0, 1.00048828125, 1.00390625, -1.00390625,
                            65504.0, 1e-8, -1e-8, float("inf"), -float("inf"), float("nan")])
    n = min(extent, special.numel())
    data[offset:offset + n] = special[:n]
    backing = data.to(dtype)
    return backing.as_strided(shape, strides, storage_offset=offset)


def cases():
    result = []
    # Empty outputs and zero-size source segments, including offset/strided views.
    for shape in ((0, 5, 7, 3), (3, 0, 7, 3), (3, 5, 0, 0), (0, 0, 0, 0),
                  (1, 1, 0, 7), (1, 1, 7, 0), (3, 5, 0, 65), (3, 5, 65, 0),
                  (65, 9, 0, 513), (65, 9, 513, 0)):
        for layout in ("contiguous", "all"):
            result.append(Case(f"empty-or-segment-{shape}-{layout}", shape,
                               (torch.float16, torch.bfloat16, torch.float32), (layout,) * 3))
    # Thresholds on both sides of common tile sizes, with full and partial tiles.
    dimensions = ((1, 1), (3, 5), (7, 9), (16, 16), (31, 33), (32, 64),
                  (63, 65), (64, 64), (96, 32), (127, 17), (128, 128),
                  (129, 65), (255, 17), (256, 64), (257, 31), (511, 33),
                  (512, 64), (513, 65), (1024, 128), (1025, 3), (33, 513))
    for dn, dr in dimensions:
        for tokens, heads in ((1, 1), (2, 8), (3, 5)):
            result.append(Case(f"tiles-{tokens}-{heads}-{dn}-{dr}", (tokens, heads, dn, dr),
                               (torch.float16, torch.bfloat16, torch.float16)))
    for tokens in (17, 33, 65, 67, 129):
        for dn, dr in ((7, 3), (129, 65), (513, 65)):
            result.append(Case(f"grid-loop-{tokens}-{dn}", (tokens, 9, dn, dr),
                               (torch.bfloat16, torch.float16, torch.bfloat16)))
    for heads in (2, 3, 4, 7, 8, 9, 15, 16, 17, 32):
        result.append(Case(f"head-tail-{heads}", (5, heads, 32, 16)))
    # All 27 source/destination dtype combinations on both main layout paths.
    for dtypes in itertools.product(DTYPES, repeat=3):
        for layout in ("contiguous", "all"):
            label = "-".join(str(dtype).split(".")[-1] for dtype in dtypes)
            result.append(Case(f"dtype-{label}-{layout}", (3, 5, 17, 9), dtypes, (layout,) * 3))
    # Independently perturb each input; k is only a shape/dtype carrier.
    for layout in ("offset", "token", "head", "column", "all", "transpose",
                   "broadcast_token", "broadcast_head"):
        for subject in range(3):
            layouts = ["contiguous"] * 3
            layouts[subject] = layout
            result.append(Case(f"view-{subject}-{layout}", (7, 9, 33, 17),
                               (torch.float32, torch.float16, torch.bfloat16), tuple(layouts)))
    for dn, dr in ((7, 3), (257, 129), (513, 65)):
        result.append(Case(f"strided-grid-loop-{dn}", (65, 9, dn, dr),
                           (torch.bfloat16, torch.float32, torch.float16),
                           ("all", "all", "transpose")))
    # Head dimensions past the widest capped tile (4096). A kernel that covers a
    # row with one tile and no column loop drops every column beyond its cap.
    for dn, dr in ((4096, 904), (4096, 4096), (5000, 32), (32, 5000),
                   (4097, 1), (513, 4096)):
        for layout in ("contiguous", "all"):
            result.append(Case(f"wide-dim-{dn}-{dr}-{layout}", (2, 3, dn, dr),
                               (torch.float16, torch.bfloat16, torch.float32),
                               (layout,) * 3))
    return result


def autotune_coverage_cases(suite):
    """Select the shapes most likely to expose config-dependent coverage bugs.

    The first autotune config still runs the complete suite.  Repeating all
    189 cases for every config is unnecessarily expensive on the CPU model;
    later configs exercise the boundary/stride/empty cases where a tile-size
    mismatch can create missing or duplicate writes.  ``--autotune-sweep-full``
    remains available for the release audit.
    """
    needles = (
        "empty-or-segment-",
        "grid-loop-",
        "strided-grid-loop-",
        "head-tail-",
        "view-",
    )
    tile_boundaries = ("-257-", "-511-", "-512-", "-513-", "-1024-", "-1025-", "-513")
    selected = []
    for case in suite:
        if case.name.startswith(needles):
            selected.append(case)
        elif case.name.startswith("tiles-") and any(token in case.name for token in tile_boundaries):
            selected.append(case)
        elif case.name.startswith("dtype-") and case.name.endswith("-all"):
            selected.append(case)
    require(selected, "autotune coverage suite must not be empty")
    return selected


def run_case(model, wrapper, case, seed, max_programs=None):
    tokens, heads, dn, dr = case.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    nope_dtype, rope_dtype, out_dtype = case.dtypes
    k = make_tensor((tokens, heads, dn + dr), out_dtype, case.layouts[0], generator)
    nope = make_tensor((tokens, heads, dn), nope_dtype, case.layouts[1], generator)
    rope = make_tensor((tokens, 1, dr), rope_dtype, case.layouts[2], generator)
    expected = torch.cat([nope, rope.expand(-1, heads, -1)], dim=-1).to(k.dtype)
    call = ValidationCall((k, nope, rope), max_programs=max_programs)
    real_empty = torch.empty

    def tracked_empty(*args, **kwargs):
        return call.register_output(real_empty(*args, **kwargs))

    model.call = call
    try:
        try:
            with patch.object(torch, "empty", tracked_empty):
                output = wrapper(k, nope, rope)
        finally:
            call.check_inputs()
        call.check_output(output, expected)
    finally:
        model.last_launches = list(call.launches)
        model.call = None
    return call


def self_check():
    """Negative controls ensure the validator actually rejects faulty kernels."""
    model = CPUModel()
    tl = model.tl
    source = torch.arange(16, dtype=torch.float32)[3:11:2]
    expected = source.clone()

    @model.jit
    def copy(out, src, N: tl.constexpr, BLOCK: tl.constexpr = 8):
        pid = tl.program_id(0).to(tl.int64)
        for first in model.kernel_range(pid * BLOCK, N, tl.num_programs(0) * BLOCK):
            cols = first.to(tl.int64) + tl.arange(0, BLOCK)
            tl.store(out + cols, tl.load(src + cols * 2, cols < N, other=0), cols < N)

    output = torch.empty_like(expected)
    call = ValidationCall((source,))
    call.register_output(output)
    model.call = call
    copy[lambda meta: (model.triton.cdiv(meta["N"], meta["BLOCK"]),)](
        src=source, out=output, N=source.numel(), num_warps=1, num_stages=1,
    )
    call.check_inputs()
    call.check_output(output, expected)
    src = call.pointer(source)
    # Huge/negative inactive addresses must never be dereferenced or truncated.
    masked = src + torch.tensor([0, -(2 ** 40), 2 ** 40], dtype=torch.int64)
    require(torch.equal(tl.load(masked, torch.tensor([True, False, False]), other=-1),
                        torch.tensor([3.0, -1.0, -1.0])), "masked-load self-check failed")

    def rejected(label, action, text):
        try:
            action()
        except (ValidationError, NotImplementedError, TypeError) as exc:
            require(text in str(exc), f"self-check {label}: unexpected error: {exc}")
        else:
            raise ValidationError(f"self-check {label}: bad operation was accepted")

    rejected("unsupported tl", lambda: tl.full((1,), 0, tl.float32), "unsupported")
    rejected("unknown launch option", lambda: copy[(1,)](output, source, 4, typo=True), "typo")
    rejected("input store", lambda: tl.store(src, 0), "protected")
    rejected("view gap load", lambda: tl.load(src + 1), "tensor view")
    rejected("negative load", lambda: tl.load(src - 4), "outside storage")
    rejected("large load", lambda: tl.load(src + 2 ** 40), "outside storage")
    dst = call.pointer(call.register_output(torch.empty(4)))
    tl.store(dst + torch.tensor([-1, 2 ** 40]), 0, torch.tensor([False, False]))
    rejected("store bounds", lambda: tl.store(dst + 4, 0), "outside storage")
    rejected("unwritten load", lambda: tl.load(dst), "unwritten")
    rejected("duplicate lanes", lambda: tl.store(dst + torch.tensor([0, 0]), 1), "within one store")
    tl.store(dst, 1)
    rejected("duplicate stores", lambda: tl.store(dst, 1), "across stores")
    missing = torch.empty_like(expected)
    incomplete = ValidationCall(())
    incomplete.register_output(missing)
    incomplete.pointer(missing)
    incomplete.launches.append(("deliberately_incomplete", (1, 1, 1)))
    rejected("missing writes", lambda: incomplete.check_output(missing, expected), "coverage")

    @model.jit
    def repeated(out):
        tl.store(out, 1)

    model.call = ValidationCall(())
    repeated_output = model.call.register_output(torch.empty(1))
    rejected("duplicate programs", lambda: repeated[(2,)](repeated_output), "across stores")
    model.call = ValidationCall((), max_programs=32)
    bounded_output = model.call.register_output(torch.empty(1))
    rejected("total grid cap", lambda: repeated[(2, 3, 6)](bounded_output), "limit is 32")
    rejected("unknown tensor", lambda: model.call.pointer(torch.empty(1)), "unregistered")
    rejected("output alias identity", lambda: model.call.pointer(bounded_output.view(1)), "identity")
    model.call = None


def validate_backend(backend, suite, verbose, source_dir=None,
                     autotune_sweep=False, autotune_sweep_full=False):
    suffix = "" if backend == "default" else f"_{backend}"
    source_dir = Path(__file__).resolve().parent if source_dir is None else Path(source_dir)
    path = source_dir / f"concat_and_cast_mha_k{suffix}.py"
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:12]
    model = CPUModel()
    module = ModuleType(f"_task78_cpu_{backend}")
    module.__file__ = str(path)
    module.__dict__["range"] = model.kernel_range
    successes, failures = 0, 0
    invoked, programs, grids = Counter(), Counter(), Counter()
    print(f"\n[{backend}] {path} sha256={digest}", flush=True)
    with model.installed():
        exec(compile(source, str(path), "exec"), module.__dict__)
        declared_objects = [obj for obj in module.__dict__.values() if isinstance(obj, PythonJIT)]
        declared = {obj.__name__ for obj in declared_objects}
        available_configs = max(
            (len(getattr(obj, "autotune_configs", ())) for obj in declared_objects),
            default=0,
        )
        config_count = available_configs if autotune_sweep and available_configs else 1
        wrapper = module.concat_and_cast_mha_k
        boundary_suite = autotune_coverage_cases(suite)
        executed_case_count = 0
        for config_index in range(config_count):
            model.autotune_index = config_index
            config_suite = suite if (not autotune_sweep or config_index == 0 or autotune_sweep_full) else boundary_suite
            executed_case_count += len(config_suite)
            for index, case in enumerate(config_suite):
                model.last_launches = []
                try:
                    call = run_case(model, wrapper, case, seed=78000 + index,
                                    max_programs=32 if backend == "ascend" else None)
                except Exception as exc:
                    failures += 1
                    suffix = f" config={config_index}" if autotune_sweep else ""
                    print(f"  FAIL {case.name}{suffix}: {exc}", flush=True)
                else:
                    successes += 1
                    if verbose:
                        suffix = f" config={config_index}" if autotune_sweep else ""
                        print(f"  PASS {case.name}{suffix}: {call.launches}", flush=True)
                finally:
                    for name, grid in model.last_launches:
                        invoked[name] += 1
                        programs[name] += grid[0] * grid[1] * grid[2]
                        grids[name, grid] += 1
    model.autotune_index = 0
    total_cases = executed_case_count
    print(f"[{backend}] {successes}/{total_cases} cases passed; {failures} failed "
          f"({config_count} autotune config(s) exercised)")
    for name in sorted(invoked):
        shapes = [grid for kernel, grid in grids if kernel == name]
        largest = max(grid[0] * grid[1] * grid[2] for grid in shapes)
        print(f"  invoked {name}: {invoked[name]} launches, {programs[name]} total programs, "
              f"{len(shapes)} grid shapes, max {largest} programs/launch")
        if verbose:
            for grid in sorted(shapes):
                print(f"    grid={grid}: {grids[name, grid]} launches")
    unseen = declared - invoked.keys()
    if unseen:
        print("  not invoked: " + ", ".join(sorted(unseen)))
    if path.read_bytes() != source:
        print("  FAIL source changed during validation; rerun against the current file")
        failures += 1
    return failures, config_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--all", action="store_true", help="validate all seven backends (default)")
    choice.add_argument("--backend", choices=BACKENDS, help="filename suffix; default selects the unsuffixed file")
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="directory containing the seven candidate files; defaults to this script's directory")
    parser.add_argument("--json", dest="json_path", type=Path, default=None,
                        help="write a machine-readable summary to this path")
    parser.add_argument("--verbose", action="store_true", help="show each successful case and its launch grid")
    parser.add_argument(
        "--autotune-sweep", action="store_true",
        help="run all cases for the first config and boundary/stride cases for every remaining config",
    )
    parser.add_argument(
        "--autotune-sweep-full", action="store_true",
        help="run all cases for every visible autotune config (slow release audit)",
    )
    args = parser.parse_args(argv)
    print(NOTICE, flush=True)
    torch.set_num_threads(1)
    with torch.no_grad():
        self_check()
        suite = cases()
        print(f"Validator self-checks passed; {len(suite)} cases per backend.", flush=True)
        failures = 0
        results = []
        for backend in (args.backend,) if args.backend else BACKENDS:
            try:
                before = failures
                sweep = args.autotune_sweep or args.autotune_sweep_full
                backend_failures, config_count = validate_backend(
                    backend, suite, args.verbose, args.source_dir, sweep,
                    args.autotune_sweep_full,
                )
                failures += backend_failures
                results.append({
                    "backend": backend,
                    "passed": backend_failures == 0,
                    "failures": backend_failures,
                    "autotune_configs": config_count,
                })
            except Exception as exc:
                failures += 1
                print(f"[{backend}] FAIL loading/running backend: {exc}", flush=True)
                results.append({
                    "backend": backend,
                    "passed": False,
                    "failures": 1,
                    "autotune_configs": None,
                    "error": str(exc),
                })
    summary = {
        "passed": failures == 0,
        "failures": failures,
        "backends": results,
        "source_dir": str(args.source_dir or Path(__file__).resolve().parent),
        "case_count": len(suite),
        "autotune_sweep": args.autotune_sweep or args.autotune_sweep_full,
        "autotune_sweep_full": args.autotune_sweep_full,
        "notice": NOTICE,
    }
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} failures. {NOTICE}")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
