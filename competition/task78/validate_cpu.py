#!/usr/bin/env python3
"""Execute Task 78 wrappers and their actual JIT function bodies on the CPU.

THIS IS NOT a Triton interpreter/compiler/perf test. It is a limited CPU
execution model using stdlib and torch only; Triton need not be installed.
Python if statements execute normally, tl vectors are torch tensors, and
programs execute serially. The imported operator module's range yields scalar
int64 tensors so dynamic loop variables support .to(); tl.static_range uses
Python's range. This does not validate compilation, device
integer/promotion rules, scheduling, races beyond duplicate output writes,
resource limits, or accelerator performance. It is not a Python sandbox.

Only the explicitly implemented Triton API is accepted. Active load/store
lanes must address the passed tensor view, including its actual storage offset
and strides; masked-off addresses may be invalid. All input backing storage
is protected. Every nonempty wrapper call must launch exactly one kernel and
write every output element exactly once across all its programs. Empty outputs
must not launch. Output values use exact numeric comparison with equal NaNs;
NaN payloads and signed-zero bit patterns are not compared.

Run from any directory:
    python3 competition/task78/validate_cpu.py             # all seven files
    python3 competition/task78/validate_cpu.py --all
    python3 competition/task78/validate_cpu.py --backend ascend
    python3 competition/task78/validate_cpu.py --backend default --verbose

Files are read as source snapshots without creating bytecode/cache files.
Kernel coverage is discovered dynamically, so newly added paths are reported.
Uninvoked definitions are reported too, but may be deliberately unused code.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import hashlib
import inspect
import itertools
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import torch


BACKENDS = ("default", "ascend", "enflame", "hygon", "iluvatar", "kunlunxin", "metax")
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
NOTICE = "THIS IS NOT a Triton interpreter/compiler/perf test; limited CPU execution model."


class ValidationError(AssertionError):
    """A semantic or memory-access contract was violated."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def storage_vector(tensor):
    """Address the entire underlying allocation, not a flattened view copy."""
    require(tensor.device.type == "cpu", "only CPU tensors are supported")
    return tensor.as_strided(
        (tensor.untyped_storage().nbytes() // tensor.element_size(),),
        (1,),
        storage_offset=0,
    )


def storage_key(tensor):
    # Unlike data_ptr(), this distinguishes separate zero-byte allocations.
    return tensor.untyped_storage()._cdata


def logical_offsets(tensor):
    offsets = torch.full(tensor.shape, tensor.storage_offset(), dtype=torch.int64)
    for axis, (size, stride) in enumerate(zip(tensor.shape, tensor.stride())):
        shape = [1] * tensor.ndim
        shape[axis] = size
        offsets += torch.arange(size, dtype=torch.int64).reshape(shape) * stride
    return offsets.reshape(-1)


def offset_tensor(value):
    value = torch.as_tensor(value, device="cpu")
    require(value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64),
            "pointer offsets must be integers")
    return value.to(torch.int64)


class Allocation:
    def __init__(self, tensor, readonly):
        self.storage = storage_vector(tensor)
        self.readonly = readonly
        self.writes = torch.zeros(self.storage.numel(), dtype=torch.int64)


class Ptr:
    """Element pointer whose offset is always a CPU int64 torch tensor."""

    def __init__(self, allocation, allowed, offsets):
        self.allocation = allocation
        self.allowed = allowed
        self.offsets = offset_tensor(offsets)

    def __add__(self, other):
        return Ptr(self.allocation, self.allowed, self.offsets + offset_tensor(other))

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-offset_tensor(other))

    def __getitem__(self, index):
        return Ptr(self.allocation, self.allowed, self.offsets[index])

    def active(self, mask, operation):
        mask = torch.as_tensor(True if mask is None else mask, device="cpu")
        require(mask.dtype == torch.bool, f"{operation}: mask must be boolean")
        # Masks and values broadcast TO the pointer shape, not vice versa.
        mask = torch.broadcast_to(mask, self.offsets.shape)
        addresses = self.offsets[mask]
        size = self.allocation.storage.numel()
        invalid = (addresses < 0) | (addresses >= size)
        require(not invalid.any(),
                f"{operation}: active address outside storage [0, {size}): "
                f"{addresses[invalid][:8].tolist()}")
        require(self.allowed[addresses].all(),
                f"{operation}: active address outside the passed tensor view")
        return mask, addresses


class RestrictedModule(ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise NotImplementedError(f"unsupported CPU-model operation: {self.__name__}.{name}")


class PythonJIT:
    """Bind real function arguments, then run the original body per program."""

    def __init__(self, function, model):
        self.fn = function
        self.model = model
        self.signature = inspect.signature(function)
        functools.update_wrapper(self, function)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            require(self.model.call is not None, "kernel launched outside a validation call")
            kwargs = dict(kwargs)
            for option in ("num_warps", "num_stages"):
                kwargs.pop(option, None)
            # Unknown options/arguments must fail; do not silently drop them.
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            launch_grid = grid(dict(bound.arguments)) if callable(grid) else grid
            if isinstance(launch_grid, int):
                launch_grid = (launch_grid,)
            launch_grid = tuple(int(x) for x in launch_grid)
            require(1 <= len(launch_grid) <= 3 and all(x > 0 for x in launch_grid),
                    f"invalid launch grid: {launch_grid}")
            launch_grid += (1,) * (3 - len(launch_grid))
            for name, value in bound.arguments.items():
                if isinstance(value, torch.Tensor):
                    bound.arguments[name] = self.model.call.pointer(value)
            call = self.model.call
            call.launches.append((self.__name__, launch_grid))
            total_programs = launch_grid[0] * launch_grid[1] * launch_grid[2]
            require(call.max_programs is None or total_programs <= call.max_programs,
                    f"total launch grid {launch_grid} has {total_programs} programs; "
                    f"limit is {call.max_programs}")
            self.model.grid = launch_grid
            try:
                for pid in itertools.product(*(range(n) for n in launch_grid)):
                    self.model.pid = pid
                    try:
                        self.fn(*bound.args, **bound.kwargs)
                    except Exception as exc:
                        raise ValidationError(
                            f"{self.__name__}, grid={launch_grid}, program={pid}: {exc}"
                        ) from exc
            finally:
                self.model.pid = self.model.grid = None
        return launch


class CPUModel:
    def __init__(self):
        self.call = None
        self.last_launches = []
        self.pid = self.grid = None
        self.triton = RestrictedModule("triton")
        self.triton.__path__ = []
        self.tl = RestrictedModule("triton.language")
        self.triton.language = self.tl
        self.triton.jit = self.jit
        self.triton.cdiv = lambda x, y: (x + y - 1) // y
        self.triton.next_power_of_2 = self.next_power_of_2
        self.tl.constexpr = type("constexpr", (), {})
        for name in ("int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"):
            setattr(self.tl, name, getattr(torch, name))
        self.tl.program_id = self.program_id
        self.tl.num_programs = self.num_programs
        self.tl.arange = self.arange
        self.tl.static_range = range
        self.tl.where = torch.where
        self.tl.maximum = lambda x, y: torch.maximum(torch.as_tensor(x), torch.as_tensor(y))
        self.tl.load = self.load
        self.tl.store = self.store

    def jit(self, function=None):
        return (lambda fn: PythonJIT(fn, self)) if function is None else PythonJIT(function, self)

    @staticmethod
    def next_power_of_2(value):
        require(value > 0, "next_power_of_2 requires a positive integer")
        return 1 << (int(value) - 1).bit_length()

    def program_id(self, axis=0):
        require(self.pid is not None and axis in (0, 1, 2), "invalid program_id context/axis")
        return torch.tensor(self.pid[axis], dtype=torch.int64)

    def num_programs(self, axis=0):
        require(self.grid is not None and axis in (0, 1, 2), "invalid num_programs context/axis")
        return torch.tensor(self.grid[axis], dtype=torch.int64)

    @staticmethod
    def kernel_range(*args):
        for value in range(*args):
            yield torch.tensor(value, dtype=torch.int64)

    @staticmethod
    def arange(start, end):
        start, end = int(start), int(end)
        require(start == 0 and end > 0 and end & (end - 1) == 0,
                "CPU model supports only tl.arange(0, positive_power_of_two)")
        return torch.arange(start, end, dtype=torch.int64)

    @staticmethod
    def load(pointer, mask=None, other=None):
        require(isinstance(pointer, Ptr), "tl.load requires a Ptr")
        active, addresses = pointer.active(mask, "load")
        allocation = pointer.allocation
        if not allocation.readonly:
            require((allocation.writes[addresses] == 1).all(), "load from unwritten output")
        if other is None:
            require(active.all(), "masked tl.load requires explicit other in this CPU model")
            other = 0
        values = torch.as_tensor(other, dtype=allocation.storage.dtype, device="cpu")
        values = torch.broadcast_to(values, pointer.offsets.shape).clone()
        values[active] = allocation.storage[addresses]
        return values

    @staticmethod
    def store(pointer, value, mask=None):
        require(isinstance(pointer, Ptr), "tl.store requires a Ptr")
        active, addresses = pointer.active(mask, "store")
        allocation = pointer.allocation
        require(not allocation.readonly or addresses.numel() == 0,
                "store to protected input/k backing storage")
        unique, counts = addresses.unique(return_counts=True)
        require((counts == 1).all(), "duplicate output writes within one store")
        require((allocation.writes[unique] == 0).all(),
                "duplicate output writes across stores/programs/launches")
        values = torch.as_tensor(value, dtype=allocation.storage.dtype, device="cpu")
        values = torch.broadcast_to(values, pointer.offsets.shape)
        allocation.storage[addresses] = values[active]
        allocation.writes[unique] += 1

    @contextmanager
    def installed(self):
        with patch.dict(sys.modules, {"triton": self.triton, "triton.language": self.tl}):
            yield


class ValidationCall:
    def __init__(self, inputs, max_programs=None):
        self.allocations = {}
        self.outputs = {}
        self.pointers = {}
        self.launches = []
        self.snapshots = []
        self.max_programs = max_programs
        for tensor in inputs:
            key = storage_key(tensor)
            if key not in self.allocations:
                self.allocations[key] = Allocation(tensor, readonly=True)
            self.snapshots.append((tensor, key, tensor.shape, tensor.stride(),
                                   tensor.storage_offset(), tensor._version,
                                   storage_vector(tensor).view(torch.uint8).clone()))

    def register_output(self, tensor):
        """Only the actual torch.empty result may become a writable argument."""
        key = storage_key(tensor)
        require(key not in self.allocations, "output allocation aliases an existing tensor")
        self.outputs[id(tensor)] = tensor
        self.allocations[key] = Allocation(tensor, readonly=False)
        return tensor

    def pointer(self, tensor):
        key = storage_key(tensor)
        require(key in self.allocations, "unregistered tensor argument; expected input or allocated output")
        allocation = self.allocations[key]
        require(allocation.readonly or self.outputs.get(id(tensor)) is tensor,
                "writable argument must be the allocated output tensor by identity")
        require(tensor.dtype == allocation.storage.dtype, "dtype-reinterpreted aliases are unsupported")
        view_key = (key, tuple(tensor.shape), tensor.stride(), tensor.storage_offset())
        if view_key not in self.pointers:
            allowed = torch.zeros(allocation.storage.numel(), dtype=torch.bool)
            allowed[logical_offsets(tensor)] = True
            self.pointers[view_key] = Ptr(allocation, allowed, tensor.storage_offset())
        return self.pointers[view_key]

    def check_inputs(self):
        for tensor, key, shape, strides, offset, version, before in self.snapshots:
            require(storage_key(tensor) == key and tensor.shape == shape
                    and tensor.stride() == strides and tensor.storage_offset() == offset,
                    "input/k storage or metadata mutated")
            require(tensor._version == version, "input/k modified in place")
            require(torch.equal(storage_vector(tensor).view(torch.uint8), before),
                    "input/k backing storage mutated (including view padding)")

    def check_output(self, output, expected):
        require(isinstance(output, torch.Tensor), "wrapper did not return a torch.Tensor")
        require(output.device.type == "cpu", "wrapper output must stay on CPU")
        require(output.shape == expected.shape and output.dtype == expected.dtype,
                f"wrong output shape/dtype: {output.shape}/{output.dtype}")
        require(output.is_contiguous(), "output must be contiguous")
        key = storage_key(output)
        allocation = self.allocations.get(key)
        require(allocation is None or not allocation.readonly, "output aliases input/k storage")
        require(self.outputs.get(id(output)) is output,
                "returned tensor is not the wrapper's allocated output by identity")
        if output.numel() == 0:
            require(not self.launches, f"empty output launched kernels: {self.launches}")
            return
        require(len(self.launches) == 1,
                f"expected a single launch, got {len(self.launches)}: {self.launches}")
        require(allocation is not None, "returned output was never passed to a kernel")
        indices = logical_offsets(output)
        counts = allocation.writes[indices]
        require((counts == 1).all(),
                f"output write coverage: missing={(counts == 0).sum().item()}, "
                f"multiple={(counts > 1).sum().item()}")
        require(allocation.writes.sum().item() == output.numel(),
                "kernel wrote outside the returned output view")
        torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)


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
    return result


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


def validate_backend(backend, suite, verbose):
    suffix = "" if backend == "default" else f"_{backend}"
    path = Path(__file__).resolve().with_name(f"concat_and_cast_mha_k{suffix}.py")
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:12]
    model = CPUModel()
    module = ModuleType(f"_task78_cpu_{backend}")
    module.__file__ = str(path)
    module.__dict__["range"] = model.kernel_range
    successes, failures = 0, 0
    invoked, programs, grids = Counter(), Counter(), Counter()
    print(f"\n[{backend}] {path.name} sha256={digest}", flush=True)
    with model.installed():
        exec(compile(source, str(path), "exec"), module.__dict__)
        declared = {obj.__name__ for obj in module.__dict__.values() if isinstance(obj, PythonJIT)}
        wrapper = module.concat_and_cast_mha_k
        for index, case in enumerate(suite):
            model.last_launches = []
            try:
                call = run_case(model, wrapper, case, seed=78000 + index,
                                max_programs=32 if backend == "ascend" else None)
            except Exception as exc:
                failures += 1
                print(f"  FAIL {case.name}: {exc}", flush=True)
            else:
                successes += 1
                if verbose:
                    print(f"  PASS {case.name}: {call.launches}", flush=True)
            finally:
                for name, grid in model.last_launches:
                    invoked[name] += 1
                    programs[name] += grid[0] * grid[1] * grid[2]
                    grids[name, grid] += 1
    print(f"[{backend}] {successes}/{len(suite)} cases passed; {failures} failed")
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
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--all", action="store_true", help="validate all seven backends (default)")
    choice.add_argument("--backend", choices=BACKENDS, help="filename suffix; default selects the unsuffixed file")
    parser.add_argument("--verbose", action="store_true", help="show each successful case and its launch grid")
    args = parser.parse_args(argv)
    print(NOTICE, flush=True)
    torch.set_num_threads(1)
    with torch.no_grad():
        self_check()
        suite = cases()
        print(f"Validator self-checks passed; {len(suite)} cases per backend.", flush=True)
        failures = 0
        for backend in (args.backend,) if args.backend else BACKENDS:
            try:
                failures += validate_backend(backend, suite, args.verbose)
            except Exception as exc:
                failures += 1
                print(f"[{backend}] FAIL loading/running backend: {exc}", flush=True)
    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} failures. {NOTICE}")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
