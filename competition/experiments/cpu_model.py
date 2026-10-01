"""Bounded CPU Triton model; provides no device or performance evidence."""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import itertools
import json
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch


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
    offsets = torch.full(
        tensor.shape, tensor.storage_offset(), dtype=torch.int64
    )
    for axis, (size, stride) in enumerate(zip(tensor.shape, tensor.stride())):
        shape = [1] * tensor.ndim
        shape[axis] = size
        offsets += (
            torch.arange(size, dtype=torch.int64).reshape(shape) * stride
        )
    return offsets.reshape(-1)


def offset_tensor(value):
    value = torch.as_tensor(value, device="cpu")
    require(
        value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64),
        "pointer offsets must be integers",
    )
    return value.to(torch.int64)


class Allocation:
    def __init__(self, tensor, readonly):
        self.storage = storage_vector(tensor)
        self.readonly = readonly
        self.writes = torch.zeros(self.storage.numel(), dtype=torch.int64)


class _ElementDtype:
    """Minimal stand-in for the element type of a Triton pointer."""

    def __init__(self, element_ty):
        self.element_ty = element_ty


class Ptr:
    """Element pointer whose offset is always a CPU int64 torch tensor."""

    def __init__(self, allocation, allowed, offsets):
        self.allocation = allocation
        self.allowed = allowed
        self.offsets = offset_tensor(offsets)

    @property
    def dtype(self):
        """Mirror ``ptr.dtype.element_ty`` so casts to the output dtype run."""
        return _ElementDtype(self.allocation.storage.dtype)

    def __add__(self, other):
        return Ptr(
            self.allocation, self.allowed, self.offsets + offset_tensor(other)
        )

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
        require(
            not invalid.any(),
            f"{operation}: active address outside storage [0, {size}): "
            f"{addresses[invalid][:8].tolist()}",
        )
        require(
            self.allowed[addresses].all(),
            f"{operation}: active address outside the passed tensor view",
        )
        return mask, addresses


class RestrictedModule(ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise NotImplementedError(
            f"unsupported CPU-model operation: {self.__name__}.{name}"
        )


class PythonJIT:
    """Bind real function arguments, then run the original body per program."""

    def __init__(self, function, model):
        self.fn = function
        self.model = model
        self.signature = inspect.signature(function)
        self.autotune_configs = []
        self.autotune_defaults = {}
        functools.update_wrapper(self, function)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            require(
                self.model.call is not None,
                "kernel launched outside a validation call",
            )
            kwargs = dict(kwargs)
            configs = getattr(self, "autotune_configs", [])
            if configs:
                config_index = min(self.model.autotune_index, len(configs) - 1)
                defaults = configs[config_index]
            else:
                defaults = getattr(self, "autotune_defaults", {})
            for name, value in defaults.items():
                kwargs.setdefault(name, value)
            for option in ("num_warps", "num_stages"):
                kwargs.pop(option, None)
            # Unknown options/arguments must fail; do not silently drop them.
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            launch_grid = (
                grid(dict(bound.arguments)) if callable(grid) else grid
            )
            if isinstance(launch_grid, int):
                launch_grid = (launch_grid,)
            launch_grid = tuple(int(x) for x in launch_grid)
            require(
                1 <= len(launch_grid) <= 3 and all(x > 0 for x in launch_grid),
                f"invalid launch grid: {launch_grid}",
            )
            launch_grid += (1,) * (3 - len(launch_grid))
            for name, value in bound.arguments.items():
                if isinstance(value, torch.Tensor):
                    bound.arguments[name] = self.model.call.pointer(value)
            call = self.model.call
            call.launches.append((self.__name__, launch_grid))
            total_programs = launch_grid[0] * launch_grid[1] * launch_grid[2]
            require(
                call.max_programs is None
                or total_programs <= call.max_programs,
                f"total launch grid {launch_grid} has {total_programs} programs; "
                f"limit is {call.max_programs}",
            )
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
        self.autotune_index = 0
        self.pid = self.grid = None
        self.triton = RestrictedModule("triton")
        self.triton.__path__ = []
        self.tl = RestrictedModule("triton.language")
        self.triton.language = self.tl
        self.triton.jit = self.jit
        self.triton.cdiv = lambda x, y: (x + y - 1) // y
        self.triton.next_power_of_2 = self.next_power_of_2
        # These decorators/config objects affect device compilation and
        # autotuning, not the serial CPU semantic model. Treat them as
        # identity wrappers here so candidates can be checked for indexing
        # and coverage without silently running a different algorithm.
        self.triton.autotune = self.autotune
        self.triton.heuristics = self.identity_decorator
        self.triton.Config = self.KernelConfig
        self.tl.constexpr = type("constexpr", (), {})
        for name in (
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "bfloat16",
            "float32",
            "float64",
        ):
            setattr(self.tl, name, getattr(torch, name))
        self.tl.program_id = self.program_id
        self.tl.num_programs = self.num_programs
        self.tl.arange = self.arange
        self.tl.static_range = range
        self.tl.where = torch.where
        self.tl.broadcast_to = torch.broadcast_to
        self.tl.multiple_of = lambda value, _alignment: value
        self.tl.max_contiguous = lambda value, _alignment: value
        self.tl.maximum = lambda x, y: torch.maximum(
            torch.as_tensor(x), torch.as_tensor(y)
        )
        # Elementwise math and constructors used by attention-style kernels.
        # Additions only: they change no existing indexing or coverage check.
        self.tl.exp = torch.exp
        self.tl.exp2 = torch.exp2
        self.tl.log = torch.log
        self.tl.log2 = torch.log2
        self.tl.zeros = lambda shape, dtype=None: torch.zeros(
            shape, dtype=dtype if dtype is not None else torch.float32
        )
        self.tl.full = lambda shape, value, dtype=None: torch.full(
            shape, value, dtype=dtype if dtype is not None else torch.float32
        )
        self.tl.load = self.load
        self.tl.store = self.store

    def jit(self, function=None):
        return (
            (lambda fn: PythonJIT(fn, self))
            if function is None
            else PythonJIT(function, self)
        )

    class KernelConfig:
        def __init__(self, values=None, **kwargs):
            self.values = dict(values or {})
            self.options = kwargs

    def autotune(self, configs=None, **_kwargs):
        """Expose every autotune config to the optional CPU semantic sweep."""
        configs = list(configs or ())

        def decorate(function):
            if configs and isinstance(function, PythonJIT):
                function.autotune_configs = [
                    dict(getattr(config, "values", {})) for config in configs
                ]
                function.autotune_defaults = dict(function.autotune_configs[0])
            return function

        return decorate

    @staticmethod
    def identity_decorator(*_args, **_kwargs):
        return lambda function: function

    @staticmethod
    def next_power_of_2(value):
        require(value > 0, "next_power_of_2 requires a positive integer")
        return 1 << (int(value) - 1).bit_length()

    def program_id(self, axis=0):
        require(
            self.pid is not None and axis in (0, 1, 2),
            "invalid program_id context/axis",
        )
        return torch.tensor(self.pid[axis], dtype=torch.int64)

    def num_programs(self, axis=0):
        require(
            self.grid is not None and axis in (0, 1, 2),
            "invalid num_programs context/axis",
        )
        return torch.tensor(self.grid[axis], dtype=torch.int64)

    @staticmethod
    def kernel_range(*args):
        for value in range(*args):
            yield torch.tensor(value, dtype=torch.int64)

    @staticmethod
    def arange(start, end):
        start, end = int(start), int(end)
        require(
            start == 0 and end > 0 and end & (end - 1) == 0,
            "CPU model supports only tl.arange(0, positive_power_of_two)",
        )
        return torch.arange(start, end, dtype=torch.int64)

    @staticmethod
    def load(pointer, mask=None, other=None, **kwargs):
        unsupported = set(kwargs) - {
            "cache_modifier",
            "eviction_policy",
            "padding_option",
            "boundary_check",
            "volatile",
        }
        require(
            not unsupported,
            f"tl.load unsupported keyword(s): {sorted(unsupported)}",
        )
        require(isinstance(pointer, Ptr), "tl.load requires a Ptr")
        active, addresses = pointer.active(mask, "load")
        allocation = pointer.allocation
        if not allocation.readonly:
            require(
                (allocation.writes[addresses] == 1).all(),
                "load from unwritten output",
            )
        if other is None:
            require(
                active.all(),
                "masked tl.load requires explicit other in this CPU model",
            )
            other = 0
        values = torch.as_tensor(
            other, dtype=allocation.storage.dtype, device="cpu"
        )
        values = torch.broadcast_to(values, pointer.offsets.shape).clone()
        values[active] = allocation.storage[addresses]
        return values

    @staticmethod
    def store(pointer, value, mask=None, **kwargs):
        unsupported = set(kwargs) - {"cache_modifier", "eviction_policy"}
        require(
            not unsupported,
            f"tl.store unsupported keyword(s): {sorted(unsupported)}",
        )
        require(isinstance(pointer, Ptr), "tl.store requires a Ptr")
        active, addresses = pointer.active(mask, "store")
        allocation = pointer.allocation
        require(
            not allocation.readonly or addresses.numel() == 0,
            "store to protected input/k backing storage",
        )
        unique, counts = addresses.unique(return_counts=True)
        require(
            (counts == 1).all(), "duplicate output writes within one store"
        )
        require(
            (allocation.writes[unique] == 0).all(),
            "duplicate output writes across stores/programs/launches",
        )
        values = torch.as_tensor(
            value, dtype=allocation.storage.dtype, device="cpu"
        )
        values = torch.broadcast_to(values, pointer.offsets.shape)
        allocation.storage[addresses] = values[active]
        allocation.writes[unique] += 1

    @contextmanager
    def installed(self):
        with patch.dict(
            sys.modules, {"triton": self.triton, "triton.language": self.tl}
        ):
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
            self.snapshots.append(
                (
                    tensor,
                    key,
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor._version,
                    storage_vector(tensor).view(torch.uint8).clone(),
                )
            )

    def register_output(self, tensor):
        """Only the actual torch.empty result may become a writable argument."""
        key = storage_key(tensor)
        require(
            key not in self.allocations,
            "output allocation aliases an existing tensor",
        )
        self.outputs[id(tensor)] = tensor
        self.allocations[key] = Allocation(tensor, readonly=False)
        return tensor

    def pointer(self, tensor):
        key = storage_key(tensor)
        require(
            key in self.allocations,
            "unregistered tensor argument; expected input or allocated output",
        )
        allocation = self.allocations[key]
        require(
            allocation.readonly or self.outputs.get(id(tensor)) is tensor,
            "writable argument must be the allocated output tensor by identity",
        )
        require(
            tensor.dtype == allocation.storage.dtype,
            "dtype-reinterpreted aliases are unsupported",
        )
        view_key = (
            key,
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
        )
        if view_key not in self.pointers:
            allowed = torch.zeros(allocation.storage.numel(), dtype=torch.bool)
            allowed[logical_offsets(tensor)] = True
            self.pointers[view_key] = Ptr(
                allocation, allowed, tensor.storage_offset()
            )
        return self.pointers[view_key]

    def check_inputs(self):
        for (
            tensor,
            key,
            shape,
            strides,
            offset,
            version,
            before,
        ) in self.snapshots:
            require(
                storage_key(tensor) == key
                and tensor.shape == shape
                and tensor.stride() == strides
                and tensor.storage_offset() == offset,
                "input/k storage or metadata mutated",
            )
            require(tensor._version == version, "input/k modified in place")
            require(
                torch.equal(storage_vector(tensor).view(torch.uint8), before),
                "input/k backing storage mutated (including view padding)",
            )

    def check_output(self, output, expected):
        require(
            isinstance(output, torch.Tensor),
            "wrapper did not return a torch.Tensor",
        )
        require(output.device.type == "cpu", "wrapper output must stay on CPU")
        require(
            output.shape == expected.shape and output.dtype == expected.dtype,
            f"wrong output shape/dtype: {output.shape}/{output.dtype}",
        )
        require(output.is_contiguous(), "output must be contiguous")
        key = storage_key(output)
        allocation = self.allocations.get(key)
        require(
            allocation is None or not allocation.readonly,
            "output aliases input/k storage",
        )
        require(
            self.outputs.get(id(output)) is output,
            "returned tensor is not the wrapper's allocated output by identity",
        )
        if output.numel() == 0:
            require(
                not self.launches,
                f"empty output launched kernels: {self.launches}",
            )
            return
        require(self.launches, "nonempty output launched no kernels")
        require(
            allocation is not None,
            "returned output was never passed to a kernel",
        )
        indices = logical_offsets(output)
        counts = allocation.writes[indices]
        require(
            (counts == 1).all(),
            f"output write coverage: missing={(counts == 0).sum().item()}, "
            f"multiple={(counts > 1).sum().item()}",
        )
        require(
            allocation.writes.sum().item() == output.numel(),
            "kernel wrote outside the returned output view",
        )
        torch.testing.assert_close(
            output, expected, rtol=0, atol=0, equal_nan=True
        )
