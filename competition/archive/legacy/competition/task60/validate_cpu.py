#!/usr/bin/env python3
"""Run a bounded CPU semantic model for Task 60 candidates.

This executes original Triton JIT bodies serially with torch CPU tensors. It
does not compile Triton, emulate vendor device behavior, or measure speed. The
CPUModel pointer checker is shared with Task 78 so active lanes, bounds,
coverage, duplicate writes, and input immutability use one implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "competition/task78/validate_cpu.py"


def load_cpu_model():
    spec = importlib.util.spec_from_file_location("_flagos_task78_cpu_model", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared CPU model: {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_candidate(path: Path, model) -> ModuleType:
    source = path.read_bytes()
    module = ModuleType("_task60_candidate")
    module.__file__ = str(path)
    module.__dict__["range"] = model.kernel_range
    with model.installed():
        exec(compile(source, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "clamp_position", None)):
        raise AssertionError("candidate has no public clamp_position function")
    return module


def make_input(values: list[int], dtype, strided: bool) -> torch.Tensor:
    if not strided:
        return torch.tensor(values, dtype=dtype)
    backing = torch.zeros(len(values) * 2 + 3, dtype=dtype)
    view = torch.as_strided(backing, (len(values),), (2,), storage_offset=1)
    if values:
        view.copy_(torch.tensor(values, dtype=dtype))
    return view


def run_kernel_case(wrapper, model, values: list[int], dtype, strided: bool) -> int:
    source = make_input(values, dtype, strided)
    expected = (source - 1).clamp_min(0).contiguous()
    call = model.ValidationCall((source,))
    real_empty_like = torch.empty_like
    real_empty = torch.empty

    def tracked_empty_like(*args, **kwargs):
        return call.register_output(real_empty_like(*args, **kwargs))

    def tracked_empty(*args, **kwargs):
        return call.register_output(real_empty(*args, **kwargs))

    model.call = call
    try:
        with patch.object(torch, "empty_like", tracked_empty_like), patch.object(torch, "empty", tracked_empty):
            output = wrapper(source)
        call.check_inputs()
        call.check_output(output, expected)
    finally:
        model.call = None
    return len(call.launches)


def run_ascend_host_case(wrapper, values: list[int], dtype, strided: bool) -> None:
    source = make_input(values, dtype, strided)
    before = source.clone()
    expected = (source - 1).clamp_min(0).contiguous()
    with patch.dict(os.environ, {"DNN_VENDOR": "ascend"}):
        output = wrapper(source)
    if output.shape != expected.shape or output.dtype != expected.dtype:
        raise AssertionError("Ascend dispatch changed output shape or dtype")
    if not output.is_contiguous() or not torch.equal(output, expected):
        raise AssertionError("Ascend dispatch failed the clamp reference")
    if not torch.equal(source, before):
        raise AssertionError("Ascend dispatch mutated its input")


def run(source_path: Path, verbose: bool = False) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is unavailable; local semantic state is inconclusive until a torch-enabled runner is used")
    source_bytes = source_path.read_bytes()
    print(f"source={source_path} sha256={hashlib.sha256(source_bytes).hexdigest()}")
    model_module = load_cpu_model()
    model = model_module.CPUModel()
    with model.installed():
        module = load_candidate(source_path, model)
        wrapper = module.clamp_position
        cases = 0
        launches = 0
        values_by_size = {
            0: [], 1: [-7], 31: list(range(-15, 16)), 32: list(range(-16, 16)),
            33: list(range(-16, 17)), 255: [index - 127 for index in range(255)],
            256: [index - 128 for index in range(256)],
            257: [index - 128 for index in range(257)],
            1023: [index % 23 - 11 for index in range(1023)],
            1024: [index % 29 - 14 for index in range(1024)],
            1025: [index % 31 - 15 for index in range(1025)],
        }
        for dtype in (torch.int32, torch.int64, torch.float32):
            for size, values in values_by_size.items():
                for strided in (False, True):
                    launched = run_kernel_case(wrapper, model, values, dtype, strided)
                    cases += 1
                    launches += launched
                    if verbose:
                        print(f"PASS generic size={size} dtype={dtype} strided={strided} launches={launched}")
        # Ascend's host path is ordinary torch arithmetic; exercise its dispatch
        # separately because a CPU device cannot enter the NPU Triton launcher.
        for dtype in (torch.int32, torch.int64, torch.float32):
            for size, values in values_by_size.items():
                for strided in (False, True):
                    run_ascend_host_case(wrapper, values, dtype, strided)
                    cases += 1
        print(f"PASS: {cases}/{cases} CPU semantic cases; {launches} generic Triton-model launches")
        print("LIMIT: no vendor compiler, accelerator correctness, device limits, or performance evidence")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("clamp_position.py"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        return run(args.source.resolve(), args.verbose)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
