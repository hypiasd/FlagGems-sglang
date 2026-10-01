#!/usr/bin/env python3
"""Run the bounded CPU semantic model for Task 112 candidates.

This executes the original Triton JIT body serially with torch CPU tensors
through the shared ``competition/experiments/cpu_model.py``.  It does **not**
compile Triton, emulate a vendor device, or measure speed, and it is not
evidence about any competition target.  What it does establish, for exactly
the source hash printed, is:

* the public entry runs through the real wrapper and launch path;
* every output element is written exactly once, inside the returned view;
* inputs and their backing storage are not mutated;
* the numeric result matches the official reference within the task tolerance
  for every development case, including the NaN / +inf sanitization and the
  ``exp2`` / ``log2`` branch.

Usage::

    python competition/task112/validate_cpu.py [--case ID] [--verbose]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "competition/task112"
MODEL_PATH = ROOT / "competition/experiments/cpu_model.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ATOL = 0.015
RTOL = 0.015
LSE_ATOL = 1e-4
LSE_RTOL = 1e-4
MAX_PROGRAMS = 4096


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_candidate(path, model) -> ModuleType:
    source = path.read_bytes()
    module = ModuleType("_task112_candidate")
    module.__file__ = str(path)
    module.__dict__["range"] = model.kernel_range
    with model.installed():
        exec(compile(source, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "dcp_lse_combine", None)):
        raise AssertionError(
            "candidate has no public dcp_lse_combine function"
        )
    return module


def check_coverage(cpu_model, call, tensor, label):
    if tensor.numel() == 0:
        return
    key = cpu_model.storage_key(tensor)
    allocation = call.allocations.get(key)
    if allocation is None:
        raise AssertionError(f"{label}: output was never passed to a kernel")
    if call.outputs.get(id(tensor)) is not tensor:
        raise AssertionError(
            f"{label}: returned tensor is not the allocated output"
        )
    indices = cpu_model.logical_offsets(tensor)
    counts = allocation.writes[indices]
    if not bool((counts == 1).all()):
        missing = int((counts == 0).sum())
        multiple = int((counts > 1).sum())
        raise AssertionError(
            f"{label}: write coverage missing={missing} multiple={multiple}"
        )
    if int(allocation.writes.sum()) != tensor.numel():
        raise AssertionError(
            f"{label}: kernel wrote outside the returned view"
        )


def run_case(adapter, cpu_model, model, wrapper, case, seed, verbose):
    values = adapter.inputs(case, "cpu", seed)
    expected = adapter.reference(*values)
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    call = cpu_model.ValidationCall(tensors, max_programs=MAX_PROGRAMS)
    model.call = call
    real_empty, real_empty_like = torch.empty, torch.empty_like

    def tracked_empty(*args, **kwargs):
        return call.register_output(real_empty(*args, **kwargs))

    def tracked_empty_like(*args, **kwargs):
        return call.register_output(real_empty_like(*args, **kwargs))

    try:
        with patch.object(torch, "empty", tracked_empty), patch.object(
            torch, "empty_like", tracked_empty_like
        ):
            got = wrapper(*values)
    finally:
        model.call = None
    call.check_inputs()

    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError("wrapper must return exactly two values")
    check_coverage(cpu_model, call, got[0], f"{case['id']}.out")
    try:
        torch.testing.assert_close(
            got[0], expected[0], atol=ATOL, rtol=RTOL, equal_nan=True
        )
    except AssertionError as exc:
        raise AssertionError(f"{case['id']}.out: {exc}") from exc
    if expected[1] is None:
        if got[1] is not None:
            raise AssertionError(
                f"{case['id']}: return_lse is False but an LSE came back"
            )
    else:
        if got[1] is None:
            raise AssertionError(
                f"{case['id']}: return_lse is True but no LSE came back"
            )
        check_coverage(cpu_model, call, got[1], f"{case['id']}.lse")
        try:
            torch.testing.assert_close(
                got[1].to(torch.float32),
                expected[1].to(torch.float32),
                atol=LSE_ATOL,
                rtol=LSE_RTOL,
                equal_nan=True,
            )
        except AssertionError as exc:
            raise AssertionError(f"{case['id']}.lse: {exc}") from exc
    if verbose:
        launches = sum(1 for _ in call.launches)
        print(
            f"PASS {case['id']}: N={case['n']} B={case['b']} H={case['h']} "
            f"D={case['d']} base_e={case['base_e']} "
            f"return_lse={case['return_lse']} "
            f"dead_shards={case['dead_shards']} launches={launches}"
        )
    return len(call.launches)


def run(source_path: Path, only=None, seed=0, verbose=False) -> int:
    if torch is None:
        raise RuntimeError(
            "PyTorch is unavailable; local semantic state is inconclusive "
            "until a torch-enabled runner is used"
        )
    source_bytes = source_path.read_bytes()
    print(
        f"source={source_path} sha256={hashlib.sha256(source_bytes).hexdigest()}"
    )
    cpu_model = load_module("_flagos_shared_cpu_model", MODEL_PATH)
    adapter = load_module("_task112_adapter", TASK / "adapter.py")
    model = cpu_model.CPUModel()
    with model.installed():
        module = load_candidate(source_path, model)
        wrapper = module.dcp_lse_combine
        suite = adapter.cases()
        if only:
            suite = [case for case in suite if case["id"] in only]
            if not suite:
                raise AssertionError(f"no case matches {sorted(only)}")
        cases = launches = 0
        for case in suite:
            launches += run_case(
                adapter, cpu_model, model, wrapper, case, seed, verbose
            )
            cases += 1
    print(f"PASS: {cases}/{cases} CPU semantic cases; {launches} launches")
    print(
        "LIMIT: no vendor compiler, accelerator correctness, device limits, "
        "or performance evidence; this is not target evidence"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=TASK / "dcp_lse_combine.py"
    )
    parser.add_argument("--case", action="append", dest="only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        return run(args.source.resolve(), args.only, args.seed, args.verbose)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
