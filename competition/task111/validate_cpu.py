#!/usr/bin/env python3
"""Run the bounded CPU semantic model for Task 111 candidates.

This executes the original Triton JIT body serially with torch CPU tensors
through the shared ``competition/experiments/cpu_model.py``.  It does **not**
compile Triton, emulate a vendor device, or measure speed, and it is not
evidence about any competition target.  What it does establish, for exactly the
source hash printed, is:

* the public entry runs through the real wrapper and launch path;
* every output element is written exactly once, inside the returned view;
* inputs and their backing storage (including the overlapping ``as_strided``
  source) are not mutated;
* the result equals the official reference transcript for every development
  case, including the all-invalid case that must return a plain copy.

Usage::

    python competition/task111/validate_cpu.py [--case ID] [--verbose]
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
TASK = ROOT / "competition/task111"
MODEL_PATH = ROOT / "competition/experiments/cpu_model.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    module = ModuleType("_task111_candidate")
    module.__file__ = str(path)
    module.__dict__["range"] = model.kernel_range
    with model.installed():
        exec(compile(source, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "conv_window_scatter_with_mask", None)):
        raise AssertionError(
            "candidate has no public conv_window_scatter_with_mask function"
        )
    return module


def check_coverage(cpu_model, call, tensor, label, allow_rewrite=False):
    """Require coverage, exactly once unless a multi-pass design opts in.

    A two-pass schedule (copy every slot, then overwrite the touched ones)
    legitimately writes some elements twice.  With ``allow_rewrite`` the
    duplicate writes are tolerated and *returned*, because their count is
    exactly the traffic that schedule pays for -- a measured number behind the
    cost model instead of an assertion.

    Trees outside the returned view are always rejected: writing past the view
    is a bug in every schedule.
    """
    if tensor.numel() == 0:
        return 0
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
    missing = int((counts == 0).sum())
    multiple = int((counts > 1).sum())
    if missing:
        raise AssertionError(f"{label}: write coverage missing={missing}")
    if multiple and not allow_rewrite:
        raise AssertionError(f"{label}: write coverage multiple={multiple}")
    if int(allocation.writes.sum()) - tensor.numel() != multiple:
        raise AssertionError(
            f"{label}: kernel wrote outside the returned view"
        )
    return multiple


def run_case(
    adapter,
    cpu_model,
    model,
    wrapper,
    case,
    seed,
    verbose,
    allow_rewrite=False,
):
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

    if not isinstance(got, torch.Tensor):
        raise AssertionError("wrapper must return a single tensor")
    rewrites = check_coverage(
        cpu_model, call, got, f"{case['id']}.out", allow_rewrite=allow_rewrite
    )
    try:
        adapter.check(got, expected)
    except AssertionError as exc:
        raise AssertionError(f"{case['id']}.out: {exc}") from exc

    grid = call.launches[-1][1] if call.launches else ()
    programs = 0
    for _, shape in call.launches:
        count = 1
        for axis in shape:
            count *= axis
        programs += count
    if verbose:
        print(
            f"PASS {case['id']}: layers={case['layers']} cache={case['cache']} "
            f"requests={case['requests']} draft={case['draft']} "
            f"dim={case['dim']} window={case['window']} "
            f"dtype={case['dtype']} all_invalid={case['all_invalid']} "
            f"grid={grid} programs={programs} launches={len(call.launches)} "
            f"rewrites={rewrites}"
        )
    return programs


def run(
    source_path: Path, only=None, seed=0, verbose=False, allow_rewrite=False
) -> int:
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
    adapter = load_module("_task111_adapter", TASK / "adapter.py")
    model = cpu_model.CPUModel()
    model.allow_rewrite = allow_rewrite
    with model.installed():
        module = load_candidate(source_path, model)
        wrapper = module.conv_window_scatter_with_mask
        suite = adapter.cases()
        if only:
            suite = [case for case in suite if case["id"] in only]
            if not suite:
                raise AssertionError(f"no case matches {sorted(only)}")
        cases = programs = 0
        for case in suite:
            programs += run_case(
                adapter,
                cpu_model,
                model,
                wrapper,
                case,
                seed,
                verbose,
                allow_rewrite,
            )
            cases += 1
    print(
        f"PASS: {cases}/{cases} CPU semantic cases; {programs} model programs "
        "(program count is a structural fact, not a speed measurement)"
    )
    print(
        "LIMIT: no vendor compiler, accelerator correctness, device limits, "
        "or performance evidence; this is not target evidence"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=TASK / "conv_window_scatter_with_mask.py",
    )
    parser.add_argument("--case", action="append", dest="only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--allow-rewrite",
        action="store_true",
        help="accept a multi-pass schedule that rewrites touched elements",
    )
    args = parser.parse_args()
    try:
        return run(
            args.source.resolve(),
            args.only,
            args.seed,
            args.verbose,
            args.allow_rewrite,
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
