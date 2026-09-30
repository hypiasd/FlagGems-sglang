#!/usr/bin/env python3
"""Non-target GPU smoke runner for Task 78 candidate sources.

This script executes on a machine that has a real, Triton-capable accelerator.
It reuses the *exact* Task 78 case matrix, tensor layouts, and seeded inputs from
``validate_cpu.py`` (same shapes, dtypes, strides, storage offsets, and values),
but runs the candidate wrappers through real Triton compilation and real kernel
launches instead of the CPU execution model.  That adds the signal the CPU model
structurally cannot provide: Triton/LLVM compilation, launch-argument handling,
real memory access, and per-autotune-config execution.

It is NOT target evidence.  The device is not one of the eight declared Task 78
targets, so a pass here says nothing about Iluvatar, MetaX, Enflame, Hygon,
Kunlunxin, Ascend, International A, or International B.  The report is labelled
``evidence_class: nvidia-smoke-nontarget`` and must never be recorded as
``target_validated`` or ``measured`` evidence.

Usage (normally driven by ``gpu_smoke.py``, but usable standalone):

    python3 gpu_smoke_remote.py \
        --source-dir <candidate-source-dir> \
        --validator <path-to-validate_cpu.py> \
        --out <report.json> \
        [--filter SUBSTR] [--max-cases N] [--seed N] [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

EVIDENCE_CLASS = "nvidia-smoke-nontarget"
PUBLIC_ENTRY = "concat_and_cast_mha_k"
NOTICE = (
    "Non-target GPU smoke only. The device is not one of the eight declared Task 78 "
    "targets, so this report is diagnostic evidence for candidate compilation and "
    "correctness; it never satisfies target_validated or measured, and it does not "
    "predict any target chip's performance."
)


def log(message: str) -> None:
    # Gate visibility: every enable/skip decision is announced once.
    print(f"[gpu-smoke] {message}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_validator(path: Path):
    return load_module(path, "_flagos_validate_cpu")


def find_entry(module):
    entry = getattr(module, PUBLIC_ENTRY, None)
    if callable(entry):
        return entry, PUBLIC_ENTRY
    exported = getattr(module, "__all__", None) or []
    for name in exported:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate, name
    return None, None


def make_tensor_gpu(validator, shape, dtype, layout, generator, device):
    """Build the validator's exact CPU tensor, then mirror it onto the device.

    Copying the 1-D backing storage and re-applying ``as_strided`` preserves the
    validator's values, strides, and storage offset exactly.  ``Tensor.to`` would
    not: it may materialise broadcast or otherwise non-contiguous views.
    """
    cpu = validator.make_tensor(tuple(shape), dtype, layout, generator)
    backing_cpu = cpu.new_empty(0).set_(cpu.storage())
    backing_gpu = backing_cpu.to(device=device, copy=True)
    return backing_gpu.as_strided(tuple(shape), tuple(cpu.stride()), cpu.storage_offset())


def same_values(left, right) -> bool:
    import torch

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    try:
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    except AssertionError:
        return False
    return True


def run_case(torch, validator, entry, case, device, seed):
    """Execute one case on the device. Returns a per-case result dict."""
    tokens, heads, nope_dim, rope_dim = case.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    nope_dtype, rope_dtype, out_dtype = case.dtypes
    k = make_tensor_gpu(validator, (tokens, heads, nope_dim + rope_dim), out_dtype,
                        case.layouts[0], generator, device)
    nope = make_tensor_gpu(validator, (tokens, heads, nope_dim), nope_dtype,
                           case.layouts[1], generator, device)
    rope = make_tensor_gpu(validator, (tokens, 1, rope_dim), rope_dtype,
                           case.layouts[2], generator, device)
    k_before = k.clone()
    nope_before = nope.clone()
    rope_before = rope.clone()
    expected = torch.cat([nope, rope.expand(-1, heads, -1)], dim=-1).to(k.dtype)

    started = time.perf_counter()
    output = entry(k, nope, rope)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"entry returned {type(output).__name__}, not a Tensor")
    if output.shape != expected.shape or output.dtype != expected.dtype:
        raise AssertionError(
            f"wrong output shape/dtype: {tuple(output.shape)}/{output.dtype} "
            f"!= {tuple(expected.shape)}/{expected.dtype}"
        )
    if output.device != k.device:
        raise AssertionError(f"wrong output device: {output.device} != {k.device}")
    if expected.numel() and not same_values(output, expected):
        mismatch = (output != expected) & ~(output.isnan() & expected.isnan())
        raise AssertionError(
            f"output mismatch: {int(mismatch.sum())}/{expected.numel()} elements differ"
        )
    for name, before, after in (("k", k_before, k), ("k_nope", nope_before, nope),
                                ("k_rope", rope_before, rope)):
        if not same_values(before, after):
            raise AssertionError(f"input {name} was mutated")
    return {"elapsed_seconds": round(elapsed, 6)}


def negative_control(torch, validator, device, seed):
    """Prove the checker rejects a wrong kernel; a silent checker is a hard error.

    The control must run on a case with a non-empty output: every empty case
    compares equal to every other empty result, so an empty case would let a
    broken implementation pass and make the whole smoke report untrustworthy.
    """
    for case in validator.cases():
        tokens, heads, nope_dim, rope_dim = case.shape
        if not (tokens and heads and nope_dim and rope_dim):
            continue
        if case.layouts[0] != "contiguous":
            continue
        generator = torch.Generator(device="cpu").manual_seed(seed)
        nope_dtype, rope_dtype, out_dtype = case.dtypes
        k = make_tensor_gpu(validator, (tokens, heads, nope_dim + rope_dim), out_dtype,
                            case.layouts[0], generator, device)
        nope = make_tensor_gpu(validator, (tokens, heads, nope_dim), nope_dtype,
                               case.layouts[1], generator, device)
        rope = make_tensor_gpu(validator, (tokens, 1, rope_dim), rope_dtype,
                               case.layouts[2], generator, device)
        expected = torch.cat([nope, rope.expand(-1, heads, -1)], dim=-1).to(k.dtype)
        sentinel = 1234.5

        def broken(k_arg, _nope, _rope):
            return torch.full_like(k_arg, sentinel)

        try:
            torch.testing.assert_close(torch.full_like(expected, sentinel), expected,
                                       rtol=0, atol=0, equal_nan=True)
            continue  # coincidental equality; try the next case
        except AssertionError:
            pass
        try:
            run_case(torch, validator, broken, case, device, seed)
        except Exception as exc:  # noqa: BLE001 - any rejection proves the checker works
            return {"passed": True, "case": case.name, "detected": type(exc).__name__}
        return {
            "passed": False,
            "case": case.name,
            "error": "checker accepted a deliberately wrong kernel; smoke results are untrustworthy",
        }
    return {"passed": False, "error": "no usable non-empty case for the negative control"}


def device_facts(torch):
    facts = {"device_name": None, "compute_capability": None, "driver": None,
             "torch": getattr(torch, "__version__", None), "cuda": None, "triton": None}
    try:
        import triton
        facts["triton"] = triton.__version__
    except Exception:  # noqa: BLE001 - reported as null, never fabricated
        pass
    facts["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    facts["device_name"] = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    facts["compute_capability"] = f"sm_{capability[0]}{capability[1]}"
    try:
        with open("/proc/driver/nvidia/version", encoding="utf-8") as handle:
            facts["driver"] = handle.readline().split("Kernel Module for x86_64")[-1].strip() or None
    except OSError:
        pass
    return facts


def summarize_module(torch, module):
    kernels = []
    try:
        import triton
        for name, value in vars(module).items():
            if isinstance(value, triton.runtime.jit.JITFunction):
                kernels.append(name)
    except Exception:  # noqa: BLE001 - informational only
        return None
    return sorted(kernels)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True,
                        help="path to validate_cpu.py, the case-matrix source of truth")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filter", default=None, help="only run cases whose name contains this")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-negative-control", action="store_true")
    args = parser.parse_args(argv)

    report = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": str(args.source_dir),
        "source_hashes": {},
        "device": {},
        "suite": {},
        "results": {},
        "negative_control": None,
        "passed": False,
        "notice": NOTICE,
    }

    def finish(status: int, infra_error: str | None = None) -> int:
        if infra_error:
            report["infrastructure_error"] = infra_error
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return status

    log(f"ENABLED: non-target GPU smoke on device={args.device}")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return finish(2, f"torch import failed: {exc}")
    if not torch.cuda.is_available():
        return finish(
            2,
            "torch.cuda.is_available() is False; export LD_LIBRARY_PATH for the driver "
            "libraries before running (the container hides them by default)",
        )
    try:
        torch.cuda.init()
        report["device"] = device_facts(torch)
    except Exception as exc:  # noqa: BLE001
        return finish(2, f"cannot initialise {args.device}: {exc}")

    try:
        validator = load_validator(args.validator)
    except Exception as exc:  # noqa: BLE001
        return finish(2, f"cannot load validator {args.validator}: {exc}")

    suite = validator.cases()
    if args.filter:
        suite = [case for case in suite if args.filter in case.name]
    if args.max_cases is not None:
        suite = suite[: args.max_cases]
    report["suite"] = {"case_count": len(suite), "filter": args.filter,
                       "max_cases": args.max_cases, "seed": args.seed}
    if not suite:
        return finish(2, "case selection is empty; refusing to report a pass")

    failures = []
    for path in sorted(args.source_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        record = {"passed": False, "cases_run": 0, "cases_passed": 0, "failures": [],
                  "kernels": None, "entry": None, "sha256": None}
        try:
            record["sha256"] = sha256_file(path)
            report["source_hashes"][path.name] = record["sha256"]
            module = load_module(path, f"_flagos_candidate_{path.stem}")
        except Exception as exc:  # noqa: BLE001
            record["failures"].append({"case": "<import>", "error": f"{type(exc).__name__}: {exc}",
                                       "traceback": traceback.format_exc()[-2000:]})
            report["results"][path.name] = record
            failures.append(path.name)
            continue
        entry, entry_name = find_entry(module)
        record["entry"] = entry_name
        record["kernels"] = summarize_module(torch, module)
        if entry is None:
            record["failures"].append({"case": "<entry>", "error": f"no {PUBLIC_ENTRY} entry point"})
            report["results"][path.name] = record
            failures.append(path.name)
            continue
        for case in suite:
            record["cases_run"] += 1
            try:
                detail = run_case(torch, validator, entry, case, args.device, args.seed)
                record["cases_passed"] += 1
                del detail
            except Exception as exc:  # noqa: BLE001
                record["failures"].append({
                    "case": case.name,
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                    "traceback": traceback.format_exc()[-2000:],
                })
                log(f"FAIL {path.name} :: {case.name} :: {type(exc).__name__}: {str(exc)[:200]}")
                break  # one real failure per file is enough; do not burn the GPU on the rest
        record["passed"] = record["cases_run"] > 0 and not record["failures"]
        report["results"][path.name] = record
        if not record["passed"]:
            failures.append(path.name)

    if args.skip_negative_control:
        log("SKIPPED: negative control (--skip-negative-control)")
        report["negative_control"] = {"passed": None, "skipped": True}
    else:
        try:
            report["negative_control"] = negative_control(torch, validator, args.device, args.seed)
        except Exception as exc:  # noqa: BLE001
            report["negative_control"] = {"passed": False,
                                          "error": f"{type(exc).__name__}: {exc}"}
        if report["negative_control"].get("passed") is not True:
            log("FAIL: negative control did not reject a deliberately wrong kernel")
            failures.append("<negative-control>")

    report["passed"] = not failures and bool(report["results"])
    log(f"{'PASS' if report['passed'] else 'FAIL'}: {len(failures)} file(s)/control(s) failed. {NOTICE}")
    return finish(0 if report["passed"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
