"""GPU-side execution of one frozen experiment; no competition submission."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .checking import check_inputs, snapshot_inputs
from .store import load_json, verify, write_json
from .timing import Budget, MODES, cuda_timer, sample_pair


def profiler_path(tool):
    found = shutil.which(tool)
    if found:
        return found
    patterns = {"ncu": ["/usr/local/cuda*/bin/ncu", "/opt/nvidia/nsight-compute/*/ncu"],
                "nsys": ["/opt/nvidia/nsight-systems/*/target-linux-x64/nsys", "/opt/nvidia/nsight-systems/*/bin/nsys"]}
    import glob
    choices = sorted({p for pattern in patterns[tool] for p in glob.glob(pattern) if os.access(p, os.X_OK)}, reverse=True)
    return choices[0] if choices else None


def facts(torch, gpu=0):
    import triton
    props = torch.cuda.get_device_properties(gpu)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    identity = str(getattr(props, "uuid", props.name)) + f":{gpu}"
    return {"name": props.name, "gpu": gpu, "device_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
            "capability": list(torch.cuda.get_device_capability(gpu)), "memory_bytes": props.total_memory,
            "bf16_tensor_core": props.major >= 8, "torch": torch.__version__, "triton": triton.__version__,
            "cuda": torch.version.cuda, "driver": driver.stdout.splitlines()[gpu] if driver.returncode == 0 else None}


def doctor(gpu):
    result = {"command": "doctor", "status": "unavailable", "profilers": {}}
    for tool in ("ncu", "nsys"):
        executable = profiler_path(tool)
        item = {"available": bool(executable), "executable": executable, "capture_permission": "unverified"}
        if executable:
            version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=15)
            item["version"] = version.stdout.strip()[:300]
        result["profilers"][tool] = item
    try:
        import torch
        torch.cuda.set_device(gpu)
        result["device"] = facts(torch, gpu)
        params = Path("/proc/driver/nvidia/params")
        process = Path("/proc/self/status")
        if params.exists() and process.exists():
            admin_only = "RmProfilingAdminOnly: 1" in params.read_text()
            effective = int(next(line.split()[1] for line in process.read_text().splitlines() if line.startswith("CapEff:")), 16)
            result["profilers"]["ncu"]["hardware_counter_permission"] = "restricted" if admin_only and not effective & (1 << 21) else "requires_capture_check"
        result["status"] = "ready"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return result


def entry(path, operator):
    name = "_candidate_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, operator)
    if not callable(fn):
        raise ValueError("public entry is not callable")
    return fn


def select(adapter, names, quick):
    suite = adapter.cases()
    ids = names or (adapter.quick_ids() if quick else [c["id"] for c in suite])
    by_id = {c["id"]: c for c in suite}
    if not ids or len(ids) != len(set(ids)) or set(ids) - by_id.keys():
        raise ValueError("case selection is empty, duplicated or unknown")
    return [by_id[key] for key in ids]


def wrong_like(value):
    import torch
    if isinstance(value, (tuple, list)):
        return type(value)(wrong_like(v) for v in value)
    return torch.full_like(value, 1234)


def check_negative(adapter, want):
    try:
        adapter.check(wrong_like(want), want)
    except AssertionError:
        return True
    return False  # Empty outputs cannot establish a negative control.


def run_job(job, out):
    started = time.perf_counter()
    root = Path(job["root"])
    manifest = verify(root)
    contract = load_json(root / "contract.json")
    report = {"schema_version": 1, "run_id": manifest["run_id"], "snapshot_sha256": manifest["snapshot_sha256"],
              "command": job["command"], "evidence_class": "gpu-experiment-nontarget", "status": "running",
              "rows": [], "negative_control": False, "coverage": contract["case_coverage"], "timing": {},
              "mode": job.get("mode"), "seed": manifest["seed"]}
    protocol = {path: value for path, value in manifest["hashes"].items() if path == "contract.json" or path.endswith(("timing.py", "worker.py", "checking.py", "adapter.py", "validate_cpu.py"))}
    report["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(out, report)
    import torch
    torch.cuda.set_device(job.get("gpu", 0))
    device = f"cuda:{job.get('gpu', 0)}"
    report["device"] = facts(torch, job.get("gpu", 0))
    if manifest["requirements"].get("bf16_tensor_core") and not report["device"]["bf16_tensor_core"]:
        report.update(status="unsupported", reason="implementation requires native BF16 Tensor Cores; dtype is not changed")
        write_json(out, report)
        return report
    adapter = importlib.import_module(f"competition.{manifest['task_id']}.adapter")
    suite = select(adapter, job.get("cases"), job["command"] != "test")
    report["selected_cases"] = [c["id"] for c in suite]
    filenames = sorted(p.name for p in (root / "source").glob("*.py")) if job.get("all_sources") else [contract["operator"] + ".py"]
    functions = {name: entry(root / "source" / name, contract["operator"]) for name in filenames}
    report["timing"]["initialization_seconds"] = time.perf_counter() - started
    mode = job.get("mode", "quick")
    budget = None
    prepared = []
    compile_started = time.perf_counter()
    if job["command"] == "bench":
        cache = Path(os.environ.get("TRITON_CACHE_DIR", "/nonexistent"))
        report["compile_cache"] = {"files_before": sum(p.is_file() for p in cache.rglob("*"))}
        # Compile each selected specialization before the bounded bench phase.
        for case in suite:
            values = adapter.inputs(case, device, manifest["seed"])
            original_inputs = snapshot_inputs(values)
            outputs = {name: fn(*values) for name, fn in functions.items()}
            torch.cuda.synchronize()
            check_inputs(original_inputs)
            prepared.append((case, values, outputs))
        report["timing"]["prepare_and_compile_seconds"] = time.perf_counter() - compile_started
        report["compile_cache"]["files_after"] = sum(p.is_file() for p in cache.rglob("*"))
        budget = Budget(job.get("budget_seconds", MODES[mode]["budget"]))
        report["budget_seconds"] = budget.seconds
    for index, case in enumerate(suite):
        if budget and not budget.fits():
            break
        values = prepared[index][1] if prepared else adapter.inputs(case, device, manifest["seed"])
        before = snapshot_inputs(values)
        want = adapter.reference(*values)
        torch.cuda.synchronize()
        report["negative_control"] |= check_negative(adapter, want)
        for name, fn in functions.items():
            if budget and not budget.fits():
                break
            row = {"case": case["id"], "source": name, "metadata": adapter.metadata(case), "correct": False}
            try:
                got = fn(*values)
                torch.cuda.synchronize()
                adapter.check(got, want)
                check_inputs(before)
                row["correct"] = True
                if budget:
                    row["measurement"] = sample_pair(lambda: fn(*values), lambda: adapter.reference(*values), cuda_timer(torch), budget, mode)
                    check_inputs(before)
                elif job["command"] == "profile-target":
                    for _ in range(3):
                        fn(*values)
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                    with torch.cuda.nvtx.range("flagos-operator"):
                        fn(*values)
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            report["rows"].append(row)
            if budget:
                report["timing"]["bench_wall_seconds"] = budget.elapsed()
                report["timing"]["budget_overrun_seconds"] = max(0.0, budget.elapsed() - budget.seconds)
            write_json(out, report)
        del values, before, want
    expected = len(suite) * len(functions)
    complete = len(report["rows"]) == expected and all(r["correct"] and not r.get("error") and (not budget or r.get("measurement", {}).get("status") == "complete") for r in report["rows"])
    report["status"] = "passed" if complete and report["negative_control"] else ("failed" if any(not r["correct"] or r.get("error") for r in report["rows"]) else "incomplete")
    completed = {(r["case"], r["source"]) for r in report["rows"] if r["correct"] and not r.get("error") and (not budget or r.get("measurement", {}).get("status") == "complete")}
    report["missing_executions"] = [{"case": c["id"], "source": name} for c in suite for name in functions if (c["id"], name) not in completed]
    report["missing_cases"] = [c["id"] for c in suite if any((c["id"], name) not in completed for name in functions)]
    report["full_matrix_passed"] = job["command"] == "test" and not job.get("cases") and complete and report["negative_control"]
    report["timing"]["worker_wall_seconds"] = time.perf_counter() - started
    if budget:
        report["timing"]["bench_wall_seconds"] = budget.elapsed()
        report["timing"]["budget_overrun_seconds"] = max(0.0, budget.elapsed() - budget.seconds)
    write_json(out, report)
    return report


def profile_job(job, out):
    tool = job["tool"]
    executable = profiler_path(tool)
    if not executable:
        return {"status": "unavailable", "command": "profile", "tool": tool, "reason": "profiler is not installed"}
    capture = Path(out).parent / "capture"
    target_out = Path(out).parent / "target.json"
    target_job = {**job, "command": "profile-target"}
    target_path = Path(out).parent / "target-job.json"
    write_json(target_path, target_job)
    target = [sys.executable, "-m", "competition.experiments.worker", "--job", str(target_path), "--out", str(target_out)]
    ncu_metrics = ["--metrics", "launch__registers_per_thread,launch__shared_mem_per_block,launch__grid_size,launch__block_size,launch__waves_per_multiprocessor"] if job.get("ncu_mode") == "launch" else ["--set", "basic"]
    options = ([*ncu_metrics, "--profile-from-start", "off", "--clock-control", "none", "--force-overwrite", "--export", str(capture)] if tool == "ncu" else
               ["profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true", "--output", str(capture)])
    try:
        proc = subprocess.run([executable, *options, *target], capture_output=True, text=True, timeout=job.get("profile_timeout", 120))
        (Path(out).parent / "profiler.log").write_text(proc.stdout + proc.stderr)
        suffix = ".ncu-rep" if tool == "ncu" else ".nsys-rep"
        artifact = capture.with_suffix(suffix)
        target_report = load_json(target_out) if target_out.exists() else {}
        success = proc.returncode == 0 and artifact.is_file() and artifact.stat().st_size > 0 and target_report.get("status") == "passed"
        stats = (["--import", str(artifact), "--csv", "--page", "raw"] if tool == "ncu" else
                 ["stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum", "--format", "csv", str(artifact)])
        if success:
            summary = subprocess.run([executable, *stats], capture_output=True, text=True, timeout=60)
            (Path(out).parent / "metrics.csv").write_text(summary.stdout)
            success = summary.returncode == 0 and bool(summary.stdout.strip())
        return {**target_report, "command": "profile", "tool": tool, "status": "passed" if success else "failed",
                "capture_permission": "verified" if success else "not_verified", "artifact": artifact.name if artifact.is_file() else None,
                "metric_scope": job.get("ncu_mode", "basic") if tool == "ncu" else "cuda-timeline",
                "hardware_counters": "not_requested" if tool == "ncu" and job.get("ncu_mode") == "launch" else "verified" if tool == "ncu" and success else "not_verified",
                "returncode": proc.returncode, "reason": None if success else "ERR_NVGPUCTRPERM" if "ERR_NVGPUCTRPERM" in proc.stdout + proc.stderr else "inspect profiler.log; no valid capture claimed"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "command": "profile", "tool": tool, "capture_permission": "not_verified"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    try:
        job = load_json(args.job) if args.job else {}
        report = doctor(args.gpu) if args.doctor else (profile_job(job, args.out) if job["command"] == "profile" else run_job(job, args.out))
    except Exception as exc:
        existing = load_json(args.out) if args.out.exists() else {}
        report = {**existing, "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    write_json(args.out, report)
    print(json.dumps({key: report[key] for key in ("status", "command", "error") if key in report}), flush=True)
    return 0 if report["status"] in {"passed", "ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
