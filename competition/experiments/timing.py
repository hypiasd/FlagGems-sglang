"""Budget-aware sampling; injectable clock/timer for meaningful offline checks."""
from __future__ import annotations

import math
import statistics
import time

MODES = {
    "quick": {"budget": 30.0, "groups": 3, "group_ms": 10.0, "cap": 256, "timeout": 120},
    "confirm": {"budget": 120.0, "groups": 5, "group_ms": 100.0, "cap": 4096, "timeout": 210},
}


class Budget:
    def __init__(self, seconds, clock=time.perf_counter):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("budget must be positive and finite")
        self.clock = clock
        self.started = clock()
        self.seconds = seconds

    def remaining(self):
        return max(0.0, self.seconds - (self.clock() - self.started))

    def fits(self, estimated_seconds=0):
        return self.remaining() > max(0.001, estimated_seconds)

    def elapsed(self):
        return self.clock() - self.started


def repeats(estimate_ms, target_ms, cap, remaining_seconds):
    # Leave room for the next group and report flush; include enqueue wall time.
    affordable = int(remaining_seconds * 1000 * 0.8 / max(estimate_ms, 0.001))
    return max(0, min(cap, max(1, int(target_ms / max(estimate_ms, 0.001))), affordable))


def sample_pair(candidate, reference, timer, budget, mode="quick"):
    settings = MODES[mode]
    functions = {"candidate": candidate, "reference": reference}
    estimates, results, counts = {}, {key: [] for key in functions}, {key: [] for key in functions}
    for key, fn in functions.items():
        if not budget.fits():
            break
        wall = budget.clock()
        estimates[key] = timer(fn, 1)
        estimates[key] = max(estimates[key], (budget.clock() - wall) * 1000)
    for key, fn in functions.items():
        if key not in estimates or not budget.fits(estimates[key] / 1000):
            continue
        n = repeats(estimates[key], 5.0, 16, budget.remaining())
        if n:
            timer(fn, n)
    for group in range(settings["groups"]):
        order = ("candidate", "reference") if group % 2 == 0 else ("reference", "candidate")
        for key in order:
            if key not in estimates:
                continue
            n = repeats(estimates[key], settings["group_ms"], settings["cap"], budget.remaining())
            if not n or not budget.fits(n * estimates[key] / 1000):
                continue
            wall = budget.clock()
            ms = timer(functions[key], n)
            results[key].append(ms)
            counts[key].append(n)
            estimates[key] = max(ms, (budget.clock() - wall) * 1000 / n)
    complete = all(len(values) == settings["groups"] for values in results.values())
    summary = {}
    for key, values in results.items():
        median = statistics.median(values) if values else None
        spread = (max(values) - min(values)) / median if median and values else None
        summary[key] = {"median_ms": median, "spread_fraction": spread, "samples_ms": values, "repeats": counts[key]}
    speedup = summary["reference"]["median_ms"] / summary["candidate"]["median_ms"] if complete and summary["candidate"]["median_ms"] else None
    return {"status": "complete" if complete else "insufficient", **summary, "speedup": speedup,
            "confirm_recommended": bool(speedup and (abs(speedup - 1) <= 0.05 or any(v["spread_fraction"] > 0.05 for v in summary.values()))),
            "budget_overrun_seconds": max(0.0, budget.elapsed() - budget.seconds)}


def cuda_timer(torch):
    def measure(fn, count):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(count):
            fn()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) / count
    return measure
