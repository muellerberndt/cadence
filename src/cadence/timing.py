"""Timing a settlement: latency per decision, its spread, and what the machine was doing.

A wall-clock number in a receipt is a fact about a program on a machine on a
day. This module records enough of the machine to read it: how many threads
the arithmetic libraries were allowed, which cores the process was pinned to
when the platform can say, the load average, and how many times the scheduler
took the core away during the measurement. ``latency`` times one warm
settlement per decision, many times, and reports the distribution rather than
a mean, because a controller in a body cares about the slow tail.
"""

from __future__ import annotations

import os
import platform
import resource
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["environment", "latency"]


def environment() -> dict[str, Any]:
    """The machine as the arithmetic saw it: threads, pinning, load, versions."""
    threads = {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS",
        )
    }
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):  # Linux: the cores this process may run on
        affinity = sorted(os.sched_getaffinity(0))
    try:
        load: tuple[float, ...] | None = os.getloadavg()
    except OSError:  # pragma: no cover
        load = None
    out: dict[str, Any] = {
        "machine": platform.machine(),
        "system": platform.system(),
        "cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "threads": threads,
        "affinity": affinity,
        "load_average": load,
    }
    for name in ("numpy", "numba", "torch"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def latency(decide: Callable[[], Any], *, repeats: int = 1000, warmup: int = 20) -> dict[str, Any]:
    """Time ``decide()`` ``repeats`` times; microseconds at the median and the tails.

    Also counts the scheduler's interventions over the measurement: voluntary
    context switches (the process gave the core up) and involuntary ones (the
    core was taken), from ``getrusage``. A tail far above the median with many
    involuntary switches is the machine, not the net.
    """
    for _ in range(warmup):
        decide()
    before = resource.getrusage(resource.RUSAGE_SELF)
    samples = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter_ns()
        decide()
        samples[i] = time.perf_counter_ns() - t0
    after = resource.getrusage(resource.RUSAGE_SELF)
    micro = samples / 1000.0
    p = np.percentile(micro, [50, 90, 99, 100])
    return {
        "repeats": repeats,
        "p50_us": float(p[0]),
        "p90_us": float(p[1]),
        "p99_us": float(p[2]),
        "max_us": float(p[3]),
        "mean_us": float(micro.mean()),
        "jitter": float((p[2] - p[0]) / p[0]) if p[0] > 0 else None,  # (p99 - p50) / p50
        "voluntary_switches": int(after.ru_nvcsw - before.ru_nvcsw),
        "involuntary_switches": int(after.ru_nivcsw - before.ru_nivcsw),
        "environment": environment(),
    }
