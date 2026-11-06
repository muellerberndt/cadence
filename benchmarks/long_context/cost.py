"""The read's cost against its slots (cadence#268): seconds, sweeps and memory per moment.

One process per cell. A newborn ``Fd`` brain at ``inputs`` wide one-hot tokens, ``streams`` streams
and ``capacity`` slots hears ``capacity`` moments to fill them, then ``acts`` moments are timed,
each split into the read (``Recall.stimulate``), the write (``Recall.witness``) and the rest (the
settle and the brain's own work); then ``lessons`` lesson-and-act moments, with the address and
position lesson (``Recall.learn``) apart. The association is a founder's 64 neurons whatever the
inputs. Sweeps come from ``last_settlement`` (acts) and the lesson's report. Memory is the slots'
arrays per stream, and the parameters come from ``Brain.parameters``.

    PYTHONPATH=src python benchmarks/long_context/cost.py --out rows.jsonl --workers 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

FD: dict[str, Any] = {
    "episodic": False, "teaching": "deep", "recall": True, "recall_recency": 0.0,
    "learning_surprise": True, "recall_boundary": True, "recall_anchor": True,
    "recall_slope": 0.1, "recall_position_rate": 0.01, "recall_relative": True,
}


def run_cell(spec: dict[str, Any]) -> dict[str, Any]:
    warnings.simplefilter("ignore")
    import cadence as cd
    from cadence.stream import Recall

    inputs, streams, capacity, position = (spec[k] for k in ("inputs", "streams", "capacity",
                                                              "position"))
    timers = {"read": 0.0, "write": 0.0, "learn": 0.0}
    calls = {"read": 0, "write": 0, "learn": 0}
    stimulate, witness, learn = Recall.stimulate, Recall.witness, Recall.learn

    def timed(name: str, function: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t = time.perf_counter()
            out = function(*args, **kwargs)
            timers[name] += time.perf_counter() - t
            calls[name] += 1
            return out
        return wrapped

    Recall.stimulate = timed("read", stimulate)
    Recall.witness = timed("write", witness)
    Recall.learn = timed("learn", learn)
    actions = min(8, inputs)
    options = {**FD, **(spec.get("options") or {})}
    if spec.get("trace") is False:
        options["working_memory_amplitude"] = 0.0
    brain = cd.Brain.compose(inputs, actions, modules=(64,), seed=int(spec.get("founder", 0)),
                             recall_capacity=capacity, recall_position=position, **options)
    rng = np.random.default_rng(np.random.SeedSequence([int(spec.get("founder", 0)), 268]))

    def moment() -> np.ndarray:
        x = np.zeros((streams, inputs))
        x[np.arange(streams), rng.integers(actions, size=streams)] = 1.0
        return x

    brain.reset()
    started = time.monotonic()
    fill = 0
    while brain.recall.slots < capacity:
        brain.act(moment(), greedy=True)
        fill += 1
    fill_seconds = time.monotonic() - started
    for key in timers:
        timers[key], calls[key] = 0.0, 0
    # acts at full capacity
    acts, sweeps = int(spec.get("acts", 200)), 0.0
    t0 = time.perf_counter()
    for _ in range(acts):
        brain.act(moment(), greedy=True)
        sweeps += float((brain.last_settlement or {}).get("steps", 0))
    act_seconds = time.perf_counter() - t0
    act_read, act_write = timers["read"], timers["write"]
    for key in timers:
        timers[key], calls[key] = 0.0, 0
    # a lesson before each act at full capacity, answers drawn at random: its settles and its steps
    lessons, row_sweeps = int(spec.get("lessons", 50)), 0.0
    t0 = time.perf_counter()
    for _ in range(lessons):
        x = moment()
        report = brain.lesson(x, rng.integers(actions, size=streams))
        row_sweeps += float(report.get("total_row_sweeps", 0.0))
        brain.act(x, greedy=True)
    lesson_seconds = time.perf_counter() - t0
    recall = brain.recall
    slot_bytes = sum(getattr(recall, n).nbytes for n in ("keys", "values", "nexts", "held"))
    return {
        **spec,
        "slots": recall.slots, "buckets": recall.position_buckets,
        "parameters": int(brain.parameters()),
        "fill": fill, "fill_seconds": round(fill_seconds, 3),
        "act_ms": round(1e3 * act_seconds / acts, 4),
        "act_read_ms": round(1e3 * act_read / acts, 4),
        "act_write_ms": round(1e3 * act_write / acts, 4),
        "act_settle_ms": round(1e3 * (act_seconds - act_read - act_write) / acts, 4),
        "act_sweeps": round(sweeps / acts, 2),
        "lesson_moment_ms": round(1e3 * lesson_seconds / lessons, 4),
        "lesson_learn_ms": round(1e3 * timers["learn"] / lessons, 4),
        "lesson_read_ms": round(1e3 * timers["read"] / lessons, 4),
        "lesson_write_ms": round(1e3 * timers["write"] / lessons, 4),
        "lesson_row_sweeps": round(row_sweeps / lessons, 2),
        "slot_bytes_per_stream": slot_bytes / streams,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    specs = []
    for inputs, streams, capacity in product((12, 64), (1, 16), (64, 128, 512, 1280, 4096)):
        for position in sorted({32, capacity}):
            specs.append({"inputs": inputs, "streams": streams, "capacity": capacity,
                          "position": position})
        specs.append({"inputs": inputs, "streams": streams, "capacity": capacity, "position": 32,
                      "options": {"recall_grid": (7, 8, 9, 11)}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink, get_context("spawn").Pool(args.workers,
                                                               maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(run_cell, specs, chunksize=1):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
