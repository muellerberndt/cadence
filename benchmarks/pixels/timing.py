"""Seconds per frame of ``act`` and ``lesson`` at 84 by 84 (cadence#284): the ``fields`` arm's
brain on one backend, at 1 and 16 streams, and whether one stream acts within a VizDoom tic at
35 per second (28.6 ms).

Each reading builds the brain, warms it with ``WARM`` moments, then times ``MOMENTS`` greedy acts
in the streams and ``MOMENTS`` lessons of the same frames, wall-clock, and reports the median
and the mean per moment and per frame. The fields learn at every timed moment, as in a life.
The fields stage runs on the host whatever the backend.

    PYTHONPATH=src python benchmarks/pixels/timing.py --backend cpu --out results/timing.jsonl
    PYTHONPATH=src python benchmarks/pixels/timing.py --backend mlx --out results/timing.jsonl
    PYTHONPATH=src python benchmarks/pixels/timing.py --backend torch --device cuda --out ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
TIC = 1.0 / 35.0  # a VizDoom tic
WARM, MOMENTS = 4, 32


def _load(name: str) -> Any:
    key = f"pixels_{name}"  # a name of this chamber's own: other benchmarks have a chamber.py
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


ch = _load("chamber")
px = _load("pixels")


def reading(backend: str, device: str | None, streams: int, shape: str, arm: str) -> dict[str, Any]:
    from cadence import Brain

    frame_shape = tuple(ch.SHAPES[shape])
    settings: dict[str, Any] = {"recall": False, **px.ARMS[arm]["compose"], "backend": backend}
    if device is not None:
        settings["device"] = device
    brain = Brain.compose(px.inputs_of(arm, frame_shape), ch.ACTIONS, seed=0, **settings)
    batches = ch.Schedule(0, frame_shape, rows=streams).batches(WARM + MOMENTS)
    frames = [(px.seen(arm, f), labels) for f, labels in batches]
    times: dict[str, list[float]] = {"act": [], "lesson": []}
    for index, (x, labels) in enumerate(frames):
        for kind in ("lesson", "act"):
            start = time.perf_counter()
            if kind == "act":
                brain.act(x, greedy=True)
            else:
                brain.lesson(x, labels)
            if index >= WARM:
                times[kind].append(time.perf_counter() - start)
    out: dict[str, Any] = {
        "backend": backend, "device": device, "streams": streams, "shape": shape, "arm": arm,
        "placement": {k: v for k, v in brain.placement.items() if k in ("backend", "device",
                                                                         "precision")},
        "machine": platform.platform(), "processor": platform.processor(),
        "parameters": brain.parameters(),
    }
    for kind, values in times.items():
        median = float(np.median(values))
        out[kind] = {"median_per_moment": median, "mean_per_moment": float(np.mean(values)),
                     "median_per_frame": median / streams}
    out["one_stream_within_tic"] = bool(streams == 1 and out["act"]["median_per_moment"] < TIC)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--device", default=None)
    parser.add_argument("--streams", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--shape", default="grey84", choices=tuple(ch.SHAPES))
    parser.add_argument("--arm", default="fields")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import warnings

    warnings.simplefilter("ignore", RuntimeWarning)
    with out.open("a") as handle:
        for streams in args.streams:
            result = reading(args.backend, args.device, streams, args.shape, args.arm)
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
