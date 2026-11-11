"""Seconds per lesson of a deep apical visual brain and a flat founder brain, on a device.

A lesson is ``Brain.lesson``: a free settle to the learner's tolerance, two nudged phases of the
brain's teaching recipe, their contrast and one parameter update. Two brains:

* ``deep``: a ventral-stream anatomy over a 32 x 32 RGB retina. Four cortical stages of unshared
  receptive fields (V1: 8 maps of 5 x 5 fields at stride 2; V2: 16 maps of 3 x 3; V4: 32 maps of
  3 x 3 at stride 2; IT: 128 neurons reading all of V4), an association region of 64 and five
  word slots of four motor neurons. Every pair of neighbouring stages after the retina is joined
  reciprocally, the top-down half on the apical compartment (``apical=0.25``), at the teaching
  point ``"deep"`` (normalized, nudged phases of a whole 32 sweeps): 7,956 neurons and 903,520
  synapses.
* ``flat``: ``Brain.compose(3072, 20, slots=(4,) * 5)`` with every founder setting.

Each lesson draws ``batch`` rows from a fixed pool of random observations and labels::

    python benchmarks/torch_gpu/lessons.py --device cuda:0 --batches 64 256 1024 4096
    OMP_NUM_THREADS=1 python benchmarks/torch_gpu/lessons.py --backend cpu --batches 64
    python benchmarks/torch_gpu/lessons.py --device cuda:0 --precision float32 --brains deep
    python benchmarks/torch_gpu/lessons.py --device cuda:0 --split --batches 64

``--split`` synchronizes the device around each settle, residual check, apical-gain read and
update and reports the time of each part of a lesson (the parts then no longer overlap, so
the split's total exceeds the plain timing). The script uses the public API only, so it times
any library version.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

import cadence as cd

IMAGE, CHANNELS, SLOTS = (32, 32), 3, (4, 4, 4, 4, 4)
# (name, kind, maps, field, stride): the founder ventral stages
STAGES = (("v1", "local", 8, 5, 2), ("v2", "local", 16, 3, 1), ("v4", "local", 32, 3, 2),
          ("it", "dense", 128, 0, 0))
ASSOCIATION = 64
DEEP = {
    "apical": 0.25,
    "teaching": "deep",
    "learning_surprise": False,
    "episodic": False,
    "working_memory_amplitude": 0.0,
}


def ventral(seed: int = 0) -> cd.Connectome:
    """The deep anatomy as one connectome, bands of rows cutting the block layout."""
    from cadence.blocks import layout as block_layout

    h, w = IMAGE
    regions = [("sensory", h * w * CHANNELS)]
    grids = [("sensory", h, w, CHANNELS)]
    rows, cols, maps = h, w, CHANNELS
    shapes = []
    for name, kind, size, field, stride in STAGES:
        if kind == "local":
            rows, cols = (rows - field) // stride + 1, (cols - field) // stride + 1
            shapes.append((name, kind, size, field, stride, rows, cols))
            regions.append((name, rows * cols * size))
            grids.append((name, rows, cols, size))
        else:
            shapes.append((name, kind, size, 0, 0, 1, 1))
            regions.append((name, size))
        maps = size
    del maps
    regions += [("association", ASSOCIATION), ("motor", sum(SLOTS))]
    start, n = {}, 0
    for name, size in regions:
        start[name] = n
        n += size
    rng = np.random.default_rng(seed)
    pre_all, post_all, sign_all = [], [], []
    below = (start["sensory"], h, w, CHANNELS)
    below_size = h * w * CHANNELS
    names = [s[0] for s in shapes] + ["association", "motor"]
    sizes = [size for _, size in regions[1:]]
    for index, name in enumerate(names):
        shape = shapes[index] if index < len(shapes) else None
        if shape is not None and shape[1] == "local":
            _, _, size, k, s, srows, scols = shape
            b0, b_rows, b_cols, b_maps = below
            y, x, f, dy, dx, c = np.meshgrid(
                np.arange(srows), np.arange(scols), np.arange(size), np.arange(k),
                np.arange(k), np.arange(b_maps), indexing="ij")
            pre = (b0 + ((y * s + dy) * b_cols + (x * s + dx)) * b_maps + c).ravel()
            post = (start[name] + (y * scols + x) * size + f).ravel()
            limit = np.sqrt(6.0 / (b_maps * k * k + size * k * k))
        else:
            a = np.arange(below[0], below[0] + below_size)
            b = np.arange(start[name], start[name] + sizes[index])
            pre, post = (v.ravel() for v in np.meshgrid(a, b, indexing="ij"))
            limit = np.sqrt(6.0 / (below_size + sizes[index]))
        sign = rng.choice([-1.0, 1.0], size=pre.size) * rng.uniform(0, 1, pre.size) * limit
        pre_all.append(pre)
        post_all.append(post)
        sign_all.append(sign)
        if index > 0:  # the retina drives V1 one way; every later pair is reciprocal
            pre_all.append(post)
            post_all.append(pre)
            sign_all.append(sign)
        if shape is not None and shape[1] == "local":
            below = (start[name], shape[5], shape[6], shape[2])
        else:
            below = (start[name], 1, 1, sizes[index])
        below_size = sizes[index]
    pre, post, sign = np.concatenate(pre_all), np.concatenate(post_all), np.concatenate(sign_all)
    populations: dict[str, Any] = {name: range(start[name], start[name] + size)
                                   for name, size in regions}

    def build(band: int) -> cd.Connectome:
        pops = dict(populations)
        for name, g_rows, g_cols, g_maps in grids if band else ():
            width = g_cols * g_maps
            for b, r0 in enumerate(range(0, g_rows, band)):
                r1 = min(g_rows, r0 + band)
                pops[f"{name}/band{b}"] = list(range(start[name] + r0 * width,
                                                     start[name] + r1 * width))
        return cd.Connectome.from_synapses(n, pre=pre, post=post, sign=sign, populations=pops)

    best = None
    for band in [0, *range(1, h + 1)]:
        candidate = build(band)
        lay = block_layout(candidate)
        if lay.ranges == 1 and n > 1:
            continue
        if best is None or lay.size < best[0]:
            best = (lay.size, candidate)
    assert best is not None
    return best[1]


def make(brain_name: str, backend: str, device: str | None, precision: str | None) -> cd.Brain:
    extra: dict[str, Any] = {"backend": backend}
    if backend == "torch":
        extra["device"] = device
    if brain_name == "deep":
        brain = cd.Brain.compose(IMAGE[0] * IMAGE[1] * CHANNELS, sum(SLOTS), connectome=ventral(),
                                 slots=SLOTS, seed=0, **DEEP, **extra)
    else:
        brain = cd.Brain.compose(IMAGE[0] * IMAGE[1] * CHANNELS, sum(SLOTS), slots=SLOTS, seed=0,
                                 **{"recall": False, **extra})
    if precision is not None and backend == "torch":
        g = brain.learner.brain
        brain.learner.brain = cd.NeuralGraph(
            g.connectome, g.neuron_model, efficacy=g.efficacy, bias=g.bias, log_gain=g.log_gain,
            backend="torch", device=device, precision=precision, apical=g.apical)
    return brain


def synchronize(device: str | None) -> None:
    if device is not None and device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def splitting(device: str | None) -> dict[str, list[float]]:
    """Time every settle, residual, apical-gain read and update with the device synchronized."""
    parts: dict[str, list[float]] = defaultdict(list)
    graph, learner = cd.NeuralGraph, cd.Learner

    def timed(owner: Any, name: str, label: Any) -> None:
        original = getattr(owner, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            synchronize(device)
            t = time.perf_counter()
            out = original(*args, **kwargs)
            synchronize(device)
            parts[label(args, kwargs)].append(time.perf_counter() - t)
            return out

        setattr(owner, name, wrapper)

    timed(graph, "settle_batch", lambda a, k: "settle_nudged" if k.get("nudge") is not None
          else "settle_free")
    timed(graph, "residual", lambda a, k: "residual")
    timed(graph, "apical_gain", lambda a, k: "apical_gain")
    timed(learner, "update", lambda a, k: "contrast_and_update")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--brains", nargs="+", default=["deep", "flat"], choices=["deep", "flat"])
    parser.add_argument("--backend", default="torch", choices=["torch", "cpu"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default=None, choices=["float64", "float32"])
    parser.add_argument("--batches", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--lessons", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = args.device if args.backend == "torch" else None
    rng = np.random.default_rng(7)
    pool = 4096
    x = rng.uniform(0.0, 1.0, (pool, IMAGE[0] * IMAGE[1] * CHANNELS))
    y = rng.integers(0, 4, (pool, len(SLOTS)))
    results = []
    parts = splitting(device) if args.split else None
    for name in args.brains:
        brain = make(name, args.backend, device, args.precision)
        for batch in args.batches:
            draw = np.random.default_rng(batch)

            def lesson(brain: cd.Brain = brain, draw: Any = draw,
                       batch: int = batch) -> dict[str, float]:
                rows = draw.integers(0, pool, batch)
                return dict(brain.lesson(x[rows], y[rows]))

            for _ in range(args.warmup):
                lesson()
            if parts is not None:
                parts.clear()
            synchronize(device)
            t = time.perf_counter()
            reports = [lesson() for _ in range(args.lessons)]
            synchronize(device)
            seconds = (time.perf_counter() - t) / args.lessons
            kernel = getattr(brain.learner.brain, "_torch", None)
            row = {
                "brain": name, "backend": args.backend, "device": device,
                "precision": args.precision, "batch": batch, "s_per_lesson": round(seconds, 4),
                "free_sweeps": float(np.mean([r.get("free_steps", np.nan) for r in reports])),
                "neurons": brain.connectome.n, "synapses": brain.connectome.synapses,
                "transport": kernel.describe() if hasattr(kernel, "describe") else None,
                "library": cd.__version__,
            }
            if parts is not None:
                row["split_s_per_lesson"] = {k: round(sum(v) / args.lessons, 4)
                                             for k, v in sorted(parts.items())}
            results.append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
