"""What a lives axis on MLX would cost: the transport of one sweep, per life, on the Apple GPU.

A sweep of ``cadence.Lives`` on torch moves every life's activation through its own synapses:
per range of postsynaptic neurons the layout reaches, one batched product of each life's row with
its own dense block (``_LivesTorchKernel._synaptic_input``). One brain's streams share one block,
one product of all rows. This times both on MLX for the chamber's brain without the store
(``Brain.compose(35, 18, episodic=False)``), its blocks as ``Lives`` lays them out, in float32
on the GPU, at ``--lives`` lives: the seconds per sweep and per life and sweep, each the median
of ``--repeats`` timings of ``--sweeps`` sweeps evaluated together::

    python benchmarks/device/lives_mlx_cost.py --lives 12 128 1024

This is a benchmark of a source checkout: ``PYTHONPATH=src`` from the repository's root, with
``mlx`` and ``torch`` installed (torch lays out the blocks).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import warnings

import mlx.core as mx
import numpy as np

import cadence as cd


def blocks() -> list[tuple[np.ndarray, int, int, int]]:
    """The chamber brain's blocks as ``Lives`` lays them out: per range reached, the presynaptic
    neurons, the range and its width."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lives = cd.Lives(35, 18, 1, backend="torch", device="cpu", episodic=False)
    kernel = lives.brain.brain._torch
    out = []
    for place, b0, b1, view in kernel._blocks():
        rows = view.shape[1]
        if isinstance(place, slice):
            index = np.arange(place.start, place.stop)
        else:
            index = place.cpu().numpy()
        assert len(index) == rows
        out.append((index, b0, b1, rows))
    return out


def timed(step, sweeps: int, repeats: int) -> float:
    """The median seconds of one sweep, ``sweeps`` sweeps evaluated together per timing."""
    mx.eval(step(mx.zeros(1)))  # compiled and placed before the clock
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = mx.zeros(1)
        for _ in range(sweeps):
            out = step(out)
        mx.eval(out)
        times.append((time.perf_counter() - start) / sweeps)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lives", nargs="+", type=int, default=[12, 128, 1024])
    parser.add_argument("--sweeps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    layout = blocks()
    n = 251
    rng = np.random.default_rng(0)
    print(json.dumps({"machine": platform.machine(), "mlx": mx.__version__,
                      "device": str(mx.default_device()),
                      "blocks": [(rows, b1 - b0) for _, b0, b1, rows in layout]}), flush=True)
    for lives in args.lives:
        s = mx.array(rng.random((lives, n), dtype=np.float32))
        own = [mx.array(rng.normal(size=(lives, rows, b1 - b0)).astype(np.float32))
               for _, b0, b1, rows in layout]
        shared = [mx.array(rng.normal(size=(rows, b1 - b0)).astype(np.float32))
                  for _, b0, b1, rows in layout]
        index = [mx.array(i.astype(np.int32)) for i, _, _, _ in layout]

        def per_life(carry, own=own, s=s, index=index):
            parts = [mx.matmul(mx.take(s, i, axis=1)[:, None, :], w)[:, 0, :]
                     for i, w in zip(index, own, strict=True)]
            return carry + sum(mx.sum(p) for p in parts)

        def one_brain(carry, shared=shared, s=s, index=index):
            parts = [mx.take(s, i, axis=1) @ w for i, w in zip(index, shared, strict=True)]
            return carry + sum(mx.sum(p) for p in parts)

        a = timed(per_life, args.sweeps, args.repeats)
        b = timed(one_brain, args.sweeps, args.repeats)
        print(json.dumps({
            "lives": lives, "lives_sweep": a, "streams_sweep": b,
            "lives_per_life_sweep": a / lives, "streams_per_life_sweep": b / lives,
            "ratio": a / b,
        }), flush=True)


if __name__ == "__main__":
    main()
