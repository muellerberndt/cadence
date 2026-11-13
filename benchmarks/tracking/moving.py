"""What a moment of the map costs while things move out of view: two hands on straight reaches
across the chamber's blind spot, read by a map with its taught physics (a map file that
``mapped.py --save`` wrote). Each moment is timed with its inventory (``things``), as a page
reads it. The same file runs under Pyodide: mount ``cadence`` and this folder and run it there.

    python benchmarks/tracking/mapped.py --founder 0 --physics '{}' --watch 100 --save taught.npz
    python benchmarks/tracking/moving.py --map taught.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world as w  # noqa: E402

STARTS = ((-0.9, 0.85), (0.4, 1.3))  # two reaches that cross the blind spot of the first pose


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)
    from cadence.space.map import Map

    geometry = w.Geometry()
    pose = np.array(w.POSES[geometry.joints][0])
    rng = np.random.default_rng(args.seed)
    paths = []
    for start in STARTS:
        moves, _ = w.reach(rng, geometry, np.array(start), 150, "straight", 1 / 30)
        paths.append(np.array(start) + np.cumsum(moves, axis=0))

    def lit(points: np.ndarray) -> np.ndarray:
        return np.asarray(w.visible(geometry, points, pose), dtype=bool)

    times, moving = [], []
    for _ in range(args.repeats):
        held = Map.load(args.map)
        if held.physics is None:
            raise SystemExit(f"{args.map} holds no physics")
        held.physics.learning = False
        held.reset()
        for t in range(150):
            seen = [(kind, path[t] + rng.normal(0.0, 0.01, 2)) for kind, path in enumerate(paths)
                    if bool(lit(path[t][None])[0])]
            began = time.perf_counter()
            held.moment(seen=seen, lit=lit)
            things = held.things()
            times.append((time.perf_counter() - began) * 1000.0)
            moving.append(sum(entry["state"] == "moving" for entry in things))
    ms, count = np.array(times), np.array(moving)
    print(f"{len(ms)} moments, {int((count >= 2).sum())} with two things moving out of view")
    for name, chosen in (("every moment", np.ones(len(ms), dtype=bool)),
                         ("two moving out of view", count >= 2)):
        if chosen.any():
            print(f"{name}: median {np.median(ms[chosen]):.1f} ms, 95th percentile "
                  f"{np.percentile(ms[chosen], 95):.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
