"""The map of things in the chamber: a scene scored with ``cadence.space.map`` beside the controls.

A scene of one thing (generated, or a ``cadence-tracking-scene/1`` file) is given to the map as
the camera gives it: a report of the hand while it is seen, and at every step the cells the
camera sees. The map's answer at a step is the mean place of the hand's activity (of the
thing a sense wrote last, where the map holds two of the kind); at a step
where the map holds the hand with no place, the last place it held is scored and the step is
counted. Printed beside it: last seen and the constant-velocity Kalman filter.

    python benchmarks/tracking/mapped.py --scenes benchmarks/tracking/scenes/promise.json
    python benchmarks/tracking/mapped.py --founder 0 --reach 0.15 0.3 0.6 --empty near keep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking as tr  # noqa: E402
import world as w  # noqa: E402

HAND = 0
ERRANDS = ("left", "visit", "pause", "hops", "slowhand")


def lit_cells(scene: w.Scene, t: int, centers: np.ndarray) -> np.ndarray:
    """Which cells the camera sees at step ``t`` of a scene of one thing: none while the
    camera is off (a step at which the thing lies in the camera's view with no report)."""
    if not scene.in_view[t, 0] and bool(
            w.visible(scene.geometry, scene.truth[t, :1], scene.joints[t])[0]):
        return np.zeros(len(centers), dtype=bool)
    return np.asarray(w.visible(scene.geometry, centers, scene.joints[t]), dtype=bool)


def follow(scene: w.Scene, settings: dict[str, Any] | None = None
           ) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """A scene of one thing through the map: answers ``(steps, 1, 2)``, per step whether the
    map held a place, whether the hand's cells held the true place and how many cells they
    were, and milliseconds a step."""
    from cadence.space.map import Map

    feed = tr.given(scene)
    held = Map(scene.geometry.plane, **(settings or {}))
    answers = np.zeros((scene.steps, 1, 2))
    placed = np.ones(scene.steps, dtype=bool)
    inside = np.ones(scene.steps, dtype=bool)
    cells = np.ones(scene.steps)
    answers[0, 0] = feed["report"][0]
    held.moment(seen=[(HAND, feed["report"][0])])
    nx, ny = held.cells
    began = time.perf_counter()
    for t in range(1, scene.steps):
        seen = [(HAND, feed["report"][t])] if feed["seen"][t] else []
        held.moment(seen=seen, lit=lit_cells(scene, t, held.centers))
        mass = held.sheet(HAND).ravel()
        if (mass > 0).any():  # of several things of the kind, the one a sense wrote last
            mass = mass * (held.clock <= held.clock[mass > 0].min() + 0.5)
        placed[t] = bool(mass.sum() > 0)
        answers[t, 0] = (mass @ held.centers) / mass.sum() if placed[t] else answers[t - 1, 0]
        at = np.floor((scene.truth[t, 0] - held._low) / held.size).astype(np.intp)
        own = int(np.clip(at[0], 0, nx - 1)) * ny + int(np.clip(at[1], 0, ny - 1))
        inside[t], cells[t] = bool(mass[own] > 0), float((mass > 0).sum())
    ms = (time.perf_counter() - began) * 1000.0 / max(scene.steps - 1, 1)
    return answers, {"placed": placed, "inside": inside, "cells": cells}, ms


def score(scenes: list[w.Scene], answers: list[np.ndarray],
          more: list[dict[str, np.ndarray]] | None = None) -> dict[str, Any]:
    """The chamber's table over ``scenes`` and, for the map, the shares of the steps out of
    view at which it held a place and at which the hand's cells held the true place, with the
    cells' mean number there."""
    out = tr.table(scenes, answers)
    if more is not None:
        hidden = [~s.in_view[:, 0] & (np.arange(s.steps) > 0) for s in scenes]
        for name in ("placed", "inside", "cells"):
            values = np.concatenate([m[name][h] for m, h in zip(more, hidden, strict=True)])
            out[name] = float(values.mean()) if len(values) else None
    return out


def line(name: str, table: dict[str, Any]) -> str:
    def cm(value: float | None) -> str:
        return "  -  " if value is None else f"{100 * value:5.1f}"

    text = f"{name:24} in view {cm(table['in_view'])}  out of view {cm(table['out_of_view'])}  |"
    text += " ".join(f" {k} {cm(v['error'])}" for k, v in table["by_unseen"].items())
    if table.get("panels"):
        text += "  |" + " ".join(f" {k} {cm(v['error'])}" for k, v in table["panels"].items())
    if table.get("placed") is not None:
        text += (f"  | holds a place {table['placed']:.2f}, cells hold the hand "
                 f"{table['inside']:.2f} at {table['cells']:.0f} cells")
    return text


def read(scenes: list[w.Scene], arms: dict[str, dict[str, Any]], kalman: dict[str, float] | None
         ) -> dict[str, Any]:
    """Every arm's table on ``scenes``: last seen, the Kalman filter where fitted, and the map
    at each of ``arms`` (a name and the map's settings)."""
    out: dict[str, Any] = {"last seen": score(scenes, [tr.follow(s, tr.Still)[0] for s in scenes])}
    if kalman is not None:
        out["Kalman"] = score(scenes, [tr.follow(s, tr.ConstantVelocity, kalman)[0]
                                       for s in scenes])
    for name, settings in arms.items():
        got = [follow(s, settings) for s in scenes]
        out[name] = score(scenes, [g[0] for g in got], [g[1] for g in got])
        out[name]["milliseconds"] = float(np.mean([g[2] for g in got]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes", type=Path, nargs="*", default=[])
    parser.add_argument("--founder", type=int)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--quota", type=int, default=10)
    parser.add_argument("--each", type=int, default=8, help="scenes of each errand kind")
    parser.add_argument("--reach", type=float, nargs="+", default=[0.15])
    parser.add_argument("--empty", nargs="+", default=["near"])
    parser.add_argument("--cells", type=int, nargs=2, default=[32, 26])
    parser.add_argument("--fit", type=int, default=100, help="scenes the Kalman filter is "
                        "fitted on (with --founder)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    arms = {f"map, {empty}" + (f", reach {reach:g} m" if empty == "near" else ""):
            {"cells": tuple(args.cells), "empty": empty, "reach": reach}
            for empty in args.empty for reach in (args.reach if empty == "near" else [0.15])}
    panels: dict[str, list[w.Scene]] = {path.stem: [w.load(path)] for path in args.scenes}
    kalman = None
    if args.founder is not None:
        kalman = tr.fit_kalman(w.scenes("other", tr.seed(args.founder, "train"), args.fit))
        panels["a person's hand"] = tr.test_scenes("other", args.founder, args.test, args.quota)
        base = tr.seed(args.founder, "test") + 7000
        panels["errands"] = [s for index, kind in enumerate(ERRANDS)
                             for s in w.scenes("other", base + index, args.each, kind=kind)]
    rows: dict[str, Any] = {}
    for name, scenes in panels.items():
        rows[name] = read(scenes, arms, kalman)
        print(f"== {name} ({len(scenes)} scenes)", flush=True)
        for arm, table in rows[name].items():
            print(line(arm, table), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"founder": args.founder, "arms": {
            k: {**v, "cells": list(v["cells"])} for k, v in arms.items()}, "readings": rows},
            indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
