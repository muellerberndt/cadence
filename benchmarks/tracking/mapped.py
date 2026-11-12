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


def follow(scene: w.Scene, held: Any = None, learn: bool = False, felt: Any = None
           ) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """A scene of one thing through a map (``held``, a ``Map`` or its settings; a new map
    where omitted): answers ``(steps, 1, 2)``, per step whether the map held a place and, for
    the smallest sets of cells holding .90 and .95 of the hand's activity, whether each held the
    true place and how many cells it was, and milliseconds a step. With ``learn`` a map with
    physics takes its lessons. ``felt`` names the place the arm's joint readings name while
    the hand is out of view, a function of the readings: the program's forward kinematics
    (``kinematics``) or a ``Sense`` taught on the arm (``Sense.named``)."""
    from cadence.space.map import Map

    feed = tr.given(scene)
    if not isinstance(held, Map):
        held = Map(scene.geometry.plane, **(held or {}))
    held.reset()
    answers = np.zeros((scene.steps, 1, 2))
    placed = np.ones(scene.steps, dtype=bool)
    more = {name: np.ones(scene.steps) for name in ("in90", "cells90", "in95", "cells95")}
    answers[0, 0] = feed["report"][0]
    held.moment(seen=[(HAND, feed["report"][0])], learn=learn)
    nx, ny = held.cells
    began = time.perf_counter()
    for t in range(1, scene.steps):
        seen = [(HAND, feed["report"][t])] if feed["seen"][t] else []
        named = None if felt is None or feed["seen"][t] else felt(feed["joints"][t])
        held.moment(seen=seen, felt=[] if named is None else [(HAND, named)],
                    lit=lit_cells(scene, t, held.centers), learn=learn)
        mass = held.sheet(HAND).ravel()
        if (mass > 0).any():  # of several things of the kind, the one a sense wrote last
            mass = mass * (held.clock <= held.clock[mass > 0].min() + 0.5)
        placed[t] = bool(mass.sum() > 0)
        answers[t, 0] = (mass @ held.centers) / mass.sum() if placed[t] else answers[t - 1, 0]
        at = np.floor((scene.truth[t, 0] - held._low) / held.size).astype(np.intp)
        own = int(np.clip(at[0], 0, nx - 1)) * ny + int(np.clip(at[1], 0, ny - 1))
        order = np.argsort(-mass)
        total = np.cumsum(mass[order])
        for rate in (90, 95):
            need = int(np.searchsorted(total, rate / 100 * total[-1])) + 1 if placed[t] else 0
            more[f"in{rate}"][t] = float(own in order[:need])
            more[f"cells{rate}"][t] = float(need)
    ms = (time.perf_counter() - began) * 1000.0 / max(scene.steps - 1, 1)
    return answers, {"placed": placed.astype(float), **more}, ms


def score(scenes: list[w.Scene], answers: list[np.ndarray],
          more: list[dict[str, np.ndarray]] | None = None) -> dict[str, Any]:
    """The chamber's table over ``scenes`` and, for the map, the shares of the steps out of
    view at which it held a place and at which the hand's cells held the true place, with the
    cells' mean number there."""
    out = tr.table(scenes, answers)
    if more is not None:
        hidden = [~s.in_view[:, 0] & (np.arange(s.steps) > 0) for s in scenes]
        for name in ("placed", "in90", "cells90", "in95", "cells95"):
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
        text += (f"  | area .90 holds {table['in90']:.2f} at {table['cells90']:.0f} cells, "
                 f".95 holds {table['in95']:.2f} at {table['cells95']:.0f}")
    return text


def kinematics(geometry: w.Geometry) -> Any:
    """The place the arm's joint readings name by its true forward kinematics: a function the
    program computes, as a robot knows its own arm."""
    return lambda joints: w.hand(geometry, np.asarray(joints, dtype=np.float64))


def taught(scenes: list[w.Scene], held: Any, per: int = 16, rows: int = 32) -> Any:
    """A ``Sense`` of the arm's joint readings onto ``held``'s cells, taught on ``scenes`` of
    the arm's own hand at the moments the camera sees it; gives ``Sense.named``."""
    from cadence.space.map import Sense

    geometry = scenes[0].geometry
    arm = Sense(held, ranges=[tuple(limit) for limit in geometry.limits],
                per=(per,) * geometry.joints, rate=0.5)
    readings, places = [], []
    for scene in scenes:
        seen = scene.in_view[:, 0]
        readings.append(scene.joints[seen])
        places.append(scene.report[seen, 0])
    joints, at = np.concatenate(readings), np.concatenate(places)
    order = np.random.default_rng(0).permutation(len(joints))
    for k in range(0, len(order) - rows + 1, rows):
        arm.teach(joints[order[k : k + rows]], at[order[k : k + rows]])
    return arm.named


MIX = (("straight", 0.10), ("slow", 0.10), ("stop", 0.10), ("turn", 0.10), ("left", 0.15),
       ("visit", 0.15), ("pause", 0.10), ("hops", 0.10), ("slowhand", 0.10))


def nursery(founder: int, count: int) -> list[w.Scene]:
    """A founder's nursery of ``count`` scenes of a person's hand in the shares of ``MIX``
    (declared in PROTOCOL.md), shuffled: half of each kind in the chamber's geometry and half
    among discs of the scene's own."""
    out: list[w.Scene] = []
    base = tr.seed(founder, "train") + 5000
    for index, (kind, share) in enumerate(MIX):
        n = round(share * count)
        out += w.scenes("other", base + 10 * index, n // 2, kind=kind)
        out += w.scenes("other", base + 10 * index + 1, n - n // 2, kind=kind, vary=True)
    order = np.random.default_rng(base).permutation(len(out))
    return [out[i] for i in order]


def read(scenes: list[w.Scene], arms: dict[str, Any], kalman: dict[str, float] | None,
         controls: bool = True) -> dict[str, Any]:
    """Every arm's table on ``scenes``: last seen and the Kalman filter where fitted (with
    ``controls``), and the map for each of ``arms`` (a name and the map's settings, or a map)."""
    out: dict[str, Any] = {}
    if controls:
        out["last seen"] = score(scenes, [tr.follow(s, tr.Still)[0] for s in scenes])
        if kalman is not None:
            out["Kalman"] = score(scenes, [tr.follow(s, tr.ConstantVelocity, kalman)[0]
                                           for s in scenes])
    for name, held in arms.items():
        got = [follow(s, held) for s in scenes]
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
    parser.add_argument("--physics", type=json.loads, help="learned physics: its settings, "
                        "trained on --train nursery scenes of --founder")
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--watched", type=int, nargs="+", default=[],
                        help="read the physics after these numbers of scenes watched")
    parser.add_argument("--save", type=Path, help="write the trained map here")
    parser.add_argument("--own", type=int, default=0, help="read the arm's own hand: a sense "
                        "of its joint readings taught on this many of --founder's scenes")
    parser.add_argument("--per", type=int, nargs="+", default=[16],
                        help="tuned readings a joint of the taught sense")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    arms: dict[str, Any] = {
        f"map, {empty}" + (f", reach {reach:g} m" if empty == "near" else ""):
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
    if args.own and args.founder is not None:  # the arm's own hand, its joint readings given
        from cadence.space.map import Map

        scenes = tr.test_scenes("own", args.founder, args.test)
        geometry = scenes[0].geometry
        training = w.scenes("own", tr.seed(args.founder, "train"), args.own)
        held = Map(geometry.plane, cells=tuple(args.cells))
        rows["own"] = read(scenes, {"map, near, reach 0.15 m": arms[next(iter(arms))]}, None)
        for name, felt in [("map, felt by the program's forward kinematics",
                            kinematics(geometry))] + [
                (f"map, felt by a sense of {per} readings a joint", taught(training, held, per))
                for per in args.per]:
            got = [follow(s, held, felt=felt) for s in scenes]
            rows["own"][name] = score(scenes, [g[0] for g in got], [g[1] for g in got])
        rows["own"]["true kinematics on the joint readings"] = score(
            scenes, [w.hand(s.geometry, s.joints)[:, None, :] for s in scenes])
        print(f"== the arm's own hand ({len(scenes)} scenes)", flush=True)
        for arm, table in rows["own"].items():
            print(line(arm, table), flush=True)
    curve: list[dict[str, Any]] = []
    if args.physics is not None and args.founder is not None:  # learned physics, as it watches
        from cadence.space.map import Map

        training = nursery(args.founder, args.train)
        held = Map(training[0].geometry.plane, cells=tuple(args.cells), physics=args.physics)
        began = time.monotonic()
        watched = 0
        for mark in sorted(set([*args.watched, args.train])):
            while watched < min(mark, len(training)):
                follow(training[watched], held, learn=True)
                watched += 1
            held.teach()
            name = f"map with physics, {watched} scenes watched"
            point = {"watched": watched, "lessons": int(held.physics.lessons),
                     "seconds": time.monotonic() - began, "readings": {}}
            for panel, scenes in panels.items():
                point["readings"][panel] = read(scenes, {name: held}, kalman, controls=False)
                print(line(f"{panel}: {watched} watched", point["readings"][panel][name]),
                      flush=True)
            curve.append(point)
        if args.save:
            held.save(args.save)
    if args.out:
        args.out.write_text(json.dumps({"founder": args.founder, "arms": {
            k: {**v, "cells": list(v["cells"])} for k, v in arms.items()}, "readings": rows,
            "physics": args.physics, "curve": curve}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
