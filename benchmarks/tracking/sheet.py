"""The settled belief sheet in the chamber: a nursery, lessons, and readings beside the controls.

``cadence.space`` with ``sheet=`` holds a person's hand as activity on a sheet of cells. This
runner raises one in a nursery of scenes, reads it on a founder's held-out panels, on a panel of
a visitor's and a worker's hands and on recorded scene files, and reads the classical belief
(``belief.py``, its hazard fitted on the same nursery) and last seen on the same scenes.

The nursery's world (``nursery``; the shares are declared in ``PROTOCOL.md``): hands that go
straight, slow, stop and turn; hands left out of view; hands that go into a shadow, rest there
and come back; hands that pause, hop and move slowly. Half of the scenes lie among occluding
discs of their own, a third of those with the camera off for a span.

Lessons are of two kinds, stated with every reading: ``camera`` (a practice span's unread
reports teach the sheet) and ``truth`` (the simulator's true place teaches it at every step
with no report read).

    python benchmarks/tracking/sheet.py --founder 0 --train 800 --lessons truth \
        --scenes benchmarks/tracking/scenes/promise.json --out sheet.json
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
import belief as bf  # noqa: E402
import tracking as tr  # noqa: E402
import world as w  # noqa: E402

# the nursery's world: each kind's share of the scenes (declared in PROTOCOL.md)
MIX = (("straight", 0.10), ("slow", 0.10), ("stop", 0.10), ("turn", 0.10), ("left", 0.15),
       ("visit", 0.15), ("pause", 0.10), ("hops", 0.10), ("slowhand", 0.10))
ERRANDS = ("left", "visit", "pause", "hops", "slowhand")


def nursery(founder: int, count: int, name: str = "train") -> list[w.Scene]:
    """A founder's nursery of ``count`` scenes in the shares of ``MIX``, shuffled: half of each
    kind in the chamber's geometry and half among discs of the scene's own."""
    out: list[w.Scene] = []
    base = tr.seed(founder, name) + 5000
    for index, (kind, share) in enumerate(MIX):
        n = round(share * count)
        out += w.scenes("other", base + 10 * index, n // 2, kind=kind)
        out += w.scenes("other", base + 10 * index + 1, n - n // 2, kind=kind, vary=True)
    order = np.random.default_rng(base).permutation(len(out))
    return [out[i] for i in order]


def errands(founder: int, each: int = 12) -> list[w.Scene]:
    """A founder's held-out panel of a visitor's and a worker's hands: ``each`` scenes of every
    kind of ``ERRANDS``, in the chamber's geometry."""
    base = tr.seed(founder, "test") + 7000
    return [s for index, kind in enumerate(ERRANDS)
            for s in w.scenes("other", base + index, each, kind=kind)]


def lit_cells(scene: w.Scene, t: int, centers: np.ndarray) -> np.ndarray:
    """Which cells the camera sees at step ``t`` of a scene of one thing: none while the
    camera is off (a step at which the thing lies in the camera's view with no report)."""
    if not scene.in_view[t, 0] and bool(
            w.visible(scene.geometry, scene.truth[t, :1], scene.joints[t])[0]):
        return np.zeros(len(centers), dtype=bool)
    return np.asarray(w.visible(scene.geometry, centers, scene.joints[t]), dtype=bool)


class Sheeted:
    """The sheet arm: a ``Space`` beside a sheet, one row a scene."""

    def __init__(self, geometry: w.Geometry, founder: int, cells: tuple[int, int],
                 settings: dict[str, Any] | None = None, lessons: str = "camera"):
        from cadence.space import Interface, Space

        settings = dict(settings or {})
        sheet = {"cells": cells, **settings.pop("sheet", {})}
        small = dict(place=8, moved=5, motion=9, counted=3, unseen=4, heard=(6, 4), felt=4,
                     hidden=16, shift=12, spread=6, found=8, keep=4)
        self.interface = Interface(plane=geometry.plane, joints=geometry.joints,
                                   ear=geometry.ear, limits=geometry.limits,
                                   **settings.pop("interface", small))
        self.space = Space(self.interface, seed=founder, backend="cpu", sheet=sheet, **settings)
        self.lessons = lessons

    def run(self, scenes: list[w.Scene], teach: bool = False
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scenes of one length side by side: answers ``(scenes, steps, 2)``, whether each
        answer's area held the true place, and the area's size in cells."""
        space, steps = self.space, scenes[0].steps
        feeds = [tr.given(s) for s in scenes]
        centers = space.sheet.centers
        answers = np.zeros((len(scenes), steps, 2))
        held = np.ones((len(scenes), steps), dtype=bool)
        size = np.ones((len(scenes), steps))
        space.begin(np.stack([f["report"][0] for f in feeds]))
        answers[:, 0] = space.where()[0]
        for t in range(1, steps):
            seen = np.stack([f["seen"][t] for f in feeds])
            report = np.stack([f["report"][t] for f in feeds])
            lit = np.stack([lit_cells(s, t, centers) for s in scenes])
            truth = np.stack([s.truth[t, 0] for s in scenes])
            answers[:, t] = space.advance(
                seen, report, in_sight=lit, learn="sheet" if teach else False,
                truth=truth if teach and self.lessons == "truth" else None)
            held[:, t] = space.sheet.holds(truth)
            size[:, t] = space.area().reshape(len(scenes), -1).sum(axis=1)
            held[seen, t] = True  # in view the place is the report's
        return answers, held, size

    def train(self, scenes: list[w.Scene], rows: int, log: Any = None) -> None:
        by_length: dict[int, list[w.Scene]] = {}
        done = 0
        for scene in scenes:
            batch = by_length.setdefault(scene.steps, [])
            batch.append(scene)
            if len(batch) == rows:
                self.run(batch, teach=True)
                done += len(batch)
                batch.clear()
                if log is not None and done % (rows * 10) == 0:
                    log(done)
        for batch in by_length.values():
            if batch:
                self.run(batch, teach=True)

    def answer(self, scenes: list[w.Scene], rows: int = 16
               ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Each scene's answers ``(steps, 1, 2)``, area flags and area sizes, lessons off."""
        out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        by_length: dict[int, list[int]] = {}
        for i, scene in enumerate(scenes):
            by_length.setdefault(scene.steps, []).append(i)
        for members in by_length.values():
            for k in range(0, len(members), rows):
                part = members[k : k + rows]
                a, h, z = self.run([scenes[i] for i in part])
                for j, i in enumerate(part):
                    out[i] = (a[j][:, None, :], h[j], z[j])
        return ([out[i][0] for i in range(len(scenes))], [out[i][1] for i in range(len(scenes))],
                [out[i][2] for i in range(len(scenes))])


def score(scenes: list[w.Scene], answers: list[np.ndarray], held: list[np.ndarray] | None = None,
          size: list[np.ndarray] | None = None) -> dict[str, Any]:
    """Errors over ``scenes`` of one thing: the mean in view and out of view (metres), by panel
    where the scenes have panels, the share of steps out of view whose area held the true place
    and the area's mean size there."""
    table = tr.table(scenes, answers)
    out: dict[str, Any] = {"in_view": table["in_view"], "out_of_view": table["out_of_view"],
                           "panels": {k: v["error"] for k, v in table.get("panels", {}).items()}}
    hidden = [~s.in_view[:, 0] for s in scenes]
    for h in hidden:
        h[0] = False
    if held is not None:
        out["area_holds"] = float(np.concatenate([a[h] for a, h in zip(held, hidden, strict=True)]
                                                 ).mean())
    if size is not None:
        out["area_cells"] = float(np.concatenate([a[h] for a, h in zip(size, hidden, strict=True)]
                                                 ).mean())
    return out


def control(scenes: list[w.Scene], hazard: float, read: str = "mean", cell: float = 0.04
            ) -> dict[str, Any]:
    """The classical belief on ``scenes`` at one hazard."""
    got = [bf.follow(s, hazard, read, cell) for s in scenes]
    return score(scenes, [g[0] for g in got], [g[1] for g in got], [g[2] for g in got])


def last_seen(scenes: list[w.Scene]) -> dict[str, Any]:
    return score(scenes, [tr.follow(s, tr.Still)[0] for s in scenes])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--founder", type=int, default=0)
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--lessons", choices=("camera", "truth"), default="camera")
    parser.add_argument("--cells", type=int, nargs=2, default=[32, 26])
    parser.add_argument("--settings", type=json.loads, default={})
    parser.add_argument("--scenes", type=Path, nargs="*", default=[])
    parser.add_argument("--fit", type=int, default=60, help="nursery scenes the control's "
                        "hazard is fitted on (0: no control)")
    parser.add_argument("--quota", type=int, default=40)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    began = time.monotonic()
    training = nursery(args.founder, args.train)
    arm = Sheeted(training[0].geometry, args.founder, (args.cells[0], args.cells[1]),
                  args.settings, args.lessons)

    def log(done: int) -> None:
        shares = arm.space.sheet.shares()
        print(f"  {done} scenes, {arm.space.sheet.lessons} lessons, "
              f"{time.monotonic() - began:.0f} s: keep {np.round(shares['keep'], 3).tolist()} stop "
              f"{np.round(shares['stop'], 4).tolist()}", flush=True)

    arm.train(training, args.rows, log)
    shares = {k: v.tolist() for k, v in arm.space.sheet.shares().items()}
    row: dict[str, Any] = {
        "arm": "sheet", "founder": args.founder, "train": args.train, "lessons": args.lessons,
        "cells": list(args.cells), "settings": args.settings, "mix": dict(MIX),
        "sheet_lessons": int(arm.space.sheet.lessons), "shares": shares,
        "kernel": arm.space.sheet.kernel.tolist(), "parameters": 90, "readings": {},
        "train_seconds": time.monotonic() - began}
    if args.save:
        arm.space.save(args.save, card={
            "lessons": args.lessons, "nursery": dict(MIX), "scenes": args.train,
            "founder": args.founder, "sheet": "settled belief sheet",
            "regions beside the sheet": "feed-forward baseline, untaught"})
    panels: dict[str, list[w.Scene]] = {
        "other": tr.test_scenes("other", args.founder, args.test, args.quota),
        "errands": errands(args.founder)}
    for path in args.scenes:
        panels[path.stem] = [w.load(path)]
    hazard = bf.fit_hazard(training[: args.fit]) if args.fit else None
    row["hazard"] = hazard
    for name, scenes in panels.items():
        answers, held, size = arm.answer(scenes)
        reading: dict[str, Any] = {"sheet": score(scenes, answers, held, size),
                                   "last_seen": last_seen(scenes)}
        if hazard is not None:
            reading["control"] = control(scenes, hazard)
            reading["control_peak"] = control(scenes, hazard, "peak")
        row["readings"][name] = reading
        cells = [f"{arm_name} {100 * r['out_of_view']:.1f}"
                 + (f" (area {r['area_holds']:.2f})" if "area_holds" in r else "")
                 + "".join(f" {k} {100 * v:.1f}" for k, v in r["panels"].items())
                 for arm_name, r in reading.items()]
        print(f"{name}: " + " | ".join(cells), flush=True)
    row["seconds"] = time.monotonic() - began
    if args.out:
        args.out.write_text(json.dumps(row, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
