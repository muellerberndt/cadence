"""A belief over places, as a control: a grid filter with a sheet of mass that moves and a sheet at
rest.

This is a classical filter in plain NumPy, a control beside the Kalman filter. It sizes what a
belief over places can reach on the chamber's scenes before a brain region holds one. It is
given what the page gives a space: the camera's reports and, at every step, where the camera
would see a thing at this step (``in_sight``).

- **Two sheets** over the plane's cells: mass that moves and mass at rest.
- **Predict.** The moving sheet is shifted by the thing's seen movement (the slope of the line
  through its latest reports in a row), and a share ``hazard`` of it comes to rest where it is.
  Mass that would leave the plane rests at the edge.
- **Evidence.** A report puts all mass in the report's cell: on the moving sheet where the seen
  movement is above its noise, else at rest. No report empties every cell the camera sees.
- **Answers.** The point is the belief's mean, or its peak (``read="peak"``). The area is the
  smallest set of cells that holds ``RATE`` of the mass.

    python benchmarks/tracking/belief.py --scene benchmarks/tracking/scenes/promise.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking as tr  # noqa: E402
import world as w  # noqa: E402

RATE = 0.90


class Belief:
    def __init__(self, plane: tuple[float, float, float, float], cell: float = 0.04,
                 hazard: float = 0.01, noise: float = 0.01, read: str = "mean"):
        x_low, x_high, y_low, y_high = plane
        self.cell, self.hazard, self.noise, self.read = cell, hazard, noise, read
        self.nx = int(round((x_high - x_low) / cell))
        self.ny = int(round((y_high - y_low) / cell))
        self.low = np.array([x_low, y_low])
        xs = x_low + (np.arange(self.nx) + 0.5) * cell
        ys = y_low + (np.arange(self.ny) + 0.5) * cell
        self.centers = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)  # (nx, ny, 2)
        self.cells = self.nx * self.ny

    def begin(self, start: np.ndarray) -> None:
        self.moving = np.zeros((self.nx, self.ny))
        self.rest = np.zeros((self.nx, self.ny))
        self.trail = [np.asarray(start, dtype=np.float64)]
        self.velocity, self.gap = np.zeros(2), False
        self._write(np.asarray(start, dtype=np.float64), False)

    def _index(self, point: np.ndarray) -> tuple[int, int]:
        at = np.floor((point - self.low) / self.cell).astype(np.intp)
        return int(np.clip(at[0], 0, self.nx - 1)), int(np.clip(at[1], 0, self.ny - 1))

    def _write(self, point: np.ndarray, moves: bool) -> None:
        self.moving[:], self.rest[:] = 0.0, 0.0
        (self.moving if moves else self.rest)[self._index(point)] = 1.0
        self.exact = point.copy()  # a report is exact to the camera's noise, finer than a cell

    def _shift(self, sheet: np.ndarray, by: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``sheet`` moved by ``by`` metres, mass shared between the cells it lands between; the
        mass that would leave the plane is given back apart, in the edge cells it left by."""
        out, lost = np.zeros_like(sheet), np.zeros_like(sheet)
        fx, fy = by / self.cell
        ix, iy = int(np.floor(fx)), int(np.floor(fy))
        for dx, wx in ((ix, 1 - (fx - ix)), (ix + 1, fx - ix)):
            for dy, wy in ((iy, 1 - (fy - iy)), (iy + 1, fy - iy)):
                if wx * wy <= 0:
                    continue
                src = sheet * (wx * wy)
                xs = np.clip(np.arange(self.nx) + dx, 0, self.nx - 1)
                ys = np.clip(np.arange(self.ny) + dy, 0, self.ny - 1)
                inside = ((np.arange(self.nx) + dx >= 0) & (np.arange(self.nx) + dx < self.nx)
                          )[:, None] & ((np.arange(self.ny) + dy >= 0)
                                        & (np.arange(self.ny) + dy < self.ny))[None, :]
                np.add.at(out, (xs[:, None], ys[None, :]), src * inside)
                np.add.at(lost, (xs[:, None], ys[None, :]), src * ~inside)
        return out, lost

    def step(self, seen: bool, report: np.ndarray, in_sight: Any) -> None:
        """One step: predict, then the report, or the lack of one where the camera sees."""
        moved, lost = self._shift(self.moving, self.velocity)
        self.rest = self.rest + self.hazard * moved + lost
        self.moving = (1.0 - self.hazard) * moved
        self.exact = None
        if seen:
            if self.gap:  # the line through the reports begins anew at a return
                self.trail, self.gap = [], False
            self.trail.append(np.asarray(report, dtype=np.float64))
            run = np.array(self.trail[-20:])
            loud = len(run) >= 3
            self.velocity = np.zeros(2)
            if loud:
                t = np.arange(len(run)) - (len(run) - 1) / 2.0
                self.velocity = (t[:, None] * run).sum(axis=0) / (t * t).sum()
                if len(run) >= 6:  # a line of six reports or more says whether it moves
                    loud = bool(np.linalg.norm(self.velocity) > 3.0 * np.sqrt(2.0) * self.noise
                                / np.sqrt((t * t).sum()))
                if not loud:
                    self.velocity = np.zeros(2)
            self._write(self.trail[-1], loud)
            return
        self.gap = True
        dark = ~np.asarray(in_sight(self.centers.reshape(-1, 2)), dtype=bool).reshape(
            self.nx, self.ny)
        moving, rest = self.moving * dark, self.rest * dark
        total = moving.sum() + rest.sum()
        if total > 1e-12:  # where all of it lay in sight the belief is kept as it was
            self.moving, self.rest = moving / total, rest / total
        else:
            total = self.moving.sum() + self.rest.sum()
            self.moving, self.rest = self.moving / total, self.rest / total

    def where(self) -> tuple[np.ndarray, np.ndarray]:
        """The point, and the area: a mask of the smallest set of cells holding ``RATE``."""
        mass = self.moving + self.rest
        flat = mass.reshape(-1)
        order = np.argsort(-flat)
        count = int(np.searchsorted(np.cumsum(flat[order]), RATE * flat.sum())) + 1
        area = np.zeros(self.cells, dtype=bool)
        area[order[:count]] = True
        if self.exact is not None:
            return self.exact.copy(), area.reshape(self.nx, self.ny)
        if self.read == "peak":
            point = self.centers.reshape(-1, 2)[int(order[0])]
        else:
            point = (self.centers * mass[..., None]).sum(axis=(0, 1)) / mass.sum()
        return np.asarray(point, dtype=np.float64), area.reshape(self.nx, self.ny)

    def holds(self, area: np.ndarray, truth: np.ndarray) -> bool:
        return bool(area[self._index(np.asarray(truth, dtype=np.float64))])


def follow(scene: w.Scene, hazard: float, read: str = "mean", cell: float = 0.04
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the belief over a scene of one thing: answers ``(steps, 1, 2)``, whether the area
    held the true position, the area's size in cells, and milliseconds a step."""
    feed = tr.given(scene)
    belief = Belief(scene.geometry.plane, cell, hazard, read=read)
    belief.begin(feed["report"][0])
    answers = np.zeros((scene.steps, 1, 2))
    held = np.ones(scene.steps, dtype=bool)
    size = np.ones(scene.steps)
    answers[0, 0] = feed["report"][0]
    began = time.perf_counter()
    for t in range(1, scene.steps):
        def in_sight(points: np.ndarray, t: int = t) -> np.ndarray:
            return w.visible(scene.geometry, points, scene.joints[t])

        belief.step(bool(feed["seen"][t]), feed["report"][t], in_sight)
        answers[t, 0], area = belief.where()
        held[t], size[t] = belief.holds(area, scene.truth[t, 0]), area.sum()
    return answers, held, size, (time.perf_counter() - began) * 1000.0 / (scene.steps - 1)


def read_scene(scene: w.Scene, hazards: tuple[float, ...]) -> None:
    out = ~scene.in_view[:, 0]
    out[0] = False
    still = tr.errors(scene, tr.follow(scene, tr.Still)[0])[:, 0]
    print(f"last seen: out of view {100 * still[out].mean():.1f} cm")
    since = scene.unseen()[:, 0]
    for hazard in hazards:
        for read in ("mean", "peak"):
            answers, held, size, ms = follow(scene, hazard, read)
            err = tr.errors(scene, answers)[:, 0]
            worse = sum(err[out & (since == k)].mean() > still[out & (since == k)].mean() + 1e-9
                        for k in np.unique(since[out]))
            print(f"hazard {hazard:.3f} {read:4}: in view {100 * err[~out][1:].mean():.1f} cm, out "
                  f"of view {100 * err[out].mean():5.1f} cm (max {100 * err[out].max():5.1f}), "
                  f"area holds {held[out].mean():.2f} of the hidden steps at {size[out].mean():.0f}"
                  f" cells, step counts worse than last seen {worse} of "
                  f"{len(np.unique(since[out]))}, {ms:.2f} ms a step")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--founder", type=int)
    parser.add_argument("--hazards", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    args = parser.parse_args(argv)
    if args.scene:
        read_scene(w.load(args.scene), tuple(args.hazards))
    if args.founder is not None:
        scenes = tr.test_scenes("other", args.founder)
        for hazard in args.hazards:
            for read in ("mean", "peak"):
                got = [follow(s, hazard, read) for s in scenes]
                table = tr.table(scenes, [g[0] for g in got])
                held = np.concatenate([g[1][~s.in_view[:, 0]] for g, s in
                                       zip(got, scenes, strict=True)])
                panels = "  ".join(f"{k} {100 * v['error']:.1f}" for k, v in
                                   table["panels"].items())
                print(f"founder {args.founder} hazard {hazard:.3f} {read:4}: in view "
                      f"{100 * table['in_view']:.1f}, out of view {100 * table['out_of_view']:.1f}"
                      f" | {panels} | area holds {held.mean():.2f} out of view")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
