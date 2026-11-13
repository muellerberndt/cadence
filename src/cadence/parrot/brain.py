"""The parrot's first brain: two eyes, a map of what it sees and a sense of distance it learns.

One moment of the brain, from the two eyes' images and the body's own joint readings:

1. **Early vision** (``Vision``): the room's colours learned as ground, the things found as
   figures in each eye, and the disparity units over the region both eyes see. A figure of the
   left eye matched with one of the right eye is one sighting seen by both eyes.
2. **What it is** (``Kinds``): each sighting's look settles the category neurons; a new sort of
   thing takes a new kind.
3. **How far it is** (``Distance``): the sighting's cues settle the distance neurons. A newborn
   brain has no sense of distance.
4. **Where it is** (``cadence.space.map.Map``): the place is the eye's ray (from the eye's
   place and the figure's place in the image, as the body knows its own head) at the distance
   read. A sighting whose distance read is weak that lies on the ray toward a thing of its kind
   the map holds is that thing, kept at the place the map holds it: the distance that keeps it
   at one place. The map holds every thing where it was last seen, in view and out of it.
5. **The teacher: a thing stays where it is while the head turns.** Each eye's sightings of the
   last moments are kept. When the head has turned, the eye looks from another place (the eyes
   sit in front of the head's pivot, so a turn moves them), and two sightings of one kind, from
   two places, whose rays meet, are one thing at the place where they meet. The distance from
   each eye to that place is the lesson for the cues that eye saw. No distance comes from the
   world; the lessons come from the parrot's own sightings and its own head.
6. **Where to look** (born, the optic tectum's orienting): every few moments the head turns to
   bring a thing seen to the side in front of the beak, or to a new direction.

``silence(units)`` holds neurons at rest inside their settles, for the chamber's ablations;
``record`` keeps what the probes read.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..space.map import Map
from .distance import Distance
from .eyes import directions
from .kinds import Kinds, look
from .vision import Figure, Vision
from .world import EYE_AXIS, EYE_FIELD, NAMES, Body, Eyes

# the founder's genes: the hand-set wiring and settings each part is built with
GENES: dict[str, Any] = {
    # early vision
    "channels": [-2.0, 16.0, 0.5],  # the disparity units' preferred disparities, degrees
    "colour": 0.12, "match": 0.15, "min_thickness": 2.2,
    # the distance neurons and their cues
    "sizes": 8, "heights": 8, "units": 24, "span": [0.1, 1.6], "width": 1.5, "rate": 0.05,
    "floor": 0.02, "families": ["disparity", "size", "elevation"], "wiring": "full",
    # the teacher
    "window": 10, "parallax": 1.0, "gap": 0.012, "sources": ["turns"], "lessons": 8,
    # placing on the map
    "keep_angle": 6.0, "keep": 0.15, "default": 0.4,
    "cells": [54, 37], "empty": "keep", "step": 0.10, "stride": 0.25, "reach": 0.15,
    # kinds
    "capacity": 12, "vigilance": 0.2,
    # where to look
    "fixation": 5, "orient": 0.6, "scan": 150.0,
}
PLANE = (-0.8, 0.8, -0.55, 0.55)  # the table, (x low, x high, z low, z high)


@dataclass
class Sighting:
    """A thing seen at one moment: from ``origin`` (an eye, or the point between the eyes for a
    thing both see) along ``direction`` (world), its ``kind``, its distance ``code`` (the cues)
    and the distance read; ``truth`` is the chamber's thing (never read by the brain)."""

    moment: int
    eye: int  # 0 left, 1 right, 2 both
    origin: np.ndarray
    direction: np.ndarray
    kind: int | None
    code: np.ndarray
    distance: float = math.nan
    strength: float = 0.0
    place: np.ndarray | None = None
    kept: bool = False
    truth: int = -1
    figures: tuple[int, ...] = ()
    disparity: float = math.nan


def triangulate(o1: np.ndarray, u1: np.ndarray, o2: np.ndarray, u2: np.ndarray
                ) -> tuple[np.ndarray, float, float, float, float]:
    """Where two rays come nearest: the middle point, the gap there, the distances along each
    ray and the angle between them (radians)."""
    b = float(u1 @ u2)
    w0 = o1 - o2
    d, e = float(u1 @ w0), float(u2 @ w0)
    denom = 1.0 - b * b
    if denom < 1e-12:
        return o1, math.inf, math.nan, math.nan, 0.0
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    p1, p2 = o1 + s * u1, o2 + t * u2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2)), s, t, math.acos(min(1.0, abs(b)))


def flat(vector: np.ndarray) -> np.ndarray:
    """A 3-vector's part on the table's plane, ``(x, z)``."""
    return np.array([vector[0], vector[2]])


class Parrot:
    """The parrot's brain and body. ``genes`` change the founder's (``GENES``)."""

    def __init__(self, genes: dict[str, Any] | None = None, *, seed: int = 0,
                 backend: Any = "cpu"):
        self.genes = dict(GENES)
        self.genes.update(genes or {})
        g = self.genes
        self.seed = int(seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, 332]))
        low, high, step = (float(v) for v in g["channels"])
        self.vision = Vision(channels=np.arange(low, high + 1e-9, step), colour=g["colour"],
                             match=g["match"], min_thickness=g["min_thickness"])
        wiring = None
        probe = Distance(self.vision.units, sizes=g["sizes"], heights=g["heights"],
                         units=g["units"], families=tuple(g["families"]))
        if isinstance(g["wiring"], str) and g["wiring"] == "random":
            # random wiring of the same size: as many pairs, drawn at random among every input
            # neuron and every distance neuron
            count = len(probe.pairs)
            every = np.stack(np.meshgrid(np.arange(probe.width_in), probe.output,
                                         indexing="ij"), axis=-1).reshape(-1, 2)
            pick = self.rng.choice(len(every), size=min(count, len(every)), replace=False)
            wiring = every[np.sort(pick)]
        elif not isinstance(g["wiring"], str):
            wiring = np.asarray(g["wiring"], dtype=np.int64)
        self.distance = Distance(self.vision.units, sizes=g["sizes"], heights=g["heights"],
                                 units=g["units"], span=tuple(g["span"]), width=g["width"],
                                 rate=g["rate"], floor=g["floor"],
                                 families=tuple(g["families"]), wiring=wiring, backend=backend)
        self.kinds = Kinds(g["capacity"], vigilance=g["vigilance"], backend=backend)
        self.map = Map(PLANE, tuple(g["cells"]), units=g["capacity"], empty=g["empty"],
                       step=g["step"], stride=g["stride"], reach=g["reach"], backend=backend)
        self.body = Body()
        self.sightings: deque[list[Sighting]] = deque(maxlen=int(g["window"]))
        self.moments = 0
        self.lessons = 0
        self.record: list[dict[str, Any]] | None = None
        self.silenced = np.zeros(0, dtype=np.int64)
        self._next_look = 0
        self.last: dict[str, Any] = {}
        self.regions = self._regions()

    # -- the brain's regions, for the brain view, the probes and silencing

    def _regions(self) -> dict[str, slice]:
        d = self.distance
        sizes = [("disparity", d.counts["disparity"]), ("size", d.counts["size"]),
                 ("elevation", d.counts["elevation"]), ("distance", d.units),
                 ("look", self.kinds.features), ("kind", self.kinds.capacity),
                 ("map", self.map.neurons)]
        out: dict[str, slice] = {}
        start = 0
        for name, many in sizes:
            out[name] = slice(start, start + int(many))
            start += int(many)
        return out

    @property
    def neurons(self) -> int:
        return int(max(s.stop for s in self.regions.values()))

    def parameters(self) -> dict[str, int]:
        """Learned parameters by part (the map's graph has none)."""
        return {"distance": self.distance.parameters(), "kinds": self.kinds.parameters()}

    def silence(self, units: np.ndarray | list[int] | None) -> None:
        """Hold these neurons (indices into the brain's regions, ``regions``) at rest inside
        their settles; None or empty silences none."""
        chosen = np.unique(np.asarray([] if units is None else units, dtype=np.int64))
        self.silenced = chosen
        self.distance.mask = np.ones(self.distance.neurons)
        self.kinds.mask = np.ones(self.kinds.neurons)
        r = self.regions
        for unit in chosen:
            if r["disparity"].start <= unit < r["elevation"].stop:
                self.distance.mask[unit - r["disparity"].start] = 0.0
            elif r["distance"].start <= unit < r["distance"].stop:
                self.distance.mask[self.distance.width_in + unit - r["distance"].start] = 0.0
            elif r["look"].start <= unit < r["look"].stop:
                self.kinds.mask[self.kinds.capacity + unit - r["look"].start] = 0.0
            elif r["kind"].start <= unit < r["kind"].stop:
                self.kinds.mask[unit - r["kind"].start] = 0.0
            elif not r["map"].start <= unit < r["map"].stop:
                raise ValueError(f"no neuron {int(unit)} in the brain")

    # -- one moment

    def _ray(self, eyes: Eyes, eye: int, figure: Figure) -> tuple[np.ndarray, np.ndarray]:
        head = directions(np.array(figure.azimuth), np.array(figure.elevation))
        return eyes.places[eye].copy(), np.asarray(eyes.rotation @ head)

    def lit(self, eyes: Eyes) -> Any:
        """The table's places the eyes see now: within either eye's field (5 degrees in from
        its edges), 5 cm to 1.5 m from the head's pivot."""
        yaw = math.radians(self.body.joint("head_yaw"))
        pivot = flat(eyes.pivot)
        edge = EYE_AXIS + EYE_FIELD - math.radians(5.0)

        def seen(points: np.ndarray) -> np.ndarray:
            p = np.asarray(points, dtype=np.float64).reshape(-1, 2) - pivot
            a = np.arctan2(p[:, 0], p[:, 1]) - yaw
            a = (a + math.pi) % (2.0 * math.pi) - math.pi
            far = np.linalg.norm(p, axis=1)
            return np.asarray((np.abs(a) <= edge) & (far >= 0.05) & (far <= 1.5))

        return seen

    def moment(self, images: np.ndarray, *, learn: bool = True, ids: np.ndarray | None = None,
               act: bool = True) -> dict[str, Any]:
        """One moment: the two eyes' images ``(2, rows, columns, 3)`` uint8, taken at the body's
        present pose. With ``learn`` the ground, the kinds and the sense of distance take their
        lessons; with ``act`` the head turns where the reflex looks. ``ids`` is the chamber's
        truth, kept with the record only."""
        eyes = self.body.eyes()
        figures = self.vision.see(images, learn=learn, ids=ids)
        sightings: list[Sighting] = []
        per_eye: list[Sighting] = []
        taken: set[int] = set()
        for i, fig in enumerate(figures):
            size = math.sqrt(max(fig.area, 1e-9))
            o, u = self._ray(eyes, fig.eye, fig)
            code = self.distance.code(fig.disparity, size, math.degrees(fig.elevation))
            per_eye.append(Sighting(self.moments, fig.eye, o, u, None, code, truth=fig.truth,
                                    figures=(i,)))
            if i in taken:
                continue
            if fig.pair is not None and fig.eye == 0:
                other = figures[fig.pair]
                taken.add(fig.pair)
                head = directions(np.array(0.5 * (fig.azimuth + other.azimuth)),
                                  np.array(0.5 * (fig.elevation + other.elevation)))
                both = Sighting(self.moments, 2, eyes.middle, np.asarray(eyes.rotation @ head),
                                None, code, truth=fig.truth, figures=(i, fig.pair))
                if fig.disparity is not None:
                    both.disparity = float(self.vision.channels[int(fig.disparity.argmax())])
                sightings.append(both)
            else:
                sightings.append(Sighting(self.moments, fig.eye, o, u, None, code,
                                          truth=fig.truth, figures=(i,)))
        # what each is
        looks = np.array([look(np.mean([figures[k].rgb for k in s.figures], axis=0))
                          for s in sightings]).reshape(-1, self.kinds.features)
        kinds, kind_activity = self.kinds.see(looks, learn=learn)
        for s, k in zip(sightings, kinds, strict=True):
            s.kind = k
        for s in per_eye:
            owner = next(x for x in sightings if s.figures[0] in x.figures)
            s.kind = owner.kind
        # how far
        codes = np.array([s.code for s in sightings]).reshape(-1, self.distance.width_in)
        activity = (self.distance.settle(codes) if len(codes)
                    else np.zeros((0, self.distance.neurons)))
        far, strength = self.distance.read(activity)
        for s, d, q in zip(sightings, far, strength, strict=True):
            s.distance, s.strength = float(d), float(q)
        # where: the ray at the distance read, or the place the map holds on the ray
        reports = []
        held = [t for t in self.map.things() if t["place"] is not None]
        for s in sightings:
            if s.kind is None:
                continue
            u = flat(s.direction)
            norm = float(np.linalg.norm(u))
            if norm < 1e-6:
                continue
            u = u / norm
            o = flat(s.origin)
            reach = s.distance
            if not (s.strength >= self.genes["keep"] and np.isfinite(reach)):
                best, best_angle = None, math.radians(self.genes["keep_angle"])
                for t in held:
                    if t["kind"] != s.kind:
                        continue
                    to = np.asarray(t["place"]) - o
                    along = float(to @ u)
                    if along <= 0:
                        continue
                    angle = math.atan2(abs(float(to[0] * u[1] - to[1] * u[0])), along)
                    if angle < best_angle:
                        best, best_angle = along, angle
                if best is not None:
                    reach, s.kept = best, True
                elif not np.isfinite(reach):
                    reach = float(self.genes["default"])
            s.place = o + reach * u
            if PLANE[0] <= s.place[0] <= PLANE[1] and PLANE[2] <= s.place[1] <= PLANE[3]:
                reports.append((int(s.kind), s.place))
        self.map.moment(seen=reports, lit=self.lit(eyes))
        # the teacher
        taught = 0
        if learn:
            taught = self._teach(per_eye)
        self.sightings.append(per_eye)
        if self.record is not None:
            for s, a, ka in zip(sightings, activity, kind_activity, strict=True):
                self.record.append({
                    "moment": self.moments, "eye": s.eye, "truth": s.truth, "kind": s.kind,
                    "distance": s.distance, "strength": s.strength, "kept": s.kept,
                    "place": None if s.place is None else s.place.tolist(),
                    "origin": s.origin.tolist(), "direction": s.direction.tolist(),
                    "disparity": s.disparity, "cues": a[: self.distance.width_in].copy(),
                    "units": a[self.distance.output].copy(),
                    "look": ka[self.kinds.feature].copy(), "kinds": ka[self.kinds.category].copy(),
                    "yaw": self.body.joint("head_yaw")})
        # where to look next
        if act:
            self.body.move(self._look(figures))
        self.moments += 1
        self.last = {"figures": figures, "sightings": sightings, "taught": taught,
                     "reports": reports}
        return self.last

    def _teach(self, now: list[Sighting]) -> int:
        """Lessons from sightings of one thing at two places of the eyes that agree."""
        g = self.genes
        rows: list[np.ndarray] = []
        targets: list[float] = []
        earlier = [s for moment in self.sightings for s in moment]
        if "eyes" in g["sources"]:
            earlier = earlier + now
        least = math.radians(g["parallax"])
        used = 0
        for s in now:
            if s.kind is None or used >= g["lessons"]:
                continue
            best: tuple[float, Sighting, np.ndarray] | None = None
            for e in earlier:
                if e is s or e.kind != s.kind or (e.moment == s.moment and e.eye == s.eye):
                    continue
                if e.moment == s.moment and "eyes" not in g["sources"]:
                    continue
                p, gap, a, b, angle = triangulate(s.origin, s.direction, e.origin, e.direction)
                if not (gap <= g["gap"] and angle >= least and a > 0.05 and b > 0.05):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, e, p)
            if best is None:
                continue
            _, e, p = best
            ds = float(np.linalg.norm(flat(p - s.origin)))
            de = float(np.linalg.norm(flat(p - e.origin)))
            if not (0.08 <= ds <= 1.5 and 0.08 <= de <= 1.5):
                continue
            rows += [s.code, e.code]
            targets += [ds, de]
            used += 1
        if rows:
            self.distance.teach(np.array(rows), np.array(targets))
            self.lessons += len(rows)
        return len(rows)

    def _look(self, figures: list[Figure]) -> np.ndarray:
        """The head's targets: every ``fixation`` moments the head turns to bring a thing seen
        to the side in front of the beak (``orient`` of the turns) or to a new direction."""
        targets = np.full(len(NAMES), np.nan)
        targets[NAMES.index("head_pitch")] = targets[NAMES.index("head_roll")] = 0.0
        if self.moments < self._next_look:
            return targets
        g = self.genes
        yaw = self.body.joint("head_yaw")
        side = [f for f in figures if abs(math.degrees(f.azimuth)) > 20.0]
        if side and self.rng.random() < g["orient"]:
            chosen = side[int(self.rng.integers(len(side)))]
            goal = yaw + math.degrees(chosen.azimuth)
        else:
            goal = float(self.rng.uniform(-g["scan"], g["scan"]))
        goal = float(np.clip(goal, -170.0, 170.0))
        targets[NAMES.index("head_yaw")] = Body.scaled(
            np.where(np.arange(len(NAMES)) == NAMES.index("head_yaw"), goal,
                     self.body.angles))[NAMES.index("head_yaw")]
        self._next_look = self.moments + int(g["fixation"]) + int(self.rng.integers(0, 3))
        return targets

    def reset(self) -> None:
        """A new scene: the map empties and the head returns to rest; what was learned stays."""
        self.map.reset()
        self.body.reset()
        self.sightings.clear()
        self._next_look = self.moments


__all__ = ["GENES", "PLANE", "Parrot", "Sighting", "flat", "triangulate"]
