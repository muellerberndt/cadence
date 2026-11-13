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

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..space.map import Map
from .distance import FAMILIES, Distance
from .eyes import directions
from .focus import Choice, Focus
from .kinds import Kinds, look
from .vision import Figure, Vision
from .world import EYE_AXIS, EYE_FIELD, HIGH, LOW, NAMES, Body, Eyes

# The founder's genes: the hand-set wiring and settings each part is built with. Names follow
# the parrot's wiring guide (docs/parrot-brain.md, "The gene space"); a gene this brain does not
# express is carried in a genome unchanged. "wiring" is no gene: "random" is the control arm of
# random wiring of the same size.
GENES: dict[str, Any] = {
    # early vision: figures and the disparity units (the Wulst's binocular units)
    "figure_thickness": 2.2, "disparity_form": "edges", "disparity_channels": 37,
    "disparity_max": 16.0, "disparity_tuning": 0.5, "disparity_colour": 0.12,
    "disparity_match": 0.15,
    # the parallax units: one eye before and after the head moved
    "parallax_delay": 2, "parallax_channels": 12, "parallax_baselines": 6,
    "parallax_window": 8.0, "parallax_least": 0.003,
    # the reach: the distance neurons and which cues project onto them
    "reach_bins": 24, "reach_span": [0.12, 1.2], "reach_width": 1.5, "reach_rate": 0.3,
    "reach_floor": 0.02, "size_units": 8, "elevation_units": 8,
    "p_wulst_dl": 1, "p_glimpse_dl": 1, "p_both_cues": 0, "wiring": "full",
    # the teacher: two views of one thing that agree
    "agreement_views": "before-after", "agreement_scale": 0.012, "agreement_parallax": 1.0,
    "agreement_baseline": 0.003, "agreement_window": 10, "agreement_lessons": 8,
    # the step from gaze to place, and the map
    "place_keep": "one", "place_keep_angle": 6.0, "place_default": 0.4,
    "map_cells": 54, "empty_rule": "keep", "match_step": 0.10, "match_stride": 0.25,
    "map_reach": 0.15,
    # kinds
    "kinds_capacity": 12, "kinds_vigilance": 0.2,
    # the focus: the tectum and the isthmi
    "focus_appear": 1.0, "focus_move": 0.8, "novelty": 0.6, "isthmi_local": 1.2,
    "focus_habituate": 6.0, "focus_return": 30.0, "isthmi_global": 2.0, "focus_explore": 0.05,
    "focus_fixation": 5,
    # the head-bob rhythm on the neck's lean (held at zero in the control)
    "peer_amplitude": 0.0, "peer_period": 6,
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
    # where the sighting is in its moment's images: (moment, ((eye, column, row), ...)), for a
    # reach that reads the images themselves (a control)
    query: tuple[int, tuple[tuple[int, float, float], ...]] = (0, ())


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


def _plain(value: Any) -> Any:
    """A gene's value as JSON: arrays as lists, numbers as numbers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"no JSON for {type(value).__name__}")


def flat(vector: np.ndarray) -> np.ndarray:
    """A 3-vector's part on the table's plane, ``(x, z)``."""
    return np.array([vector[0], vector[2]])


class Parrot:
    """The parrot's brain and body. ``genes`` change the founder's (``GENES``)."""

    def __init__(self, genes: dict[str, Any] | None = None, *, seed: int = 0,
                 control: Any = None, voice: Any = None, backend: Any = "cpu"):
        # ``control``: an object that takes the distance neurons' place, reading the eyes'
        # images (``see(moment, images)``, ``read(queries)``, ``teach(queries, distances)``):
        # the matched controls of the chamber
        self.control = control
        # ``voice``: when to speak, learned (``cadence.parrot.voice.Voice``): it hears each
        # moment's frames and its onset opens the beak; a pistachio pays its act
        self.voice = voice
        self.genes = dict(GENES)
        self.genes.update(genes or {})
        g = self.genes
        self.seed = int(seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, 332]))
        channels = np.linspace(-2.0, float(g["disparity_max"]), int(g["disparity_channels"]))
        families = (["disparity", "parallax"] if int(g["p_wulst_dl"]) else []) + (
            ["size_one", "elevation_one"] if int(g["p_glimpse_dl"]) else []) + (
            ["size_both", "elevation_both"] if int(g["p_both_cues"]) else [])
        if not families:
            raise ValueError("the reach needs one projection of cues or more")
        self.vision = Vision(channels=channels, colour=g["disparity_colour"],
                             match=g["disparity_match"], min_thickness=g["figure_thickness"],
                             form=g["disparity_form"], tuning=g["disparity_tuning"])
        wiring = None

        def reach(wiring: np.ndarray | None = None) -> Distance:
            return Distance(self.vision.units, sizes=int(g["size_units"]),
                            heights=int(g["elevation_units"]), units=int(g["reach_bins"]),
                            shifts=int(g["parallax_channels"]),
                            baselines=int(g["parallax_baselines"]),
                            span=(float(g["reach_span"][0]), float(g["reach_span"][1])),
                            width=float(g["reach_width"]), rate=float(g["reach_rate"]),
                            floor=float(g["reach_floor"]), families=tuple(families),
                            wiring=wiring, backend=backend)

        probe = reach()
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
        self.distance = reach(wiring)
        self.kinds = Kinds(g["kinds_capacity"], vigilance=g["kinds_vigilance"], backend=backend)
        cells = (int(g["map_cells"]), max(3, round(int(g["map_cells"]) * (PLANE[3] - PLANE[2])
                                                    / (PLANE[1] - PLANE[0]))))
        self.map = Map(PLANE, cells, units=g["kinds_capacity"], empty=g["empty_rule"],
                       step=g["match_step"], stride=g["match_stride"], reach=g["map_reach"],
                       backend=backend)
        self.focus = Focus(appear=g["focus_appear"], move=g["focus_move"], novel=g["novelty"],
                           hold=g["isthmi_local"], habituate=g["focus_habituate"],
                           return_time=g["focus_return"],
                           inhibition=g["isthmi_global"], backend=backend)
        self.focused: Choice | None = None
        self.cue: dict[int, float] | None = None  # a heard thing's kind and weight (later)
        self.body = Body()
        self.sightings: deque[list[Sighting]] = deque(maxlen=int(g["agreement_window"]))
        self.moments = 0
        self.lessons = 0
        self.record: list[dict[str, Any]] | None = None
        self.taught: list[dict[str, Any]] | None = None  # the teacher's lessons, for a reader
        self.silenced = np.zeros(0, dtype=np.int64)
        self._next_look = 0
        self.last: dict[str, Any] = {}
        self.seconds: dict[str, float] = {}  # time spent by part, over the moments lived
        self.regions = self._regions()

    # -- the brain's regions, for the brain view, the probes and silencing

    def _regions(self) -> dict[str, slice]:
        d = self.distance
        sizes = [(name, d.counts[name]) for name in FAMILIES] + [("distance", d.units),
                 ("look", self.kinds.features), ("kind", self.kinds.capacity),
                 ("focus", self.focus.slots), ("map", self.map.neurons)]
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

    @property
    def settling(self) -> np.ndarray:
        """The neurons ``silence`` can hold at rest: every region but the map's sheet."""
        return np.arange(self.regions["disparity"].start, self.regions["focus"].stop)

    def silence(self, units: np.ndarray | list[int] | None) -> None:
        """Hold these neurons (indices into the brain's regions, ``regions``) at rest inside
        their settles; None or empty silences none."""
        chosen = np.unique(np.asarray([] if units is None else units, dtype=np.int64))
        self.silenced = chosen
        self.distance.mask = np.ones(self.distance.neurons)
        self.kinds.mask = np.ones(self.kinds.neurons)
        self.focus.mask = np.ones(self.focus.slots)
        r = self.regions
        for unit in chosen:
            if r["disparity"].start <= unit < r["elevation_both"].stop:
                self.distance.mask[unit - r["disparity"].start] = 0.0
            elif r["distance"].start <= unit < r["distance"].stop:
                self.distance.mask[self.distance.width_in + unit - r["distance"].start] = 0.0
            elif r["look"].start <= unit < r["look"].stop:
                self.kinds.mask[self.kinds.capacity + unit - r["look"].start] = 0.0
            elif r["kind"].start <= unit < r["kind"].stop:
                self.kinds.mask[unit - r["kind"].start] = 0.0
            elif r["focus"].start <= unit < r["focus"].stop:
                self.focus.mask[unit - r["focus"].start] = 0.0
            else:
                raise ValueError(f"neuron {int(unit)} is not one the brain can silence (the "
                                 "map's sheet is carried state)")

    # -- one moment

    def _ray(self, eyes: Eyes, eye: int, figure: Figure) -> tuple[np.ndarray, np.ndarray]:
        head = directions(np.array(figure.azimuth), np.array(figure.elevation))
        return eyes.places[eye].copy(), np.asarray(eyes.rotation @ head)

    def lit(self, eyes: Eyes) -> Any:
        """The table's places the eyes see at this pose: within either eye's field (5 degrees
        in from its edges), 5 cm to 1.5 m from the head's pivot."""
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
               act: bool = True, frames: np.ndarray | None = None, reward: float | None = None,
               done: bool = False) -> dict[str, Any]:
        """One moment: the two eyes' images ``(2, rows, columns, 3)`` uint8, taken at the body's
        present pose. With ``learn`` the ground, the kinds and the sense of distance take their
        lessons; with ``act`` the head turns where the reflex looks. ``ids`` is the chamber's
        truth, kept with the record only."""
        clock = time.perf_counter()
        eyes = self.body.eyes()
        if self.control is not None:
            self.control.see(self.moments, images)
        figures = self.vision.see(images, learn=learn, ids=ids)
        clock = self._time("vision", clock)
        sightings: list[Sighting] = []
        per_eye: list[Sighting] = []
        taken: set[int] = set()
        for i, fig in enumerate(figures):
            o, u = self._ray(eyes, fig.eye, fig)
            per_eye.append(Sighting(self.moments, fig.eye, o, u, None, np.zeros(0),
                                    truth=fig.truth, figures=(i,)))
            if i in taken:
                continue
            if fig.pair is not None and fig.eye == 0:
                other = figures[fig.pair]
                taken.add(fig.pair)
                head = directions(np.array(0.5 * (fig.azimuth + other.azimuth)),
                                  np.array(0.5 * (fig.elevation + other.elevation)))
                both = Sighting(self.moments, 2, eyes.middle, np.asarray(eyes.rotation @ head),
                                None, np.zeros(0), truth=fig.truth, figures=(i, fig.pair))
                if fig.disparity is not None and fig.disparity.sum() > 0:
                    both.disparity = float((fig.disparity * self.vision.channels).sum()
                                           / fig.disparity.sum())
                sightings.append(both)
            else:
                sightings.append(Sighting(self.moments, fig.eye, o, u, None, np.zeros(0),
                                          truth=fig.truth, figures=(i,)))
        clock = self._time("sightings", clock)
        # what each is
        looks = np.array([look(np.mean([figures[k].rgb for k in s.figures], axis=0))
                          for s in sightings]).reshape(-1, self.kinds.features)
        kinds, kind_activity = self.kinds.see(looks, learn=learn)
        for s, k in zip(sightings, kinds, strict=True):
            s.kind = k
        for s in per_eye:
            owner = next(x for x in sightings if s.figures[0] in x.figures)
            s.kind = owner.kind
        clock = self._time("kinds", clock)
        # the cues of each eye's sighting: the disparity units for a thing both eyes see, the
        # parallax units for a thing one eye sees after the head moved
        for x in per_eye:
            fig = figures[x.figures[0]]
            x.code = self.distance.code(fig.disparity, math.sqrt(max(fig.area, 1e-9)),
                                        math.degrees(fig.elevation),
                                        None if fig.disparity is not None else self._parallax(x))
        for s in sightings:
            s.code = per_eye[s.figures[0]].code
        for x in per_eye + sightings:
            x.query = (self.moments, tuple((figures[k].eye, figures[k].col, figures[k].row)
                                           for k in x.figures))
        # how far
        codes = np.array([s.code for s in sightings]).reshape(-1, self.distance.width_in)
        if self.control is not None:
            activity = np.zeros((len(codes), self.distance.neurons))
            far, strength = self.control.read([s.query for s in sightings])
        else:
            activity = (self.distance.settle(codes) if len(codes)
                        else np.zeros((0, self.distance.neurons)))
            far, strength = self.distance.read(activity)
        for s, d, q in zip(sightings, far, strength, strict=True):
            s.distance, s.strength = float(d), float(q)
        clock = self._time("distance", clock)
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
            # a thing one eye sees, on the ray toward a thing of its kind the map holds, is kept
            # where the map holds it; a thing both eyes see is placed at the distance read
            weak = s.eye != 2 if self.genes["place_keep"] == "one" else not s.strength >= float(
                self.genes["place_keep"])
            if weak or not np.isfinite(reach):
                best, best_angle = None, math.radians(self.genes["place_keep_angle"])
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
                    reach = float(self.genes["place_default"])
            s.place = o + reach * u
            if PLANE[0] <= s.place[0] <= PLANE[1] and PLANE[2] <= s.place[1] <= PLANE[3]:
                reports.append((int(s.kind), s.place))
        clock = self._time("place", clock)
        self.map.moment(seen=reports, lit=self.lit(eyes))
        clock = self._time("map", clock)
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
                    "yaw": self.body.joint("head_yaw"), "eyes": eyes.places.copy(),
                    "rotation": eyes.rotation.copy()})
        clock = self._time("teacher", clock)
        # the voice: whether it starts, from what the ear holds; its onset opens the beak
        vocalized = False
        if self.voice is not None:
            vocalized = bool(self.voice.moment(frames, reward=reward, done=done))
        # where to look next
        if act:
            targets = self._look()
            if self.voice is not None:
                targets[NAMES.index("beak_open")] = 0.0 if vocalized else -1.0
            self.body.move(targets)
        self._time("look", clock)
        self.moments += 1
        self.last = {"figures": figures, "sightings": sightings, "taught": taught,
                     "vocalized": vocalized,
                     "reach": activity, "kind_activity": kind_activity,
                     "reports": reports}
        return self.last

    def _parallax(self, s: Sighting) -> tuple[float, float] | None:
        """The parallax units' cue for a thing one eye sees: where the same eye saw a thing of
        its kind ``parallax_delay`` moments before (the nearest in direction, within
        ``parallax_window`` degrees), the shift between the two rays (degrees, toward where the
        eye moved from) and how far the eye moved across the line of sight (metres). None where
        the eye did not move or saw no such thing."""
        k = int(self.genes["parallax_delay"])
        if s.kind is None or len(self.sightings) < k:
            return None
        best, angle = None, math.radians(float(self.genes["parallax_window"]))
        for e in self.sightings[-k]:
            if e.eye != s.eye or e.kind != s.kind:
                continue
            a = math.acos(min(1.0, float(s.direction @ e.direction)))
            if a < angle:
                best, angle = e, a
        if best is None:
            return None
        moved = s.origin - best.origin
        across = moved - float(moved @ s.direction) * s.direction
        b = float(np.linalg.norm(across))
        if b < float(self.genes["parallax_least"]):
            return None
        shift = float((best.direction - s.direction) @ (across / b))
        return math.degrees(max(shift, 0.0)), b

    def _time(self, part: str, since: float) -> float:
        now = time.perf_counter()
        self.seconds[part] = self.seconds.get(part, 0.0) + now - since
        return now

    def cost(self) -> dict[str, Any]:
        """The cost of the moments lived: seconds a moment by part, and the multiply-accumulates
        of the settles (the distance neurons, the kinds and the map's sheet)."""
        moments = max(self.moments, 1)
        macs = {"distance": self.distance.work["macs"], "kinds": self.kinds.work["macs"]}
        # the map's settle: two sweeps over its synapses a moment
        macs["map"] = 2 * int(self.map.connectome.synapses) * self.moments
        return {"moments": self.moments,
                "ms": {k: 1000.0 * v / moments for k, v in self.seconds.items()},
                "macs": {k: v / moments for k, v in macs.items()}}

    def _teach(self, now: list[Sighting]) -> int:
        """Lessons from sightings of one thing at two places of the eyes that agree."""
        g = self.genes
        rows: list[np.ndarray] = []
        targets: list[float] = []
        queries: list[Any] = []
        earlier = [s for moment in self.sightings for s in moment]
        views = str(g["agreement_views"])
        if views not in ("before-after", "binocular", "both"):
            raise ValueError("agreement_views is 'before-after', 'binocular' or 'both'")
        if views == "binocular":
            earlier = list(now)
        elif views == "both":
            earlier = earlier + now
        least = math.radians(g["agreement_parallax"])
        used = 0
        for s in now:
            if s.kind is None or used >= g["agreement_lessons"]:
                continue
            best: tuple[float, Sighting, np.ndarray, float] | None = None
            for e in earlier:
                if e is s or e.kind != s.kind or (e.moment == s.moment and e.eye == s.eye):
                    continue
                if (e.moment == s.moment) != (views != "before-after") and views != "both":
                    continue
                if e.moment != s.moment and float(np.linalg.norm(e.origin - s.origin)) < float(
                        g["agreement_baseline"]):
                    continue
                p, gap, a, b, angle = triangulate(s.origin, s.direction, e.origin, e.direction)
                if not (gap <= g["agreement_scale"] and angle >= least and a > 0.05 and b > 0.05):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, e, p, angle)
            if best is None:
                continue
            gap, e, p, angle = best
            ds = float(np.linalg.norm(flat(p - s.origin)))
            de = float(np.linalg.norm(flat(p - e.origin)))
            if not (0.08 <= ds <= 1.5 and 0.08 <= de <= 1.5):
                continue
            rows += [s.code, e.code]
            targets += [ds, de]
            queries += [s.query, e.query]
            used += 1
            if self.taught is not None:
                for x, d in ((s, ds), (e, de)):
                    self.taught.append({"moment": self.moments, "truth": x.truth,
                                        "origin": x.origin.copy(), "target": d, "eye": x.eye,
                                        "binocular": bool(x.code[: self.distance.counts[
                                            "disparity"]].any()),
                                        "angle": math.degrees(angle), "gap": gap})
        if rows:
            if self.control is not None:
                self.control.teach(queries, np.array(targets))
            else:
                self.distance.teach(np.array(rows), np.array(targets))
            self.lessons += len(rows)
        return len(rows)

    def _look(self) -> np.ndarray:
        """The head's targets: the focus (one winner among the things on the map) turns the head
        so that it lies in front of the beak, as far as the neck allows. With no focus, and at
        ``explore`` of the moments, the head looks to a new direction, held ``fixation``
        moments."""
        targets = np.full(len(NAMES), np.nan)
        targets[NAMES.index("head_pitch")] = targets[NAMES.index("head_roll")] = 0.0
        k = NAMES.index("head_yaw")
        g = self.genes
        if float(g["peer_amplitude"]) > 0.0:
            # the head-bob rhythm: the neck leans forward and back, which moves the eyes
            n = NAMES.index("neck_pitch")
            phase = 2.0 * math.pi * self.moments / max(int(g["peer_period"]), 2)
            lean = float(g["peer_amplitude"]) * HIGH[n] * 0.5 * (1.0 + math.sin(phase))
            targets[n] = 2.0 * (lean - LOW[n]) / (HIGH[n] - LOW[n]) - 1.0
        self.focused = self.focus.choose(self.map.things(), self.moments, self.cue)
        if self.moments < self._next_look:
            return targets
        if self.focused is not None and self.rng.random() >= float(g["focus_explore"]):
            to = self.focused.place - flat(self.body.eyes().pivot)
            goal = math.degrees(math.atan2(float(to[0]), float(to[1])))
        else:
            goal = float(self.rng.uniform(LOW[k], HIGH[k]))
            self._next_look = self.moments + int(g["focus_fixation"])
        goal = float(np.clip(goal, LOW[k], HIGH[k]))
        targets[k] = 2.0 * (goal - LOW[k]) / (HIGH[k] - LOW[k]) - 1.0
        return targets

    # -- what a file keeps

    FORMAT = "cadence-parrot-brain-arrays/1"

    def arrays(self) -> dict[str, np.ndarray]:
        """What was learned, as arrays: the ground's colours, the kinds and the reach."""
        out = {f"ground.{k}": v for k, v in self.vision.ground.arrays().items()}
        out.update({f"kinds.{k}": v for k, v in self.kinds.arrays().items()})
        out.update({f"distance.{k}": v for k, v in self.distance.arrays().items()})
        return out

    def save(self, path: str | Path, *, card: dict[str, Any] | None = None) -> Path:
        """Write the genes, a card and what was learned to one ``.npz`` (no pickled object)."""
        target = Path(path)
        record = {"format": self.FORMAT, "genes": self.genes, "seed": self.seed,
                  "moments": self.moments, "lessons": self.lessons, "card": card or {}}
        text = json.dumps(record, default=_plain)
        payload: dict[str, Any] = {"record": np.frombuffer(text.encode("utf-8"), dtype=np.uint8),
                                   **self.arrays()}
        np.savez(target, **payload)
        return target if target.suffix == ".npz" else target.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path, *, backend: Any = "cpu") -> Parrot:
        """A brain from ``save``'s file: its genes and what it learned."""
        with np.load(Path(path), allow_pickle=False) as data:
            record = json.loads(bytes(data["record"]).decode("utf-8"))
            if record.get("format") != cls.FORMAT:
                raise ValueError(f"{path} is no {cls.FORMAT} file")
            arrays = {k: np.asarray(data[k]) for k in data.files if k != "record"}
        genes = dict(record["genes"])
        if isinstance(genes.get("wiring"), list):
            genes["wiring"] = np.asarray(genes["wiring"], dtype=np.int64)
        brain = cls(genes, seed=int(record["seed"]), backend=backend)
        brain.vision.ground.take({k[7:]: v for k, v in arrays.items() if k.startswith("ground.")})
        brain.kinds.take({k[6:]: v for k, v in arrays.items() if k.startswith("kinds.")})
        brain.distance.take({k[9:]: v for k, v in arrays.items() if k.startswith("distance.")})
        brain.moments, brain.lessons = int(record["moments"]), int(record["lessons"])
        return brain

    def reset(self) -> None:
        """A new scene: the map empties and the head returns to rest; what was learned stays."""
        self.map.reset()
        self.body.reset()
        self.sightings.clear()
        self.focus.reset()
        if self.voice is not None:
            self.voice.level = None
        self.focused = None
        self._next_look = self.moments


__all__ = ["GENES", "PLANE", "Parrot", "Sighting", "flat", "triangulate"]
