"""The map as the source of the glimpse (cadence#342).

The eyes report every thing in view to a ``cadence.space.map.Map``: its place around the body
and, as values carried on its cells, each sense's tuned cells (hue, lightness, shape, size,
surface) as codes. The thing the brain attends is read back from the map with ``Map.thing_at``:
its codes as the map holds them, and its direction and distance from the body, coded on the
eyes' own tuned cells. That read enters the brain where the eyes' direct report of the thing
in focus entered (the glimpse), so the brain's settle, lessons and voice are the meaning
chamber's. The scene's slots stay the eyes' report of what is in view.

A thing behind a screen is not reported and its place is not lit, so the map holds it where
it was last sensed, with its values; a sound that names its place (``heard``) moves it there
with its values.

The map takes a kind's number with each report, which says which report is which thing. Here
it is the world's kind of the thing; the brain reads no kind unit, only the values and the
place. The focus is given as the kind of the thing the trainers attend, and where that thing is
comes from the map.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import lane
import numpy as np
from cadence.meaning.host import MeaningParrot

from cadence.space.map import Map

nt = lane.night
PLANE, CELLS = (-2.0, 2.0, -2.0, 2.0), (32, 32)  # things lie 0.5 to 1.8 m from the body
KIND = {k: i for i, k in enumerate({**nt.KINDS, **nt.NEW_KINDS})}
SCREEN = 0.25  # metres around a hidden thing that the eyes do not see


def unit_fields(eyes: Any, glimpse: np.ndarray) -> np.ndarray:
    """Every sense field of the glimpse at unit length (a field with nothing in it stays zero):
    the equal-senses rule of the sparse mixed code."""
    g = np.array(glimpse, dtype=np.float64, copy=True)
    edges = np.cumsum([0, *eyes.parts.values()])
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        n = float(np.linalg.norm(g[a:b]))
        g[a:b] = g[a:b] / n if n > 0 else 0.0
    return g


def groups(eyes: Any) -> dict[str, int]:
    """The map's value groups: one for each sense whose tuned cells ride on a thing's place."""
    return {"hue": eyes.hues, "lightness": eyes.lights, "shape": eyes.shapes * eyes.per_shape,
            "size": eyes.sizes, "surface": eyes.surfaces * eyes.per_surface}


def codes(eyes: Any, seen: Any) -> dict[str, np.ndarray]:
    """What the eyes report of a thing as codes on the map's value groups."""
    slot = eyes.slot(seen)
    edges = np.cumsum([0, *eyes.parts.values()])
    part = {name: slot[a:b] for name, a, b in zip(eyes.parts, edges[:-1], edges[1:],
                                                   strict=True)}
    return {name: part[name] for name in groups(eyes)}


class MapEyes:
    """One body's map and the read of the thing it attends. Bodies of one layout share one
    map's graph; each holds its own carried state and gives it to the graph for its turn."""

    CARRIED = ("potential", "clock", "lost", "doubt", "reading", "_sensed", "_views")
    graphs: ClassVar[dict[Any, Map]] = {}

    def __init__(self, eyes: Any):
        self.eyes = eyes
        layout = tuple(groups(eyes).items())
        if layout not in self.graphs:
            # rooms hold one thing a kind: a report of a kind anywhere is that thing, moved
            self.graphs[layout] = Map(PLANE, CELLS, units=len(KIND), values=groups(eyes),
                                      stride=4.0)
        self.map = self.graphs[layout]
        self.shown: Any = None
        self.reset()

    def mine(self) -> Map:
        """The shared graph holding this body's carried state."""
        for name, value in self.held.items():
            setattr(self.map, name, value)
        return self.map

    def reset(self) -> None:
        self.map.reset()
        self.held = {name: getattr(self.map, name) for name in self.CARRIED}
        self.read: dict[Any, np.ndarray] = {}

    def look(self, things: list[tuple[str, np.ndarray, Any]], hidden: Any = (),
             heard: Any = ()) -> None:
        """One moment of the eyes: every thing in view (its kind, place and what the eyes report
        of it) goes to the map. The places in ``hidden`` lie behind a screen: unlit, and nothing
        there is reported. ``heard`` holds sounds that name a kind's place."""
        screens = [np.asarray(p, dtype=np.float64) for p in hidden]

        def lit(points: np.ndarray) -> np.ndarray:
            out = np.ones(len(points), dtype=bool)
            for at in screens:
                out &= np.linalg.norm(points - at, axis=1) > SCREEN
            return out

        self.mine().moment(seen=[(KIND[k], place, codes(self.eyes, seen))
                                 for k, place, seen in things],
                           heard=[(KIND[k], place) for k, place in heard],
                           lit=lit if screens else None)
        self.held = {name: getattr(self.map, name) for name in self.CARRIED}
        self.read = {}

    def glimpse(self, kind: str, heading: float, salient: bool = True) -> np.ndarray:
        """The map's read of the thing of ``kind`` in the eyes' slot layout: its codes as the
        map holds them, its direction from ``heading`` and its distance from the body on the
        eyes' tuned cells. Zeros where the map holds no place for the kind."""
        key = (kind, float(heading), bool(salient))
        if key not in self.read:
            e = self.eyes
            held = self.mine()
            cells = held.sheet(KIND[kind]).ravel() > 0
            got = held.thing_at(None, cells=cells, heading=heading) if cells.any() else None
            if got is None or got["kind"] != KIND[kind]:
                self.read[key] = np.zeros(e.slot_width)
            else:
                v = got["values"]
                self.read[key] = np.concatenate([
                    v["hue"], v["lightness"], v["shape"], v["size"],
                    e._direction(got["bearing"]),
                    e._distance(math.log(max(got["distance"], 1e-3))),
                    [1.0 if salient else 0.0], v["surface"]])
        return self.read[key]


def show(me: MapEyes, ex: Any) -> None:
    """An exchange begins: the eyes look over the room and the map takes it. Then what the
    exchange hides or moves (``extra``): ``hidden`` things go behind a screen, and a ``moved``
    thing (``{kind: its place before}``) was first seen at its earlier place and, hidden,
    made a sound at its place of the exchange."""
    me.reset()
    room, extra = ex.room, ex.extra
    hidden = [t for t in room.things if t.kind in extra.get("hidden", ())]
    was = extra.get("moved", {})
    me.look([(t.kind, was.get(t.kind, t.place), room.seen_of(t)) for t in room.things])
    if hidden:
        screens = [was.get(t.kind, t.place) for t in hidden] + [
            t.place for t in hidden if t.kind in was]
        in_view = [(t.kind, t.place, room.seen_of(t)) for t in room.things if t not in hidden]
        me.look(in_view, hidden=screens)
        if was:
            me.look(in_view, hidden=screens,
                    heard=[(t.kind, t.place) for t in hidden if t.kind in was])
    me.shown = ex


def observe(self: Any, exs: list[Any], t: int, own: list[str | None], praise: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    """``night.Parrot.observe`` with the glimpse read from the map (``self.source`` "map") or
    from the eyes ("eyes", the meaning chamber's own); a hidden thing is in no slot of the
    scene, and the eyes report nothing of it."""
    source = getattr(self, "source", "map")
    if not hasattr(self, "mapped"):
        self.mapped = []
    while len(self.mapped) < len(exs):
        self.mapped.append(MapEyes(self.eyes))
    rows = []
    if self.mixed:
        if t == 0 or len(self.heard_trace) != len(exs):  # an exchange begins
            self.heard_trace = np.zeros((len(exs), self.ear))
        decay = float(self.genome.get("mixed_decay", 0.7))
        for i, ex in enumerate(exs):
            word = ex.heard[t] if t < len(ex.heard) else None
            self.heard_trace[i] = decay * self.heard_trace[i] + self.forms(word)
    for i, ex in enumerate(exs):
        word = ex.heard[t] if t < len(ex.heard) else None
        code = self.forms(word) + self.forms(own[i])
        if praise[i]:
            code = code + self.forms(nt.PRAISE)
        if self.records is not None:  # a recalled word, replayed to the ear
            code = code + self.recalled[i]
        me = self.mapped[i]
        if source == "map" and (t == 0 or me.shown is not ex):
            show(me, ex)
        things = ex.room.things
        hidden = ex.extra.get("hidden", ())
        seen = [ex.room.seen_of(things[int(j)]) for j in rng.permutation(len(things))
                if things[int(j)].kind not in hidden]
        focus = next((x for x in things if x.focus), None)
        if ex.scan and 0 <= t - ex.turn < len(ex.scan):  # the focus visits each thing
            focus = ex.scan[t - ex.turn]
        if focus is None:
            glimpse = np.zeros(self.eyes.slot_width)
        elif source == "map":
            glimpse = me.glimpse(focus.kind, ex.room.heading, focus.focus)
        else:
            glimpse = self.eyes.slot(None if focus.kind in hidden else ex.room.seen_of(focus))
        parts = [glimpse, self.eyes.scene(seen)]
        if self.mixed:
            parts.append(self.mixed_code(glimpse, self.heard_trace[i]))
        rows.append(np.concatenate([*parts, code]))
    return np.asarray(rows)


def install(source: str = "map", fields: bool = True) -> None:
    """Every ``night.Parrot`` made from here on observes through ``observe`` (``source`` "map"
    or "eyes"); with ``fields`` its sparse mixed code takes each sense field at unit length."""
    nt.Parrot.observe = observe
    nt.Parrot.source = source
    if fields and not getattr(nt.Parrot, "_fields", False):
        plain = nt.Parrot.mixed_code
        nt.Parrot.mixed_code = lambda self, glimpse, trace: plain(
            self, unit_fields(self.eyes, glimpse), trace)
        nt.Parrot._fields = True


class MappedHost(MeaningParrot):
    """The page's host with the glimpse read from a map (``source`` "map") or from the eyes,
    its sparse mixed code taking each sense field at unit length. ``see`` reports the things to
    the map; a thing with ``hidden`` set lies behind a screen: it is in no slot of the scene
    and the map holds it as it was last seen."""

    def __init__(self, *args: Any, source: str = "map", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.source = source
        self.me = MapEyes(self.eyes)
        self.room: Any = None
        self.told: list[dict[str, Any]] = []
        self.attended: dict[str, Any] | None = None

    def see(self, things: list[dict[str, Any]], glimpse: dict[str, Any] | None = None) -> None:
        room = tuple(sorted((t["kind"], round(float(t["direction"]), 6),
                             round(float(t["distance"]), 6)) for t in things))
        if room != self.room:  # another room: the map begins empty
            self.me.reset()
            self.room = room
        self.told = list(things)[: self.eyes.slots]
        self.things = [self.seen(t) for t in self.told if not t.get("hidden")]
        self.attended = glimpse if glimpse is not None else next(
            (t for t in self.told if t.get("focus")), None)
        self.glimpsed = None

        def place(t: dict[str, Any]) -> np.ndarray:
            return float(t["distance"]) * np.array([math.cos(t["direction"]),
                                                    math.sin(t["direction"])])

        self.me.look([(t["kind"], place(t), self.seen(t)) for t in self.told
                      if not t.get("hidden")],
                     hidden=[place(t) for t in self.told if t.get("hidden")])

    def observation(self) -> np.ndarray:
        order = self.rng.permutation(len(self.things))
        scene = self.eyes.scene([self.things[int(i)] for i in order])
        t = self.attended
        if t is None:
            glimpse = np.zeros(self.eyes.slot_width)
        elif self.source == "map":
            glimpse = self.me.glimpse(t["kind"], 0.0, bool(t.get("focus", False)))
        else:
            glimpse = self.eyes.slot(None if t.get("hidden") else self.seen(t))
        ear = self.code(self.heard) + self.code(self.own) + self.recalled
        g = unit_fields(self.eyes, glimpse)
        g = g / max(float(np.linalg.norm(g)), 1e-9)
        trace = self.trace / max(float(np.linalg.norm(self.trace)), 1e-9)
        drive = np.concatenate([g, trace]) @ self.mixed_w
        code = np.zeros(self.mixed_w.shape[1])
        code[np.argpartition(drive, -self.mixed_on)[-self.mixed_on:]] = 1.0
        return np.concatenate([glimpse, scene, code, ear])[None]
