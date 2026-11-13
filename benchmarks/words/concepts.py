"""The words chamber with concepts the brain learns from its senses (cadence#326, learned
concepts).

The parrot's eyes report each thing in view as raw readings: its shape and size as three features
(roundness, elongation, size, each in 0 to 1), its place and its hue (an angle). No reading says
which kind or which colour a thing is. Each founder has its own **teacher**, who names by its own
categories:

- **colours**: six bands of the hue circle whose boundaries lie at 60 degrees apart, each moved by
  a draw within ``SHIFT`` degrees for the founder, so one teacher's red ends where another's does
  not;
- **kinds**: twelve shapes, each a point of the feature space drawn for the founder (at least
  ``APART`` from every other), a thing of the kind drawn around it at ``SPREAD`` a feature;
- **places around the body**: in front, to the right, behind, to the left, each boundary moved
  from the quarters by a draw within ``TURN`` degrees.

Held out of every training room: a gap of ``GAP`` degrees inside each colour's band, and for each
kind one side of one feature (things of the kind in training lie on the other side).

The script is the protocol 1 chamber's (``words.py``: naming rooms, practice, the test, new
words), its things given raw readings and its places and answers the teacher's, and adds:

- ``held_rooms`` test rooms whose things have held-out hues and held-out shapes;
- **probes** read after the life, with nothing taught: the brain's word for hues around the
  circle, for fresh things of each trained kind and for directions around the body, against the
  teacher's (C1); for the held-out hues and shapes (C2); and the hue and the direction where the
  brain's word changes near each of the teacher's boundaries (C3).

``brain``: ``"learned"`` (``cadence.words.concepts.Concepts``), ``"senses"`` (the same regions
taught by the senses alone: no naming lesson), ``"given"`` (protocol 1's given concepts: the
thing's kind, the colour band of the hue at the unmoved boundaries, the unmoved quarters).

    python concepts.py <seed> [json settings]    # one life, its readings as JSON
    python concepts.py run <out> <first> <count> [jobs] [arms]
    python concepts.py read <out>
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import words  # noqa: E402
import world  # noqa: E402

SHAPE = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))  # roundness, elongation, size
SPREAD, APART, EDGE = 0.04, 0.3, 0.1  # a kind's spread a feature; the least distance of kinds
SHIFT, TURN = math.radians(15.0), math.radians(15.0)  # the teacher's moves of the boundaries
GAP, MARGIN = math.radians(6.0), math.radians(6.0)  # a colour's held-out gap, kept from edges
DEFAULTS = {**words.DEFAULTS, "naming_rooms": 160, "held_rooms": 16, "joint": True,
            "naming_rates": (1 / 3, 1 / 3, 1 / 3)}
TAU = 2.0 * math.pi


@dataclass
class Teacher:
    """A founder's teacher: where its colour names change (``bounds``, radians, ascending from
    the first colour's start), where its place names change around the body (``turns``: the
    boundaries front/right, right/behind, behind/left, left/front, radians from the heading),
    each kind's shape (``shapes``), each colour's held-out gap (its centre) and each kind's
    held-out side (``side``: the feature and the sign)."""

    bounds: np.ndarray
    turns: np.ndarray
    shapes: np.ndarray
    gaps: np.ndarray
    side: list[tuple[int, int]]

    def colour(self, hue: float) -> int:
        """The colour the teacher names a hue."""
        h = (float(hue) - self.bounds[0]) % TAU
        return int(np.searchsorted((self.bounds - self.bounds[0]) % TAU, h, side="right") - 1)

    def band(self, colour: int) -> tuple[float, float]:
        """A colour's band: its start and its width, radians."""
        start = float(self.bounds[colour])
        end = float(self.bounds[(colour + 1) % len(self.bounds)])
        return start, (end - start) % TAU

    def place(self, angle: float) -> int:
        """The place the teacher names a direction (radians from where the body faces,
        counter-clockwise): 0 in front, 1 to the right, 2 behind, 3 to the left."""
        a = float(np.angle(np.exp(1j * angle)))
        fr, rb, bl, lf = (float(t) for t in self.turns)
        if fr <= a < lf:
            return 0
        if rb <= a < fr:
            return 1
        if lf <= a < bl:
            return 3
        return 2

    def gap(self, colour: int) -> tuple[float, float]:
        return float(self.gaps[colour] - GAP / 2), float(self.gaps[colour] + GAP / 2)


def teacher(seed: int) -> Teacher:
    """The founder's teacher, drawn from its own stream."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 77]))
    start = float(rng.uniform(0.0, TAU))  # the first colour's band starts here
    bounds = (start + np.arange(world.COLOURS) * TAU / world.COLOURS
              + rng.uniform(-SHIFT, SHIFT, world.COLOURS)) % TAU
    turns = np.radians([-45.0, -135.0, 135.0, 45.0]) + rng.uniform(-TURN, TURN, 4)
    shapes: list[np.ndarray] = []
    while len(shapes) < world.KINDS:
        s = rng.uniform(EDGE, 1.0 - EDGE, len(SHAPE))
        if all(np.linalg.norm(s - o) >= APART for o in shapes):
            shapes.append(s)
    out = Teacher(bounds=bounds, turns=turns, shapes=np.array(shapes),
                  gaps=np.zeros(world.COLOURS), side=[])
    for c in range(world.COLOURS):
        first, width = out.band(c)
        out.gaps[c] = first + rng.uniform(MARGIN + GAP / 2, width - MARGIN - GAP / 2)
    out.side = [(int(rng.integers(len(SHAPE))), int(rng.choice([-1, 1])))
                for _ in range(world.KINDS)]
    return out


def hue_for(tea: Teacher, colour: int, rng: np.random.Generator, held: bool = False) -> float:
    """A hue of a colour: in its held-out gap, or anywhere in its band out of the gap."""
    if held:
        low, high = tea.gap(colour)
        return float(rng.uniform(low, high) % TAU)
    first, width = tea.band(colour)
    low, high = tea.gap(colour)
    while True:
        h = first + float(rng.uniform(0.0, width))
        if not low <= h <= high and not low <= h + TAU <= high:
            return float(h % TAU)


def shape_for(tea: Teacher, kind: int, rng: np.random.Generator, held: bool = False) -> np.ndarray:
    """A thing of a kind's features: around the kind's shape, on its training side, or past one
    spread on its held-out side."""
    s = tea.shapes[kind] + rng.normal(0.0, SPREAD, len(SHAPE))
    feature, sign = tea.side[kind]
    d = s[feature] - tea.shapes[kind][feature]
    if held:
        s[feature] = tea.shapes[kind][feature] + sign * (SPREAD + abs(d))
    elif sign * d > 0:
        s[feature] = tea.shapes[kind][feature] - d
    return np.asarray(np.clip(s, 0.0, 1.0))


def dress(tea: Teacher, r: world.Room, rng: np.random.Generator, held: bool = False) -> None:
    """Each thing of a room takes raw readings: a hue of its colour and a shape of its kind."""
    for t in r.things:
        t.hue = hue_for(tea, t.colour, rng, held)  # type: ignore[attr-defined]
        t.shape = shape_for(tea, t.kind, rng, held)  # type: ignore[attr-defined]


def direction(heading: float, point: np.ndarray) -> float:
    """A point's direction from the room's middle, radians from the heading."""
    return float(np.angle(np.exp(1j * (math.atan2(point[1], point[0]) - heading))))


def reports(r: world.Room, heading: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """What the eyes report facing ``heading``: each thing in view, its raw readings."""
    return [(t.shape.copy(), t.place.copy(), float(t.hue))  # type: ignore[attr-defined]
            for t in r.things if world.in_view(heading, t.place)[0]]


# -- the teacher's questions, at the teacher's places


def where(tea: Teacher, lex: world.Lexicon, r: world.Room, kind: int) -> world.Question:
    q = world.where(lex, r, kind)
    things = [t for t in r.things if t.kind == kind]
    if things:
        q.answer = [lex.place[tea.place(direction(r.heading, things[0].place))]]
    return q


def by_place(tea: Teacher, lex: world.Lexicon, r: world.Room, place: int) -> world.Question:
    q = world.by_place(lex, r, place)
    things = [t for t in r.things if tea.place(direction(r.heading, t.place)) == place]
    q.answer = sorted(lex.kind[t.kind] for t in things) or [lex.none]
    q.about = "several" if things else "absent"
    return q


def room_questions(tea: Teacher, lex: world.Lexicon, r: world.Room, rng: np.random.Generator,
                   *, absent: list[int], foils: list[int]) -> list[world.Question]:
    """The protocol 1 chamber's question forms about a room, at the teacher's places."""
    out: list[world.Question] = []
    for t in r.things:
        out += [where(tea, lex, r, t.kind), world.colour_of(lex, r, t.kind)]
    colours = sorted({t.colour for t in r.things})
    out.append(world.by_colour(lex, r, int(rng.choice(colours))))
    out += [by_place(tea, lex, r, p) for p in range(len(world.PLACE_NAMES))]
    gone = [k for k in absent if k not in r.kinds()]
    if gone:
        k = int(rng.choice(gone))
        out += [where(tea, lex, r, k), world.colour_of(lex, r, k)]
    if foils:
        f = int(rng.choice(foils))
        out += [where(tea, lex, r, -(f + 1)), world.colour_of(lex, r, -(f + 1))]
    return out


def script(seed: int, **given: Any) -> list[tuple[Any, ...]]:
    """A founder's life as events, as ``words.script`` gives them, with raw readings and the
    teacher's places; the held-out rooms close the test."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 3]))
    looks = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 78]))
    tea = teacher(seed)
    lex = world.lexicon(seed)
    unseen = world.unseen_colours(seed)
    trained = list(range(world.TRAINED))
    events: list[tuple[Any, ...]] = []

    def new_room(r: world.Room, held: bool = False) -> None:
        dress(tea, r, looks, held)
        events.append(("room", r))

    # 1. naming
    for _ in range(settings["naming_rooms"]):
        r = world.room(rng, trained, int(rng.integers(4, 7)), avoid=unseen)
        new_room(r)
        for _ in range(settings["namings"]):
            heading = float(rng.uniform(-math.pi, math.pi))
            events.append(("turn", heading))
            r.heading = heading
            t = r.things[int(rng.integers(len(r.things)))]
            u = rng.random()
            rates = settings["naming_rates"]
            word = (lex.kind[t.kind] if u < rates[0] else lex.colour[t.colour]
                    if u < rates[0] + rates[1]
                    else lex.place[tea.place(direction(heading, t.place))])
            if settings["joint"] and u < rates[0] + rates[1]:
                # the parrot turns its head to the thing whose kind or colour it is told; its
                # body, and so what lies in front of it and behind it, stays as it is
                events.append(("look", direction(0.0, t.place) + float(looks.uniform(-0.5, 0.5))))
            events.append(("name", t.place.copy(), word, "naming"))

    # 2. practice: kinds 0 to 3 only
    practised = list(range(4))
    for _ in range(settings["practice_rooms"]):
        r = world.room(rng, practised, 3, avoid=unseen)
        new_room(r)
        asked = room_questions(tea, lex, r, rng, absent=practised, foils=list(range(6)))
        for i in rng.permutation(len(asked))[: settings["asked"]]:
            events.append(("practice", asked[int(i)]))

    # 3. the test
    for _ in range(settings["test_rooms"]):
        r = world.room(rng, trained, int(rng.integers(4, 7)), avoid=unseen)
        new_room(r)
        for q in room_questions(tea, lex, r, rng, absent=trained, foils=list(range(6, 12))):
            events.append(("ask", words.gate_of(q), q))
    for _ in range(settings["unseen_rooms"]):
        k = int(rng.choice(trained))
        r = world.room(rng, [j for j in trained if j != k], int(rng.integers(3, 6)),
                       avoid=unseen)
        r.things.append(world.Thing(k, unseen[k], words._free_place(rng, r)))
        new_room(r)
        events += [("ask", "W3_colour", world.colour_of(lex, r, k)),
                   ("ask", "W3_where", where(tea, lex, r, k)),
                   ("ask", "W3_by_colour", world.by_colour(lex, r, unseen[k]))]
    for _ in range(settings["held_rooms"]):
        r = world.room(rng, trained, int(rng.integers(4, 7)), avoid=unseen)
        new_room(r, held=True)
        for t in r.things:
            events += [("ask", "C2_colour", world.colour_of(lex, r, t.kind)),
                       ("ask", "C2_where", where(tea, lex, r, t.kind))]

    # 4. new words
    new = list(range(world.TRAINED, world.KINDS))
    for n in new:
        r = world.room(rng, trained, 3, avoid=unseen)
        r.things.append(world.Thing(n, int(rng.integers(world.COLOURS)), words._free_place(rng, r)))
        new_room(r)
        # the parrot is shown the new thing: it turns to it, and hears its word three times
        events.append(("turn", direction(0.0, r.things[-1].place) + float(
            looks.uniform(-0.7, 0.7))))
        events += [("name", r.things[-1].place.copy(), lex.kind[n], "new") for _ in range(3)]
    for _ in range(settings["new_rooms"]):
        r = world.room(rng, trained + new, int(rng.integers(4, 7)), avoid=unseen)
        if not set(new) & set(r.kinds()):
            r.things.append(world.Thing(int(rng.choice(new)), int(rng.integers(world.COLOURS)),
                                        words._free_place(rng, r)))
        new_room(r)
        for t in r.things:
            if t.kind in new:
                events += [("ask", "W2_where", where(tea, lex, r, t.kind)),
                           ("ask", "W2_colour", world.colour_of(lex, r, t.kind))]
    return events


# -- the brains


class Given:
    """Protocol 1's given concepts on raw readings: the thing's kind (the teacher's own shape
    nearest its features), its hue's band at the unmoved boundaries, the unmoved quarters around
    the body. Nothing is learned."""

    def __init__(self, tea: Teacher):
        from cadence.words import Body

        self.tea, self.body, self.learning = tea, Body(), True
        # the unmoved boundaries: 60-degree bands placed where the teacher's moved ones lie on
        # the mean (the moves of one founder's boundaries, given to no brain)
        offsets = (tea.bounds - np.arange(world.COLOURS) * TAU / world.COLOURS) % TAU
        self.origin = float(np.angle(np.exp(1j * offsets).mean()) % TAU)
        self.log = {"lessons": 0, "splits": 0, "skipped": 0}

    def sizes(self) -> dict[str, int]:
        return {"kind": world.KINDS, "colour": world.COLOURS, "place": len(world.PLACE_NAMES)}

    def percept(self, shape: Any, hue: float) -> tuple[int, int]:
        kind = int(np.argmin(np.linalg.norm(self.tea.shapes - np.asarray(shape), axis=1)))
        colour = int(((float(hue) - self.origin) % TAU) // (TAU / world.COLOURS))
        return kind, colour

    def perceive(self, seen: Any, lit: Any = None) -> list[tuple[int, np.ndarray, dict[str, int]]]:
        out = []
        for shape, place, hue in seen:
            kind, colour = self.percept(shape, hue)
            out.append((kind, np.asarray(place).copy(), {"colour": colour}))
        return out

    def parameters(self) -> int:
        return 0

    def work(self) -> dict[str, int]:
        return {"settles": 0, "macs": 0}


def make_brain(seed: int, brain: str = "learned", **settings: Any) -> tuple[Any, Any]:
    """The founder's brain and its concepts. ``"learned"`` and ``"senses"``:
    ``cadence.words.concepts.LearnedWords`` on ``Concepts`` (24 kind and 12 colour concept
    neurons, the map's units); ``"given"``: protocol 1's ``cadence.words.Words`` on the given
    concepts. ``lexicon_rate`` is the lexicon's rate (0.2)."""
    from cadence.space.map import Map
    from cadence.words import Words
    from cadence.words.concepts import Concepts, LearnedWords

    rate = float(settings.pop("lexicon_rate", 0.2))
    if brain == "given":
        given = Given(teacher(seed))
        room = Map(world.PLANE, world.CELLS, units=world.KINDS,
                   values={"colour": world.COLOURS}, seed=int(seed))
        return Words(room, body=given.body, rate=rate, seed=int(seed)), given
    kinds = int(settings.pop("kinds", 24))
    colours = int(settings.pop("colours", 12))
    if brain == "senses":
        settings = {"vigilance": {"kind": 0.5, "colour": 0.15, "place": 0.15}, **settings,
                    "naming": False}
    room = Map(world.PLANE, world.CELLS, units=kinds, values={"colour": colours},
               seed=int(seed))
    held = Concepts(room, SHAPE, **settings)
    return LearnedWords(room, held, rate=rate, seed=int(seed)), held


def see(brain: Any, held: Any, r: world.Room, gaze: float) -> None:
    """One moment of sight facing ``gaze``: the eyes' raw reports, through the brain's
    concepts."""
    from cadence.words.concepts import LearnedWords

    raw = reports(r, gaze)
    if isinstance(brain, LearnedWords):
        brain.look(raw, world.camera(gaze))
    else:
        brain.look(held.perceive(raw), world.camera(gaze))


def look_around(brain: Any, held: Any, r: world.Room) -> None:
    brain.map.reset()
    for i in range(9):
        heading = r.heading + (i % 8) * math.pi / 4 if i < 8 else r.heading
        brain.body.heading = heading
        see(brain, held, r, heading)


def turn(brain: Any, held: Any, r: world.Room, heading: float) -> None:
    r.heading = heading
    brain.body.heading = heading
    see(brain, held, r, heading)


def look(held: Any, brain: Any, r: world.Room, gaze: float) -> None:
    """The parrot turns its head to ``gaze``: one moment of sight there, its body as it is."""
    see(brain, held, r, gaze)


def ablate(brain: Any) -> Any:
    """W6's twin: the brain's map, body and senses, and its lexicon with every word it heard
    and its synapses at zero."""
    from cadence.words import Lexicon, Words

    arrays = brain.lexicon.arrays()
    arrays["efficacy"] = np.zeros_like(arrays["efficacy"])
    return Words(brain.map, senses=brain.senses, body=brain.body,
                 lexicon=Lexicon.from_arrays(brain.lexicon.settings, arrays))


def held_rate(brain: Any, held: Any, r: world.Room) -> list[bool]:
    """For each thing out of view: whether the map holds its percept (the kind and colour
    concepts its readings settle on) within 0.15 m of its place (the map's own rate)."""
    out = []
    for t in r.things:
        if world.in_view(r.heading, t.place)[0]:
            continue
        kind, colour = held.percept(t.shape, t.hue)  # type: ignore[attr-defined]
        places = brain.map.where_is(kind) if kind is not None else []
        near = [p for p in places if np.linalg.norm(p - t.place) <= 0.15]
        out.append(bool(near) and brain.map.values_at(near[0])["colour"] == colour)
    return out


def life(seed: int, *, brain: str = "learned", record: Any = None, keep: Any = None,
         **given: Any) -> dict[str, Any]:
    """One founder's life on its script; the readings by gate, the probes. ``keep`` (a list)
    takes the brain and its concepts at the end."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    options = {k: v for k, v in given.items() if k not in DEFAULTS}
    events = script(seed, **settings)
    lex = world.lexicon(seed)
    english = lex.english()
    tea = teacher(seed)
    mind, held = make_brain(seed, brain, **options)
    if keep is not None:
        keep += [mind, held]
    readings: dict[str, list[bool]] = {}
    rates: list[bool] = []
    seconds: dict[str, float] = {}
    phase, began = "naming", time.time()
    r: world.Room | None = None
    rated: world.Room | None = None
    ablated: Any = None
    for event in events:
        kind = event[0]
        now = {"room": None, "turn": None, "look": None,
               "name": event[3] if kind == "name" else None,
               "practice": "practice", "ask": "test"}[kind]
        if now is not None and now != phase:
            seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
            phase, began = now, time.time()
        held.learning = phase != "test"
        if kind == "room":
            r = event[1]
            look_around(mind, held, r)
        elif kind == "turn":
            assert r is not None
            turn(mind, held, r, event[1])
        elif kind == "look":
            assert r is not None
            look(held, mind, r, event[1])
        elif kind == "name":
            mind.name(event[2], event[1])
        elif kind == "practice":
            q = event[1]
            mind.practice(q.text, q.answer if len(q.answer) > 1 else q.answer[0])
        else:
            gate, q = event[1], event[2]
            assert r is not None
            if rated is not r and gate[:2] in ("W1", "W4", "W5"):
                rated = r
                rates += held_rate(mind, held, r)
            said = mind.ask(q.text)
            ok = world.right(q, said)
            readings.setdefault(gate, []).append(ok)
            if gate.startswith("W1"):
                ablated = ablated if ablated is not None else ablate(mind)
                readings.setdefault("W6_ablated", []).append(world.right(q, ablated.ask(q.text)))
            if gate.startswith("W1") and q.about == "out_of_view":
                readings.setdefault("W4_out_of_view", []).append(ok)
            if record is not None:
                record.append({"gate": gate, "question": q.text, "answer": q.answer,
                               "said": said, "right": ok,
                               "english": " ".join(english.get(w, w) for w in q.text.split()),
                               "said_english": [english.get(w, w) for w in said]})
    seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
    held.learning = False
    began = time.time()
    probed = probes(seed, mind, held, tea, lex)
    seconds["probes"] = time.time() - began
    out = words.readings_of(seed, settings, readings, rates, mind, seconds)
    out["W4_map_rate"] = float(np.mean(rates)) if rates else float("nan")
    out["C2_room"] = float(np.mean(readings.get("C2_colour", []) + readings.get("C2_where", [])))
    out.update(probed)
    out["brain"] = brain
    out["parameters"] = int(mind.lexicon.connectome.synapses // 2) + held.parameters()
    out["macs"] = int(out["macs"]) + held.work()["macs"]
    out["concepts_used"] = {g: getattr(held, "regions", {}).get(g).used
                            for g in ("kind", "colour", "place")} if brain != "given" else {}
    out["concepts_log"] = dict(held.log)
    return out


# -- the probes


def _said(mind: Any, group: str, concepts: list[int | None]) -> list[str | None]:
    return [None if c is None else mind.lexicon.say((group, int(c))) for c in concepts]


def probes(seed: int, mind: Any, held: Any, tea: Teacher, lex: world.Lexicon) -> dict[str, Any]:
    """C1 (the brain's words for fresh readings against the teacher's), C2 (held-out readings)
    and C3 (where the brain's words change near each of the teacher's boundaries)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 79]))
    out: dict[str, Any] = {}

    def colours(hues: np.ndarray) -> list[str | None]:
        return _said(mind, "colour", [held.percept(tea.shapes[0], h)[1] for h in hues])

    def kinds(shapes: list[np.ndarray]) -> list[str | None]:
        return _said(mind, "kind", [held.percept(s, 0.0)[0] for s in shapes])

    def places(angles: np.ndarray) -> list[str | None]:
        mind.body.heading = 0.0
        points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return _said(mind, "place", [mind.body.place(p) for p in points])

    # C1: hues at every degree out of the gaps, fresh things of each trained kind, directions
    hues = np.radians(np.arange(0.0, 360.0, 1.0))
    out_of_gaps = np.array([not any(_within(h, *tea.gap(c)) for c in range(world.COLOURS))
                            for h in hues])
    said = colours(hues[out_of_gaps])
    out["C1_colour"] = float(np.mean([w == lex.colour[tea.colour(h)]
                                      for w, h in zip(said, hues[out_of_gaps], strict=True)]))
    things = [(k, shape_for(tea, k, rng)) for k in range(world.TRAINED) for _ in range(20)]
    said = kinds([s for _, s in things])
    out["C1_kind"] = float(np.mean([w == lex.kind[k] for w, (k, _) in zip(said, things,
                                                                            strict=True)]))
    angles = np.radians(np.arange(-180.0, 180.0, 1.0))
    said = places(angles)
    out["C1_place"] = float(np.mean([w == lex.place[tea.place(a)] for w, a in zip(said, angles,
                                                                                    strict=True)]))
    # C2: the held-out hues and shapes
    gap_hues = np.array([hue_for(tea, c, rng, held=True) for c in range(world.COLOURS)
                         for _ in range(10)])
    said = colours(gap_hues)
    out["C2_hue"] = float(np.mean([w == lex.colour[tea.colour(h)]
                                   for w, h in zip(said, gap_hues, strict=True)]))
    things = [(k, shape_for(tea, k, rng, held=True)) for k in range(world.TRAINED)
              for _ in range(10)]
    said = kinds([s for _, s in things])
    out["C2_shape"] = float(np.mean([w == lex.kind[k] for w, (k, _) in zip(said, things,
                                                                             strict=True)]))
    # C3: each boundary's place in the brain's words, a half degree at a time within 30 degrees
    errors, moved, found = [], [], []
    canon = (tea.bounds - np.arange(world.COLOURS) * TAU / world.COLOURS) % TAU
    origin = float(np.angle(np.exp(1j * canon).mean()))
    for c in range(world.COLOURS):
        b = float(tea.bounds[c])
        sweep = b + np.radians(np.arange(-30.0, 30.25, 0.5))
        said = colours(sweep % TAU)
        before, after = lex.colour[(c - 1) % world.COLOURS], lex.colour[c]
        at = _switch(sweep, said, before, after)
        errors.append(math.degrees(at - b))
        unmoved = origin + c * TAU / world.COLOURS
        moved.append(math.degrees(float(np.angle(np.exp(1j * (b - unmoved))))))
        found.append(math.degrees(float(np.angle(np.exp(1j * (at - unmoved))))))
    out["C3_colour_error"] = errors
    out["C3_colour_moved"] = moved
    out["C3_colour_found"] = found
    p_errors, p_moved, p_found = [], [], []
    # each turn's places below and above it (counter-clockwise): front/right at -45 degrees has
    # the right below it and the front above it, and so on round the body
    sides = [(1, 0), (2, 1), (3, 2), (0, 3)]
    unmoved_turns = np.radians([-45.0, -135.0, 135.0, 45.0])
    for i, b in enumerate(tea.turns):
        sweep = float(b) + np.radians(np.arange(-30.0, 30.25, 0.5))
        said = places(sweep)
        at = _switch(sweep, said, lex.place[sides[i][0]], lex.place[sides[i][1]])
        p_errors.append(math.degrees(at - float(b)))
        p_moved.append(math.degrees(float(b - unmoved_turns[i])))
        p_found.append(math.degrees(at - float(unmoved_turns[i])))
    out["C3_place_error"] = p_errors
    out["C3_place_moved"] = p_moved
    out["C3_place_found"] = p_found
    return out


def _within(h: float, low: float, high: float) -> bool:
    return low <= h <= high or low <= h + TAU <= high or low <= h - TAU <= high


def _switch(sweep: np.ndarray, said: list[str | None], before: str, after: str) -> float:
    """The point of a sweep where the words said fit best a change from ``before`` to
    ``after``: the split that the most words agree with."""
    a = np.array([w == before for w in said], dtype=float)
    b = np.array([w == after for w in said], dtype=float)
    agree = [a[:i].sum() + b[i:].sum() for i in range(len(said) + 1)]
    best = np.flatnonzero(np.asarray(agree) == max(agree))
    i = int(best[len(best) // 2])
    if i == 0:
        return float(sweep[0])
    if i == len(said):
        return float(sweep[-1])
    return float((sweep[i - 1] + sweep[i]) / 2)


# -- the declared run


ARMS = ("learned", "senses", "given")
CONTROLS = ("lstm_online", "lstm_offline", "transformer_online", "transformer_offline")
PARAMETERS = 56864  # the learned brain's parameters at the defaults

# the declared gates: (reading, threshold); a gate holds where the mean over the founders and at
# least ten of twelve founders reach it
GATES = {
    "C1": [("C1_colour", 0.90), ("C1_kind", 0.95), ("C1_place", 0.90)],
    "C2": [("C2_hue", 0.90), ("C2_shape", 0.90), ("C2_room", 0.90)],
    "W1": [("W1", 0.95), ("W1_named_only", 0.95)],
    "W2": [("W2", 0.90)],
    "W3": [("W3", 0.90)],
    "W4": [("W4_never_seen", 0.95), ("W4_never_taught", 0.95)],
    "W5": [("W5_by_colour", 0.90), ("W5_by_place", 0.90)],
}
READINGS = ["C1_colour", "C1_kind", "C1_place", "C2_hue", "C2_shape", "C2_room", "W1",
            "W1_named_only", "W2", "W3", "W4_out_of_view", "W4_map_rate", "W4_never_seen",
            "W4_never_taught", "W5_by_colour", "W5_by_place", "W6_ablated"]


def jobs(out: Path, first: int, count: int, arms: Any = ARMS + CONTROLS,
         settings: dict[str, Any] | None = None) -> list[tuple[Path, list[str]]]:
    python = sys.executable
    todo = []
    for seed in range(first, first + count):
        for arm in arms:
            if arm in CONTROLS:
                model, mode = arm.split("_")
                todo.append((out / f"{arm}_{seed}.json",
                             [python, str(HERE / "concepts_controls.py"), str(seed), model, mode,
                              str(PARAMETERS)]))
                continue
            told = json.dumps({"brain": arm, **(settings or {})})
            todo.append((out / f"{arm}_{seed}.json",
                         [python, str(HERE / "concepts.py"), str(seed), told]))
    return todo


def run(out: Path, first: int, count: int, workers: int, arms: Any = ARMS + CONTROLS,
        settings: dict[str, Any] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}

    def one(job: tuple[Path, list[str]]) -> str:
        path, command = job
        if path.exists():
            return f"kept {path.name}"
        done = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
        if done.returncode:
            path.with_suffix(".err").write_text(done.stderr)
            return f"failed {path.name}"
        part = path.with_suffix(".part")
        part.write_text(done.stdout)
        part.rename(path)
        return f"wrote {path.name}"

    with ThreadPoolExecutor(workers) as pool:
        for line in pool.map(one, jobs(out, first, count, arms, settings)):
            print(line, flush=True)


def boundaries(lives: list[dict[str, Any]], sense: str) -> dict[str, float]:
    """C3 over every founder's boundaries of a sense: the slope of where the brain's words change
    on where the teacher's do (both from the unmoved place), the mean distance between the two,
    and the share of the boundaries the teacher moved by 5 degrees or more on which the brain's
    lies nearer its own teacher's than the unmoved place."""
    moved = np.concatenate([life[f"C3_{sense}_moved"] for life in lives])
    found = np.concatenate([life[f"C3_{sense}_found"] for life in lives])
    error = np.abs(np.concatenate([life[f"C3_{sense}_error"] for life in lives]))
    far = np.abs(moved) >= 5.0
    return {"slope": float(np.polyfit(moved, found, 1)[0]), "error": float(error.mean()),
            "nearer": float(np.mean(np.abs(found - moved)[far] < np.abs(found)[far])),
            "boundaries": int(len(moved))}


def read(out: Path) -> dict[str, Any]:
    """The table of every arm and each gate's verdict on the learned brain."""
    arms: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out.glob("*.json")):
        arms.setdefault(path.stem.rsplit("_", 1)[0], []).append(json.loads(path.read_text()))
    head = ["arm", "founders", "parameters", "multiply-accumulates", "seconds"] + READINGS
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for arm, lives in arms.items():
        row = [arm, str(len(lives)), str(lives[0].get("parameters", "")),
               f"{np.mean([life.get('macs', 0) for life in lives]):.3g}",
               f"{np.mean([sum(life['seconds'].values()) for life in lives]):.0f}"]
        for name in READINGS:
            values = [life[name] for life in lives if name in life and life[name] == life[name]]
            row.append(f"{np.mean(values):.3f}" if values else "")
        print("| " + " | ".join(row) + " |")
    verdicts: dict[str, Any] = {}
    brain = arms.get("learned", [])
    for gate, needs in GATES.items():
        held, parts = True, []
        for name, threshold in needs:
            values = np.array([life[name] for life in brain])
            reach = int((values >= threshold).sum())
            ok = bool(len(values)) and values.mean() >= threshold and reach >= min(
                10, len(values))
            held &= ok
            parts.append(f"{name} mean {values.mean():.3f}, {reach} of {len(values)} at "
                         f"{threshold}")
        verdicts[gate] = held
        print(f"{gate}: {'holds' if held else 'fails'} ({'; '.join(parts)})")
    for arm in ("learned", "senses", "given"):
        for sense in ("colour", "place"):
            if arms.get(arm):
                c3 = boundaries(arms[arm], sense)
                print(f"C3 {arm} {sense}: slope {c3['slope']:.3f}, mean error "
                      f"{c3['error']:.2f} degrees, nearer its own teacher {c3['nearer']:.3f} "
                      f"({c3['boundaries']} boundaries)")
    if brain:
        c3 = {s: boundaries(brain, s) for s in ("colour", "place")}
        verdicts["C3"] = all(c3[s]["slope"] >= 0.8 and c3[s]["error"] <= 4.0 for s in c3) and (
            c3["colour"]["nearer"] >= 0.8)
        print(f"C3: {'holds' if verdicts['C3'] else 'fails'}")
        gaps = [abs(life["W4_out_of_view"] - life["W4_map_rate"]) for life in brain]
        verdicts["W4_out_of_view"] = max(gaps) <= 0.02
        print(f"W4 out of view: {'holds' if verdicts['W4_out_of_view'] else 'fails'} (largest "
              f"gap from the map's own rate {max(gaps):.3f})")
        ablated = [life["W6_ablated"] for life in brain]
        verdicts["W6"] = max(ablated) <= 0.05
        print(f"W6 ablation: the lexicon's synapses at zero answer W1 at {max(ablated):.3f} at "
              "most")
    return verdicts


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 15,
            sys.argv[6].split(",") if len(sys.argv) > 6 else ARMS + CONTROLS,
            json.loads(sys.argv[7]) if len(sys.argv) > 7 else None)
    elif sys.argv[1] == "read":
        read(Path(sys.argv[2]))
    else:
        told = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        print(json.dumps(life(int(sys.argv[1]), **told), default=float))
