"""A founder's life in the words chamber, and its readings (cadence#326).

The brain is ``cadence.words.Words``: a map of things with each thing's colour on its own sheet
(``Map(values={"colour": 6})``), the robot's body and a lexicon learned from naming. One life:

1. **Naming.** In each of ``naming_rooms`` rooms of trained kinds (no kind in its unseen colour)
   the robot looks around (eight moments, an eighth of a turn apart), and the teacher names
   ``namings`` things: each time the robot turns to a drawn heading, attends to a thing (in view
   or held out of it) and hears one word, its kind's, its colour's or its place's (at the rates
   ``world.NAMING``).
2. **Practice.** In each of ``practice_rooms`` rooms of three things of kinds 0 to 3 the teacher
   asks ``asked`` questions and says the answer: where a thing is, its colour, what has a
   colour, what is at a place around the robot, a kind not in the room, a foil.
3. **The test.** In ``test_rooms`` fresh rooms of trained kinds, and ``unseen_rooms`` rooms with
   a thing in its kind's unseen colour, every question form is asked; nothing is taught.
4. **New words.** Each new kind is shown in a room and named three times; then ``new_rooms``
   fresh rooms with new kinds are asked where they are and their colour.

Kinds 4 to 7 are named and never asked about in practice: their questions read words learned
from naming alone.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import world  # noqa: E402

DEFAULTS = {"naming_rooms": 40, "namings": 15, "practice_rooms": 24, "asked": 10,
            "test_rooms": 30, "unseen_rooms": 16, "new_rooms": 16}


def make_brain(seed: int, **settings: Any) -> Any:
    """The founder's brain: a map of the chamber's plane with one unit a kind and the colours
    on its sheet, the robot at the middle, and a fresh lexicon."""
    from cadence.space.map import Map
    from cadence.words import Words

    held = Map(world.PLANE, world.CELLS, units=world.KINDS, values={"colour": world.COLOURS},
               seed=int(seed))
    return Words(held, **settings)


def look_around(brain: Any, r: world.Room) -> None:
    """A new room: the map is emptied, the robot looks around in eight moments and turns back
    to the room's heading."""
    brain.map.reset()
    for i in range(9):
        heading = r.heading + (i % 8) * math.pi / 4 if i < 8 else r.heading
        brain.body.heading = heading
        brain.map.moment(seen=world.seen(r, heading), lit=world.camera(heading))


def turn(brain: Any, r: world.Room, heading: float) -> None:
    """The robot turns to ``heading``: one moment of sight there."""
    r.heading = heading
    brain.body.heading = heading
    brain.map.moment(seen=world.seen(r, heading), lit=world.camera(heading))


def naming_word(lex: world.Lexicon, r: world.Room, t: world.Thing,
                rng: np.random.Generator) -> str:
    u = rng.random()
    if u < world.NAMING[0]:
        return lex.kind[t.kind]
    if u < world.NAMING[0] + world.NAMING[1]:
        return lex.colour[t.colour]
    return lex.place[world.place_of(r.heading, t.place)]


def room_questions(lex: world.Lexicon, r: world.Room, rng: np.random.Generator, *,
                   absent: list[int], foils: list[int]) -> list[world.Question]:
    """Every question form about a room: where each thing is and its colour, what has a colour
    held in the room, what is at each place around the robot, a kind not in the room, a foil."""
    out: list[world.Question] = []
    for t in r.things:
        out += [world.where(lex, r, t.kind), world.colour_of(lex, r, t.kind)]
    colours = sorted({t.colour for t in r.things})
    out.append(world.by_colour(lex, r, int(rng.choice(colours))))
    out += [world.by_place(lex, r, p) for p in range(len(world.PLACE_NAMES))]
    gone = [k for k in absent if k not in r.kinds()]
    if gone:
        k = int(rng.choice(gone))
        out += [world.where(lex, r, k), world.colour_of(lex, r, k)]
    if foils:
        f = int(rng.choice(foils))
        out += [world.where(lex, r, -(f + 1)), world.colour_of(lex, r, -(f + 1))]
    return out


def held_rate(brain: Any, r: world.Room) -> list[bool]:
    """For each thing out of view: whether the map holds its kind within 0.15 m of its place,
    with its colour (the map's own rate)."""
    out = []
    for t in r.things:
        if world.in_view(r.heading, t.place)[0]:
            continue
        places = brain.map.where_is(t.kind)
        near = [p for p in places if np.linalg.norm(p - t.place) <= 0.15]
        out.append(bool(near) and brain.map.values_at(near[0])["colour"] == t.colour)
    return out


def script(seed: int, **given: Any) -> list[tuple[Any, ...]]:
    """A founder's life as events, the same for the brain and the controls:

    - ``("room", room)``: a new room; the robot looks around and faces ``room.heading``;
    - ``("turn", heading)``: the robot turns to ``heading`` (one moment of sight);
    - ``("name", place, word, phase)``: the robot attends to the thing at ``place`` and hears
      ``word`` (``phase`` is ``"naming"`` or ``"new"``);
    - ``("practice", question)``: a question and the teacher's answer;
    - ``("ask", gate, question)``: a question of the test, answered and never taught."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 3]))
    lex = world.lexicon(seed)
    unseen = world.unseen_colours(seed)
    trained = list(range(world.TRAINED))
    events: list[tuple[Any, ...]] = []

    # 1. naming
    for _ in range(settings["naming_rooms"]):
        r = world.room(rng, trained, int(rng.integers(4, 7)), avoid=unseen)
        events.append(("room", r))
        for _ in range(settings["namings"]):
            heading = float(rng.uniform(-math.pi, math.pi))
            events.append(("turn", heading))
            r.heading = heading
            t = r.things[int(rng.integers(len(r.things)))]
            events.append(("name", t.place.copy(), naming_word(lex, r, t, rng), "naming"))

    # 2. practice: kinds 0 to 3 only
    practised = list(range(4))
    for _ in range(settings["practice_rooms"]):
        r = world.room(rng, practised, 3, avoid=unseen)
        events.append(("room", r))
        asked = room_questions(lex, r, rng, absent=practised, foils=list(range(6)))
        for i in rng.permutation(len(asked))[: settings["asked"]]:
            events.append(("practice", asked[int(i)]))

    # 3. the test: fresh rooms of trained kinds, and rooms with a thing in its unseen colour
    for _ in range(settings["test_rooms"]):
        r = world.room(rng, trained, int(rng.integers(4, 7)), avoid=unseen)
        events.append(("room", r))
        for q in room_questions(lex, r, rng, absent=trained, foils=list(range(6, 12))):
            events.append(("ask", gate_of(q), q))
    for _ in range(settings["unseen_rooms"]):
        k = int(rng.choice(trained))
        r = world.room(rng, [j for j in trained if j != k], int(rng.integers(3, 6)),
                       avoid=unseen)
        r.things.append(world.Thing(k, unseen[k], _free_place(rng, r)))
        events.append(("room", r))
        events += [("ask", "W3_colour", world.colour_of(lex, r, k)),
                   ("ask", "W3_where", world.where(lex, r, k)),
                   ("ask", "W3_by_colour", world.by_colour(lex, r, unseen[k]))]

    # 4. new words: each new kind shown and named three times, then asked in fresh rooms
    new = list(range(world.TRAINED, world.KINDS))
    for n in new:
        r = world.room(rng, trained, 3, avoid=unseen)
        r.things.append(world.Thing(n, int(rng.integers(world.COLOURS)), _free_place(rng, r)))
        events.append(("room", r))
        events += [("name", r.things[-1].place.copy(), lex.kind[n], "new") for _ in range(3)]
    for _ in range(settings["new_rooms"]):
        r = world.room(rng, trained + new, int(rng.integers(4, 7)), avoid=unseen)
        if not set(new) & set(r.kinds()):
            r.things.append(world.Thing(int(rng.choice(new)), int(rng.integers(world.COLOURS)),
                                        _free_place(rng, r)))
        events.append(("room", r))
        for t in r.things:
            if t.kind in new:
                events += [("ask", "W2_where", world.where(lex, r, t.kind)),
                           ("ask", "W2_colour", world.colour_of(lex, r, t.kind))]
    return events


def gate_of(q: world.Question) -> str:
    """The gate a question of a test room reads."""
    if q.form in ("where", "colour") and q.about in ("in_view", "out_of_view"):
        return "W1_named_only" if q.kind is not None and q.kind >= 4 else "W1_practised"
    if q.about == "foil":
        return "W4_never_taught"
    if q.about == "absent" and q.form in ("where", "colour"):
        return "W4_never_seen"
    return "W5_by_colour" if q.form == "by_colour" else "W5_by_place"


def life(seed: int, *, record: Any = None, brain: Any = None, **given: Any) -> dict[str, Any]:
    """One founder's life on its script; the readings by gate. ``record`` (a list) takes every
    question asked in the test with what was said."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    brain_settings = {k: v for k, v in given.items() if k not in DEFAULTS}
    events = script(seed, **settings)
    lex = world.lexicon(seed)
    english = lex.english()
    brain = make_brain(seed, **brain_settings) if brain is None else brain
    readings: dict[str, list[bool]] = {}
    held: list[bool] = []
    seconds: dict[str, float] = {}
    phase, began = "naming", time.time()
    r: world.Room | None = None
    held_for: world.Room | None = None  # the room whose things out of view were counted
    ablated: Any = None  # W6: the same map and body, the lexicon's synapses at zero
    for event in events:
        kind = event[0]
        now = {"room": None, "turn": None, "name": event[3] if kind == "name" else None,
               "practice": "practice", "ask": "test"}[kind]
        if now is not None and now != phase:
            seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
            phase, began = now, time.time()
        if kind == "room":
            r = event[1]
            look_around(brain, r)
        elif kind == "turn":
            assert r is not None
            turn(brain, r, event[1])
        elif kind == "name":
            brain.name(event[2], event[1])
        elif kind == "practice":
            q = event[1]
            brain.practice(q.text, q.answer if len(q.answer) > 1 else q.answer[0])
        else:
            gate, q = event[1], event[2]
            assert r is not None
            if held_for is not r and gate[:2] in ("W1", "W4", "W5"):
                held_for = r
                held += held_rate(brain, r)
            said = brain.ask(q.text)
            ok = world.right(q, said)
            readings.setdefault(gate, []).append(ok)
            if gate.startswith("W1"):
                ablated = ablated if ablated is not None else ablate(brain)
                readings.setdefault("W6_ablated", []).append(world.right(q, ablated.ask(q.text)))
            if gate.startswith("W1") and q.about == "out_of_view":
                readings.setdefault("W4_out_of_view", []).append(ok)
            if record is not None:
                record.append({"gate": gate, "question": q.text, "form": q.form,
                               "about": q.about, "answer": q.answer, "said": said, "right": ok,
                               "english": " ".join(english.get(w, w) for w in q.text.split()),
                               "said_english": [english.get(w, w) for w in said]})
    seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
    return readings_of(seed, settings, readings, held, brain, seconds)


def readings_of(seed: int, settings: dict[str, Any], readings: dict[str, list[bool]],
                held: list[bool], brain: Any, seconds: dict[str, float]) -> dict[str, Any]:
    """A life's readings: each gate's share right and its count, the gates' pools, the map's
    own rate out of view, the brain's parameters and work."""
    out: dict[str, Any] = {"seed": int(seed), **settings}
    for gate, oks in sorted(readings.items()):
        out[gate] = float(np.mean(oks)) if oks else float("nan")
        out[f"{gate}_n"] = len(oks)

    def pool(*names: str) -> float:
        oks = [ok for n in names for ok in readings.get(n, [])]
        return float(np.mean(oks)) if oks else float("nan")

    out["W1"] = pool("W1_practised", "W1_named_only")
    out["W2"] = pool("W2_where", "W2_colour")
    out["W3"] = pool("W3_colour", "W3_by_colour")
    out["W4_map_rate"] = float(np.mean(held)) if held else float("nan")
    out["W4_map_n"] = len(held)
    out["W5"] = pool("W5_by_colour", "W5_by_place")
    if brain is not None:
        out["parameters"] = brain_parameters(brain)
        out["work"] = dict(brain.map.work)
        out["lessons"] = brain.lexicon.lessons
        out["words"] = len(brain.lexicon.forms)
        synapses = int(brain.map.connectome.synapses)
        out["macs"] = int(brain.lexicon.work["macs"] + synapses * (
            brain.map.work["moments"] + 2 * brain.map.work["questions"]))
    out["seconds"] = seconds
    return out


def _free_place(rng: np.random.Generator, r: world.Room) -> np.ndarray:
    for _ in range(1000):
        radius, a = rng.uniform(0.6, 1.8), rng.uniform(-math.pi, math.pi)
        p = np.array([radius * math.cos(a), radius * math.sin(a)])
        if all(np.linalg.norm(p - t.place) >= 0.45 for t in r.things):
            return p
    raise RuntimeError("no free place in the room")


def ablate(brain: Any) -> Any:
    """W6's twin: the brain's map and body, and its lexicon with every word it heard and its
    synapses at zero."""
    from cadence.words import Lexicon, Words

    arrays = brain.lexicon.arrays()
    arrays["efficacy"] = np.zeros_like(arrays["efficacy"])
    return Words(brain.map, lexicon=Lexicon.from_arrays(brain.lexicon.settings, arrays),
                 body=brain.body)


def brain_parameters(brain: Any) -> int:
    """The brain's learned parameters: one efficacy a reciprocal pair of the lexicon (the
    map's graph is wiring with no taught synapse)."""
    return int(brain.lexicon.connectome.synapses // 2)


if __name__ == "__main__":
    import json

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    told = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(json.dumps(life(seed, **told), indent=1, default=float))
