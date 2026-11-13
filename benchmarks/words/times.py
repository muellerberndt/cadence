"""The words chamber's rung of time (cadence#326, W8 to W10): "where was the ball?", "where is
the ball?" and "where will the ball be?".

The brain is ``cadence.words.episodes.Times``: the brain of ``words.py`` with its records (the
places things left, keyed by the thing and the time), its forecast and a lexicon whose concepts
add the three sources. The time words ("was", "will") are nonce words drawn for each founder
beside the founder's other words; "is" is the founder's word of protocol 1. One life:

1. **Naming**, as in protocol 1: ``naming_rooms`` rooms of the 8 trained kinds, ``namings``
   namings a room (a kind's word, a colour's or a place's).
2. **Practice**: ``practice_rooms`` rooms of 4 to 6 trained things. The parrot looks around; the
   teacher carries ``carries`` things each to a new place (at least 0.8 m from where it was and
   in another place around the parrot), and the parrot looks around ``looks`` times. The teacher
   asks where each carried thing was and where it is, where an unmoved thing is and its colour,
   and says each answer.
3. **The test**: ``test_rooms`` fresh rooms, carried as in practice, nothing taught: where each
   carried thing was and where it is. Each past question is asked again with the records
   barred.

A question about time names the thing by its kind's word: ``where <was|is> the <kind>``.

    python times.py <founder> ['{"test_rooms": 4}']    one brain's life, its readings
    python times.py run <out> <first founder> <founders> [jobs]
    python times.py read <out>

``run`` writes one JSON file a life (``brain_<seed>.json``, ``<model>_<mode>_<seed>.json`` for
the controls, at the brain's parameter count); ``read`` prints the table and each gate's verdict.
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

import words  # noqa: E402
import world  # noqa: E402

DEFAULTS = {"naming_rooms": 40, "namings": 15, "practice_rooms": 40, "test_rooms": 30,
            "carries": 2, "looks": 2, "watch_rooms": 0, "rollers": 2, "watch": 40,
            "roll_rooms": 0, "roll_tests": 0, "ahead": 10, "physics": False, "rest": 0}
APART = 0.8  # a carried thing goes at least this far from where it was (metres)
HELD = 0.15  # the map holds a thing where it holds its kind within this distance of its place
TIMES = ("was", "is", "will")  # the forms of the questions about time, in the sources' order
EDGE = math.radians(10.0)  # a rolled thing's places lie this far inside their places' edges


def edge(heading: float, point: np.ndarray) -> float:
    """How far a point's direction from the parrot lies from the nearest edge between two
    places around it (radians)."""
    a = math.atan2(point[1], point[0]) - heading
    return abs((a % (math.pi / 2)) - math.pi / 4)


def time_words(seed: int) -> dict[str, str]:
    """The founder's words for time: "was" and "will" drawn as nonce words apart from the
    founder's other words, and "is", the founder's word of protocol 1."""
    lex = world.lexicon(seed)
    taken = set(lex.kind + lex.colour + lex.place + list(lex.question.values()) + [lex.none]
                + lex.foils)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 11]))
    drawn: list[str] = []
    while len(drawn) < 2:
        form = "".join(str(rng.choice(list(world.CONSONANTS))) + str(rng.choice(
            list(world.VOWELS))) for _ in range(int(rng.integers(2, 4))))
        if form not in taken and form not in drawn:
            drawn.append(form)
    return {"was": drawn[0], "is": lex.question["is"], "will": drawn[1]}


def make_brain(seed: int, *, physics: Any = False, ahead: int = 10, **settings: Any) -> Any:
    """The founder's brain in time: the map of protocol 1 (one unit a kind, the colours on its
    sheet), with learned physics where ``physics`` (True, or the physics' settings), and a
    ``Times`` brain that forecasts ``ahead`` moments on."""
    from cadence.space.map import Map
    from cadence.words.episodes import Times

    held = Map(world.PLANE, world.CELLS, units=world.KINDS, values={"colour": world.COLOURS},
               seed=int(seed), physics=(dict(physics) if isinstance(physics, dict)
                                        else True if physics else None))
    return Times(held, seed=int(seed), ahead=int(ahead), **settings)


def look(brain: Any, r: world.Room) -> None:
    """The parrot looks around in eight moments and turns back to the room's heading."""
    for i in range(9):
        heading = r.heading + (i % 8) * math.pi / 4 if i < 8 else r.heading
        brain.body.heading = heading
        brain.moment(seen=world.seen(r, heading), lit=world.camera(heading))


def turn(brain: Any, r: world.Room, heading: float) -> None:
    r.heading = heading
    brain.body.heading = heading
    brain.moment(seen=world.seen(r, heading), lit=world.camera(heading))


def carry_to(rng: np.random.Generator, heading: float, at: dict[int, np.ndarray],
             kind: int) -> np.ndarray:
    """A new place for the carried thing of ``kind`` (``at`` holds every thing's place): free of
    the others, at least ``APART`` from where it was, and in another place around the parrot
    facing ``heading``."""
    was = at[kind]
    for _ in range(10000):
        radius, a = rng.uniform(0.6, 1.8), rng.uniform(-math.pi, math.pi)
        p = np.array([radius * math.cos(a), radius * math.sin(a)])
        if (np.linalg.norm(p - was) >= APART
                and world.place_of(heading, p) != world.place_of(heading, was)
                and all(np.linalg.norm(p - q) >= 0.45 for k, q in at.items() if k != kind)):
            return p
    raise RuntimeError("no place to carry the thing to")


def carry(r: world.Room, kind: int, place: np.ndarray) -> None:
    """The teacher carries the room's thing of ``kind`` to ``place``."""
    next(t for t in r.things if t.kind == kind).place = np.asarray(place, dtype=np.float64).copy()


def question(lex: world.Lexicon, said: dict[str, str], heading: float, kind: int, when: str,
             place: np.ndarray) -> world.Question:
    """``where <when> the <kind>``, answered by the place around the parrot facing ``heading``
    of ``place``."""
    q = lex.question
    text = f"{q['where']} {said[when]} {q['the']} {lex.kind[kind]}"
    return world.Question(text, when, [lex.place[world.place_of(heading, place)]], "moved",
                          kind)


def roll_path(rng: np.random.Generator, at: dict[int, np.ndarray], kind: int, moments: Any,
              ahead: int, *, heading: float | None = None,
              distinct: bool = False) -> tuple[np.ndarray, int, float] | None:
    """A roll of the thing of ``kind`` (``at`` holds every thing's place): a velocity of .03 to
    .06 m a moment, the moments it rolls while the parrot watches (``moments``: a number, or a
    range drawn from) and the heading the parrot watches from (``heading``, else drawn within
    .5 radians of the thing). The path to ``ahead`` moments after stays .15 m inside the plane,
    .4 m from the parrot and .35 m from every other thing; with ``distinct`` the places around
    the parrot where it began, where it is when the watch ends and where it is ``ahead`` moments
    later are three, each ``EDGE`` or more from the edges of its place. None where no roll is
    found."""
    start = at[kind]
    for _ in range(4000):
        speed, a = rng.uniform(0.03, 0.06), rng.uniform(-math.pi, math.pi)
        v = speed * np.array([math.cos(a), math.sin(a)])
        m = int(moments) if isinstance(moments, int) else int(rng.integers(*moments))
        h = (math.atan2(start[1], start[0]) + rng.uniform(-0.5, 0.5) if heading is None
             else float(heading))
        path = start + np.arange(m + ahead + 1)[:, None] * v
        if np.abs(path).max() > 1.85 or np.linalg.norm(path, axis=1).min() < 0.4:
            continue
        if any(np.linalg.norm(path - q, axis=1).min() < 0.35 for k, q in at.items()
               if k != kind):
            continue
        three = (start, path[m], path[m + ahead])
        if distinct and (len({world.place_of(h, p) for p in three}) < 3
                         or min(edge(h, p) for p in three) < EDGE):
            continue
        return v, m, float(h)
    return None


def roll(r: world.Room, kind: int, velocity: np.ndarray) -> None:
    """The room's thing of ``kind`` rolls at ``velocity`` (metres a moment) from now on."""
    r.extra.setdefault("rolling", {})[int(kind)] = np.asarray(velocity, dtype=np.float64).copy()


def watch(brain: Any, r: world.Room, heading: float, moments: int) -> None:
    """The parrot faces ``heading`` for ``moments`` moments; each moment every rolling thing
    moves on and the parrot sees what is in view."""
    r.heading = heading
    brain.body.heading = heading
    for _ in range(int(moments)):
        for t in r.things:
            if t.kind in r.extra.get("rolling", {}):
                t.place = t.place + r.extra["rolling"][t.kind]
        brain.moment(seen=world.seen(r, heading), lit=world.camera(heading))


def script(seed: int, **given: Any) -> list[tuple[Any, ...]]:
    """A founder's life as events:

    - ``("room", room)``: a new room; the parrot looks around and faces ``room.heading``;
    - ``("turn", heading)``; ``("name", place, word, phase)``: as in ``words.script``;
    - ``("carry", kind, place)``: the teacher carries the thing of ``kind`` to ``place``;
    - ``("look",)``: the parrot looks around;
    - ``("practice", question)``; ``("ask", gate, question, extra)``: a test question, with
      ``extra`` holding the carried thing's places (``was``, ``now``)."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 12]))
    lex = world.lexicon(seed)
    said = time_words(seed)
    trained = list(range(world.TRAINED))
    events: list[tuple[Any, ...]] = []

    for _ in range(settings["naming_rooms"]):
        r = world.room(rng, trained, int(rng.integers(4, 7)))
        events.append(("room", r))
        for _ in range(settings["namings"]):
            heading = float(rng.uniform(-math.pi, math.pi))
            events.append(("turn", heading))
            r.heading = heading
            t = r.things[int(rng.integers(len(r.things)))]
            events.append(("name", t.place.copy(), words.naming_word(lex, r, t, rng), "naming"))

    for _ in range(settings["watch_rooms"]):  # things roll while the parrot watches
        r = world.room(rng, trained, int(rng.integers(4, 7)))
        events.append(("room", r))
        at = {t.kind: t.place.copy() for t in r.things}
        first = r.things[int(rng.integers(len(r.things)))].place
        heading = math.atan2(first[1], first[0]) + float(rng.uniform(-0.5, 0.5))
        seen = [t.kind for t in r.things if world.in_view(heading, t.place)[0]]
        for kind in [seen[int(i)] for i in rng.permutation(len(seen))][: settings["rollers"]]:
            path = roll_path(rng, at, kind, settings["watch"], 0, heading=heading)
            if path is not None:
                events.append(("roll", kind, path[0]))
        if settings["rest"]:  # the parrot watches the things at rest before they roll
            events.insert(len(events) - sum(e[0] == "roll" for e in events[-settings["rollers"]:]),
                          ("watch", heading, settings["rest"]))
        events.append(("watch", heading, settings["watch"]))

    def rolled_room() -> tuple[world.Room, dict[str, Any], float] | None:
        """A room one of whose things rolls while the parrot watches: the thing's ``kind``, its
        place where it began (``was``), when the watch ends (``now``) and ``ahead`` moments
        later (``will``), three places around the parrot; and the heading it watches from."""
        r = world.room(rng, trained, int(rng.integers(4, 7)))
        at = {t.kind: t.place.copy() for t in r.things}
        for i in rng.permutation(len(r.things)):
            kind = r.things[int(i)].kind
            path = roll_path(rng, at, kind, (16, 29), settings["ahead"], distinct=True)
            if path is None:
                continue
            v, m, heading = path
            events.extend([("room", r)] + ([("watch", heading, settings["rest"])]
                                           if settings["rest"] else [])
                          + [("roll", kind, v), ("watch", heading, m)])
            return r, {"kind": kind, "was": at[kind], "now": at[kind] + m * v,
                       "will": at[kind] + (m + settings["ahead"]) * v}, heading
        return None

    def carried_room() -> tuple[world.Room, list[tuple[int, np.ndarray, np.ndarray]]]:
        """A room whose things the teacher carries: ``(kind, was, now)`` a carried thing. The
        room's things keep their first places: a life moves them at the carry events."""
        r = world.room(rng, trained, int(rng.integers(4, 7)))
        events.append(("room", r))
        at = {t.kind: t.place.copy() for t in r.things}
        moved = []
        for i in rng.permutation(len(r.things))[: settings["carries"]]:
            kind = r.things[int(i)].kind
            was = at[kind]
            at[kind] = carry_to(rng, r.heading, at, kind)
            events.append(("carry", kind, at[kind].copy()))
            moved.append((kind, was, at[kind].copy()))
        events.extend([("look",)] * settings["looks"])
        return r, moved

    for _ in range(settings["practice_rooms"]):
        r, moved = carried_room()
        asked = []
        for kind, was, now in moved:
            asked += [question(lex, said, r.heading, kind, "was", was),
                      question(lex, said, r.heading, kind, "is", now)]
        still = [t for t in r.things if all(t.kind != m for m, _, _ in moved)]
        if still:
            t = still[int(rng.integers(len(still)))]
            asked += [world.where(lex, r, t.kind), world.colour_of(lex, r, t.kind)]
        events += [("practice", asked[int(i)]) for i in rng.permutation(len(asked))]

    rolled = 0
    while rolled < settings["roll_rooms"]:
        made = rolled_room()
        if made is None:
            continue
        r, it, heading = made
        rolled += 1
        asked = [question(lex, said, heading, it["kind"], when, it[place])
                 for when, place in (("was", "was"), ("is", "now"), ("will", "will"))]
        still = [t for t in r.things if t.kind != it["kind"]]
        t = still[int(rng.integers(len(still)))]
        asked += [world.where(lex, world.Room(r.things, heading), t.kind),
                  world.colour_of(lex, world.Room(r.things, heading), t.kind)]
        events += [("practice", asked[int(i)]) for i in rng.permutation(len(asked))]

    for _ in range(settings["test_rooms"]):
        r, moved = carried_room()
        for kind, was, now in moved:
            extra = {"was": was, "now": now}
            events += [("ask", "W8_past", question(lex, said, r.heading, kind, "was", was),
                        extra),
                       ("ask", "W8_present", question(lex, said, r.heading, kind, "is", now),
                        extra)]
        still = [t for t in r.things if all(t.kind != m for m, _, _ in moved)]
        if still:  # the questions of protocol 1 about a thing that stayed
            t = still[int(rng.integers(len(still)))]
            extra = {"was": t.place.copy(), "now": t.place.copy()}
            events += [("ask", "still_where", world.where(lex, r, t.kind), extra),
                       ("ask", "still_colour", world.colour_of(lex, r, t.kind), extra)]

    rolled = 0
    while rolled < settings["roll_tests"]:
        made = rolled_room()
        if made is None:
            continue
        r, it, heading = made
        rolled += 1
        for gate, when, place in (("W8_past", "was", "was"), ("W8_present", "is", "now"),
                                  ("W9_future", "will", "will")):
            events.append(("ask", gate, question(lex, said, heading, it["kind"], when,
                                                 it[place]), {**it, "rolled": True}))
        still = [t for t in r.things if t.kind != it["kind"]]
        t = still[int(rng.integers(len(still)))]
        here = world.Room(r.things, heading)
        extra = {"was": t.place.copy(), "now": t.place.copy(), "will": t.place.copy()}
        events += [("ask", "still_where", world.where(lex, here, t.kind), extra),
                   ("ask", "still_will", question(lex, said, heading, t.kind, "will", t.place),
                    extra)]
    return events


def held_at(brain: Any, kind: int, place: np.ndarray) -> bool:
    """The map's own answer: it holds the kind within ``HELD`` of ``place`` and nowhere else
    (no place of the kind farther than twice that)."""
    places = brain.map.where_is(kind)
    apart = [float(np.linalg.norm(p - place)) for p in places]
    return any(d <= HELD for d in apart) and all(d <= 2 * HELD for d in apart)


def forecast_holds(brain: Any, kind: int, place: np.ndarray) -> tuple[bool, float]:
    """The forecast's own answer: whether every place the forecast ``ahead`` moments on holds
    of the kind lies in the place around the parrot of ``place`` (and it holds one), and its
    distance in metres from ``place`` (the nearest)."""
    from cadence.words.episodes import forecast

    ahead = forecast(brain.map, brain.ahead)
    places = ahead.where_is(kind)
    if not len(places):
        return False, float("nan")
    around = world.place_of(brain.body.heading, place)
    right = all(world.place_of(brain.body.heading, p) == around for p in places)
    return right, float(min(np.linalg.norm(p - place) for p in places))


def life(seed: int, *, record: Any = None, brain: Any = None, **given: Any) -> dict[str, Any]:
    """One founder's life on its script; the readings by gate. ``record`` (a list) takes every
    test question with what was said and what each source found."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    brain_settings = {k: v for k, v in given.items() if k not in DEFAULTS}
    events = script(seed, **settings)
    lex = world.lexicon(seed)
    said_as = time_words(seed)
    english = {**lex.english(), said_as["was"]: "was", said_as["will"]: "will"}
    if brain is None:
        brain = make_brain(seed, physics=settings["physics"], ahead=int(settings["ahead"]),
                           **brain_settings)
    readings: dict[str, list[bool]] = {}
    seconds: dict[str, float] = {}
    phase, began = "naming", time.time()
    r: world.Room | None = None
    for event in events:
        kind = event[0]
        now = {"room": None, "turn": None, "name": "naming", "carry": None, "look": None,
               "roll": None, "watch": None, "practice": "practice", "ask": "test"}[kind]
        if now is not None and now != phase:
            seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
            phase, began = now, time.time()
        if kind == "room":
            r = event[1]
            brain.enter()
            look(brain, r)
        elif kind == "turn":
            assert r is not None
            turn(brain, r, event[1])
        elif kind == "name":
            brain.name(event[2], event[1])
        elif kind == "carry":
            assert r is not None
            carry(r, event[1], event[2])
        elif kind == "look":
            assert r is not None
            look(brain, r)
        elif kind == "roll":
            assert r is not None
            roll(r, event[1], event[2])
        elif kind == "watch":
            assert r is not None
            watch(brain, r, event[1], event[2])
        elif kind == "practice":
            q = event[1]
            brain.practice(q.text, q.answer if len(q.answer) > 1 else q.answer[0])
        else:
            gate, q, extra = event[1], event[2], event[3]
            kept = brain.map.arrays()
            said = brain.ask(q.text)
            ok = world.right(q, said)
            readings.setdefault(gate, []).append(ok)
            readings.setdefault("W9_map_kept", []).append(all(
                np.array_equal(kept[k], v) for k, v in brain.map.arrays().items()))
            found = brain.last["found"]
            concepts = {s: sorted({c for k, c in found[s] if k == q.kind}) for s in found}
            three = [concepts[s][0] if len(concepts[s]) == 1 else None
                     for s in ("records", "map", "forecast")]
            if None not in three and len(set(three)) == 3 and q.form in TIMES:
                # W10: the three sources hold the thing in three places; is the asked one said?
                group, value = three[TIMES.index(q.form)]
                readings.setdefault(f"W10_three_{q.form}", []).append(said == [
                    {"kind": lex.kind, "colour": lex.colour, "place": lex.place}[group][value]])
            if gate == "W9_future":
                right, error = forecast_holds(brain, q.kind, extra["will"])
                readings.setdefault("W9_forecast_rate", []).append(right)
                readings.setdefault("W9_error", []).append(error)
                readings.setdefault("W9_now_rate", []).append(
                    world.place_of(brain.body.heading, extra["now"])
                    == world.place_of(brain.body.heading, extra["will"]))
            asked = "records" if q.form == "was" else "map"
            other = "map" if q.form == "was" else "records"
            apart = (len(concepts[asked]) == 1 and len(concepts[other]) == 1
                     and concepts[asked] != concepts[other])
            if apart:  # W10: the sources tell the times apart; does the brain?
                group, value = concepts[asked][0]
                told = said == [{"kind": lex.kind, "colour": lex.colour,
                                 "place": lex.place}[group][value]]
                readings.setdefault(f"W10_{q.form}", []).append(told)
            if gate == "W8_present":
                readings.setdefault("W8_map_rate", []).append(held_at(brain, q.kind,
                                                                      extra["now"]))
            barred_said: list[str] = []
            if gate == "W8_past":
                brain.records.barred = True
                try:
                    barred_said = brain.ask(q.text)
                finally:
                    brain.records.barred = False
                readings.setdefault("W8_barred", []).append(world.right(q, barred_said))
                readings.setdefault("W8_barred_refused", []).append(
                    barred_said in ([lex.none], [""]))
            if record is not None:
                record.append({"gate": gate, "question": q.text, "answer": q.answer,
                               "read": brain.last.get("read"),
                               "said": said, "right": ok, "barred": barred_said,
                               "sources": {s: [list(c) for c in v] for s, v in concepts.items()},
                               "gains": [round(float(g), 3) for g in brain.last["gains"]],
                               "english": " ".join(english.get(w, w) for w in q.text.split()),
                               "said_english": [english.get(w, w) for w in said]})
    seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
    return readings_of(seed, settings, readings, brain, seconds)


def readings_of(seed: int, settings: dict[str, Any], readings: dict[str, list[bool]],
                brain: Any, seconds: dict[str, float]) -> dict[str, Any]:
    """A life's readings: each gate's share and count, the pools, parameters and work."""
    out: dict[str, Any] = {"seed": int(seed), **settings}
    for gate, oks in sorted(readings.items()):
        finite = [x for x in oks if x == x]
        out[gate] = float(np.mean(finite)) if finite else float("nan")
        out[f"{gate}_n"] = len(oks)
    oks = readings.get("W10_was", []) + readings.get("W10_is", []) + readings.get(
        "W10_will", [])
    out["W10"] = float(np.mean(oks)) if oks else float("nan")
    out["W10_n"] = len(oks)
    oks = [ok for when in TIMES for ok in readings.get(f"W10_three_{when}", [])]
    out["W10_three"] = float(np.mean(oks)) if oks else float("nan")
    out["W10_three_n"] = len(oks)
    if brain is not None:
        out["parameters"] = int(brain.lexicon.connectome.synapses // 2)
        out["records"] = int(brain.records.store.count)
        out["work"] = {**brain.map.work, **brain.work, **brain.records.work}
        synapses = int(brain.map.connectome.synapses)
        out["macs"] = int(brain.lexicon.work["macs"] + brain.records.work["macs"] + synapses * (
            brain.map.work["moments"] + 2 * brain.map.work["questions"]
            + brain.work["forecast_moments"] + 2 * brain.work["forecast_questions"]))
    out["seconds"] = seconds
    return out


# -- the matched controls: an LSTM and a transformer on the same script as tokens


def tokens(seed: int) -> Any:
    """The controls' vocabulary of protocol 1 with the marker of a carry and the time words."""
    import controls

    tok = controls.Tokens(world.lexicon(seed))
    said = time_words(seed)
    for name in ("MOVE", said["was"], said["will"]):
        tok.index[name] = len(tok.names)
        tok.names.append(name)
    return tok


def sequences(seed: int, **given: Any) -> list[tuple[str, Any, Any]]:
    """The script as the controls' items, in the brain's order: ``("train", pairs, None)`` for a
    naming (its two sequences, as in protocol 1) or a practice question, ``("ask", gate,
    (prompt, question))`` for a test question. A question about a room whose things were
    carried is prompted with the room as first seen, ``MOVE`` and the room as seen after the
    carries (each thing's kind, colour and place around the parrot); a naming with the room as
    the parrot faces."""
    import controls

    tok = tokens(seed)
    out: list[tuple[str, Any, Any]] = []
    r: world.Room | None = None
    scene: list[int] = []
    for event in script(seed, **given):
        kind = event[0]
        if kind == "room":
            r = event[1]
            scene = controls.room_tokens(tok, r)
        elif kind == "turn":
            assert r is not None
            r.heading = event[1]
        elif kind == "carry":
            assert r is not None
            carry(r, event[1], event[2])
        elif kind == "look":
            assert r is not None
            seen = controls.room_tokens(tok, r)
            if seen != scene[-len(seen):]:
                scene = scene + [tok("MOVE")] + seen
        elif kind == "name":
            assert r is not None
            place, word = event[1], event[2]
            t = min(r.things, key=lambda th: float(np.linalg.norm(th.place - place)))
            concepts = [tok(f"kind{t.kind}"), tok(f"colour{t.colour}"),
                        tok(f"place{world.place_of(r.heading, t.place)}")]
            now = controls.room_tokens(tok, r)
            out.append(("train", [
                (now + [tok("NAME")] + concepts + [tok("SAY")], [tok(word), tok("END")]),
                (now + [tok("NAME"), tok(word), tok("SAY")], concepts + [tok("END")])], None))
        else:
            q = event[1] if kind == "practice" else event[2]
            prompt = scene + [tok("ASK")] + [tok(w) for w in q.text.split()] + [tok("SAY")]
            if kind == "practice":
                out.append(("train", [(prompt, [tok(w) for w in q.answer] + [tok("END")])],
                            None))
            else:
                out.append(("ask", event[1], (prompt, q)))
    return out


def control(seed: int, model: str, mode: str, parameters: int, **given: Any) -> dict[str, Any]:
    """One control's life on the founder's script (``controls.run``'s training, on this
    chamber's items); the readings by gate, its parameters and work."""
    import controls
    import torch

    torch.set_num_threads(1)
    tok = tokens(seed)
    items = sequences(seed, **given)
    width = controls.matched_width(model, parameters, len(tok))
    net = controls.Control(model, width, len(tok), seed, controls.RATES[mode])
    began = time.monotonic()
    train = [pairs for kind, pairs, _ in items if kind == "train"]
    if mode == "online":
        for pairs in train:
            net.update(pairs)
    else:
        flat = [p for pairs in train for p in pairs]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 13]))
        order = rng.permutation(len(flat))
        aside = [flat[i] for i in order[: len(flat) // 10]]
        fit = [flat[i] for i in order[len(flat) // 10:]]
        best, best_state, since, since_halved = math.inf, None, 0, 0
        rate = controls.RATES[mode]
        for _ in range(controls.EPOCHS):
            shuffled = rng.permutation(len(fit))
            for start in range(0, len(fit), controls.BATCH):
                net.update([fit[int(i)] for i in shuffled[start: start + controls.BATCH]])
            with torch.no_grad():
                held = float(np.mean([net.loss(aside[i: i + controls.BATCH]).item()
                                      for i in range(0, len(aside), controls.BATCH)]))
            if held < best - 1e-6:
                best, since, since_halved = held, 0, 0
                best_state = {k: v.clone() for k, v in net.net.state_dict().items()}
            else:
                since += 1
                since_halved += 1
                if since_halved >= controls.HALVE and rate > controls.FLOOR:
                    rate = max(rate / 2.0, controls.FLOOR)
                    since_halved = 0
                    for group in net.opt.param_groups:
                        group["lr"] = rate
            if since >= controls.PATIENCE:
                break
        if best_state is not None:
            net.net.load_state_dict(best_state)
    trained_at = time.monotonic()
    readings: dict[str, list[bool]] = {}
    for kind, gate, second in items:
        if kind == "ask":
            prompt, q = second
            said = [tok.names[t] for t in net.say(prompt, tok("END"))]
            readings.setdefault(gate, []).append(world.right(q, said))
    out = readings_of(seed, {}, readings, None,
                      {"train": trained_at - began, "test": time.monotonic() - trained_at})
    out.update({"model": model, "mode": mode, "width": width,
                "parameters": controls.count(net.net), "updates": net.work["updates"],
                "macs": net.work["macs"]})
    return out


# -- the declared run: every arm's lives on the founders, and the gates read from their files

PARAMETERS = 256 * (world.KINDS + world.COLOURS + 4 + 3 + 1 + 3)  # one efficacy a pair
CONTROLS = [(m, mode) for m in ("lstm", "transformer") for mode in ("online", "offline")]
READINGS = ["W8_past", "W8_present", "W8_map_rate", "W8_barred", "W8_barred_refused", "W10",
            "W10_was", "W10_is", "still_where", "still_colour"]


def jobs(out: Path, first: int, count: int) -> list[tuple[Path, list[str]]]:
    python = sys.executable
    todo = []
    for seed in range(first, first + count):
        todo.append((out / f"brain_{seed}.json", [python, str(HERE / "times.py"), str(seed)]))
        for model, mode in CONTROLS:
            code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import times; "
                    f"print(json.dumps(times.control({seed}, {model!r}, {mode!r}, "
                    f"{PARAMETERS}), default=float))")
            todo.append((out / f"{model}_{mode}_{seed}.json", [python, "-c", code]))
    return todo


def run(out: Path, first: int, count: int, workers: int) -> None:
    import gates

    out.mkdir(parents=True, exist_ok=True)
    keep = gates.jobs
    gates.jobs = lambda o, f, c: jobs(o, f, c)  # gates.run's pool over this chamber's jobs
    try:
        gates.run(out, first, count, workers)
    finally:
        gates.jobs = keep


def read(out: Path) -> dict[str, Any]:
    """The table of every arm and each gate's verdict, from a run's files."""
    import json

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
            row.append(f"{np.mean(values):.3f} ({min(values):.2f}-{max(values):.2f})"
                       if values else "")
        print("| " + " | ".join(row) + " |")
    brain = arms.get("brain", [])
    verdicts: dict[str, Any] = {}

    def gate(name: str, threshold: float) -> bool:
        values = np.array([life[name] for life in brain])
        reach = int((values >= threshold).sum())
        ok = bool(len(values)) and values.mean() >= threshold and reach >= min(10, len(values))
        print(f"{name}: {'holds' if ok else 'fails'} (mean {values.mean():.3f}, {reach} of "
              f"{len(values)} at {threshold})")
        return ok

    verdicts["W8_past"] = gate("W8_past", 0.90)
    gaps = [abs(life["W8_present"] - life["W8_map_rate"]) for life in brain]
    verdicts["W8_present"] = bool(gaps) and max(gaps) <= 0.02
    print(f"W8 present: {'holds' if verdicts['W8_present'] else 'fails'} (largest gap from the "
          f"map's own rate {max(gaps) if gaps else float('nan'):.3f})")
    barred = [life["W8_barred"] for life in brain]
    verdicts["W8_barred"] = bool(barred) and max(barred) <= 0.25
    print(f"W8 barred: {'holds' if verdicts['W8_barred'] else 'fails'} (the past with the "
          f"records barred at {max(barred) if barred else float('nan'):.3f} at most; chance .25)")
    verdicts["W8"] = verdicts["W8_past"] and verdicts["W8_present"] and verdicts["W8_barred"]
    verdicts["W10_was_is"] = gate("W10", 0.95)
    return verdicts


if __name__ == "__main__":
    import json

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 16)
    elif len(sys.argv) > 1 and sys.argv[1] == "read":
        read(Path(sys.argv[2]))
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        told = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        print(json.dumps(life(seed, **told), indent=1, default=float))
