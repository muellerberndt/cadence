"""The relations chamber (cadence#327): a parrot learns words for where things are from each
other, and from itself, from questions answered for it.

The world is the words chamber's (``world.py``): rooms of 4 m by 4 m, the parrot on its perch
at the middle, things of the 8 trained kinds at least 0.6 m from it and 0.45 m from each
other. The relations, each a nonce word drawn for the founder:

- **front, right, behind, left** of a reference (a thing, or the parrot itself): the quarter of
  the turn the thing lies in from the reference, in the frame of where the parrot faces (the
  parrot's own front, carried to the reference: "in front of the ball" is farther along the way
  the parrot faces);
- **near** (under ``NEAR`` metres from the reference) and **far** (``NEAR`` or more).

The teacher asks only questions whose answer is clear: every thing a relation is judged on lies
at least ``MARGIN_ANGLE`` from a quarter's edge, or ``MARGIN_DISTANCE`` from ``NEAR``.

Question forms (nonce words; English for the reader):

| Form | Words | Answer |
| --- | --- | --- |
| things | "what is R the X" / "what is R you" | the things in relation R, or "i do not know" |
| where | "where is the X from the Y" / "... from you" | the X's quarter word from the Y |
| how far | "howfar is the X from the Y" / "... from you" | near or far |
| is it | "is the X R the Y" / "is the X R you" | yes or no |

One life (``script``): naming (kinds, and "you" while the parrot attends to itself); practice
from the parrot ("where" and "how far" from it); practice of every form; the test in fresh
rooms, after the relation word held out of the life so far is taught in ``examples``
answered questions (``taught``: "what is R the X?" answered with the things, or "where is the
X from the Y?" answered with R); the test asks every form about the practised relations and
about the new one. Kind pairs drawn for the founder (``held_pairs``) never
meet in a practice room.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import world  # noqa: E402

RELATIONS = ("front", "right", "behind", "left", "near", "far")
NEAR = 1.0  # metres: under it near, from it far
MARGIN_ANGLE = math.radians(10.0)
MARGIN_DISTANCE = 0.15
FRAME = ("from", "howfar", "yes", "no")
KINDS = world.TRAINED
SALT = 327
FORMS = ("things", "things_you", "where", "where_you", "howfar", "howfar_you", "is", "is_you")
DEFAULTS = {"naming_rooms": 30, "namings": 12, "selves": 10, "body_rooms": 16, "body_asked": 8,
            "practice_rooms": 40, "asked": 10, "test_rooms": 30, "test_asked": 2,
            "examples": 4, "held_pairs": 5, "taught": "things"}


@dataclass
class Nonce:
    """A founder's nonce words for the chamber: the base lexicon's kinds, question words and
    nothing answer, and the relations and frame words drawn beside them."""

    base: world.Lexicon
    relation: list[str]
    frame: dict[str, str]

    def english(self) -> dict[str, str]:
        out = self.base.english()
        out.update(zip(self.relation, RELATIONS, strict=True))
        out.update({w: n for n, w in self.frame.items()})
        return out


def words_of(seed: int) -> Nonce:
    drawn = world.vocabulary(seed, {"relation": RELATIONS, "frame": FRAME}, salt=SALT)
    return Nonce(world.lexicon(seed), [drawn["relation"][n] for n in RELATIONS],
                 drawn["frame"])


# -- the truth of a relation


def placed(origin: np.ndarray, point: np.ndarray, heading: float
           ) -> tuple[int, float, float, float]:
    """A point from an origin, the parrot facing ``heading``: its quarter (0 front, 1 right, 2
    behind, 3 left), the angle to the nearest quarter's edge, its distance and that
    distance's gap to ``NEAR``."""
    d = np.asarray(point, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    a = float(np.angle(np.exp(1j * (math.atan2(d[1], d[0]) - heading))))
    quarter = int(((-a + math.pi / 4) % (2 * math.pi)) // (math.pi / 2))
    edge = (a - math.pi / 4) % (math.pi / 2)
    far = float(np.hypot(d[0], d[1]))
    return quarter, min(edge, math.pi / 2 - edge), far, abs(far - NEAR)


def holds(relation: int, origin: np.ndarray, point: np.ndarray, heading: float) -> bool:
    quarter, _, far, _ = placed(origin, point, heading)
    if relation < 4:
        return quarter == relation
    return (far < NEAR) == (relation == 4)


def clear(relation: int, origin: np.ndarray, point: np.ndarray, heading: float) -> bool:
    """Whether the teacher judges a relation clearly here."""
    _, edge, _, gap = placed(origin, point, heading)
    return edge >= MARGIN_ANGLE if relation < 4 else gap >= MARGIN_DISTANCE


# -- the teacher's questions


@dataclass
class Asked:
    text: str
    form: str  # one of FORMS
    answer: list[str]
    relation: int  # the relation named or answered (0 to 5)
    pair: tuple[int, ...]  # the kinds the question places from each other (reference first)
    room: list[int]  # the kinds in the room
    thing: int | None = None  # the kind of the thing a "where", "how far" or "is" places


ORIGIN = np.zeros(2)


def things_in(w: Nonce, r: world.Room, relation: int, ref: world.Thing | None
              ) -> Asked | None:
    """"what is R the X" (``ref``) or "what is R you" (None): None where a thing is not
    clear."""
    q = w.base.question
    origin = ORIGIN if ref is None else ref.place
    others = [t for t in r.things if t is not ref]
    if not all(clear(relation, origin, t.place, r.heading) for t in others):
        return None
    members = [t for t in others if holds(relation, origin, t.place, r.heading)]
    answer = sorted(w.base.kind[t.kind] for t in members) or [w.base.none]
    tail = q["you"] if ref is None else f"{q['the']} {w.base.kind[ref.kind]}"
    return Asked(f"{q['what']} {q['is']} {w.relation[relation]} {tail}",
                 "things_you" if ref is None else "things", answer, relation,
                 () if ref is None else (ref.kind,), r.kinds())


def where_from(w: Nonce, r: world.Room, x: world.Thing, ref: world.Thing | None, *,
               far: bool = False) -> Asked | None:
    """"where is the X from the Y" (or "howfar ...", ``far``), from the parrot where ``ref``
    is None: None where the answer is not clear."""
    q = w.base.question
    origin = ORIGIN if ref is None else ref.place
    quarter, _, distance, _ = placed(origin, x.place, r.heading)
    relation = (4 if distance < NEAR else 5) if far else quarter
    if not clear(relation, origin, x.place, r.heading):
        return None
    tail = q["you"] if ref is None else f"{q['the']} {w.base.kind[ref.kind]}"
    head = w.frame["howfar"] if far else q["where"]
    form = ("howfar" if far else "where") + ("_you" if ref is None else "")
    return Asked(f"{head} {q['is']} {q['the']} {w.base.kind[x.kind]} {w.frame['from']} {tail}",
                 form, [w.relation[relation]], relation,
                 () if ref is None else (ref.kind, x.kind), r.kinds(), x.kind)


def is_it(w: Nonce, r: world.Room, x: world.Thing, relation: int, ref: world.Thing | None
          ) -> Asked | None:
    """"is the X R the Y" or "is the X R you": None where the relation is not clear."""
    q = w.base.question
    origin = ORIGIN if ref is None else ref.place
    if not clear(relation, origin, x.place, r.heading):
        return None
    yes = holds(relation, origin, x.place, r.heading)
    tail = q["you"] if ref is None else f"{q['the']} {w.base.kind[ref.kind]}"
    return Asked(f"{q['is']} {q['the']} {w.base.kind[x.kind]} {w.relation[relation]} {tail}",
                 "is_you" if ref is None else "is", [w.frame["yes" if yes else "no"]],
                 relation, () if ref is None else (ref.kind, x.kind), r.kinds(), x.kind)


def every_question(w: Nonce, r: world.Room, relations: list[int], rng: np.random.Generator,
                   forms: tuple[str, ...] = FORMS) -> dict[str, list[Asked]]:
    """Every clear question of the forms about a room, by form; "is it" questions balanced
    between yes and no."""
    out: dict[str, list[Asked]] = {f: [] for f in forms}
    things = r.things
    directions = [k for k in relations if k < 4]
    for ref in [None, *things]:
        you = "_you" if ref is None else ""
        for k in relations:
            if f"things{you}" in forms:
                a = things_in(w, r, k, ref)
                if a is not None:
                    out[f"things{you}"].append(a)
        for x in things:
            if x is ref:
                continue
            if f"where{you}" in forms:
                a = where_from(w, r, x, ref)
                if a is not None and a.relation in directions:
                    out[f"where{you}"].append(a)
            if f"howfar{you}" in forms:
                a = where_from(w, r, x, ref, far=True)
                if a is not None and a.relation in relations:
                    out[f"howfar{you}"].append(a)
            if f"is{you}" in forms:
                for k in relations:
                    a = is_it(w, r, x, k, ref)
                    if a is not None:
                        out[f"is{you}"].append(a)
    for form in [f for f in forms if f.startswith("is")]:
        yes = [a for a in out[form] if a.answer == [w.frame["yes"]]]
        no = [a for a in out[form] if a.answer != [w.frame["yes"]]]
        n = min(len(yes), len(no))
        picked = [yes[int(i)] for i in rng.permutation(len(yes))[:n]] + [
            no[int(i)] for i in rng.permutation(len(no))[:n]]
        out[form] = [picked[int(i)] for i in rng.permutation(len(picked))]
    return out


def draw(rng: np.random.Generator, pool: dict[str, list[Asked]], count: int) -> list[Asked]:
    """``count`` questions: a form drawn evenly among those with questions left, then one of its
    questions."""
    left = {f: list(qs) for f, qs in pool.items() if qs}
    out: list[Asked] = []
    while left and len(out) < count:
        form = sorted(left)[int(rng.integers(len(left)))]
        out.append(left[form].pop(int(rng.integers(len(left[form])))))
        if not left[form]:
            del left[form]
    return out


def held_out_pairs(seed: int, count: int) -> set[frozenset[int]]:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), SALT, 2]))
    pairs = [frozenset((a, b)) for a in range(KINDS) for b in range(a + 1, KINDS)]
    return {pairs[int(i)] for i in rng.permutation(len(pairs))[:count]}


def held_out_word(seed: int) -> int:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), SALT, 4]))
    return int(rng.integers(len(RELATIONS)))


def a_room(rng: np.random.Generator, held: set[frozenset[int]] | None = None) -> world.Room:
    """A room of 4 to 6 trained things; with ``held``, no two of its kinds a held-out pair."""
    for _ in range(10000):
        r = world.room(rng, list(range(KINDS)), int(rng.integers(4, 7)))
        kinds = r.kinds()
        if not held or not any(frozenset((a, b)) in held for i, a in enumerate(kinds)
                               for b in kinds[i + 1:]):
            return r
    raise RuntimeError("no room without a held-out pair")


def script(seed: int, **given: Any) -> list[tuple[Any, ...]]:
    """A founder's life as events, the same for the brain and the controls: ``("room", room)``,
    ``("turn", heading)``, ``("name", place, word)``, ``("self", word)`` (the parrot attends to
    itself), ``("practice", asked)`` and ``("ask", gate, asked)``."""
    s = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), SALT, 3]))
    w = words_of(seed)
    held = held_out_pairs(seed, s["held_pairs"])
    new = held_out_word(seed)
    known = [k for k in range(len(RELATIONS)) if k != new]
    events: list[tuple[Any, ...]] = []

    # 1. naming: kinds, and the parrot itself
    for i in range(s["naming_rooms"]):
        r = a_room(rng)
        events.append(("room", r))
        for _ in range(s["namings"]):
            heading = float(rng.uniform(-math.pi, math.pi))
            events.append(("turn", heading))
            r.heading = heading
            t = r.things[int(rng.integers(len(r.things)))]
            events.append(("name", t.place.copy(), w.base.kind[t.kind]))
        if i < s["selves"]:
            events.append(("self", w.base.question["you"]))

    # 2. practice from the parrot: where things are from it, and how far
    for _ in range(s["body_rooms"]):
        r = a_room(rng, held)
        events.append(("room", r))
        pool = every_question(w, r, known, rng, ("where_you", "howfar_you"))
        events += [("practice", a) for a in draw(rng, pool, s["body_asked"])]

    # 3. practice of every form
    for _ in range(s["practice_rooms"]):
        r = a_room(rng, held)
        events.append(("room", r))
        pool = every_question(w, r, known, rng)
        events += [("practice", a) for a in draw(rng, pool, s["asked"])]

    # 4. the held-out relation word, taught in a few answered questions. ``taught="things"``:
    # "what is R the X?" in the words the parrot knows, answered with the things (never none,
    # one of them each time at another part of what R does not name), R heard for the first
    # time at the first; ``"where"``: "where is the X from the Y?" (or
    # "how far"), answered with R, the examples spanning what R names and what it does not (a
    # word for a quarter across its quarter, each at another distance; a word for a distance
    # across its range, each in another quarter)
    far = new >= 4
    taught = 0
    while taught < s["examples"]:
        r = a_room(rng)
        if s["taught"] == "things":
            found = [a for a in every_question(w, r, [new], rng, ("things",))["things"]
                     if a.answer != [w.base.none] and taught % SPANS in others_of(a, r)]
        else:
            form = "howfar" if far else "where"
            found = [a for a in every_question(w, r, [new], rng, (form,))[form]
                     if a.relation == new and other_of(a, r) == taught % SPANS
                     and span_of(a, r) == taught % SPANS]
        if not found:
            continue
        events += [("room", r), ("practice", found[int(rng.integers(len(found)))])]
        taught += 1

    # 5. the test: fresh rooms, every form, the practised relations and the new one, nothing
    # taught
    for _ in range(s["test_rooms"]):
        r = a_room(rng)
        events.append(("room", r))
        pool = every_question(w, r, known, rng)
        for qs in pool.values():
            for i in rng.permutation(len(qs))[: s["test_asked"]]:
                a = qs[int(i)]
                events.append(("ask", gate_of(a), a))
        for asked in every_question(w, r, [new], rng).values():
            kept = [a for a in asked if a.relation == new]
            for i in rng.permutation(len(kept))[: s["test_asked"]]:
                events.append(("ask", "R5_new", kept[int(i)]))
    return events


def other_of(a: Asked, r: world.Room) -> int:
    """What the answer to a "where" or "how far" question about two things does not name: the
    quarter of a distance's example; for a quarter's example its distance, in ``SPANS`` parts
    of 0.45 m to 3.6 m on a log scale."""
    things = {t.kind: t for t in r.things}
    origin, point = things[a.pair[0]].place, things[a.pair[1]].place
    quarter, _, distance, _ = placed(origin, point, r.heading)
    if a.relation >= 4:
        return quarter
    part = math.log(distance / 0.45) / math.log(3.6 / 0.45)
    return min(max(int(part * SPANS), 0), SPANS - 1)


SPANS = 4  # the parts of a relation's range a new word's examples span in turn


def span_of(a: Asked, r: world.Room) -> int:
    """Which of ``SPANS`` parts of its relation's range the answer to a "where" or "how far"
    question about two things lies in: across a quarter (clear of its edges), across near
    (from the closest two things come to the edge) or across far (from the edge outward, on a
    log scale to 3 m and beyond)."""
    things = {t.kind: t for t in r.things}
    origin, point = things[a.pair[0]].place, things[a.pair[1]].place
    d = point - origin
    angle = float(np.angle(np.exp(1j * (math.atan2(d[1], d[0]) - r.heading))))
    distance = float(np.hypot(d[0], d[1]))
    if a.relation < 4:
        offset = float(np.angle(np.exp(1j * (angle + a.relation * math.pi / 2))))
        part = (offset + math.pi / 4 - MARGIN_ANGLE) / (math.pi / 2 - 2 * MARGIN_ANGLE)
    elif a.relation == 4:
        part = (distance - 0.45) / (NEAR - MARGIN_DISTANCE - 0.45)
    else:
        part = math.log(distance / (NEAR + MARGIN_DISTANCE)) / math.log(3.0 / (
            NEAR + MARGIN_DISTANCE))
    return min(max(int(part * SPANS), 0), SPANS - 1)


def others_of(a: Asked, r: world.Room) -> set[int]:
    """What the answer to "what is R the X?" does not name, for each thing in it (as
    ``other_of``: the distance part of a quarter's thing, the quarter of a distance's)."""
    things = {t.kind: t for t in r.things}
    origin = things[a.pair[0]].place
    out = set()
    for t in r.things:
        if t.kind == a.pair[0] or not holds(a.relation, origin, t.place, r.heading):
            continue
        quarter, _, distance, _ = placed(origin, t.place, r.heading)
        part = math.log(distance / 0.45) / math.log(3.6 / 0.45)
        out.add(quarter if a.relation >= 4 else min(max(int(part * SPANS), 0), SPANS - 1))
    return out


def gate_of(a: Asked) -> str:
    """The gate a test question reads."""
    gate = {"things": "R1_thing", "things_you": "R1_you", "where": "R2_thing",
            "howfar": "R2_thing", "where_you": "R2_you", "howfar_you": "R2_you",
            "is": "R3_thing", "is_you": "R3_you"}[a.form]
    return str(gate)


def held_pair(a: Asked, held: set[frozenset[int]]) -> bool:
    """Whether a question about things places a held-out pair: the two things named, or the
    reference and a thing of its room."""
    if len(a.pair) == 2:
        return frozenset(a.pair) in held
    if len(a.pair) == 1:
        return any(frozenset((a.pair[0], k)) in held for k in a.room if k != a.pair[0])
    return False


def right(a: Asked, said: list[str]) -> bool:
    return sorted(set(said)) == sorted(set(a.answer)) and len(said) == len(a.answer)


# -- the brain's life


def make_brain(seed: int, **settings: Any) -> Any:
    """The founder's brain: a map of the chamber's plane, one unit a kind, the parrot at the
    middle, a lexicon of ``capacity`` words, and the relations part."""
    from cadence.space.map import Map
    from cadence.words import Words as Brain
    from cadence.words.relations import Relations

    part = {k: settings.pop(k) for k in list(settings) if k in (
        "directions", "sharpness", "distances", "nearest", "farthest", "width", "tries", "norm",
        "contrast")}
    settings.setdefault("capacity", 64)
    settings.setdefault("heard_only", True)  # the lexicon's opt-ins the part needs
    settings.setdefault("pattern_back", True)
    settings.setdefault("shares", {"kind": 0.4, "place": 0.3})  # the words chamber's shares
    held = Map(world.PLANE, world.CELLS, units=KINDS, seed=int(seed))
    return Brain(held, parts=[Relations(**part)], **settings)


def sight(r: world.Room, heading: float) -> list[tuple[int, np.ndarray]]:
    """What the parrot sees facing ``heading``: each thing in view, its kind and place."""
    return [(t.kind, t.place.copy()) for t in r.things if world.in_view(heading, t.place)[0]]


def look_around(brain: Any, r: world.Room) -> None:
    """A new room: the map is emptied, the parrot looks around in eight moments and turns its
    head back to the room's heading."""
    brain.map.reset()
    for i in range(9):
        heading = r.heading + (i % 8) * math.pi / 4 if i < 8 else r.heading
        brain.body.heading = heading
        brain.look(seen=sight(r, heading), lit=world.camera(heading))


def turn(brain: Any, r: world.Room, heading: float) -> None:
    r.heading = heading
    brain.body.heading = heading
    brain.look(seen=sight(r, heading), lit=world.camera(heading))


def part_of(brain: Any) -> Any:
    from cadence.words.relations import Relations

    return next(p for p in brain.parts if isinstance(p, Relations))


def ablate(brain: Any) -> Any:
    """The brain's map, body and lexicon with every synapse of the relation groups at zero."""
    from cadence.words import Lexicon
    from cadence.words import Words as Brain
    from cadence.words.relations import RELATION_GROUPS, Relations

    lexicon = brain.lexicon
    arrays = lexicon.arrays()
    cut = np.concatenate([lexicon.concept[lexicon.groups[g]] for g in RELATION_GROUPS])
    touched = np.isin(np.asarray(lexicon.connectome.pre), cut) | np.isin(
        np.asarray(lexicon.connectome.post), cut)
    arrays["efficacy"] = np.where(touched, 0.0, arrays["efficacy"])
    twin = Lexicon.from_arrays(lexicon.settings, arrays)
    return Brain(brain.map, parts=[Relations(**part_of(brain).settings)], lexicon=twin,
                 body=brain.body, senses=brain.senses)


def life(seed: int, *, record: Any = None, brain: Any = None, **given: Any) -> dict[str, Any]:
    """One founder's life on its script; the readings by gate. ``record`` (a list) takes every
    question asked in the test with what was said."""
    settings = {k: v for k, v in given.items() if k in DEFAULTS}
    brain_settings = {k: v for k, v in given.items() if k not in DEFAULTS}
    events = script(seed, **settings)
    w = words_of(seed)
    english = w.english()
    held = held_out_pairs(seed, {**DEFAULTS, **settings}["held_pairs"])
    brain = make_brain(seed, **brain_settings) if brain is None else brain
    readings: dict[str, list[bool]] = {}
    seconds: dict[str, float] = {}
    phase, began = "naming", time.time()
    r: world.Room | None = None
    ablated: Any = None
    for event in events:
        kind = event[0]
        now = {"practice": "practice", "ask": "test"}.get(kind)
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
        elif kind == "self":
            part_of(brain).name_self(brain, event[1])
        elif kind == "practice":
            a = event[1]
            brain.practice(a.text, a.answer if len(a.answer) > 1 else a.answer[0])
        else:
            gate, a = event[1], event[2]
            said = brain.ask(a.text)
            ok = right(a, said)
            readings.setdefault(gate, []).append(ok)
            readings.setdefault(f"{gate}_{a.form}", []).append(ok)
            if gate != "R5_new" and held_pair(a, held):
                readings.setdefault("R4_held", []).append(ok)
            if gate[:2] in ("R1", "R2") and a.answer != [w.base.none]:
                ablated = ablated if ablated is not None else ablate(brain)
                readings.setdefault("R6_ablated", []).append(right(a, ablated.ask(a.text)))
            if record is not None:
                record.append({"gate": gate, "form": a.form, "question": a.text,
                               "answer": a.answer, "said": said, "right": ok,
                               "english": " ".join(english.get(x, x) for x in a.text.split()),
                               "said_english": [english.get(x, x) for x in said],
                               "answer_english": [english.get(x, x) for x in a.answer]})
    seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
    return readings_of(seed, settings, readings, brain, seconds)


def readings_of(seed: int, settings: dict[str, Any], readings: dict[str, list[bool]],
                brain: Any, seconds: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"seed": int(seed), **settings, "held_word": RELATIONS[
        held_out_word(seed)]}
    for gate, oks in sorted(readings.items()):
        out[gate] = float(np.mean(oks)) if oks else float("nan")
        out[f"{gate}_n"] = len(oks)
    if brain is not None:
        out["parameters"] = int(brain.lexicon.connectome.synapses // 2)
        out["lessons"] = brain.lexicon.lessons
        out["words"] = len(brain.lexicon.forms)
        synapses = int(brain.map.connectome.synapses)
        out["macs"] = int(brain.lexicon.work["macs"] + synapses * (
            brain.map.work["moments"] + 2 * brain.map.work["questions"]))
    out["seconds"] = seconds
    return out


if __name__ == "__main__":
    import json

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    told = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    record: list[Any] = []
    out = life(seed, record=record, **told)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
