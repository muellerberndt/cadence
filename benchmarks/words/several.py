"""The words chamber's rung of several things (cadence#328): several things of one kind told apart
by their colours, answers that name several things, "how many?" and "is there?".

The parrot is the chamber's (``world.py``): it sits at the room's middle, looks around and turns
its head. A room holds 4 to 6 things, and one or two of its kinds come two or three times, each
time in another colour ("the red ball", "the blue ball"); no colour is held by more than three
things. Every word is a nonce word drawn for the founder: the chamber's words of ``world.py``
and, drawn here, the question words of the new forms, the four number words (none to three)
and the two answers of "is there?" (no, yes).

One life (``script``):

1. **Naming**: ``naming_rooms`` rooms of the eight trained kinds, ``namings`` namings a room, as
   in ``words.py``: the parrot turns its head, attends to a thing and hears its kind's, its
   colour's or its place's word.
2. **Practice**: ``practice_rooms`` rooms of kinds 0 to 3, ``asked`` questions a room with the
   teacher's answer, drawn from the forms below in the teacher's order: the first
   ``counting_rooms`` rooms ask every form but "is there?", the rest every form. The number
   words and the answers to "is there?" are heard here alone: no naming shows them.
3. **The test**: ``test_rooms`` fresh rooms of the trained kinds, every form; nothing taught.
   Kinds 4 to 7 are named and never asked about in practice.

The forms (English for the reader; the parrot hears nonce words):

- ``apart``: where is the red ball? (two or three balls): the red one's place around the parrot;
- ``where``, ``colour``: where is the ball? what colour is the ball?: every ball's place, or
  colour;
- ``by_colour``, ``by_place``: what is red? what is behind you?: every thing's kind;
- ``count_kind``, ``count_colour``, ``count_both``: how many balls / red things / red balls are
  there?: the number word;
- ``there_kind``, ``there_both``: is there a ball? is there a red ball?: no or yes;
- foils: how many / is there / where is the (a word never taught)?: i do not know.

    python several.py 2800                       # one life of the brain, its readings as JSON
    python several.py run <out> 32800000 12 [jobs]   # the declared run, every arm
    python several.py read <out>                 # the table and the gates
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

import controls  # noqa: E402
import words as base  # noqa: E402
import world  # noqa: E402

DEFAULTS = {"naming_rooms": 40, "namings": 15, "practice_rooms": 40, "asked": 14,
            "test_rooms": 30, "counting_rooms": 12}
LATER = ("there_kind", "there_both")  # the forms the first ``counting_rooms`` rooms do not ask
QUESTION_WORDS = ("how", "many", "are", "there", "things", "a")
NUMBER_NAMES = ("zero", "one", "two", "three")
TRUTH_NAMES = ("no", "yes")
MOST = 3  # things of one kind, and things of one colour, in a room at most
NUMBERS = 6  # the brain's number sense: counts from none to five or more


@dataclass
class Words:
    """A founder's nonce words: the chamber's (``world.lexicon``) and this rung's."""

    base: world.Lexicon
    question: dict[str, str]
    number: list[str]
    truth: list[str]

    def english(self) -> dict[str, str]:
        out = self.base.english()
        out.update({w: n for n, w in self.question.items()})
        out.update(zip(self.number, NUMBER_NAMES, strict=True))
        out.update(zip(self.truth, TRUTH_NAMES, strict=True))
        return out


def lexicon(seed: int) -> Words:
    """The founder's words: the chamber's (``world.lexicon``), and this rung's question words,
    number words and answers to "is there?" (``world.vocabulary``, distinct from the others)."""
    drawn = world.vocabulary(seed, {"question": QUESTION_WORDS, "number": NUMBER_NAMES,
                                    "truth": TRUTH_NAMES}, salt=328)
    return Words(world.lexicon(seed), drawn["question"],
                 [drawn["number"][n] for n in NUMBER_NAMES],
                 [drawn["truth"][n] for n in TRUTH_NAMES])


# -- rooms with several things of one kind


def room(rng: np.random.Generator, kinds: list[int], count: int, *, repeated: int) -> world.Room:
    """A room of ``count`` things whose kinds are drawn from ``kinds``: ``repeated`` kinds come two
    or three times, each time in another colour, the others once. No colour is held by more than
    ``MOST`` things; things are 0.6 to 1.8 m from the parrot and 0.45 m apart or more."""
    for _ in range(1000):
        chosen = [int(k) for k in rng.choice(kinds, size=min(count, len(kinds)), replace=False)]
        plan: list[int] = []
        for i, kind in enumerate(chosen):
            plan += [kind] * (int(rng.integers(2, MOST + 1)) if i < repeated else 1)
        plan = plan[:count] if len(plan) >= count else plan
        colours: list[int] = []
        for i, kind in enumerate(plan):
            used = {c for k, c in zip(plan[:i], colours, strict=True) if k == kind}
            full = {c for c in range(world.COLOURS) if colours.count(c) >= MOST}
            options = [c for c in range(world.COLOURS) if c not in used | full]
            colours.append(int(rng.choice(options)))
        if len(plan) == count:
            break
    things: list[world.Thing] = []
    for kind, colour in zip(plan, colours, strict=True):
        things.append(world.Thing(kind, colour, base._free_place(rng, world.Room(things))))
    return world.Room(things, heading=float(rng.uniform(-math.pi, math.pi)))


# -- the teacher's questions


def _view(r: world.Room, things: list[world.Thing]) -> str:
    """Whether the things an answer names are in view, out of view or both."""
    seen = {bool(world.in_view(r.heading, t.place)[0]) for t in things}
    return "in_view" if seen == {True} else "out_of_view" if seen == {False} else "both"


def _thing_word(x: Words, kind: int) -> str:
    return x.base.kind[kind] if kind >= 0 else x.base.foils[-kind - 1]


def apart(x: Words, r: world.Room, t: world.Thing) -> world.Question:
    q = x.base.question
    text = f"{q['where']} {q['is']} {q['the']} {x.base.colour[t.colour]} {x.base.kind[t.kind]}"
    return world.Question(text, "apart", [x.base.place[world.place_of(r.heading, t.place)]],
                          _view(r, [t]), t.kind)


def where(x: Words, r: world.Room, kind: int) -> world.Question:
    q = x.base.question
    text = f"{q['where']} {q['is']} {q['the']} {_thing_word(x, kind)}"
    things = [t for t in r.things if t.kind == kind]
    answer = sorted(x.base.place[world.place_of(r.heading, t.place)] for t in things)
    return world.Question(text, "where", answer or [x.base.none], _about(r, things, kind), kind)


def colour(x: Words, r: world.Room, kind: int) -> world.Question:
    q = x.base.question
    text = f"{q['what']} {q['colour']} {q['is']} {q['the']} {_thing_word(x, kind)}"
    things = [t for t in r.things if t.kind == kind]
    answer = sorted(x.base.colour[t.colour] for t in things)
    return world.Question(text, "colour", answer or [x.base.none], _about(r, things, kind), kind)


def _about(r: world.Room, things: list[world.Thing], kind: int) -> str:
    if kind < 0:
        return "foil"
    if not things:
        return "absent"
    return "several" if len(things) > 1 else _view(r, things)


def count_kind(x: Words, r: world.Room, kind: int) -> world.Question:
    q = x.question
    text = f"{q['how']} {q['many']} {_thing_word(x, kind)} {q['are']} {q['there']}"
    things = [t for t in r.things if t.kind == kind]
    answer = [x.base.none] if kind < 0 else [x.number[len(things)]]
    return world.Question(text, "count_kind", answer, "foil" if kind < 0 else _view(r, things)
                          if things else "absent", kind)


def count_colour(x: Words, r: world.Room, c: int) -> world.Question:
    q = x.question
    text = f"{q['how']} {q['many']} {x.base.colour[c]} {q['things']} {q['are']} {q['there']}"
    things = [t for t in r.things if t.colour == c]
    return world.Question(text, "count_colour", [x.number[len(things)]],
                          _view(r, things) if things else "absent")


def count_both(x: Words, r: world.Room, kind: int, c: int) -> world.Question:
    q = x.question
    text = (f"{q['how']} {q['many']} {x.base.colour[c]} {x.base.kind[kind]} {q['are']} "
            f"{q['there']}")
    things = [t for t in r.things if t.kind == kind and t.colour == c]
    return world.Question(text, "count_both", [x.number[len(things)]],
                          _view(r, things) if things else "absent", kind)


def there_kind(x: Words, r: world.Room, kind: int) -> world.Question:
    q, b = x.question, x.base.question
    text = f"{b['is']} {q['there']} {q['a']} {_thing_word(x, kind)}"
    things = [t for t in r.things if t.kind == kind]
    answer = [x.base.none] if kind < 0 else [x.truth[int(bool(things))]]
    return world.Question(text, "there_kind", answer, "foil" if kind < 0 else _view(r, things)
                          if things else "absent", kind)


def there_both(x: Words, r: world.Room, kind: int, c: int) -> world.Question:
    q, b = x.question, x.base.question
    text = f"{b['is']} {q['there']} {q['a']} {x.base.colour[c]} {x.base.kind[kind]}"
    things = [t for t in r.things if t.kind == kind and t.colour == c]
    return world.Question(text, "there_both", [x.truth[int(bool(things))]],
                          _view(r, things) if things else "absent", kind)


def room_questions(x: Words, r: world.Room, rng: np.random.Generator, *, kinds: list[int],
                   foils: list[int]) -> list[world.Question]:
    """Every form about a room: each thing of a kind held several times told apart; where and
    what colour each kind is; what has a colour and what is at each place around the parrot;
    how many of each kind held, of a kind not held, of a colour, of a kind in a colour it has and
    one it has not; whether there is a kind held, one not held, a kind in a colour it has and
    in one it has not; and a foil in three forms."""
    held = sorted(set(r.kinds()))
    gone = [k for k in kinds if k not in held]
    out: list[world.Question] = []
    for kind in held:
        things = [t for t in r.things if t.kind == kind]
        if len(things) > 1:
            out += [apart(x, r, t) for t in things]
        out += [where(x, r, kind), colour(x, r, kind), count_kind(x, r, kind),
                there_kind(x, r, kind)]
        t = things[int(rng.integers(len(things)))]
        other = int(rng.choice([c for c in range(world.COLOURS)
                                if c not in {s.colour for s in things}]))
        out += [count_both(x, r, kind, t.colour), count_both(x, r, kind, other),
                there_both(x, r, kind, t.colour), there_both(x, r, kind, other)]
    colours = sorted({t.colour for t in r.things})
    out.append(world.by_colour(x.base, r, int(rng.choice(colours))))
    out += [world.by_place(x.base, r, p) for p in range(len(world.PLACE_NAMES))]
    out += [count_colour(x, r, c) for c in colours]
    absent = [c for c in range(world.COLOURS) if c not in colours]
    if absent:
        out.append(count_colour(x, r, int(rng.choice(absent))))
    if gone:
        k = int(rng.choice(gone))
        out += [where(x, r, k), count_kind(x, r, k), there_kind(x, r, k)]
    if foils:
        f = -(int(rng.choice(foils)) + 1)
        out += [where(x, r, f), count_kind(x, r, f), there_kind(x, r, f)]
    return out


def gate_of(q: world.Question) -> str:
    """The reading a test question counts for."""
    part = "named_only" if q.kind is not None and q.kind >= 4 else "practised"
    if q.about == "foil":
        return "S5_never_taught"
    if q.form == "apart":
        return f"S1_{part}"
    if q.form.startswith("count"):
        return f"S3_{part}" if q.form != "count_colour" else "S3_colour"
    if q.form.startswith("there"):
        return f"S4_{part}"
    if q.form in ("where", "colour"):
        return "S2_several" if q.about == "several" else (
            "S5_never_seen" if q.about == "absent" else "one")
    return "S2_several" if len(q.answer) > 1 else "one"


# -- a life


def script(seed: int, **given: Any) -> list[tuple[Any, ...]]:
    """A founder's life as events, the same for the brain and the controls (the events of
    ``words.script``: ``room``, ``turn``, ``name``, ``practice`` and ``ask``)."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 328, 3]))
    x = lexicon(seed)
    trained = list(range(world.TRAINED))
    events: list[tuple[Any, ...]] = []
    for _ in range(settings["naming_rooms"]):
        r = room(rng, trained, int(rng.integers(4, 7)), repeated=int(rng.integers(0, 3)))
        events.append(("room", r))
        for _ in range(settings["namings"]):
            heading = float(rng.uniform(-math.pi, math.pi))
            events.append(("turn", heading))
            r.heading = heading
            t = r.things[int(rng.integers(len(r.things)))]
            events.append(("name", t.place.copy(), base.naming_word(x.base, r, t, rng), "naming"))
    practised = list(range(4))
    for index in range(settings["practice_rooms"]):
        r = room(rng, practised, int(rng.integers(3, 6)), repeated=int(rng.integers(1, 3)))
        events.append(("room", r))
        asked = [q for q in room_questions(x, r, rng, kinds=practised, foils=list(range(6)))
                 if index >= settings["counting_rooms"] or q.form not in LATER]
        for i in rng.permutation(len(asked))[: settings["asked"]]:
            events.append(("practice", asked[int(i)]))
    for _ in range(settings["test_rooms"]):
        r = room(rng, trained, int(rng.integers(4, 7)), repeated=int(rng.integers(1, 3)))
        events.append(("room", r))
        for q in room_questions(x, r, rng, kinds=trained, foils=list(range(6, 12))):
            events.append(("ask", gate_of(q), q))
    return events


def make_brain(seed: int, **settings: Any) -> Any:
    """The founder's brain: the chamber's map (one unit a kind, the colours on its sheet), the
    parrot at the middle, the part of several things, and a fresh lexicon."""
    from cadence.space.map import Map
    from cadence.words import Words
    from cadence.words.several import Several

    held = Map(world.PLANE, world.CELLS, units=world.KINDS, values={"colour": world.COLOURS},
               seed=int(seed))
    return Words(held, parts=[Several(numbers=NUMBERS)], seed=int(seed), **settings)


def map_rate(brain: Any, r: world.Room) -> tuple[list[bool], list[bool]]:
    """The map's own rates in a room: for each thing, whether the map holds its kind within
    0.15 m of its place with its colour; for each kind held, whether the map's places of the
    kind are as many as the kind's things."""
    held, counted = [], []
    for kind in sorted(set(r.kinds())):
        places = brain.map.where_is(kind)
        things = [t for t in r.things if t.kind == kind]
        counted.append(len(places) == len(things))
        for t in things:
            near = [p for p in places if np.linalg.norm(p - t.place) <= 0.15]
            held.append(bool(near) and brain.map.values_at(near[0])["colour"] == t.colour)
    return held, counted


def ablate(brain: Any) -> Any:
    """The twin of the ablation: the brain's map and body, and its lexicon with every word it
    heard and its synapses at zero."""
    from cadence.words import Lexicon, Words
    from cadence.words.several import Several

    arrays = brain.lexicon.arrays()
    arrays["efficacy"] = np.zeros_like(arrays["efficacy"])
    return Words(brain.map, parts=[Several(numbers=NUMBERS)], body=brain.body,
                 lexicon=Lexicon.from_arrays(brain.lexicon.settings, arrays))


def life(seed: int, *, record: Any = None, brain: Any = None, **given: Any) -> dict[str, Any]:
    """One founder's life on its script; the readings by gate. ``record`` (a list) takes every
    test question with what was said."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    events = script(seed, **settings)
    x = lexicon(seed)
    english = x.english()
    brain = make_brain(seed, **{k: v for k, v in given.items() if k not in DEFAULTS}) if (
        brain is None) else brain
    readings: dict[str, list[bool]] = {}
    held: list[bool] = []
    counted: list[bool] = []
    seconds: dict[str, float] = {}
    phase, began = "naming", time.time()
    r: world.Room | None = None
    rated: world.Room | None = None
    ablated: Any = None
    for event in events:
        kind = event[0]
        now = {"room": None, "turn": None, "name": "naming", "practice": "practice",
               "ask": "test"}[kind]
        if now is not None and now != phase:
            seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
            phase, began = now, time.time()
        if kind == "room":
            r = event[1]
            base.look_around(brain, r)
        elif kind == "turn":
            assert r is not None
            base.turn(brain, r, event[1])
        elif kind == "name":
            brain.name(event[2], event[1])
        elif kind == "practice":
            q = event[1]
            brain.practice(q.text, q.answer if len(q.answer) > 1 else q.answer[0])
        else:
            gate, q = event[1], event[2]
            assert r is not None
            if rated is not r:
                rated = r
                h, c = map_rate(brain, r)
                held += h
                counted += c
            said = brain.ask(q.text)
            ok = world.right(q, said)
            readings.setdefault(gate, []).append(ok)
            if q.about in ("out_of_view", "both"):
                readings.setdefault("S5_out_of_view", []).append(ok)
            if gate[:2] in ("S1", "S3", "S4"):
                ablated = ablated if ablated is not None else ablate(brain)
                readings.setdefault("S6_ablated", []).append(world.right(q, ablated.ask(q.text)))
            if record is not None:
                record.append({"gate": gate, "question": q.text, "form": q.form,
                               "about": q.about, "answer": q.answer, "said": said, "right": ok,
                               "english": " ".join(english.get(w, w) for w in q.text.split()),
                               "answer_english": [english.get(w, w) for w in q.answer],
                               "said_english": [english.get(w, w) for w in said]})
    seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
    out = readings_of(seed, settings, readings, seconds)
    out["S5_map_rate"] = float(np.mean(held)) if held else float("nan")
    out["S5_map_counted"] = float(np.mean(counted)) if counted else float("nan")
    out["parameters"] = int(brain.lexicon.connectome.synapses // 2)
    out["work"] = dict(brain.map.work)
    out["lessons"] = brain.lexicon.lessons
    out["words"] = len(brain.lexicon.forms)
    synapses = int(brain.map.connectome.synapses)
    out["macs"] = int(brain.lexicon.work["macs"] + synapses * (
        brain.map.work["moments"] + 2 * brain.map.work["questions"]))
    return out


def readings_of(seed: int, settings: dict[str, Any], readings: dict[str, list[bool]],
                seconds: dict[str, float]) -> dict[str, Any]:
    """A life's readings: each gate's share right and its count, and the pooled readings."""
    out: dict[str, Any] = {"seed": int(seed), **settings}
    for gate, oks in sorted(readings.items()):
        out[gate] = float(np.mean(oks)) if oks else float("nan")
        out[f"{gate}_n"] = len(oks)

    def pool(*names: str) -> float:
        oks = [ok for n in names for ok in readings.get(n, [])]
        return float(np.mean(oks)) if oks else float("nan")

    out["S1"] = pool("S1_practised", "S1_named_only")
    out["S3"] = pool("S3_practised", "S3_named_only", "S3_colour")
    out["S4"] = pool("S4_practised", "S4_named_only")
    out["seconds"] = seconds
    return out


# -- the controls: the same script as tokens (``controls.py``)


class Tokens(controls.Tokens):
    """The controls' vocabulary: the chamber's, and this rung's words."""

    def __init__(self, x: Words):
        super().__init__(x.base)
        extra = list(x.question.values()) + x.number + x.truth
        self.words += extra
        self.names += extra
        self.index = {n: i for i, n in enumerate(self.names)}


def sequences(seed: int, **given: Any) -> list[tuple[str, Any, Any]]:
    """The script as the controls' items, in the brain's order (``controls.sequences``)."""
    tok = Tokens(lexicon(seed))
    out: list[tuple[str, Any, Any]] = []
    r: world.Room | None = None
    for event in script(seed, **given):
        kind = event[0]
        if kind == "room":
            r = event[1]
        elif kind == "turn":
            assert r is not None
            r.heading = event[1]
        elif kind == "name":
            assert r is not None
            place, word = event[1], event[2]
            t = min(r.things, key=lambda th: float(np.linalg.norm(th.place - place)))
            concepts = [tok(f"kind{t.kind}"), tok(f"colour{t.colour}"),
                        tok(f"place{world.place_of(r.heading, t.place)}")]
            scene = controls.room_tokens(tok, r)
            pair = [(scene + [tok("NAME")] + concepts + [tok("SAY")], [tok(word), tok("END")]),
                    (scene + [tok("NAME"), tok(word), tok("SAY")], concepts + [tok("END")])]
            out.append(("train", pair, None))
        else:
            assert r is not None
            q = event[1] if kind == "practice" else event[2]
            prompt = controls.room_tokens(tok, r) + [tok("ASK")] + [
                tok(w) for w in q.text.split()] + [tok("SAY")]
            if kind == "practice":
                out.append(("train", [(prompt, [tok(w) for w in q.answer] + [tok("END")])],
                            None))
            else:
                out.append(("ask", event[1], (prompt, q)))
    return out


def control_life(seed: int, model: str, mode: str, parameters: int,
                 **given: Any) -> dict[str, Any]:
    """One control's life on the founder's script, trained as ``controls.run`` trains (one
    pass online, or epochs with early stopping offline); the readings by gate."""
    import torch

    torch.set_num_threads(1)
    tok = Tokens(lexicon(seed))
    items = sequences(seed, **given)
    width = controls.matched_width(model, parameters, len(tok))
    control = controls.Control(model, width, len(tok), seed, controls.RATES[mode])
    began = time.monotonic()
    train = [pairs for kind, pairs, _ in items if kind == "train"]
    if mode == "online":
        for pairs in train:
            control.update(pairs)
    else:
        flat = [p for pairs in train for p in pairs]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 328, 9]))
        order = rng.permutation(len(flat))
        aside = [flat[i] for i in order[: len(flat) // 10]]
        fit = [flat[i] for i in order[len(flat) // 10:]]
        best, best_state, since, since_halved = math.inf, None, 0, 0
        rate = controls.RATES[mode]
        for _ in range(controls.EPOCHS):
            shuffled = rng.permutation(len(fit))
            for start in range(0, len(fit), controls.BATCH):
                control.update([fit[int(i)] for i in shuffled[start: start + controls.BATCH]])
            with torch.no_grad():
                loss = float(np.mean([control.loss(aside[i: i + controls.BATCH]).item()
                                      for i in range(0, len(aside), controls.BATCH)]))
            if loss < best - 1e-6:
                best, since, since_halved = loss, 0, 0
                best_state = {k: v.clone() for k, v in control.net.state_dict().items()}
            else:
                since += 1
                since_halved += 1
                if since_halved >= controls.HALVE and rate > controls.FLOOR:
                    rate, since_halved = max(rate / 2.0, controls.FLOOR), 0
                    for group in control.opt.param_groups:
                        group["lr"] = rate
            if since >= controls.PATIENCE:
                break
        if best_state is not None:
            control.net.load_state_dict(best_state)
    trained_at = time.monotonic()
    readings: dict[str, list[bool]] = {}
    for kind, first, second in items:
        if kind != "ask":
            continue
        gate, (prompt, q) = first, second
        said = [tok.names[t] for t in control.say(prompt, tok("END"))]
        ok = world.right(q, said)
        readings.setdefault(gate, []).append(ok)
        if q.about in ("out_of_view", "both"):
            readings.setdefault("S5_out_of_view", []).append(ok)
    out = readings_of(seed, {}, readings, {"train": trained_at - began,
                                           "test": time.monotonic() - trained_at})
    out.update({"model": model, "mode": mode, "width": width,
                "parameters": controls.count(control.net), "updates": control.work["updates"],
                "macs": control.work["macs"]})
    return out


# -- the declared run and its reading

PARAMETERS = 256 * (world.KINDS + world.COLOURS + 4 + 5 + 1 + NUMBERS + 2)  # one a synapse pair
SIZES = {"matched": PARAMETERS, "large": 9 * PARAMETERS}
CONTROLS = [(m, mode, "matched") for m in ("lstm", "transformer") for mode in ("online", "offline")
            ] + [(m, "offline", "large") for m in ("lstm", "transformer")]
UNORDERED = {"counting_rooms": 0}  # the brain with every form asked from the first practice room
# the declared gates: (reading, threshold); a gate holds where the mean over the founders and at
# least ten of twelve founders reach it
GATES = {
    "S1": [("S1_practised", 0.90), ("S1_named_only", 0.90)],
    "S2": [("S2_several", 0.90)],
    "S3": [("S3_practised", 0.90), ("S3_named_only", 0.90), ("S3_colour", 0.90)],
    "S4": [("S4_practised", 0.90), ("S4_named_only", 0.90)],
    "S5": [("S5_never_taught", 0.95), ("S5_never_seen", 0.95)],
}
READINGS = ["S1", "S1_practised", "S1_named_only", "S2_several", "S3", "S3_practised",
            "S3_named_only", "S3_colour", "S4", "S4_practised", "S4_named_only",
            "S5_never_taught", "S5_never_seen", "S5_out_of_view", "S5_map_rate",
            "S5_map_counted", "one", "S6_ablated"]


def jobs(out: Path, first: int, count: int) -> list[tuple[Path, list[str]]]:
    python = sys.executable
    todo = []
    for seed in range(first, first + count):
        todo.append((out / f"brain_{seed}.json", [python, str(HERE / "several.py"), str(seed)]))
        todo.append((out / f"unordered_{seed}.json", [python, str(HERE / "several.py"), str(seed),
                                                      json.dumps(UNORDERED)]))
        for model, mode, size in CONTROLS:
            code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import several; "
                    f"print(json.dumps(several.control_life({seed}, {model!r}, {mode!r}, "
                    f"{SIZES[size]}), default=float))")
            todo.append((out / f"{model}_{mode}_{size}_{seed}.json", [python, "-c", code]))
    return todo


def run(out: Path, first: int, count: int, workers: int) -> None:
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
        for line in pool.map(one, jobs(out, first, count)):
            print(line, flush=True)


def read(out: Path) -> dict[str, Any]:
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
    verdicts: dict[str, Any] = {}
    brain = arms.get("brain", [])
    for gate, needs in GATES.items():
        held, parts = True, []
        for name, threshold in needs:
            values = np.array([life[name] for life in brain])
            reach = int((values >= threshold).sum())
            ok = bool(len(values)) and values.mean() >= threshold and reach >= min(10, len(values))
            held &= ok
            parts.append(f"{name} mean {values.mean():.3f}, {reach} of {len(values)} at "
                         f"{threshold}")
        verdicts[gate] = held
        print(f"{gate}: {'holds' if held else 'fails'} ({'; '.join(parts)})")
    gaps = [life["S5_map_rate"] - life["S5_out_of_view"] for life in brain]
    verdicts["S5_out_of_view"] = bool(gaps) and max(gaps) <= 0.02
    print(f"S5 out of view: {'holds' if verdicts['S5_out_of_view'] else 'fails'} (largest "
          f"shortfall from the map's own rate {max(gaps) if gaps else float('nan'):.3f})")
    ablated = [life["S6_ablated"] for life in brain]
    verdicts["S6"] = bool(ablated) and max(ablated) <= 0.05
    print(f"S6 ablation: the lexicon's synapses at zero answer S1, S3 and S4 at "
          f"{max(ablated) if ablated else float('nan'):.3f} at most")
    return verdicts


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 16)
    elif len(sys.argv) > 1 and sys.argv[1] == "read":
        read(Path(sys.argv[2]))
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2800
        told = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        print(json.dumps(life(seed, **told), indent=1, default=float))
