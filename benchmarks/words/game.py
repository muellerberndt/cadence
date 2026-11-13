"""The training game: words linked by pistachios (cadence#326, protocol 3).

The parrot learns a word for a thing as an African Grey does: hearing a word while it focuses on
a thing leaves a mark on their links (``Lexicon.mark``), the mark fades, and a pistachio makes the
marks of that moment lasting links (``Lexicon.confirm``). No naming lesson is given. One room of
the game, round by round:

1. **Look and listen**: the trainer holds up a thing (the parrot's focus moves to it) and says one
   of its words twice: its kind's, its colour's or its place's around the parrot.
2. **The parrot's turn**: "what is this?" ("what colour is this?", "where is this?"); the parrot
   says its best guess, at first the word it heard last; a right one earns a pistachio, a wrong
   one none, and the trainer says the word again.
3. **Understanding by acting**: the trainer says a word of a thing in the room; the parrot turns
   to one; the right one earns a pistachio.
4. **Two trainers** (from the second third of the rooms on): the parrot watches one trainer
   answer the other about a thing; a right answer is paid with a pistachio the parrot sees.
5. **Thinning out** (the last third): a right answer earns a pistachio one time in two.

Arms: ``game`` (every step paid), ``naming`` (no pistachio ever: the marks fade), ``withheld``
(no pistachio at steps 2 to 4 for a colour's word; every other word paid). The test is protocol
1's: fresh rooms of trained kinds, the rooms of unseen colours and four new kinds, each taught
by three rounds of the game.
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

DEFAULTS = {"game_rooms": 60, "rounds": 8, "test_rooms": 30, "unseen_rooms": 16,
            "new_rooms": 16}
PAUSE = 6  # moments between the steps of a round: a mark left unpaid fades before the next
ARMS = ("game", "naming", "withheld")
GROUPS = ("kind", "colour", "place")


def this_word(seed: int) -> str:
    """The founder's nonce word for "this", drawn apart from its other words."""
    return world.vocabulary(seed, {"game": ["this"]}, salt=11)["game"]["this"]


def word_of(lex: world.Lexicon, r: world.Room, t: world.Thing, group: str) -> str:
    if group == "kind":
        return lex.kind[t.kind]
    if group == "colour":
        return lex.colour[t.colour]
    return lex.place[world.place_of(r.heading, t.place)]


def question_of(lex: world.Lexicon, this: str, group: str) -> str:
    q = lex.question
    return {"kind": f"{q['what']} {q['is']} {this}",
            "colour": f"{q['what']} {q['colour']} {q['is']} {this}",
            "place": f"{q['where']} {q['is']} {this}"}[group]


def named_by(lex: world.Lexicon, r: world.Room, word: str) -> list[int]:
    """The things of a room a word names (its kind, its colour or its place)."""
    return [i for i, t in enumerate(r.things) if word in (
        lex.kind[t.kind], lex.colour[t.colour], lex.place[world.place_of(r.heading, t.place)])]


def _group(rng: np.random.Generator) -> str:
    u = rng.random()
    return "kind" if u < world.NAMING[0] else (
        "colour" if u < world.NAMING[0] + world.NAMING[1] else "place")


def _round(events: list, rng: np.random.Generator, lex: world.Lexicon, this: str,
           r: world.Room, t: world.Thing, group: str, *, stage: str,
           steps: tuple[int, ...] = (1, 2, 3, 4)) -> None:
    """One round of the game about thing ``t``: its events."""
    heading = float(rng.uniform(-math.pi, math.pi))
    events.append(("turn", heading))
    r.heading = heading
    word = word_of(lex, r, t, group)
    events += [("focus", t.place.copy()), ("listen", word), ("moment",), ("listen", word),
               ("moment",)]
    if 2 in steps:
        events += [("answer", question_of(lex, this, group), word, group, stage)]
        events += [("moment",)] * PAUSE
    if 3 in steps:
        other = r.things[int(rng.integers(len(r.things)))]
        g = _group(rng)
        said = word_of(lex, r, other, g)
        events += [("point", said, [x.place.copy() for x in r.things], named_by(lex, r, said),
                    g, stage)]
        events += [("moment",)] * PAUSE
    if 4 in steps and stage != "early":
        seen = r.things[int(rng.integers(len(r.things)))]
        g = _group(rng)
        right = word_of(lex, r, seen, g)
        pool = {"kind": lex.kind[: world.TRAINED], "colour": lex.colour, "place": lex.place}[g]
        said = right if rng.random() < 0.8 else str(rng.choice([w for w in pool if w != right]))
        events += [("watch", seen.place.copy(), said, said == right, g, stage)]
        events += [("moment",)] * PAUSE


def script(seed: int, **given: Any) -> list[tuple[Any, ...]]:
    """The founder's game and test as events, the same for every arm and the controls."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 13]))
    lex = world.lexicon(seed)
    this = this_word(seed)
    unseen = world.unseen_colours(seed)
    trained = list(range(world.TRAINED))
    events: list[tuple[Any, ...]] = []
    rooms = settings["game_rooms"]
    for index in range(rooms):
        stage = "early" if index < rooms // 3 else ("middle" if index < 2 * rooms // 3
                                                    else "thinning")
        r = world.room(rng, trained, int(rng.integers(4, 7)), avoid=unseen)
        events.append(("room", r))
        for _ in range(settings["rounds"]):
            t = r.things[int(rng.integers(len(r.things)))]
            _round(events, rng, lex, this, r, t, _group(rng), stage=stage)
    test = words.script(seed, naming_rooms=0, namings=0, practice_rooms=0,
                        test_rooms=settings["test_rooms"], unseen_rooms=settings["unseen_rooms"],
                        new_rooms=0)
    events += [e for e in test if e[0] in ("room", "ask")]
    new = list(range(world.TRAINED, world.KINDS))
    for n in new:  # each new kind: three rounds of the game about it
        r = world.room(rng, trained, 3, avoid=unseen)
        r.things.append(world.Thing(n, int(rng.integers(world.COLOURS)),
                                    words._free_place(rng, r)))
        events.append(("room", r))
        for _ in range(3):
            _round(events, rng, lex, this, r, r.things[-1], "kind", stage="middle",
                   steps=(1, 2))
    for _ in range(settings["new_rooms"]):
        r = world.room(rng, trained + new, int(rng.integers(4, 7)), avoid=unseen)
        if not set(new) & set(r.kinds()):
            r.things.append(world.Thing(int(rng.choice(new)), int(rng.integers(world.COLOURS)),
                                        words._free_place(rng, r)))
        events.append(("room", r))
        for t in r.things:
            if t.kind in new:
                events += [("ask", "W2_where", world.where(lex, r, t.kind)),
                           ("ask", "W2_colour", world.colour_of(lex, r, t.kind))]
    return events


def paid(arm: str, group: str, stage: str, rng: np.random.Generator) -> bool:
    """Whether a right answer earns a pistachio in an arm."""
    if arm == "naming" or (arm == "withheld" and group == "colour"):
        return False
    return stage != "thinning" or rng.random() < 0.5


def life(seed: int, arm: str = "game", *, lexicon: Any = None, record: Any = None,
         **given: Any) -> dict[str, Any]:
    """One founder's game in an arm; the readings. ``lexicon`` puts another lexicon in the
    brain's place (the W7 arms of ``arms.py``)."""
    if arm not in ARMS:
        raise ValueError(f"arm is one of {ARMS}")
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    brain_settings = {k: v for k, v in given.items() if k not in DEFAULTS}
    events = script(seed, **settings)
    lex = world.lexicon(seed)
    brain = words.make_brain(seed, **brain_settings)
    if lexicon is not None:
        brain.lexicon = lexicon(brain.lexicon)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 14]))
    readings: dict[str, list[bool]] = {}
    tally: dict[str, list[bool]] = {}
    seconds: dict[str, float] = {}
    began, phase = time.time(), "game"
    r: world.Room | None = None
    for event in events:
        kind = event[0]
        if kind == "ask" and phase == "game":
            seconds["game"] = time.time() - began
            readings.update(word_readings(brain, lex))
            began, phase = time.time(), "test"
        if kind == "room":
            r = event[1]
            words.look_around(brain, r)
        elif kind == "turn":
            assert r is not None
            words.turn(brain, r, event[1])
        elif kind == "focus":
            brain.focus(event[1])
        elif kind == "listen":
            brain.listen(event[1])
        elif kind == "moment":
            brain.moment()
        elif kind == "answer":
            _, question, word, group, stage = event
            right = brain.answer_focus(question, explore=True) == [word]
            tally.setdefault(f"turn_{stage}", []).append(right)
            if right and paid(arm, group, stage, rng):
                brain.reward(1.0)
            elif not right:
                brain.listen(word)  # the trainer says the word again
        elif kind == "point":
            _, word, places, right_ones, group, stage = event
            right = brain.point(word, places) in right_ones
            tally.setdefault(f"point_{stage}", []).append(right)
            brain.last = {}  # no question: a pistachio here makes links alone
            if right and paid(arm, group, stage, rng):
                brain.reward(1.0)
        elif kind == "watch":
            _, place, word, right, group, stage = event
            brain.focus(place)
            brain.listen(word)
            brain.last = {}
            if right and paid(arm, group, stage, rng):
                brain.reward(1.0)  # the pistachio the parrot sees given
        elif kind == "ask":
            gate, q = event[1], event[2]
            said = brain.ask(q.text)
            ok = world.right(q, said)
            readings.setdefault(gate, []).append(ok)
            if gate == "W5_by_place" and q.about == "several":
                readings.setdefault("W5_by_place_held", []).append(ok)
            if gate.startswith("W1"):
                readings.setdefault(f"W1_{q.form}", []).append(ok)
            if record is not None:
                record.append({"gate": gate, "question": q.text, "answer": q.answer,
                               "said": said, "right": ok})
    seconds["test"] = time.time() - began
    out = words.readings_of(seed, settings, readings, [], brain, seconds)
    out["arm"] = arm
    for name, oks in tally.items():
        out[name] = float(np.mean(oks)) if oks else float("nan")
    return out


def word_readings(brain: Any, lex: world.Lexicon) -> dict[str, list[bool]]:
    """After the game: whether each trained word, heard alone, names its concept above the
    floor (understood), and whether each concept, said, is that word (said), by group."""
    lexicon = brain.lexicon
    rest = lexicon.hear([])
    out: dict[str, list[bool]] = {}
    pairs = ([("kind", k, lex.kind[k]) for k in range(world.TRAINED)]
             + [("colour", c, lex.colour[c]) for c in range(world.COLOURS)]
             + [("place", p, lex.place[p]) for p in range(len(world.PLACE_NAMES))])
    for group, value, word in pairs:
        heard = lexicon.hear(word) - rest
        concept = lexicon.index(group, value)
        understood = bool(heard.max() > brain.floor and int(np.argmax(heard)) == concept)
        out.setdefault(f"understood_{group}", []).append(understood)
        out.setdefault(f"said_{group}", []).append(lexicon.say((group, value)) == word)
    return out


if __name__ == "__main__":
    import json

    seed, arm = int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "game"
    given = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    print(json.dumps(life(seed, arm, **given), indent=1, default=float))
