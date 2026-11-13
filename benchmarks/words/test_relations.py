"""The relations chamber (cadence#327): its teacher, its script and a small life."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import relations as R  # noqa: E402

SMALL = {"naming_rooms": 3, "namings": 6, "selves": 2, "body_rooms": 2, "body_asked": 4,
         "practice_rooms": 2, "asked": 6, "test_rooms": 1, "test_asked": 1, "examples": 4}


def test_the_teacher_asks_only_clear_questions() -> None:
    events = R.script(5, **SMALL)

    def asked(script: list) -> list:
        return [(e[0], e[-1].text, e[-1].answer) for e in script if e[0] in ("practice", "ask")]

    assert asked(events) == asked(R.script(5, **SMALL))  # one script for brain and controls
    words = R.words_of(5)
    assert len(set(words.relation)) == 6 and not set(words.relation) & set(
        words.base.every())
    held = R.held_out_pairs(5, R.DEFAULTS["held_pairs"])
    new = R.held_out_word(5)
    room = None
    taught = 0
    for event in events:
        if event[0] == "room":
            room = event[1]
        if event[0] not in ("practice", "ask"):
            continue
        a = event[-1]
        assert room is not None
        things = {t.kind: t for t in room.things}
        origin = R.ORIGIN if len(a.pair) == 0 else things[a.pair[0]].place
        judged = [things[a.thing]] if a.thing is not None else [
            t for t in room.things if len(a.pair) == 0 or t.kind != a.pair[0]]
        assert all(R.clear(a.relation, origin, t.place, room.heading) for t in judged)
        if event[0] == "practice":
            if a.relation == new:  # the held-out word's own examples, after practice
                assert R.span_of(a, room) == taught % R.SPANS
                assert R.other_of(a, room) == taught % R.SPANS
                taught += 1
            else:
                kinds = room.kinds()
                assert not any(frozenset((x, y)) in held for x in kinds for y in kinds
                               if x != y)
    assert taught == SMALL["examples"]


def test_the_truth_of_a_relation() -> None:
    origin = np.zeros(2)
    assert R.holds(0, origin, np.array([1.0, 0.1]), 0.0)  # ahead: in front
    assert R.holds(3, origin, np.array([0.1, 1.0]), 0.0)  # counter-clockwise: to the left
    assert R.holds(1, origin, np.array([0.1, -1.0]), 0.0)
    assert R.holds(4, origin, np.array([0.5, 0.0]), 0.0) and R.holds(
        5, origin, np.array([1.5, 0.0]), 0.0)
    assert not R.clear(0, origin, np.array([1.0, 0.95]), 0.0)  # at a quarter's edge
    assert not R.clear(4, origin, np.array([1.05, 0.0]), 0.0)  # at near's edge


def test_a_small_life_reads_every_gate() -> None:
    out = R.life(5, **SMALL)
    for gate in ("R1_thing", "R1_you", "R2_thing", "R2_you", "R3_thing", "R3_you", "R5_new",
                 "R6_ablated"):
        assert gate in out and out[f"{gate}_n"] > 0
    assert out["parameters"] == 64 * 37  # the lexicon: one efficacy a reciprocal pair
    assert out["macs"] > 0 and out["held_word"] in R.RELATIONS
