"""The several-things rung's contracts (cadence#328): the words, the rooms, the teacher's order,
the scoring, a short life and the controls' sequences."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import several  # noqa: E402
import world  # noqa: E402

SHORT = {"naming_rooms": 16, "namings": 15, "practice_rooms": 20, "asked": 14, "test_rooms": 2,
         "counting_rooms": 8}


def test_the_words_are_distinct_and_drawn_for_each_founder() -> None:
    x = several.lexicon(0)
    b = x.base
    every = (b.kind + b.colour + b.place + list(b.question.values()) + [b.none] + b.foils
             + list(x.question.values()) + x.number + x.truth)
    assert len(set(every)) == len(every)
    assert several.lexicon(0) == x and several.lexicon(1).number != x.number
    assert x.english()[x.number[2]] == "two" and x.english()[x.truth[1]] == "yes"


def test_rooms_hold_several_of_a_kind_in_distinct_colours() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = several.room(rng, list(range(8)), int(rng.integers(4, 7)), repeated=2)
        kinds = r.kinds()
        assert max(kinds.count(k) for k in kinds) >= 2
        for k in set(kinds):
            colours = [t.colour for t in r.things if t.kind == k]
            assert len(set(colours)) == len(colours) <= several.MOST
        assert max([t.colour for t in r.things].count(c) for c in range(6)) <= several.MOST
        places = np.array([t.place for t in r.things])
        apart = np.linalg.norm(places[:, None] - places[None], axis=-1)
        assert (apart[np.triu_indices(len(places), 1)] >= 0.45).all()


def test_the_teacher_asks_in_order() -> None:
    events = several.script(5, **SHORT)
    rooms: list[list[str]] = []  # the forms asked in each practice room, in order
    for e in events:
        if e[0] == "room":
            rooms.append([])
        elif e[0] == "practice":
            rooms[-1].append(e[1].form)
    practice = [r for r in rooms if r][: SHORT["practice_rooms"]]
    counting = SHORT["counting_rooms"]
    assert not any(f.startswith("there") for r in practice[:counting] for f in r)
    assert any(f.startswith("count") for r in practice[:counting] for f in r)
    assert any(f.startswith("there") for r in practice[counting:] for f in r)
    asked = {e[1] for e in several.script(5) if e[0] == "ask"}  # the full test
    assert asked >= {"S1_practised", "S1_named_only", "S2_several", "S3_practised",
                     "S3_named_only", "S3_colour", "S4_practised", "S4_named_only",
                     "S5_never_taught", "S5_never_seen"}
    named_only = set(several.lexicon(5).base.kind[4: world.TRAINED])
    assert not any(named_only & set(e[1].text.split()) for e in events if e[0] == "practice")


def test_a_count_and_a_truth_are_the_words_of_the_room() -> None:
    x = several.lexicon(2)
    ball = world.Thing(0, 1, np.array([1.0, 0.0]))
    red_ball = world.Thing(0, 0, np.array([0.0, 1.0]))
    cup = world.Thing(1, 0, np.array([-1.0, 0.0]))
    r = world.Room([ball, red_ball, cup], heading=0.0)
    assert several.count_kind(x, r, 0).answer == [x.number[2]]
    assert several.count_colour(x, r, 0).answer == [x.number[2]]
    assert several.count_both(x, r, 1, 1).answer == [x.number[0]]
    assert several.there_both(x, r, 0, 0).answer == [x.truth[1]]
    assert several.there_kind(x, r, 2).answer == [x.truth[0]]
    assert several.there_kind(x, r, -1).answer == [x.base.none]
    q = several.apart(x, r, red_ball)
    assert q.answer == [x.base.place[world.place_of(0.0, red_ball.place)]]
    assert several.where(x, r, 0).about == "several" and len(several.where(x, r, 0).answer) == 2


def test_a_short_life() -> None:
    record: list[dict] = []
    out = several.life(3, record=record, **SHORT)
    assert out["S3"] >= 0.9 and out["S5_never_taught"] == 1.0 and out["S6_ablated"] == 0.0
    assert out["parameters"] == several.PARAMETERS and out["macs"] > 0
    assert all(set(r["said"]) <= set(several.lexicon(3).english()) | {""} for r in record)


def test_the_controls_read_the_same_script() -> None:
    pytest.importorskip("torch")
    import controls

    items = several.sequences(5, **SHORT)
    script = several.script(5, **SHORT)
    assert sum(k == "ask" for k, _, _ in items) == sum(e[0] == "ask" for e in script)
    tok = several.Tokens(several.lexicon(5))
    assert all(tok(w) >= 0 for w in several.lexicon(5).number)
    width = controls.matched_width("lstm", several.PARAMETERS, len(tok))
    net = controls.Control("lstm", width, len(tok), 0, 0.003)
    assert abs(controls.count(net.net) - several.PARAMETERS) < 300
    kind, pairs, _ = next(i for i in items if i[0] == "train")
    assert np.isfinite(net.update(pairs))
