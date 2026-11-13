"""The words chamber's contracts: the nonce words, the script, the scoring, a short life and the
controls' sequences."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import controls  # noqa: E402
import words  # noqa: E402
import world  # noqa: E402

SHORT = {"naming_rooms": 16, "namings": 15, "practice_rooms": 10, "asked": 10, "test_rooms": 3,
         "unseen_rooms": 3, "new_rooms": 3}


def test_the_nonce_words_are_distinct_and_drawn_for_each_founder() -> None:
    a, b = world.lexicon(0), world.lexicon(1)
    every = a.kind + a.colour + a.place + list(a.question.values()) + [a.none] + a.foils
    assert len(set(every)) == len(every) == world.KINDS + world.COLOURS + 4 + 6 + 1 + 12
    assert all(set(w) <= set(world.CONSONANTS + world.VOWELS) for w in every)
    assert a.kind != b.kind and world.lexicon(0) == a
    assert a.english()[a.colour[0]] == "red"


def test_the_script_is_the_founders_own() -> None:
    first, again = words.script(5, **SHORT), words.script(5, **SHORT)
    assert [e[0] for e in first] == [e[0] for e in again]
    asked = [e for e in first if e[0] == "ask"]
    assert {e[1] for e in asked} >= {"W1_practised", "W1_named_only", "W2_where", "W3_colour",
                                     "W4_never_seen", "W4_never_taught", "W5_by_colour",
                                     "W5_by_place"}
    practised = [e[1] for e in first if e[0] == "practice"]
    lex = world.lexicon(5)
    named_only = set(lex.kind[4: world.TRAINED])
    assert not any(named_only & set(q.text.split() + q.answer) for q in practised)


def test_an_answer_is_its_words_as_a_set() -> None:
    q = world.Question("a b", "by_colour", ["x", "y"], "several")
    assert world.right(q, ["y", "x"]) and not world.right(q, ["x"])
    assert not world.right(q, ["x", "y", "y"])


def test_a_short_life() -> None:
    record: list[dict] = []
    out = words.life(3, record=record, **SHORT)
    assert out["W1"] >= 0.9 and out["W4_never_taught"] == 1.0 and out["W6_ablated"] == 0.0
    assert out["parameters"] == 256 * 26 and out["macs"] > 0
    assert all(set(r["said"]) <= set(world.lexicon(3).english()) | {""} for r in record)


def test_the_controls_read_the_same_script() -> None:
    pytest.importorskip("torch")
    items = controls.sequences(5, **SHORT)
    script = words.script(5, **SHORT)
    assert sum(k == "ask" for k, _, _ in items) == sum(e[0] == "ask" for e in script)
    tok = controls.Tokens(world.lexicon(5))
    kind, pairs, _ = next(i for i in items if i[0] == "train")
    (prompt, target), (prompt2, target2) = pairs
    assert tok.names[prompt[-1]] == "SAY" and tok.names[target[-1]] == "END"
    assert tok.names[target2[0]].startswith("kind")
    width = controls.matched_width("lstm", 6656, len(tok))
    control = controls.Control("lstm", width, len(tok), 0, 0.003)
    assert abs(controls.count(control.net) - 6656) < 300
    assert np.isfinite(control.update(pairs))


def test_the_world_grows_by_configuration() -> None:
    words_of = world.vocabulary(5, {"time": ["was", "will"], "number": ["one", "two"]}, salt=10)
    drawn = [f for group in words_of.values() for f in group.values()]
    assert len(set(drawn)) == 4 and not set(drawn) & set(world.lexicon(5).every())
    assert words_of == world.vocabulary(5, {"time": ["was", "will"], "number": ["one", "two"]},
                                        salt=10)
    r = world.room(np.random.default_rng(0), list(range(world.TRAINED)), 4, groups={"size": 3})
    assert all(set(t.values) == {"size"} and 0 <= t.values["size"] < 3 for t in r.things)
    assert all(set(v) == {"colour", "size"} for _, _, v in world.seen(r, r.heading))
