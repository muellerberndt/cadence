"""The rung of time's contracts: the time words, the script, a short life and the controls'
sequences (``times.py``, #326 W8 to W10)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import times  # noqa: E402
import world  # noqa: E402

SHORT = {"naming_rooms": 12, "namings": 15, "practice_rooms": 10, "test_rooms": 4}


def test_the_time_words_are_nonce_words_of_the_founder() -> None:
    lex, said = world.lexicon(7), times.time_words(7)
    others = set(lex.kind + lex.colour + lex.place + list(lex.question.values()) + [lex.none]
                 + lex.foils)
    assert said["is"] == lex.question["is"]
    assert said["was"] not in others and said["will"] not in others
    assert len({said["was"], said["is"], said["will"]}) == 3
    assert times.time_words(7) == said and times.time_words(8) != said


def test_the_script_carries_things_to_another_place() -> None:
    events = times.script(5, **SHORT)
    assert [e[0] for e in events] == [e[0] for e in times.script(5, **SHORT)]
    carries = [e for e in events if e[0] == "carry"]
    assert len(carries) == 2 * (SHORT["practice_rooms"] + SHORT["test_rooms"])
    asked = [e for e in events if e[0] == "ask"]
    assert {e[1] for e in asked} == {"W8_past", "W8_present", "still_where", "still_colour"}
    room = None
    for e in events:
        room = e[1] if e[0] == "room" else room
        if e[0] == "ask" and e[1] == "W8_past":
            q, extra = e[2], e[3]
            assert np.linalg.norm(extra["was"] - extra["now"]) >= times.APART
            now = world.lexicon(5).place[world.place_of(room.heading, extra["now"])]
            assert q.answer != [now]  # the past is in another place around the parrot


def test_a_short_life() -> None:
    record: list[dict] = []
    out = times.life(3, record=record, **SHORT)
    assert out["W8_past"] >= 0.75 and out["W8_barred"] <= 0.25
    assert abs(out["W8_present"] - out["W8_map_rate"]) <= 0.125
    assert out["parameters"] == times.PARAMETERS == 256 * 29 and out["macs"] > 0
    assert out["records"] >= 2 * SHORT["practice_rooms"]
    past = [r for r in record if r["gate"] == "W8_past"]
    assert all(r["sources"]["records"] for r in past)


def test_the_controls_read_the_same_script() -> None:
    pytest.importorskip("torch")
    import controls

    items = times.sequences(5, **SHORT)
    script = times.script(5, **SHORT)
    assert sum(k == "ask" for k, _, _ in items) == sum(e[0] == "ask" for e in script)
    tok = times.tokens(5)
    _, gate, (prompt, q) = next(i for i in items if i[0] == "ask")
    assert tok("MOVE") in prompt and len(prompt) <= controls.MAX_LENGTH - 2
    assert tok.names[prompt[-1]] == "SAY"
    width = controls.matched_width("transformer", times.PARAMETERS, len(tok))
    net = controls.Control("transformer", width, len(tok), 0, 0.003)
    assert abs(controls.count(net.net) - times.PARAMETERS) < 0.1 * times.PARAMETERS
