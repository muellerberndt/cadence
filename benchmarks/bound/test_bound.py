"""The bound chamber's instruments (cadence#342): the map's read of the attended thing against
the eyes' report, and the store's read of a word alone, on a newborn brain. The tests need a
checkout of the meaning chamber's branch (``lane.py``) and are skipped without one."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
try:
    import lane
except ImportError:
    pytest.skip("no checkout of lane-0100-meaning beside the repository",
                allow_module_level=True)
import completion  # noqa: E402
import mapped  # noqa: E402

nt = lane.night


def test_the_maps_read_of_a_thing_is_what_the_eyes_report_of_it() -> None:
    from cadence.meaning import Eyes

    eyes, rng = Eyes(slots=5, surfaces=4), np.random.default_rng(0)
    me = mapped.MapEyes(eyes)
    for _ in range(4):
        room = nt.room(rng, 4, set())
        me.reset()
        me.look([(t.kind, t.place, room.seen_of(t)) for t in room.things])
        for t in room.things:
            t.focus = True
            assert np.allclose(me.glimpse(t.kind, room.heading), eyes.slot(room.seen_of(t)),
                               atol=2e-3)
            t.focus = False


def test_a_hidden_thing_is_held_and_follows_a_sound() -> None:
    from cadence.meaning import Eyes

    eyes, rng = Eyes(slots=5, surfaces=4), np.random.default_rng(1)
    me = mapped.MapEyes(eyes)
    room = nt.room(rng, 3, set())
    x, rest = room.things[0], room.things[1:]
    x.focus = True
    whole = eyes.slot(room.seen_of(x))
    me.look([(t.kind, t.place, room.seen_of(t)) for t in room.things])
    in_view = [(t.kind, t.place, room.seen_of(t)) for t in rest]
    for _ in range(5):  # behind a screen: nothing of it is reported, its place is unlit
        me.look(in_view, hidden=[x.place])
    assert np.allclose(me.glimpse(x.kind, room.heading), whole, atol=2e-3)
    was = x.place.copy()
    x.place = nt.free_place(rng, [t.place for t in room.things])
    me.look(in_view, hidden=[was, x.place], heard=[(x.kind, x.place)])  # it moved, unseen
    assert np.allclose(me.glimpse(x.kind, room.heading), eyes.slot(room.seen_of(x)), atol=2e-3)
    assert not np.allclose(me.glimpse(x.kind, room.heading), whole, atol=2e-3)  # its place


def test_a_word_alone_reads_the_records_of_the_things_it_names(tmp_path: Path) -> None:
    """On a newborn brain: the sparse mixed code of a word heard alone finds the records of
    the things named by that word, their form comes back, and a population the records hold
    otherwise stays silent. The read needs no trained synapse."""
    from cadence.longterm import LongTermMemory

    p = nt.Parrot(lane.SEED, dict(lane.GENOME), streams=32)
    p.brain.save(tmp_path / "brain.npz")
    ch = completion.Chamber(tmp_path)
    rng = np.random.default_rng(7)
    named = [(r, t.kind) for r, t in lane.introspect.scenes(ch.p, rng, 96, ch.held, test=False)]
    ch.stores = {"code": LongTermMemory(width=ch.p.mixed, value_width=ch.w + ch.ear,
                                        capacity=256, cosine=True, revise=0.9999)}
    for chunk, m in completion.batches(named):
        ch.live([r for r, _ in chunk], [[None, None, w] for _, w in chunk], rng,
                write=[True] * m + [False] * (32 - m))
    assert ch.stores["code"].count == 96
    empty = nt.Room([], heading=0.0)
    for word in ("apple", "cup", "key"):
        ch.live([empty] * 32, [[None, None, word]] * 32, rng, key="code", heads=5, after=1)
        assert all(named[slot][1] == word for slot in ch.told[0]) and len(ch.told[0]) == 5
        back = ch.returned[0]
        assert all(back[ch.lines[f"shape {k}"]].any() for k in range(6))
        assert np.allclose(back[ch.lines["word"]], ch.p.forms(word))
        assert not back[ch.lines["direction"]].any()  # the things lay in every direction


def test_a_heard_name_draws_the_focus_to_the_thing_it_names(tmp_path: Path) -> None:
    """The born rule on a newborn brain: at the moment a thing's name is heard, the focus is
    on the thing of that name the map holds; a question that says no name draws no focus. The
    rule needs no trained synapse. With records of fewer things a recalled form is a coarser
    prototype: at 160 things a ball's name draws the focus to an apple in one room of 32."""
    import named

    p = nt.Parrot(lane.SEED, dict(lane.GENOME), streams=32)
    p.brain.save(tmp_path / "brain.npz")
    ch = named.Named(tmp_path, 320, 5, np.random.default_rng(7))
    ch.draw = True
    rng = np.random.default_rng(3)
    rooms = named.rooms_of_three(rng, ch.held, 32)
    quiet = np.zeros(32, dtype=bool)
    asked = [r.things[0].kind for r in rooms]
    exs = [nt.make_exchange(r, nt.Question(["where", "is", "the", kind], ["front"], "direction"),
                            "turn", rng) for r, kind in zip(rooms, asked, strict=True)]
    for t in range(3):  # "where is the": no name heard
        ch.p.observe(exs, t, [None] * 32, quiet, rng)
    assert ch.drawn == [None] * 32
    x = ch.p.observe(exs, 3, [None] * 32, quiet, rng)  # the name
    drawn = [d == kind for d, kind in zip(ch.drawn, asked, strict=True)]
    assert all(d is None or d == kind for d, kind in zip(ch.drawn, asked, strict=True))
    assert np.mean(drawn) >= 0.9
    for i in np.flatnonzero(drawn)[:4]:  # the glimpse is the map's read of the named thing
        thing = rooms[i].things[0]
        thing.focus = True
        assert np.allclose(x[i, : ch.w], ch.eyes.slot(rooms[i].seen_of(thing)), atol=2e-3)
        thing.focus = False
