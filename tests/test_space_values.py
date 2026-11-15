"""Values carried with a thing on the map of things (``Map(values=)``, #326)."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.space.map import Map

BALL, CUP, BOX = 0, 1, 2
RED, BLUE = 0, 1


def _room() -> Map:
    return Map((-1.0, 1.0, -1.0, 1.0), (24, 24), units=8, values={"colour": 6})


def test_two_red_things_stay_two_things() -> None:
    room = _room()
    room.moment(seen=[(BALL, [0.5, 0.5], {"colour": RED}), (CUP, [-0.5, 0.3], {"colour": RED}),
                      (BOX, [0.0, -0.6], {"colour": BLUE})])
    for _ in range(20):
        room.moment()
    assert room.width == 14 and room.values == {"colour": slice(8, 14)}
    assert [(t["kind"], t["values"]) for t in room.things()] == [
        (BALL, {"colour": RED}), (CUP, {"colour": RED}), (BOX, {"colour": BLUE})]
    red = room.where_is_value("colour", RED)
    assert len(red) == 2 and np.allclose(sorted(red.tolist()), [[-0.5, 0.3], [0.5, 0.5]],
                                         atol=1e-3)
    assert room.values_at([0.5, 0.5]) == {"colour": RED} and room.what_is_at([0.5, 0.5]) == BALL
    assert room.values_at([0.9, -0.9]) == {"colour": None}


def test_a_value_moves_with_its_thing() -> None:
    room = _room()
    room.moment(seen=[(BALL, [0.5, 0.5], {"colour": RED})])
    room.moment(heard=[(BALL, [0.7, 0.6])])  # moved unseen, a report with no colour
    assert room.values_at([0.7, 0.6]) == {"colour": RED}
    assert room.values_at([0.5, 0.5]) == {"colour": None}
    room.moment(seen=[(BALL, [0.7, 0.6], {"colour": BLUE})])  # seen again, another colour
    assert room.values_at([0.7, 0.6]) == {"colour": BLUE}
    assert len(room.where_is_value("colour", RED)) == 0
    room.moment(lit=lambda p: p[:, 0] > 0.6)  # its place seen empty: out of sight near there
    (ball,) = room.things()
    assert ball["state"] == "near" and ball["values"] == {"colour": BLUE}
    assert ball["place"][0] < 0.6
    assert np.allclose(room.where_is_value("colour", BLUE), [ball["place"]], atol=1e-6)


def test_cells_as_the_question_and_a_file() -> None:
    import tempfile
    from pathlib import Path

    room = _room()
    room.moment(seen=[(BOX, [0.0, -0.6], {"colour": BLUE}), (CUP, [0.0, 0.6], {"colour": 3})])
    lower = room.centers[:, 1] < -0.3
    assert room.what_is_at(None, cells=lower) == BOX
    assert room.values_at(None, cells=lower) == {"colour": BLUE}
    again = Map.load(room.save(Path(tempfile.mkdtemp()) / "room.npz"))
    assert again.values == room.values and again.parts()[0]["values"] == {"colour": 6}
    assert [(t["kind"], t["values"]) for t in again.things()] == [
        (CUP, {"colour": 3}), (BOX, {"colour": BLUE})]


def test_values_are_refused_where_they_do_not_fit() -> None:
    with pytest.raises(ValueError, match="values="):
        Map((-1.0, 1.0, -1.0, 1.0)).moment(seen=[(0, [0.0, 0.0], {"colour": 1})])
    with pytest.raises(ValueError, match="no value"):
        _room().moment(seen=[(0, [0.0, 0.0], {"colour": 6})])
    with pytest.raises(ValueError, match="no value"):
        _room().moment(seen=[(0, [0.0, 0.0], {"size": 0})])
    with pytest.raises(ValueError, match="count of values"):
        Map((-1.0, 1.0, -1.0, 1.0), values={"colour": 0})
    plain = Map((-1.0, 1.0, -1.0, 1.0))
    assert plain.width == plain.units and "values" not in plain.parts()[0]
    plain.moment(seen=[(0, [0.0, 0.0])])
    assert "values" not in plain.things()[0]


def test_a_moving_thing_keeps_its_colour_out_of_sight() -> None:
    """A red ball rolls behind an occluder: a taught physics carries it on its line out of view,
    and its colour goes with it, on the map and in the map's forecast."""
    from cadence.space.physics import Physics

    physics = Physics()  # as if it had watched a world where things seldom stop or turn
    physics.stats[:] = [0.005e6, 0.005e6, 1e6, 0.5, 1.0, 0.25, 1.0, 1e5, 1e5 * 0.015**2, 1e5]
    physics.fit()
    room = Map((-1.0, 1.0, -0.2, 1.4), values={"colour": 6}, physics=physics)
    before_occluder = lambda points: np.asarray(points)[:, 0] < 0.0  # noqa: E731
    rng = np.random.default_rng(0)
    truth = np.array([-0.8, 0.6]) + np.arange(70)[:, None] * np.array([0.015, 0.0])
    for t in range(70):
        seen = ([(BALL, truth[t] + rng.normal(0.0, 0.01, 2), {"colour": RED})]
                if truth[t, 0] < 0.0 else [])
        room.moment(seen=seen, lit=before_occluder)
    (ball,) = [t for t in room.things() if t["kind"] == BALL]
    assert ball["state"] == "moving" and ball["values"] == {"colour": RED}
    assert np.linalg.norm(ball["place"] - truth[-1]) < 0.06  # carried on its line
    assert room.values_at(ball["place"]) == {"colour": RED}
    assert room.what_is_at(ball["place"]) == BALL
    ahead = room.forecast(10)
    core = min(ahead.where_is(BALL), key=lambda p: np.linalg.norm(p - (truth[-1] + [0.15, 0.0])))
    assert ahead.values_at(core) == {"colour": RED} and ahead.what_is_at(core) == BALL
    assert np.allclose(room.where_is_value("colour", RED), [ball["place"]], atol=0.03)


def _tuned(value: float, many: int) -> np.ndarray:
    """A sense's tuned cells for a value in 0 to 1: a code of ``many`` amounts."""
    return np.exp(-0.5 * ((value - np.linspace(0.0, 1.0, many)) * (many - 1)) ** 2)


def test_a_code_is_read_back_whole_with_the_place_in_the_bodys_frame() -> None:
    """A value given as a code (a sense's tuned cells) rides on the thing's place and comes back
    as it was reported; ``thing_at`` reads the thing whole, its place from where the body
    points."""
    room = Map((-2.0, 2.0, -2.0, 2.0), (32, 32), units=8, values={"hue": 12, "size": 4})
    hue, other = _tuned(0.3, 12), 0.05 * _tuned(0.8, 12)  # a pale thing: a weak code
    room.moment(seen=[(BALL, [0.7, -0.4], {"hue": hue, "size": 2}),
                      (CUP, [0.7, 0.1], {"hue": other})])
    for _ in range(10):
        room.moment()
    ball = room.thing_at([0.7, -0.4], heading=np.pi / 2)
    assert ball is not None and ball["kind"] == BALL
    assert np.allclose(ball["values"]["hue"], hue, atol=1e-3)
    assert np.allclose(ball["values"]["size"], [0, 0, 1, 0], atol=1e-3)
    assert np.allclose(ball["place"], [0.7, -0.4], atol=1e-3)
    assert np.isclose(ball["distance"], np.hypot(0.7, 0.4), atol=1e-3)
    assert np.isclose(ball["bearing"], np.arctan2(-0.4, 0.7) - np.pi / 2, atol=1e-3)  # right
    cup = room.thing_at([0.7, 0.1], body=(0.7, -0.9))
    assert cup is not None and cup["kind"] == CUP
    assert np.allclose(cup["values"]["hue"], other, atol=1e-3)
    assert not cup["values"]["size"].any()  # a group the report named none of
    assert np.isclose(cup["bearing"], np.pi / 2, atol=1e-3) and np.isclose(cup["distance"], 1.0,
                                                                         atol=1e-3)
    assert room.thing_at([-1.5, 1.5]) is None
    assert room.values_at([0.7, -0.4]) == {"hue": int(np.argmax(hue)), "size": 2}
    # the cells of a kind as the question: the thing wherever the map holds it
    again = room.thing_at(None, cells=room.sheet(BALL).ravel() > 0)
    assert again is not None and np.allclose(again["values"]["hue"], hue, atol=1e-3)


def test_a_code_goes_with_its_thing_unseen() -> None:
    hue = _tuned(0.6, 12)
    room = Map((-2.0, 2.0, -2.0, 2.0), (32, 32), units=8, values={"hue": 12})
    room.moment(seen=[(BALL, [0.5, 0.5], {"hue": hue})])
    room.moment(heard=[(BALL, [0.8, 0.3])])  # moved unseen: a sound names its place alone
    moved = room.thing_at([0.8, 0.3])
    assert moved is not None and np.allclose(moved["values"]["hue"], hue, atol=1e-3)
    assert room.thing_at([0.5, 0.5]) is None
    # a place seen empty: the code goes onto the cells out of sight with the thing
    lit = lambda points: np.asarray(points)[:, 0] < 0.85  # noqa: E731
    room = Map((-2.0, 2.0, -2.0, 2.0), (32, 32), units=8, values={"hue": 12})
    room.moment(seen=[(BALL, [0.8, 0.0], {"hue": hue})], lit=lit)
    room.moment(lit=lit)
    (ball,) = room.things()
    assert ball["state"] == "near"
    near = room.thing_at(None, cells=np.isin(np.arange(room.count), ball["cells"]))
    assert near is not None and np.allclose(near["values"]["hue"], hue, atol=1e-3)


def test_a_code_is_refused_by_what_is_wrong_with_it() -> None:
    room = Map((-1.0, 1.0, -1.0, 1.0), (24, 24), units=8, values={"hue": 12})
    for bad in (np.ones(11), 2.0 * np.ones(12), -np.ones(12), np.full(12, np.nan)):
        with pytest.raises(ValueError, match="amounts in 0 to 1"):
            room.moment(seen=[(BALL, [0.0, 0.0], {"hue": bad})])
    with pytest.raises(ValueError, match="amounts in 0 to 1"):
        room.moment(seen=[(BALL, [0.0, 0.0], {"shade": np.ones(3)})])
