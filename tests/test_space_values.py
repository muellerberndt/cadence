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
