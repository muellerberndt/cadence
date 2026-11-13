"""Time in words: the records of where things were, the forecast and the words for time
(``cadence.words.episodes``, #326 W8 to W10)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadence.space.map import Map
from cadence.words.episodes import SOURCES, Clock, Records, Times, forecast


def test_the_clock_drifts() -> None:
    clock = Clock(units=64, drift=0.99, seed=3)
    contexts = [clock.now.copy()]
    for _ in range(200):
        clock.tick()
        contexts.append(clock.now.copy())
    assert clock.moments == 200
    assert all(abs(float(np.linalg.norm(c)) - 1.0) < 1e-12 for c in contexts)
    near = [float(contexts[200] @ contexts[200 - lag]) for lag in (1, 10, 100)]
    assert near[0] > near[1] > near[2] and near[0] > 0.95
    again = Clock(units=64, drift=0.99, seed=3)
    again.tick()
    assert np.array_equal(again.now, contexts[1])
    with pytest.raises(ValueError, match="drift"):
        Clock(drift=1.0)


def _code(kind: int, width: int = 6) -> np.ndarray:
    out = np.zeros(width)
    out[kind] = 1.0
    return out


def test_records_by_the_thing_and_the_time() -> None:
    records = Records(6, capacity=16, seed=1)
    assert records.recall(_code(0)) is None  # nothing written
    records.write(_code(0), (1.0, 0.0))
    for _ in range(30):
        records.moment()
    records.write(_code(1), (0.0, 1.0))
    records.write(_code(0), (-1.0, 0.0))  # the same thing left another place, later
    for _ in range(5):
        records.moment()
    place, held = records.recall(_code(0))  # type: ignore[misc]
    assert np.allclose(place, (-1.0, 0.0)) and np.array_equal(held, _code(0))
    assert np.allclose(records.recall(_code(1))[0], (0.0, 1.0))  # type: ignore[index]
    assert records.recall(_code(2)) is None  # a thing never written: another thing's record
    records.barred = True
    assert records.recall(_code(0)) is None
    assert records.work["writes"] == 3 and records.work["reads"] == 3


def _room() -> Map:
    return Map((-2.0, 2.0, -2.0, 2.0), (24, 24), units=3, values={"colour": 2})


def _look(brain: Times, things: dict[int, tuple[np.ndarray, int]], heading: float) -> None:
    """The parrot looks around in eight moments and turns back to ``heading``."""
    for i in range(9):
        brain.body.heading = heading + (i % 8) * math.pi / 4 if i < 8 else heading
        face = brain.body.heading

        def lit(points: np.ndarray, face: float = face) -> np.ndarray:
            a = np.arctan2(points[:, 1], points[:, 0]) - face
            return np.asarray(np.abs(np.angle(np.exp(1j * a))) <= math.radians(60))

        seen = [(k, p, {"colour": c}) for k, (p, c) in things.items() if lit(p[None])[0]]
        brain.moment(seen=seen, lit=lit)


def test_a_thing_carried_away_is_written_and_recalled() -> None:
    brain = Times(_room())
    things = {0: (np.array([1.2, 0.0]), 1), 1: (np.array([0.0, 1.2]), 0)}
    brain.enter()
    _look(brain, things, 0.0)
    assert brain.records.store.count == 0  # nothing moved: nothing written
    things[0] = (np.array([-1.2, 0.1]), 1)  # the teacher carries the ball behind the parrot
    _look(brain, things, 0.0)
    _look(brain, things, 0.0)
    assert brain.records.store.count == 1
    place, held = brain.records.recall(brain.code(0))  # type: ignore[misc]
    assert np.linalg.norm(place - (1.2, 0.0)) < 0.15 and held[3 + 1] == 1.0  # with its colour
    assert brain.body.place(place) == 0 and brain.body.place(things[0][0]) == 2
    found = brain.sources(("kind", 0), 2)  # where (the read of a place) is the ball?
    assert found == {"records": [(0, ("place", 0))], "map": [(0, ("place", 2))],
                     "forecast": [(0, ("place", 2))]}
    brain.enter()  # a new room: the records stay
    assert brain.records.store.count == 1


def _taught(times: bool = True) -> tuple[Times, dict[str, str]]:
    """A brain taught the words of three kinds and four places by naming, then "was" and "is"
    by questions about things the teacher carried (or, without ``times``, where things are
    with no word for time, nothing carried)."""
    rng = np.random.default_rng(0)
    brain = Times(_room(), seed=0)
    kinds, places = ["dako", "mipu", "sefa"], ["fobe", "riza", "kosa", "lume"]
    said = {"was": "tavi", "is": "nole"}

    def new_room() -> dict[int, tuple[np.ndarray, int]]:
        turn = rng.uniform(0.0, 2 * math.pi)
        return {k: (1.2 * np.array([math.cos(turn + 2.1 * k), math.sin(turn + 2.1 * k)]),
                    int(rng.integers(2))) for k in range(3)}

    for _ in range(12):  # naming
        things = new_room()
        brain.enter()
        _look(brain, things, 0.0)
        for _ in range(6):
            k = int(rng.integers(3))
            p = things[k][0]
            word = kinds[k] if rng.random() < 0.6 else places[brain.body.place(p)]
            brain.name(word, p)
    for _ in range(10):  # practice: a thing carried to the opposite side
        things = new_room()
        brain.enter()
        _look(brain, things, 0.0)
        k = int(rng.integers(3))
        was = things[k][0]
        if not times:
            brain.practice(f"wo ta {kinds[k]}", places[brain.body.place(was)])
            continue
        things[k] = (-was, things[k][1])
        _look(brain, things, 0.0)
        _look(brain, things, 0.0)
        for when, p in (("was", was), ("is", -was)):
            brain.practice(f"wo {said[when]} ta {kinds[k]}", places[brain.body.place(p)])
    return brain, {**said, **{f"kind{k}": w for k, w in enumerate(kinds)},
                   **{f"place{p}": w for p, w in enumerate(places)}}


def test_was_and_is_are_learned_and_the_past_comes_from_the_records() -> None:
    brain, words = _taught()
    when = brain.lexicon.groups["when"]
    assert int(np.argmax(brain.lexicon.hear(words["was"])[when])) == SOURCES.index("records")
    assert int(np.argmax(brain.lexicon.hear(words["is"])[when])) == SOURCES.index("map")
    things = {0: (np.array([1.2, 0.3]), 0), 1: (np.array([-0.2, 1.3]), 1),
              2: (np.array([0.1, -1.4]), 0)}
    brain.enter()
    _look(brain, things, 0.0)
    things[2] = (np.array([-1.3, -0.2]), 0)  # carried from the right to behind
    _look(brain, things, 0.0)
    _look(brain, things, 0.0)
    assert brain.ask(f"wo {words['was']} ta {words['kind2']}") == [words["place1"]]
    assert brain.ask(f"wo {words['is']} ta {words['kind2']}") == [words["place2"]]
    gains = brain.last["gains"]
    assert gains[SOURCES.index("map")] == 1.0 and gains[SOURCES.index("records")] < 0.5
    brain.records.barred = True  # the records barred: the past is not answered
    assert words["place1"] not in brain.ask(f"wo {words['was']} ta {words['kind2']}")
    brain.records.barred = False
    assert brain.ask(f"wo {words['is']} ta {words['kind0']}") == [words["place0"]]


def test_a_question_that_names_no_time_asks_the_map() -> None:
    brain, words = _taught(times=False)  # nothing moved: no lesson of time was taken
    assert brain.lexicon.counts.get(("when",)) is None
    things = {0: (np.array([1.2, 0.3]), 0), 1: (np.array([-0.2, 1.3]), 1)}
    brain.enter()
    _look(brain, things, 0.0)
    assert brain.ask(f"wo ta {words['kind1']}") == [words["place3"]]
    assert brain.last["gains"].tolist() == [0.0, 1.0, 0.0]


def test_the_forecast_of_a_map_without_physics_is_the_map() -> None:
    held = _room()
    assert forecast(held, 8) is held
    brain = Times(held)
    with pytest.raises(ValueError, match="leave"):
        Times(_room(), leave=0.01, still=0.02)
    with pytest.raises(ValueError, match="when"):
        from cadence.words import Lexicon

        Times(_room(), lexicon=Lexicon({"kind": 3}))
    assert brain.lexicon.concepts == 3 + 2 + 4 + 3 + 1 + 3


def test_things_first_seen_together_are_told_apart_by_the_physics() -> None:
    """Two things first seen in one moment take two identities in the motion records, so one
    that rolls out of view is carried on by the physics while the other is seen."""
    from cadence.space.physics import IDENT, Physics

    held = Map((-2.0, 2.0, -2.0, 2.0), (24, 24), units=3, values={"colour": 2},
               physics=Physics(prior=20.0))
    lit = lambda p: p[:, 0] > 0.0  # noqa: E731
    rng = np.random.default_rng(0)
    for _ in range(12):  # watching things roll teaches the physics
        held.reset()
        start, velocity = rng.uniform(0.4, 1.5, (2, 2)), rng.uniform(-0.03, 0.03, (2, 2))
        for t in range(40):
            places = start + t * velocity
            held.moment(seen=[(k, places[k], {"colour": k}) for k in range(2)
                              if lit(places[k][None])[0]], lit=lit)
    held.reset()
    ball, cup = np.array([0.5, 0.5]), np.array([1.2, -0.6])
    held.moment(seen=[(0, ball, {"colour": 1}), (1, cup, {"colour": 0})], lit=lit)
    idents = {k: set(held.motion[:, k, IDENT][held.motion[:, k, IDENT] > 0]) for k in (0, 1)}
    assert idents[0] and idents[1] and not idents[0] & idents[1]
    velocity = np.array([-0.04, 0.0])
    for t in range(1, 31):  # the ball rolls out of view; the cup stays in view
        place = ball + t * velocity
        held.moment(seen=([(0, place, {"colour": 1})] if lit(place[None])[0] else [])
                    + [(1, cup, {"colour": 0})], lit=lit)
    found = held.where_is(0)
    assert len(found) == 1 and np.linalg.norm(found[0] - (ball + 30 * velocity)) < 0.1
    assert held.values_at(found[0])["colour"] == 1
