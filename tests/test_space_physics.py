"""Learned physics on the map (``cadence.space.physics``)."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.space.map import Map
from cadence.space.physics import CORE, COUNT, EST, FIELDS, HIDDEN, IDENT, VEL, Physics

PLANE = (-1.0, 1.0, -0.2, 1.4)
INSIDE = (np.array([-0.98, -0.18]), np.array([0.98, 1.38]))


def everywhere(points: np.ndarray) -> np.ndarray:
    return np.ones(len(points), dtype=bool)


def left_half(points: np.ndarray) -> np.ndarray:
    """A camera that sees the plane left of x = 0."""
    return np.asarray(points)[:, 0] < 0.0


def line_of(steps: int, start=(-0.8, 0.6), velocity=(0.015, 0.0), noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(steps)[:, None]
    truth = np.asarray(start) + t * np.asarray(velocity)
    return truth, truth + rng.normal(0.0, noise, truth.shape)


def taught(rest: float = 0.005, leave: float = 0.005) -> Physics:
    """A physics as if it had watched a world with these shares (its statistics set so)."""
    ph = Physics()
    exposure = 1e6
    ph.stats[:] = [rest * exposure, leave * exposure, exposure, 0.5, 1.0, 0.25, 1.0, 1e5,
                   1e5 * 0.015**2, 1e5]
    ph.fit()
    return ph


def test_the_line_reads_a_velocity_and_a_surprise_ends_it():
    ph = Physics(learning=False)
    truth, reports = line_of(40)
    rec = None
    for r in reports:
        rec = ph.see(rec, r, 1.0, INSIDE)
    assert rec is not None and rec.shape == (FIELDS,)
    assert np.allclose(rec[VEL], [0.015, 0.0], atol=0.002)
    assert np.linalg.norm(rec[EST] - truth[-1]) < 0.01
    # it stops: the line ends within a few reports and reads a velocity near zero
    for _ in range(12):
        rec = ph.see(rec, truth[-1] + np.random.default_rng(1).normal(0, 0.01, 2), 1.0, INSIDE)
    assert rec[COUNT] < 15
    assert np.linalg.norm(rec[VEL]) < 0.005


def test_expectation_maximisation_reads_a_rest_share():
    """Lessons from lines that come to rest at a known share a moment: the estimate finds it."""
    rng = np.random.default_rng(3)
    ph = Physics(prior=0.0)
    rest = 0.02
    for _ in range(3000):
        k = int(rng.integers(1, 31))
        v = np.array([0.015, 0.0])
        stop = rng.geometric(rest)  # the moment it comes to rest
        went = min(stop - 1, k)
        r = np.array([-0.5, 0.6]) + went * v + rng.normal(0, 0.01, 2)
        ph.watch(np.array([[-0.5, 0.6]]), v[None], np.array([30.0]), r, np.array([k]), INSIDE)
    assert ph.estimate[0] == pytest.approx(rest, rel=0.35)
    assert ph.estimate[1] < 0.01


def test_an_untaught_physics_leaves_the_map_as_it_is():
    plain = Map(PLANE)
    with_physics = Map(PLANE, physics=Physics(learning=False))
    truth, reports = line_of(60)
    for t in range(60):
        seen = [(0, reports[t])] if t < 40 else []
        plain.moment(seen=seen, lit=left_half)
        with_physics.moment(seen=seen, lit=left_half)
        assert np.array_equal(plain.potential, with_physics.potential)
        assert np.array_equal(plain.clock, with_physics.clock)
    assert not with_physics.physics.taught


def test_a_taught_physics_carries_a_thing_out_of_view_along_its_line():
    held = Map(PLANE, physics=taught())
    truth, reports = line_of(70)
    for t in range(70):
        seen = [(0, reports[t])] if truth[t, 0] < 0.0 else []
        held.moment(seen=seen, lit=left_half)
    (entry,) = [e for e in held.things() if e["kind"] == 0]
    assert entry["state"] == "moving"
    assert np.allclose(entry["velocity"], [0.015, 0.0], atol=0.003)
    # the map's answer follows the hand into the dark; the map without physics holds it back
    plain = Map(PLANE)
    for t in range(70):
        plain.moment(seen=[(0, reports[t])] if truth[t, 0] < 0.0 else [], lit=left_half)
    error = np.linalg.norm(entry["place"] - truth[-1])
    still = min(np.linalg.norm(p - truth[-1]) for p in plain.where_is(0))
    assert error < 0.06 < still


def test_sight_takes_what_it_sees_and_the_parts_keep_their_sum():
    held = Map(PLANE, physics=taught())
    truth, reports = line_of(50)
    for t in range(50):
        held.moment(seen=[(0, reports[t])] if t < 30 else [], lit=left_half if t < 30 else None)
    whole = float(np.maximum(held.potential[:, 0], 0).sum())
    # the camera now sees all of the plane right of x = 0.1: what moved there is taken away
    # (a cell under the floor is no part of a thing: the sum is kept to a part in a thousand)
    held.moment(lit=lambda p: np.asarray(p)[:, 0] > 0.1)
    assert float(np.maximum(held.potential[:, 0], 0).sum()) == pytest.approx(whole, rel=1e-3)
    assert not (held.sheet(0).ravel()[held.centers[:, 0] > 0.2] > 1e-6).any()


def test_a_forecast_moves_on_and_leaves_the_map_as_it_is():
    held = Map(PLANE, physics=taught())
    truth, reports = line_of(30)
    for t in range(30):
        held.moment(seen=[(0, reports[t])], lit=everywhere)
    before = {k: v.copy() for k, v in held.arrays().items()}
    lessons = held.physics.lessons
    ahead = held.forecast(10)
    for name, value in held.arrays().items():
        assert np.array_equal(value, before[name])
    assert held.physics.lessons == lessons and held.physics.learning
    (place,) = [e["place"] for e in ahead.things() if e["kind"] == 0]
    assert place[0] > truth[-1, 0] + 0.05  # ahead of where it was last seen
    with pytest.raises(ValueError):
        Map(PLANE).forecast(1)


def test_a_thing_moves_with_its_whole_code():
    codes = np.zeros((2, 6))
    codes[0, [0, 1, 2]] = 1.0
    codes[1, [3, 4, 5]] = 1.0
    held = Map(PLANE, codes=codes, physics=taught())
    truth, reports = line_of(30)
    for t in range(30):
        held.moment(seen=[(0, reports[t])], lit=everywhere)
    ahead = held.forecast(15)
    (entry,) = ahead.things()
    assert entry["kind"] == 0 and entry["state"] == "moving"
    assert ahead.what_is_at(entry["place"], within=0.1) == 0


def test_a_map_file_keeps_what_the_physics_learned(tmp_path):
    held = Map(PLANE, physics=taught())
    truth, reports = line_of(30)
    for t in range(30):
        held.moment(seen=[(0, reports[t])], lit=everywhere)
    path = held.save(tmp_path / "map.npz")
    again = Map.load(path)
    assert again.physics is not None
    assert again.physics.describe() == held.physics.describe()
    assert np.array_equal(again.motion, held.motion)
    assert np.allclose(again.forecast(8).where_is(0), held.forecast(8).where_is(0))


def test_physics_settings_are_checked():
    with pytest.raises(ValueError):
        Physics(line=2)
    assert Map(PLANE, physics=True).physics is not None
    assert Map(PLANE, physics={"line": 20}).physics.line == 20
    assert Map(PLANE).physics is None
    assert "motion" not in Map(PLANE).arrays()
    rec = np.zeros(FIELDS)
    assert rec[CORE + 7] == 0


# -- several things, other senses and the map's own rules (the review of #335)


def run(held: Map, steps: int, things: dict, lit, *, hidden_after=None, senses=None):
    """Things on lines ``(start, velocity)`` by kind (a list for two of a kind), reported by the
    camera where ``lit`` sees them and until ``hidden_after`` moments (by index)."""
    rng = np.random.default_rng(7)
    flat = [(kind, i, line) for kind, lines in things.items()
            for i, line in enumerate(lines if isinstance(lines, list) else [lines])]
    truth = {}
    for t in range(steps):
        reports = {"seen": [], "heard": [], "felt": []}
        for kind, i, (start, velocity) in flat:
            at = np.asarray(start) + t * np.asarray(velocity)
            truth[(kind, i)] = at
            sense = (senses or {}).get((kind, i), "seen")
            limit = (hidden_after or {}).get((kind, i))
            shown = limit is None or t < limit
            if sense != "seen" and shown:
                reports[sense].append((kind, at))
            elif shown and (lit is None or bool(lit(at[None])[0])):
                reports["seen"].append((kind, at + rng.normal(0.0, 0.01, 2)))
        held.moment(lit=lit, **reports)
    return truth


def test_two_things_first_seen_in_one_moment_are_two_things():
    held = Map(PLANE, physics=taught())
    truth = run(held, 75, {0: ((-0.8, 0.3), (0.012, 0.0)), 1: ((-0.8, 0.9), (0.012, 0.004))},
                left_half)
    idents = {int(i) for i in np.unique(held.motion[:, :, IDENT]) if i > 0}
    assert len(idents) == 2
    entries = {e["kind"]: e for e in held.things()}
    for kind in (0, 1):
        assert entries[kind]["state"] == "moving"
        assert np.linalg.norm(entries[kind]["place"] - truth[(kind, 0)]) < 0.08
    assert abs(entries[1]["velocity"][1] - 0.004) < 0.002 and abs(entries[0]["velocity"][1]) < 0.002


def test_a_report_of_one_thing_leaves_another_carried_thing_of_its_kind():
    """A hand moves out of view while a second hand of the same kind is still reported."""
    held = Map(PLANE, physics=taught())
    lines = {0: [((-0.8, 0.3), (0.012, 0.0)), ((0.5, 1.2), (0.0, 0.0))]}
    truth = run(held, 60, lines, None, hidden_after={(0, 0): 30})
    places = held.where_is(0)
    assert len(places) >= 2  # the hand at rest in view, and the one carried on
    assert min(np.linalg.norm(p - truth[(0, 0)]) for p in places) < 0.08
    assert min(np.linalg.norm(p - truth[(0, 1)]) for p in places) < 0.02


def test_a_thing_seen_at_rest_is_the_maps_to_hold():
    """At rest, then no longer reported: with physics the map answers as it does without."""
    for seed in range(12):
        rng = np.random.default_rng(seed)
        plain, held = Map(PLANE), Map(PLANE, physics=taught())
        for t in range(48):
            seen = [(0, np.array([-0.05, 0.6]) + rng.normal(0.0, 0.01, 2))] if t < 40 else []
            for m in (plain, held):
                m.moment(seen=seen, lit=left_half)
        assert np.array_equal(plain.potential, held.potential)
        assert [e["state"] for e in plain.things()] == [e["state"] for e in held.things()]


def test_a_place_another_sense_names_takes_no_part_in_the_line():
    """Seen and felt at once (the arm's own hand): the line and its lessons are the camera's."""
    alone, both = Physics(prior=0.0), Physics(prior=0.0)
    a, b = Map(PLANE, physics=alone), Map(PLANE, physics=both)
    rng = np.random.default_rng(2)
    for t in range(100):
        at = np.array([-0.6 + 0.012 * t, 0.6])
        seen = [(0, at + rng.normal(0.0, 0.01, 2))]
        a.moment(seen=seen, lit=everywhere)
        b.moment(seen=seen, felt=[(0, at + rng.normal(0.0, 0.04, 2))], lit=everywhere)
    assert both.lessons == alone.lessons > 0
    assert np.array_equal(both.stats, alone.stats)


def test_a_thing_that_has_come_to_rest_is_no_longer_moving():
    held = Map(PLANE, physics=taught(rest=0.02))
    run(held, 30, {0: ((-0.8, 0.6), (0.015, 0.0))}, None)
    states = []
    for _ in range(600):
        held.moment()
        states.append(held.things()[0]["state"])
    (entry,) = held.things()
    # moving while half of it or more moves, then in doubt at rest, with no velocity
    assert states[0] == "moving" and entry["state"] == "near" and "velocity" not in entry
    assert states.index("near") < 100 and "moving" not in states[states.index("near"):]
    # under a thousandth of it still moved: all of it is at rest, and the map's to hold
    assert not (held.motion[:, :, HIDDEN] > 0).any()


def test_keep_and_clear_are_honoured_for_a_thing_the_physics_carries():
    lines = {0: ((-0.8, 0.3), (0.012, 0.0))}
    kept = Map(PLANE, empty="keep", physics=taught())
    run(kept, 30, lines, everywhere)
    for _ in range(5):
        kept.moment(lit=everywhere)  # no report, the whole plane in sight
    (entry,) = kept.things()
    assert entry["place"] is not None  # sight takes nothing away
    cleared = Map(PLANE, empty="clear", physics=taught())
    run(cleared, 30, lines, everywhere)
    cleared.moment(lit=everywhere)
    assert [e["state"] for e in cleared.things()] == ["not known where"]
    # vanished in plain sight a step from a shadow: the map's near rule decides, as without physics
    beside = {0: ((-0.1, 0.2), (0.0, 0.012))}
    near, plain = Map(PLANE, physics=taught()), Map(PLANE)
    for m in (near, plain):
        run(m, 30, beside, left_half)
        m.moment(lit=left_half)
    assert [e["state"] for e in plain.things()] == ["near"]
    assert [e["state"] for e in near.things()] == ["near"]
    assert np.allclose(near.things()[0]["place"], plain.things()[0]["place"])


def test_a_frozen_physics_stays_frozen_in_its_file(tmp_path):
    held = Map(PLANE, physics=taught())
    held.physics.learning = False
    again = Map.load(held.save(tmp_path / "frozen.npz"))
    assert again.physics.learning is False
    held.physics.learning = True
    assert Map.load(held.save(tmp_path / "learning.npz")).physics.learning is True


def test_a_carried_thing_leaves_other_kinds_as_they_are():
    """A thing of another kind at rest out of view, under a carried thing's path."""
    held, plain = Map(PLANE, physics=taught()), Map(PLANE)
    for m in (held, plain):
        m.moment(seen=[(1, (0.33, 0.3))], lit=everywhere)
    lines = {0: ((-0.6, 0.3), (0.012, 0.0))}
    for m in (held, plain):
        run(m, 110, lines, left_half)
    one = {e["kind"]: e for e in held.things()}[1]
    other = {e["kind"]: e for e in plain.things()}[1]
    assert one["state"] == other["state"] == "held"
    assert np.allclose(one["place"], other["place"]) and one["since"] == other["since"]


def test_a_carried_thing_that_comes_back_elsewhere_is_one_thing():
    """Hidden while moving, then seen far from every part of it: no part of it is left behind."""
    held = Map(PLANE, physics=taught())
    run(held, 60, {0: ((-0.8, 0.3), (0.012, 0.0))}, None, hidden_after={(0, 0): 30})
    assert held.things()[0]["state"] == "moving"
    held.moment(seen=[(0, (0.6, 1.2))])
    (entry,) = held.things()
    assert entry["state"] == "seen" and np.allclose(entry["place"], [0.6, 1.2], atol=0.001)
    assert len(held.where_is(0)) == 1
