"""Learned physics on the map (``cadence.space.physics``)."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.space.map import Map
from cadence.space.physics import CORE, COUNT, EST, FIELDS, VEL, Physics

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
