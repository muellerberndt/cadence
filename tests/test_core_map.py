"""The map of things in the native core (``cadence.core.parts.map``): its state crosses and
comes back the same bytes on both sides, and a moment (reports by every sense, values, a camera
as flags or as a function, learned physics carrying things out of view, the cells' surprise),
a question by place, the inventory, the picture, the forecast and the moves give in the core
what the Python gives (where the core library is built)."""

from __future__ import annotations

import copy
import functools

import numpy as np
import pytest

from cadence.core import Core, _default_library, pack
from cadence.core.parts import part
from cadence.space.map import Map

MAP = part("map")


def wild_map(**settings) -> Map:
    """Wild's map: four kinds, a hue as four value units, physics and surprise."""
    options = dict(units=4, values={"hue": 4}, reach=3.0, step=1.8, stride=6.0, margin=0.3,
                   physics={"noise": 0.3}, surprise=True)
    options.update(settings)
    return Map((0.0, 24.0, 0.0, 24.0), (8, 8), **options)


def hue(rng: np.random.Generator) -> np.ndarray:
    return rng.random(4)


def life(held: Map, moments: int = 70, seed: int = 0):
    """Moments of a life on ``held``: a thing that moves along a line, seen until it leaves
    the camera's half of the plane (its physics is taught by then); a thing that stays and is
    seen empty; sounds; moments with the camera as flags and with no camera. Yields each
    moment's arguments before it is lived."""
    rng = np.random.default_rng(seed)
    for t in range(moments):
        place = np.array([2.0 + 0.45 * t, 12.0 + 0.2 * np.sin(t / 3.0)])
        place = place + rng.normal(0.0, 0.05, 2)
        edge = 14.0 if t < 45 else 30.0

        def lit(points, edge=edge):
            return np.asarray(points)[:, 0] < edge

        seen = []
        if place[0] < edge:
            seen.append((1, place, {"hue": hue(rng)}) if held.values else (1, place))
        if t < 20 or 30 <= t < 34:
            seen.append((0, np.array([12.0, 20.0]) + rng.normal(0.0, 0.05, 2)))
        heard = [(2, rng.random(2) * 24.0)] if t % 7 == 3 else []
        felt = [(3, np.array([20.0, 4.0]))] if t % 13 == 6 else []
        camera = (rng.random(64) < 0.5) if t % 9 == 4 else (None if t % 17 == 8 else lit)
        yield {"seen": seen, "heard": heard, "felt": felt, "lit": camera}


@functools.cache
def maps() -> tuple[Map, ...]:
    """Maps along lives: fresh, then every few moments, on Wild's map and on maps that keep or
    clear a thing seen empty, hold no values or no physics."""
    out = []
    for held in (wild_map(), wild_map(empty="clear"), wild_map(empty="keep", values=None),
                 wild_map(physics=None, surprise=None)):
        out.append(copy.deepcopy(held))
        for k, given in enumerate(life(held)):
            held.moment(**given)
            if k % 10 == 9:
                out.append(copy.deepcopy(held))
    return tuple(out)


def test_states_cover_physics_and_doubt():
    seen = [MAP.state_of(m) for m in maps()]
    assert any(m.physics is not None and m.physics.taught for m in maps())
    assert any(s.get("motion", np.zeros((1, 1, 63)))[:, :, 11].max() > 0 for s in seen)
    assert any(s["doubt"].max() > 0.5 for s in seen)


def test_state_round_trips_in_python():
    for held in maps():
        state = MAP.state_of(held)
        fresh = copy.deepcopy(held)
        MAP.restore(fresh, {k: np.array(v) for k, v in state.items()})
        assert pack(MAP.state_of(fresh)) == pack(state)


core_built = pytest.mark.skipif(not _default_library().exists(),
                                reason="the core library is not built")


@core_built
def test_state_round_trips_in_the_core():
    core = Core()
    for held in maps():
        state = MAP.state_of(held)
        after, out = core.call("map", "state", state, {})
        assert not out and pack(after) == pack(state)


def same(ours: dict, theirs: dict) -> None:
    """Integers and flags equal, floats within 1e-9 (NumPy's own tanh and its BLAS sums differ
    from the core's in the last bits)."""
    assert sorted(ours) == sorted(theirs)
    for key, value in theirs.items():
        x, y = np.asarray(ours[key]), np.asarray(value)
        assert x.shape == y.shape and x.dtype == y.dtype, key
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-9, err_msg=key)
        else:
            assert np.array_equal(x, y), key


def called(core: Core, held: Map, method: str, *args, **kwargs):
    """``method`` in the Python on ``held`` and in the core on its state: both compared."""
    before = MAP.state_of(held)
    args, kwargs, after = MAP.calling(held, method, args, kwargs)
    value = getattr(held, method)(*args, **kwargs)
    arguments, result = MAP.METHODS[method]
    ours, out = core.call("map", method, before, {**arguments(*args, **kwargs), **after()})
    same(out, result(value))
    same(ours, MAP.state_of(held))
    return value


@core_built
@pytest.mark.parametrize("settings", [{}, {"empty": "clear"}, {"empty": "keep", "values": None},
                                      {"physics": None, "surprise": None}])
def test_lives_in_the_core(settings):
    core = Core()
    held = wild_map(**settings)
    for k, given in enumerate(life(held)):
        called(core, held, "moment", **given)
        place = held.centers[(7 * k) % held.count]
        called(core, held, "thing_at", place, body=(12.0, 12.0), heading=0.3 * k)
        if k % 5 == 0:
            called(core, held, "things")
            called(core, held, "sheet")
            called(core, held, "sheet", 1)
            called(core, held, "thing_at", None, cells=np.arange(held.count) % 3 == 0)
            called(core, held, "thing_at", place, 4.0, body=(1.0, 2.0))
            called(core, held, "moves", held.potential * 0.9)
        if k % 10 == 0 and held.physics is not None:
            called(core, held, "forecast", 3)
    assert held.physics is None or held.physics.taught


@core_built
def test_the_core_refuses_what_the_python_refuses():
    core = Core()
    held = copy.deepcopy(maps()[3])
    state = MAP.state_of(held)
    told = MAP.reports(held, seen=[(1, (3.0, 4.0))])
    with pytest.raises(RuntimeError):  # a place that is not finite
        core.call("map", "moment", state, {**told, "reports.place": np.array([[np.nan, 1.0]]),
                                           "lit.kind": np.int64(0)})
    with pytest.raises(RuntimeError):  # a kind the map does not hold
        core.call("map", "moment", state, {**told, "reports.kind": np.array([9]),
                                           "lit.kind": np.int64(0)})
    with pytest.raises(RuntimeError):  # a camera asked what the Python did not ask
        core.call("map", "moment", state, {**told, "lit.kind": np.int64(2),
                                           "lit.points": np.zeros((0, 2)),
                                           "lit.bounds": np.zeros(1, dtype=np.int64),
                                           "lit.seen": np.zeros(0, dtype=bool)})
    plain = MAP.state_of(wild_map(physics=None, surprise=None))
    with pytest.raises(RuntimeError):  # a forecast without physics
        core.call("map", "forecast", plain, {"moments": np.int64(1)})
