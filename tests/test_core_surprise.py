"""The surprise and the focus in the native core (``cadence.core.parts.surprise`` and
``orienting``): their state crosses and comes back the same bytes on both sides, and each
method in the core gives what the Python gives (where the core library is built)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cadence.core import Core, _default_library, pack
from cadence.core.parts import part
from cadence.surprise import Orienting, Surprise

SURPRISE, ORIENTING = part("surprise"), part("orienting")


def surprises() -> list[Surprise]:
    """Surprises along a life: fresh, after moments of one and of several streams, some
    streams not carried, a moment that teaches nothing, and after a reset."""
    rng = np.random.default_rng(5)
    held = Surprise(7, tolerance=1.5, floor=0.03, adapt=0.4, recover=0.02)
    out = [copy.deepcopy(held)]
    for streams in (1, 1, 3, 3, 2, 40):
        sizes = rng.exponential(0.2, (streams, 7)) * rng.choice([1.0, 10.0], (streams, 7))
        carried = rng.random(streams) < 0.7 if streams > 1 else None
        held.observe(sizes if streams > 1 else sizes[0], carried=carried,
                     learn=streams != 2)
        out.append(copy.deepcopy(held))
    held.reset(2)
    out.append(copy.deepcopy(held))
    return out


def orientings() -> list[Orienting]:
    """Competitions along a life: no focus, a focus taken, held, renewed where it moves, lost
    under ``least``, places inhibited and released."""
    rng = np.random.default_rng(9)
    held = Orienting(hold=0.7, least=0.4, reach=0.3, release=0.8)
    out = [copy.deepcopy(held)]
    cells = rng.random((16, 2))
    for k in range(30):
        s = rng.exponential(0.3, 16) * (rng.random(16) < 0.3)
        named = (rng.random(3) * (k % 4 == 0), rng.random((3, 2)))
        held.attend({"map": (s, cells), "named": named})
        out.append(copy.deepcopy(held))
    return out


@pytest.mark.parametrize("module,objects", [(SURPRISE, surprises), (ORIENTING, orientings)])
def test_state_round_trips_in_python(module, objects):
    for held in objects():
        state = module.state_of(held)
        fresh = copy.deepcopy(held)
        module.restore(fresh, {k: np.array(v) for k, v in state.items()})
        assert pack(module.state_of(fresh)) == pack(state)


def test_states_cover_focus_and_inhibition():
    seen = [ORIENTING.state_of(h) for h in orientings()]
    assert any(bool(s["focused"]) for s in seen) and any(not bool(s["focused"]) for s in seen)
    assert any(len(s["inhibited.dim"]) > 1 for s in seen)
    assert any(int(s["focus.held"]) > 1 for s in seen)


core_built = pytest.mark.skipif(not _default_library().exists(),
                                reason="the core library is not built")


@core_built
@pytest.mark.parametrize("name,objects", [("surprise", surprises), ("orienting", orientings)])
def test_state_round_trips_in_the_core(name, objects):
    core = Core()
    for held in objects():
        state = part(name).state_of(held)
        after, out = core.call(name, "state", state, {})
        assert not out and pack(after) == pack(state)


def same(ours: dict, theirs: dict) -> None:
    assert list(ours) == list(theirs)
    for key, value in theirs.items():
        x, y = np.asarray(ours[key]), np.asarray(value)
        assert x.shape == y.shape and x.dtype == y.dtype, key
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-12, err_msg=key)
        else:
            assert np.array_equal(x, y), key


@core_built
def test_surprise_calls_in_the_core():
    core = Core()
    rng = np.random.default_rng(2)
    for held in surprises():
        for method, args, kwargs in (
            ("observe", (rng.exponential(0.3, 7),), {}),
            ("observe", (rng.exponential(0.3, (4, 7)),),
             {"carried": np.array([True, False, True, True])}),
            ("observe", (rng.exponential(0.3, (3, 7)),), {"learn": False}),
            ("moment", (), {}),
            ("reset", (3,), {}),
        ):
            before = SURPRISE.state_of(held)
            theirs = copy.deepcopy(held)
            arguments, result = SURPRISE.METHODS[method]
            value = getattr(theirs, method)
            value = value(*args, **kwargs) if callable(value) else value
            after, out = core.call("surprise", method, before, arguments(*args, **kwargs))
            same(out, result(value))
            same(after, SURPRISE.state_of(theirs))


@core_built
def test_attend_in_the_core():
    core = Core()
    rng = np.random.default_rng(4)
    cells = rng.random((16, 2))
    for held in orientings():
        for _ in range(4):
            sources = {"map": (rng.exponential(0.4, 16) * (rng.random(16) < 0.4), cells),
                       "senses": (rng.random(5), rng.random((5, 2)))}
            before = ORIENTING.state_of(held)
            theirs = copy.deepcopy(held)
            focus = theirs.attend(sources)
            after, out = core.call("orienting", "attend", before,
                                   ORIENTING.METHODS["attend"][0](sources))
            same(out, ORIENTING.METHODS["attend"][1](focus))
            same(after, ORIENTING.state_of(theirs))
            held = theirs
        before = ORIENTING.state_of(held)
        held.reset()
        after, out = core.call("orienting", "reset", before, {})
        assert not out
        same(after, ORIENTING.state_of(held))


@core_built
def test_the_core_refuses_what_the_python_refuses():
    core = Core()
    surprise, orienting = surprises()[2], orientings()[3]
    with pytest.raises(ValueError):
        core.call("surprise", "observe", SURPRISE.state_of(surprise),
                  {"sizes": np.array([np.nan] * 7), "learn": np.bool_(True)})
    with pytest.raises(ValueError):
        core.call("surprise", "observe", SURPRISE.state_of(surprise),
                  {"sizes": np.zeros(6), "learn": np.bool_(True)})
    with pytest.raises(ValueError):
        core.call("orienting", "attend", ORIENTING.state_of(orienting),
                  {"surprise": np.array([np.inf]), "places": np.zeros((1, 2)),
                   "sense": np.zeros(1, dtype=np.int64)})
    bad = {**SURPRISE.state_of(surprise), "config.floor": np.float64(0.0)}
    with pytest.raises(ValueError):
        core.call("surprise", "state", bad, {})
