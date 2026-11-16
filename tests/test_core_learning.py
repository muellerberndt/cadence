"""The learner, the actor and critic and the maturity in the native core
(``cadence.core.parts.learner``, ``actor``, ``maturity``): their state crosses and comes back the
same bytes on both sides, and every call a composed brain's life makes of them gives in the core
what it gives in the Python (where the core library is built), within the rounding of the block
products' and the critic's dot products' order (natively NumPy sums them in its BLAS's order)."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

import cadence as cd
from cadence.core import Core, Refused, _default_library, pack
from cadence.core.parts import part

LEARNER, ACTOR, MATURITY = part("learner"), part("actor"), part("maturity")


def brain(seed: int = 3) -> Any:
    """A composed brain as Wild composes one: the deep lesson's compartment, the recall memory
    with two heads, the carried routes' gains and a maturity, aroused throughout its youth."""
    return cd.Brain.compose(6, 3, modules=(12, 12), apical=0.25, teaching="deep", seed=seed,
                            maturity=True, recall_heads=2, recall_capacity=8,
                            arousal={"youth": 40})


def live(held: Any, moments: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for t in range(moments):
        held.live(rng.random((1, 6)), reward=None if t == 0 else [float(rng.random() - 0.3)])


@contextmanager
def recording(module: Any) -> Iterator[list[tuple[Any, ...]]]:
    """Every call of the part's methods with its state before, its arguments, what it returned
    and its state after, as ``kernels/core/check/replay.py`` records them."""
    kept: list[tuple[Any, ...]] = []
    saved = {}
    for name, (arguments, result) in module.METHODS.items():
        method = saved[name] = getattr(module.CLASS, name)

        def recorded(self: Any, *args: Any, _m: Any = method, _n: str = name, _a: Any = arguments,
                     _r: Any = result, **kwargs: Any) -> Any:
            before = module.state_of(self)
            value = _m(self, *args, **kwargs)
            kept.append((_n, before, _a(*args, **kwargs), _r(value), module.state_of(self)))
            return value
        setattr(module.CLASS, name, recorded)
    try:
        yield kept
    finally:
        for name, method in saved.items():
            setattr(module.CLASS, name, method)


def same(ours: dict[str, Any], theirs: dict[str, Any], atol: float = 1e-11) -> None:
    assert set(ours) == set(theirs), set(ours) ^ set(theirs)
    for key, value in theirs.items():
        x, y = np.asarray(ours[key]), np.asarray(value)
        assert x.shape == y.shape, key
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=0, atol=atol, err_msg=key)
        else:
            assert np.array_equal(x, y.astype(x.dtype)), key


def test_states_round_trip_in_python():
    held = brain()
    for moments in (0, 1, 6):
        live(held, moments, seed=moments)
        actor = held.basal_ganglia
        for module, obj, place in ((ACTOR, actor, "basal_ganglia"),
                                   (LEARNER, actor.learner, "learner"),
                                   (MATURITY, actor.maturity, "maturity")):
            state = module.state_of(obj)
            fresh = copy.deepcopy(held).basal_ganglia
            target = {"basal_ganglia": fresh, "learner": fresh.learner,
                      "maturity": fresh.maturity}[place]
            module.restore(target, {k: np.array(v) for k, v in state.items()})
            assert pack(module.state_of(target)) == pack(state), place


core_built = pytest.mark.skipif(not _default_library().exists(),
                                reason="the core library is not built")


@core_built
def test_states_round_trip_in_the_core():
    core = Core()
    held = brain()
    for moments in (0, 1, 6):
        live(held, moments, seed=moments)
        actor = held.basal_ganglia
        for name, module, obj in (("actor", ACTOR, actor), ("learner", LEARNER, actor.learner),
                                  ("maturity", MATURITY, actor.maturity)):
            state = module.state_of(obj)
            after, out = core.call(name, "state", state, {})
            assert not out and pack(after) == pack(state), name


@core_built
@pytest.mark.parametrize("name,module", [("actor", ACTOR), ("learner", LEARNER),
                                         ("maturity", MATURITY)])
def test_a_life_s_calls_in_the_core(name, module):
    core = Core()
    held = brain(seed=5)
    with recording(module) as kept:
        live(held, 14, seed=2)
    methods = {entry[0] for entry in kept}
    assert {"act", "learn"} <= methods if name == "actor" else methods
    for method, before, args, theirs_out, theirs_after in kept:
        after, out = core.call(name, method, before, args)
        same(out, theirs_out)
        same(after, theirs_after)


@core_built
def test_the_core_refuses_what_the_python_refuses():
    core = Core()
    held = brain()
    live(held, 3)
    actor = held.basal_ganglia
    state = ACTOR.state_of(actor)
    n = actor.n
    for args in (
        ACTOR.METHODS["act"][0](np.zeros((1, n + 1))),
        ACTOR.METHODS["act"][0](np.full((1, n), np.nan)),
        ACTOR.METHODS["act"][0](np.zeros((1, n)), temperature=-1.0),
    ):
        with pytest.raises(Refused):
            core.call("actor", "act", state, args)
    actor._pending = None
    with pytest.raises(RuntimeError):
        actor.learn(np.zeros(1), np.zeros(1, bool), np.zeros((1, n)))
    with pytest.raises(Refused):
        core.call("actor", "learn", ACTOR.state_of(actor),
                  ACTOR.METHODS["learn"][0](np.zeros(1), np.zeros(1, bool), np.zeros((1, n))))
    learner = LEARNER.state_of(actor.learner)
    with pytest.raises(Refused):
        core.call("learner", "apply", learner,
                  LEARNER.METHODS["apply"][0](np.full(actor.synapses, np.nan), np.zeros(n)))
    with pytest.raises(Refused):
        core.call("learner", "targets", learner, LEARNER.METHODS["targets"][0](np.array([99])))
