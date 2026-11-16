"""The neural graph in the native core (``cadence.core.parts.graph``): its state crosses and
comes back the same bytes on both sides, and each settle, solve and check in the core gives
what the Python gives (where the core library is built), within the rounding of the block
products' order (natively NumPy and Numba sum a dense block in their BLAS's order, the core in
order, as NumPy without BLAS does)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import cadence as cd
from cadence.brain import Brain, Nudge
from cadence.connectome import Connectome
from cadence.core import Core, _default_library, pack
from cadence.core.parts import part
from cadence.neuron import Adaptation, NeuronModel

GRAPH = part("graph")
SETTLE, SOLVE, CHECK, GAIN = (GRAPH.METHODS[k] for k in (
    "settle_batch", "equilibrate", "residual", "apical_gain"))


def deep() -> Brain:
    """A composed brain's graph with the deep lesson's apical compartment."""
    return cd.Brain.compose(6, 3, modules=(12, 12), apical=0.25, teaching="deep", seed=3,
                            recall=False).brain


def plain() -> Brain:
    """A composed brain's graph without a compartment, at a step of one (the fast solve's)."""
    return cd.Brain.compose(5, 3, modules=(10,), seed=1, recall=False).brain


def adapting() -> Brain:
    """A graph whose neurons adapt, with parallel synapses and three ranges."""
    rng = np.random.default_rng(7)
    n = 18
    pre = rng.integers(0, n, 120)
    post = rng.integers(0, n, 120)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    pre, post = np.concatenate([pre, pre[:5]]), np.concatenate([post, post[:5]])  # parallel
    sign = rng.choice([-1.0, 1.0], len(pre))
    c = Connectome(n, pre, post, np.ones(len(pre)), sign,
                   populations={"a": tuple(range(0, 6)), "b": tuple(range(6, 12))})
    model = NeuronModel(dt=0.4, slope=2.0, threshold=0.5, gain=0.3,
                        adaptation=Adaptation(tau_steps=6.0, strength=0.4), leak=0.05)
    return Brain(c, model, bias=rng.normal(0, 0.1, n))


GRAPHS = (deep, plain, adapting)
core_built = pytest.mark.skipif(not _default_library().exists(),
                                reason="the core library is not built")


@pytest.mark.parametrize("make", GRAPHS)
def test_state_round_trips_in_python(make):
    graph = make()
    state = GRAPH.state_of(graph)
    fresh = copy.copy(graph)
    GRAPH.restore(fresh, {k: np.array(v) for k, v in state.items()})
    assert pack(GRAPH.state_of(fresh)) == pack(state)


@core_built
@pytest.mark.parametrize("make", GRAPHS)
def test_state_round_trips_in_the_core(make):
    state = GRAPH.state_of(make())
    after, out = Core().call("graph", "state", state, {})
    assert not out and pack(after) == pack(state)


def same(ours: dict, theirs: dict, atol: float = 1e-12) -> None:
    assert set(ours) == set(theirs), set(ours) ^ set(theirs)
    for key, value in theirs.items():
        x, y = np.asarray(ours[key]), np.asarray(value)
        assert x.shape == y.shape and x.dtype == y.dtype, key
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=0, atol=atol, err_msg=key)
        else:
            assert np.array_equal(x, y), key


def nudges(graph: Brain, batch: int, rng: np.random.Generator) -> list[Nudge]:
    n = graph.connectome.n
    out_mask = np.zeros(n)
    out_mask[-4:] = 1.0
    groups = np.full(n, -1)
    groups[-4:-2], groups[-2:] = 0, 1
    target = np.zeros((batch, n))
    target[:, -1] = 1.0
    gain = None
    if graph.apical is not None:
        gain = rng.uniform(0.1, 0.5, (batch, len(graph.apical.members)))
    anchor_gain = np.zeros(n)
    anchor_gain[:3] = 0.2
    return [
        Nudge(target, out_mask, 0.3),
        Nudge(target[0], out_mask, 0.2, softmax_temperature=0.25, groups=groups,
              weight=rng.normal(size=batch), apical_gain=gain),
        Nudge(target, out_mask * 0.5, -0.1, softmax_temperature=np.where(groups == 1, 0.3, 0.2),
              groups=groups, anchor=rng.random(n), anchor_gain=anchor_gain),
    ]


@core_built
@pytest.mark.parametrize("make", GRAPHS)
def test_calls_in_the_core(make):
    core = Core()
    rng = np.random.default_rng(11)
    graph = make()
    n = graph.connectome.n
    state = GRAPH.state_of(graph)
    for batch in (1, 3):
        drive = rng.normal(0, 1.5, (batch, n))
        mask = (rng.random(n) > 0.1).astype(float)
        free = graph.settle_batch(drive, steps=5)
        for kwargs in ({"steps": 0}, {"steps": 0, "state": free}, {"steps": 9},
                       {"steps": 40, "tolerance": 1e-3, "state": free},
                       {"steps": 6, "mask": mask}, {"steps": 4, "mask": mask[None, :] * 0.5}):
            value = graph.settle_batch(drive, **kwargs)
            _, out = core.call("graph", "settle_batch", state, SETTLE[0](drive, **kwargs))
            same(out, SETTLE[1](value))
        for nudge in nudges(graph, batch, rng):
            for kwargs in ({"steps": 7, "state": free, "nudge": nudge},
                           {"steps": 0, "state": free, "nudge": nudge}):
                value = graph.settle_batch(drive, **kwargs)
                _, out = core.call("graph", "settle_batch", state, SETTLE[0](drive, **kwargs))
                same(out, SETTLE[1](value))
            for per_unit in (False, True):
                value = graph.residual(drive, free, nudge=nudge, per_unit=per_unit)
                _, out = core.call("graph", "residual", state,
                                   CHECK[0](drive, free, nudge=nudge, per_unit=per_unit))
                same(out, CHECK[1](value))
            kwargs = {"budget": 40, "chunk": 8, "tolerance": 1e-6, "state": free, "nudge": nudge}
            value = graph.equilibrate(drive, **kwargs)
            _, out = core.call("graph", "equilibrate", state, SOLVE[0](drive, **kwargs))
            same(out, SOLVE[1](value), atol=1e-11)
        for kwargs in ({}, {"state": free}, {"fast": True, "state": free}, {"damping": 2},
                       {"mask": mask, "budget": 30, "chunk": 7}, {"budget": 0},
                       {"budget": 60, "tolerance": 1e-9, "damping": 1}):
            value = graph.equilibrate(drive, **kwargs)
            _, out = core.call("graph", "equilibrate", state, SOLVE[0](drive, **kwargs))
            same(out, SOLVE[1](value), atol=1e-11)
            # a state the solve made settles on in the core from what it published
            settled = value.state
            later = graph.settle_batch(drive, steps=3, state=settled)
            _, out = core.call("graph", "settle_batch", state,
                               SETTLE[0](drive, steps=3, state=settled))
            same(out, SETTLE[1](later), atol=1e-11)
        for per_unit in (False, True):
            value = graph.residual(drive, free, mask=mask, per_unit=per_unit)
            _, out = core.call("graph", "residual", state,
                               CHECK[0](drive, free, mask=mask, per_unit=per_unit))
            same(out, CHECK[1](value))
        value = graph.apical_gain(drive, free)
        _, out = core.call("graph", "apical_gain", state, GAIN[0](drive, free))
        same(out, GAIN[1](value))


@core_built
def test_the_core_refuses_what_the_python_refuses():
    core = Core()
    graph = deep()
    n = graph.connectome.n
    state = GRAPH.state_of(graph)
    free = graph.settle_batch(np.zeros((1, n)), steps=2)
    for drive, kwargs in (
        (np.zeros((1, n + 1)), {}),
        (np.full((1, n), np.nan), {}),
        (np.zeros((1, n)), {"mask": np.full(n, 2.0)}),
        (np.zeros((1, n)), {"steps": -1}),
        (np.zeros((2, n)), {"state": free}),
    ):
        with pytest.raises(ValueError):
            graph.settle_batch(drive, **kwargs)
        with pytest.raises(RuntimeError):
            core.call("graph", "settle_batch", state, SETTLE[0](drive, **kwargs))
    held = Nudge(np.zeros(n), np.ones(n), 0.1, apical_gain=np.ones((1, 5)))
    with pytest.raises(ValueError):
        graph.settle_batch(np.zeros((1, n)), steps=1, nudge=held)
    with pytest.raises(RuntimeError):
        core.call("graph", "settle_batch", state,
                  SETTLE[0](np.zeros((1, n)), steps=1, nudge=held))
    with pytest.raises(RuntimeError):
        core.call("graph", "equilibrate", state, SOLVE[0](np.zeros((1, n)), chunk=0))
    bad = {**state, "efficacy": np.full_like(state["efficacy"], np.inf)}
    with pytest.raises(RuntimeError):
        core.call("graph", "state", bad, {})
