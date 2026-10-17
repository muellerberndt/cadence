"""Contracts of the learned-read prototype."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cadence import Brain
from cadence import brain as cadence_brain
from cadence.brain import Brain as NeuralGraph

HERE = Path(__file__).resolve().parent
# the learned read enters the settle through the nudge's drive, which the NumPy solver calls
# at every step and the compiled dense kernel never does
pytestmark = pytest.mark.skipif(
    cadence_brain._FUSED,
    reason="the learned read enters the settle through the nudge's drive, which the compiled "
    "dense kernel never calls; run with CADENCE_FUSED=0",
)


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tb = _load("token_binding")
lr = _load("learned_read")
READ = {"alpha": lr.ALPHA}


def _uncoupled(seed: int):
    """The same brain with its graph returned to the plain class: no coupling anywhere."""
    brain = lr.ReadBrain.build(seed, read=dict(READ))
    graph = brain.learner.brain
    graph.__class__ = NeuralGraph
    del graph.coupling
    return brain


def test_without_read_regions_the_brain_is_the_composed_one() -> None:
    built = lr.ReadBrain.build(3)
    composed = Brain.compose(tb.INPUTS, tb.VALUES, modules=(tb.WIDTH,), lateral=tb.LATERAL, seed=3)
    np.testing.assert_array_equal(built.brain.efficacy, composed.brain.efficacy)
    assert built.coupling is None


def test_the_coupling_is_the_bilinear_exchange() -> None:
    brain = lr.ReadBrain.build(0, read=dict(READ))
    coupling = brain.coupling
    rng = np.random.default_rng(0)
    coupling.memory.reset(3)
    coupling.memory.strength = rng.random(coupling.memory.strength.shape)
    s = rng.random((3, brain.connectome.n))
    out = coupling.drive(s)
    m = coupling.memory.strength
    q, r = s[:, coupling.query], s[:, coupling.recall]
    for b in range(3):
        np.testing.assert_allclose(out[b, coupling.recall], lr.ALPHA * q[b] @ m[b])
        np.testing.assert_allclose(out[b, coupling.query], lr.ALPHA * m[b] @ r[b])
    others = np.setdiff1d(np.arange(brain.connectome.n), np.r_[coupling.query, coupling.recall])
    assert not out[:, others].any()
    coupling.rows = np.array([2, 0])
    np.testing.assert_allclose(coupling.matrix(2), m[[2, 0]])
    assert coupling.matrix(5) is None


def test_an_empty_memory_reproduces_the_uncoupled_brain() -> None:
    coupled, plain = lr.ReadBrain.build(1, read=dict(READ)), _uncoupled(1)
    trials = tb.draw(np.random.default_rng(1), 4, tb.holdout(1))
    first = tb.events(trials)[0]
    np.testing.assert_array_equal(coupled.act(first, greedy=True), plain.act(first, greedy=True))
    np.testing.assert_allclose(
        coupled.basal_ganglia.state.activation, plain.basal_ganglia.state.activation
    )
    # the half-step fallback: a budget too small for the first attempt
    coupled.reset(), plain.reset()
    coupled.coupling.memory.reset(0)
    drive = coupled.stimulus(first)
    a = coupled._equilibrate(drive, None, budget=4, tolerance=1e-12)
    b = Brain._equilibrate(plain, drive, None, budget=4, tolerance=1e-12)
    assert a.damping_halvings == b.damping_halvings == 1
    np.testing.assert_allclose(a.state.activation, b.state.activation)
    np.testing.assert_allclose(a.residual, b.residual)


def test_a_written_memory_enters_the_settle_and_its_residual() -> None:
    brain = lr.ReadBrain.build(2, read=dict(READ))
    trials = tb.draw(np.random.default_rng(2), 4, tb.holdout(2))
    seq = tb.events(trials)
    brain.reset()
    for x in seq[:-1]:
        brain.act(x, greedy=True)
    assert brain.coupling.memory.strength.any()
    drive = brain.stimulus(seq[-1])
    state = brain._qualified(drive, brain._activity(), operation="act")
    graph = brain.learner.brain
    coupled_residual = graph.residual(drive, state)
    plain_residual = NeuralGraph.residual(graph, drive, state)
    assert np.max(coupled_residual) <= brain.learner.config.tolerance
    assert np.max(plain_residual) > np.max(coupled_residual)
    recall = state.activation[:, brain.coupling.recall]
    assert recall.max() > 0


def test_lessons_keep_the_coupling_and_move_the_query_projection() -> None:
    brain = lr.ReadBrain.build(4, read=dict(READ), learning_surprise=True)
    connectome = brain.connectome
    association = np.asarray(connectome.populations["association"])
    query = np.asarray(connectome.populations["query"])
    pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
    joins = (np.isin(pre, association) & np.isin(post, query)) | (
        np.isin(pre, query) & np.isin(post, association)
    )
    assert joins.any()
    before = brain.brain.efficacy[joins].copy()
    coupling = brain.coupling
    held = tb.holdout(4)
    rng = np.random.default_rng(4)
    work = tb.new_work()
    for _ in range(3):
        assert lr.train_round(brain, tb.draw(rng, tb.BATCH, held), work)
    assert work["lesson_refusals"] == 0 and work["lessons"] == 3
    assert isinstance(brain.learner.brain, lr.CoupledGraph)
    assert brain.coupling is coupling and coupling.rows is None
    if work["rows_taught"]:
        assert np.any(brain.brain.efficacy[joins] != before)


def test_forks_carry_their_own_memory() -> None:
    brain = lr.ReadBrain.build(5, read=dict(READ))
    trials = tb.draw(np.random.default_rng(5), 3, tb.holdout(5))
    brain.reset()
    brain.act(tb.events(trials)[0], greedy=True)
    twin = tb.fork(brain)
    assert isinstance(twin.learner.brain, lr.CoupledGraph)
    assert twin.coupling is not brain.coupling
    assert twin.coupling.memory is not brain.coupling.memory
    before = brain.coupling.memory.strength.copy()
    twin.act(tb.events(trials)[1], greedy=True)
    np.testing.assert_array_equal(brain.coupling.memory.strength, before)


def test_a_heard_query_starts_as_the_present_event_and_recalls_its_fact() -> None:
    brain = lr.ReadBrain.build(6, read={"alpha": 0.5, "rule": "delta", "heard": 3.0})
    plain = lr.ReadBrain.build(6, read={"alpha": 0.5, "rule": "delta"})
    trials = tb.draw(np.random.default_rng(6), 6, tb.holdout(6))
    seq = tb.events(trials)
    for b in (brain, plain):
        b.reset()
        for x in seq[:-1]:
            b.act(x, greedy=True)
    drive = brain.stimulus(seq[-1])
    np.testing.assert_allclose(drive[:, brain.coupling.query], 3.0 * seq[-1])
    assert not plain.stimulus(seq[-1])[:, plain.coupling.query].any()
    state = brain._qualified(drive, brain._activity(), operation="act")
    recall = state.activation[:, brain.coupling.recall]
    rows = np.arange(6)
    asked = tb.coordinate("value") + tb.answers(trials)
    other = trials["values"][rows, 1 - trials["query"]]
    differs = other != tb.answers(trials)
    gap = recall[rows, asked] - recall[rows, tb.coordinate("value") + other]
    # a weak coupling discriminates; a strong one saturates on every stored fact
    assert np.mean(gap[differs] > 0) >= 0.75
