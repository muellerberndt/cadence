"""``learning_fast_settle`` (opt-in): a free solve in which a row stops at the check it qualifies
and the activation is made in arithmetic that vectorizes: the same equations, order and tolerance
in fewer neuron updates, with other bits."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import cadence as cd

pytest.importorskip("numba")


def brain(**options) -> cd.Brain:
    return cd.Brain.compose(8, 4, modules=(32,), seed=3, recall=True, recall_founder=True, episodic=False,
                            learning_surprise=True, **options)


def stream(subject: cd.Brain, moments: int = 12, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(moments):
        out.append(subject.act(rng.random((4, 8)), greedy=True))
    return out


def test_a_newborn_without_the_option_reads_as_before() -> None:
    assert cd.LearnerConfig().fast_settle is False
    assert "fast_settle" not in cd.LearnerConfig().to_dict()
    plain, off = brain(), brain(learning_fast_settle=False)
    for a, b in zip(stream(plain), stream(off), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(plain.basal_ganglia.state.v, off.basal_ganglia.state.v)
    assert "row_sweeps" not in plain.last_settlement


def test_the_solve_meets_the_tolerance_in_fewer_row_sweeps() -> None:
    subject = brain()
    graph = subject.brain
    rng = np.random.default_rng(1)
    stream(subject, 4)
    state = subject.basal_ganglia.state
    drive = subject.stimulus(rng.random((4, 8)))
    fast = graph.equilibrate(drive, state=state, budget=512, tolerance=3e-3, fast=True)
    plain = graph.equilibrate(drive, state=state, budget=512, tolerance=3e-3)
    assert fast.qualified.all()
    assert fast.row_sweeps is not None and fast.row_sweeps <= fast.steps * 4
    assert fast.row_sweeps < plain.steps * 4
    assert plain.row_sweeps is None
    assert fast.state.activity_change is None and plain.state.activity_change is not None
    # what it publishes is the rule's, and its report is what a check of the state reads
    np.testing.assert_allclose(fast.state.activation, graph.neuron_model.activation(fast.state.v),
                               atol=1e-11)
    np.testing.assert_allclose(graph.residual(drive, fast.state), fast.residual, atol=1e-11)
    # the same fixed point to about the tolerance
    tight = graph.equilibrate(drive, state=state, budget=4096, tolerance=1e-10)
    closer = graph.equilibrate(drive, state=state, budget=4096, tolerance=1e-10, fast=True)
    np.testing.assert_allclose(closer.state.v, tight.state.v, atol=1e-7)


def test_the_state_it_leaves_reads_as_any() -> None:
    subject = brain()
    graph = subject.brain
    rng = np.random.default_rng(2)
    drive = subject.stimulus(rng.random((4, 8)))
    fast = graph.equilibrate(drive, budget=256, tolerance=3e-3, fast=True)
    plain = cd.BrainState(np.array(fast.state.v), np.array(fast.state.activation),
                          np.array(fast.state.adaptation), fast.state.steps)
    np.testing.assert_allclose(graph.residual(drive, fast.state), graph.residual(drive, plain),
                               atol=1e-11)
    np.testing.assert_allclose(graph.settle_batch(drive, steps=4, state=fast.state).v,
                               graph.settle_batch(drive, steps=4, state=plain).v, atol=1e-11)
    # a second solve from it, and the fast solve from a state another solve left
    fast2 = graph.equilibrate(drive, state=fast.state, budget=64, tolerance=1e-6, fast=True)
    assert fast2.qualified.all()


def test_where_it_does_not_apply_the_ordinary_solve_runs() -> None:
    subject = cd.Brain.compose(8, 4, modules=(32,), seed=3, episodic=False, recall=False)
    graph = subject.brain
    drive = subject.stimulus(np.random.default_rng(3).random((2, 8)))
    n = graph.connectome.n
    mask = np.ones(n)
    mask[0] = 0.5
    for kw in (dict(mask=mask),):
        a = graph.equilibrate(drive, budget=64, tolerance=3e-3, fast=True, **kw)
        b = graph.equilibrate(drive, budget=64, tolerance=3e-3, **kw)
        np.testing.assert_array_equal(a.state.v, b.state.v)
        assert a.row_sweeps is None
    halved = graph._with_dt(0.5)
    a = halved.equilibrate(drive, budget=64, tolerance=3e-3, fast=True)
    np.testing.assert_array_equal(a.state.v, halved.equilibrate(drive, budget=64, tolerance=3e-3).state.v)


def test_a_brain_reports_it_and_a_checkpoint_keeps_it(tmp_path) -> None:
    subject = brain(learning_fast_settle=True)
    assert subject.describe()["genes"]["learning_fast_settle"] is True
    stream(subject, 6)
    report = subject.last_settlement
    assert report["row_sweeps"] <= report["steps"] * 4
    assert report["qualified"]
    assert subject.parameters() == brain().parameters()
    loaded = cd.Brain.load(subject.save(tmp_path / "fast.npz"))
    assert loaded.learner.config.fast_settle is True
    twin = copy.deepcopy(subject)
    x = np.random.default_rng(7).random((4, 8))
    action = subject.act(x, greedy=True)
    np.testing.assert_array_equal(twin.act(x, greedy=True), action)
    np.testing.assert_array_equal(loaded.act(x, greedy=True), action)
    np.testing.assert_allclose(loaded.basal_ganglia.state.v, subject.basal_ganglia.state.v, atol=1e-9)


def test_retune_takes_it_and_grow_and_graft_carry_it() -> None:
    subject = brain()
    subject.retune(learning_fast_settle=True)
    assert subject.learner.config.fast_settle is True
    stream(subject, 3)
    assert "row_sweeps" in subject.last_settlement
    plain = cd.Brain.compose(8, 4, modules=(32,), seed=3, episodic=False, learning_fast_settle=True, recall=False)
    assert plain.grow(recall=True, recall_founder=True).learner.config.fast_settle is True
    donor = cd.Brain.compose(8, 4, modules=(32,), seed=4, episodic=False, recall=False)
    grafted = copy.deepcopy(plain).graft(donor, ["association", "motor"])
    assert grafted.learner.config.fast_settle is True
    with pytest.raises(ValueError, match="fast_settle"):
        cd.LearnerConfig(fast_settle=1)


def test_lessons_keep_their_phases_and_learn() -> None:
    subject = brain(learning_fast_settle=True)
    rng = np.random.default_rng(5)
    x = np.eye(8)[rng.integers(0, 8, 4)]
    labels = rng.integers(0, 4, 4)
    for _ in range(40):
        subject.lesson(x, labels)
        subject.act(x, greedy=True)
    assert subject.learner.contrast_updates > 0


def test_a_graph_with_its_own_solve_keeps_it() -> None:
    from cadence.brain import Brain as NeuralGraph

    seen = []

    class Own(NeuralGraph):
        def equilibrate(self, drive, **options):
            seen.append(sorted(options))
            return super().equilibrate(drive, **options)

    subject = brain(learning_fast_settle=True)
    subject.brain.__class__ = Own
    stream(subject, 2)
    assert seen and all("fast" not in names for names in seen)
    assert "row_sweeps" not in subject.last_settlement


def test_a_spent_budget_reports_the_rows_as_they_stand() -> None:
    subject = brain()
    graph = subject.brain
    drive = subject.stimulus(np.random.default_rng(4).random((4, 8)))
    fast = graph.equilibrate(drive, budget=3, tolerance=1e-12, fast=True)
    assert fast.steps == 3
    assert not fast.qualified.any()
    np.testing.assert_allclose(graph.residual(drive, fast.state), fast.residual, atol=1e-11)
    none = graph.equilibrate(drive, budget=0, tolerance=1e-12, fast=True)
    assert none.steps == 0
    assert not none.qualified.any()
