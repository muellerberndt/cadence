"""Skipping the first attempts of the damping schedule: a different path, the same equations."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd
from cadence.nudged_settle import seam_copy


def orbit_graph():
    """Uniform motor inhibition: the full step falls into a period-two orbit."""
    pre, post = np.where(~np.eye(36, dtype=bool))
    connectome = cd.Connectome.from_synapses(36, pre=pre, post=post, sign=np.full(len(pre), -0.5))
    return cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1))


def composed(settle=None, tolerance=1e-8):
    brain = cd.Brain.compose(inputs=6, actions=36, modules=(10,), seed=5, lateral=-0.5)
    brain.learner.config = replace(
        brain.learner.config, qualified=True, damping=3, free_steps=8192, nudged_steps=8192,
        tolerance=tolerance,
    )
    brain.learner.nudged_settle = settle
    return brain


def lessons(brain, count=5):
    rng = np.random.default_rng(11)
    for _ in range(count):
        yield brain.stimulus(rng.random((3, 6)), memory=False), rng.integers(0, 36, 3)


def parameters(learner):
    return learner.brain.efficacy.copy(), learner.brain.bias.copy()


def test_first_halving_zero_is_the_released_schedule():
    graph = orbit_graph()
    drive = np.full((2, 36), 0.2)
    base = graph.equilibrate(drive, budget=256, tolerance=3e-3, damping=3)
    same = graph.equilibrate(drive, budget=256, tolerance=3e-3, damping=3, first_halving=0)
    assert (base.steps, base.damping_halvings, base.residual_checks) == (
        same.steps, same.damping_halvings, same.residual_checks)
    np.testing.assert_array_equal(base.state.v, same.state.v)
    with pytest.raises(ValueError, match="first_halving"):
        graph.equilibrate(drive, damping=1, first_halving=2)
    with pytest.raises(ValueError, match="first_halving"):
        graph.equilibrate(drive, damping=3, first_halving=-1)


def test_skipping_the_orbit_settles_the_same_equations_in_fewer_sweeps():
    graph = orbit_graph()
    drive = np.full((2, 36), 0.2)
    base = graph.equilibrate(drive, budget=1024, tolerance=1e-8, damping=3)
    fast = graph.equilibrate(drive, budget=1024, tolerance=1e-8, damping=3, first_halving=2)
    assert base.qualified.all() and fast.qualified.all()
    assert fast.damping_halvings >= 2
    assert fast.steps < base.steps
    # The residual is the original dt=1 model's, recomputed independently here.
    assert graph.residual(drive, fast.state).max() <= 1e-8
    np.testing.assert_allclose(fast.state.activation, base.state.activation, atol=1e-7)


def test_default_learner_has_no_nudged_settle_and_an_unchanged_report():
    brain = composed()
    assert brain.learner.nudged_settle is None
    drive, labels = next(lessons(brain))
    _, report = brain.learner.step(drive, labels)
    assert not any(key.startswith("nudged_settle") for key in report)


def test_nudged_settle_requires_qualified_teaching_and_a_valid_schedule():
    brain = cd.Brain.compose(inputs=6, actions=36, modules=(10,), seed=5)
    brain.learner.nudged_settle = cd.NudgedSettle()
    drive, labels = next(lessons(brain))
    before = parameters(brain.learner)
    with pytest.raises(ValueError, match="qualified"):
        brain.learner.step(drive, labels)
    np.testing.assert_array_equal(parameters(brain.learner)[0], before[0])
    deep = composed(cd.NudgedSettle(first_halving=4))
    with pytest.raises(ValueError, match="damping"):
        deep.learner.step(drive, labels)
    for bad in ({"first_halving": -1}, {"first_halving": True}, {"probe": -1.0}):
        with pytest.raises(ValueError):
            cd.NudgedSettle(**bad)
    with pytest.raises(ValueError, match="NudgedSettle"):
        cd.Learner(brain.brain, list(brain.motor_index), nudged_settle=object())


def test_same_lessons_reach_the_same_parameters_in_fewer_nudged_sweeps():
    plain, fast = composed(), composed(cd.NudgedSettle())
    base_sweeps = fast_sweeps = 0.0
    for (drive, labels), _ in zip(lessons(plain), lessons(fast), strict=True):
        _, base = plain.learner.step(drive, labels)
        _, cand = fast.learner.step(drive, labels)
        assert cand["qualified"] == 1.0
        base_sweeps += base["nudged_steps"] + base["opposite_steps"]
        fast_sweeps += (cand["nudged_steps"] + cand["opposite_steps"]
                        + cand["nudged_settle_probe_steps"] + cand["nudged_settle_discarded_steps"])
    for a, b in zip(parameters(plain.learner), parameters(fast.learner), strict=True):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-6)
    assert fast_sweeps < base_sweeps


def test_a_failed_probe_falls_back_to_the_default_phase_exactly():
    plain, guarded = composed(tolerance=1e-6), composed(cd.NudgedSettle(probe=0.0), 1e-6)
    drive, labels = next(lessons(plain))
    _, base = plain.learner.step(drive, labels)
    learned, cand = guarded.learner.step(drive, labels)
    assert cand["nudged_settle_fallbacks"] == 2.0
    assert cand["nudged_settle_discarded_steps"] > 0
    assert cand["total_steps"] == (base["total_steps"] + cand["nudged_settle_probe_steps"]
                                   + cand["nudged_settle_discarded_steps"])
    for a, b in zip(parameters(plain.learner), parameters(guarded.learner), strict=True):
        np.testing.assert_array_equal(b, a)


def test_a_refused_lesson_leaves_parameters_unchanged():
    brain = composed(cd.NudgedSettle())
    drive, labels = next(lessons(brain))
    before = parameters(brain.learner)
    with pytest.warns(RuntimeWarning, match="nudged_steps"):
        brain.learner.config = replace(brain.learner.config, nudged_steps=1, tolerance=1e-12)
    with pytest.raises(cd.LearningPhaseError):
        brain.learner.step(drive, labels)
    for a, b in zip(parameters(brain.learner), before, strict=True):
        np.testing.assert_array_equal(a, b)


def test_seam_copy_moves_only_neurons_the_nudge_reaches():
    connectome = cd.layered(6, 12, 3, seed=3, density=1.0)
    graph = cd.NeuralGraph(connectome, cd.learning_neuron_model())
    learner = cd.Learner(graph, connectome.populations["output"])
    drive = np.zeros((2, connectome.n))
    drive[:, connectome.populations["input"]] = 0.5
    free = learner.free(drive)
    start = seam_copy(learner, free, learner.nudge_for(learner.targets(np.array([0, 2])), 0.1))
    moved = np.abs(start.activation - free.activation).max(axis=0)
    assert np.all(moved[list(connectome.populations["input"])] == 0)
    assert np.all(moved[list(connectome.populations["output"])] > 0)
    assert np.all(moved[list(connectome.populations["hidden"])] > 0)
