"""An opt-in starting state for qualified nudged phases: where settling starts, not where it ends."""

import numpy as np
import pytest

import cadence as cd


def learner(recovery=None, *, tolerance=1e-9, budget=4096, seed=3, hidden=12, **config):
    connectome = cd.layered(6, hidden, 3, seed=seed, density=1.0)
    graph = cd.NeuralGraph(connectome, cd.learning_neuron_model())
    cfg = cd.LearnerConfig(
        qualified=True, damping=3, free_steps=budget, nudged_steps=budget,
        tolerance=tolerance, **config,
    )
    return cd.Learner(graph, connectome.populations["output"], cfg, recovery=recovery)


def lesson(seed=0, rows=5):
    rng = np.random.default_rng(seed)
    return rng.random((rows, 6 + 12 + 3)) * np.r_[np.ones(6), np.zeros(15)], rng.integers(0, 3, rows)


def parameters(model):
    return model.brain.efficacy.copy(), model.brain.bias.copy()


def test_default_learner_has_no_recovery_and_an_unchanged_report():
    drive, labels = lesson()
    plain = learner()
    assert plain.recovery is None
    _, report = plain.step(drive, labels)
    assert "recovery_row_seam_passes" not in report
    assert "recovery" not in plain.to_dict()


def test_recovery_requires_qualified_teaching_and_a_recovery_start():
    connectome = cd.layered(6, 12, 3, seed=3)
    graph = cd.NeuralGraph(connectome, cd.learning_neuron_model())
    finite = cd.Learner(graph, connectome.populations["output"], recovery=cd.RecoveryStart())
    drive, labels = lesson()
    before = parameters(finite)
    with pytest.raises(ValueError, match="qualified"):
        finite.step(drive, labels)
    np.testing.assert_array_equal(parameters(finite)[0], before[0])
    with pytest.raises(ValueError, match="RecoveryStart"):
        cd.Learner(graph, connectome.populations["output"], recovery=object())
    for bad in ({"decay": 1.0}, {"ridge": -1.0}, {"fit": 1}):
        with pytest.raises(ValueError):
            cd.RecoveryStart(**bad)


def test_same_lessons_reach_the_same_parameters_within_the_tolerance():
    plain, started = learner(), learner(cd.RecoveryStart())
    for k in range(6):
        drive, labels = lesson(k)
        _, base = plain.step(drive, labels)
        _, cand = started.step(drive, labels)
        assert cand["qualified"] == 1.0
        assert cand["recovery_row_seam_passes"] == 2 * 2 * started.recovery.depth * len(labels)
    for a, b in zip(parameters(plain), parameters(started), strict=True):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-7)


def test_any_start_settles_to_the_same_qualified_contrast():
    model = learner()
    drive, labels = lesson(7)
    target = model.targets(labels)
    free = model.free(drive)
    # A deliberately wrong channel: random gains move the start, not the equilibrium.
    recovery = cd.RecoveryStart()
    recovery._bind(model)
    recovery._moments["ff"][:] = 1.0
    recovery._moments["fy"][:] = np.random.default_rng(1).normal(0, 3, free.v.shape[1])
    states = {}
    for name in ("default", "random"):
        out = []
        for sign in (1.0, -1.0):
            nudge = model.nudge_for(target, sign * model.config.beta)
            start = free if name == "default" else recovery.start(model, free, nudge)
            assert (start is free) == (name == "default")
            phase = model._qualified_phase(drive, start, 4096, nudge)
            assert phase.qualified.all()
            out.append(phase.state)
        states[name] = model.contrast(free, *out)
    np.testing.assert_allclose(states["random"][0], states["default"][0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(states["random"][1], states["default"][1], rtol=0, atol=1e-6)


def test_start_differs_from_the_free_state_and_moves_toward_the_settled_phase():
    model = learner(cd.RecoveryStart(fit=False))
    drive, labels = lesson(2)
    free = model.free(drive)
    nudge = model.nudge_for(model.targets(labels), model.config.beta)
    start = model.recovery.start(model, free, nudge)
    settled = model._qualified_phase(drive, free, 4096, nudge).state
    shells = model.recovery.shells()
    assert set(np.unique(shells)) == {-1, 0, 1}
    assert np.all(shells[model.output_index] == 0)
    moved = np.abs(start.activation - free.activation).max(axis=0)
    assert np.all(moved[shells < 0] == 0)  # inputs never receive the nudge
    assert np.all(moved[shells >= 0] > 0)
    before = np.abs(free.activation - settled.activation).sum()
    after = np.abs(start.activation - settled.activation).sum()
    assert after < before


def test_gains_fit_the_settled_records_never_the_rendered_start():
    drive, labels = lesson(4)
    model = learner()
    free = model.free(drive)
    nudge = model.nudge_for(model.targets(labels), model.config.beta)
    settled = model._qualified_phase(drive, free, 4096, nudge).state
    fresh, skewed = cd.RecoveryStart(), cd.RecoveryStart()
    fresh._bind(model)
    skewed._bind(model)
    skewed._moments["ff"][:] = 1.0
    skewed._moments["fy"][:] = np.random.default_rng(2).normal(0, 3, free.v.shape[1])
    held = {k: skewed._moments[k].copy() for k in ("ff", "fy")}
    fresh.observe(model, free, settled, nudge)
    skewed.observe(model, free, settled, nudge)
    for name in ("ff", "fy"):  # the evidence is the same whatever the current gains render
        np.testing.assert_allclose(
            skewed._moments[name] - skewed.decay * held[name], fresh._moments[name],
            rtol=1e-12, atol=1e-15,
        )
    assert fresh.observed == skewed.observed == 1
    copy = learner(cd.RecoveryStart(fit=False))
    copy.step(drive, labels)
    np.testing.assert_array_equal(copy.recovery.gains(), np.ones(copy.brain.connectome.n))
    report = copy.recovery.fidelity()
    assert [row["shell"] for row in report] == [0, 1]
    for row in report:
        assert row["gains"] == pytest.approx(row["copy"])


def test_a_refused_lesson_leaves_parameters_and_recovery_statistics_unchanged():
    model = learner(cd.RecoveryStart())
    drive, labels = lesson(5)
    model.step(drive, labels)
    held = {k: v.copy() for k, v in model.recovery._moments.items()}
    before = parameters(model)
    with pytest.warns(RuntimeWarning, match="nudged_steps"):
        model.config = cd.LearnerConfig(
            qualified=True, damping=0, free_steps=4096, nudged_steps=1, tolerance=1e-12,
        )
    with pytest.raises(cd.LearningPhaseError):
        model.step(*lesson(6))
    for name, value in held.items():
        np.testing.assert_array_equal(model.recovery._moments[name], value)
    for x, y in zip(parameters(model), before, strict=True):
        np.testing.assert_array_equal(x, y)


def test_composed_brain_accepts_a_recovery_start():
    from dataclasses import replace

    brain = cd.Brain.compose(inputs=4, actions=2, modules=(8,), seed=7)
    brain.learner.config = replace(
        brain.learner.config, qualified=True, damping=3, free_steps=512, nudged_steps=512,
        tolerance=1e-7,
    )
    brain.learner.recovery = cd.RecoveryStart()
    observation = np.array([[1.0, 0.0, 0.0, 0.0]])
    _, report = brain.learner.step(brain.stimulus(observation, memory=False), np.array([0]))
    assert report["qualified"] == 1.0
    assert brain.learner.recovery.observed == 2
