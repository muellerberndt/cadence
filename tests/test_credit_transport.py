"""Keep the free equilibrium while transporting a motor-only signed contrast."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


def chain(backend, factor, qualified=False):
    cn = cd.Connectome.from_synapses(
        6, pre=[0, 1, 2, 2, 2, 3, 4, 5], post=[1, 2, 1, 3, 4, 2, 2, 3],
        sign=[.85, .6, .6, .5, -.3, .5, -.3, .1],
        populations={"sensory": [0], "deep": [1], "upper": [2], "motor": [3, 4], "spare": [5]},
    )
    top = (cn.pre > cn.post) & np.isin(cn.post, [1, 2])
    apical = cd.Apical(top, np.array([-1, 0, 1, -1, -1, -1]), .1)
    neural = cd.NeuralGraph(
        cn, cd.learning_neuron_model(), apical=apical,
        bias=np.array([0., .3, .4, .2, .2, .2]), backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    x = np.array([[.65, 0., 0., 0., 0., 0.], [.85, 0., 0., 0., 0., 0.]])
    cfg = cd.LearnerConfig(
        beta=1e-3, credit_transport=factor, credit_window=30,
        free_steps=160, nudged_steps=160, qualified=qualified,
    )
    return cd.Learner(neural, [3, 4], cfg), x


@pytest.mark.parametrize(("backend", "qualified"), [("cpu", False), ("cpu", True), ("torch", False)])
def test_transport_changes_the_signed_response_not_the_free_answer(backend, qualified):
    if backend != "cpu":
        pytest.importorskip(backend)
    baseline, x = chain(backend, 1., qualified)
    boosted, _ = chain(backend, 3., qualified)
    np.testing.assert_array_equal(baseline.predict(x), boosted.predict(x))
    old, report = baseline.step(x, np.array([0, 1]))
    new, changes = boosted.step(x, np.array([0, 1]))
    assert report["credit_transport_factor"] == 1.
    assert changes["credit_transport_factor"] == 3.
    np.testing.assert_allclose(old.free.activation, new.free.activation, atol=1e-12)
    assert changes["credit_deep_contrast"] > 3. * report["credit_deep_contrast"]
    assert changes["credit_transport_row_sweeps"] > 0.
    assert changes["nudged_residual"] < 2e-6 and changes["opposite_residual"] < 2e-6


def test_transport_configuration_survives_saved_continuation(tmp_path):
    brain = cd.Brain.compose(
        4, 32, slots=(2,) * 16, modules=(12, 8, 6), seed=7, backend="cpu",
        apical=.25, learning_credit_transport=3., learning_surprise=False,
    )
    path = tmp_path / "transport.npz"
    x, target = np.eye(4)[:2], np.zeros((2, 16), dtype=int)
    brain.lesson(x, target)
    brain.save(path)
    resumed = cd.Brain.load(path, backend="cpu")
    assert resumed.learner.config.credit_transport == 3.
    np.testing.assert_array_equal(brain.act(x, greedy=True), resumed.act(x, greedy=True))
    assert brain.lesson(x, target) == resumed.lesson(x, target)
    np.testing.assert_array_equal(brain.brain.efficacy, resumed.brain.efficacy)


def test_zero_force_centered_feedback_keeps_the_original_equilibrium():
    from cadence.brain import Nudge

    learner, x = chain("cpu", 3.)
    graph = learner.brain
    free = graph.equilibrate(x, budget=160, tolerance=2e-9)
    assert free.qualified.all()
    gain = graph.apical_gain(x, free.state)
    _, raw = graph._split_input(free.state.activation, *graph._apical_transports())
    expected = graph.apical.drive(raw, raw, gain)
    pull = Nudge(np.zeros(x.shape[1]), np.zeros(x.shape[1]), 0., apical_gain=3. * gain)
    centered = graph.equilibrate(
        x - 2. * expected, state=free.state, nudge=pull, budget=160, tolerance=2e-9,
    )
    assert centered.qualified.all() and centered.steps == 0


def test_transport_never_runs_during_reward_only_live():
    args = dict(apical=.2, modules=(8, 6), arousal={"youth": 8}, backend="cpu", seed=3)
    original = cd.Brain.compose(4, 2, **args)
    transport = cd.Brain.compose(4, 2, **args, learning_credit_transport=3.)
    for t in range(8):
        options = {} if t == 0 else {"reward": [float(t % 2)]}
        cue = np.eye(4)[[t % 4]]
        np.testing.assert_array_equal(original.live(cue, **options), transport.live(cue, **options))
        assert all("credit_transport" not in key for key in transport.last_learning)
    np.testing.assert_array_equal(original.brain.weights, transport.brain.weights)


def test_transport_refuses_large_movements_and_runs_the_ordinary_lesson():
    learner, x = chain("cpu", 3.)
    learner.config = replace(learner.config, beta=.5)
    _, report = learner.step(x, np.array([0, 1]))
    assert report["credit_transport_trials"] == 3.
    assert report["credit_transport_factor"] == 0.
    assert report["accepted"] == 1.
    assert learner.contrast_updates == 1


@pytest.mark.parametrize("qualified", [False, True])
def test_transport_cannot_override_a_zero_signed_sweep_budget(qualified):
    learner, x = chain("cpu", 3., qualified)
    learner.config = replace(learner.config, nudged_steps=0)
    starting = learner.brain.efficacy.copy()
    if qualified:
        with pytest.raises(cd.LearningPhaseError):
            learner.step(x, np.array([0, 1]))
        assert learner.contrast_updates == 0
    else:
        _, report = learner.step(x, np.array([0, 1]))
        assert report["nudged_steps"] == report["opposite_steps"] == 0
        assert report["credit_transport_factor"] == 0.
        assert "credit_transport_reference_sweeps" not in report
    np.testing.assert_array_equal(learner.brain.efficacy, starting)
