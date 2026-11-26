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


def _refusals(caught) -> list[str]:
    return [str(w.message) for w in caught if "credit_transport admitted no factor" in str(w.message)]


def test_a_run_of_lessons_without_a_factor_is_reported_and_said_once():
    import warnings

    learner, x = chain("cpu", 3.)
    labels = np.array([0, 1])
    refusing = replace(learner.config, beta=.5, credit_window=3)  # movement too large to admit
    learner.config = refusing
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reports = [learner.step(x, labels)[1] for _ in range(5)]
    assert [r["credit_transport_factor"] for r in reports] == [0.] * 5
    assert [r["credit_transport_refused"] for r in reports] == [1.] * 5
    said = _refusals(caught)
    assert len(said) == 1  # at the third lesson of the run, and not again while it lasts
    assert "in 3 lessons in a row" in said[0] and "none of the 3 candidate solves" in said[0]
    assert "within 160 signed sweeps" in said[0]
    # an admitted factor ends the run
    learner.config = replace(refusing, beta=1e-3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, admitted = learner.step(x, labels)
    assert admitted["credit_transport_factor"] > 0. and admitted["credit_transport_refused"] == 0.
    assert not _refusals(caught)
    # and the next run of the window's length is said again
    learner.config = refusing
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = [learner.step(x, labels)[1] for _ in range(3)]
    assert [r["credit_transport_refused"] for r in again] == [1.] * 3 and len(_refusals(caught)) == 1


def test_a_report_is_a_function_of_its_lesson_and_a_learner_without_transport_holds_no_field():
    import warnings

    # two learners at different points of a run report the same lesson alike: the run's
    # length is not in the report, so a loaded brain's reports are its saver's
    late, x = chain("cpu", 3.)
    early, _ = chain("cpu", 3.)
    for learner in (late, early):
        learner.config = replace(learner.config, beta=.5, credit_window=30)
    labels = np.array([0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            late.step(x, labels)
        early.brain = late.brain  # the same parameters, a run of none behind it
        early.velocity, early.velocity_bias = late.velocity.copy(), late.velocity_bias.copy()
        early.contrast_updates, early.updates = late.contrast_updates, late.updates
        after_three, after_none = late.step(x, labels)[1], early.step(x, labels)[1]
    assert after_three.keys() == after_none.keys()
    assert after_three["credit_transport_refused"] == after_none["credit_transport_refused"] == 1.
    assert after_three == pytest.approx(after_none, rel=1e-9, abs=1e-12)
    admitting, x = chain("cpu", 3.)
    admitting.config = replace(admitting.config, credit_window=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reports = [admitting.step(x, labels)[1] for _ in range(4)]
    assert all(r["credit_transport_factor"] == 3. for r in reports)
    assert all(r["credit_transport_refused"] == 0. for r in reports) and not _refusals(caught)
    plain, x = chain("cpu", 0.)
    _, report = plain.step(x, labels)
    assert not any(name.startswith("credit_transport") for name in report)


def test_the_warning_names_a_reference_that_did_not_qualify_and_a_graph_without_a_branch():
    import warnings

    # a reference budget of two sweeps cannot reach the reference's tolerance
    learner, x = chain("cpu", 3.)
    learner.config = replace(learner.config, free_steps=2, credit_window=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reports = [learner.step(x, np.array([0, 1]))[1] for _ in range(2)]
    assert all(r["credit_transport_reference_refused"] == 1. for r in reports)
    assert all(r["credit_transport_refused"] == 1. for r in reports)
    said = _refusals(caught)
    assert len(said) == 1 and "the zero-force reference did not qualify" in said[0]
    assert "against 2e-07 within 2 sweeps" in said[0]
    # without an apical compartment the setting has no branch to act on; the run is the
    # founder window of 15 lessons where no credit window is set
    cn = cd.Connectome.from_synapses(
        4, pre=[0, 1, 2, 3], post=[1, 2, 1, 1], sign=[.8, .6, .6, -.3],
        populations={"sensory": [0], "deep": [1], "motor": [2, 3]},
    )
    flat = cd.Learner(
        cd.NeuralGraph(cn, cd.learning_neuron_model()), [2, 3],
        cd.LearnerConfig(beta=1e-3, credit_transport=3., free_steps=160, nudged_steps=160),
    )
    drive = np.array([[.7, 0., 0., 0.]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reports = [flat.step(drive, np.array([0]))[1] for _ in range(16)]
    assert all(r["credit_transport_refused"] == 1. for r in reports)
    said = _refusals(caught)
    assert len(said) == 1 and "in 15 lessons in a row" in said[0]
    assert "no apical compartment" in said[0]
