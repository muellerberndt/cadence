"""Operating defaults chosen by the readout they serve: issues 124, 125, 126, 127, 220.

The hand-set defaults remain the controls. These tests pin what each default
resolves to, that composed brains settle at those defaults, and that the
diagnostics name the cause when a budget or rate is the problem.
"""

import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import cadence as cd


def motor_synapses(brain):
    """The motor-to-motor synapse signs of a composed brain."""
    w = brain.brain.connectome
    motor = np.zeros(w.n, dtype=bool)
    motor[brain.motor_index] = True
    edges = motor[w.pre] & motor[w.post]
    return w.sign[edges]


# -- issues 124 and 220: an unset motor lateral is 0.0 at every menu size


@pytest.mark.parametrize("actions", [2, 4, 8, 9, 36])
def test_an_unset_lateral_wires_no_motor_competition(actions: int) -> None:
    brain = cd.Brain.compose(inputs=8, actions=actions, modules=(8,), seed=0, recall=False)
    assert len(motor_synapses(brain)) == 0


def test_an_explicit_lateral_is_used_as_given():
    brain = cd.Brain.compose(inputs=8, actions=36, modules=(8,), seed=0, lateral=-0.5, recall=False)
    signs = motor_synapses(brain)
    assert len(signs) == 36 * 35 and np.all(signs == -0.5)
    signs = motor_synapses(cd.Brain.compose(inputs=8, actions=4, modules=(8,), seed=0, lateral=-0.5, recall=False))
    assert len(signs) == 4 * 3 and np.all(signs == -0.5)
    assert (
        len(
            motor_synapses(cd.Brain.compose(inputs=8, actions=4, modules=(8,), seed=0, lateral=0.0, recall=False))
        )
        == 0
    )


def test_the_composed_layout_resolves_the_unset_lateral_as_compose_does():
    for actions in (4, 36):
        layout = cd.Genome.composed(8, actions, modules=(8,))
        brain = cd.Brain(cd.develop(layout, seed=0))
        assert len(motor_synapses(brain)) == 0
    signs = motor_synapses(cd.Brain(cd.develop(cd.Genome.composed(8, 4, modules=(8,), lateral=-0.5), seed=0)))
    assert len(signs) == 12 and np.all(signs == -0.5)


def test_wide_composed_brain_settles_undamped_at_the_default():
    # The issue-124 defect: at lateral -0.5 a 36-action composed brain does not
    # settle an undamped free solve at all; at the resolved default it settles
    # in the ordinary few dozen sweeps.
    rng = np.random.default_rng(3)
    brain = cd.Brain.compose(inputs=64, actions=36, modules=(64, 32), seed=1, recall=False)
    drive = brain.stimulus(rng.uniform(0, 1, (8, 64)), memory=False)
    phase = brain.brain.equilibrate(drive, budget=256, tolerance=3e-3)
    assert bool(np.all(phase.qualified)) and phase.steps <= 64


def test_eight_action_composed_brain_settles_undamped_at_the_default():
    # The issue-220 defect: at lateral -0.5 an 8-action composed brain settling 16 rows
    # together oscillates in its undamped free solve, so greedy acts fall to the damping
    # fallback and a lesson's free phase runs out its budget. At the resolved default the
    # undamped solve, the acts and the lesson all settle in the ordinary few dozen sweeps.
    def life(brain: cd.Brain) -> tuple[list[Mapping[str, Any]], dict[str, float]]:
        rng = np.random.default_rng(3)
        settles = []
        for _ in range(4):
            brain.act(rng.uniform(0, 1, (16, 64)), greedy=True)
            settled = brain.last_settlement
            assert settled is not None
            settles.append(settled)
        brain.lesson(rng.uniform(0, 1, (16, 64)), rng.integers(0, 8, 16))
        return settles, brain.last_learning

    competing = cd.Brain.compose(inputs=64, actions=8, modules=(64, 32), seed=1, lateral=-0.5, recall=False)
    with pytest.warns(RuntimeWarning, match="entire free_steps budget"):
        settles, lesson = life(competing)
    assert any(settled["damping_halvings"] > 0 for settled in settles)
    assert lesson["demonstration_free_budget_exhausted"] == 1.0

    brain = cd.Brain.compose(inputs=64, actions=8, modules=(64, 32), seed=1, recall=False)
    drive = brain.stimulus(np.random.default_rng(3).uniform(0, 1, (16, 64)), memory=False)
    phase = brain.brain.equilibrate(drive, budget=256, tolerance=3e-3)
    assert bool(np.all(phase.qualified)) and phase.steps <= 64
    budget = brain.learner.config.free_steps
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        settles, lesson = life(brain)
    for settled in settles:
        assert settled["qualified"] and settled["damping_halvings"] == 0
        assert settled["steps"] <= 64 < budget == settled["budget"]
    assert lesson["demonstration_free_budget_exhausted"] == 0.0
    assert lesson["demonstration_free_steps"] <= 64 < budget


def test_invalid_lateral_is_still_rejected():
    with pytest.raises(ValueError, match="lateral"):
        cd.Brain.compose(inputs=8, actions=4, modules=(8,), lateral=np.nan, recall=False)
    with pytest.raises(ValueError, match="lateral"):
        cd.Brain.compose(inputs=8, actions=4, modules=(8,), lateral=True, recall=False)


# -- issue 125: calibration targets the competitive operating point by default


def readout_learner(outputs=36, **config):
    """Independent output neurons: every gain settles, and the operating point is exact."""
    graph = cd.NeuralGraph(
        cd.Connectome.from_synapses(outputs, pre=[], post=[]),
        cd.learning_neuron_model(dt=1),
    )
    return cd.Learner(
        graph,
        np.arange(outputs),
        cd.LearnerConfig(
            qualified=True,
            free_steps=512,
            nudged_steps=512,
            tolerance=3e-3,
            **config,
        ),
    )


def driven_readout(**config):
    """36 input neurons each driving one output neuron: gain moves the readout."""
    graph = cd.NeuralGraph(
        cd.Connectome.from_synapses(
            72, pre=np.arange(36), post=np.arange(36, 72), sign=np.ones(36)
        ),
        cd.learning_neuron_model(dt=1),
    )
    return cd.Learner(
        graph,
        np.arange(36, 72),
        cd.LearnerConfig(
            qualified=True,
            free_steps=512,
            nudged_steps=512,
            tolerance=3e-3,
            **config,
        ),
    )


def test_default_calibration_targets_the_top_output_not_the_mean():
    rng = np.random.default_rng(0)
    drive = np.zeros((4, 72))
    drive[:, :36] = rng.uniform(0.1, 0.9, (4, 36))
    learner = driven_readout()
    competitive = learner.calibrate(drive)
    report = learner.last_calibration
    assert report["target"] == "top" and report["level"] == 0.5
    chosen = next(x for x in report["candidates"] if x["gain"] == competitive)
    assert chosen["gap"] == abs(chosen["top_output"] - 0.5)
    assert chosen["top_output"] == min(
        (x["top_output"] for x in report["candidates"] if x["admitted"]),
        key=lambda top: abs(top - 0.5),
    )

    saturating = driven_readout()
    mean_target = saturating.calibrate(drive, level=0.5)
    mean_report = saturating.last_calibration
    assert mean_report["target"] == "mean"
    picked = next(x for x in mean_report["candidates"] if x["gain"] == mean_target)
    assert picked["gap"] == abs(picked["mean_output"] - 0.5)
    # Driving the mean of 36 outputs to 0.5 needs a larger gain than placing
    # the winner at 0.5: the whole readout is pushed up together (issue 125).
    assert mean_target > competitive
    assert picked["mean_output"] > chosen["mean_output"]


def test_single_output_calibration_is_unchanged_by_the_competitive_default():
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1], count=[1])

    def learner():
        return cd.Learner(
            cd.NeuralGraph(graph, cd.learning_neuron_model(dt=1)),
            [1],
            cd.LearnerConfig(qualified=True, free_steps=32, tolerance=1e-9),
        )

    drive = np.array([[1.0, 0.0]])
    assert learner().calibrate(drive) == learner().calibrate(drive, level=0.5)


def test_slotted_calibration_takes_one_winner_per_slot():
    base = readout_learner(outputs=36)
    learner = cd.Learner(base.brain, np.arange(36), base.config, slots=(12, 24))
    rng = np.random.default_rng(1)
    drive = rng.uniform(0.0, 0.4, (2, 36))
    learner.calibrate(drive, grid=[1.0])
    candidate = learner.last_calibration["candidates"][0]
    cfg = learner.config
    phase = learner.brain.equilibrate(
        drive, budget=cfg.free_steps, tolerance=cfg.tolerance, damping=cfg.damping
    )
    outputs = phase.state.activation[:, learner.output_index]
    expected = np.concatenate([outputs[:, :12].max(axis=1), outputs[:, 12:].max(axis=1)]).mean()
    assert candidate["top_output"] == pytest.approx(expected, abs=1e-12)
    assert candidate["mean_output"] == pytest.approx(outputs.mean(), abs=1e-12)
    assert candidate["top_output"] > candidate["mean_output"]


# -- issue 126: the bias rate follows the synapse rate unless chosen


def test_eta_bias_defaults_to_a_tenth_of_eta():
    assert cd.LearnerConfig().eta_bias == pytest.approx(0.02)
    assert cd.LearnerConfig(eta=0.0015).eta_bias == pytest.approx(0.00015)
    assert cd.LearnerConfig(eta=0.5).eta_bias == pytest.approx(0.05)
    assert cd.LearnerConfig(eta=0.5, eta_bias=0.02).eta_bias == 0.02


def test_replace_keeps_the_resolved_bias_rate_unless_rederived():
    config = cd.LearnerConfig(eta=0.2)
    lowered = replace(config, eta=0.002, eta_bias=None)
    assert lowered.eta_bias == pytest.approx(0.0002)
    with pytest.warns(RuntimeWarning, match="bias step dominates"):
        kept = replace(config, eta=0.002)
    assert kept.eta_bias == pytest.approx(0.02)
    restored = cd.LearnerConfig(**lowered.to_dict())
    assert restored == lowered


def test_dominating_bias_rate_warns_and_zero_eta_stays_silent():
    with pytest.warns(RuntimeWarning, match="bias step dominates"):
        cd.LearnerConfig(eta=0.0015, eta_bias=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd.LearnerConfig(eta=0.0, eta_bias=0.02)  # bias-only learning is a control
        cd.LearnerConfig(eta=0.2, eta_bias=0.2)


def test_composed_default_bias_rate_follows_the_composed_eta():
    config = cd.Brain.compose(inputs=4, actions=2, modules=(4,), seed=0, recall=False).learner.config
    assert config.eta == 0.5 and config.eta_bias == pytest.approx(0.05)


# -- issue 127: a finite free phase that uses its whole budget says so


def exhausted_learner(tolerance=3e-3):
    pre, post = np.where(~np.eye(36, dtype=bool))
    graph = cd.NeuralGraph(
        cd.Connectome.from_synapses(36, pre=pre, post=post, sign=np.full(len(pre), -0.5)),
        cd.learning_neuron_model(dt=1),
    )
    return cd.Learner(
        graph,
        np.arange(36),
        cd.LearnerConfig(
            free_steps=2,
            nudged_steps=2,
            tolerance=tolerance,
        ),
    )


def test_exhausted_finite_free_phase_warns_and_is_counted():
    learner = exhausted_learner()
    with pytest.warns(RuntimeWarning, match="entire free_steps budget") as caught:
        _, report = learner.step(np.full((1, 36), 0.2), np.array([0]))
    assert report["free_budget_exhausted"] == 1.0
    assert report["accepted"] == 1.0  # finite teaching retains its law
    assert any("free_residual" in str(w.message) for w in caught)


def test_exhausted_lessons_are_counted_and_the_tenth_says_how_many():
    learner = exhausted_learner()
    assert (learner.finite_lessons, learner.exhausted_lessons) == (0, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(10):
            learner.step(np.full((1, 36), 0.2), np.array([0]))
    assert (learner.finite_lessons, learner.exhausted_lessons) == (10, 10)
    said = [str(w.message) for w in caught if "entire free_steps budget" in str(w.message)]
    assert len(said) == 10 and len(set(said[:9])) == 1  # one text, so shown once by default
    assert said[9].endswith("in 10 of this learner's 10 finite lessons so far.")
    settled = readout_learner()
    settled.config = replace(settled.config, qualified=False)
    settled.step(np.full((1, 36), 0.2), np.array([0]))
    assert (settled.finite_lessons, settled.exhausted_lessons) == (1, 0)


def test_fixed_length_phases_and_settled_phases_stay_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, report = exhausted_learner(tolerance=None).step(np.full((1, 36), 0.2), np.array([0]))
    assert report["free_budget_exhausted"] == 0.0

    settled = readout_learner()
    settled.config = replace(settled.config, qualified=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, report = settled.step(np.full((1, 36), 0.2), np.array([0]))
    assert report["free_budget_exhausted"] == 0.0
    assert report["free_steps"] < 512


def test_brain_step_reports_the_exhausted_demonstration():
    composed = cd.Brain.compose(inputs=4, actions=2, modules=(4,), seed=0, recall=False)
    brain = cd.Brain.compose(
        inputs=4,
        actions=2,
        modules=(4,),
        seed=0,
        learning=replace(composed.learner.config, free_steps=1), recall=False,
    )
    # The lesson is taught and counted before the following answer, whose
    # one-sweep budget then refuses without undoing the learning.
    with pytest.warns(RuntimeWarning, match="entire free_steps budget"):
        with pytest.raises(RuntimeError, match="did not settle"):
            brain.step(np.array([[0.6, 0.1, 0.2, 0.9]]), teacher=np.array([1]))
    assert brain.last_learning["demonstration_free_budget_exhausted"] == 1.0
    assert brain.last_learning["demonstration_accepted"] == 1.0


# -- issue 143: the actor's bias rate follows its synapse rate


def test_actor_eta_bias_defaults_to_a_tenth_of_eta():
    assert cd.ActorCriticConfig().eta_bias == pytest.approx(0.05)  # eta 0.5: unchanged
    assert cd.ActorCriticConfig(eta=0.002).eta_bias == pytest.approx(0.0002)
    assert cd.ActorCriticConfig(eta=0.0).eta_bias == 0.0
    assert cd.ActorCriticConfig(eta=0.002, eta_bias=0.0001).eta_bias == 0.0001


def test_actor_replace_keeps_the_resolved_bias_rate_unless_rederived():
    config = cd.ActorCriticConfig(eta=0.5)
    lowered = replace(config, eta=0.002, eta_bias=None)
    assert lowered.eta_bias == pytest.approx(0.0002)
    with pytest.warns(RuntimeWarning, match="ActorCriticConfig.*bias step dominates"):
        kept = replace(config, eta=0.002)
    assert kept.eta_bias == pytest.approx(0.05)


def test_actor_dominating_bias_rate_warns_and_zero_eta_stays_silent():
    with pytest.warns(RuntimeWarning, match="ActorCriticConfig.*bias step dominates"):
        cd.ActorCriticConfig(eta=0.002, eta_bias=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd.ActorCriticConfig(eta=0.0, eta_bias=0.05)  # bias-only learning is a control
        cd.ActorCriticConfig(eta=0.2, eta_bias=0.2)


def test_composed_actor_keeps_its_measured_bias_rate():
    brain = cd.Brain.compose(inputs=4, actions=3, modules=(8,), seed=0, recall=False)
    config = brain.basal_ganglia.config
    assert config.eta == 1.0 and config.eta_bias == 0.05
    saved = config.to_dict()
    assert saved["eta_bias"] == 0.05 and cd.ActorCriticConfig(**saved) == config
