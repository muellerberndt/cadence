"""Main-interface qualification and honest accepted/refused teaching work."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


@pytest.mark.parametrize("qualified", [False, True])
def test_batched_lesson_reports_exposure_and_every_required_phase(qualified):
    wire = cd.Connectome.from_synapses(
        2, pre=np.array([0, 1]), post=np.array([1, 0]), sign=np.array([0.2, 0.2]),
    )
    graph = cd.NeuralGraph(wire, cd.learning_neuron_model(dt=0.5))
    learner = cd.Learner(graph, [0, 1], cd.LearnerConfig(
        qualified=qualified, free_steps=128, nudged_steps=128, tolerance=3e-3,
    ))
    states, report = learner.step(np.array([[0.2, 0.0], [0.0, 0.2], [0.3, 0.1]]),
                                  np.array([0, 1, 0]))
    assert report["attempted_presentations"] == report["accepted_presentations"] == 3
    phases = ("free", "nudged", "opposite")
    assert report["total_steps"] == sum(getattr(states, name).steps for name in phases)
    assert report["total_row_sweeps"] == 3 * report["total_steps"]
    assert report["total_row_residual_checks"] == 3 * report["total_residual_checks"]
    assert report["total_stagnation_checks"] == sum(
        report[name + "_stagnation_checks"] for name in phases
    )
    assert report["qualified"] == float(qualified)


def test_successful_teacher_work_survives_main_step_and_checkpoint(tmp_path):
    brain = cd.Brain.compose(2, 2, modules=(4,), seed=0, learning=cd.LearnerConfig(
        qualified=True, free_steps=512, nudged_steps=512, tolerance=3e-3,
    ), recall=False)
    brain.step(np.array([[1.0, 0.0]]), teacher=np.array([0]))
    report = brain.last_learning
    assert report["demonstrations"] == report["demonstration_accepted_presentations"] == 1
    assert report["demonstration_qualified"] == 1
    assert report["demonstration_total_steps"] == sum(
        report["demonstration_" + name + "_steps"]
        for name in ("free", "nudged", "opposite")
    )
    restored = cd.Brain.load(brain.save(tmp_path / "taught.npz"))
    assert restored.last_learning == report


def test_refused_teacher_keeps_prior_accepted_reward_and_attempted_work():
    brain = cd.Brain.compose(2, 2, modules=(4,), seed=0, recall=False)
    x = np.array([[1.0, 0.0]])
    brain.step(x)
    brain.learner.config = replace(
        brain.learner.config, qualified=True, free_steps=512, nudged_steps=0,
    )
    before_reward_updates = brain.basal_ganglia.updates
    before_contrast_updates = brain.learner.contrast_updates
    with pytest.raises(cd.LearningPhaseError) as failed:
        brain.step(x, reward=np.array([1.0]), done=np.array([False]), teacher=np.array([0]))
    assert failed.value.phase == "nudged"
    assert brain.basal_ganglia.updates == before_reward_updates + 1
    assert brain.learner.contrast_updates == before_contrast_updates
    assert brain.basal_ganglia._pending is None  # actual reward is consumed exactly once
    assert "td_error" in brain.last_learning
    assert brain.last_learning["demonstrations"] == 0
    assert brain.last_learning["demonstration_accepted_presentations"] == 0
    assert brain.last_learning["demonstration_attempted_presentations"] == 1
    for name, value in failed.value.report.items():
        assert brain.last_learning["demonstration_" + name] == value


def test_fit_cannot_publish_a_finite_accuracy_as_a_qualified_brain_score():
    brain = cd.Brain.compose(2, 36, modules=(4,), seed=0, learning=cd.LearnerConfig(
        free_steps=1, nudged_steps=0, tolerance=3e-3, eta=0, eta_bias=0,
    ), recall=False)
    x, labels = np.array([[1.0, 0.0]]), np.array([0])
    # One transient sweep happens to choose the teacher label, despite failing the equations.
    assert brain.learner.accuracy(brain.stimulus(x, memory=False), labels) == 1
    with pytest.raises(RuntimeError, match="no action issued"):
        brain.fit(x, labels, epochs=1)
    assert brain.learner.updates == 1  # the accepted lesson remains; its score was refused


def test_a_finite_lessons_diagnostic_check_reads_the_phases_own_nudge(monkeypatch):
    """Each nudged phase and its residual diagnostic share one Nudge (its kernel arrays made
    once); the opposite phase's is the pull reversed; the public phase builds its own."""
    wire = cd.Connectome.from_synapses(
        2, pre=np.array([0, 1]), post=np.array([1, 0]), sign=np.array([0.2, 0.2]),
    )
    graph = cd.NeuralGraph(wire, cd.learning_neuron_model(dt=0.5))
    learner = cd.Learner(graph, [0, 1], cd.LearnerConfig(
        free_steps=64, nudged_steps=16, tolerance=3e-3,
    ))
    settled, checked = [], []
    settle, residual = cd.NeuralGraph.settle_batch, cd.NeuralGraph.residual

    def recorded_settle(self, *args, **kwargs):
        if kwargs.get("nudge") is not None:
            settled.append(kwargs["nudge"])
        return settle(self, *args, **kwargs)

    def recorded_residual(self, *args, **kwargs):
        if kwargs.get("nudge") is not None:
            checked.append(kwargs["nudge"])
        return residual(self, *args, **kwargs)

    monkeypatch.setattr(cd.NeuralGraph, "settle_batch", recorded_settle)
    monkeypatch.setattr(cd.NeuralGraph, "residual", recorded_residual)
    drive, labels = np.array([[0.2, 0.0], [0.0, 0.2]]), np.array([0, 1])
    states, report = learner.step(drive, labels)
    assert [nudge.beta for nudge in settled] == [0.1, -0.1]
    assert len(checked) == 2 and settled[0] is checked[0] and settled[1] is checked[1]
    assert report["nudged_residual_checks"] == report["opposite_residual_checks"] == 1
    pushed = learner.nudged(drive, states.free, learner.targets(labels), sign=-1.0)
    assert settled[-1].beta == -0.1 and settled[-1] is not settled[1] and pushed.steps > 0
