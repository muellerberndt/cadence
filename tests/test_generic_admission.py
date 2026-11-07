"""Live answers require a fresh whole-graph residual without losing real outcomes."""

from dataclasses import replace

import numpy as np
import pytest

from cadence import Brain


def checkpoint(brain, path):
    """Compare all saved arrays and metadata, including RNG and pending eligibility."""
    with np.load(brain.save(path), allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def assert_checkpoint_equal(before, after):
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)


def composed(**options):
    return Brain.compose(2, 2, modules=(5, 4), observers=(3,), seed=3, **{"recall": False, **options})


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_exhausted_live_repair_preserves_full_pending_checkpoint(tmp_path, backend):
    if backend == "torch":
        pytest.importorskip("torch")
    brain = composed(backend=backend, device="cpu" if backend == "torch" else None)
    brain.hippocampus.observe(np.array([[1.0, 0.0]]), np.array([[0.7, -0.2]]))
    brain.act([[0.2, -0.1]])
    assert brain.basal_ganglia._pending is not None
    brain.learner.config = replace(brain.learner.config, free_steps=1, tolerance=1e-12)
    before = checkpoint(brain, tmp_path / "before")
    with pytest.raises(RuntimeError, match="no action issued"):
        brain.act([[-0.3, 0.2]])
    assert_checkpoint_equal(before, checkpoint(brain, tmp_path / "after"))


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("coordinate", ["v", "bias"])
def test_matching_and_restored_cache_cannot_hide_observer_residual(tmp_path, restore, coordinate):
    brain = composed(episodic=False, working_memory_amplitude=0)
    observation = np.array([[0.2, -0.1]])
    brain.act(observation)
    # Keep matching cache keys but disturb a non-output observer coordinate.
    state = brain.basal_ganglia.state
    observer = brain.connectome.populations["observer_0"][0]
    if coordinate == "v":
        state.v[0, observer] += 0.5
    else:
        bias = brain.brain.bias.copy()
        bias[observer] += 0.5
        brain.brain.bias = bias
    brain._prepared = observation.copy()
    brain.learner.config = replace(brain.learner.config, free_steps=0)
    drive = brain.stimulus(observation)
    assert brain.brain.residual(drive, state).max() > brain.learner.config.tolerance
    if restore:
        brain = Brain.load(brain.save(tmp_path / "unqualified-cache"))
    before = checkpoint(brain, tmp_path / "before")
    with pytest.raises(RuntimeError, match="no action issued"):
        brain.act(observation)
    assert_checkpoint_equal(before, checkpoint(brain, tmp_path / "after"))


def test_one_unqualified_stream_refuses_the_entire_action_batch(tmp_path):
    brain = composed(episodic=False, working_memory_amplitude=0)
    observation = np.array([[0.2, -0.1], [0.1, -0.2]])
    brain.act(observation)
    brain.learner.config = replace(brain.learner.config, free_steps=0)
    brain.basal_ganglia.state.v[1, brain.motor_index[0]] += 0.5
    residual = brain.brain.residual(brain.stimulus(observation), brain.basal_ganglia.state)
    assert residual[0] <= brain.learner.config.tolerance < residual[1]
    before = checkpoint(brain, tmp_path / "before")
    with pytest.raises(RuntimeError, match="no action issued"):
        brain.act(observation)
    assert_checkpoint_equal(before, checkpoint(brain, tmp_path / "after"))


def test_zero_budget_admits_a_freshly_checked_existing_equilibrium(monkeypatch):
    brain = composed(episodic=False, working_memory_amplitude=0)
    observation = [[0.2, -0.1]]
    brain.act(observation, greedy=True)
    brain.learner.config = replace(brain.learner.config, free_steps=0)
    checked = []
    residual = brain.brain.residual

    def record(*args, **kwargs):
        result = residual(*args, **kwargs)
        checked.append(result.copy())
        return result

    monkeypatch.setattr(brain.brain, "residual", record)
    assert brain.act(observation, greedy=True).shape == (1,)
    assert checked and all(np.all(value <= brain.learner.config.tolerance) for value in checked)
    assert brain.basal_ganglia.state.steps == 0


@pytest.mark.parametrize("operation", ["predict", "accuracy"])
def test_independent_answers_also_refuse_without_consuming_pending_outcomes(tmp_path, operation):
    brain = composed()
    brain.act([[0.2, -0.1]])
    brain.learner.config = replace(brain.learner.config, free_steps=0)
    before = checkpoint(brain, tmp_path / "before")
    args = ([[0.3, -0.2]],) if operation == "predict" else ([[0.3, -0.2]], [0])
    with pytest.raises(RuntimeError, match="no action issued"):
        getattr(brain, operation)(*args)
    assert_checkpoint_equal(before, checkpoint(brain, tmp_path / "after"))


def test_real_feedback_is_consumed_once_before_refusal_and_act_retries_only_the_answer(tmp_path):
    brain = composed()
    brain.step([[0.2, -0.1]])
    brain.learner.config = replace(brain.learner.config, free_steps=0, tolerance=1e-12)
    control = Brain.load(brain.save(tmp_path / "pending"))
    following, reward, done = [[-0.1, 0.3]], np.array([0.4]), np.array([False])
    # This independent control admits the actual outcome but asks for no next action.
    control.last_learning = control.learn(reward, done, following)
    with pytest.raises(RuntimeError, match="no action issued"):
        brain.step(following, reward=reward, done=done)
    assert brain.basal_ganglia.updates == control.basal_ganglia.updates == 1
    assert brain.hippocampus.writes == control.hippocampus.writes == 1
    assert brain.basal_ganglia._pending is None
    accepted_feedback = checkpoint(brain, tmp_path / "feedback")
    assert_checkpoint_equal(accepted_feedback, checkpoint(control, tmp_path / "control"))
    with pytest.raises(RuntimeError, match="preceding action"):
        brain.step(following, reward=reward, done=done)
    assert_checkpoint_equal(accepted_feedback, checkpoint(brain, tmp_path / "duplicate"))
    for candidate in (brain, control):
        candidate.learner.config = replace(
            candidate.learner.config, free_steps=1024, tolerance=3e-3
        )
    np.testing.assert_array_equal(brain.act(following), control.act(following))
    assert brain.basal_ganglia.updates == brain.hippocampus.writes == 1
    assert_checkpoint_equal(
        checkpoint(brain, tmp_path / "retried"), checkpoint(control, tmp_path / "control-retried")
    )


@pytest.mark.parametrize("operation", ["act", "step", "feedback"])
def test_changed_batch_identity_is_rejected_before_state_or_feedback_changes(tmp_path, operation):
    brain = composed()
    brain.act([[0.2, -0.1]])
    before = checkpoint(brain, tmp_path / "before")
    observations = [[0.2, -0.1], [0.1, -0.2]]
    with pytest.raises(ValueError):
        if operation == "feedback":
            brain.step(observations, reward=np.ones(2), done=np.zeros(2, bool))
        else:
            getattr(brain, operation)(observations)
    assert_checkpoint_equal(before, checkpoint(brain, tmp_path / "after"))


@pytest.mark.parametrize("operation", ["act", "predict", "accuracy"])
def test_disabled_legacy_stopping_tolerance_cannot_publish_an_answer(tmp_path, operation):
    brain = composed()
    brain.act([[0.2, -0.1]])
    brain.learner.config = replace(brain.learner.config, tolerance=None)
    before = checkpoint(brain, tmp_path / "before")
    args = ([[0.3, -0.2]],) if operation != "accuracy" else ([[0.3, -0.2]], [0])
    with pytest.raises(ValueError, match="residual tolerance"):
        getattr(brain, operation)(*args)
    assert_checkpoint_equal(before, checkpoint(brain, tmp_path / "after"))


def test_changed_batch_cannot_teach_after_a_greedy_live_action(tmp_path):
    brain = composed()
    brain.act([[0.2, -0.1]], greedy=True)
    assert brain.basal_ganglia._pending is None
    before = checkpoint(brain, tmp_path / "before")
    with pytest.raises(ValueError, match="live streams"):
        brain.step([[0.2, -0.1], [0.1, -0.2]], teacher=[0, 1])
    assert_checkpoint_equal(before, checkpoint(brain, tmp_path / "after"))
