"""Public action/outcome custody with the policy frozen to isolate associative memory.

The full plastic actor/teacher path has separate transaction and acquisition tests.
Here the environment rewards the action actually returned, without replacing a
pending moment or supplying a preferred action as its reward owner.
"""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


def component_brain():
    return cd.Brain.compose(
        4, 4, modules=(8,), seed=3,
        learning=cd.LearnerConfig(
            eta=0, eta_bias=0, free_steps=1024, nudged_steps=128, tolerance=3e-3,
        ),
        reward=cd.ActorCriticConfig(
            gamma=0, lam=0, eta=0, eta_bias=0, eta_critic=0,
        ), recall=False,
    )


def checkpoint(brain, path):
    with np.load(brain.save(path), allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def assert_checkpoint_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def assert_one_actual_write(brain, cue, action):
    # A unit cue, reward=1 and default salience=abs(reward) give slow rate .1.
    # The fast delta correction then supplies the remaining .9. Other actions
    # are unobserved, and other cue rows have never occurred.
    expected = np.zeros((4, 4))
    expected[np.argmax(cue[0]), action] = 1
    memory = brain.hippocampus
    assert isinstance(memory, cd.SynapticMemory)
    assert memory.consolidation == 0.05 and memory.rate == memory.amplitude == 1
    assert memory.writes == 1
    np.testing.assert_allclose(memory.consolidated, 0.1 * expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(memory.strength, expected[None], rtol=0, atol=1e-15)
    np.testing.assert_allclose(memory.recall(cue), expected[np.argmax(cue[0])][None])


def test_actual_sampled_outcome_survives_pending_and_recall_checkpoints(tmp_path):
    brain = component_brain()
    cue, following = np.eye(4)[[2]], np.eye(4)[[0]]
    efficacy, bias = brain.brain.efficacy.copy(), brain.brain.bias.copy()
    action = int(brain.act(cue)[0])
    restored = cd.Brain.load(brain.save(tmp_path / "pending-actual-action"))
    assert_checkpoint_equal(
        checkpoint(brain, tmp_path / "pending-original"),
        checkpoint(restored, tmp_path / "pending-restored"),
    )

    for candidate in (brain, restored):
        candidate.learn(np.array([1.0]), np.array([True]), following)
        assert_one_actual_write(candidate, cue, action)
        assert candidate.basal_ganglia.updates == 1
        assert candidate.learner.contrast_updates == 0
        np.testing.assert_array_equal(candidate.brain.efficacy, efficacy)
        np.testing.assert_array_equal(candidate.brain.bias, bias)
    assert_checkpoint_equal(
        checkpoint(brain, tmp_path / "outcome-original"),
        checkpoint(restored, tmp_path / "outcome-restored"),
    )

    # Neural/working-state reset preserves the record; no teacher is queried.
    brain.reset()
    restored = cd.Brain.load(brain.save(tmp_path / "recorded-outcome"))
    for candidate in (brain, restored):
        assert int(candidate.act(cue, greedy=True)[0]) == action
        assert candidate.last_settlement["qualified"]
        assert candidate.last_settlement["max_residual"] <= 3e-3
        assert_one_actual_write(candidate, cue, action)
        assert candidate.basal_ganglia.updates == 1
        assert candidate.learner.contrast_updates == 0
        np.testing.assert_array_equal(candidate.brain.efficacy, efficacy)
        np.testing.assert_array_equal(candidate.brain.bias, bias)
    assert_checkpoint_equal(
        checkpoint(brain, tmp_path / "recall-original"),
        checkpoint(restored, tmp_path / "recall-restored"),
    )

    # Clearing only the fast residual leaves the measured partial slow value.
    brain.hippocampus.reset(1)
    expected_cold = np.zeros((1, 4))
    expected_cold[0, action] = 0.1
    np.testing.assert_allclose(brain.hippocampus.recall(cue), expected_cold)


def test_a_teacher_label_without_an_outcome_does_not_write_association():
    brain = component_brain()
    brain.step(np.eye(4)[[2]], teacher=np.array([3]))
    assert brain.last_learning["demonstrations"] == 1
    assert brain.hippocampus.writes == 0
    assert not brain.hippocampus.consolidated.any()
    assert not brain.hippocampus.strength.any()


def test_refused_action_cannot_replace_the_owner_of_an_actual_outcome(tmp_path):
    brain = component_brain()
    cue, refused = np.eye(4)[[2]], np.eye(4)[[0]]
    action = int(brain.act(cue)[0])
    qualified_config = brain.learner.config
    brain.learner.config = replace(qualified_config, free_steps=0, tolerance=1e-12)
    before = checkpoint(brain, tmp_path / "before-refusal")
    with pytest.raises(RuntimeError, match="no action issued"):
        brain.act(refused)
    assert_checkpoint_equal(before, checkpoint(brain, tmp_path / "after-refusal"))

    brain.learner.config = qualified_config
    brain.learn(np.array([1.0]), np.array([True]), refused)
    assert_one_actual_write(brain, cue, action)
    np.testing.assert_array_equal(brain.hippocampus.recall(refused), np.zeros((1, 4)))
    before = checkpoint(brain, tmp_path / "once")
    with pytest.raises(RuntimeError, match="non-greedy act first"):
        brain.learn(np.array([1.0]), np.array([True]), refused)
    assert_checkpoint_equal(before, checkpoint(brain, tmp_path / "duplicate-refused"))
