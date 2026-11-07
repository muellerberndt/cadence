"""Acquired ordinary behavior and composition options have public-API checks.

Arbitrary depth/observer wiring is not a guarantee of uniform task accuracy.
The broader declared matrix, including failures, is retained as release evidence.
"""

import numpy as np
import pytest

import cadence as cd


def blobs(seed):
    rng = np.random.default_rng(seed)
    x = np.zeros((120, 8))
    x[:60, :4] = 1
    x[60:, 4:] = 1
    return np.clip(x + 0.3 * rng.standard_normal(x.shape), 0, 1), np.repeat([0, 1], 60)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_composition_acquires_labels_with_qualified_free_answers(seed):
    brain = cd.Brain.compose(8, 2, seed=seed, recall=False)
    training, labels = blobs(0)
    held, expected = blobs(7)
    before = brain.accuracy(held, expected)
    # Keep the public default 30 epochs and the original blobs task's batch size.
    # Targets are independent labels, never the brain's own guessed answers.
    brain.fit(training, labels, batch=20)
    assert brain.learner.contrast_updates == 180
    assert brain.accuracy(training, labels) >= 0.95
    assert brain.accuracy(held, expected) >= 0.95 > before
    assert isinstance(brain.working_memory, cd.Trace)
    assert isinstance(brain.hippocampus, cd.SynapticMemory)
    assert not any(name.startswith("observer_") for name in brain.connectome.populations)


def same_checkpoint(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize(
    "modules,observers",
    [((8,), ()), ((8, 5), ()), ((8, 5, 3), ()), ((8, 5), (3,)), ((8, 5), (3, 2))],
)
def test_composed_depth_and_observation_preserve_actual_feedback_and_private_memory(
    modules, observers, tmp_path
):
    brain = cd.Brain.compose(3, 3, modules=modules, observers=observers, seed=2, recall=False)
    cue = np.array([[0.6, -0.3, 0.1], [-0.2, 0.4, 0.3]])
    actual_action = brain.act(cue)
    before = brain.save(tmp_path / "before")
    restored = cd.Brain.load(before)
    private = brain.imagine([np.zeros_like(cue), cue * 0.2])
    assert len(private) == 2 and all(np.all(phase.qualified) for phase in private)
    same_checkpoint(before, brain.save(tmp_path / "after-private"))
    # The independent environment assigns rewards to the preceding actual actions.
    reward = (actual_action == np.array([0, 2])).astype(float)
    done = np.array([False, True])
    following = cue * 0.1
    for owner in (brain, restored):
        owner.learn(reward, done, following)
    np.testing.assert_array_equal(brain.act(following), restored.act(following))
    same_checkpoint(brain.save(tmp_path / "continued"), restored.save(tmp_path / "restored"))


@pytest.mark.parametrize("episodic", [False, True])
def test_composition_forwards_memory_and_learning_options_through_saved_continuation(
    episodic, tmp_path
):
    learning = cd.LearnerConfig(free_steps=1024, tolerance=1e-4, nudged_steps=20)
    reward = cd.ActorCriticConfig(gamma=0.0, lam=0.0)
    brain = cd.Brain.compose(
        np.int64(3),
        np.int64(2),
        modules=(5, 4),
        observers=(2,),
        seed=np.int64(1),
        episodic=episodic,
        consolidation=0.2,
        working_memory_decay=0.7,
        working_memory_amplitude=0.4,
        learning=learning,
        reward=reward, recall=False,
    )
    assert (brain.hippocampus is not None) is episodic
    assert brain.working_memory.decay == 0.7 and brain.working_memory.amplitude == 0.4
    brain.act([[0.2, -0.1, 0.3]])
    restored = cd.Brain.load(brain.save(tmp_path / "options"))
    assert restored.learner.config == learning
    assert restored.basal_ganglia.config == reward
    assert restored.working_memory.decay == 0.7
    assert restored.working_memory.amplitude == 0.4
    if episodic:
        assert restored.hippocampus.consolidation == 0.2
    else:
        assert restored.hippocampus is None
    for owner in (brain, restored):
        owner.learn(np.array([0.4]), np.array([False]), [[0.1, 0.0, 0.2]])
    np.testing.assert_array_equal(brain.act([[0.1, 0.0, 0.2]]), restored.act([[0.1, 0.0, 0.2]]))
    same_checkpoint(brain.save(tmp_path / "live"), restored.save(tmp_path / "restored"))


@pytest.mark.parametrize("inputs,actions", [(0, 2), (2, 0), (True, 2), (2, True)])
def test_invalid_public_port_counts_are_refused(inputs, actions):
    with pytest.raises(ValueError, match="positive integers"):
        cd.Brain.compose(inputs, actions, modules=(4,), recall=False)


@pytest.mark.parametrize(
    "inputs,actions,modules,observers,seed",
    [
        (1, 1, (1,), (), 0),
        (4, 2, (16, 8), (), 7),
        (8, 2, (32, 16), (8,), 7),
    ],
)
def test_minimum_and_published_layouts_issue_qualified_actions(
    inputs, actions, modules, observers, seed
):
    brain = cd.Brain.compose(
        inputs, actions, modules=modules, observers=observers, seed=seed, recall=False
    )
    observation = np.linspace(0.1, 0.4, inputs)[None, :]
    drive = brain.stimulus(observation)
    choice = brain.act(observation)
    assert choice.shape == (1,) and 0 <= choice[0] < actions
    residual = brain.brain.residual(drive, brain.basal_ganglia.state)
    assert np.max(residual) <= brain.learner.config.tolerance
    branch = brain.imagine([observation * 0.5])
    assert len(branch) == 1 and np.all(branch[0].qualified)
