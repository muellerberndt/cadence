"""What an imagined branch reads it does not write, and what a refused outcome wrote it takes
back: the held message's counts under an imagined moment, a fading store's depth and the
episodic hold's context under a refused outcome. Each brain's file before and after is the
same, and the outcome retried equals a first call on a twin that never saw the refusal."""

from __future__ import annotations

import copy
import warnings
from dataclasses import replace

import numpy as np
import pytest

import cadence as cd

QUALIFIED = cd.LearnerConfig(
    qualified=True, damping=3, free_steps=1024, nudged_steps=128, tolerance=3e-3
)


def saved(brain: cd.Brain, path) -> dict[str, np.ndarray]:
    with np.load(copy.deepcopy(brain).save(path, compressed=False)) as data:
        return {name: np.array(data[name]) for name in data.files}


def changed(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> list[str]:
    names = set(before) | set(after)
    return sorted(
        name for name in names
        if name not in before or name not in after
        or before[name].shape != after[name].shape
        or (before[name].dtype.kind in "fiub" and not np.array_equal(before[name], after[name]))
        or (before[name].dtype.kind not in "fiub" and name != "description"
            and not np.array_equal(before[name], after[name]))
    )


@pytest.mark.parametrize("options", [{}, {"message_boundary": 5}], ids=["roles", "boundary"])
def test_an_imagined_branch_leaves_the_held_message_as_it_was(options, tmp_path):
    def build() -> cd.Brain:
        return cd.Brain.compose(inputs=6, actions=3, modules=(16,), seed=0, message=12,
                                message_winners=2, message_roles=((0, 1), (2, 3)), **options)

    eye = np.eye(6)
    brain, twin = build(), build()
    for x in (eye[[0]], eye[[2]]):
        brain.act(x, greedy=True)
        twin.act(x, greedy=True)
    before = saved(brain, tmp_path / "before.npz")
    counts = brain.message.counts.copy()
    brain.imagine([eye[[1]], eye[[0]], eye[[3]], eye[[5]]])
    np.testing.assert_array_equal(brain.message.counts, counts)
    assert changed(before, saved(brain, tmp_path / "after.npz")) == []
    brain.act(eye[[1]], greedy=True)  # the next real moment joins the block it would have
    twin.act(eye[[1]], greedy=True)
    np.testing.assert_array_equal(brain.message.state, twin.message.state)
    np.testing.assert_array_equal(brain.stimulus(eye[[4]]), twin.stimulus(eye[[4]]))


def refused(brain: cd.Brain, reward, done, following) -> None:
    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=0)
    try:
        with pytest.raises(cd.LearningPhaseError):
            brain.learn(reward, done, following)
    finally:
        brain.learner.config = config


@pytest.mark.parametrize("done", [False, True], ids=["going on", "ended"])
def test_a_refused_outcome_leaves_a_fading_store_as_it_was(done, tmp_path):
    warnings.simplefilter("ignore")
    brain = cd.Brain.compose(inputs=5, actions=3, modules=(10,), seed=5, learning=QUALIFIED,
                             recall=False, memory_fade=1e-3)
    x, y = np.array([[1.0, 0, 1, 0, 0]]), np.array([[1.0, 1, 0, 0, 1]])
    brain.step(x)
    twin = copy.deepcopy(brain)
    before = saved(brain, tmp_path / "before.npz")
    depth = brain.hippocampus.fade_depth.copy()
    for _ in range(3):
        refused(brain, np.array([1.0]), np.array([done]), y)
    np.testing.assert_array_equal(brain.hippocampus.fade_depth, depth)
    assert changed(before, saved(brain, tmp_path / "after.npz")) == []
    brain.learn(np.array([1.0]), np.array([done]), y)
    twin.learn(np.array([1.0]), np.array([done]), y)
    assert changed(saved(twin, tmp_path / "twin.npz"), saved(brain, tmp_path / "retry.npz")) == []


@pytest.mark.parametrize("hold", ["unchanged", "punished"])
def test_a_refused_end_of_episode_leaves_the_episodic_context_as_it_was(hold, tmp_path):
    warnings.simplefilter("ignore")
    brain = cd.Brain.compose(inputs=5, actions=3, modules=(10,), seed=5, learning=QUALIFIED,
                             recall=False, memory_choice=True, memory_context_hold=hold)
    a, b = np.array([[1.0, 0, 1, 0, 0]]), np.array([[1.0, 1, 0, 0, 1]])
    brain.step(a)
    brain.step(b, reward=[0.0])
    brain.step(b, reward=[0.0])  # a second decision on one observation: the hold keys it
    twin = copy.deepcopy(brain)
    episodic = brain.memories["episodic"]
    held = {name: None if getattr(episodic, name) is None else getattr(episodic, name).copy()
            for name in episodic._STATE}
    before = saved(brain, tmp_path / "before.npz")
    refused(brain, np.array([0.0]), np.array([True]), b)
    for name, value in held.items():
        if value is None:
            assert getattr(episodic, name) is None
        else:
            np.testing.assert_array_equal(getattr(episodic, name), value, err_msg=name)
    assert changed(before, saved(brain, tmp_path / "after.npz")) == []
    # the stream goes on, the episode not ended: both read and write under one context
    brain.step(b, reward=[1.0])
    twin.step(b, reward=[1.0])
    assert changed(saved(twin, tmp_path / "twin.npz"), saved(brain, tmp_path / "on.npz")) == []
