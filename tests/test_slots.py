"""Slotted outputs: a whole utterance settles at once, one softmax per slot."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import cadence as cd


def test_slotted_targets_and_predictions_have_one_choice_per_slot() -> None:
    w = cd.layered(6, 8, 3 * 4, density=1.0, seed=0)  # three slots of four choices
    learner = cd.Learner(cd.Settlement(w, cd.learning_rule(dt=1.0)), w.sets["output"], slots=3)
    target = learner.targets(np.array([[0, 1, 3], [2, 2, 2]]))
    out = target[:, learner.output_index].reshape(2, 3, 4)
    assert out.sum(axis=2).tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert out[0, 2, 3] == 1.0 and out[1, 0, 2] == 1.0
    drive = np.zeros((2, w.n))
    assert learner.predict(drive).shape == (2, 3)
    assert learner.nudge_for(target, 0.1).groups is not None
    with pytest.raises(ValueError):
        learner.targets(np.array([0, 1]))
    with pytest.raises(ValueError):
        cd.Learner(cd.Settlement(w, cd.learning_rule()), w.sets["output"], slots=5)


def test_a_slotted_net_learns_a_whole_pattern_at_once() -> None:
    """Six inputs, two slots: each slot names which of its three inputs is largest, and the two
    settle and learn together."""
    rng = np.random.default_rng(1)
    slots, choices = 2, 3
    x = rng.random((64, 6))
    y = np.stack([x[:, :3].argmax(axis=1), x[:, 3:].argmax(axis=1)], axis=1)
    w = cd.layered(6, 16, slots * choices, density=1.0, seed=2)
    cfg = cd.LearnerConfig(
        eta=2.0, beta=0.1, temperature=0.1, tolerance=3e-3, nudged_steps=12, free_steps=60
    )
    learner = cd.Learner(
        cd.Settlement(w, cd.learning_rule(dt=1.0)), w.sets["output"], cfg, slots=slots
    )
    drive = learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, w.n - 6))))
    for epoch in range(25):
        learner.config = dataclasses.replace(cfg, eta=2.0 * 0.9**epoch)
        for s in range(0, 64, 16):
            learner.step(drive[s : s + 16], y[s : s + 16])
    learner.config = dataclasses.replace(cfg, tolerance=1e-4)
    assert (learner.predict(drive) == y).mean() > 0.6  # chance is a third per slot
