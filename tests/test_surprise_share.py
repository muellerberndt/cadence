"""Surprise lessons in proportion to their surprise (cadence#239): with ``surprise_share`` a
lesson's step is scaled by the share of its rows it taught."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

import cadence as cd


def brains(**options) -> tuple[cd.Brain, cd.Brain]:
    plain = cd.Brain.compose(6, 4, seed=7, episodic=False, learning_surprise=True, **options)
    shared = cd.Brain.compose(
        6, 4, seed=7, episodic=False, learning_surprise=True, learning_surprise_share=True,
        **options,
    )
    return plain, shared


def test_the_founder_config_leaves_the_setting_out():
    assert "surprise_share" not in cd.LearnerConfig().to_dict()
    assert cd.LearnerConfig(surprise_share=True).to_dict()["surprise_share"] is True
    _, shared = brains()
    assert shared.learner.config.surprise_share is True


def test_a_lesson_moves_the_net_by_the_share_of_its_rows_it_taught():
    plain, shared = brains()
    rng = np.random.default_rng(0)
    x = rng.random((8, 6))
    answer = np.asarray(plain.act(x, greedy=True)).reshape(-1)
    shared.act(x, greedy=True)
    labels = answer.copy()
    labels[:3] = (answer[:3] + 1) % 4  # three of eight rows are wrong
    before = np.asarray(plain.brain.efficacy).copy()
    plain.lesson(x, labels)
    shared.lesson(x, labels)
    full = np.asarray(plain.brain.efficacy) - before
    part = np.asarray(shared.brain.efficacy) - before
    assert np.abs(full).sum() > 0
    np.testing.assert_allclose(part, full * 3 / 8, rtol=1e-9, atol=1e-12)


def test_a_lesson_whose_rows_are_all_wrong_moves_the_net_in_full():
    plain, shared = brains()
    x = np.random.default_rng(1).random((4, 6))
    answer = np.asarray(plain.act(x, greedy=True)).reshape(-1)
    shared.act(x, greedy=True)
    labels = (answer + 1) % 4
    before = np.asarray(plain.brain.efficacy).copy()
    plain.lesson(x, labels)
    shared.lesson(x, labels)
    np.testing.assert_array_equal(np.asarray(plain.brain.efficacy),
                                  np.asarray(shared.brain.efficacy))
    assert np.abs(np.asarray(plain.brain.efficacy) - before).sum() > 0


def test_a_checkpoint_keeps_the_setting(tmp_path):
    _, shared = brains()
    copy = cd.Brain.load(shared.save(tmp_path / "brain.npz"))
    assert copy.learner.config.surprise_share is True
    twin = deepcopy(shared)
    rng = np.random.default_rng(2)
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(twin.step(x, teacher=label), copy.step(x, teacher=label))
