"""A lesson's step that follows the running error (cadence#258): with ``running_error`` a
lesson's step is scaled by the running share of lesson rows answered wrong, to a power."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd


def brains(**options) -> tuple[cd.Brain, cd.Brain]:
    plain = cd.Brain.compose(6, 4, seed=7, episodic=False, learning_surprise=True,
                             **{"recall": False, **options})
    tracked = cd.Brain.compose(6, 4, seed=7, episodic=False, learning_surprise=True,
                               learning_running_error=0.9, **{"recall": False, **options})
    return plain, tracked


def lesson(brain: cd.Brain, x: np.ndarray, wrong: int) -> tuple[np.ndarray, float]:
    """The lesson's change of the efficacies and the share of its rows it taught."""
    answer = np.asarray(brain.act(x, greedy=True)).reshape(-1)
    labels = answer.copy()
    labels[:wrong] = (answer[:wrong] + 1) % 4
    before = np.asarray(brain.brain.efficacy).copy()
    report = brain.lesson(x, labels)
    taught = report["accepted_presentations"] / report["attempted_presentations"]
    return np.asarray(brain.brain.efficacy) - before, taught


def test_the_founder_config_leaves_the_setting_out():
    assert "running_error" not in cd.LearnerConfig().to_dict()
    assert "running_error_power" not in cd.LearnerConfig().to_dict()
    assert cd.LearnerConfig(running_error=0.9).to_dict()["running_error"] == 0.9
    _, tracked = brains()
    assert tracked.learner.config.running_error == 0.9
    assert tracked.learner.error_level == 1.0
    with pytest.raises(ValueError):
        cd.LearnerConfig(running_error=1.0)
    with pytest.raises(ValueError):
        cd.LearnerConfig(running_error_power=0.0)


def test_a_lesson_steps_by_the_running_error_to_its_power():
    plain, tracked = brains()
    x = np.random.default_rng(0).random((8, 6))
    full, taught = lesson(plain, x, 2)
    part, same = lesson(tracked, x, 2)
    assert 0 < taught == same < 1
    level = 0.9 + 0.1 * taught  # the running error after one lesson, from one at birth
    assert np.abs(full).sum() > 0
    assert tracked.learner.error_level == pytest.approx(level)
    np.testing.assert_allclose(part, full * level**0.5, rtol=1e-9, atol=1e-12)


def test_the_level_runs_over_every_lesson_right_or_wrong():
    _, tracked = brains()
    rng = np.random.default_rng(1)
    level = 1.0
    shares = []
    for wrong in (0, 3, 0, 0, 1, 4):
        _, taught = lesson(tracked, rng.random((4, 6)), wrong)
        shares.append(taught)
        level = 0.9 * level + 0.1 * taught
        assert tracked.learner.error_level == pytest.approx(level, abs=1e-12)
    assert 0.0 in shares  # a lesson the brain answers right counts as zero


def test_without_the_option_the_brain_is_unchanged_to_the_bit():
    a = cd.Brain.compose(6, 4, seed=3, recall=False)
    b = cd.Brain.compose(6, 4, seed=3, recall=False, learning_running_error=0.0)
    rng = np.random.default_rng(2)
    for _ in range(6):
        x = rng.random((4, 6))
        y = rng.integers(4, size=4)
        for brain in (a, b):
            brain.act(x, greedy=True)
            brain.lesson(x, y)
    np.testing.assert_array_equal(np.asarray(a.brain.efficacy), np.asarray(b.brain.efficacy))
    assert a.learner.error_level == b.learner.error_level == 1.0


def test_lessons_without_surprise_judge_the_free_answer():
    brain = cd.Brain.compose(6, 4, seed=7, recall=False, learning_running_error=0.5)
    x = np.random.default_rng(3).random((4, 6))
    answer = np.asarray(brain.act(x, greedy=True)).reshape(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain.lesson(x, (answer + 1) % 4)
    assert 0.5 <= brain.learner.error_level <= 1.0


def test_the_level_is_saved_grown_and_retuned(tmp_path):
    _, tracked = brains()
    x = np.random.default_rng(4).random((4, 6))
    for _ in range(3):
        lesson(tracked, x, 1)
    level = tracked.learner.error_level
    assert level < 1.0
    again = cd.Brain.load(tracked.save(tmp_path / "tracked.npz"))
    assert again.learner.error_level == level
    assert again.describe()["genes"]["learning_running_error"] == 0.9
    grown = tracked.grow(slots=[4, 4])
    assert grown.learner.error_level == level
    plain = cd.Brain.compose(6, 4, seed=7, recall=False)
    plain.retune(learning_running_error=0.9)
    assert plain.learner.config.running_error == 0.9 and plain.learner.error_level == 1.0
