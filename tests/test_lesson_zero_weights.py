"""Rows of weight zero in a lesson, on a brain that has already learned: a lesson whose rows all
weigh zero is the lesson of no row (no parameter, no optimizer moment, no count of updates and
no demonstration moves, whatever the optimizer's history), a row of weight zero among others
shows no demonstration and its label changes nothing, and the brain learns on afterwards as
if the ignored rows had not been taught."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import cadence as cd

try:
    import torch  # noqa: F401

    TORCH = True
except ImportError:  # pragma: no cover - the CI installs torch with the dev extra
    TORCH = False

PLACES = [
    pytest.param({"backend": "cpu"}, id="cpu"),
    pytest.param(
        {"backend": "torch", "device": "cpu", "precision": "float64"}, id="torch",
        marks=pytest.mark.skipif(not TORCH, reason="torch is not installed"),
    ),
    pytest.param(
        {"backend": "mlx"}, id="mlx",
        marks=pytest.mark.skipif("mlx" not in cd.available_backends(),
                                 reason="MLX is not available"),
    ),
]
X = np.eye(12)[[1, 7, 3]]
STREAMS = {"brain", "meta", "actor/_drive"}  # the report of the latest lesson, what a moment heard


def taught(place: dict, **settings) -> cd.Brain:
    """A brain after one lesson that moved it: its optimizer holds a history."""
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, **place, **settings)
    report = brain.lesson(X, (brain.predict(X) + 1) % 5)
    assert report["scale_step"] > 0 and brain.learner.updates == 1
    return brain


def saved(brain: cd.Brain, path) -> dict[str, np.ndarray]:
    with np.load(brain.save(path)) as held:
        return {name: held[name].copy() for name in held.files}


@pytest.mark.parametrize("place", PLACES)
@pytest.mark.parametrize("surprise", [True, False])
def test_a_lesson_of_weight_zero_moves_nothing_of_a_brain_that_has_learned(
        place, surprise, tmp_path):
    brain = taught(place, learning_surprise=surprise)
    before = saved(brain, tmp_path / "before.npz")
    counts = (brain.learner.updates, brain.learner.contrast_updates)
    report = brain.lesson(X, (brain.predict(X) + 2) % 5, weights=np.zeros(3))
    after = saved(brain, tmp_path / "after.npz")
    assert (brain.learner.updates, brain.learner.contrast_updates) == counts
    assert report["attempted_presentations"] == 3.0 and report["accepted_presentations"] == 0.0
    assert report["nudged_steps"] == report["opposite_steps"] == 0.0
    assert set(before) == set(after)
    for name in set(before) - STREAMS:  # parameters, moments, every memory
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)
    for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
        assert name in before and np.abs(before[name]).max() > 0  # the history it was held at
    counted = json.loads(str(after["brain"]))["last_learning"]
    assert counted["demonstrations"] == 3.0  # the report counts the lesson


@pytest.mark.parametrize("place", PLACES)
def test_the_brain_learns_on_as_if_the_ignored_lesson_had_not_been_taught(place):
    quiet = dict(recall=False, episodic=False, working_memory_amplitude=0.0, history_decays=())
    brain = taught(place, learning_surprise=False, **quiet)
    twin = copy.deepcopy(brain)
    brain.lesson(X, np.array([4, 4, 4]), weights=np.zeros(3))
    labels = (brain.predict(X) + 3) % 5
    assert brain.lesson(X, labels) == twin.lesson(X, labels)
    np.testing.assert_array_equal(brain.brain.efficacy, twin.brain.efficacy)
    np.testing.assert_array_equal(brain.brain.bias, twin.brain.bias)
    assert brain.learner.updates == twin.learner.updates == 2


def test_weights_of_zero_are_refused_as_any_weights_are():
    brain = taught({"backend": "cpu"})
    for weights in (np.zeros(2), np.array([0.0, np.nan, 0.0]), np.array([0.0, -1.0, 0.0])):
        with pytest.raises(ValueError, match="one finite nonnegative number per row"):
            brain.lesson(X, np.array([0, 1, 2]), weights=weights)
    drive = brain.stimulus(X, memory=False)
    with pytest.raises(ValueError, match="one entry per batch row"):
        brain.learner.step(drive, np.array([0, 1, 2]), weight=np.zeros(2))
    with pytest.raises(ValueError, match="must be finite"):
        brain.learner.step(drive, np.array([0, 1, 2]), weight=np.array([0.0, np.inf, 0.0]))
    assert brain.learner.updates == 1


def test_the_rows_surprise_teaches_pull_nothing_where_they_weigh_zero():
    """The learner judges by its own free phase: the rows that differ weigh zero, the row that
    weighs agrees, so no nudged phase runs and nothing moves."""
    brain = taught({"backend": "cpu"}, learning_surprise=True)
    drive = brain.stimulus(X, memory=False)
    answer = brain.predict(X)
    labels = answer.copy()
    labels[1:] = (labels[1:] + 1) % 5
    before = (brain.brain.efficacy.copy(), brain.learner.velocity.copy(), brain.learner.updates)
    learned, report = brain.learner.step(drive, labels, weight=np.array([1.0, 0.0, 0.0]))
    assert learned.taught is not None and not learned.taught.any()
    assert report["accepted_presentations"] == 0.0 and report["nudged_steps"] == 0.0
    np.testing.assert_array_equal(brain.brain.efficacy, before[0])
    np.testing.assert_array_equal(brain.learner.velocity, before[1])
    assert brain.learner.updates == before[2]


def moved(rows, labels, weights, **settings) -> np.ndarray:
    """How far one lesson moves a new brain's synapses."""
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, backend="cpu",
                             learning_surprise=False, **settings)
    start = brain.brain.efficacy.copy()
    brain.lesson(X[rows], np.array(labels), weights=weights)
    return np.asarray(brain.brain.efficacy - start)


@pytest.mark.parametrize("settings", [{}, {"teaching": "founder"}])
def test_the_label_of_a_row_of_weight_zero_changes_nothing(settings):
    beside = moved([0, 1], [2, 0], np.array([1.0, 0.0]), **settings)
    assert np.abs(beside).max() > 0
    np.testing.assert_array_equal(beside, moved([0, 1], [2, 4], np.array([1.0, 0.0]), **settings))


def test_a_row_of_weight_zero_adds_nothing_whatever_it_shows():
    """At a new brain's teaching point the phases run their sweeps whatever the rows hold, so
    another observation in the row of weight zero leaves the lesson as it is. (Phases that
    stop on the batch's movement, as the unnormalized point's do, end where their slowest row
    ends: there a row's observation can move the sweep the others are read at.)"""
    beside = moved([0, 1], [2, 0], np.array([1.0, 0.0]))
    np.testing.assert_array_equal(beside, moved([0, 2], [2, 0], np.array([1.0, 0.0])))


def test_a_weight_common_to_every_row_divides_out_of_the_step_s_size_under_normalization():
    """The weights act through their ratios at the normalized point a new brain is taught at:
    three times the weights leave the size of the step, where the unnormalized point's step
    grows with them."""
    rows, labels, weights = [0, 1, 2], [2, 0, 4], np.array([1.0, 2.0, 0.5])

    def grown(**settings) -> float:
        once = np.linalg.norm(moved(rows, labels, weights, **settings))
        thrice = np.linalg.norm(moved(rows, labels, 3.0 * weights, **settings))
        return float(thrice / once)

    assert 0.8 < grown() < 1.25
    assert grown(teaching="founder") > 2.0


def test_mixed_weights_continue_alike_after_a_save_and_a_load(tmp_path):
    brain = taught({"backend": "cpu"}, learning_surprise=False)
    brain.lesson(X, np.array([0, 1, 2]), weights=np.array([2.0, 0.0, 1.0]))
    again = cd.Brain.load(brain.save(tmp_path / "between.npz"))
    for held in (brain, again):
        held.lesson(X, np.array([3, 3, 3]), weights=np.array([0.0, 1.0, 0.5]))
    np.testing.assert_array_equal(brain.brain.efficacy, again.brain.efficacy)
    np.testing.assert_array_equal(brain.learner.velocity, again.learner.velocity)


# -- the demonstrations an episodic memory records


EPISODIC = [
    pytest.param({}, id="records"),
    pytest.param({"memory_context_hold": "unchanged"}, id="context held"),
]


def recorded(brain: cd.Brain, path) -> dict[str, np.ndarray]:
    return {name: value for name, value in saved(brain, path).items()
            if name.startswith("memories/")}


@pytest.mark.parametrize("settings", EPISODIC)
def test_a_lesson_of_weight_zero_records_no_demonstration(settings, tmp_path):
    brain = cd.Brain.compose(12, 5, modules=(16,), memory_choice=True, learning_surprise=False,
                             backend="cpu", seed=0, **settings)
    cues = np.eye(12)[[0, 1]]
    before = recorded(brain, tmp_path / "before.npz")
    brain.lesson(cues, [3, 4], weights=[0.0, 0.0])
    after = recorded(brain, tmp_path / "after.npz")
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)
    assert brain.hippocampus.read(cues)[[0, 1], [3, 4]].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("settings", EPISODIC)
def test_a_row_of_weight_zero_among_others_records_no_demonstration(settings):
    def votes(weights) -> list[float]:
        brain = cd.Brain.compose(12, 5, modules=(16,), memory_choice=True,
                                 learning_surprise=False, backend="cpu", seed=0, **settings)
        cues = np.eye(12)[[0, 1]]
        brain.lesson(cues, [3, 4], weights=weights)
        return brain.hippocampus.read(cues)[[0, 1], [3, 4]].tolist()

    shown, ignored = votes([1.0, 0.0])
    assert shown > 0 and ignored == 0.0
    assert votes([1.0, 1.0])[1] > 0  # at weight one the second row is a demonstration
