"""A lesson of a distribution per row (``targets``) and of weighted rows (``weights``), both
opt-in: a lesson without them, or with every weight one, is the lesson without either to the
bit; a one-hot distribution teaches as its label does; a distribution teaches its shares; a row of
weight zero adds nothing to the contrast; and every refusal comes before anything changes."""

from __future__ import annotations

import copy
import warnings

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
        {"backend": "torch", "device": "cpu"}, id="torch",
        marks=pytest.mark.skipif(not TORCH, reason="torch is not installed"),
    ),
]


def parameters(brain: cd.Brain) -> np.ndarray:
    return np.concatenate([brain.brain.efficacy, brain.brain.bias])


def streaming(place: dict, **settings) -> tuple[cd.Brain, np.ndarray, np.ndarray]:
    """A brain whose streams are live (a greedy act holds nothing for reward), its rows and
    their labels."""
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, **place, **settings)
    rng = np.random.default_rng(1)
    x = (rng.random((6, 12)) < 0.3).astype(float)
    labels = rng.integers(0, 5, 6)
    brain.act((rng.random((6, 12)) < 0.3).astype(float), greedy=True)
    return brain, x, labels


@pytest.mark.parametrize("place", PLACES)
@pytest.mark.parametrize("surprise", [False, True])
def test_weights_of_one_are_the_lesson_without_weights(place, surprise):
    brain, x, labels = streaming(place, learning_surprise=surprise)
    twin = copy.deepcopy(brain)
    report = brain.lesson(x, labels)
    weighted = twin.lesson(x, labels, weights=np.ones(len(x)))
    assert report == weighted
    np.testing.assert_array_equal(parameters(brain), parameters(twin))
    np.testing.assert_array_equal(brain.act(x, greedy=True), twin.act(x, greedy=True))


@pytest.mark.parametrize("place", PLACES)
def test_a_one_hot_distribution_teaches_as_its_label_does(place):
    brain, x, labels = streaming(place, learning_surprise=False)
    twin = copy.deepcopy(brain)
    report = brain.lesson(x, labels)
    shown = twin.lesson(x, targets=np.eye(5)[labels])
    assert report == shown
    np.testing.assert_array_equal(parameters(brain), parameters(twin))
    np.testing.assert_array_equal(brain.act(x, greedy=True), twin.act(x, greedy=True))


def test_a_one_hot_distribution_per_slot_teaches_as_its_labels_do():
    judge = cd.Brain.compose(7, 4, slots=(2, 2), modules=(16,), seed=0, learning_surprise=False)
    cells = np.eye(7)[[1, 2, 4, 5]]
    labels = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])
    twin = copy.deepcopy(judge)
    judge.lesson(cells, labels)
    twin.lesson(cells, targets=np.concatenate([np.eye(2)[labels[:, 0]],
                                               np.eye(2)[labels[:, 1]]], axis=1))
    np.testing.assert_array_equal(parameters(judge), parameters(twin))


@pytest.mark.parametrize("place", PLACES)
def test_targets_teach_every_row_under_surprise(place):
    """A distribution has no single answer to agree with: under ``learning_surprise`` a lesson of
    targets teaches every row, as a lesson of labels does without the surprise rule."""
    judged, x, labels = streaming(place, learning_surprise=True)
    every, _, _ = streaming(place, learning_surprise=False)
    report = judged.lesson(x, targets=np.eye(5)[labels])
    every.lesson(x, labels)
    assert report["accepted_presentations"] == len(x)
    np.testing.assert_array_equal(parameters(judged), parameters(every))


def test_a_distribution_teaches_its_shares():
    brain = cd.Brain.compose(12, 5, seed=3)
    x = np.eye(12)[[2]]
    shares = np.array([[0.6, 0.0, 0.4, 0.0, 0.0]])
    for _ in range(40):
        brain.lesson(x, targets=shares)
    free = brain.learner.free(brain.stimulus(x, memory=False))
    motor = np.asarray(free.activation)[0, brain.learner.output_index]
    assert motor[0] > motor[2] > max(motor[1], motor[3], motor[4])


@pytest.mark.parametrize("place", PLACES)
def test_a_row_of_weight_zero_adds_nothing_to_the_contrast(place):
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, **place)
    drive = brain.stimulus(np.eye(12)[[1, 7]], memory=False)
    learned, _ = brain.learner.step(drive, np.array([0, 3]), weight=np.array([1.0, 0.0]))
    nudged, opposite = np.asarray(learned.nudged.activation), np.asarray(learned.opposite.activation)
    np.testing.assert_array_equal(nudged[1], opposite[1])
    assert not np.array_equal(nudged[0], opposite[0])


@pytest.mark.parametrize("place", PLACES)
def test_a_lesson_of_weight_zero_teaches_nothing(place, tmp_path):
    """Neither the graph, the optimizer's history nor the recall memory's maps move; the
    streams heard the moment, and the report counts the lesson."""
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, **place)
    brain.act(np.eye(12)[[2, 5, 9]], greedy=True)
    before = brain.save(tmp_path / "before.npz")
    report = brain.lesson(np.eye(12)[[1, 7, 3]], np.array([0, 1, 2]), weights=np.zeros(3))
    after = brain.save(tmp_path / "after.npz")
    assert report["address_rows"] == 0.0
    streams = {"actor/_drive", "brain", "meta"}  # what the act heard, the report's counts
    with np.load(before) as old, np.load(after) as new:
        assert set(old.files) - streams == set(new.files) - streams
        moved = {k for k in set(old.files) - streams if not np.array_equal(old[k], new[k])}
    assert moved == set()


def test_weights_move_the_brain_toward_the_heavier_rows():
    """Two rows of one observation with different labels: the heavier row's label gains on the
    other."""
    x = np.eye(12)[[4, 4]]
    margins = []
    for weights in ([3.0, 1.0], [1.0, 1.0], [1.0, 3.0]):
        brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, learning_surprise=False)
        for _ in range(30):
            brain.lesson(x, np.array([1, 3]), weights=np.array(weights))
        free = brain.learner.free(brain.stimulus(x[:1], memory=False))
        motor = np.asarray(free.activation)[0, brain.learner.output_index]
        margins.append(motor[1] - motor[3])
    assert margins[0] > margins[1] > margins[2]


def test_targets_record_no_demonstration_in_an_episodic_memory(tmp_path):
    def records(brain: cd.Brain, name: str) -> dict[str, np.ndarray]:
        with np.load(brain.save(tmp_path / name)) as saved:
            return {k: saved[k] for k in saved.files if k.startswith("memories/episodic")}

    brain, x, labels = streaming({"backend": "cpu"}, memory_choice=True, learning_surprise=False)
    shown = copy.deepcopy(brain)
    before = records(brain, "before.npz")
    brain.lesson(x, targets=np.eye(5)[labels])
    after = records(brain, "after.npz")
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)
    shown.lesson(x, labels)  # a label is a demonstration, and the memory records it
    labelled = records(shown, "labelled.npz")
    assert any(
        labelled[k].shape != before[k].shape or not np.array_equal(labelled[k], before[k])
        for k in before
    ) or labelled.keys() != before.keys()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "labels or targets"),
        ({"labels": [0, 1, 2], "targets": np.eye(5)[[0, 1, 2]]}, "labels or targets"),
        ({"targets": np.eye(5)[[0, 1]]}, "match the observation batch"),
        ({"targets": np.eye(4)[[0, 1, 2]]}, "one probability per output"),
        ({"targets": np.full((3, 5), 0.3)}, "sum to one"),
        ({"targets": np.tile([1.5, -0.5, 0, 0, 0], (3, 1))}, "nonnegative"),
        ({"targets": np.full((3, 5), np.nan)}, "finite"),
        ({"labels": [0, 1, 2], "weights": [1.0, -1.0, 1.0]}, "nonnegative"),
        ({"labels": [0, 1, 2], "weights": [1.0, 1.0]}, "one finite"),
        ({"labels": [0, 1, 2], "weights": [1.0, np.inf, 1.0]}, "one finite"),
    ],
)
def test_a_refused_lesson_changes_nothing(arguments, message):
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3)
    x = np.eye(12)[[1, 7, 3]]
    before = parameters(brain)
    with pytest.raises(ValueError, match=message):
        brain.lesson(x, **arguments)
    np.testing.assert_array_equal(parameters(brain), before)
    assert brain.last_learning == {}


def test_the_rehearsal_reads_labels_at_weight_one():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, learning_rehearsal=1.0)
    x = np.eye(12)[[1, 7, 3]]
    with pytest.raises(ValueError, match="learning_rehearsal"):
        brain.lesson(x, targets=np.eye(5)[[0, 1, 2]])
    with pytest.raises(ValueError, match="learning_rehearsal"):
        brain.lesson(x, [0, 1, 2], weights=[2.0, 1.0, 1.0])
    brain.lesson(x, [0, 1, 2], weights=np.ones(3))  # weight one is the lesson without weights


def test_declared_tokens_judge_labels():
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, recall_tokens=(0, 1, 2, -1, -1))
    with pytest.raises(ValueError, match="recall_tokens"):
        brain.lesson(np.eye(12)[[1]], targets=np.eye(5)[[0]])
