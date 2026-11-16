"""A lesson that leaves a motor slot open (a label of ``OPEN``, -1, in a brain of several
slots): the slot is neither judged nor pulled at that row, a lesson that labels every slot keeps
its bits, and every refusal comes before anything changes."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import cadence as cd
from cadence.learning import OPEN

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
SLOTS = (4, 3)


def parameters(brain: cd.Brain) -> np.ndarray:
    return np.concatenate([brain.brain.efficacy, brain.brain.bias])


def slotted(**settings) -> tuple[cd.Brain, np.ndarray, np.ndarray]:
    """A brain of two motor slots, rows of observations and a label for every slot."""
    brain = cd.Brain.compose(
        10, sum(SLOTS), modules=(16,), seed=5, slots=SLOTS, **{"recall": False, **settings}
    )
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, (6, 10))
    labels = np.stack([rng.integers(0, SLOTS[0], 6), rng.integers(0, SLOTS[1], 6)], axis=1)
    return brain, x, labels


def test_open_is_a_lessons_label_of_a_brain_with_several_slots() -> None:
    brain, _, labels = slotted()
    learner = brain.learner
    opened = labels.copy()
    opened[::2, 1] = OPEN
    assert OPEN == -1
    np.testing.assert_array_equal(learner._labels(opened, open=True), opened)
    with pytest.raises(ValueError, match="outside its output slot"):
        learner._labels(opened)  # a caller that is no lesson (accuracy, fit) labels every slot
    with pytest.raises(ValueError, match="outside its output slot"):
        learner._labels(np.where(opened < 0, -2, opened), open=True)
    whole = opened.copy()
    whole[0] = OPEN
    with pytest.raises(ValueError, match="labels at least one slot"):
        learner._labels(whole, open=True)
    with pytest.raises(ValueError, match="outside its output slot"):
        brain.accuracy(np.zeros((6, 10)), opened)
    single = cd.Brain.compose(10, 4, modules=(16,), seed=5, recall=False)
    with pytest.raises(ValueError, match="outside its output slot"):
        single.lesson(np.zeros((2, 10)), np.array([0, OPEN]))


def test_an_open_slot_holds_what_it_says_in_the_free_phase_and_is_not_pulled() -> None:
    brain, x, labels = slotted()
    learner = brain.learner
    opened = labels.copy()
    opened[:3, 1] = OPEN
    target = learner.targets(opened)
    second = learner.output_index[SLOTS[0]:]
    assert not target[:3][:, second].any()  # no target before the free phase
    assert (target[3:][:, second].sum(axis=1) == 1.0).all()
    free = learner.free(np.asarray(brain.stimulus(x, memory=False), dtype=float))
    filled = learner._open_targets(target, free, opened < 0)
    np.testing.assert_array_equal(filled[3:], target[3:])
    first = learner.output_index[:SLOTS[0]]
    np.testing.assert_array_equal(filled[:, first], target[:, first])
    np.testing.assert_allclose(filled[:3][:, second].sum(axis=1), 1.0)
    # the pull on an open slot is zero at the free state, and the labelled slots keep theirs
    pull = learner.nudge_for(filled, learner.config.beta).drive(np.asarray(free.activation))
    assert np.abs(pull[:3][:, second]).max() < 1e-12
    assert np.abs(pull[:3][:, first]).max() > 1e-4
    assert np.abs(pull[3:][:, second]).max() > 1e-4
    assert learner._open_targets(target, free, None) is target


@pytest.mark.parametrize("place", PLACES)
def test_a_lesson_teaches_the_labelled_slot_and_leaves_the_open_one(place: dict) -> None:
    brain, x, labels = slotted(teaching="normalized", **place)
    # the second slot's labels are none of the brain's own answers, so only lessons give them
    labels[:, 1] = (copy.deepcopy(brain).act(x, greedy=True)[:, 1] + 1) % SLOTS[1]
    other = copy.deepcopy(brain)
    opened = labels.copy()
    opened[:, 1] = OPEN
    for _ in range(200):
        brain.reset()
        brain.lesson(x, opened)
        other.reset()
        other.lesson(x, labels)
    said = copy.deepcopy(brain).act(x, greedy=True)
    shown = copy.deepcopy(other).act(x, greedy=True)
    np.testing.assert_array_equal(said[:, 0], labels[:, 0])
    # lessons that label the second slot teach it; no lesson taught the open slot those labels
    assert (shown[:, 1] == labels[:, 1]).sum() >= 5
    assert (said[:, 1] == labels[:, 1]).sum() < (shown[:, 1] == labels[:, 1]).sum()


def test_a_lesson_does_not_judge_an_open_slot() -> None:
    brain, x, _ = slotted(learning_surprise=True)
    labels = copy.deepcopy(brain).act(x, greedy=True)  # the answer the lesson judges
    wrong = labels.copy()
    wrong[:, 1] = (wrong[:, 1] + 1) % SLOTS[1]
    assert copy.deepcopy(brain).lesson(x, wrong)["agreed_presentations"] == 0.0
    opened = wrong.copy()
    opened[:, 1] = OPEN  # the slot whose label the brain does not give is left open
    kept = parameters(brain).copy()
    report = brain.lesson(x, opened)
    assert report["agreed_presentations"] == 6.0 and report["accepted_presentations"] == 0.0
    assert report["agreed_slots"] == 6.0  # the labelled slots; an open slot shows nothing
    np.testing.assert_array_equal(parameters(brain), kept)
    # a row is taught where a labelled slot differs, whatever its open slot says
    mixed = opened.copy()
    mixed[:2, 0] = (mixed[:2, 0] + 1) % SLOTS[0]
    report = brain.lesson(x, mixed)
    assert report["accepted_presentations"] == 2.0 and report["agreed_presentations"] == 4.0


def test_a_lesson_that_labels_every_slot_keeps_its_bits() -> None:
    brain, x, labels = slotted(recall=True)
    same = copy.deepcopy(brain)
    learner = same.learner
    taken = []
    filling = learner._open_targets
    learner._open_targets = lambda target, free, opened: (  # type: ignore[method-assign]
        taken.append(opened) or filling(target, free, opened)
    )
    for _ in range(3):
        for b in (brain, same):
            b.step(x, teacher=labels)
    assert taken and all(opened is None for opened in taken)
    np.testing.assert_array_equal(parameters(brain), parameters(same))


def test_the_other_lesson_paths_take_open_slots() -> None:
    brain, x, labels = slotted(recall=True)
    opened = labels.copy()
    opened[1::2, 0] = OPEN
    brain.step(x, teacher=opened)  # the streaming lesson
    assert brain.last_learning["demonstrations"] == 6.0
    compiled, _, _ = slotted(recall=True, learning_compiled_moment=True)
    for _ in range(3):
        compiled.step(x, teacher=opened)
    assert compiled.last_learning["demonstration_attempted_presentations"] == 6.0
    rehearsing, _, _ = slotted(recall=True, learning_rehearsal=1.0)
    for _ in range(3):
        rehearsing.step(x, teacher=opened)
    assert (rehearsing.rehearsal_labels == OPEN).any()
    qualified = cd.Learner(
        brain.learner.brain, brain.learner.output_index, slots=SLOTS,
        config=cd.LearnerConfig(qualified=True, tolerance=1e-6, free_steps=400, nudged_steps=400),
    )
    drive = np.asarray(brain.stimulus(x, memory=False), dtype=float)
    learned, report = qualified.step(drive, opened)
    assert report["accepted"] == 1.0 and learned.nudged is not None


def test_open_slots_are_refused_where_a_memory_keeps_every_slot() -> None:
    tokens = cd.Brain.compose(
        10, sum(SLOTS), modules=(16,), seed=5, slots=SLOTS,
        recall_tokens=[0, 1, 2, 3, -1, -1, -1],
    )
    episodic = cd.Brain.compose(
        10, sum(SLOTS), modules=(16,), seed=5, slots=SLOTS, recall=False, arousal=True,
        memory_choice=True,
    )
    x = np.random.default_rng(0).uniform(0, 1, (2, 10))
    opened = np.array([[1, OPEN], [OPEN, 2]])
    for brain, match in ((tokens, "recall_tokens"), (episodic, "episodic")):
        kept = parameters(brain).copy()
        with pytest.raises(ValueError, match=match):
            brain.lesson(x, opened)
        with pytest.raises(ValueError, match=match):
            brain.step(x, teacher=opened)
        np.testing.assert_array_equal(parameters(brain), kept)
        brain.lesson(x, np.array([[1, 0], [2, 2]]))  # every slot labelled: taught as before
