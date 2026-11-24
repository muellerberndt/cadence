"""A correction the brain would refuse at any step is refused whole, and one a settle ends
leaves the brain at rest (cadence.speech.correct.Rehearsal.correct)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from cadence import Brain
from cadence.speech.correct import Rehearsal


def brain(seed: int = 0, **options: Any) -> Brain:
    return Brain.compose(8, 4, seed=seed, backend="cpu", learning_surprise=False, **options)


def utterance(rng: np.random.Generator, steps: int) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 4, size=steps)
    rows = np.eye(4)[labels].repeat(2, axis=1) + 0.1 * rng.standard_normal((steps, 8))
    return rows, labels


def efficacy(of: Brain) -> np.ndarray:
    return np.array(of.brain.efficacy, copy=True)


def test_a_label_of_the_last_step_that_the_brain_lacks_teaches_no_step_before_it():
    rng = np.random.default_rng(0)
    rows, labels = utterance(rng, 6)
    labels[-1] = 4  # a fifth action of four, at the last step
    pupil, twin = brain(), brain()
    before = efficacy(pupil)
    with pytest.raises(ValueError, match="a label lies outside its output slot's choices"):
        Rehearsal(rows=0).correct(pupil, rows, labels)
    assert pupil.learner.updates == 0 and pupil.basal_ganglia.state is None
    np.testing.assert_array_equal(efficacy(pupil), before)
    labels[-1] = 3  # the label set right, the utterance is taught once
    Rehearsal(rows=0).correct(pupil, rows, labels)
    Rehearsal(rows=0).correct(twin, rows, labels)
    assert pupil.learner.updates == twin.learner.updates == 6
    np.testing.assert_array_equal(efficacy(pupil), efficacy(twin))


Banks = tuple[Rehearsal, Rehearsal]


def a_label_the_brain_lacks(rng: np.random.Generator, banks: Banks) -> tuple[Any, ...]:
    rows, labels = utterance(rng, 6)
    labels[-1] = 4
    return (rows, labels), {}, "a label lies outside its output slot's choices"


def rows_of_another_width(rng: np.random.Generator, banks: Banks) -> tuple[Any, ...]:
    rows, labels = utterance(rng, 6)
    return (rows[:, :7], labels), {}, "observations need 8 values per row, got 7"


def targets_over_three_actions(rng: np.random.Generator, banks: Banks) -> tuple[Any, ...]:
    rows, _ = utterance(rng, 6)
    return (rows,), {"targets": np.full((6, 3), 1 / 3)}, "targets must have shape"


def targets_of_a_late_step_off_one(rng: np.random.Generator, banks: Banks) -> tuple[Any, ...]:
    rows, labels = utterance(rng, 6)
    targets = np.eye(4)[labels]
    targets[4] = [0.25005, 0.25, 0.25, 0.25]  # one as the bank reads it, not as the brain does
    return (rows,), {"targets": targets}, "target probabilities must sum to one"


def a_banked_label_the_brain_lacks(rng: np.random.Generator, banks: Banks) -> tuple[Any, ...]:
    rows, _ = utterance(rng, 6)
    for bank in banks:  # kept for a brain of five actions, and of the correction's own length
        bank.remember(rows, np.full(6, 4))
    return utterance(rng, 6), {}, "the brain refuses the lessons beside the bank's utterances"


@pytest.mark.parametrize(
    "case",
    [a_label_the_brain_lacks, rows_of_another_width, targets_over_three_actions,
     targets_of_a_late_step_off_one, a_banked_label_the_brain_lacks],
)
def test_a_refused_correction_leaves_a_living_brain_and_the_bank_as_they_were(case):
    rng = np.random.default_rng(1)
    # the twin is given the same utterances and never asked for a correction
    bank, twin = (Rehearsal(rows=2, weight=0.25, seed=3) for _ in range(2))
    for _ in range(3):
        earlier = utterance(rng, 9)
        bank.remember(*earlier)
        twin.remember(*earlier)
    given, named, message = case(rng, (bank, twin))
    pupil = brain(seed=1)
    moment = utterance(rng, 1)[0]
    pupil.act(moment)  # a sampled action awaits its outcome
    stream, before = pupil.basal_ganglia.state, efficacy(pupil)
    assert pupil.pending_feedback
    with pytest.raises(ValueError, match=message):
        bank.correct(pupil, *given, **named)
    assert pupil.basal_ganglia.state is stream and pupil.pending_feedback
    assert pupil.learner.updates == 0
    np.testing.assert_array_equal(efficacy(pupil), before)
    for (rows, labels), (theirs, said) in zip(bank.chosen(6), twin.chosen(6), strict=True):
        np.testing.assert_array_equal(rows, theirs)  # the bank has drawn nothing
        np.testing.assert_array_equal(labels, said)
    pupil.step(moment, reward=[1.0])  # the outcome is still the brain's to take


def test_a_settle_refused_at_a_later_step_keeps_the_lessons_before_it_and_rests_the_brain():
    labels = np.array([0, 1, 2, 3, 0, 1])
    rows = np.zeros((6, 8))
    rows[2:] = np.eye(4)[labels[2:]].repeat(2, axis=1)  # two silent steps, then sound
    # one sweep and a rescue of one settle the silent steps and not the third
    pupil, by_hand = (brain(learning_free_steps=1) for _ in range(2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="brain did not settle") as refused:
            Rehearsal(rows=0).correct(pupil, rows, labels)
        for t in range(3):  # the three lessons given before the act that refused
            by_hand.lesson(rows[t : t + 1], labels[t : t + 1])
            if t < 2:
                by_hand.act(rows[t : t + 1], greedy=True)
    assert pupil.basal_ganglia.state is None
    assert pupil.learner.updates == by_hand.learner.updates == 3
    np.testing.assert_array_equal(efficacy(pupil), efficacy(by_hand))
    assert refused.value.__notes__ == [
        "the correction stopped at step 2 of 6: what its first 3 steps taught stays learned, "
        "and the brain is at rest"
    ]
    pupil.retune(learning_free_steps=1024)  # a budget that settles: the rest, each step once
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = Rehearsal(rows=0).correct(pupil, rows[3:], labels[3:])
    assert report["steps"] == 3 and report["refused"] == 0
    assert pupil.learner.updates == 6 and pupil.basal_ganglia.state is None
