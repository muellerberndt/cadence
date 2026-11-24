"""Independent samples read no bound record: ``predict``, ``accuracy``, ``fit`` and
``stimulus(memory=False)`` give the same drive, answer and score on two brains that differ only
in their records, by name and as rows of the whole width, where ``act`` and
``stimulus(memory=True)`` read what the records bring back."""

from __future__ import annotations

import numpy as np

from cadence import Brain

EYES, EAR = np.eye(16), np.eye(16)
CUE = {"eyes": EYES[[0]]}
CUES = {"eyes": EYES[[0, 3, 5, 9]]}
LABELS = np.array([1, 0, 3, 2])


def reader(**options) -> Brain:
    return Brain.compose(senses={"eyes": 16, "ear": 16}, code=64, bound=8, modules=(8,),
                         actions=4, backend="cpu", seed=0, **options)


def told(brain: Brain) -> Brain:
    """The same brain after it sensed each cue with a sound: records, and no lesson."""
    efficacy = brain.brain.efficacy.copy()
    for k in (0, 3, 5, 9):
        brain.act({"eyes": EYES[[k]], "ear": EAR[[(k + 4) % 16]]}, greedy=True)
    np.testing.assert_array_equal(brain.brain.efficacy, efficacy)
    brain.reset()  # the streams afresh; the records stay
    assert brain.bound.store.count
    return brain


def whole(named: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([named["eyes"], np.zeros((len(named["eyes"]), 16))], axis=1)


def test_a_record_changes_what_the_brain_acts_on_and_not_an_independent_sample():
    fresh, taught = reader(), told(reader())
    assert taught.complete(CUE)["ear"].any()  # the records bring a sound back for the cue
    assert np.abs(taught.stimulus(CUE) - fresh.stimulus(CUE)).max() > 0.01
    for cue in (CUE, whole(CUE), CUES, whole(CUES)):
        np.testing.assert_array_equal(
            taught.stimulus(cue, memory=False), fresh.stimulus(cue, memory=False))
        np.testing.assert_array_equal(taught.predict(cue), fresh.predict(cue))
    np.testing.assert_array_equal(  # by name and as a row: the same independent sample
        taught.stimulus(CUE, memory=False), taught.stimulus(whole(CUE), memory=False))


def test_accuracy_scores_the_same_with_and_without_records():
    fresh, taught = reader(), told(reader())
    for cues in (CUES, whole(CUES)):
        assert taught.accuracy(cues, LABELS) == fresh.accuracy(cues, LABELS)
    assert taught.accuracy(CUES, LABELS) == taught.accuracy(whole(CUES), LABELS)


def test_fit_teaches_and_scores_the_same_with_and_without_records():
    fresh, taught = reader(), told(reader())
    scores = fresh.fit(CUES, LABELS, epochs=3, batch=2)
    assert taught.fit(CUES, LABELS, epochs=3, batch=2) == scores
    np.testing.assert_array_equal(taught.brain.efficacy, fresh.brain.efficacy)
    np.testing.assert_array_equal(taught.brain.bias, fresh.brain.bias)


def test_an_independent_sample_changes_nothing_of_the_brain(tmp_path):
    brain = told(reader())
    with np.load(brain.save(tmp_path / "before.npz")) as saved:
        before = {name: saved[name].copy() for name in saved.files}
    given = brain.__dict__.get("_bound_given")
    brain.predict(CUE)
    brain.accuracy(CUES, LABELS)
    brain.stimulus(CUE, memory=False)
    assert brain.__dict__.get("_bound_given") is given
    with np.load(brain.save(tmp_path / "after.npz")) as saved:
        assert set(saved.files) == set(before)
        for name in saved.files:
            np.testing.assert_array_equal(saved[name], before[name], err_msg=name)
