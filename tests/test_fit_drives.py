"""``fit`` teaches each minibatch on the drive the brain's present parameters give its rows. A
learned code's drive reads the weights the lessons move, so a fit equals the explicit loop that
makes each minibatch's drive when its lesson begins: the same rows in the same order, the same
drives, parameters, optimizer history, scores and answers. Without a learned code the drive of
a row is fixed, and the fit equals the same loop too."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import cadence as cd

RNG = np.random.default_rng(197)
X = RNG.random((64, 32))
Y = RNG.integers(0, 4, len(X))
COMPOSITIONS = [
    pytest.param({"learned_code": 12}, id="learned code"),
    pytest.param({"code": 64}, id="fixed code"),
    pytest.param({}, id="no code"),
]
FIELDS = pytest.param({"learned_fields": 4, "recall": False}, id="learned fields")
MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")


def pupil(**options) -> cd.Brain:
    return cd.Brain.compose(32, 4, modules=(16,), backend="cpu", learning_surprise=False,
                            seed=2, **options)


def explicit(brain: cd.Brain, epochs: int, batch: int) -> tuple[list, list[float]]:
    """The fit as a loop of lessons, each minibatch's drive made as its lesson begins."""
    brain.reset()
    taught, scores = [], []
    for _ in range(epochs):
        order = brain.rng.permutation(len(X))
        for start in range(0, len(X), batch):
            rows = order[start:start + batch]
            drive = brain.stimulus(X[rows], memory=False)
            taught.append((drive.copy(), Y[rows].copy()))
            brain.learner.step(drive, Y[rows])
        scores.append(brain.accuracy(X, Y))
    return taught, scores


@pytest.mark.parametrize("options", COMPOSITIONS)
def test_a_fit_is_the_loop_that_makes_each_minibatch_s_drive_at_its_lesson(options):
    fitted, looped = pupil(**options), pupil(**options)
    seen = []
    step = fitted.learner.step

    def recorded(drive, labels, *args, **kwargs):
        seen.append((np.array(drive, copy=True), np.array(labels, copy=True)))
        return step(drive, labels, *args, **kwargs)

    with patch.object(fitted.learner, "step", recorded):
        scores = fitted.fit(X, Y, batch=8, epochs=3)
    taught, expected = explicit(looped, epochs=3, batch=8)
    assert len(seen) == len(taught) == 24 and scores == expected
    for (drive, labels), (wanted, shown) in zip(seen, taught, strict=True):
        np.testing.assert_array_equal(labels, shown)
        np.testing.assert_array_equal(drive, wanted)
    np.testing.assert_array_equal(fitted.brain.efficacy, looped.brain.efficacy)
    np.testing.assert_array_equal(fitted.brain.bias, looped.brain.bias)
    for name in MOMENTS:
        np.testing.assert_array_equal(getattr(fitted.learner, name),
                                      getattr(looped.learner, name), err_msg=name)
    assert fitted.learner.updates == looped.learner.updates == 24
    np.testing.assert_array_equal(fitted.predict(X), looped.predict(X))


def test_a_learned_code_s_drive_follows_the_weights_a_fit_moves():
    """What the fit has to follow: after one lesson the same rows drive the brain otherwise."""
    brain = pupil(learned_code=12)
    before = brain.stimulus(X[:8], memory=False).copy()
    brain.learner.step(before, Y[:8])
    assert np.abs(brain.stimulus(X[:8], memory=False) - before).max() > 1e-6
    plain = pupil()
    fixed = plain.stimulus(X[:8], memory=False).copy()
    plain.learner.step(fixed, Y[:8])
    np.testing.assert_array_equal(plain.stimulus(X[:8], memory=False), fixed)


def eyes(**options) -> cd.Brain:
    """A brain of 16 x 16 frames: learned fields read frames of two dimensions."""
    return cd.Brain.compose((16, 16), 4, modules=(16,), backend="cpu", learning_surprise=False,
                            seed=2, **options)


@pytest.mark.parametrize("options", [*COMPOSITIONS, FIELDS])
def test_a_fit_refuses_its_input_before_anything_changes(options, tmp_path):
    framed = "learned_fields" in options
    brain = eyes(**options) if framed else pupil(**options)
    X = RNG.random((64, 256)) if framed else globals()["X"]
    with np.load(brain.save(tmp_path / "before.npz")) as saved:
        before = {name: saved[name].copy() for name in saved.files}
    bad = X.copy()
    bad[5, 3] = np.nan
    for arguments in ((bad, Y), (X, Y[:-1]), (X[:, :-1], Y), (X, np.full(len(X), 4))):
        with pytest.raises(ValueError):
            brain.fit(*arguments, batch=8, epochs=1)
    for options_ in ({"epochs": -1}, {"batch": 0}):
        with pytest.raises(ValueError):
            brain.fit(X, Y, **options_)
    with np.load(brain.save(tmp_path / "after.npz")) as saved:
        assert set(saved.files) == set(before)
        for name in saved.files:
            np.testing.assert_array_equal(saved[name], before[name], err_msg=name)
