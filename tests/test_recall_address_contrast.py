"""The address lesson as a contrast (cadence#261): the error is the named keys less the attended
ones, zero where the read is what the lesson names."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd

RNG = np.random.default_rng(0)
X = [RNG.random((2, 6)) for _ in range(10)]


def brain_of(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=True, recall_founder=True, **options)


def stored(brain: cd.Brain, query: np.ndarray) -> cd.Recall:
    """Three one-hot moments in every stream and ``query`` addressed."""
    recall = brain.recall
    for x in np.eye(6)[:3]:
        brain.act(np.repeat(x[None], len(query), axis=0), greedy=True)
    recall.stimulate(np.zeros((len(query), brain.connectome.n)), query)
    return recall


def test_the_lesson_has_no_setting():
    assert "recall_address_contrast" not in brain_of().describe()["genes"]
    with pytest.raises((TypeError, ValueError), match="recall_address_contrast"):
        brain_of(recall_address_contrast=True)
    with pytest.raises((TypeError, ValueError), match="recall_address_contrast"):
        brain_of().retune(recall_address_contrast=True)


def test_the_error_is_the_named_keys_less_the_attended_ones():
    brain = brain_of(recall_recency=0.0, recall_position=0)
    recall = stored(brain, np.eye(6)[[0, 1]])
    moment = recall.last
    d_value = np.eye(6)[[1, 2]]  # the movement names the second and the third slots
    found = recall._row_steps(moment, np.arange(2), d_value, None, 0, None)
    assert found is not None
    error, named, prior = found[4], found[6], found[7]
    keys = recall.keys
    expected = np.einsum("bm,bmk->bk", named - np.exp(prior), keys)
    np.testing.assert_allclose(error, expected)
    assert not np.allclose(error, np.einsum("bm,bmk->bk", named, keys) - moment.address)


def test_a_read_that_is_what_the_lesson_names_takes_no_step():
    brain = brain_of(recall_recency=0.0, recall_position=0, recall_temperature=0.001)
    recall = stored(brain, np.eye(6)[[1]])  # a sharp read lands on the second moment
    before = [a.copy() for a in recall.head_map(0)]
    found = recall._row_steps(recall.last, np.arange(1), np.eye(6)[[1]], None, 0, None)
    assert found is not None
    np.testing.assert_allclose(found[4], 0.0, atol=1e-9)
    assert recall._step(recall.last, np.arange(1), np.eye(6)[[1]], None, found=found)["address_rows"] == 1
    for a, b in zip(before, recall.head_map(0), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_a_checkpoint_continues_the_lessons(tmp_path):
    brain = brain_of()
    rng = np.random.default_rng(2)
    for x in X[:6]:
        brain.step(x, teacher=rng.integers(0, 4, size=2))
    assert brain.recall.head_map(0)[0].any()
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    rng = np.random.default_rng(4)
    for x in X[6:9]:
        label = rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
        for a, b in zip(brain.recall.head_map(0), copy.recall.head_map(0), strict=True):
            np.testing.assert_array_equal(a, b)


def test_reward_takes_the_same_error():
    brain = brain_of(recall_address_reward=0.5)
    before = brain.recall.head_map(0)[0].copy()
    brain.step(X[0])
    for x in X[1:5]:
        brain.step(x, reward=np.array([1.0, -1.0]))
    assert not np.array_equal(before, brain.recall.head_map(0)[0])


def _lessons(brain: cd.Brain, count: int = 6) -> list[dict]:
    rng = np.random.default_rng(7)
    streams = X[0].shape[0]
    return [brain.lesson(x, rng.integers(0, 4, size=streams)) for x in X[:count]]


@pytest.mark.parametrize(
    "options",
    [{"recall_heads": 2},
     {"recall_window": 2}, {"recall_action": True}, {"recall_boundary": True},
     {"recall_anchor": True, "recall_relative": True, "recall_slope": 0.1}],
    ids=["competing", "window", "action", "boundary", "positions"],
)
def test_the_contrast_steps_beside_every_option_of_the_read(options):
    brain = brain_of(**options)
    for x in X[:3]:
        brain.act(x, greedy=True)
    reports = _lessons(brain)
    assert any(r["address_rows"] for r in reports)
    assert all(r["address_refused"] == 0.0 for r in reports)
    for h in range(brain.recall.heads):
        assert all(np.isfinite(a).all() for a in brain.recall.head_map(h))


def test_a_stream_that_began_since_holds_fewer_slots_and_the_contrast_stays_finite():
    brain = brain_of()
    for x in X[:4]:
        brain.act(x, greedy=True)
    brain.recall.reset(2, rows=np.array([1]))  # the second stream holds nothing now
    brain.act(X[4], greedy=True)
    assert not brain.recall.held.all()
    report = brain.lesson(X[5], np.array([1, 2]))
    assert report["address_refused"] == 0.0
    assert all(np.isfinite(a).all() for a in brain.recall.head_map(0))


def test_the_report_gives_the_contrast_as_the_error():
    brain = brain_of(recall_recency=0.0, recall_position=0, recall_temperature=0.001)
    recall = stored(brain, np.eye(6)[[1]])
    found = recall._row_steps(recall.last, np.arange(1), np.eye(6)[[1]], None, 0, None)
    report = recall._step(recall.last, np.arange(1), np.eye(6)[[1]], None, found=found)
    assert report["address_rows"] == 1.0 and report["address_error"] < 1e-9
