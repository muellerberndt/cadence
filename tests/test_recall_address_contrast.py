"""The address lesson as a contrast (``recall_address_contrast``, cadence#261): the error is the
named keys less the attended ones, zero where the read is what the lesson names."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd

RNG = np.random.default_rng(0)
X = [RNG.random((2, 6)) for _ in range(10)]
ON = {"recall_address_contrast": True}


def brain_of(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=True, **options)


def stored(brain: cd.Brain, query: np.ndarray) -> cd.Recall:
    """Three one-hot moments in every stream and ``query`` addressed."""
    recall = brain.recall
    for x in np.eye(6)[:3]:
        brain.act(np.repeat(x[None], len(query), axis=0), greedy=True)
    recall.stimulate(np.zeros((len(query), brain.connectome.n)), query)
    return recall


def test_it_is_an_option_that_adds_no_parameter_and_a_newborn_reads_as_without_it():
    plain, brain = brain_of(), brain_of(**ON)
    assert not plain.recall.address_contrast and brain.recall.address_contrast
    assert brain.parameters() == plain.parameters()
    assert brain.describe()["genes"]["recall_address_contrast"] is True
    assert "recall_address_contrast" not in plain.describe()["genes"]
    with pytest.raises(ValueError, match="must be boolean"):
        brain_of(recall_address_contrast=1)
    for x in X[:5]:
        np.testing.assert_array_equal(plain.act(x, greedy=True), brain.act(x, greedy=True))
    np.testing.assert_array_equal(plain.recall.attended_heads[0], brain.recall.attended_heads[0])


def test_the_error_is_the_named_keys_less_the_attended_ones():
    brain = brain_of(**ON, recall_recency=0.0, recall_position=0)
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
    brain = brain_of(**ON, recall_recency=0.0, recall_position=0, recall_temperature=0.001)
    recall = stored(brain, np.eye(6)[[1]])  # a sharp read lands on the second moment
    before = [a.copy() for a in recall.head_map(0)]
    found = recall._row_steps(recall.last, np.arange(1), np.eye(6)[[1]], None, 0, None)
    assert found is not None
    np.testing.assert_allclose(found[4], 0.0, atol=1e-9)
    assert recall._step(recall.last, np.arange(1), np.eye(6)[[1]], None, found=found)["address_rows"] == 1
    for a, b in zip(before, recall.head_map(0), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_lessons_differ_from_the_distance_and_a_checkpoint_continues(tmp_path):
    def taught(**options) -> cd.Brain:
        brain = brain_of(**options)
        rng = np.random.default_rng(2)
        for x in X[:6]:
            brain.step(x, teacher=rng.integers(0, 4, size=2))
        return brain

    plain, brain = taught(), taught(**ON)
    assert not np.array_equal(plain.recall.head_map(0)[0], brain.recall.head_map(0)[0])
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert copy.recall.address_contrast
    rng = np.random.default_rng(4)
    for x in X[6:9]:
        label = rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
        for a, b in zip(brain.recall.head_map(0), copy.recall.head_map(0), strict=True):
            np.testing.assert_array_equal(a, b)


def test_a_retune_takes_it_and_acts_as_built():
    rng = np.random.default_rng(2)
    labels = [rng.integers(0, 4, size=2) for _ in range(6)]
    built, retuned = brain_of(**ON), brain_of()
    retuned.retune(recall_address_contrast=True)
    for x, label in zip(X[:6], labels, strict=True):
        np.testing.assert_array_equal(built.step(x, teacher=label), retuned.step(x, teacher=label))
    np.testing.assert_array_equal(built.recall.head_map(0)[0], retuned.recall.head_map(0)[0])


def test_it_is_held_while_a_sampled_actions_address_contrast_is_pending():
    brain = brain_of(recall_address_reward=0.5)
    brain.step(X[0])
    with pytest.raises(ValueError, match="cannot change while sampled feedback"):
        brain.retune(recall_address_contrast=True)


def test_reward_takes_the_same_error():
    brain = brain_of(**ON, recall_address_reward=0.5)
    before = brain.recall.head_map(0)[0].copy()
    brain.step(X[0])
    for x in X[1:5]:
        brain.step(x, reward=np.array([1.0, -1.0]))
    assert not np.array_equal(before, brain.recall.head_map(0)[0])


def test_a_grown_head_and_a_graft_keep_it():
    brain = brain_of(**ON)
    rng = np.random.default_rng(3)
    for x in X[:3]:
        brain.step(x, teacher=rng.integers(0, 4, size=2))
    assert brain.grow(recall_heads=2).recall.address_contrast
    assert brain.graft(brain_of(), ["association", "motor"]).recall.address_contrast
