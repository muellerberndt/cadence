"""The recall memory's anchor (cadence#237): a position term over each slot's place in the
stream's current episode, beside the term over its age."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def anchored(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, seed=4, recall=True, **options)


def lessons(brain: cd.Brain, count: int = 12, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(count):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))


def test_anchor_is_an_option_beside_the_position_term():
    plain = anchored()
    assert not plain.recall.anchor and plain.recall.anchor_maps == []
    brain = anchored(recall_anchor=True)
    count = brain.recall.position_buckets
    assert [m.shape for m in brain.recall.anchor_maps[0]] == [(6, count + 1), (64, count + 1), (count + 1,)]
    assert brain.describe()["memories"]["recall"]["anchor"] is True
    with pytest.raises(ValueError, match="needs position"):
        anchored(recall_position=0, recall_anchor=True)


def test_each_slot_is_placed_in_its_streams_episode():
    brain = anchored(recall_anchor=True)
    recall = brain.recall
    lessons(brain, 3)
    np.testing.assert_array_equal(recall.told, [3, 3])
    np.testing.assert_array_equal(recall.anchored_buckets(), [[0, 1, 2], [0, 1, 2]])
    recall.end(2, rows=np.array([1]))  # stream 1 ends an episode and keeps its slots
    lessons(brain, 1, seed=1)
    np.testing.assert_array_equal(recall.told, [4, 1])
    before = recall.position_buckets  # the bucket of a slot written before the episode began
    np.testing.assert_array_equal(
        recall.anchored_buckets(), [[0, 1, 2, 3], [before, before, before, 0]]
    )


def test_a_newborn_anchor_reads_as_the_memory_without_it():
    rng = np.random.default_rng(1)
    plain, brain = anchored(), anchored(recall_anchor=True)
    for _ in range(4):
        x = rng.random((2, 6))
        np.testing.assert_array_equal(plain.act(x, greedy=True), brain.act(x, greedy=True))
        np.testing.assert_array_equal(
            np.asarray(plain.basal_ganglia.state.activation),
            np.asarray(brain.basal_ganglia.state.activation),
        )


def test_lessons_step_the_anchored_maps_and_a_checkpoint_continues(tmp_path):
    brain = anchored(recall_anchor=True)
    lessons(brain)
    assert np.abs(brain.recall.anchor_maps[0][2]).sum() > 0
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    np.testing.assert_array_equal(copy.recall.told, brain.recall.told)
    rng = np.random.default_rng(2)
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
    for kept, saved in zip(copy.recall.anchor_maps[0], brain.recall.anchor_maps[0], strict=True):
        np.testing.assert_array_equal(kept, saved)


def test_the_anchored_term_enters_the_position_term():
    brain = anchored(recall_anchor=True, recall_recency=0.0)
    recall = brain.recall
    rng = np.random.default_rng(0)
    for _ in range(6):
        brain.act(rng.random((2, 6)))
    e_x, e_c, e = recall.anchor_maps[0]
    bias = np.arange(1.0, len(e) + 1.0)  # one value per anchored bucket
    recall.anchor_maps = [(e_x, e_c, bias)]
    p_x, p_c, p = recall.position_maps[0]
    event, cortex = rng.random((2, 6)), np.zeros((2, p_c.shape[0]))
    table = (event @ p_x + p)[:, recall.buckets()]
    anchored_term = recall.position_term(0, event, cortex) - table
    expected = np.take_along_axis(np.repeat(bias[None], 2, axis=0), recall.anchored_buckets(), axis=1)
    np.testing.assert_allclose(anchored_term, expected)
    assert len(set(expected[0].tolist())) > 1  # the slots sit at different places of the episode
