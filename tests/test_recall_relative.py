"""A read placed relative to the previous read (cadence#234): a position term over each slot's
age relative to the slot the stream's previous read attended most."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def relative(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, seed=5, recall=True, **options)


def test_relative_is_an_option_beside_the_position_term():
    plain = relative()
    assert not plain.recall.relative and plain.recall.relative_maps == []
    brain = relative(recall_relative=True)
    width = 2 * cd.Recall.RELATIVE + 2
    assert [m.shape for m in brain.recall.relative_maps[0]] == [(6, width), (64, width), (width,)]
    assert brain.describe()["memories"]["recall"]["relative"] is True
    with pytest.raises(ValueError, match="needs position"):
        relative(recall_position=0, recall_relative=True)


def test_the_focus_follows_the_slot_the_read_attended_most():
    brain = relative(recall_relative=True)
    recall = brain.recall
    rng = np.random.default_rng(0)
    brain.act(rng.random((2, 6)))
    np.testing.assert_array_equal(recall.focus, [-1, -1])  # no slot to read at the first moment
    np.testing.assert_array_equal(recall.relative_buckets(), [[17], [17]])
    brain.act(rng.random((2, 6)))
    np.testing.assert_array_equal(recall.focus, [1, 1])  # the one slot, one moment older
    brain.act(rng.random((2, 6)))
    attended = recall.attended_heads[0]  # this act's read, over the two slots held before its write
    expected = recall.ages()[: attended.shape[1]][np.argmax(attended, axis=1)]
    np.testing.assert_array_equal(recall.focus, expected)
    offsets = recall.ages()[None, :] - expected[:, None]
    np.testing.assert_array_equal(recall.relative_buckets(), offsets + cd.Recall.RELATIVE)


def test_a_newborn_relative_read_reads_as_the_memory_without_it():
    rng = np.random.default_rng(1)
    plain, brain = relative(), relative(recall_relative=True)
    for _ in range(4):
        x = rng.random((2, 6))
        np.testing.assert_array_equal(plain.act(x, greedy=True), brain.act(x, greedy=True))
        np.testing.assert_array_equal(
            np.asarray(plain.basal_ganglia.state.activation),
            np.asarray(brain.basal_ganglia.state.activation),
        )


def test_lessons_step_the_relative_maps_and_a_checkpoint_continues(tmp_path):
    brain = relative(recall_relative=True)
    rng = np.random.default_rng(2)
    for _ in range(12):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))
    assert np.abs(brain.recall.relative_maps[0][2]).sum() > 0
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    np.testing.assert_array_equal(copy.recall.focus, brain.recall.focus)
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))


def test_the_relative_term_enters_the_position_term():
    brain = relative(recall_relative=True, recall_recency=0.0)
    recall = brain.recall
    rng = np.random.default_rng(0)
    for _ in range(6):
        brain.act(rng.random((2, 6)))
    assert (recall.focus >= 0).all()  # a read before: the offsets are relative to it
    q_x, q_c, q = recall.relative_maps[0]
    bias = np.arange(1.0, len(q) + 1.0)  # one value per relative bucket
    recall.relative_maps = [(q_x, q_c, bias)]
    p_x, p_c, p = recall.position_maps[0]
    event, cortex = rng.random((2, 6)), np.zeros((2, p_c.shape[0]))
    table = (event @ p_x + p)[:, recall.buckets()]
    relative_term = recall.position_term(0, event, cortex) - table
    expected = np.take_along_axis(np.repeat(bias[None], 2, axis=0), recall.relative_buckets(), axis=1)
    np.testing.assert_allclose(relative_term, expected)
    assert len(set(expected[0].tolist())) > 1
