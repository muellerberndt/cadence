"""The recall memory's boundary (``recall_boundary``): a stream that begins holds a silent slot
first, zero key and zero value, whose next the stream's first event teaches. The start of an
episode then has a predecessor as every later event has, so a cue can read it through either
port. The founder has none; the option is described and saved where it is on.
"""
from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def composed(**options):
    return cd.Brain.compose(8, 3, modules=(16,), seed=3, episodic=False, recall=True, recall_founder=True, **options)


def event(*symbols):
    return np.eye(8)[list(symbols)]


def test_the_founder_has_none_and_the_option_is_described_where_on():
    founder = composed()
    assert founder.recall.boundary is False and cd.Recall.options()["boundary"] is False
    assert "recall_boundary" not in founder.describe()["genes"]
    assert composed(recall_boundary=True).describe()["genes"]["recall_boundary"] is True
    with pytest.raises(ValueError, match="boundary must be boolean"):
        composed(recall_boundary=1)


def test_the_first_event_teaches_the_boundary_its_next():
    brain = composed(recall_boundary=True)
    for s in (2, 5, 1):
        brain.act(event(s), greedy=True)
    recall = brain.recall
    assert recall.slots == 4 and recall.writes == 3  # the boundary is no written moment
    assert not recall.keys[0, 0].any() and not recall.values[0, 0].any()
    np.testing.assert_array_equal(recall.nexts[0, :3], event(2, 5, 1))
    np.testing.assert_array_equal(recall.values[0, 1:], event(2, 5, 1))
    brain.reset()  # the stream begins again at a boundary of its own
    brain.act(event(7), greedy=True)
    assert recall.slots == 2
    np.testing.assert_array_equal(recall.nexts[0, 0], event(7)[0])


def test_the_slots_after_the_boundary_are_the_founders():
    founder, bounded = composed(), composed(recall_boundary=True)
    for s in (4, 0, 6, 6):
        for brain in (founder, bounded):
            brain.act(event(s), greedy=True)
    for name in ("keys", "values", "held"):
        np.testing.assert_array_equal(getattr(bounded.recall, name)[:, 1:], getattr(founder.recall, name))
    np.testing.assert_array_equal(bounded.recall.nexts[:, 1:], founder.recall.nexts)


def test_a_stream_that_begins_alone_holds_its_boundary_in_the_latest_slot():
    brain = composed(recall_boundary=True)
    for x in (event(1, 2, 3), event(4, 5, 6)):
        brain.act(x, greedy=True)
    recall = brain.recall
    assert recall.slots == 3
    recall.reset(3, rows=np.array([1]))
    assert recall.slots == 3  # the other streams' slots keep their ages
    np.testing.assert_array_equal(recall.held, [[True] * 3, [False, False, True], [True] * 3])
    assert not recall.keys[1, -1].any() and not recall.values[1, -1].any()
    assert not recall.nexts[1, -1].any()
    np.testing.assert_array_equal(recall.values[[0, 2], -1], event(4, 6))
    brain.act(event(7, 0, 2), greedy=True)
    np.testing.assert_array_equal(recall.nexts[:, 2], event(7, 0, 2))


def test_an_end_of_every_stream_opens_one_boundary():
    brain = composed(recall_boundary=True)
    for s in (2, 5):
        brain.act(event(s), greedy=True)
    brain.reset(keep=("slots",))
    brain.reset(keep=("slots",))  # a second end before any moment keeps the one boundary
    recall = brain.recall
    assert recall.slots == 4 and not recall.ended.any()
    brain.act(event(6), greedy=True)
    assert not recall.nexts[0, 2].any()  # the closed episode's latest slot learns no successor
    np.testing.assert_array_equal(recall.nexts[0, 3], event(6)[0])


def test_an_end_of_some_streams_keeps_the_founders_rule():
    brain = composed(recall_boundary=True)
    brain.act(event(1, 2), greedy=True)
    recall = brain.recall
    recall.end(2, rows=np.array([0]))
    assert recall.slots == 2 and recall.ended.tolist() == [True, False]
    brain.act(event(3, 4), greedy=True)
    assert not recall.nexts[0, 1].any()
    np.testing.assert_array_equal(recall.nexts[1, 1], event(4)[0])


def test_the_boundary_is_saved_loaded_and_retuned(tmp_path):
    brain = composed(recall_boundary=True)
    for s in (2, 5):
        brain.act(event(s), greedy=True)
    loaded = cd.Brain.load(brain.save(tmp_path / "bounded.npz"))
    assert loaded.recall.boundary is True
    for name in ("keys", "values", "nexts", "held"):
        np.testing.assert_array_equal(getattr(loaded.recall, name), getattr(brain.recall, name))
    for b in (brain, loaded):
        b.reset()
    np.testing.assert_array_equal(brain.act(event(3), greedy=True), loaded.act(event(3), greedy=True))
    founder = composed()
    founder.act(event(1), greedy=True)
    founder.retune(recall_boundary=True)  # a stream that began before the retune has none
    assert founder.recall.boundary is True
    founder.act(event(1), greedy=True)
    assert founder.recall.slots == 2
    founder.reset()  # the next stream begins at one
    founder.act(event(1), greedy=True)
    assert founder.recall.slots == 2 and not founder.recall.keys[0, 0].any()


def test_a_stream_reset_alone_then_an_end_of_every_stream_leaves_no_dead_slot():
    brain = composed(recall_boundary=True)
    brain.act(event(1, 2), greedy=True)
    recall = brain.recall
    recall.reset(2, rows=np.array([0]))
    recall.end(2)
    brain.act(event(3, 4), greedy=True)
    dead = (recall.held & ~recall.keys.any(axis=2) & ~recall.values.any(axis=2)
            & ~recall.nexts.any(axis=2))
    assert not dead.any()  # every boundary learned its stream's first event
    for row, symbol in ((0, 3), (1, 4)):
        boundary = recall.held[row] & ~recall.keys[row].any(axis=1)
        np.testing.assert_array_equal(recall.nexts[row][boundary][-1], event(symbol)[0])


def test_a_memory_of_one_slot_holds_no_boundary():
    brain = composed(recall_boundary=True, recall_capacity=1)
    for s in (2, 5):
        brain.act(event(s), greedy=True)
    np.testing.assert_array_equal(brain.recall.values[0], event(5))
    brain.reset(keep=("slots",))
    assert brain.recall.slots == 1 and brain.recall.ended.all()
