"""The recall memory writes a new slot in the room behind its bank, and the bank stays a view in
the order every read sums it: a write costs one slot, and every read and write is the one a full
copy of the bank per moment gives, to the bit."""

from __future__ import annotations

import copy

import numpy as np

import cadence as cd


def brain(capacity: int = 8) -> cd.Brain:
    return cd.Brain.compose(6, 3, modules=(16,), seed=0, recall=True, recall_founder=True, recall_capacity=capacity,
                            episodic=False)


def live(subject: cd.Brain, moments: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    actions = []
    for _ in range(moments):
        x = np.eye(6)[rng.integers(0, 6, 2)]
        actions.append(subject.act(x, greedy=True))
    return np.array(actions)


def test_a_write_keeps_the_bank_in_its_room_and_the_room_moves_when_it_runs_out() -> None:
    subject = brain(capacity=8)
    live(subject, 1)
    recall = subject.recall
    room = recall.keys.base
    assert room is not None and room.shape[1] >= recall.keys.shape[1]
    rooms = {id(room)}
    for _ in range(40):
        live(subject, 1)
        assert recall.keys.base is not None
        assert recall.keys.shape[1] <= 8
        assert recall.keys.base.shape[1] <= 16  # at most twice the capacity
        rooms.add(id(recall.keys.base))
    assert 1 < len(rooms) < 40  # most writes stayed in their room


def test_every_read_and_write_is_the_one_a_full_copy_per_moment_gives() -> None:
    subject, copied = brain(capacity=8), brain(capacity=8)
    rng = np.random.default_rng(3)
    for moment in range(50):
        x = np.eye(6)[rng.integers(0, 6, 2)]
        recall = copied.recall
        # the same memory with its slots in plain arrays of their own at every moment
        recall.keys, recall.values, recall.nexts, recall.held = (
            np.array(a) for a in (recall.keys, recall.values, recall.nexts, recall.held)
        )
        np.testing.assert_array_equal(subject.act(x, greedy=True), copied.act(x, greedy=True))
        for name in ("keys", "values", "nexts", "held"):
            a, b = getattr(subject.recall, name), getattr(copied.recall, name)
            assert a.shape == b.shape and a.tobytes() == b.tobytes(), (moment, name)
        np.testing.assert_array_equal(subject.basal_ganglia.state.v, copied.basal_ganglia.state.v)


def test_a_snapshot_is_not_written_into() -> None:
    subject = brain(capacity=8)
    live(subject, 4)
    recall = subject.recall
    held = recall.snapshot()
    before = {name: np.array(value) for name, value in held.items()}
    live(subject, 3, seed=1)
    for name, value in before.items():
        np.testing.assert_array_equal(held[name], value)
    recall.restore(held)
    assert recall.nexts is held["nexts"]
    live(subject, 2, seed=2)  # a write after a restore leaves the restored arrays as they were
    for name, value in before.items():
        np.testing.assert_array_equal(held[name], value)


def test_a_copy_writes_its_own_slots() -> None:
    subject = brain(capacity=8)
    live(subject, 3)
    twin = copy.deepcopy(subject)
    assert twin.recall.keys.base is None or twin.recall.keys.base is not subject.recall.keys.base
    before = np.array(subject.recall.keys)
    live(twin, 3, seed=1)
    np.testing.assert_array_equal(subject.recall.keys, before)
    assert not np.shares_memory(twin.recall.keys, subject.recall.keys)


def test_a_long_bank_costs_a_write_of_one_slot() -> None:
    """At a large capacity a write moves no slot on most moments."""
    subject = brain(capacity=256)
    live(subject, 40)
    recall = subject.recall
    moved = 0
    for _ in range(40):
        before = recall.keys.base
        live(subject, 1)
        moved += recall.keys.base is not before
    assert moved <= 2
