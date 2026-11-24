"""The record a reward is awaited for: a moment sensed between the act and its reward, or
another stream's moment, does not take its slot. A reward pays the moment the brain acted at
and no other, in a store of one record, in a full store whose records are all kept, and
after a save and a load."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import Brain
from cadence.bound import BoundRecords
from cadence.longterm import LongTermMemory


def speaker(capacity: int) -> Brain:
    return Brain.compose(senses={"eyes": 16, "ear": 16}, actions=2, modules=(8,), code=64,
                         voice="ear", bound=capacity, arousal=True, seed=0, backend="cpu")


def seen(eye: int, said: int | None = None, rows: int = 1) -> dict[str, np.ndarray]:
    out = {"eyes": np.repeat(np.eye(16)[[eye]], rows, axis=0)}
    if said is not None:
        out["voice"] = np.repeat(np.eye(16)[[said]], rows, axis=0)
    return out


def kept_full(brain: Brain, eyes: tuple[int, ...]) -> None:
    """Fill the store with records, each told until it is kept for good."""
    for eye in eyes:
        for _ in range(20):
            brain.act(seen(eye, said=eye + 1), greedy=True)
    store = brain.bound.store
    assert store.count == store.capacity == len(eyes) and store.kept.all()


@pytest.mark.parametrize("reward", [1.0, -1.0])
def test_a_sensed_moment_leaves_the_record_a_reward_is_awaited_for(reward):
    brain = speaker(1)
    brain.step(seen(0, said=1))
    slot = int(brain.bound.written[0])
    acted = brain.bound.store.values[slot].copy()
    brain.wait(seen(6, said=7))  # a store of one record: the sensed moment is not written
    assert int(brain.bound.written[0]) == slot
    np.testing.assert_array_equal(brain.bound.store.values[slot], acted)
    brain.live({}, reward=[reward])
    np.testing.assert_array_equal(brain.bound.store.values[slot], acted)
    if reward > 0:
        assert brain.bound.paid[slot] == BoundRecords.PAY and not brain.bound.refused[slot]
    else:
        assert brain.bound.paid[slot] == 0.0 and brain.bound.refused[slot]


def test_in_a_full_store_of_kept_records_the_sensed_moments_take_other_slots():
    brain = speaker(4)
    kept_full(brain, (0, 4, 8, 12))
    brain.step(seen(2, said=3))  # replaces the shallowest kept record
    slot = int(brain.bound.written[0])
    acted = brain.bound.store.values[slot].copy()
    for eye in (5, 9, 13):  # each would be the shallowest record's replacement
        brain.wait(seen(eye, said=eye + 1))
        np.testing.assert_array_equal(brain.bound.store.values[slot], acted)
    brain.live({}, reward=[1.0])
    assert brain.bound.paid[slot] == BoundRecords.PAY
    assert brain.bound.paid.sum() == BoundRecords.PAY  # and no other record was paid


def test_without_a_moment_in_between_the_store_replaces_as_it_did():
    """A life that only acts awaits no reward when it writes: the store's own rule decides."""
    brain, twin = speaker(2), speaker(2)
    for held in (brain, twin):
        kept_full(held, (0, 8))
    for eye in (3, 5, 11):
        brain.step(seen(eye, said=eye + 1), reward=[1.0] if brain.pending_feedback else None)
        twin.act(seen(eye, said=eye + 1), greedy=True)
    np.testing.assert_array_equal(brain.bound.store.keys, twin.bound.store.keys)
    np.testing.assert_array_equal(brain.bound.store.values, twin.bound.store.values)


def test_a_loaded_brain_still_awaits_the_reward_of_the_moment_it_acted_at(tmp_path):
    brain = speaker(1)
    brain.step(seen(0, said=1))
    slot = int(brain.bound.written[0])
    acted = brain.bound.store.values[slot].copy()
    again = Brain.load(brain.save(tmp_path / "awaiting.npz"))
    assert int(again.bound.written[0]) == slot
    again.wait(seen(6, said=7))
    again.live({}, reward=[1.0])
    np.testing.assert_array_equal(again.bound.store.values[slot], acted)
    assert again.bound.paid[slot] == BoundRecords.PAY


def test_two_streams_of_one_moment_take_slots_of_their_own():
    brain = speaker(2)
    kept_full(brain, (0, 8))
    brain.reset()  # two streams from here on; the records stay
    both = {"eyes": np.eye(16)[[3, 11]], "voice": np.eye(16)[[4, 12]]}
    brain.step(both)
    slots = brain.bound.written.copy()
    assert slots[0] != slots[1] and (slots >= 0).all()
    brain.bound.pay([1.0, 0.0])  # each stream's reward, as the next moment's outcome pays it
    assert brain.bound.paid[slots[0]] == BoundRecords.PAY and not brain.bound.refused[slots[0]]
    assert brain.bound.paid[slots[1]] == 0.0 and brain.bound.refused[slots[1]]


def test_a_stream_without_a_slot_left_writes_nothing_and_is_paid_nothing():
    brain = speaker(1)
    both = {"eyes": np.eye(16)[[3, 11]], "voice": np.eye(16)[[4, 12]]}
    brain.step(both)
    assert brain.bound.written.tolist() == [0, -1]  # one record: the first stream's
    brain.bound.pay([0.0, 1.0])
    assert brain.bound.paid.sum() == 0.0  # the second stream's reward has no record to pay
    assert brain.bound.refused[0]  # the first stream's own reward, of zero, refuses its record


# -- the store's write with slots to spare


def full(**options) -> LongTermMemory:
    memory = LongTermMemory(8, 4, 3, **options)
    for k in range(3):
        memory.write(np.eye(8)[k], np.eye(4)[k % 4])
    return memory


@pytest.mark.parametrize("options", [{}, {"fade": False}, {"keep": 2.0}])
def test_a_write_replaces_no_spared_record(options):
    plain, sparing = full(**options), full(**options)
    taken, _ = plain.write(np.eye(8)[5], np.eye(4)[1])
    slot, revised = sparing.write(np.eye(8)[5], np.eye(4)[1], spare=[taken])
    assert not revised and slot not in (taken, -1)
    np.testing.assert_array_equal(sparing.keys[taken], np.eye(8)[taken])
    # a spare that names no held slot leaves the write as it is
    other = full(**options)
    assert other.write(np.eye(8)[5], np.eye(4)[1], spare=[-1]) == (taken, False)


@pytest.mark.parametrize("options", [{}, {"fade": False}, {"keep": 2.0}])
def test_a_write_with_every_record_spared_writes_nothing(options):
    memory = full(**options)
    before = {name: value.copy() for name, value in memory.arrays().items()}
    writes = memory.writes
    assert memory.write(np.eye(8)[5], np.eye(4)[1], spare=[0, 1, 2]) == (-1, False)
    assert memory.writes == writes and memory.count == 3
    for name, value in memory.arrays().items():
        np.testing.assert_array_equal(value, before[name], err_msg=name)


def test_a_spared_record_told_again_is_still_revised():
    memory = full(revise=0.9)
    slot, revised = memory.write(np.eye(8)[1], np.eye(4)[3], spare=[0, 1, 2])
    assert (slot, revised) == (1, True)
    np.testing.assert_array_equal(memory.values[1], np.eye(4)[3])
