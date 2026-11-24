"""A record of a bound moment is one event: a situation and the sound said in it. A moment told
again revises the record of the same event and no other, so a reward pays the sound the brain
said, in either order of two sounds whose keys lie near. The streams of one moment that act at
one event share its record, a reward any of them is paid pays it, and a record awaiting its
reward is not forgotten before the reward comes."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def speaker(capacity: int = 8, code: int = 24, seed: int = 4) -> cd.Brain:
    return cd.Brain.compose(senses={"eyes": 16, "ear": 16}, actions=2, modules=(8,), code=code,
                            voice="ear", bound=capacity, arousal=True, seed=seed, backend="cpu")


def seen(eye: int, said: int | None = None, rows: int = 1) -> dict[str, np.ndarray]:
    out = {"eyes": np.eye(16)[[eye] * rows]}
    if said is not None:
        out["voice"] = np.eye(16)[[said] * rows]
    return out


def said(brain: cd.Brain, eye: int) -> tuple[list, int]:
    answer = brain.say(seen(eye))
    return list(brain.bound.said_from), int(np.asarray(answer).argmax())


def test_the_two_sounds_lie_near_enough_to_be_recognised_as_one_moment():
    brain = speaker()
    keys = [brain._bound_keyed(brain._sensed_rows(seen(14, s), complete=False))[0] for s in (2, 3)]
    assert float(keys[0] @ keys[1]) >= brain.bound.REVISE  # what the tests below rely on


def test_a_reward_pays_the_sound_said_after_another_sound_was_refused():
    brain = speaker()
    brain.step(seen(14, 3))
    brain.step(seen(14, 3), reward=[0.0])  # 3 refused
    brain.step(seen(14, 2), reward=[0.0])  # 3 refused again; 2 said
    slot = int(brain.bound.written[0])
    first, last = brain.bound.fields[brain.bound._voice_field()]
    np.testing.assert_array_equal(brain.bound.store.values[slot, first:last], np.eye(16)[2])
    brain.live(seen(14), reward=[1.0])  # 2 paid
    assert said(brain, 14) == (["said"], 2)


def test_a_refusal_of_a_new_sound_leaves_the_paid_sound_paid():
    brain = speaker()
    brain.step(seen(14, 2))
    brain.step(seen(14, 2), reward=[1.0])  # 2 paid
    brain.step(seen(14, 3), reward=[1.0])  # 2 paid again; 3 said
    slot = int(brain.bound.written[0])
    first, last = brain.bound.fields[brain.bound._voice_field()]
    np.testing.assert_array_equal(brain.bound.store.values[slot, first:last], np.eye(16)[3])
    brain.live(seen(14), reward=[0.0])  # 3 refused
    assert brain.bound.refused[slot]
    assert said(brain, 14) == (["said"], 2)


def test_a_moment_told_again_with_its_own_sound_still_deepens_its_record():
    brain = speaker()
    brain.step(seen(14, 2))
    slot = int(brain.bound.written[0])
    deep = float(brain.bound.store.deep[slot])
    brain.step(seen(14, 2), reward=[1.0])
    assert int(brain.bound.written[0]) == slot and brain.bound.store.count == 1
    assert brain.bound.store.deep[slot] > deep


@pytest.mark.parametrize("rewards", [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
def test_streams_that_act_at_one_event_share_its_record_and_a_payment_wins(rewards):
    brain = speaker()
    brain.step(seen(14, 2, rows=2))
    first, second = (int(slot) for slot in brain.bound.written)
    assert first == second >= 0 and brain.bound.store.count == 1
    brain.bound.pay(rewards)
    assert brain.bound.paid[first] == brain.bound.PAY and not brain.bound.refused[first]
    assert brain.say(seen(14, rows=2)).shape == (2, 16)
    assert brain.bound.said_from == ["said", "said"]


def test_streams_refused_together_refuse_the_record():
    brain = speaker()
    brain.step(seen(14, 2, rows=2))
    slot = int(brain.bound.written[0])
    brain.bound.pay([0.0, -1.0])
    assert brain.bound.refused[slot] and brain.bound.paid[slot] == 0.0


def test_many_streams_seeing_the_same_events_fill_the_store_with_one_record_an_event():
    brain = speaker(capacity=16)
    for t in range(6):
        e = t % 3
        moment = seen(10 + e, 2 + e, rows=8)
        brain.step(moment) if t == 0 else brain.step(moment, reward=np.ones(8))
    assert brain.bound.store.count == 3
    assert np.all(brain.bound.paid[:3] > 0)


def test_a_refusal_one_stream_brought_is_lifted_when_the_sound_is_paid_later():
    brain = speaker(capacity=16, code=64, seed=0)
    both = {"eyes": np.eye(16)[[14, 14]], "voice": np.eye(16)[[2, 2]]}
    away = {"eyes": np.eye(16)[[3, 5]]}
    brain.step(both)
    brain.step(away, reward=[1.0, 0.0])  # one stream paid, one not
    alone = {"eyes": np.eye(16)[[14, 5]], "voice": np.stack([np.eye(16)[2], np.zeros(16)])}
    for _ in range(3):  # stream 0 says 2 at eye 14 again and is paid each time
        brain.step(alone, reward=[0.0, 0.0])
        brain.step(away, reward=[1.0, 0.0])
    assert said(brain, 14) == (["said"], 2)


@pytest.mark.parametrize("capacity, waits", [(4, 36), (1, 9)])
def test_a_record_awaiting_its_reward_is_not_forgotten_before_the_reward(capacity, waits):
    brain = speaker(capacity=capacity, code=64, seed=0)
    brain.step(seen(0, 1))
    slot = int(brain.bound.written[0])
    store = brain.bound.store
    for k in range(waits):  # the outcome comes late: the stream senses on
        brain.wait(seen(2 + k % 12))
    assert store.faded[slot] >= store.forget and not store.gone[slot]
    brain.live({}, reward=[2.0])
    assert store.kept[slot] and not store.gone[slot]
    assert said(brain, 0) == (["said"], 1)


def test_a_record_paid_after_it_was_forgotten_is_held_again():
    """A file of an earlier release can hold an awaited record already forgotten."""
    brain = speaker(capacity=4, code=64, seed=0)
    brain.step(seen(0, 1))
    slot = int(brain.bound.written[0])
    brain.bound.store.gone[slot] = True
    brain.live({}, reward=[2.0])
    assert not brain.bound.store.gone[slot]
    assert said(brain, 0) == (["said"], 1)
