"""``cadence.Message``: an episode held as phased unions of a fixed sparse code (cadence#286)."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def _moments(rng: np.random.Generator, count: int, inputs: int = 35, active: int = 3) -> np.ndarray:
    x = np.zeros((count, inputs))
    for row in x:
        row[rng.choice(inputs, size=active, replace=False)] = 1.0
    return x


def test_a_code_is_ones_at_its_winners_and_silence_codes_to_nothing():
    message = cd.Message(35, units=70, winners=5, seed=1)
    x = _moments(np.random.default_rng(0), 6)
    code = message.code(x)
    assert code.shape == (6, 70) and set(np.unique(code)) == {0.0, 1.0}
    assert (code.sum(axis=1) == 5).all()
    assert not message.code(np.zeros((2, 35))).any()
    assert np.array_equal(code, cd.Message(35, units=70, winners=5, seed=1).code(x))
    assert not np.array_equal(code, cd.Message(35, units=70, winners=5, seed=2).code(x))


def test_a_code_binds_what_a_moment_holds_together():
    # two moments that share two of three active inputs share part of their code and no more
    message = cd.Message(35, seed=3)
    a, b, c = np.zeros((3, 35))
    a[[0, 4, 5]] = 1.0
    b[[0, 4, 6]] = 1.0
    c[[1, 7, 8]] = 1.0
    code = message.code(np.stack([a, b, c]))
    shared, apart = float(code[0] @ code[1]), float(code[0] @ code[2])
    assert 0 < shared < 5 and apart < shared


def test_each_moment_joins_the_union_of_its_phase():
    message = cd.Message(35, phases=3, units=40, winners=4, seed=0)
    x = _moments(np.random.default_rng(1), 5)
    for moment in x:
        message.hear(moment[None])
    assert message.state.shape == (1, 120) and message.width == 120
    held = message.state.reshape(3, 40)
    codes = message.code(x)
    assert np.array_equal(held[0], np.maximum(codes[0], codes[3]))  # moments 0 and 3
    assert np.array_equal(held[1], np.maximum(codes[1], codes[4]))
    assert np.array_equal(held[2], codes[2])
    assert message.told.tolist() == [5]
    assert np.array_equal(message.key()[0], codes.max(axis=0))


def test_the_key_is_the_same_in_any_order_and_the_state_keeps_the_order():
    rng = np.random.default_rng(2)
    x = _moments(rng, 4)
    forward, backward = cd.Message(35, seed=5), cd.Message(35, seed=5)
    for moment in x:
        forward.hear(moment[None])
    for moment in x[::-1]:
        backward.hear(moment[None])
    assert np.array_equal(forward.key(), backward.key())
    assert not np.array_equal(forward.state, backward.state)
    other = cd.Message(35, seed=5)
    for moment in _moments(rng, 4):
        other.hear(moment[None])
    assert float(forward.key()[0] @ other.key()[0]) < float(forward.key()[0] @ forward.key()[0])


def test_streams_hold_their_own_messages_and_begin_again_alone():
    message = cd.Message(35, phases=2, units=30, winners=3, seed=0)
    x = _moments(np.random.default_rng(3), 6).reshape(3, 2, 35)
    for moment in x:
        message.hear(moment)
    kept = message.state[1].copy()
    message.reset(2, rows=np.array([0]))
    assert not message.state[0].any() and message.told.tolist() == [0, 3]
    assert np.array_equal(message.state[1], kept)
    message.hear(x[0])  # stream 0 is at its first phase again, stream 1 at its fourth moment
    assert np.array_equal(message.state[0, :30], message.code(x[0])[0])
    message.keep(np.array([1]))
    assert message.state.shape == (1, 60) and message.told.tolist() == [4]
    message.hear(np.zeros((4, 35)))  # another batch begins the streams afresh
    assert message.state.shape == (4, 60) and not message.state.any()


def test_a_branch_is_private_and_the_arrays_are_what_a_checkpoint_keeps():
    message = cd.Message(35, seed=0)
    x = _moments(np.random.default_rng(4), 3)
    message.hear(x[:1])
    branch = message.branch()
    branch.hear(x[1:2])
    assert message.told.tolist() == [1] and branch.told.tolist() == [2]
    assert set(message.arrays()) == {"trace", "told"}
    again = cd.Message(**message.to_dict())
    again.trace, again.told = (a.copy() for a in message.arrays().values())
    again.hear(x[1:2])
    assert np.array_equal(again.state, branch.state)


def test_the_fan_in_is_raised_until_every_channel_is_read():
    message = cd.Message(200, units=16, winners=2, fan_in=4, seed=0)
    assert message.to_dict()["fan_in"] == 50  # two readers of each of 400 channels over 16 units
    assert (message.code(np.eye(200)).sum(axis=1) > 0).all()  # a single input never codes to zero


@pytest.mark.parametrize("settings", [
    {"phases": 0}, {"units": 0}, {"winners": 0}, {"winners": 80}, {"fan_in": 0}, {"seed": -1},
    {"phases": True},
])
def test_refusals(settings):
    with pytest.raises(ValueError):
        cd.Message(35, **settings)


def test_observations_are_checked():
    message = cd.Message(35)
    with pytest.raises(ValueError):
        message.hear(np.zeros((1, 34)))
    with pytest.raises(ValueError):
        message.hear(np.full((1, 35), np.nan))
