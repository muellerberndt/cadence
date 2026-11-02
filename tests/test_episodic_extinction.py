"""Extinction of a paid episodic record when failures pay nothing (cadence#210).

With ``memory_extinction`` above zero, a zero outcome writes a learned zero for the chosen
action where the records' vote for it and its direct payment in this place both reach the
threshold. Only a known outcome with a reward counts as a direct payment: replayed and
demonstrated credit is never extinguished. The rule is Sean McClane's (cadence#210).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from checkpoints import metadata

import cadence as cd


def episodic(*args, **options) -> cd.Brain:
    return cd.Brain.compose(*args, **{"memory_choice": True, "memory_context": 0.0, **options})


def one_hot(index: int, width: int) -> np.ndarray:
    x = np.zeros((1, width))
    x[0, index] = 1.0
    return x


def place(brain: cd.Brain, x: np.ndarray, action: int) -> float:
    return float(brain.hippocampus.recall(x)[0, action])


def outcome(brain: cd.Brain, x: np.ndarray, action: int, reward: float, after: np.ndarray,
            done: bool = False) -> None:
    """The outcome of ``action`` taken on ``x``: a decision there, the action executed."""
    brain.act(x)
    brain._moment = (x.copy(), np.array([action]))
    brain.learn(np.array([reward]), np.array([done]), after)


def test_extinction_is_an_opt_in_threshold_of_the_episodic_choice():
    founder = episodic(3, 2, seed=0)
    assert founder.memory_extinction == 0.0 and founder.memories["episodic"].paid is None
    assert "extinction" not in founder.describe()["memories"]["episodic"]
    brain = episodic(3, 2, seed=0, memory_extinction=0.5)
    assert brain.memory_extinction == 0.5 and brain.memories["episodic"].paid is not None
    assert brain.describe()["memories"]["episodic"]["extinction"] == 0.5
    with pytest.raises(ValueError, match="need memory_choice"):
        cd.Brain.compose(3, 2, memory_extinction=0.5)
    with pytest.raises(ValueError, match="memory_extinction"):
        episodic(3, 2, memory_extinction=1.5)


@pytest.mark.parametrize("extinction", [0.0, 0.5])
def test_a_zero_outcome_extinguishes_a_record_its_action_earned_here(extinction):
    brain = episodic(3, 3, seed=4, memory_extinction=extinction)
    x, after = one_hot(0, 3), one_hot(1, 3)
    for _ in range(2):
        outcome(brain, x, 2, 1.0, after)
    paid = place(brain, x, 2)
    assert paid > 0.5
    outcome(brain, x, 2, 0.0, after)  # the action stopped paying, without a punishment
    if extinction:
        settled = place(brain, x, 2)
        assert settled < 0.5 <= paid  # a learned zero: the record moved toward nothing
        outcome(brain, x, 2, 0.0, after)  # below the threshold: nothing more is written
        assert place(brain, x, 2) == pytest.approx(settled)
    else:
        assert place(brain, x, 2) == pytest.approx(paid)


def test_a_demonstrated_record_is_not_extinguished():
    brain = episodic(3, 3, seed=4, memory_extinction=0.5)
    x, after = one_hot(0, 3), one_hot(1, 3)
    brain.step(x, teacher=[2])
    shown = place(brain, x, 2)
    assert shown >= 0.5
    for _ in range(3):
        outcome(brain, x, 2, 0.0, after)
    assert place(brain, x, 2) == pytest.approx(shown)


def test_replayed_credit_is_not_extinguished():
    """A reward at the end of an attempt replays credit to the decision that led to it; that
    decision was never paid itself, so its zero outcomes leave the credit. One action, so
    every decision is the one replayed."""
    brain = episodic(3, 1, seed=4, memory_extinction=0.5, memory_replay=8)
    start, middle, end = one_hot(0, 3), one_hot(1, 3), one_hot(2, 3)
    for _ in range(6):
        brain.act(start)
        brain.learn(np.zeros(1), np.zeros(1, bool), middle)
        brain.act(middle)
        brain.learn(np.ones(1), np.ones(1, bool), end)  # paid at the end of the attempt
    credited = place(brain, start, 0)
    assert credited >= 0.5  # replayed from the reward at the end
    for _ in range(3):
        brain.act(start)
        brain.learn(np.zeros(1), np.zeros(1, bool), middle)  # nothing, in the middle
    assert place(brain, start, 0) == pytest.approx(credited)


def test_the_reversal_is_acquired_when_failures_pay_nothing():
    """The issue's bandit: arm 0 pays for 120 trials, then arm 1; the unpaid arm pays 0."""
    observation = np.array([[1.0]])

    def late(seed: int, extinction: float) -> float:
        brain = cd.Brain.compose(1, 2, modules=(16,), seed=seed, arousal=True,
                                 memory_choice=True, memory_extinction=extinction)
        action, hits = int(brain.live(observation)[0]), []
        for t in range(240):
            target = int(t >= 120)
            hits.append(action == target)
            action = int(brain.live(observation, reward=[float(action == target)])[0])
        return float(np.mean(hits[-40:]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = [late(seed, 0.0) for seed in range(8)]
        on = [late(seed, 0.5) for seed in range(8)]
    acquired = sum(b >= 0.8 for b in on), sum(b >= 0.8 for b in off)
    assert acquired[0] >= 6 and acquired[1] <= 5, (on, off)
    assert np.mean(on) - np.mean(off) >= 0.2, (on, off)


def test_an_extinguishing_brain_saves_and_continues_exactly(tmp_path):
    brain = episodic(3, 3, seed=7, memory_extinction=0.5, arousal=True)
    x, after = one_hot(0, 3), one_hot(1, 3)
    for _ in range(2):
        outcome(brain, x, 2, 1.0, after)
    brain.step(after)  # an issued action owns the next outcome
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert copy.memory_extinction == 0.5
    for name in ("strength", "consolidated"):
        np.testing.assert_array_equal(
            getattr(copy.memories["episodic"].paid, name),
            getattr(brain.memories["episodic"].paid, name),
        )
    for t in range(4):
        reward = [0.0 if t % 2 else 1.0]
        np.testing.assert_array_equal(
            brain.step(one_hot(t % 3, 3), reward=reward), copy.step(one_hot(t % 3, 3), reward=reward)
        )
    np.testing.assert_allclose(copy.hippocampus.consolidated, brain.hippocampus.consolidated)
    founder = episodic(3, 3, seed=7)
    founder.step(x)
    meta = metadata(np.load(founder.save(tmp_path / "founder.npz")))
    assert "extinction" not in meta["episodic"]


def test_grow_keeps_the_extinction_and_the_direct_payments():
    brain = episodic(3, 3, seed=7, memory_extinction=0.5)
    x, after = one_hot(0, 3), one_hot(1, 3)
    outcome(brain, x, 2, 1.0, after)
    grown = brain.grow(efference_amplitude=1.0)
    assert grown.memory_extinction == 0.5
    np.testing.assert_array_equal(
        grown.memories["episodic"].paid.consolidated,
        brain.memories["episodic"].paid.consolidated,
    )


@pytest.mark.parametrize("amplitude", [None, 3.0])
def test_extinction_reads_the_direct_payment_in_reward_units_after_a_retune(amplitude):
    brain = cd.Brain.compose(4, 2, seed=0, memory_choice=True, memory_extinction=0.5)
    if amplitude is not None:
        brain.retune(memory_amplitude=amplitude)
    episodic = brain.memories["episodic"]
    x = brain.observation_code(np.array([[1.0, 0.0, 0.0, 0.0]]))
    action, learned, done = np.array([0]), np.array([True]), np.array([False])
    for _ in range(6):  # the action pays here, directly
        episodic.outcome(x, action, np.array([1.0]), done, learned, gamma=0.9)
    assert float(episodic.held(x, x, action, episodic.paid)[0]) == pytest.approx(1.0)
    before = float(episodic.held(x, None, action)[0])
    episodic.outcome(x, action, np.array([0.0]), done, learned, gamma=0.9)  # it stops paying
    assert float(episodic.held(x, None, action)[0]) < before - 0.3
