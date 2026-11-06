"""Decisions kept out of the episodic temporal context (cadence#202).

``memory_context_hold="unchanged"`` keys a decision taken on the observation of the stream's
previous decision as if that decision had not been taken, the replay path's rule applied to
the key; ``"punished"`` takes a decision whose outcome was below zero, in the middle of an
attempt, back out of the context. Will Glynn measured both on his Stone Age chains
(cadence#202: EXP-57 for the observation, EXP-66 for the punishment).
"""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.memories import command


def episodic(*args, **options) -> cd.Brain:
    return cd.Brain.compose(*args, **{"memory_choice": True, **options})


def one_hot(index: int, width: int) -> np.ndarray:
    x = np.zeros((1, width))
    x[0, index] = 1.0
    return x


def key(brain: cd.Brain, x: np.ndarray) -> np.ndarray:
    return brain.memories["episodic"].key(brain.observation_code(x))


def test_the_hold_is_an_opt_in_setting_of_the_temporal_context():
    founder = episodic(3, 2, seed=0)
    assert founder.memories["episodic"].hold is None
    assert "hold" not in founder.describe()["memories"]["episodic"]
    for hold in ("unchanged", "punished"):
        brain = episodic(3, 2, seed=0, memory_context_hold=hold)
        assert brain.memories["episodic"].hold == hold
        assert brain.describe()["memories"]["episodic"]["hold"] == hold
    with pytest.raises(ValueError, match="memory_context_hold"):
        episodic(3, 2, memory_context_hold="never")
    with pytest.raises(ValueError, match="needs memory_choice and memory_context"):
        episodic(3, 2, memory_context=0.0, memory_context_hold="unchanged")
    with pytest.raises(ValueError, match="needs memory_choice and memory_context"):
        cd.Brain.compose(3, 2, memory_context_hold="unchanged")


@pytest.mark.parametrize("hold", [None, "unchanged"])
def test_a_decision_that_left_the_observation_unchanged_stays_out_of_the_key(hold):
    brain = episodic(3, 3, seed=1, memory_context_hold=hold)
    records = brain.memories["episodic"]
    a, b = one_hot(0, 3), one_hot(1, 3)
    code_a, code_b = brain.observation_code(a), brain.observation_code(b)
    first = records.key(code_a)
    brain.act(a)
    again = records.key(code_a)  # the same observation: the first decision left it unchanged
    if hold == "unchanged":
        np.testing.assert_array_equal(again, first)
    else:
        assert not np.array_equal(again, first)
    second = brain.act(a)
    if hold == "unchanged":
        # B is keyed after the last decision on A alone, as if the first had not been taken
        context = first.copy()
        context[:, records.width:] += (1.0 - records.share) * command(
            second, records.motor, records.offsets
        )
        np.testing.assert_allclose(records.key(code_b), records.keyed(code_b, context))


def test_a_punished_decision_leaves_the_temporal_context():
    brain = episodic(3, 3, seed=2, memory_context_hold="punished")
    a, b = one_hot(0, 3), one_hot(1, 3)
    brain.act(a)
    records = brain.memories["episodic"]
    before = records.before.copy()
    moved = records.context.copy()
    assert not np.array_equal(moved, before)
    brain.learn(np.array([-1.0]), np.zeros(1, bool), b)  # punished in the middle of the attempt
    np.testing.assert_array_equal(records.context, before)
    brain.act(b)
    kept = records.context.copy()
    brain.learn(np.array([1.0]), np.zeros(1, bool), a)  # paid: the decision stays
    np.testing.assert_array_equal(records.context, kept)
    plain = episodic(3, 3, seed=2)
    plain.act(a)
    moved = plain.memories["episodic"].context.copy()
    plain.learn(np.array([-1.0]), np.zeros(1, bool), b)
    np.testing.assert_array_equal(plain.memories["episodic"].context, moved)


@pytest.mark.parametrize("hold", ["unchanged", "punished"])
def test_a_holding_brain_saves_and_continues_exactly(tmp_path, hold):
    brain = episodic(3, 3, seed=3, memory_context_hold=hold)
    for t in range(4):
        brain.step(one_hot(t % 2, 3), reward=None if t == 0 else [-1.0 if t % 2 else 0.0])
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert copy.memories["episodic"].hold == hold
    for name in ("context", "before", "seen"):
        np.testing.assert_array_equal(
            getattr(copy.memories["episodic"], name), getattr(brain.memories["episodic"], name)
        )
    for t in range(5):
        reward = [-1.0 if t % 2 else 0.0]
        np.testing.assert_array_equal(
            brain.step(one_hot(t % 3, 3), reward=reward), copy.step(one_hot(t % 3, 3), reward=reward)
        )


def test_grow_keeps_the_hold():
    brain = episodic(3, 3, seed=4, memory_context_hold="unchanged")
    brain.act(one_hot(0, 3))
    grown = brain.grow(efference_amplitude=1.0)
    assert grown.memories["episodic"].hold == "unchanged"


def test_an_imagined_branch_keys_its_decisions_as_the_stream_keys_its_own():
    """Two decisions on one observation: the second is keyed in the context before the first
    (the hold), on the stream and on a branch alike."""
    from copy import deepcopy

    x, y = one_hot(0, 3), one_hot(1, 3)
    for hold in (None, "unchanged"):
        brain = episodic(3, 3, seed=1, memory_context=0.5, memory_context_hold=hold)
        brain.act(y, greedy=True)
        records, code = brain.memories["episodic"], brain.observation_code(x)
        stream = deepcopy(brain)
        stream.act(x, greedy=True)
        branch = records.branch()
        first = branch.key(code)
        branch.issue(first, np.asarray(brain.act(x, greedy=True)), code)
        np.testing.assert_array_equal(branch.key(code), stream.memories["episodic"].key(code))


def test_imagine_issues_its_decisions_through_the_shared_bookkeeping(monkeypatch):
    from cadence.memories import EpisodicRecords

    issued = []
    original = EpisodicRecords.issue

    def spy(self, key, action, code):
        issued.append(np.array(code))
        return original(self, key, action, code)

    monkeypatch.setattr(EpisodicRecords, "issue", spy)
    brain = episodic(3, 3, seed=1, memory_context=0.5, memory_context_hold="unchanged",
                     efference_amplitude=1.0)  # with an efference copy the branch issues acts
    x = one_hot(0, 3)
    brain.act(one_hot(1, 3), greedy=True)
    issued.clear()
    brain.imagine([x, x])
    assert len(issued) == 2
    for code in issued:
        np.testing.assert_array_equal(code, brain.observation_code(x))


def test_a_waited_moment_ends_the_run_the_hold_keys_out():
    """live on X, then Y sensed without a decision: the next decision on X is keyed in the
    context after Y, as without the hold; the hold removes the previous decision, not what
    was sensed after it."""
    x, y = one_hot(0, 3), one_hot(1, 3)
    keys = {}
    for hold in (None, "unchanged"):
        brain = episodic(3, 2, seed=0, arousal=True, memory_context=0.5,
                         memory_context_hold=hold)
        brain.live(x)
        brain.wait(y)
        keys[hold] = key(brain, x)
    np.testing.assert_array_equal(keys["unchanged"], keys[None])
    held = episodic(3, 2, seed=0, arousal=True, memory_context=0.5,
                    memory_context_hold="unchanged")
    held.live(x)
    assert not np.array_equal(key(held, x), keys[None])  # without the wait the hold applies
