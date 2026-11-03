"""Retuning the recall read's reward settings keeps the actor's eligibility consistent with
them: a pending outcome's contrast keeps its rates, a trace of another column layout is
dropped, and reward-trained position maps keep their trace through unrelated retunes."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import cadence as cd


def lived(**options) -> cd.Brain:
    brain = cd.Brain.compose(6, 3, modules=(16,), seed=0, recall=True, **options)
    rng = np.random.default_rng(0)
    for i in range(4):
        brain.step(rng.random((2, 6)), **({} if i == 0 else {"reward": rng.normal(size=2)}))
    return brain


@pytest.mark.parametrize("field", ["position_reward", "position_rate"])
def test_position_reward_settings_wait_for_a_pending_outcome(field):
    brain = lived(recall_position_reward=4.0, recall_address_reward=1.0)
    assert brain.basal_ganglia._pending_memory is not None
    with pytest.raises(ValueError, match="pending"):
        brain.retune(**{f"recall_{field}": 2.0 * getattr(brain.recall, field)})


def test_a_trace_of_another_layout_is_dropped_and_the_brain_saves_and_loads(tmp_path):
    brain = lived(recall_position_reward=4.0, recall_address_reward=1.0)
    brain.learn(np.array([1.0, -1.0]), np.zeros(2, bool), np.random.default_rng(1).random((2, 6)))
    assert brain.basal_ganglia.trace_memory is not None
    brain.retune(recall_position_reward=0.0)
    trace = brain.basal_ganglia.trace_memory
    assert trace is None or trace.shape[-1] == brain.recall.reward_columns
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    x = np.random.default_rng(2).random((2, 6))
    np.testing.assert_array_equal(brain.step(x), copy.step(x))


def test_an_unrelated_retune_keeps_the_position_trace():
    brain = lived(recall_position_reward=4.0)
    twin = deepcopy(brain)
    twin.retune(actor_gamma=twin.basal_ganglia.config.gamma)  # the same value
    x = np.random.default_rng(1).random((2, 6))
    for b in (brain, twin):
        b.learn(np.array([1.0, -1.0]), np.zeros(2, bool), x)
    for kept, retuned in zip(brain.recall.position_maps[0], twin.recall.position_maps[0],
                             strict=True):
        np.testing.assert_array_equal(kept, retuned)


def test_position_reward_needs_the_position_term():
    with pytest.raises(ValueError, match="needs position"):
        cd.Brain.compose(6, 3, seed=0, recall=True, recall_position=0,
                         recall_position_reward=1.0)
    brain = cd.Brain.compose(6, 3, seed=0, recall=True, recall_position=0)
    with pytest.raises(ValueError, match="needs position"):
        brain.retune(recall_position_reward=1.0)


def test_the_position_rate_retunes_freely_where_reward_does_not_train_the_position_maps():
    brain = lived(recall_address_reward=1.0)  # position_reward at its founder 0
    assert brain.basal_ganglia._pending_memory is not None
    brain.retune(recall_position_rate=0.05)
    assert brain.recall.position_rate == 0.05


def test_retune_refuses_a_position_reward_at_a_frozen_table():
    brain = cd.Brain.compose(6, 4, seed=0, recall=True, recall_position_reward=1.0)
    with pytest.raises(ValueError, match="position_rate above 0"):
        brain.retune(recall_position_rate=0.0)
    frozen = cd.Brain.compose(6, 4, seed=0, recall=True, recall_position_rate=0.0)
    with pytest.raises(ValueError, match="position_rate above 0"):
        frozen.retune(recall_position_reward=1.0)
    frozen.retune(recall_position_rate=0.05, recall_position_reward=1.0)  # both at once
    assert frozen.recall.position_reward == 1.0 and frozen.recall.position_rate == 0.05
