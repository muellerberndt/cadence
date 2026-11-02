"""Reward eligibility for the recall read's position maps (cadence#236): at each sampled act
the position lesson's step joins the eligibility the actor traces, and dopamine steps the maps
by ``position_reward`` (founder 0, which leaves them to lessons alone)."""

from __future__ import annotations

import numpy as np

import cadence as cd


def life(position_reward: float, *, anchor: bool = False, moments: int = 60, seed: int = 0):
    brain = cd.Brain.compose(
        4, 2, seed=seed, recall=True, arousal=True, recall_position_reward=position_reward,
        recall_anchor=anchor,
    )
    action = brain.live(np.eye(4)[[0]])
    for t in range(moments):
        action = brain.live(np.eye(4)[[t % 4]], reward=[float(action[0] == t % 2)])
    return brain


def moved(maps) -> bool:
    return any(np.abs(m).sum() > 0 for head in maps for m in head)


def test_the_founder_leaves_the_position_maps_to_lessons():
    brain = life(0.0)
    assert brain.recall.position_reward == 0.0 and not moved(brain.recall.position_maps)
    assert brain.basal_ganglia.trace_memory is None


def test_reward_alone_steps_the_position_maps_and_the_anchored_ones():
    brain = life(1.0, anchor=True)
    assert moved(brain.recall.position_maps) and moved(brain.recall.anchor_maps)
    w_x, w_c, b = brain.recall.head_map(0)  # the address map stays with address_reward at 0
    assert not (w_x.any() or w_c.any() or b.any())
    trace = brain.basal_ganglia.trace_memory
    assert trace is not None and trace.shape[-1] == brain.recall.reward_columns


def test_a_life_with_position_reward_saves_and_continues_exactly(tmp_path):
    brain = life(1.0, moments=30)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    for t in range(6):
        x, reward = np.eye(4)[[t % 4]], [float(t % 3 == 0)]
        np.testing.assert_array_equal(brain.live(x, reward=reward), copy.live(x, reward=reward))
    for kept, saved in zip(copy.recall.position_maps[0], brain.recall.position_maps[0], strict=True):
        np.testing.assert_allclose(kept, saved, rtol=0, atol=1e-12)
