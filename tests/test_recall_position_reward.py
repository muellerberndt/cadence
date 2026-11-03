"""Reward eligibility for the recall read's position maps (cadence#236): at each sampled act
the position lesson's step joins the eligibility the actor traces, and dopamine steps the maps
by ``position_reward`` (founder 0, which leaves them to lessons alone). The reward step moves
the rows of the event heard, in the address map as well; the cortex and bias rows, which every
moment shares, learn from lessons alone."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd

X = np.eye(4)


def life(position_reward: float, *, anchor: bool = False, moments: int = 60, seed: int = 0,
         **options):
    brain = cd.Brain.compose(
        4, 2, seed=seed, recall=True, arousal=True, recall_position_reward=position_reward,
        recall_anchor=anchor, **options,
    )
    action = brain.live(X[[0]])
    for t in range(moments):
        action = brain.live(X[[t % 4]], reward=[float(action[0] == t % 2)])
    return brain


def moved(maps) -> bool:
    return any(np.abs(m).sum() > 0 for head in maps for m in head)


def first_contrast(**options) -> np.ndarray:
    """The read's first eligibility that is not zero in a life of rewards."""
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=1, recall=True, arousal=True, **options)
    action = brain.live(X[[0]])
    for t in range(1, 30):
        pending = brain.basal_ganglia._pending_memory
        if pending is not None and pending.any():
            return pending
        action = brain.live(X[[t % 4]], reward=[float(action[0] == t % 2)])
    raise AssertionError("no sampled act kept an eligibility")


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


def test_position_reward_needs_a_position_rate_above_zero():
    """With the table frozen a lesson's step is zero, so reward could train nothing: refused."""
    import pytest

    with pytest.raises(ValueError, match="position_rate above 0"):
        cd.Brain.compose(6, 4, seed=0, recall=True, recall_position_reward=1.0,
                         recall_position_rate=0.0)


def test_reward_steps_the_event_rows_alone():
    """Under reward the position, anchored and address maps move on the rows of the event
    heard and nowhere else; a lesson moves every row."""
    brain = life(1.0, anchor=True, recall_address_reward=1.0)
    recall = brain.recall
    w_x, w_c, b = recall.head_map(0)
    assert w_x.any() and not (w_c.any() or b.any())
    for kind in (recall.position_maps, recall.anchor_maps):
        (p_x, p_c, p), = kind
        assert p_x.any() and not (p_c.any() or p.any())
    taught = cd.Brain.compose(4, 2, seed=0, recall=True, arousal=True, recall_anchor=True)
    for t in range(6):
        taught.act(X[[t % 4]], greedy=True)
    taught.lesson(X[[1]], np.array([1]))
    (p_x, p_c, p), = taught.recall.position_maps
    assert p_c.any() and p.any()


@pytest.mark.parametrize(
    "options",
    [{"recall_position_reward": 1.0, "recall_anchor": True}, {"recall_address_reward": 1.0}],
    ids=["position", "address"],
)
def test_the_eligibility_holds_the_event_rows_alone(options):
    """The read's eligibility of a sampled act: the event's rows (the first four of
    ``[x, c, 1]``) carry the lesson's step, the cortex and bias rows are zero."""
    contrast = first_contrast(**options)
    assert contrast.shape[1] == 4 + 8 + 1
    assert contrast[:, :4].any() and not contrast[:, 4:].any()
