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
        4, 2, seed=seed, recall=True, recall_founder=True, arousal=True, recall_position_reward=position_reward,
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
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=1, recall=True, recall_founder=True, arousal=True, **options)
    action = brain.live(X[[0]])
    for t in range(1, 30):
        pending = brain.basal_ganglia._pending_memory
        if pending is not None and pending.any():
            return brain.recall.full_eligibility(pending)
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
    assert trace is not None
    assert brain.recall.full_eligibility(trace).shape[-1] == brain.recall.reward_columns


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
        cd.Brain.compose(6, 4, seed=0, recall=True, recall_founder=True, recall_position_reward=1.0,
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
    taught = cd.Brain.compose(4, 2, seed=0, recall=True, recall_founder=True, arousal=True, recall_anchor=True)
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


def test_the_reward_step_does_not_follow_the_lessons_rate():
    # the same life with the table taught by lessons at the founder's rate and at a tenth of
    # it: no lesson is given, so reward alone moves the maps, and it moves them alike
    founder_rate = life(1.0, anchor=True, recall_position_rate=0.1)
    slow_table = life(1.0, anchor=True, recall_position_rate=0.01)
    assert cd.Recall.REWARD_RATE == cd.Recall.options()["position_rate"] == 0.1
    for a, b in zip(founder_rate.recall.position_maps, slow_table.recall.position_maps,
                    strict=True):
        for m, n in zip(a, b, strict=True):
            np.testing.assert_array_equal(m, n)
    assert moved(slow_table.recall.position_maps)


def test_the_actor_keeps_the_columns_that_can_step_over_the_event_rows():
    """The default memory's eligibility at speech's width would be 141 MB per row in full: the
    actor keeps the position's and the anchored columns of the event's rows, and the checkpoint's
    layout is made from them."""
    brain = life(1.0, anchor=True)
    recall, agent = brain.recall, brain.basal_ganglia
    assert recall.reward_blocks == [
        (recall.address_width, recall.position_buckets),
        (recall.address_width + recall.position_buckets, recall.position_buckets + 1),
    ]
    width = 2 * recall.position_buckets + 1
    assert agent.trace_memory.shape == (1, recall.map_width, width)
    full = recall.full_eligibility(agent.trace_memory)
    assert full.shape == (1, recall.map_width + len(recall.association) + 1, recall.reward_columns)
    np.testing.assert_array_equal(recall.kept_eligibility(full), agent.trace_memory)
    assert not full[:, recall.map_width:].any() and not full[:, :, :recall.address_width].any()


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(("width", "association", "columns", "size"), [
    (4, 8, 4, 3), (35, 64, 35, 34), (64, 64, 64, 36), (256, 128, 256, 36), (1024, 256, 1024, 36),
])
def test_the_report_of_a_reward_step_reads_as_the_norm_of_the_whole_eligibility(
    seed, width, association, columns, size
):
    """Without the zeros of the long runs each term keeps its lane in the dot, so the norm reads as
    the whole vector's, to the last bits a BLAS of another machine moves."""
    from cadence.stream import _laid_out_norm

    rng = np.random.default_rng(seed)
    position = rng.normal(size=(width, size)) * 10.0 ** rng.uniform(-6, 0)
    anchored = rng.normal(size=(width, size + 1)) * 10.0 ** rng.uniform(-6, 0)
    rest = association + 1
    whole = np.concatenate([
        np.zeros((width + rest) * columns), position.ravel(), np.zeros(rest * size),
        anchored.ravel(), np.zeros(rest * (size + 1)),
    ])
    pieces = [(width + rest) * columns, position, rest * size, anchored, rest * (size + 1)]
    assert _laid_out_norm(pieces) == pytest.approx(np.linalg.norm(whole), rel=1e-14)


def test_a_retuned_address_reward_keeps_the_eligibility_as_it_lay():
    brain = life(1.0, anchor=True)
    recall, agent = brain.recall, brain.basal_ganglia
    if agent._pending is not None:  # the setting waits for the outcome of a sampled act
        brain.learn(np.zeros(1), np.zeros(1, dtype=bool), X[[0]])
    before = recall.full_eligibility(agent.trace_memory)
    brain.retune(recall_address_reward=0.5)  # its columns join, at zero
    assert agent.trace_memory.shape[-1] == recall.address_width + 2 * recall.position_buckets + 1
    np.testing.assert_array_equal(recall.full_eligibility(agent.trace_memory), before)
    brain.retune(recall_address_reward=0.0)
    np.testing.assert_array_equal(
        recall.full_eligibility(agent.trace_memory)[:, :, recall.address_width:],
        before[:, :, recall.address_width:],
    )
