"""A reloaded recall memory keeps its per-stream state the shape of its streams, so a partial
episode end or a reset that keeps the slots continues as the living brain does."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


@pytest.mark.parametrize(
    "options",
    [{}, {"recall_anchor": True}, {"recall_relative": True}, {"recall_heads": 2,
                                                              "recall_boundary": True}],
    ids=["founder", "anchor", "relative", "heads"],
)
def test_a_reloaded_recall_brain_continues_through_partial_ends(options, tmp_path):
    brain = cd.Brain.compose(6, 3, modules=(16,), seed=0, recall=True, **options)
    rng = np.random.default_rng(0)
    brain.step(rng.random((2, 6)))
    for _ in range(3):
        brain.step(rng.random((2, 6)), reward=rng.normal(size=2))
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    for twin in (brain, copy):
        assert len(twin.recall.told) == len(twin.recall.focus) == 2
    for _ in range(4):
        x, reward = rng.random((2, 6)), rng.normal(size=2)
        done = rng.random(2) < 0.5
        np.testing.assert_array_equal(
            brain.step(x, reward=reward, done=done), copy.step(x, reward=reward, done=done)
        )
    brain.reset(keep=("slots",))
    copy.reset(keep=("slots",))
    x = rng.random((2, 6))
    np.testing.assert_array_equal(brain.act(x, greedy=True), copy.act(x, greedy=True))


@pytest.mark.parametrize(
    "options",
    [{}, {"recall_anchor": True}, {"recall_relative": True}, {"recall_slope": 0.1},
     {"recall_when": True, "recall_anchor": True}],
    ids=["founder", "anchor", "relative", "slope", "when_anchor"],
)
def test_a_batch_of_other_streams_after_a_reset_that_keeps_the_slots(options):
    brain = cd.Brain.compose(6, 3, modules=(16,), seed=0, recall=True, **options)
    rng = np.random.default_rng(0)
    for _ in range(3):
        brain.act(rng.random((2, 6)), greedy=True)
    brain.reset(keep=("slots",))
    assert np.asarray(brain.act(rng.random((3, 6)), greedy=True)).shape == (3,)
    assert np.asarray(brain.act(rng.random((3, 6)), greedy=True)).shape == (3,)
