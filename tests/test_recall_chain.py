"""Chained recall heads (cadence#234): each head after the first adds to its address what the
head before it read, through a map that starts at zero and the address lesson steps."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def chained(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, seed=3, recall=True, recall_heads=2, **options)


def lessons(brain: cd.Brain, count: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(count):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))


def test_chain_is_an_option_of_several_heads():
    plain = chained()
    assert not plain.recall.chain and plain.recall.chain_maps == []
    assert "chain" not in plain.describe()["memories"]["recall"]
    brain = chained(recall_chain=True)
    assert brain.recall.chain and brain.describe()["memories"]["recall"]["chain"] is True
    assert [m.shape for m in brain.recall.chain_maps] == [(2 * 6, 6)]  # recall and next reads
    with pytest.raises(ValueError, match="heads of 2 or more"):
        cd.Brain.compose(6, 4, recall=True, recall_chain=True)
    with pytest.raises(ValueError, match="address_reward"):
        chained(recall_chain=True, recall_address_reward=1.0)


def test_a_newborn_chain_reads_as_the_heads_without_it():
    rng = np.random.default_rng(1)
    plain, brain = chained(), chained(recall_chain=True)
    for _ in range(4):
        x = rng.random((2, 6))
        np.testing.assert_array_equal(plain.act(x, greedy=True), brain.act(x, greedy=True))
        np.testing.assert_array_equal(
            np.asarray(plain.basal_ganglia.state.activation),
            np.asarray(brain.basal_ganglia.state.activation),
        )


def test_lessons_step_the_chain_map():
    brain = chained(recall_chain=True)
    lessons(brain)
    assert np.abs(brain.recall.chain_maps[0]).sum() > 0


def test_a_chained_brain_saves_and_continues_exactly(tmp_path):
    brain = chained(recall_chain=True)
    lessons(brain)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    np.testing.assert_array_equal(copy.recall.chain_maps[0], brain.recall.chain_maps[0])
    rng = np.random.default_rng(2)
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
    np.testing.assert_array_equal(copy.recall.chain_maps[0], brain.recall.chain_maps[0])
