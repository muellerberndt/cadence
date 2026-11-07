"""``dense_limit`` at composition: a brain whose blocks exceed the graph's founder 2048 squared
entries settles on the block transport and the compiled kernels instead of the sparse one, to the
same equations. A brain with the recall memory takes the blocks' width unless it names a limit, and
a brain without the memory composes as before."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def wide(**options) -> cd.Brain:
    return cd.Brain.compose(3000, 8, modules=(512,), seed=0, recall=True, recall_founder=True, episodic=False, **options)


def test_a_founder_composes_with_the_founders_limit() -> None:
    assert cd.Brain.compose(6, 3).brain.dense_limit == 2048


def test_the_limit_takes_a_wide_brain_onto_the_blocks() -> None:
    sparse, dense = wide(dense_limit=2048), wide(dense_limit=4096)
    assert not sparse.brain._blocked
    assert dense.brain._blocked and dense.brain.dense_limit == 4096
    np.testing.assert_array_equal(sparse.brain.efficacy, dense.brain.efficacy)
    x = np.random.default_rng(0).random((2, 3000)) * 0.25
    drive = dense.stimulus(x)
    a = sparse.brain.settle_batch(drive, steps=8)
    b = dense.brain.settle_batch(drive, steps=8)
    np.testing.assert_allclose(a.activation, b.activation, atol=1e-10)
    assert sparse.parameters() == dense.parameters()


def test_a_wide_brain_with_the_memory_keeps_its_blocks_unless_it_names_a_limit() -> None:
    subject = wide()
    graph = subject.brain
    assert graph._blocked
    assert graph.layout.size <= graph.dense_limit ** 2 < (graph.dense_limit + 1) ** 2
    assert (graph.dense_limit - 1) ** 2 < graph.layout.size
    assert not wide(dense_limit=2048).brain._blocked
    # a brain under the graph's limit keeps it, and so does one without the memory at any width
    assert cd.Brain.compose(6, 3, recall=True, recall_founder=True).brain.dense_limit == 2048
    founder = cd.Brain.compose(5000, 8, modules=(1024,), seed=0, episodic=False).brain
    assert founder.layout.size > 2048 ** 2
    assert founder.dense_limit == 2048 and not founder._blocked


def test_it_is_a_construction_choice_a_checkpoint_keeps(tmp_path) -> None:
    dense = wide(dense_limit=4096)
    x = np.random.default_rng(1).random((1, 3000)) * 0.25
    dense.act(x, greedy=True)
    path = dense.save(tmp_path / "wide.npz")
    loaded = cd.Brain.load(path)
    assert loaded.brain.dense_limit == 4096 and loaded.brain._blocked
    np.testing.assert_array_equal(dense.act(x, greedy=True), loaded.act(x, greedy=True))
    with pytest.raises(ValueError, match="dense_limit"):
        dense.retune(dense_limit=2048)


def test_grow_keeps_the_limit() -> None:
    grown = cd.Brain.compose(6, 3, modules=(8,), seed=0, dense_limit=512).grow(recall=True, recall_founder=True)
    assert grown.brain.dense_limit == 512
