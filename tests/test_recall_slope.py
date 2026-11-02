"""A learned slope over age beside the position table (cadence#232): every head's position
term also leans by age at every length, one number per moment read from the moment."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def slope(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, seed=5, recall=True, **options)


def test_the_slope_is_an_option_beside_the_position_term():
    plain = slope()
    assert plain.recall.slope == 0.0 and plain.recall.slope_maps == []
    brain = slope(recall_slope=0.1)
    assert [m.shape for m in brain.recall.slope_maps[0]] == [(6,), (64,), (1,)]
    assert brain.describe()["memories"]["recall"]["slope"] == 0.1
    with pytest.raises(ValueError, match="needs position"):
        slope(recall_position=0, recall_slope=0.1)
    with pytest.raises(ValueError):
        slope(recall_slope=-0.1)


def test_the_lean_grows_with_age_at_every_length():
    brain = slope(recall_slope=0.1)
    recall = brain.recall
    rng = np.random.default_rng(0)
    for _ in range(12):
        brain.act(rng.random((2, 6)))
    s_x, s_c, s = recall.slope_maps[0]
    recall.slope_maps = [(s_x, s_c, np.array([2.0]))]  # lean old
    event, cortex = rng.random((2, 6)), np.zeros((2, 64))
    p_x, _, p = recall.position_maps[0]
    table = (event @ p_x + p)[:, recall.buckets()]
    lean = recall.position_term(0, event, cortex) - table
    np.testing.assert_allclose(lean, np.repeat(2.0 * recall.ages()[None, :] / 8.0, 2, axis=0))


def test_a_newborn_slope_reads_as_the_memory_without_it():
    rng = np.random.default_rng(1)
    plain, brain = slope(), slope(recall_slope=0.1)
    for _ in range(4):
        x = rng.random((2, 6))
        np.testing.assert_array_equal(plain.act(x, greedy=True), brain.act(x, greedy=True))
        np.testing.assert_array_equal(
            np.asarray(plain.basal_ganglia.state.activation),
            np.asarray(brain.basal_ganglia.state.activation),
        )


@pytest.mark.parametrize("position_rate", [0.1, 0.0])
def test_lessons_step_the_slope_even_with_the_table_frozen(position_rate):
    brain = slope(recall_slope=0.1, recall_position_rate=position_rate)
    table = [a.copy() for a in brain.recall.position_maps[0]]
    rng = np.random.default_rng(2)
    for _ in range(12):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))
    assert np.abs(brain.recall.slope_maps[0][2]).sum() > 0
    moved = any(not np.array_equal(a, b) for a, b in zip(table, brain.recall.position_maps[0], strict=True))
    assert moved == (position_rate > 0)


def test_a_checkpoint_keeps_the_slope_and_continues(tmp_path):
    brain = slope(recall_slope=0.1)
    rng = np.random.default_rng(3)
    for _ in range(12):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    for kept, loaded in zip(brain.recall.slope_maps[0], copy.recall.slope_maps[0], strict=True):
        np.testing.assert_array_equal(kept, loaded)
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
    plain = slope()
    path = plain.save(tmp_path / "plain.npz")
    assert cd.Brain.load(path).recall.slope_maps == []


def test_parameters_count_the_maps_of_the_options():
    founder = slope().parameters()
    extended = cd.Brain.compose(6, 4, seed=5, recall=True, recall_slope=0.1, recall_anchor=True,
                                recall_relative=True).parameters()
    recall = cd.Brain.compose(6, 4, seed=5, recall=True, recall_slope=0.1, recall_anchor=True,
                              recall_relative=True).recall
    added = sum(a.size for maps in (*recall.slope_maps, *recall.anchor_maps,
                                    *recall.relative_maps) for a in maps)
    assert extended == founder + added
