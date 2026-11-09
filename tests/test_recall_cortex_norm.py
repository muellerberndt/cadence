"""``recall_cortex_norm``: every map of the read reads the cortex of the moment before at unit
norm per stream."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def _brain(**options):
    return cd.Brain.compose(12, 6, backend="cpu", recall=True, seed=3, **options)


def _life(brain, lessons=4):
    rng = np.random.default_rng(0)
    for _ in range(lessons):
        brain.reset()
        for _ in range(3):
            x = np.eye(12)[rng.integers(12, size=2)]
            brain.lesson(x, rng.integers(6, size=2))
            brain.act(x, greedy=True)


def test_the_cortex_the_maps_read_is_at_unit_norm():
    brain = _brain(recall_cortex_norm=True)
    brain.reset()
    brain.act(np.eye(12)[[1, 2]], greedy=True)
    assert np.allclose(np.linalg.norm(brain.recall.previous, axis=1), 1.0)
    assert brain.recall.cortex_norm and not brain.recall.resides


def test_the_option_is_off_by_default_and_the_life_without_it_is_unchanged():
    plain, named = _brain(), _brain(recall_cortex_norm=False)
    assert plain.recall.cortex_norm is False
    _life(plain)
    _life(named)
    for a, b in zip(plain.recall.head_map(0), named.recall.head_map(0), strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(plain.recall.previous, axis=1).max() != pytest.approx(1.0)


def test_a_life_with_it_differs_and_is_saved_with_it(tmp_path):
    plain, normed = _brain(), _brain(recall_cortex_norm=True)
    _life(plain)
    _life(normed)
    assert not np.array_equal(plain.recall.head_map(0)[1], normed.recall.head_map(0)[1])
    loaded = cd.Brain.load(normed.save(tmp_path / "normed"))
    assert loaded.recall.cortex_norm is True
    assert normed.describe()["genes"]["recall_cortex_norm"] is True
    assert "recall_cortex_norm" not in plain.describe()["genes"]


def test_it_is_refused_unless_boolean():
    with pytest.raises(ValueError, match="cortex_norm must be boolean"):
        _brain(recall_cortex_norm=1)
