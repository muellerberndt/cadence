"""Pattern separation: correlated keys become sparse codes that interfere little or not at all."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def _shared(rng: np.random.Generator, dim: int) -> np.ndarray:
    shared = rng.standard_normal(dim)
    return shared / np.linalg.norm(shared)


def _correlated_keys(
    rng: np.random.Generator, count: int, dim: int, cosine: float, shared: np.ndarray
) -> np.ndarray:
    keys = []
    for _ in range(count):
        own = rng.standard_normal(dim)
        own -= own @ shared * shared
        own /= np.linalg.norm(own)
        keys.append(np.sqrt(cosine) * shared + np.sqrt(1.0 - cosine) * own)
    return np.asarray(keys)


def _mean_cosine(x: np.ndarray) -> float:
    unit = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    gram = unit @ unit.T
    off = gram[~np.eye(len(x), dtype=bool)]
    return float(np.mean(np.abs(off)))


def _revision_accuracy(
    memory: cd.FastSynapses, keys: np.ndarray, rng: np.random.Generator, passes: int
) -> float:
    count, classes = len(keys), 8
    values = np.eye(classes)[rng.integers(0, classes, size=count)]
    for _ in range(passes):
        for t in rng.permutation(count):
            memory.observe(keys[t : t + 1], values[t : t + 1])
    reads = np.concatenate([memory.recall(keys[t : t + 1]) for t in range(count)])
    return float(np.mean(reads.argmax(axis=1) == values.argmax(axis=1)))


def test_separator_validates_and_codes_are_sparse() -> None:
    with pytest.raises(ValueError):
        cd.PatternSeparator(inputs=4, expansion=8, winners=9)
    with pytest.raises(ValueError):
        cd.PatternSeparator(inputs=4, expansion=8, winners=2, center=1.0)
    sep = cd.PatternSeparator(inputs=16, expansion=128, winners=4, seed=3)
    rng = np.random.default_rng(0)
    code = sep.code(rng.standard_normal((5, 16)))
    assert code.shape == (5, 128)
    assert np.all((code > 0).sum(axis=1) <= 4)
    x = rng.standard_normal((5, 16))
    assert np.array_equal(sep.code(x), sep.code(x))
    assert np.array_equal(
        sep.code(x), cd.PatternSeparator(inputs=16, expansion=128, winners=4, seed=3).code(x)
    )


def test_correlated_keys_become_nearly_orthogonal_codes() -> None:
    rng = np.random.default_rng(1)
    shared = _shared(rng, 32)
    keys = _correlated_keys(rng, 16, 32, 0.9, shared)
    assert _mean_cosine(keys) > 0.85
    sep = cd.PatternSeparator(inputs=32, expansion=1024, winners=8, seed=1, center=0.99)
    sep.habituate(_correlated_keys(rng, 256, 32, 0.9, shared))
    codes = sep.code(keys)
    assert _mean_cosine(codes) < 0.05
    with pytest.raises(ValueError):
        sep.habituate(np.zeros((0, 32)))


def test_disjoint_keys_are_stored_exactly() -> None:
    # the numerical twin of the Lean theorem read_writeUpTo_of_disjoint
    rng = np.random.default_rng(4)
    count, dim = 20, 64
    keys = np.zeros((count, dim))
    for t in range(count):
        keys[t, 3 * t : 3 * t + 3] = rng.uniform(0.5, 1.5, 3)
    memory = cd.FastSynapses(np.arange(dim), np.arange(dim, dim + 8), rule="delta")
    assert _revision_accuracy(memory, keys, rng, passes=1) == 1.0


def test_separation_recovers_correlated_revision_streams() -> None:
    rng = np.random.default_rng(5)
    shared = _shared(rng, 32)
    keys = _correlated_keys(rng, 64, 32, 0.9, shared)
    plain = cd.FastSynapses(np.arange(32), np.arange(32, 40), rule="delta")
    sep = cd.PatternSeparator(inputs=32, expansion=1024, winners=8, seed=5, center=0.99)
    sep.habituate(_correlated_keys(rng, 256, 32, 0.9, shared))
    separated = cd.FastSynapses(np.arange(32), np.arange(32, 40), rule="delta", separator=sep)
    before = _revision_accuracy(plain, keys, np.random.default_rng(7), passes=4)
    after = _revision_accuracy(separated, keys, np.random.default_rng(7), passes=4)
    assert separated.strength.shape[1:] == (1024, 8)
    assert separated.to_dict()["separator"]["expansion"] == 1024
    assert after >= 0.95
    assert after >= before + 0.5
