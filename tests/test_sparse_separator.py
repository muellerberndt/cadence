"""A sparse pattern separator: ``PatternSeparator(..., fan_in=k)`` and the episodic memory's
``memory_fan_in``. Each code neuron reads ``k`` inputs, so the separator's storage and work
follow the expansion and the fan-in, whatever the number of inputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd


def one_hot(index: int, width: int) -> np.ndarray:
    x = np.zeros((1, width))
    x[0, index] = 1.0
    return x


def winners(separator: cd.PatternSeparator, keys: np.ndarray) -> list[set[int]]:
    return [set(np.flatnonzero(code)) for code in separator.code(keys)]


def overlap(separator: cd.PatternSeparator, a: np.ndarray, b: np.ndarray) -> float:
    """The mean number of winners two keys share, over pairs of rows."""
    return float(np.mean([len(u & v) for u, v in zip(winners(separator, a), winners(separator, b),
                                                       strict=True)]))


def keys_at(similarity: float, inputs: int, rows: int, seed: int, shared: bool) -> tuple:
    """Pairs of random keys whose varying parts have the given cosine; ``shared`` adds a
    positive mean every key has in common, as activity patterns do."""
    rng = np.random.default_rng(seed)
    a, noise = rng.standard_normal((rows, inputs)), rng.standard_normal((rows, inputs))
    b = similarity * a + np.sqrt(1.0 - similarity**2) * noise
    if shared:
        mean = np.abs(np.random.default_rng(seed + 1).standard_normal(inputs))
        a, b = mean + 0.5 * a, mean + 0.5 * b
    return a, b


def test_the_dense_founder_is_unchanged():
    separator = cd.PatternSeparator(16, 128, 4, seed=3)
    named = cd.PatternSeparator(16, 128, 4, seed=3, fan_in=0)
    expected = np.random.default_rng(3).standard_normal((16, 128)) / 4.0
    np.testing.assert_array_equal(separator.projection, expected)
    np.testing.assert_array_equal(named.projection, expected)
    assert separator.index is None and separator.weight is None
    assert "fan_in" not in separator.to_dict() and list(separator.arrays()) == [
        "projection",
        "mean",
    ]
    keys = np.random.default_rng(4).standard_normal((5, 16))
    np.testing.assert_array_equal(separator.code(keys), named.code(keys))


def test_a_fan_in_separator_holds_the_same_bytes_whatever_the_number_of_inputs():
    narrow = cd.PatternSeparator(64, 1024, 24, seed=1, fan_in=16)
    wide = cd.PatternSeparator(12824, 1024, 24, seed=1, fan_in=16)
    for separator in (narrow, wide):
        assert separator.projection is None
        assert separator.index.shape == separator.weight.shape == (1024, 16)
        assert separator.index.dtype == np.int64
        assert (np.diff(separator.index, axis=1) > 0).all()  # distinct inputs, in order
        assert separator.index.max() < separator.inputs
        assert separator.to_dict()["fan_in"] == 16
        assert list(separator.arrays()) == ["index", "weight", "mean"]
    assert narrow.index.nbytes + narrow.weight.nbytes == wide.index.nbytes + wide.weight.nbytes
    assert narrow.index.nbytes + narrow.weight.nbytes == 1024 * 16 * 16
    dense = cd.PatternSeparator(12824, 1024, 24, seed=1)
    assert dense.projection.nbytes == 12824 * 1024 * 8
    code = wide.code(np.random.default_rng(2).standard_normal((3, 12824)))
    assert code.shape == (3, 1024) and ((code > 0).sum(axis=1) <= 24).all()


def test_a_fan_in_code_is_the_dense_code_of_its_sparse_matrix():
    sparse = cd.PatternSeparator(40, 256, 8, seed=5, fan_in=6)
    dense = cd.PatternSeparator(40, 256, 8, seed=5)
    matrix = np.zeros((40, 256))
    columns = np.repeat(np.arange(256), 6)
    matrix[sparse.index.reshape(-1), columns] = sparse.weight.reshape(-1)
    dense.projection = matrix
    keys = np.random.default_rng(6).standard_normal((7, 40))
    np.testing.assert_allclose(sparse.code(keys), dense.code(keys), rtol=1e-12, atol=1e-12)
    centered = cd.PatternSeparator(40, 256, 8, seed=5, fan_in=6, center=0.5)
    centered.habituate(keys)
    shifted = keys - centered.mean
    np.testing.assert_allclose(centered.code(keys), sparse.code(shifted), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shared", [False, True])
def test_a_fan_in_code_separates_keys_as_the_dense_code_does(shared):
    dense = cd.PatternSeparator(64, 1024, 24, seed=3)
    sparse = cd.PatternSeparator(64, 1024, 24, seed=3, fan_in=16)
    readings = []
    for similarity in (0.0, 0.5, 0.9, 1.0):
        a, b = keys_at(similarity, 64, 200, seed=11, shared=shared)
        readings.append((overlap(dense, a, b), overlap(sparse, a, b)))
    for separator in (0, 1):
        series = [reading[separator] for reading in readings]
        assert series == sorted(series) and series[-1] == 24.0  # closer keys share more winners
    for dense_overlap, sparse_overlap in readings:
        assert abs(sparse_overlap - dense_overlap) <= 3.0
    if not shared:
        assert readings[0][0] < 2.0 and readings[0][1] < 2.0  # unrelated keys: near chance


def test_the_episodic_memory_takes_a_fan_in_and_continues_exactly(tmp_path):
    brain = cd.Brain.compose(20, 2, seed=31, arousal=True, memory_choice=True, memory_fan_in=16)
    for memory in (brain.hippocampus, brain.context_memory):
        assert memory.separator.fan_in == 16 and memory.separator.projection is None
    assert brain.describe()["memory"]["separator"]["fan_in"] == 16
    for t in range(5):
        brain.step(one_hot(t % 3, 20), reward=None if t == 0 else [float(t == 3)],
                   teacher=[1] if t == 2 else None)
    path = brain.save(tmp_path / "brain.npz")
    copy = cd.Brain.load(path)
    for name in ("hippocampus", "context_memory"):
        for field in ("index", "weight", "mean"):
            np.testing.assert_array_equal(
                getattr(getattr(copy, name).separator, field),
                getattr(getattr(brain, name).separator, field),
            )
    meta = json.loads(str(np.load(path)["generic"]))
    assert meta["hippocampus"]["separator"]["fan_in"] == 16
    assert "episodic/separator/projection" not in np.load(path).files
    for t in range(6):
        reward, done = [float(t == 5)], [t == 5]
        np.testing.assert_array_equal(
            brain.step(one_hot(t % 2, 20), reward=reward, done=done),
            copy.step(one_hot(t % 2, 20), reward=reward, done=done),
        )
    for name in ("hippocampus", "context_memory"):
        np.testing.assert_array_equal(
            getattr(copy, name).consolidated, getattr(brain, name).consolidated
        )


def test_the_fan_in_founder_keeps_the_composed_brain_byte_identical(tmp_path):
    founder = cd.Brain.compose(3, 2, seed=32, memory_choice=True)
    named = cd.Brain.compose(3, 2, seed=32, memory_choice=True, memory_fan_in=0)
    for brain in (founder, named):
        for t in range(4):
            brain.step(one_hot(t % 3, 3), reward=None if t == 0 else [float(t == 2)])
    left = np.load(founder.save(tmp_path / "founder.npz"))
    right = np.load(named.save(tmp_path / "named.npz"))
    assert sorted(left.files) == sorted(right.files)
    for name in left.files:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)
    assert "fan_in" not in json.loads(str(left["generic"]))["hippocampus"]["separator"]


def test_a_sensory_key_with_a_fan_in_holds_the_same_bytes_at_any_width():
    def held(brain: cd.Brain) -> list[int]:
        return [sum(value.nbytes for name, value in m.separator.arrays().items() if name != "mean")
                for m in (brain.hippocampus, brain.context_memory)]

    # code=0 keys the wide brain's memories on the observation itself (above 1,024 values the
    # brain's code would key them, and the stores would take no separator)
    wide = cd.Brain.compose(4096, 3, seed=33, memory_choice=True, memory_fan_in=16, code=0)
    narrow = cd.Brain.compose(20, 3, seed=33, memory_choice=True, memory_fan_in=16)
    assert held(wide) == held(narrow) == [1024 * 16 * 16, 1024 * 16 * 16]


def test_the_fan_in_is_validated():
    for value in (-1, True, 2.5):
        with pytest.raises(ValueError, match="fan_in must be a nonnegative integer"):
            cd.PatternSeparator(4, 8, 2, fan_in=value)
    with pytest.raises(ValueError, match="fan_in must not exceed the inputs"):
        cd.PatternSeparator(4, 8, 2, fan_in=5)
    with pytest.raises(ValueError, match="must not exceed the 3 coordinates of the place key"):
        cd.Brain.compose(3, 2, memory_choice=True, memory_fan_in=16)
    with pytest.raises(ValueError, match="memory_fan_in needs memory_expansion"):
        cd.Brain.compose(3, 2, memory_choice=True, memory_expansion=0, memory_fan_in=2)
    with pytest.raises(ValueError, match="memory_fan_in needs memory_expansion"):
        cd.Brain.compose(3, 2, memory_fan_in=2)
    with pytest.raises(ValueError, match="construction setting"):
        cd.Brain.compose(20, 2, memory_choice=True, memory_fan_in=8).retune(memory_fan_in=4)


@pytest.mark.parametrize("tamper", ["outside", "repeated", "no_weight", "no_fan_in"])
def test_a_corrupt_sparse_separator_is_refused(tmp_path, tamper):
    brain = cd.Brain.compose(20, 2, seed=34, memory_choice=True, memory_fan_in=8)
    brain.step(one_hot(0, 20))
    path = brain.save(tmp_path / "brain.npz")
    with np.load(path) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    meta = json.loads(str(data["generic"]))
    if tamper == "outside":
        data["episodic/separator/index"][0, -1] = 64
    elif tamper == "repeated":
        data["episodic/separator/index"][0, 1] = data["episodic/separator/index"][0, 0]
    elif tamper == "no_weight":
        del data["episodic/separator/weight"]
    else:
        del meta["hippocampus"]["separator"]["fan_in"]  # the arrays of a sparse separator alone
    data["generic"] = np.array(json.dumps(meta))
    corrupt = tmp_path / "corrupt.npz"
    np.savez(corrupt, **data)
    with pytest.raises(ValueError):
        cd.Brain.load(corrupt)
