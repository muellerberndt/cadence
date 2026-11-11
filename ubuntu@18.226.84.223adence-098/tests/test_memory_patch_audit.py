"""Selective stream operations preserve the other memories and reject invalid selections."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def populated() -> cd.Brain:
    brain = cd.Brain.compose(
        3, 2, modules=(4,), seed=31, memory_choice=True, recall=True, recall_founder=True, recall_window=2
    )
    brain.act(np.eye(3), greedy=True)
    brain.act(np.roll(np.eye(3), 1, axis=1), greedy=True)
    return brain


def arrays(memory) -> dict[str, np.ndarray]:  # noqa: ANN001
    return {name: value.copy() for name, value in memory.arrays().items()}


def assert_arrays(memory, expected: dict[str, np.ndarray]) -> None:  # noqa: ANN001
    actual = memory.arrays()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value, err_msg=name)


@pytest.mark.parametrize("operation", ["reset", "end"])
def test_recall_boolean_mask_changes_only_selected_streams(operation: str) -> None:
    brain = populated()
    memory = brain.recall
    assert memory is not None
    before = arrays(memory)
    getattr(memory, operation)(3, rows=np.array([True, False, True]))

    for name in ("keys", "values", "nexts", "held"):
        np.testing.assert_array_equal(memory.arrays()[name][1], before[name][1])
    np.testing.assert_array_equal(memory.previous[1], before["previous"][1])
    np.testing.assert_array_equal(memory.past[1], before["past"][1])
    assert not memory.previous[[0, 2]].any()
    assert not memory.past[[0, 2]].any()
    if operation == "reset":
        assert not memory.held[[0, 2]].any()
        assert memory.held[1].all()
    else:
        np.testing.assert_array_equal(memory.held, before["held"])
        np.testing.assert_array_equal(memory.ended, [True, False, True])


def test_episodic_boolean_reset_preserves_other_streams_paths() -> None:
    brain = populated()
    memory = brain.memories["episodic"]
    context = memory.context.copy()
    path = list(memory.paths[1])
    memory.reset(3, rows=np.array([True, False, True]))
    assert not memory.context[[0, 2]].any()
    np.testing.assert_array_equal(memory.context[1], context[1])
    assert memory.paths[0] == [] and memory.paths[2] == []
    assert len(memory.paths[1]) == len(path) > 0
    for actual, expected in zip(memory.paths[1], path, strict=True):
        for value, earlier in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value, earlier)


@pytest.mark.parametrize("name", ["recall", "episodic"])
def test_boolean_keep_matches_the_same_explicit_stream_indices(name: str) -> None:
    masked, indexed = populated().memories[name], populated().memories[name]
    masked.keep(np.array([True, False, True]))
    indexed.keep(np.array([0, 2]))
    assert_arrays(masked, arrays(indexed))
    if name == "episodic":
        assert len(masked.paths) == len(masked.context) == len(masked.decision) == 2


@pytest.mark.parametrize("name", ["recall", "episodic"])
@pytest.mark.parametrize("operation", ["reset", "keep"])
@pytest.mark.parametrize(
    "selection",
    [
        np.array([True, False]), np.array([0.5]), np.array([[0]]), np.array([0, 3]),
        np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
        np.array([np.iinfo(np.uint64).max - 1], dtype=np.uint64),
    ],
)
def test_invalid_stream_selection_is_rejected_without_partial_changes(
    name: str, operation: str, selection: np.ndarray
) -> None:
    memory = populated().memories[name]
    before = arrays(memory)
    with pytest.raises(ValueError, match="rows"):
        if operation == "reset":
            memory.reset(3, rows=selection)
        else:
            memory.keep(selection)
    assert_arrays(memory, before)
