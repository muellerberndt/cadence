"""Index arrays that reach ``np.repeat`` and ``reduceat`` use the platform's index type.

A 32-bit NumPy (the Python a browser runs) has int32 as its index type, and its ``np.repeat`` and
``reduceat`` refuse int64 counts and offsets under the safe casting rule. On a 64-bit platform the
index type is int64. These tests set ``np.intp`` to int32 while a brain is built, as a 32-bit
platform has it, and check that the arrays follow the platform's index type.
"""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


@pytest.fixture
def index_type_int32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(np, "intp", np.int32)


@pytest.mark.parametrize("slots", [(2, 3), (3, 3), (1, 2, 4)])
def test_motor_slots_take_the_index_type(index_type_int32: None, slots: tuple[int, ...]) -> None:
    brain = cd.Brain.compose(8, sum(slots), slots=slots, seed=1)
    assert brain.learner.slot_sizes.dtype == np.dtype(np.int32)
    assert brain.learner.slot_offsets.dtype == np.dtype(np.int32)
    x = np.random.default_rng(0).random((3, 8))
    assert np.asarray(brain.act(x)).shape == (3, len(slots))
    taught = cd.Brain.compose(8, sum(slots), slots=slots, seed=1)
    taught.lesson(x, targets=np.hstack([np.full((3, k), 1.0 / k) for k in slots]))


def test_grown_slots_take_the_index_type(index_type_int32: None) -> None:
    grown = cd.Brain.compose(8, 4, seed=1).grow(slots=(4, 2))
    assert grown.learner.slot_sizes.dtype == np.dtype(np.int32)
    assert grown.learner.slot_offsets.dtype == np.dtype(np.int32)


def test_summed_input_segments_take_the_index_type(index_type_int32: None) -> None:
    connectome = cd.Brain.compose(8, 3, seed=1).connectome
    connectome.__dict__.pop("_segments", None)
    brain = cd.NeuralGraph(connectome, cd.NeuronModel()) if hasattr(cd, "NeuronModel") else None
    if brain is None:
        pytest.skip("no NeuralGraph constructor to rebuild the segments")
    assert brain._starts.dtype == np.dtype(np.int32)
