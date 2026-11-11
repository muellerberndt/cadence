"""The readback types the public API hands back: each one obtained from its real producer,
checked to be the exported class, frozen where it is declared frozen, and consistent with the
inputs that produced it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import cadence as cd


def _frozen(result: object, name: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, name, None)


# ---------------------------------------------------------------------------- continuous brains
def test_ep_structure() -> None:
    graph = cd.layered(2, 5, 1, seed=3, density=1)
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model())
    structure = cd.ep_structure(brain, fixed_inputs=graph.populations["input"])
    assert isinstance(structure, cd.EPStructure)
    _frozen(structure, "tolerance")
    assert structure.compatible and structure.to_dict()["compatible"] is True


def test_equilibrium_from_brain_equilibrate() -> None:
    brain = cd.NeuralGraph(cd.layered(2, 3, 2, seed=4), cd.learning_neuron_model())
    drive = np.zeros((2, brain.connectome.n))
    drive[:, :2] = [[1.0, 0.5], [0.2, 0.8]]
    result = brain.equilibrate(drive, budget=256, tolerance=1e-6)
    assert isinstance(result, cd.Equilibrium)
    _frozen(result, "residual")
    assert result.residual.shape == (2,) and np.isfinite(result.residual).all()
    assert 0 < result.steps <= 256


def test_settlement_record() -> None:
    graph = cd.layered(2, 3, 2, seed=4)
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend="cpu", precision="float64")
    drive = np.zeros((2, graph.n))
    drive[:, :2] = [[1.0, 0.5], [0.2, 0.8]]
    records: list[cd.SettlementRecord] = []
    with cd.record_settlements(records.append, label="probe"):
        state = brain.settle_batch(drive, steps=5)
    (record,) = records
    assert isinstance(record, cd.SettlementRecord)
    _frozen(record, "label")
    assert record.label == "probe" and record.steps == state.steps == 5
    assert record.potential.shape == record.activation.shape == (6, 2, graph.n)
    np.testing.assert_array_equal(record.drive, drive)
