"""Direct tests for cadence.reference: Ledger, settle_neuron_by_neuron, conformance."""

import numpy as np

import cadence as cd
from cadence.reference import Ledger, conformance, settle_neuron_by_neuron


def _stimulus(n: int) -> np.ndarray:
    stimulus = np.zeros(n)
    stimulus[:4] = [1.0, 0.5, 0.8, 0.2]  # drive the input layer so the dynamics are not trivial
    return stimulus


def _small_brain() -> cd.NeuralGraph:
    c = cd.layered(4, 6, 2, density=1.0, seed=1)
    return cd.NeuralGraph(c, cd.learning_neuron_model())


# ------------------------------------------------------------------ Ledger


def test_ledger_clean_when_counts_match() -> None:
    ledger = Ledger(declared_synapses=10, steps=3, transmissions=30)
    assert ledger.clean


def test_ledger_not_clean_with_undeclared() -> None:
    ledger = Ledger(declared_synapses=10, steps=3, transmissions=30, undeclared=1)
    assert not ledger.clean


def test_ledger_to_dict_has_expected_keys() -> None:
    ledger = Ledger(declared_synapses=5, steps=2, transmissions=10)
    d = ledger.to_dict()
    for key in ("declared_synapses", "steps", "transmissions", "undeclared_transmissions", "clean"):
        assert key in d


# ------------------------------------------------------------------ settle_neuron_by_neuron


def test_settle_neuron_by_neuron_output_shape() -> None:
    brain = _small_brain()
    stimulus = _stimulus(brain.connectome.n)
    trajectory, ledger = settle_neuron_by_neuron(
        brain.connectome, brain.neuron_model, stimulus, steps=10
    )
    assert trajectory.shape == (10, brain.connectome.n)


def test_settle_neuron_by_neuron_ledger_is_clean() -> None:
    brain = _small_brain()
    stimulus = _stimulus(brain.connectome.n)
    _, ledger = settle_neuron_by_neuron(
        brain.connectome, brain.neuron_model, stimulus, steps=5
    )
    assert ledger.clean


def test_settle_neuron_by_neuron_trajectory_is_finite() -> None:
    brain = _small_brain()
    stimulus = _stimulus(brain.connectome.n)
    trajectory, _ = settle_neuron_by_neuron(
        brain.connectome, brain.neuron_model, stimulus, steps=20
    )
    assert np.isfinite(trajectory).all()


# ------------------------------------------------------------------ conformance


def test_conformance_returns_expected_keys() -> None:
    brain = _small_brain()
    stimulus = _stimulus(brain.connectome.n)
    result = conformance(brain, stimulus, steps=10)
    for key in ("backend", "steps", "max_abs_deviation", "ledger", "final_active"):
        assert key in result


def test_conformance_deviation_is_small() -> None:
    brain = _small_brain()
    stimulus = _stimulus(brain.connectome.n)
    result = conformance(brain, stimulus, steps=30)
    assert result["max_abs_deviation"] < 1e-10
    moved = brain.settle(stimulus, steps=30).activation
    assert np.abs(moved).max() > 1e-3  # the check compared a moving net, not two silent ones
