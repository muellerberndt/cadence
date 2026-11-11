"""Adversarial regressions for the submission's numerical boundaries."""

import numpy as np
import pytest

import cadence as cd


def test_parameter_replacement_cannot_overflow_effective_weights():
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1], count=[1e100])
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model())
    before = brain.weights.copy()
    with pytest.raises(ValueError, match="effective synaptic weights"):
        brain.with_parameters(efficacy=np.array([1e300]))
    with pytest.raises(ValueError, match="effective synaptic weights"):
        brain.efficacy = np.array([1e300])
    np.testing.assert_array_equal(brain.weights, before)
