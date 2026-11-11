"""Apical topology and structural diagnostics must preserve their declared model scope."""

import json

import numpy as np
import pytest

import cadence as cd


def test_symmetric_apical_edges_do_not_admit_the_plain_sum_ep_argument():
    connectome = cd.Connectome.from_synapses(2, pre=[0, 1], post=[1, 0], sign=[0.4, 0.4])
    plain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
    assert cd.ep_structure(plain).compatible
    apical = cd.Apical(np.array([True, False]), np.array([0, -1]), 0.25)
    graph = cd.NeuralGraph(connectome, plain.neuron_model, apical=apical)
    with pytest.raises(ValueError, match="apical"):
        cd.ep_structure(graph)


@pytest.mark.parametrize("field,kind", [
    ("apical_synapses", "nan"), ("apical_synapses", "numeric"),
    ("apical_groups", "fraction"), ("apical_groups", "overflow"),
    ("apical_synapses", "missing"), ("apical_groups", "missing"),
    ("fraction", "bool"), ("floor", "string"),
])
def test_checkpoint_refuses_coercible_or_incomplete_compartments(tmp_path, field, kind):
    original = cd.Brain.compose(3, 2, modules=(5, 4), apical=0.25, seed=3, recall=False)
    path = original.learner.save(tmp_path / "good.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if kind == "missing":
        del arrays[field]
    elif field in ("fraction", "floor"):
        meta = json.loads(str(arrays["meta"]))
        meta["apical"][field] = True if kind == "bool" else "0.25"
        arrays["meta"] = np.array(json.dumps(meta))
    elif kind == "overflow":
        arrays[field] = np.zeros_like(arrays[field], dtype=np.uint64)
        arrays[field][0] = 2**63
    else:
        arrays[field] = arrays[field].astype(float)
        if kind in ("nan", "fraction"):
            arrays[field][0] = np.nan if kind == "nan" else 0.5
    altered = tmp_path / "altered.npz"
    np.savez(altered, **arrays)
    with pytest.raises(ValueError):
        cd.Learner.load(altered)
    # Valid writer-produced topology and continuation still round-trip exactly.
    loaded = cd.Learner.load(path)
    drive = original.stimulus([[1.0, 0.5, 0.0]])
    np.testing.assert_array_equal(loaded.free(drive).activation,
                                  original.learner.free(drive).activation)


@pytest.mark.parametrize("gain", [0.02, 0.5, 1.0, 2.0, 3.0])
def test_expert_code_step_bound_stabilizes_the_actual_model_gain(gain):
    code = cd.LearnedCode(1, 8, 2, cd.learning_neuron_model(gain=gain, leak=1.0), seed=2)
    assert code.rest_stability() >= code.stability(np.zeros((1, code.units)))
    code.restep(code.rest_stability())
    epsilon = 1e-7
    perturbation = epsilon * np.eye(code.units)
    drive = np.broadcast_to(code.thresholds, perturbation.shape)

    def advance(v):
        state = cd.BrainState(v, code.model.activation(v), np.zeros_like(v), 0)
        return code.graph.settle_batch(drive, steps=1, state=state, nudge=code.standing()).v

    jacobian = ((advance(perturbation) - advance(-perturbation)) / (2 * epsilon)).T
    assert max(abs(np.linalg.eigvals(jacobian))) < 1.0
    v = 1e-4 * code.dictionary[[0]]
    warm = cd.BrainState(v, code.model.activation(v), np.zeros_like(v), 0)
    phase = code.settle(code.thresholds[None], warm, budget=512, tolerance=1e-10, damping=0)
    assert phase.qualified.all()
