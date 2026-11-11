"""Operating points and seams: the calibrated threshold, the naive seam and the preflight."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def test_calibrate_bias_brings_populations_to_their_targets() -> None:
    connectome = cd.layered(4, 12, 2, density=0.6, seed=1)
    brain = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0))
    amp = brain.neuron_model.stimulus_amplitude
    drives = np.zeros((2, connectome.n))
    drives[0, [0, 1]] = amp
    drives[1, [2, 3]] = amp
    outputs = list(connectome.populations["output"])
    bias = cd.calibrate_bias(
        brain, drives, {"hidden": 0.3, "output": 0.5}, per_neuron=False, steps=60, tolerance=1e-4
    )
    state = brain.with_parameters(bias=bias).settle_batch(drives, steps=60, tolerance=1e-4)
    assert state.activation[:, connectome.populations["hidden"]].mean() == pytest.approx(
        0.3, abs=0.03
    )
    assert state.activation[:, outputs].mean() == pytest.approx(0.5, abs=0.03)
    # per neuron: each output cell at the target on its own, and the brain untouched
    bias2 = cd.calibrate_bias(brain, drives, {tuple(outputs): 0.5}, per_neuron=True, steps=60)
    state2 = brain.with_parameters(bias=bias2).settle_batch(drives, steps=60, tolerance=1e-4)
    for i in outputs:
        assert state2.activation[:, i].mean() == pytest.approx(0.5, abs=0.03)
    assert np.all(brain.bias == 0.0)
    with pytest.raises(ValueError):
        cd.calibrate_bias(brain, drives, {"output": 1.5})


def test_naive_efficacy_equalises_the_plastic_classes_and_keeps_the_rest() -> None:
    connectome = cd.Connectome.from_synapses(
        4, pre=[0, 1, 0, 2], post=[2, 2, 3, 3], count=[2, 8, 5, 1], sign=[1, 1, -1, 1]
    )
    plastic = np.array([True, True, False, False])
    efficacy = cd.naive_efficacy(connectome, plastic)
    weight = connectome.count * efficacy
    assert weight[0] == pytest.approx(weight[1]) and weight[0] > 0
    assert efficacy[2] == -1.0 and efficacy[3] == 1.0
    with pytest.raises(ValueError):
        cd.naive_efficacy(connectome, np.array([True]))


def test_seam_report_counts_classes_and_coverage() -> None:
    connectome = cd.Connectome.from_synapses(
        6,
        pre=[0, 1, 0, 3],
        post=[4, 4, 5, 5],
        count=[3, 7, 2, 9],
        populations={"pre": [0, 1, 2], "post": [4, 5]},
    )
    report = cd.seam_report(connectome, "pre", "post")
    assert report["classes"] == 3 and report["synapses"] == 12.0
    assert report["coverage_pre"] == pytest.approx(2 / 3)
    assert report["classes_per_post"] == {4: 2, 5: 1}


def test_preflight_names_each_finding_and_its_remedy() -> None:
    # inputs A (0, 1) and B (2, 3); hidden 4, 5 answer A, 6, 7 answer B, 8 both; outputs 9, 10 from every hidden cell
    pre = [0, 1, 0, 1, 2, 3, 2, 3, 0, 2] + [h for h in range(4, 9) for _ in (9, 10)]
    post = [4, 4, 5, 5, 6, 6, 7, 7, 8, 8] + [o for _ in range(4, 9) for o in (9, 10)]
    count = [100] * 10 + [20] * 10
    connectome = cd.Connectome.from_synapses(
        11,
        pre=pre,
        post=post,
        count=count,
        populations={"a": [0, 1], "b": [2, 3], "hidden": [4, 5, 6, 7, 8], "output": [9, 10]},
    )
    outputs = [9, 10]
    hidden = np.zeros(connectome.n, dtype=bool)
    hidden[[4, 5, 6, 7, 8]] = True
    plastic = hidden[connectome.pre]
    brain = cd.NeuralGraph(connectome, cd.NeuronModel(gain=0.05))
    amp = brain.neuron_model.stimulus_amplitude
    drives = np.zeros((2, connectome.n))
    drives[0, [0, 1]] = amp
    drives[1, [2, 3]] = amp
    calibrated = brain.with_parameters(
        bias=cd.calibrate_bias(brain, drives, {"output": 0.5}, per_neuron=True, steps=60)
    )
    clean = cd.preflight(calibrated, outputs, plastic, drives, eligibility_min=1, steps=60)
    assert clean["warnings"] == []
    assert clean["shared"][(0, 1)] == pytest.approx(0.2) and clean["eligibility"] == {
        9: [3, 3],
        10: [3, 3],
    }
    # a readout on its rail, the same code under both drives, a seam with too little active input
    on_rail = cd.preflight(brain, outputs, plastic, drives, eligibility_min=1, steps=60)
    assert any("calibrate_bias" in w and "output 9" in w for w in on_rail["warnings"])
    same = cd.preflight(calibrated, outputs, plastic, np.vstack([drives[0], drives[0]]), steps=60)
    assert any("specific fact" in w for w in same["warnings"])
    assert any("same situation" in w for w in same["warnings"]) and same["inputs"][(0, 1)] == 1.0
    assert clean["inputs"][(0, 1)] == 0.0
    thin = cd.preflight(calibrated, outputs, plastic, drives, eligibility_min=100, steps=60)
    assert any("seam is thin" in w for w in thin["warnings"])
    with pytest.raises(ValueError):
        cd.preflight(brain, outputs, np.array([True]), drives)
