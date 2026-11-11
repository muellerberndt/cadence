"""Gain selection preserves backend identity and an actual adaptive continuation."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd
from cadence.blocks import layout

MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
BACKENDS = [("cpu", None), ("torch", "float64"), ("torch", "float32"), ("mlx", None)]


def prepared_learner(backend, precision):
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "mlx":
        pytest.importorskip("mlx.core")
    wire = cd.layered(2, 3, 2, density=1, init=0.1, lateral=-0.05, seed=3)
    graph = cd.NeuralGraph(
        wire, cd.learning_neuron_model(gain=0.7, dt=1, leak=0.2),
        backend=backend, device="cpu" if backend == "torch" else None,
        precision=precision, dense_limit=16, layout=layout(wire),
        log_gain=np.linspace(-0.03, 0.03, wire.n),
        bias=np.linspace(0.01, 0.07, wire.n),
    )
    synapses = np.ones(wire.synapses, dtype=bool)
    synapses[:2] = False
    neurons = np.ones(wire.n, dtype=bool)
    neurons[[0, 3]] = False
    learner = cd.Learner(
        graph, wire.populations["output"], cd.LearnerConfig(
            qualified=True, free_steps=256, nudged_steps=256, tolerance=3e-3,
            momentum=0.6, normalize=0.7,
        ),
        plastic_synapses=synapses, plastic_neurons=neurons,
        synapse_rate=np.linspace(0.4, 1.2, wire.synapses),
    )
    drive = np.zeros((2, wire.n))
    drive[:, :2] = 0.3 * np.eye(2)
    labels = np.array([0, 1])
    for _ in range(2):
        _, report = learner.step(drive, labels)
        assert report["accepted"] == report["qualified"] == 1
        if backend == "torch":
            assert report["nudged_damping_halvings"] == report["opposite_damping_halvings"] == 0
    return learner, drive


def numeric_state(learner):
    """Read values without materializing lazy parameter or optimizer host fields."""
    graph = learner.brain
    efficacy = (
        graph._torch.host_scale() if graph._efficacy is None else graph.efficacy
    )
    bias = graph._torch.host_bias() if graph._bias is None else graph.bias
    result = {
        "efficacy": efficacy.copy(), "bias": bias.copy(), "log_gain": graph.log_gain.copy(),
        "plastic_synapses": learner.plastic_synapses.copy(),
        "plastic_neurons": learner.plastic_neurons.copy(),
        "synapse_rate": learner.synapse_rate.copy(),
        "outputs": learner.output_index.copy(),
        "updates": learner.updates, "contrast_updates": learner.contrast_updates,
    }
    held = learner.__dict__.get("_device_moments")
    for name in MOMENTS:
        result[name] = (
            held[name].detach().cpu().double().numpy().copy()
            if held is not None else getattr(learner, name).copy()
        )
    return result


def assert_numeric_state(learner, expected):
    actual = numeric_state(learner)
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def same_checkpoint(first, second):
    with np.load(first, allow_pickle=False) as aa, np.load(second, allow_pickle=False) as bb:
        assert set(aa.files) == set(bb.files)
        for name in aa.files:
            np.testing.assert_array_equal(aa[name], bb[name], err_msg=name)


@pytest.mark.parametrize("backend,precision", BACKENDS)
def test_admitted_calibration_preserves_backend_history_and_saved_next_lesson(
    tmp_path, backend, precision,
):
    learner, drive = prepared_learner(backend, precision)
    original = learner.brain
    before = numeric_state(learner)
    original_model = original.neuron_model.to_dict()
    original_config = learner.config
    held = learner.__dict__.get("_device_moments")
    assert any(np.any(before[name]) for name in MOMENTS)
    if backend == "torch":
        assert original._efficacy is original._bias is None
        assert held["holder"] is original._torch

    selected = learner.calibrate(drive, level=0.2, grid=[0.35, 0.5, 1.1])
    winner = learner.brain
    assert selected in (0.35, 0.5, 1.1) and selected != original.neuron_model.gain
    assert winner is not original and winner.connectome is original.connectome
    assert winner.backend == backend and winner.precision == precision
    assert winner.layout is original.layout and winner.dense_limit == original.dense_limit
    assert winner.neuron_model.to_dict() == {**original_model, "gain": selected}
    assert original.neuron_model.to_dict() == original_model
    assert learner.config is original_config
    assert_numeric_state(learner, before)
    if backend == "torch":
        assert str(winner._torch.device) == str(original._torch.device)
        assert winner._torch.dtype == original._torch.dtype
        assert original._efficacy is original._bias is None
        assert learner.__dict__["_device_moments"] is held
    if backend == "mlx":
        assert winner._mlx is not None
    receipt = learner.last_calibration
    assert receipt["selected_gain"] == selected
    assert receipt["attempted_candidates"] == receipt["admitted_candidates"] == 3
    assert all(attempt["qualified"] for attempt in receipt["candidates"])
    assert receipt["total_steps"] <= 3 * learner.config.free_steps

    # Exercise history migration to the winner's kernel BEFORE save can fetch
    # the resident optimizer arrays. Independently evaluate its local recurrence.
    learned, report = learner.step(drive * 0.9, np.array([1, 0]))
    assert report["accepted"] == report["qualified"] == 1
    plus = np.asarray(learned.nudged.activation, dtype=float)
    minus = np.asarray(learned.opposite.activation, dtype=float)
    wire = winner.connectome
    span = 2 * learner.config.beta
    raw = {
        "": ((plus[:, wire.pre] * plus[:, wire.post]
               - minus[:, wire.pre] * minus[:, wire.post]).mean(axis=0) / span),
        "_bias": (plus - minus).mean(axis=0) / span,
    }
    after_next = numeric_state(learner)
    for suffix, contrast in raw.items():
        expected_velocity = (
            learner.config.momentum * before["velocity" + suffix]
            + (1 - learner.config.momentum) * contrast
        )
        expected_second_moment = (
            learner.config.normalize * before["second_moment" + suffix]
            + (1 - learner.config.normalize) * contrast**2
        )
        # Float32 device products need their own arithmetic tolerance; failure
        # to retain the existing history produces much larger discrepancies.
        tolerance = 2e-7 if precision == "float32" or backend == "mlx" else 5e-14
        np.testing.assert_allclose(
            after_next["velocity" + suffix], expected_velocity, atol=tolerance, rtol=0,
        )
        np.testing.assert_allclose(
            after_next["second_moment" + suffix], expected_second_moment,
            atol=tolerance, rtol=0,
        )
    if backend == "torch":
        assert learner.__dict__["_device_moments"]["holder"] is winner._torch
        assert learner.brain._efficacy is learner.brain._bias is None

    restored = cd.Learner.load(
        learner.save(tmp_path / "calibrated.npz"),
        device="cpu" if backend == "torch" else None,
    )
    assert restored.brain.backend == backend and restored.brain.precision == precision
    assert restored.brain.neuron_model.to_dict() == winner.neuron_model.to_dict()
    assert_numeric_state(restored, after_next)
    # These are real next lessons. Their adaptive update must continue from
    # the retained history rather than silently restart after graph selection.
    outcomes = [owner.step(drive * 0.8, np.array([1, 0])) for owner in (learner, restored)]
    assert all(report["accepted"] == report["qualified"] == 1 for _, report in outcomes)
    assert learner.updates == restored.updates == before["updates"] + 2
    assert learner.contrast_updates == restored.contrast_updates == before["contrast_updates"] + 2
    assert_numeric_state(restored, numeric_state(learner))
    same_checkpoint(
        learner.save(tmp_path / "continued-live.npz"),
        restored.save(tmp_path / "continued-loaded.npz"),
    )


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_refused_torch_calibration_preserves_lazy_parameters_and_device_history(precision):
    learner, drive = prepared_learner("torch", precision)
    original = learner.brain
    held = learner.__dict__["_device_moments"]
    tensors = {name: held[name] for name in MOMENTS}
    values = {name: held[name].clone() for name in MOMENTS}
    scale, bias = original._torch.scale.clone(), original._torch.bias_param.clone()
    counts = learner.updates, learner.contrast_updates
    assert original._efficacy is original._bias is None
    assert held["holder"] is original._torch
    learner.config = replace(learner.config, free_steps=0)

    with pytest.raises(RuntimeError, match="no admissible calibration candidate"):
        learner.calibrate(drive, grid=[0.35, 0.5, 1.1])

    assert learner.brain is original and original._efficacy is original._bias is None
    assert learner.__dict__["_device_moments"] is held
    assert held["holder"] is original._torch
    torch = original._torch.torch
    for name in MOMENTS:
        assert held[name] is tensors[name]
        torch.testing.assert_close(held[name], values[name], rtol=0, atol=0)
    torch.testing.assert_close(original._torch.scale, scale, rtol=0, atol=0)
    torch.testing.assert_close(original._torch.bias_param, bias, rtol=0, atol=0)
    assert (learner.updates, learner.contrast_updates) == counts
    assert learner.last_calibration["selected_gain"] is None
    assert learner.last_calibration["admitted_candidates"] == 0
    assert learner.last_calibration["attempted_candidates"] == 3
    assert learner.last_calibration["total_steps"] == 0
    assert all(not attempt["qualified"] for attempt in learner.last_calibration["candidates"])
