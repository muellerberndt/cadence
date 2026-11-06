"""Host and compiled traces retain the contrast represented by saved phase values."""

from fractions import Fraction

import numpy as np
import pytest

import cadence as cd
from cadence.blocks import block_contrast


def host_learner(path):
    sizes = (5, 7, 3) if path == "blocks" else (2, 3, 2)
    graph = cd.layered(*sizes, density=1, seed=4)
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(), dense_limit=0 if path == "sparse" else 4096)
    if path == "blocks":
        assert brain._blocked and graph.synapses > 4 * graph.n
    elif path == "small":
        assert brain._blocked and graph.synapses <= 4 * graph.n
    else:
        assert not brain._blocked
    return cd.Learner(brain, graph.populations["output"], cd.LearnerConfig(beta=0.5))


def state(activity):
    return cd.BrainState(np.zeros_like(activity), activity, np.zeros_like(activity), 0)


@pytest.mark.parametrize("path", ["blocks", "small", "sparse"])
def test_host_one_ulp_mean_contrast(path):
    learner = host_learner(path)
    graph = learner.brain.connectome
    minus = np.full((31, graph.n), 0.5)
    plus = np.nextafter(minus, 1.0)
    # Exact rational products avoid making the reference repeat the cancellation bug.
    product = Fraction(float(plus[0, 0])) ** 2 - Fraction(0.5) ** 2
    edges, neurons = learner.contrast(state(minus), state(plus), state(minus))
    np.testing.assert_allclose(edges, float(product), rtol=1e-14, atol=1e-28)
    np.testing.assert_array_equal(neurons, (plus - minus)[0])
    if path == "blocks":
        raw = block_contrast(learner.brain.layout, plus, minus)
        np.testing.assert_allclose(raw, float(31 * product), rtol=1e-14, atol=1e-28)


@pytest.mark.parametrize("path", ["small", "sparse"])
def test_saved_host_rows_keep_opposite_change_products(tmp_path, path):
    learner = host_learner(path)
    graph = learner.brain.connectome
    minus = np.full((2, graph.n), 0.5)
    delta = np.spacing(0.5)
    plus = minus + delta
    plus[:, graph.populations["hidden"]] = 0.5 - delta
    saved = tmp_path / "phases.npz"
    np.savez(saved, plus=plus, minus=minus)
    with np.load(saved, allow_pickle=False) as arrays:
        plus, minus = arrays["plus"], arrays["minus"]
    expected = np.array(
        [
            float(Fraction(float(plus[0, a])) * Fraction(float(plus[0, b])) - Fraction(0.25))
            for a, b in zip(graph.pre, graph.post, strict=True)
        ]
    )
    assert (expected < 0).any()  # -delta**2, which separately rounded products erase
    rows, neurons = learner.contrast_rows(state(minus), state(plus), state(minus))
    np.testing.assert_allclose(rows, np.broadcast_to(expected, rows.shape), rtol=1e-14, atol=1e-45)
    np.testing.assert_array_equal(neurons, plus - minus)
    mean, _ = learner.contrast(state(minus), state(plus), state(minus))
    np.testing.assert_allclose(mean, expected, rtol=1e-14, atol=1e-45)
    from cadence import fused

    if fused.available():
        trace, trace_bias = np.zeros_like(rows), np.zeros_like(neurons)
        compiled_step, _ = fused.trace_step(
            trace, trace_bias, 0.0, plus, minus, graph.pre, graph.post, 1.0, np.ones(2)
        )
        np.testing.assert_array_equal(trace, rows)
        np.testing.assert_allclose(compiled_step, expected, rtol=1e-14, atol=1e-45)


def test_saved_device_phase_values_match_host_and_numba_traces_exactly(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("numba")
    from cadence.fused import trace_step

    graph = cd.layered(3, 4, 2, density=1, seed=9)
    config = cd.LearnerConfig(free_steps=20, nudged_steps=12, tolerance=None)
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend="torch", device="cpu")
    live = cd.Learner(brain, graph.populations["output"], config)
    drive = np.zeros((2, graph.n))
    drive[:, :3] = np.eye(3)[:2]
    free = live.free(drive)
    target = live.targets(np.array([0, 1]))
    plus = live.nudged(drive, free, target)
    minus = live.nudged(drive, free, target, sign=-1)
    raw_edges, raw_neurons = brain._torch.contrast_rows(plus.device["s"], minus.device["s"])
    span = 2 * config.beta
    expected_edges = (raw_edges / span).numpy()
    expected_neurons = (raw_neurons / span).numpy()
    saved = tmp_path / "pending-phases.npz"
    np.savez(saved, plus=plus.activation, minus=minus.activation)
    with np.load(saved, allow_pickle=False) as arrays:
        saved_plus, saved_minus = arrays["plus"], arrays["minus"]
    host = cd.Learner(cd.NeuralGraph(graph, cd.learning_neuron_model()), live.outputs, config)
    edges, neurons = host.contrast_rows(free, state(saved_plus), state(saved_minus))
    np.testing.assert_array_equal(edges, expected_edges)
    np.testing.assert_array_equal(neurons, expected_neurons)
    trace, trace_bias = np.zeros_like(edges), np.zeros_like(neurons)
    delta = np.array([0.7, -0.4])
    step_edges, step_neurons = trace_step(
        trace, trace_bias, 0.95, saved_plus, saved_minus, graph.pre, graph.post, span, delta
    )
    np.testing.assert_array_equal(trace, expected_edges)
    np.testing.assert_array_equal(trace_bias, expected_neurons)
    np.testing.assert_array_equal(step_edges, (delta[:, None] * expected_edges).mean(axis=0))
    np.testing.assert_array_equal(step_neurons, (delta[:, None] * expected_neurons).mean(axis=0))
