"""Small phase contrasts retain their sign and magnitude at the declared precision."""

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")


def make_torch(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    graph = cd.layered(2, 3, 2, density=1, seed=4)
    return cd.NeuralGraph(
        graph, cd.learning_neuron_model(), backend="torch", device=device, precision="float32"
    )


def reference(graph, plus, minus):
    """Float64 products of the exact supplied float32 values, independent of factoring."""
    plus, minus = plus.astype(np.float64), minus.astype(np.float64)
    rows = plus[:, graph.pre] * plus[:, graph.post] - minus[:, graph.pre] * minus[:, graph.post]
    return rows, plus - minus


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("fixed_inputs", [False, True])
def test_one_ulp_phase_change_keeps_aggregate_contrast(device, fixed_inputs):
    brain = make_torch(device)
    kernel = brain._torch
    graph = brain.connectome
    minus = np.full((31, graph.n), 0.5, dtype=np.float32)
    plus = np.nextafter(minus, np.float32(1))
    if fixed_inputs:
        plus[:, graph.populations["input"]] = minus[:, graph.populations["input"]]
    expected_edges, expected_neurons = reference(graph, plus, minus)
    tensors = [torch.tensor(x, device=device) for x in (plus, minus)]
    edges, neurons = kernel.contrast_tensors(*tensors)
    reverse_edges, reverse_neurons = kernel.contrast_tensors(*tensors[::-1])
    np.testing.assert_allclose(edges.cpu(), expected_edges.sum(axis=0), rtol=2e-6, atol=1e-12)
    np.testing.assert_allclose(neurons.cpu(), expected_neurons.sum(axis=0), rtol=0, atol=0)
    assert bool((edges > 0).all()) and bool((reverse_edges < 0).all())
    np.testing.assert_allclose(reverse_edges.cpu(), -edges.cpu(), rtol=2e-6, atol=1e-12)
    np.testing.assert_array_equal(reverse_neurons.cpu(), -neurons.cpu())
    zero_edges, zero_neurons = kernel.contrast_tensors(tensors[0], tensors[0])
    assert torch.count_nonzero(zero_edges) == 0 and torch.count_nonzero(zero_neurons) == 0


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_row_contrast_retains_small_product_difference(device):
    brain = make_torch(device)
    kernel, graph = brain._torch, brain.connectome
    minus = np.full((3, graph.n), 0.5, dtype=np.float32)
    delta = np.spacing(np.float32(0.5))
    plus = minus + delta
    plus[:, graph.populations["hidden"]] = minus[:, graph.populations["hidden"]] - delta
    # Edges joining opposite changes have (.5+d)(.5-d)-.25 = -d**2.
    # Rounding those two products separately incorrectly returns zero in float32.
    expected_edges, expected_neurons = reference(graph, plus, minus)
    tensors = [torch.tensor(x, device=device) for x in (plus, minus)]
    edges, neurons = kernel.contrast_rows(*tensors)
    reverse_edges, reverse_neurons = kernel.contrast_rows(*tensors[::-1])
    np.testing.assert_allclose(edges.cpu(), expected_edges, rtol=1e-6, atol=1e-21)
    np.testing.assert_array_equal(neurons.cpu(), expected_neurons)
    assert (expected_edges < 0).any() and (expected_edges > 0).any()
    np.testing.assert_array_equal(np.sign(edges.cpu().numpy()), np.sign(expected_edges))
    np.testing.assert_array_equal(np.sign(reverse_edges.cpu().numpy()), -np.sign(expected_edges))
    np.testing.assert_allclose(reverse_edges.cpu(), -edges.cpu(), rtol=1e-6, atol=1e-21)
    np.testing.assert_array_equal(reverse_neurons.cpu(), -neurons.cpu())
    zero_edges, zero_neurons = kernel.contrast_rows(tensors[0], tensors[0])
    assert torch.count_nonzero(zero_edges) == 0 and torch.count_nonzero(zero_neurons) == 0


def test_mlx_one_ulp_phase_change_keeps_aggregate_contrast():
    mx = pytest.importorskip("mlx.core")
    graph = cd.layered(2, 3, 2, density=1, seed=4)
    kernel = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend="mlx")._mlx
    minus = np.full((31, graph.n), 0.5, dtype=np.float32)
    plus = np.nextafter(minus, np.float32(1))
    expected_edges, expected_neurons = reference(graph, plus, minus)
    tensors = [mx.array(x) for x in (plus, minus)]
    edges, neurons = kernel.contrast(*tensors)
    reverse_edges, reverse_neurons = kernel.contrast(*tensors[::-1])
    np.testing.assert_allclose(edges, expected_edges.sum(axis=0), rtol=2e-6, atol=1e-12)
    np.testing.assert_array_equal(neurons, expected_neurons.sum(axis=0))
    assert (edges > 0).all() and (reverse_edges < 0).all()
    np.testing.assert_allclose(reverse_edges, -edges, rtol=2e-6, atol=1e-12)
    np.testing.assert_array_equal(reverse_neurons, -neurons)
    zero_edges, zero_neurons = kernel.contrast(tensors[0], tensors[0])
    assert not zero_edges.any() and not zero_neurons.any()
