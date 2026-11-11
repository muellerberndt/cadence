"""Audit 2026-09-15, classes 6 and 7: grouped nudges on every kernel; the block transport."""

import numpy as np
import pytest

import cadence as cd
from cadence.blocks import BlockTransport, layout
from cadence.reference import settle_neuron_by_neuron


def test_grouped_nudge_with_unequal_slots_and_weights_agrees_on_every_kernel() -> None:
    rng = np.random.default_rng(1)
    c, _ = cd.embedded(10, 2, 5, 16, 7, seed=0)
    model = cd.learning_neuron_model(dt=0.5)
    d = np.zeros((6, c.n))
    for p in range(2):
        d[np.arange(6), p * 10 + rng.integers(0, 10, 6)] = 1.0
    learner = cd.Learner(cd.NeuralGraph(c, model), c.populations["output"], slots=[3, 4])
    labels = np.stack([rng.integers(0, 3, 6), rng.integers(0, 4, 6)], axis=1)
    nudge = learner.nudge_for(learner.targets(labels), 0.3, weight=rng.normal(size=6))
    assert nudge.groups is not None and nudge.beta == pytest.approx(0.15)
    cpu = cd.NeuralGraph(c, model)
    free = cpu.settle_batch(d, steps=80)
    reference = cpu.settle_batch(d, steps=40, state=free, nudge=nudge, trajectory=True).activation
    results = {"fused": (cpu.settle_batch(d, steps=40, state=free, nudge=nudge).activation, 1e-12)}
    if "torch" in cd.available_backends():
        import torch

        devices = ["cpu"]
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            devices.append("mps")
        for device in devices:
            for dense_limit in (2048, 1):
                brain = cd.NeuralGraph(c, model, backend="torch", device=device, dense_limit=dense_limit)
                warm = brain.settle_batch(d, steps=80)
                state = brain.settle_batch(d, steps=40, state=warm, nudge=nudge)
                results[f"torch-{device}-{dense_limit}"] = (
                    np.asarray(state.activation),
                    1e-12 if device == "cpu" else 1e-5,
                )
    if "mlx" in cd.available_backends():
        brain = cd.NeuralGraph(c, model, backend="mlx")
        warm = brain.settle_batch(d, steps=80)
        results["mlx"] = (brain.settle_batch(d, steps=40, state=warm, nudge=nudge).activation, 1e-5)
    for name, (activation, tol) in results.items():
        assert np.abs(activation - reference).max() < tol, name
    # the grouped drive is one softmax per slot
    s = free.activation
    out = learner.output_index
    manual = np.zeros_like(s)
    for g in (0, 1):
        members = out[nudge.groups[out] == g]
        z = s[:, members] / 0.2
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        manual[:, members] = nudge.beta * (learner.targets(labels)[:, members] - p)
    assert nudge.weight is not None
    assert np.allclose(nudge.drive(s), manual * nudge.weight[:, None])


def _random_connectome(populations: dict[str, list[int]], seed: int = 2) -> cd.Connectome:
    rng = np.random.default_rng(seed)
    n = 60
    pre, post = rng.integers(0, n, 900), rng.integers(0, n, 900)
    keep = pre != post
    return cd.Connectome.from_synapses(
        n,
        pre=pre[keep],
        post=post[keep],
        sign=rng.choice([-1.0, 1.0], int(keep.sum())),
        populations=populations,
    )


def test_layout_ignores_scattered_populations_and_every_transport_matches_the_reference() -> None:
    c = _random_connectome(
        {"odd": list(range(1, 60, 2)), "scatter": [3, 7, 20, 21, 22], "block": list(range(30, 45))}
    )
    lay = layout(c)
    assert lay.starts.tolist() == [0, 30, 45, 60]  # only the contiguous population cuts
    model = cd.NeuronModel(gain=0.05, leak=0.1)
    drive = np.random.default_rng(0).random((4, c.n)) * 2
    reference, _ = settle_neuron_by_neuron(c, model, drive[0], steps=50)
    residuals = []
    for dense_limit in (2048, 10, 1):
        brain = cd.NeuralGraph(c, model, dense_limit=dense_limit)
        assert brain._blocked == (dense_limit == 2048)
        state = brain.settle_batch(drive, steps=50)
        assert np.abs(state.activation[0] - reference[-1]).max() < 1e-12
        residuals.append(brain.residual(drive, state))
        if dense_limit < 2048:
            assert brain._csr is not None and brain._csr[1] is not None  # the CSR path ran
    assert np.allclose(residuals[0], residuals[1]) and np.allclose(residuals[0], residuals[2])


def test_max_pairs_fallback_is_one_block_and_still_exact() -> None:
    c = _random_connectome({f"p{i}": [i] for i in range(60)})
    lay = layout(c)
    assert lay.ranges == 1 and lay.pairs == 1 and lay.size == 60 * 60
    assert lay.edge_index.max() < lay.size
    model = cd.NeuronModel(gain=0.05, leak=0.1)
    drive = np.random.default_rng(0).random((2, c.n))
    state = cd.NeuralGraph(c, model).settle_batch(drive, steps=40)
    reference, _ = settle_neuron_by_neuron(c, model, drive[0], steps=40)
    assert np.abs(state.activation[0] - reference[-1]).max() < 1e-12


def test_dense_limit_bounds_the_block_entries_not_the_neuron_count() -> None:
    c = cd.layered(3000, 50, 5, density=0.05, seed=0)  # n squared is far above 2048 squared
    brain = cd.NeuralGraph(c, cd.learning_neuron_model(), dense_limit=2048)
    assert brain.layout.size <= 2048 * 2048 and brain._blocked
    assert cd.NeuralGraph(c, cd.learning_neuron_model(), dense_limit=300)._blocked is False


def test_block_reuse_csr_and_segmented_sum_agree_and_parameters_do_not_share_caches() -> None:
    c = _random_connectome({"block": list(range(30, 45))})
    model = cd.NeuronModel(gain=0.05, leak=0.1)
    rng = np.random.default_rng(3)
    s = rng.random((5, c.n))
    sparse = cd.NeuralGraph(c, model, dense_limit=1)
    blocked = cd.NeuralGraph(c, model)
    blocks = BlockTransport(blocked.layout, blocked._blocks)
    expected = np.zeros_like(s)
    np.add.at(expected.T, c.post, (s[:, c.pre] * sparse.weights).T)
    assert np.abs(sparse._synaptic_input(s) - expected).max() < 1e-12
    assert np.abs(blocks.synaptic_input(s) - expected).max() < 1e-12
    s2 = s.copy()
    s2[:, 31] += 0.5  # a range that hears synapses moved: its product is recomputed
    expected2 = np.zeros_like(s)
    np.add.at(expected2.T, c.post, (s2[:, c.pre] * sparse.weights).T)
    assert np.abs(blocks.synaptic_input(s2) - expected2).max() < 1e-12
    halved = sparse.with_parameters(efficacy=sparse.efficacy * 0.5)
    assert halved._csr is None and sparse._csr is not None
    assert np.abs(halved._synaptic_input(s) - 0.5 * expected).max() < 1e-12
    assert np.abs(sparse._synaptic_input(s) - expected).max() < 1e-12  # the old brain is untouched
