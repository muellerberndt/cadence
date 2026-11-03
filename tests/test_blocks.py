"""The block transport: layouts, cached products, frozen ranges, and the block contrast."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.blocks import BlockTransport, block_contrast, layout


def test_layout_cuts_at_contiguous_sets_and_pairs_only_wired_ranges() -> None:
    w = cd.layered(6, 4, 3, density=1.0, seed=0)
    lay = layout(w)
    # input->hidden, hidden->output, output->hidden, and the (zero) lateral block among the outputs
    assert lay.ranges == 3 and lay.pairs == 4
    assert lay.size == 6 * 4 + 4 * 3 + 3 * 4 + 3 * 3
    assert lay.sources() == [0]
    flat = lay.flat(np.arange(1, w.synapses + 1, dtype=float))
    assert np.count_nonzero(flat) == w.synapses
    dense = np.zeros((w.n, w.n))
    dense[w.pre, w.post] = np.arange(1, w.synapses + 1)
    for k, block in enumerate(lay.blocks(flat)):
        a0, a1, b0, b1 = lay.bounds(k)
        assert np.array_equal(block, dense[a0:a1, b0:b1])
    scattered = w.with_populations(odd=[1, 3, 5])  # not one contiguous run: ignored
    assert layout(scattered).ranges == 3
    bare = cd.Connectome(w.n, w.pre, w.post, w.count, w.sign)  # no sets: one block, the full matrix
    assert layout(bare).ranges == 1 and layout(bare).pairs == 1 and layout(bare).size == w.n**2
    assert layout(cd.Connectome.from_synapses(3, pre=[], post=[])).pairs == 0


def test_block_transport_matches_the_full_matrix_and_reuses_still_ranges() -> None:
    w = cd.layered(5, 4, 3, density=1.0, seed=1)
    brain = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0))
    lay = brain.layout
    transport = BlockTransport(lay, lay.flat(brain.weights))
    rng = np.random.default_rng(0)
    s = rng.random((3, w.n))
    assert np.allclose(transport.synaptic_input(s), s @ brain.dense())
    s2 = s.copy()
    s2[:, 5:] = rng.random((3, w.n - 5))  # the inputs did not move
    first = [c is not None and c.copy() for c in transport.cache]
    out = transport.synaptic_input(s2)
    assert np.allclose(out, s2 @ brain.dense())
    assert np.array_equal(transport.cache[0], first[0])  # the input block was reused


@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_fused_kernel_agrees_with_the_loop_with_frozen_inputs(dt: float) -> None:
    w = cd.layered(7, 5, 3, density=1.0, seed=2)
    neuron_model = cd.learning_neuron_model(dt=dt)
    brain = cd.NeuralGraph(w, neuron_model)
    drive = brain.stimulus_levels(np.random.default_rng(3).random((4, w.n)) * 0.5)
    fused = brain.settle_batch(drive, steps=40)
    loop = brain.settle_batch(drive, steps=40, trajectory=True)  # the NumPy loop
    assert (
        np.array_equal(fused.activation, loop.activation)
        or np.abs(fused.activation - loop.activation).max() < 1e-15
    )
    out = list(w.populations["output"])
    mask = np.zeros(w.n)
    mask[out] = 1.0
    target = np.zeros((4, w.n))
    target[:, out] = 0.7
    nudge = cd.Nudge(target, mask, 0.1, softmax_temperature=0.2)
    a = brain.settle_batch(drive, steps=20, state=fused, nudge=nudge)
    b = brain.settle_batch(drive, steps=20, state=loop, nudge=nudge, trajectory=True)
    assert np.abs(a.activation - b.activation).max() < 1e-15
    assert cd.conformance(brain, drive[0], steps=30)["max_abs_deviation"] < 1e-12


def test_block_contrast_matches_the_gram_matrix() -> None:
    w = cd.layered(6, 5, 4, density=1.0, seed=4)
    lay = layout(w)
    rng = np.random.default_rng(5)
    s_plus, s_minus = rng.random((3, w.n)), rng.random((3, w.n))
    s_minus[:, :6] = s_plus[:, :6]  # stimulated inputs are the same in both phases
    gram = (s_plus.T @ s_plus - s_minus.T @ s_minus)[w.pre, w.post]
    assert np.allclose(block_contrast(lay, s_plus, s_minus), gram, atol=1e-12)


def test_layout_survives_with_parameters_and_learning() -> None:
    w = cd.layered(8, 6, 3, density=1.0, seed=6)
    brain = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0))
    changed = brain.with_parameters(bias=np.full(w.n, 0.01))
    assert changed.layout is brain.layout and changed.to_dict()["layout"]["blocks"] == 4
    learner = cd.Learner(brain, w.populations["output"], cd.LearnerConfig(eta=1.0, beta=0.1))
    drive = brain.stimulus_levels(np.random.default_rng(7).random((5, w.n)) * 0.5)
    learner.step(drive, np.array([0, 1, 2, 0, 1]))
    assert learner.brain.layout is brain.layout


@pytest.mark.skipif("torch" not in cd.available_backends(), reason="torch not installed")
def test_torch_precision_option_and_block_transport_agree_with_cpu() -> None:
    w = cd.layered(9, 7, 3, density=1.0, seed=8)
    neuron_model = cd.learning_neuron_model(dt=1.0)
    drive = cd.NeuralGraph(w, neuron_model).stimulus_levels(
        np.random.default_rng(9).random((3, w.n)) * 0.5
    )
    cpu = cd.NeuralGraph(w, neuron_model).settle_batch(drive, steps=30)
    single = cd.NeuralGraph(w, neuron_model, backend="torch", device="cpu", precision="float32")
    double = cd.NeuralGraph(w, neuron_model, backend="torch", device="cpu", precision="float64")
    assert np.abs(single.settle_batch(drive, steps=30).activation - cpu.activation).max() < 1e-5
    assert np.abs(double.settle_batch(drive, steps=30).activation - cpu.activation).max() < 1e-12
    assert single.with_parameters(bias=np.zeros(w.n)).precision == "float32"
    with pytest.raises(ValueError):
        cd.NeuralGraph(w, neuron_model, backend="torch", device="cpu", precision="half")


def test_the_flat_blocks_are_written_in_the_order_of_their_entries() -> None:
    """The scatter of new weights into the flat array writes it front to back, the synapses
    gathered by the layout's permutation (found once); the array is the one synapse order gives."""
    pytest.importorskip("numba")
    brain = cd.Brain.compose(6, 3, modules=(24,), seed=2)
    lay = brain.brain.layout
    held = lay._flat_order
    assert held is not None and lay._flat_order is held
    order, sorted_index = held
    assert np.array_equal(sorted_index, lay.edge_index[order])
    assert np.all(np.diff(sorted_index) > 0) and not np.all(np.diff(lay.edge_index) > 0)
    weights = np.random.default_rng(3).normal(size=brain.connectome.synapses)
    expected = np.zeros(lay.size)
    expected[lay.edge_index] = weights
    np.testing.assert_array_equal(lay.flat(weights), expected)
    # entries already in order: no permutation
    in_order = layout(cd.Connectome.from_synapses(3, pre=[0, 0, 1], post=[1, 2, 2]))
    assert np.all(np.diff(in_order.edge_index) > 0) and in_order._flat_order is None
    np.testing.assert_array_equal(in_order.flat(np.array([1.0, 2.0, 3.0]))[[1, 2, 5]], [1.0, 2.0, 3.0])


def test_a_synapse_at_every_entry_makes_the_gathered_weights_the_flat_array() -> None:
    """Where every entry holds a synapse the weights gathered in entry order are the array, with
    nothing more written; where entries stay empty the scatter writes the zeros between them."""
    pytest.importorskip("numba")
    rng = np.random.default_rng(4)
    full = cd.Brain.compose(6, 3, modules=(24,), seed=2).brain.layout
    sparse = layout(cd.layered(6, 5, 4, density=0.5, seed=1))
    assert len(full.edge_index) == full.size and len(sparse.edge_index) < sparse.size
    for lay in (full, sparse):
        assert lay._flat_order is not None
        weights = rng.normal(size=len(lay.edge_index))
        expected = np.zeros(lay.size)
        expected[lay.edge_index] = weights
        flat = lay.flat(weights)
        assert flat.dtype == np.float64 and flat.flags.c_contiguous
        assert not np.shares_memory(flat, weights)
        np.testing.assert_array_equal(flat, expected)
