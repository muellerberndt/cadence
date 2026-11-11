"""Audit 2026-09-15, class 5: traces and synaptic memories on batches and device states."""

import numpy as np
import pytest

import cadence as cd
from cadence.stream import FastSynapses, Trace


def context_graph(inputs: int, hidden: int, outputs: int, seed: int = 0) -> cd.Connectome:
    """Inputs, a context range of one neuron per hidden neuron, hidden and output neurons: the
    inputs and the context drive the hidden neurons, which exchange signals with the outputs."""
    genome = cd.Genome(
        regions=(
            cd.Region("input", inputs),
            cd.Region("context", hidden),
            cd.Region("hidden", hidden),
            cd.Region("output", outputs),
        ),
        projections=(
            cd.Projection("input", "hidden", density=1.0, sign=0.0, scale=1.0, reciprocal=False),
            cd.Projection("context", "hidden", density=1.0, sign=0.0, scale=1.0, reciprocal=False),
            cd.Projection("hidden", "output", density=1.0, sign=0.0, scale=1.0),
        ),
    )
    return cd.develop(genome, seed=seed)


def _context() -> tuple[cd.Connectome, np.ndarray]:
    rng = np.random.default_rng(0)
    c = context_graph(16, 10, 3, seed=0)
    d = np.zeros((5, c.n))
    for p in range(2):
        d[np.arange(5), p * 8 + rng.integers(0, 8, 5)] = 1.0
    return c, d


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_trace_reads_only_its_source_slice_and_resets_rows(backend: str) -> None:
    if backend == "torch":
        pytest.importorskip("torch")
    c, d = _context()
    brain = cd.NeuralGraph(c, cd.learning_neuron_model(dt=1.0), backend=backend, device="cpu" if backend == "torch" else None)
    trace = Trace(c, decay=0.5, source="hidden", target="context")
    state = brain.settle_batch(d, steps=30)
    trace.update(state)
    if backend == "torch":
        assert state.__dict__.get("activation") is None  # the slice came from the device
    hidden = list(c.populations["hidden"])
    assert np.allclose(trace.trace, 0.5 * np.asarray(state.activation)[:, hidden])
    trace.reset(5, rows=np.array([1, 3]))
    assert not trace.trace[[1, 3]].any() and trace.trace[[0, 2, 4]].any()
    assert trace.cold.tolist() == [False, True, False, True, False]
    trace.keep(np.array([0, 2]))
    assert len(trace.trace) == 2 and trace.cold.tolist() == [False, False]
    stimulated = trace.stimulate(np.zeros((7, c.n)))  # a query cannot erase live streams
    assert len(trace.trace) == 2 and trace.trace.any() and not stimulated.any()
    with pytest.raises(TypeError):
        Trace(c, decay=0.5, focus=1.0, source="hidden", target="context")


def test_synaptic_memory_salience_value_mask_and_batch_changes() -> None:
    rng = np.random.default_rng(0)
    key, value = rng.random((3, 8)), rng.random((3, 4))
    memory = cd.SynapticMemory(np.arange(8), np.arange(8, 12), consolidation=0.5, decay=0.9)
    observed = np.array([[True, False, True, True]] * 3)
    memory.observe(key, value, salience=np.array([0.0, 1.0, 100.0]), value_mask=observed)
    assert np.isfinite(memory.consolidated).all()
    assert not memory.consolidated[:, 1].any()  # an unobserved component teaches nothing
    read = memory.recall(key)
    assert np.allclose(read[:, [0, 2, 3]], value[:, [0, 2, 3]])  # the fast correction is exact
    assert not read[:, 1].any()
    held = memory.strength.copy()
    two = memory.recall(key[:2])  # another batch reads the baseline, preserving live streams
    assert np.allclose(two, memory._delta_unit(key[:2]) @ memory.consolidated)
    np.testing.assert_array_equal(memory.strength, held)
    writes = memory.writes
    memory.observe(key, value, value_mask=np.zeros((3, 4), dtype=bool))
    assert memory.writes == writes  # fully masked rows write nothing
    memory.reset(3, rows=np.array([0]))
    assert np.array_equal(memory.strength[0], memory.consolidated) and memory.mass[0] == 0.0
    plain = cd.SynapticMemory(np.arange(8), np.arange(8, 12), consolidation=0.0)
    plain.observe(key, value, salience=np.full(3, 1e9))
    assert not plain.consolidated.any()  # salience scales the consolidation rate, never creates one
    assert np.allclose(plain.recall(key), value)


def test_fast_synapses_decay_once_per_observation_and_reset_on_new_batches() -> None:
    rng = np.random.default_rng(0)
    key, value = rng.random((3, 8)), rng.random((3, 4))
    hebb = FastSynapses(np.arange(8), np.arange(8, 12), decay=0.5)
    hebb.observe(key, value)
    held = hebb.strength.copy()
    hebb.observe(key, value, write=np.zeros(3, dtype=bool))
    assert np.allclose(hebb.strength, 0.5 * held)  # no writer, still one decay
    delta = FastSynapses(np.arange(8), np.arange(8, 12), rule="delta")
    delta.observe(key, value)
    assert np.allclose(delta.recall(key), value)
    assert not delta.recall(key[:2]).any() and len(delta.strength) == 3
    assert np.allclose(delta.recall(key), value)
    delta.reset(2)
    assert len(delta.strength) == 2 and not delta.recall(key[:2]).any()
