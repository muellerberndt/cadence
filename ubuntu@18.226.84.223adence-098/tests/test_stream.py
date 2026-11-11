"""Owned state as a stimulus: the trace and the fast synapses."""

from __future__ import annotations

import numpy as np

import cadence as cd


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


def test_trace_remembers_previous_activation_when_state_storage_is_reused() -> None:
    connectome = cd.Connectome.from_synapses(
        4, pre=[], post=[], populations={"hidden": [0, 1], "context": [2, 3]}
    )
    trace = cd.Trace(connectome, decay=0.5)
    activation = np.array([[0.1, 0.7, 0.0, 0.0]])
    state = cd.BrainState(
        v=activation.copy(), activation=activation, adaptation=np.zeros_like(activation), steps=1
    )
    trace.update(state)
    activation[0, 0] = 0.5
    np.testing.assert_array_equal(trace.last, [[0.1, 0.7]])
    trace.update(state)
    expected = 0.25 * np.array([[0.1, 0.7]]) + 0.5 * np.array([[0.5, 0.7]])
    np.testing.assert_allclose(trace.trace, expected)


def test_fast_synapses_bind_a_cue_to_what_was_active_and_fade() -> None:
    # neurons 0..3 are cues, 4..6 are contents; a stream that saw cue 1 with content 6 recalls 6
    connectome = cd.layered(4, 2, 3, density=1.0, seed=0)
    fast = cd.FastSynapses(np.arange(4), np.asarray(connectome.populations["output"]), decay=0.5)
    fast.reset(2)
    s = np.zeros((2, connectome.n))
    out = np.asarray(connectome.populations["output"])
    s[0, 1] = 1.0
    s[0, out[2]] = 1.0  # stream 0: cue 1 with content 2
    s[1, 3] = 1.0
    s[1, out[0]] = 1.0  # stream 1: cue 3 with content 0
    state = cd.BrainState(v=s, activation=s, adaptation=np.zeros_like(s), steps=1)
    fast.update(state, np.array([True, True]))
    drive = np.zeros((2, connectome.n))
    drive[0, 1] = 1.0
    drive[1, 3] = 1.0
    read = fast.read(drive)
    assert read[0].argmax() == 2 and read[1].argmax() == 0
    assert np.isclose(read[0, 2], 1.0)
    fast.update(state, None)  # fades
    assert np.isclose(fast.read(drive)[0, 2], 0.5)
    drive[0, 1] = 0.0
    drive[0, 2] = 1.0  # an unseen cue reads nothing
    assert np.allclose(fast.read(drive)[0], 0.0)


def test_normalized_fast_synapses_read_an_average_of_what_followed() -> None:
    connectome = cd.layered(4, 2, 3, density=1.0, seed=0)
    out = np.asarray(connectome.populations["output"])
    fast = cd.FastSynapses(np.arange(4), out, decay=1.0, normalize=True)
    fast.reset(1)
    s = np.zeros((1, connectome.n))
    s[0, :4] = [3.0, 0.0, 0.0, 0.0]  # a key of any length is written as a unit vector
    state = cd.BrainState(v=s, activation=s, adaptation=np.zeros_like(s), steps=1)
    post = np.zeros((1, 3))
    post[0, 2] = 1.0
    fast.update(state, np.array([True]), post=post)
    fast.update(state, np.array([True]), post=post)  # written twice: still an average of one
    drive = np.zeros((1, connectome.n))
    drive[0, 0] = 0.1  # a cue of any length
    read = fast.read(drive)
    assert np.allclose(read[0], [0.0, 0.0, 1.0])
    other = np.zeros((1, 3))
    other[0, 0] = 1.0
    fast.update(
        state, np.array([True]), post=other
    )  # a third write with the same key, another follower
    assert np.allclose(fast.read(drive)[0], [1 / 3, 0.0, 2 / 3])


def test_replacing_fast_synapses_keep_one_note_per_key() -> None:
    connectome = cd.layered(3, 2, 2, density=1.0, seed=0)
    out = np.asarray(connectome.populations["output"])
    fast = cd.FastSynapses(np.arange(3), out, replace=True)
    fast.reset(1)
    s = np.zeros((1, connectome.n))
    s[0, 1] = 1.0
    state = cd.BrainState(v=s, activation=s, adaptation=np.zeros_like(s), steps=1)
    first = np.array([[1.0, 0.0]])
    second = np.array([[0.0, 1.0]])
    fast.update(state, np.array([True]), post=first)
    fast.update(state, np.array([True]), post=second)  # the same key again: the first note is gone
    drive = np.zeros((1, connectome.n))
    drive[0, 1] = 1.0
    assert np.allclose(fast.read(drive)[0], [0.0, 1.0])


def test_the_trace_reads_a_state_on_the_device_the_same_as_on_the_host() -> None:
    """A settled state that rests on the torch device is sliced there; the trace it feeds is the
    host trace to the last digit."""
    pytest = __import__("pytest")
    if "torch" not in cd.available_backends():
        pytest.skip("no torch")
    w = context_graph(3, 3, 3, seed=1)
    w = w.with_populations(afterglow=w.populations["context"])
    drive = np.zeros((2, w.n))
    drive[:, 0] = 1.0
    host = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0)).settle_batch(drive, steps=30)
    dev = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0), backend="torch", device="cpu").settle_batch(
        drive, steps=30
    )
    a = cd.Trace(w, decay=0.5, source="input", target="afterglow")
    b = cd.Trace(w, decay=0.5, source="input", target="afterglow")
    a.update(host)
    b.update(dev)
    assert np.allclose(a.trace, b.trace)
    drive[:, 1] = 1.0
    a.update(
        cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0)).settle_batch(drive, steps=30, state=host)
    )
    b.update(
        cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0), backend="torch", device="cpu").settle_batch(
            drive, steps=30, state=dev
        )
    )
    assert np.allclose(a.trace, b.trace, atol=1e-6)
