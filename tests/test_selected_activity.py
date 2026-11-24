"""``BrainState.part`` reads the activation at the neurons its index names when it is called, on
every backend: an index array written in place, another array made where a freed one stood,
repeated, reordered and negative neurons, a mask and a slice all read as the whole activation
gives them, and the array returned is the caller's own. A state resting on a device still brings a
set of neurons to the host once, whichever array names them."""

from __future__ import annotations

import numpy as np
import pytest
from places import DEVICES, PLACES, place

import cadence as cd
from cadence import brain as brain_module
from cadence.brain import _MlxPlan


def settled(name: str):
    """A composed brain's graph on ``name``, a drive of two rows, a state settled a few sweeps
    under it, the whole activation of such a state on the host, and three neurons whose
    activities differ in the first row, among those synapses reach (the ones a device's
    launch holds)."""
    brain = cd.Brain.compose(8, 3, modules=(8,), seed=2, **place(name))
    graph = brain.brain
    drive = brain.stimulus(np.eye(8)[[0, 3]], memory=False)
    whole = np.array(graph.settle_batch(drive, steps=6).activation)
    reached = np.unique(graph.connectome.post)
    order = reached[np.argsort(whole[0, reached])]
    a, b, c = (int(order[k]) for k in (0, len(order) // 2, -1))
    assert len({whole[0, a], whole[0, b], whole[0, c]}) == 3
    return graph, drive, graph.settle_batch(drive, steps=6), whole, (a, b, c)


@pytest.mark.parametrize("name", PLACES)
def test_an_index_written_in_place_reads_the_neurons_it_names_now(name):
    graph, drive, state, whole, (a, b, c) = settled(name)
    index = np.array([a, b])
    np.testing.assert_array_equal(state.part(index), whole[:, [a, b]])
    index[0] = c
    np.testing.assert_array_equal(state.part(index), whole[:, [c, b]])
    # a state settled after the write reads them too
    np.testing.assert_array_equal(graph.settle_batch(drive, steps=6).part(index), whole[:, [c, b]])
    index[:] = [b, b]  # a neuron named twice
    np.testing.assert_array_equal(state.part(index), whole[:, [b, b]])
    index[:] = [b, a]  # and the first two the other way round
    np.testing.assert_array_equal(state.part(index), whole[:, [b, a]])
    np.testing.assert_array_equal(graph.settle_batch(drive, steps=6).part(index), whole[:, [b, a]])


@pytest.mark.parametrize("name", PLACES)
def test_an_index_made_where_a_freed_one_stood_reads_its_own_neurons(name):
    """Nothing keeps the first array once it is read, so the second can take its place in
    memory, and its identity with it."""
    graph, drive, _, whole, (a, b, _c) = settled(name)
    for _ in range(20):
        state = graph.settle_batch(drive, steps=6)
        first = np.array([a])
        np.testing.assert_array_equal(state.part(first), whole[:, [a]])
        del first
        second = np.array([b])
        np.testing.assert_array_equal(state.part(second), whole[:, [b]])


@pytest.mark.parametrize("name", PLACES)
def test_any_selector_reads_as_the_whole_activation_gives_it(name):
    graph, drive, _, whole, (a, b, c) = settled(name)
    n = whole.shape[1]
    selectors = [
        np.array([c, a, a, b]), [b, c, c], np.array([a, b], dtype=np.int32),
        np.array([-1]), np.array([-3, -2, -1]), np.array([n - 2, -1]),
        np.arange(n) % 3 == 0, slice(2, 5), np.array([], dtype=np.int64),
    ]
    for selector in selectors:
        state = graph.settle_batch(drive, steps=6)
        got = state.part(selector)
        assert got.shape == whole[:, selector].shape
        np.testing.assert_array_equal(got, whole[:, selector])
        np.testing.assert_array_equal(state.part(selector), whole[:, selector])  # and again


@pytest.mark.parametrize("name", PLACES)
def test_the_array_returned_is_the_callers_own(name):
    _, _, state, whole, (a, b, _c) = settled(name)
    index = np.array([a, b])
    taken = state.part(index)
    taken[:] = -7.0
    np.testing.assert_array_equal(state.part(index), whole[:, [a, b]])


@pytest.mark.parametrize("name", DEVICES)
def test_a_set_of_neurons_comes_to_the_host_once_whichever_array_names_it(name, monkeypatch):
    """The motor index of a moment's readout, an equal array and the same array again: one read
    of the device. An array written since names other neurons, and they come once too."""
    brain = cd.Brain.compose(8, 3, modules=(8,), seed=2, **place(name))
    graph, motor = brain.brain, brain.motor_index
    drive = brain.stimulus(np.eye(8)[[0, 3]], memory=False)
    whole = np.array(graph.settle_batch(drive, steps=6).activation)
    state = graph.settle_batch(drive, steps=6)
    reads = []
    host_copy, plan_part = brain_module._host_copy, _MlxPlan.part

    def copied(x):
        reads.append("torch")
        return host_copy(x)

    def launched(self, handle, index):
        reads.append("mlx")
        return plan_part(self, handle, index)

    monkeypatch.setattr(brain_module, "_host_copy", copied)
    monkeypatch.setattr(_MlxPlan, "part", launched)
    for index in (motor, motor.copy(), motor, list(motor)):
        np.testing.assert_array_equal(state.part(index), whole[:, motor])
    assert len(reads) == 1
    assert state.__dict__["activation"] is None  # the whole activation stayed on the device
    index = motor.copy()
    index[0] = brain.sensory_index[0]
    for _ in range(3):
        np.testing.assert_array_equal(state.part(index), whole[:, index])
    assert len(reads) == 2
    if graph._torch is not None:  # the kernel keeps one device index per set of neurons
        again = graph.settle_batch(drive, steps=6)
        kept = len(graph._torch._indices)
        np.testing.assert_array_equal(again.part(motor.copy()), whole[:, motor])
        assert len(graph._torch._indices) == kept
