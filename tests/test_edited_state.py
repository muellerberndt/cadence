"""A settled state's host arrays are the caller's to write on every backend, and the state has one
authority. A state that rests on a device keeps resting there while its arrays are only read: a
settle from it uploads nothing of it and continues as from a state never read. Once an array it
handed out is written, its host arrays are the state: a settle of no sweeps, of one and of a few,
a solve and a contrast read them as they stand, as a state made new from copies of them does, and
as the NumPy backend does. No state shares the memory of another's arrays."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from places import DEVICES, PLACES, place

import cadence as cd

SWEEPS = pytest.mark.parametrize("sweeps", [0, 1, 3])
ARRAYS = ("v", "activation", "adaptation")


def settled(name: str, **options):
    """A composed brain's graph on ``name``, a drive of two rows and a state solved under it."""
    brain = cd.Brain.compose(8, 3, modules=(12,), seed=3, **options, **place(name))
    graph = brain.brain
    drive = np.array(brain.stimulus(np.eye(8)[[0, 1]], memory=False))
    return graph, drive, graph.equilibrate(drive).state


def detached(state) -> cd.BrainState:
    """A state made new from copies of ``state``'s host arrays: the host alone holds it."""
    return cd.BrainState(
        state.v.copy(), state.activation.copy(), state.adaptation.copy(), state.steps
    )


def same(a, b) -> None:
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def rests_on(graph, state) -> bool:
    """Whether ``state`` rests on the device ``graph`` settles on."""
    kernel = graph._torch if graph._torch is not None else graph._mlx
    return kernel is not None and state.device is not None and (
        state.device.get("holder") is kernel.root
    )


@pytest.mark.parametrize("name", PLACES)
@SWEEPS
def test_a_state_whose_arrays_were_written_continues_from_them(name, sweeps):
    graph, drive, state = settled(name)
    state.v[:] += 1.0
    state.activation[:] = graph.neuron_model.activation(state.v)
    twin = detached(state)
    same(graph.settle_batch(drive, steps=sweeps, state=state),
         graph.settle_batch(drive, steps=sweeps, state=twin))
    assert state.device is None  # the host arrays are the state from the write on
    np.testing.assert_array_equal(state.v, twin.v)
    # and it takes further writes as any state of the host does
    state.v[0] -= 0.25
    twin.v[0] -= 0.25
    same(graph.settle_batch(drive, steps=sweeps, state=state),
         graph.settle_batch(drive, steps=sweeps, state=twin))


@pytest.mark.parametrize("name", PLACES)
@pytest.mark.parametrize("written", ARRAYS)
def test_one_array_written_alone_is_read_with_the_others_as_the_device_held_them(name, written):
    graph, drive, state = settled(name)
    reference = detached(graph.equilibrate(drive).state)  # what the device holds, on the host
    getattr(state, written)[1] += 0.5
    getattr(reference, written)[1] += 0.5
    same(graph.settle_batch(drive, steps=2, state=state),
         graph.settle_batch(drive, steps=2, state=reference))
    same(state, reference)  # the arrays not written read as the device held them
    solved = graph.equilibrate(drive, state=state, budget=64, tolerance=1e-4)
    expected = graph.equilibrate(drive, state=reference, budget=64, tolerance=1e-4)
    assert solved.steps == expected.steps
    np.testing.assert_array_equal(solved.residual, expected.residual)


@pytest.mark.parametrize("name", DEVICES)
@SWEEPS
def test_a_state_that_was_read_and_not_written_stays_on_its_device(name, sweeps):
    """Reading every array brings it to the host once; a settle from the state then reads
    nothing more of it and uploads nothing of it. The same state, once written, is uploaded."""
    graph, drive, state = settled(name)
    unread = graph.equilibrate(drive).state
    read = {field: getattr(state, field).copy() for field in ARRAYS}
    assert rests_on(graph, state)

    def forbidden(key):
        raise AssertionError(f"a settle from a state that rests on its device fetched {key}")

    state.device["fetch"] = forbidden
    continued = graph.settle_batch(drive, steps=sweeps, state=state)
    assert rests_on(graph, state) and rests_on(graph, continued)
    same(continued, graph.settle_batch(drive, steps=sweeps, state=unread))
    for field in ARRAYS:  # and the arrays read before are still the caller's own, unchanged
        np.testing.assert_array_equal(getattr(state, field), read[field])
    state.adaptation[0, 0] += 0.5
    same(graph.settle_batch(drive, steps=sweeps, state=state),
         graph.settle_batch(drive, steps=sweeps, state=detached(state)))
    assert not rests_on(graph, state)


@pytest.mark.parametrize("name", PLACES)
def test_no_state_shares_the_memory_of_anothers_arrays(name):
    """A state continued without a sweep, a state settled from host arrays and a copy by
    ``dataclasses.replace`` each read as they were made, whatever is written elsewhere."""
    graph, drive, state = settled(name)
    rests = graph.settle_batch(drive, steps=0, state=state)
    held = {field: getattr(rests, field).copy() for field in ARRAYS}
    state.v[:] += 5.0
    state.adaptation[:] += 5.0
    for field in ARRAYS:
        np.testing.assert_array_equal(getattr(rests, field), held[field], err_msg=field)
    start = detached(rests)
    start.adaptation[:] = 0.25
    onward = graph.settle_batch(drive, steps=2, state=start)
    start.adaptation[:] = 7.0
    start.v[:] = -3.0
    if graph.neuron_model.adaptation is None:  # carried through a settle untouched
        np.testing.assert_array_equal(onward.adaptation, np.full_like(held["v"], 0.25))
    np.testing.assert_array_equal(
        onward.v, graph.settle_batch(drive, steps=2, state=detached(rests)).v
    )


@pytest.mark.parametrize("name", PLACES)
def test_a_replaced_state_reads_its_own_arrays(name):
    """``dataclasses.replace`` hands the new state the arrays the old one handed out."""
    graph, drive, state = settled(name)
    unread = graph.equilibrate(drive).state
    copy = replace(state, steps=7)
    assert copy.steps == 7
    if name != "numpy":
        assert rests_on(graph, copy)  # unwritten, it continues where the old one rests
    same(graph.settle_batch(drive, steps=2, state=copy),
         graph.settle_batch(drive, steps=2, state=unread))
    copy.v[:] += 1.0
    same(graph.settle_batch(drive, steps=2, state=copy),
         graph.settle_batch(drive, steps=2, state=detached(copy)))


@pytest.mark.parametrize("name", PLACES)
def test_a_learner_contrasts_a_written_phase_as_it_stands(name):
    """The contrast of two phases reads their activations: a phase whose activation a caller
    wrote is read on the host, on a device as on the NumPy backend."""
    brain = cd.Brain.compose(8, 3, modules=(12,), seed=3, **place(name))
    reference = cd.Brain.compose(8, 3, modules=(12,), seed=3, backend="cpu")
    drive = np.array(brain.stimulus(np.eye(8)[[0, 1]], memory=False))
    contrasts = []
    for one in (brain, reference):
        learner = one.learner
        free = learner.free(drive)
        nudged = learner.nudged(drive, free, learner.targets(np.array([0, 2])))
        motor = one.motor_index
        nudged.activation[:, motor] = [[0.9, 0.1, 0.0], [0.0, 0.2, 0.8]]
        contrasts.append(learner.contrast(free, nudged))
        if one is brain and name != "numpy":
            assert one.brain.contrast_on_device(nudged, free) is None
    for ours, theirs in zip(contrasts[0], contrasts[1], strict=True):
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-4)
    assert np.abs(contrasts[1][1][reference.motor_index]).max() > 0.1  # the write is in it
