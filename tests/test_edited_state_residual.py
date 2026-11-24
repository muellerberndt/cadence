"""``NeuralGraph.residual`` checks the state it is given as it stands. A state settled on a device
keeps the check of its device while its arrays are only read; once a caller wrote one, the
residual, each neuron's own error, a solve and the apical gain are those of the host arrays, equal
to the float64 reference of a state made new from them, on every backend."""

from __future__ import annotations

import numpy as np
import pytest
from places import DEVICES, PLACES, place

import cadence as cd

TOLERANCE = 1e-4
APICAL = {"modules": (32, 16), "apical": 0.25, "recall": False}


def solved(name: str, **options):
    """A composed brain's graph on ``name``, its learner, a drive of two rows and a state solved
    under it."""
    options = {"modules": (12,), **options}
    brain = cd.Brain.compose(8, 3, seed=3, **options, **place(name))
    graph = brain.brain
    drive = np.array(brain.stimulus(np.eye(8)[[0, 1]], memory=False))
    phase = graph.equilibrate(drive, tolerance=TOLERANCE)
    assert phase.qualified.all()
    return graph, brain.learner, drive, phase.state


def detached(state) -> cd.BrainState:
    return cd.BrainState(
        state.v.copy(), state.activation.copy(), state.adaptation.copy(), state.steps
    )


@pytest.mark.parametrize("name", PLACES)
@pytest.mark.parametrize("nudged", [False, True], ids=["free", "nudged"])
def test_the_residual_of_a_written_state_is_that_of_its_host_arrays(name, nudged):
    graph, learner, drive, state = solved(name)
    nudge = None
    if nudged:
        nudge = learner.nudge_for(learner.targets(np.array([0, 2])), 0.2)
        state = graph.equilibrate(drive, nudge=nudge, tolerance=TOLERANCE).state
    assert (graph.residual(drive, state, nudge=nudge) <= 2 * TOLERANCE).all()
    state.v[0, 0] += 1.0
    reference = graph.residual(drive, detached(state), nudge=nudge, on_device=False)
    assert reference[0] > 0.9 and reference[1] <= 2 * TOLERANCE  # the write, and its row alone
    np.testing.assert_allclose(
        graph.residual(drive, state, nudge=nudge), reference, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        graph.residual(drive, state, nudge=nudge, per_unit=True),
        graph.residual(drive, detached(state), nudge=nudge, per_unit=True), rtol=0, atol=1e-9,
    )
    # a solve from the written state does the work the write asks for
    again = graph.equilibrate(drive, state=state, nudge=nudge, tolerance=TOLERANCE)
    assert again.steps > 0 and again.qualified.all()
    final = graph.residual(drive, detached(again.state), nudge=nudge, on_device=False)
    assert (final <= 2 * TOLERANCE).all()


@pytest.mark.parametrize("name", DEVICES)
def test_a_state_read_and_not_written_keeps_the_check_of_its_device(name):
    graph, _, drive, state = solved(name)
    first = graph.residual(drive, state)
    seen = state.activation.copy()  # the activation comes to the host; no potential does
    np.testing.assert_array_equal(graph.residual(drive, state), first)
    reference = graph._torch is not None and not graph._torch.float64_reference  # MPS
    assert (state.__dict__.get("v") is None) is not reference
    assert state.device is not None
    # the activation alone, written: the equations are those of the potentials, as on the host
    state.activation[:] = 1.0 - seen
    assert (graph.residual(drive, state) <= 2 * TOLERANCE).all()
    assert state.device is None and state.__dict__.get("v") is not None
    np.testing.assert_allclose(
        graph.residual(drive, state),
        graph.residual(drive, detached(state), on_device=False), rtol=0, atol=1e-9,
    )


@pytest.mark.parametrize("name", PLACES)
def test_the_apical_gain_of_a_written_state_is_read_from_its_host_arrays(name):
    graph, _, drive, state = solved(name, **APICAL)
    before = graph.apical_gain(drive, state)
    state.v[:] += 0.3
    state.activation[:] = graph.neuron_model.activation(state.v)
    after = graph.apical_gain(drive, state)
    assert np.abs(after - before).max() > 1e-3
    np.testing.assert_allclose(
        after, graph.apical_gain(drive, detached(state)), rtol=0, atol=1e-5
    )
