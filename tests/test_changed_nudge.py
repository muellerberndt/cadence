"""A nudge's arrays are the caller's to write between calls. After a write in place to its target,
its row weights, its mask or its anchor, ``NeuralGraph.residual`` reports the equations under the
nudge as it reads now, a settle sweeps under it and ``equilibrate`` does the work the new boundary
asks for, on every backend and with the compiled CPU kernels on and off. A nudge nobody wrote
keeps what was made from it, and on MLX the check its launch made."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from places import PLACES, place

import cadence as cd
from cadence import brain as brain_module
from cadence import fused

TOLERANCE = 1e-4
KERNELS = pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "numpy-loop"])


def kernels(monkeypatch, compiled: bool) -> None:
    """The compiled CPU kernels as this run has them, or the NumPy loop in their place."""
    if compiled and not brain_module._FUSED:
        pytest.skip("CADENCE_FUSED=0 reads the NumPy loop alone")
    monkeypatch.setattr(brain_module, "_FUSED", compiled)


def taught(name: str):
    """A composed brain's graph on ``name``, a drive of two rows, its learner, and a state
    settled under the learner's nudge toward the first action at a large beta."""
    brain = cd.Brain.compose(8, 3, modules=(12,), seed=3, **place(name))
    graph, learner = brain.brain, brain.learner
    drive = np.array(brain.stimulus(np.eye(8)[[0, 1]], memory=False))
    nudge = learner.nudge_for(learner.targets(np.array([0, 0])), 0.3, weight=np.ones(2))
    solved = graph.equilibrate(drive, nudge=nudge, budget=512, tolerance=TOLERANCE)
    assert solved.qualified.all()
    assert (graph.residual(drive, solved.state, nudge=nudge) <= 2 * TOLERANCE).all()
    return graph, learner, drive, nudge, solved.state


def reads_as_written(graph, drive, nudge, state) -> None:
    """The check, a few sweeps and a solve under ``nudge`` are those under a nudge made new from
    what it holds now, which the state settled before the write does not satisfy."""
    fresh = replace(nudge)  # its own copies of the arrays, and nothing kept from earlier calls
    reference = graph.residual(drive, state, nudge=fresh, on_device=False)
    assert (reference > 10 * TOLERANCE).any()
    np.testing.assert_allclose(
        graph.residual(drive, state, nudge=nudge), reference, rtol=0, atol=1e-9
    )
    swept = graph.settle_batch(drive, steps=3, state=state, nudge=nudge)
    again = graph.settle_batch(drive, steps=3, state=state, nudge=fresh)
    np.testing.assert_array_equal(swept.v, again.v)
    solved = graph.equilibrate(drive, state=state, nudge=nudge, budget=512, tolerance=TOLERANCE)
    assert solved.steps > 0 and solved.qualified.all()
    final = graph.residual(drive, solved.state, nudge=fresh, on_device=False)
    assert (final <= 2 * TOLERANCE).all()
    np.testing.assert_allclose(
        graph.residual(drive, solved.state, nudge=nudge), final, rtol=0, atol=1e-5
    )


@pytest.mark.parametrize("name", PLACES)
@KERNELS
def test_a_target_written_in_place_is_checked_and_solved_as_written(name, compiled, monkeypatch):
    kernels(monkeypatch, compiled)
    graph, learner, drive, nudge, state = taught(name)
    nudge.target[:] = learner.targets(np.array([1, 1]))
    reads_as_written(graph, drive, nudge, state)


@pytest.mark.parametrize("name", PLACES)
@KERNELS
def test_row_weights_written_in_place_are_checked_and_solved_as_written(
    name, compiled, monkeypatch
):
    kernels(monkeypatch, compiled)
    graph, _, drive, nudge, state = taught(name)
    nudge.weight[:] = [-1.0, 0.25]  # the first row pushed away, the second pulled a quarter
    reads_as_written(graph, drive, nudge, state)


@pytest.mark.parametrize("name", PLACES)
@KERNELS
def test_a_mask_written_in_place_is_checked_and_solved_as_written(name, compiled, monkeypatch):
    kernels(monkeypatch, compiled)
    graph, _, drive, nudge, state = taught(name)
    nudge.mask[:] = 0.0  # no neuron is pulled: the free equations
    free = graph.residual(drive, state, on_device=False)
    np.testing.assert_allclose(
        graph.residual(drive, state, nudge=nudge), free, rtol=0, atol=1e-9
    )
    reads_as_written(graph, drive, nudge, state)


@pytest.mark.parametrize("name", PLACES)
@KERNELS
def test_a_target_and_an_anchor_given_once_for_every_row_are_read_as_written(
    name, compiled, monkeypatch
):
    """One row of targets and of anchors serves every row of the batch: what the kernels read
    of them is made from the nudge, and made again after a write."""
    kernels(monkeypatch, compiled)
    brain = cd.Brain.compose(8, 3, modules=(12,), seed=3, **place(name))
    graph, n = brain.brain, brain.connectome.n
    drive = np.array(brain.stimulus(np.eye(8)[[0, 1]], memory=False))
    motor = brain.motor_index
    mask, target, gain = np.zeros(n), np.zeros(n), np.zeros(n)
    mask[motor], target[motor[0]], gain[motor] = 1.0, 1.0, 0.5
    nudge = cd.Nudge(target, mask, 0.3, anchor=np.zeros(n), anchor_gain=gain)
    solved = graph.equilibrate(drive, nudge=nudge, budget=512, tolerance=TOLERANCE)
    assert solved.qualified.all()
    nudge.target[motor] = [0.0, 1.0, 0.0]
    reads_as_written(graph, drive, nudge, solved.state)
    solved = graph.equilibrate(drive, nudge=nudge, budget=512, tolerance=TOLERANCE)
    nudge.anchor[motor] = 0.8
    reads_as_written(graph, drive, nudge, solved.state)


@pytest.mark.skipif(not brain_module._FUSED, reason="CADENCE_FUSED=0 reads the NumPy loop alone")
def test_the_kernels_read_whole_rows_where_they_rest_and_keep_what_is_made():
    brain = cd.Brain.compose(8, 3, modules=(12,), seed=3, backend="cpu")
    learner, n = brain.learner, brain.connectome.n
    nudge = learner.nudge_for(learner.targets(np.array([0, 2])), 0.3, weight=np.ones(2))
    first = fused._nudge_arrays(nudge, 2, n)
    again = fused._nudge_arrays(nudge, 2, n)
    assert first[0] is nudge.target and first[6] is nudge.weight  # read where they rest
    assert all(a is b for a, b in zip(first, again, strict=True) if isinstance(a, np.ndarray))
    nudge.mask[brain.motor_index[0]] = 0.0  # the group's ids and its mask are made again
    made = fused._nudge_arrays(nudge, 2, n)
    assert made[1] is not first[1] and made[1][brain.motor_index[0]] == 0.0
    assert made[0] is nudge.target
    # a target that cannot be written is copied once, and the copy holds while it reads the same
    frozen = replace(nudge)
    frozen.target.flags.writeable = False
    copied = fused._nudge_arrays(frozen, 2, n)[0]
    assert copied is not frozen.target and copied.flags.writeable
    assert fused._nudge_arrays(frozen, 2, n)[0] is copied


@pytest.mark.parametrize("name", ["mlx"])
def test_a_nudge_nobody_wrote_reads_the_check_its_launch_made(name):
    graph, learner, drive, nudge, state = taught(name)
    held = graph.residual(drive, state, nudge=nudge)
    assert (held <= 2 * TOLERANCE).all()
    assert state.__dict__.get("v") is None  # the launch's own check: no state came to the host
    nudge.target[:] = learner.targets(np.array([2, 2]))
    assert (graph.residual(drive, state, nudge=nudge) > 10 * TOLERANCE).any()
    assert state.__dict__.get("v") is not None  # the float64 reference read it there
    nudge.target[:] = learner.targets(np.array([0, 0]))  # written back: the launch's check holds
    np.testing.assert_array_equal(graph.residual(drive, state, nudge=nudge), held)
