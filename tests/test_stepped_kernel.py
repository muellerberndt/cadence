"""A damped solve on torch settles each halved step on a kernel the live kernel keeps
(``_TorchKernel.stepped``): made once per step from the parameters on the device, given new
parameters where they changed, its states resting on the live kernel; the solve is the one a kernel
made afresh for that step gives, a lesson whose phases halve updates on the device, and the kernels
hold no reference cycle. A settle's movement stays on the device until it is read."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module

torch = pytest.importorskip("torch")


def graph_on_torch(seed: int = 3) -> tuple[cd.NeuralGraph, np.ndarray]:
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=seed)
    graph = cd.NeuralGraph(
        brain.brain.connectome, brain.brain.neuron_model, backend="torch", device="cpu",
        efficacy=brain.brain.efficacy, bias=brain.brain.bias,
    )
    return graph, brain.stimulus(np.eye(12)[[1, 7, 3]], memory=False)


def test_each_halved_step_has_one_kernel_made_once(monkeypatch):
    graph, drive = graph_on_torch()
    graph.equilibrate(drive, budget=64, tolerance=1e-14, damping=2)
    kernel = graph._torch
    assert sorted(k.dt for k in kernel._stepped) == [0.25, 0.5]
    made = []
    construct = brain_module._TorchKernel.__init__

    def counted(self, *args, **kwargs):
        made.append(self)
        construct(self, *args, **kwargs)

    monkeypatch.setattr(brain_module._TorchKernel, "__init__", counted)
    for _ in range(3):
        phase = graph.equilibrate(drive, budget=64, tolerance=1e-14, damping=2)
        assert phase.damping_halvings == 2
        assert phase.state.device["holder"] is kernel  # the halved state rests on the live kernel
    assert made == []


def test_a_halved_solve_is_the_solve_of_a_kernel_made_for_its_step():
    graph, drive = graph_on_torch()
    start = graph.settle_batch(drive, steps=3)
    for dt in (0.5, 0.25):
        kept = graph._with_dt(dt)
        fresh = cd.NeuralGraph(
            graph.connectome, graph.neuron_model.replace(dt=dt), backend="torch", device="cpu",
            efficacy=graph.efficacy, bias=graph.bias,
        )
        ours = kept.settle_batch(drive, steps=7, state=start)
        theirs = fresh.settle_batch(drive, steps=7, state=start)
        for name in ("v", "activation", "adaptation", "activity_change"):
            np.testing.assert_array_equal(getattr(ours, name), getattr(theirs, name))
        np.testing.assert_array_equal(kept.residual(drive, ours), fresh.residual(drive, theirs))


def test_a_kept_kernel_takes_the_live_parameters():
    graph, drive = graph_on_torch()
    stepped = graph._with_dt(0.5)._torch
    moved = graph._torch.scale * 1.01
    changed = graph._with_device_parameters(moved, graph._torch.bias_param + 0.01)
    again = changed._with_dt(0.5)._torch
    assert again is stepped
    assert again._version == changed._torch._version
    assert again.scale is changed._torch.scale  # the device's parameters, with no copy
    np.testing.assert_array_equal(again.bias.cpu().numpy(), changed._torch.bias.cpu().numpy())
    fresh = cd.NeuralGraph(
        graph.connectome, graph.neuron_model.replace(dt=0.5), backend="torch", device="cpu",
        efficacy=changed.efficacy, bias=changed.bias,
    )
    np.testing.assert_array_equal(
        changed._with_dt(0.5).settle_batch(drive, steps=5).v,
        fresh.settle_batch(drive, steps=5).v,
    )


def test_a_lesson_whose_phases_halve_updates_on_the_device():
    """Every phase of these lessons halves its step; their states rest on the live kernel, so the
    contrast and the optimizer's history stay on the device, as in a lesson that does not halve."""
    brain = cd.Brain.compose(
        12, 5, modules=(16,), seed=3, backend="torch", device="cpu", learning_qualified=True,
        learning_surprise=False, learning_tolerance=1e-3, learning_free_steps=24,
        learning_nudged_steps=24,
    )
    x, labels = np.eye(12)[[1, 7, 3]], np.array([0, 1, 2])
    kernel = brain.brain._torch
    for _ in range(2):
        brain.lesson(x, labels)
        for phase in ("free", "nudged", "opposite"):
            assert brain.last_learning[f"demonstration_{phase}_damping_halvings"] > 0
        assert brain.learner.__dict__.get("_device_moments") is not None
    assert brain.brain._torch is kernel


def test_the_kernels_hold_no_reference_cycle():
    graph, drive = graph_on_torch()
    gc.collect()
    gc.disable()
    try:
        phase = graph.equilibrate(drive, budget=64, tolerance=1e-14, damping=2)
        kernels = [weakref.ref(graph._torch)] + [
            weakref.ref(k) for k in graph._torch._stepped.values()
        ]
        del graph, phase
        assert all(kernel() is None for kernel in kernels)
    finally:
        gc.enable()


def test_a_settles_movement_stays_on_the_device_until_read(monkeypatch):
    graph, drive = graph_on_torch()
    copies = []
    host_copy = brain_module._host_copy

    def counted(x):
        copies.append(tuple(x.shape))
        return host_copy(x)

    monkeypatch.setattr(brain_module, "_host_copy", counted)
    state = graph.settle_batch(drive, steps=6)
    assert copies == []
    change = state.activity_change
    assert copies == [(3,)]
    reference = cd.NeuralGraph(
        graph.connectome, graph.neuron_model, efficacy=graph.efficacy, bias=graph.bias
    ).settle_batch(drive, steps=6)
    np.testing.assert_allclose(change, reference.activity_change, rtol=1e-12, atol=1e-12)
