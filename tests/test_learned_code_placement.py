"""A learned code's placement is whole or not at all. A move to a torch device, or back to the
host, that fails while the sub-network of the new place is made leaves the region where it was:
its device, the graph at the model's step, the graph of its present step and the lateral weights
its step's tests read on the device. The region then reads its step, settles and learns as one
that never tried, and the same move completes when it is asked again. Read on a torch CPU device
and on CUDA where there is one."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd
from cadence import code as code_module
from cadence.brain import _TorchKernel

torch = pytest.importorskip("torch")

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]
TIGHT = 1e-9  # float64 on both sides
INPUTS, ACTIONS = 12, 5
OPTIONS = {"modules": (32, 16), "learned_code": 24, "learning_fast_settle": False}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def host(seed: int = 3) -> cd.Brain:
    return cd.Brain.compose(INPUTS, ACTIONS, seed=seed, backend="cpu", **OPTIONS)


def observations(rng: np.random.Generator, rows: int) -> np.ndarray:
    return np.eye(INPUTS)[rng.integers(INPUTS, size=rows)] + 0.01 * rng.random((rows, INPUTS))


def held(region) -> tuple:
    """What a placement moves: the device, the two graphs and the weights the step's tests read."""
    return region.device, region._base, region.graph, region._coupled


def lives_as(region, twin, atol: float) -> None:
    """The step's test, a settle and a learning step of ``region`` read as ``twin``'s."""
    rng = np.random.default_rng(11)
    potentials = rng.normal(region.model.threshold, 0.4, (3, region.units))
    assert region.stability(potentials) == twin.stability(potentials)
    sensory, context = rng.random((3, region.inputs)), rng.random((3, region.context))
    budget = 512 << region.halvings
    phases = [
        one.settle(one.stimulus(sensory, context), None, budget=budget, tolerance=1e-9)
        for one in (twin, region)
    ]
    assert all(np.all(phase.qualified) for phase in phases)
    np.testing.assert_allclose(
        phases[1].state.activation, phases[0].state.activation, rtol=0, atol=atol
    )
    for one, phase in zip((twin, region), phases, strict=True):
        assert one.learn(sensory, np.asarray(phase.state.activation), context)
    for name in ("dictionary", "weights", "thresholds"):
        np.testing.assert_allclose(
            getattr(region, name), getattr(twin, name), rtol=0, atol=atol, err_msg=name
        )
    np.testing.assert_array_equal(region.graph.efficacy, region.lateral_efficacy())
    np.testing.assert_array_equal(region.graph.bias, -region.thresholds)


@pytest.mark.parametrize("device", DEVICES)
def test_a_move_to_a_device_that_fails_leaves_the_region_on_the_host(device, monkeypatch):
    """The kernel of the model's step is made, then the kernel of the halved step fails."""
    region, twin = host().learned_code, host().learned_code
    region.restep(1)
    twin.restep(1)
    before = held(region)
    stepped = _TorchKernel.stepped

    def fails(self, neuron_model):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(_TorchKernel, "stepped", fails)
    with pytest.raises(RuntimeError, match="out of memory"):
        region.place(device)
    monkeypatch.setattr(_TorchKernel, "stepped", stepped)
    assert held(region) == before
    assert region.device is None and region._base is None and region.graph.backend == "cpu"
    assert region.graph.neuron_model.dt == twin.graph.neuron_model.dt
    lives_as(region, twin, atol=0.0)  # the host's own arithmetic on both
    region.place(device)  # the same move, asked again
    assert region.device == torch.device(device) and region._base.backend == "torch"
    assert region.graph.backend == "torch" and region.graph._torch.root is region._base._torch
    assert region.graph.neuron_model.dt == twin.graph.neuron_model.dt
    lives_as(region, twin, atol=TIGHT)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("halvings", [0, 2])
def test_a_move_to_the_host_that_fails_leaves_the_region_on_its_device(
    device, halvings, monkeypatch
):
    region, twin = host().learned_code, host().learned_code
    for one in (region, twin):
        one.place(device)
        one.restep(halvings)
    rng = np.random.default_rng(4)
    potentials = rng.normal(region.model.threshold, 0.4, (2, region.units))
    region.stability(potentials)  # the lateral weights of the step's tests rest on the device
    before = held(region)
    assert before[3] is not None
    graph = code_module.NeuralGraph

    def fails(*args, **kwargs):
        raise MemoryError("the host's graph could not be made")

    monkeypatch.setattr(code_module, "NeuralGraph", fails)
    with pytest.raises(MemoryError, match="could not be made"):
        region.place(None)
    monkeypatch.setattr(code_module, "NeuralGraph", graph)
    assert held(region) == before
    assert region.device == torch.device(device) and region.graph.backend == "torch"
    assert region.graph._torch.root is region._base._torch
    lives_as(region, twin, atol=TIGHT)
    region.place(None)  # the same move, asked again
    assert region.device is None and region._base is None and region.graph.backend == "cpu"
    assert region.graph.neuron_model.dt == twin.graph.neuron_model.dt
    lives_as(region, twin, atol=TIGHT)


@pytest.mark.parametrize("device", DEVICES)
def test_a_brain_lives_on_after_a_placement_that_failed(device, monkeypatch):
    """Lessons and acts after the failed move are the ones of a brain that never tried."""
    reference, brain = host(), host()
    rng = np.random.default_rng(0)

    def taught(moments: int) -> None:
        for _ in range(moments):
            x, labels = observations(rng, 2), rng.integers(ACTIONS, size=2)
            for one in (reference, brain):
                one.lesson(x, labels)
            np.testing.assert_array_equal(
                brain.act(x, greedy=True), reference.act(x, greedy=True)
            )

    taught(3)
    region = brain.learned_code
    region.restep(max(1, region.halvings))
    reference.learned_code.restep(region.halvings)
    before = held(region)

    def fails(self, neuron_model):
        raise RuntimeError("CUDA out of memory")

    with monkeypatch.context() as patched:
        patched.setattr(_TorchKernel, "stepped", fails)
        with pytest.raises(RuntimeError, match="out of memory"):
            region.place(device)
    assert held(region) == before
    taught(3)
    for name in ("dictionary", "weights", "thresholds"):
        np.testing.assert_array_equal(
            getattr(region, name), getattr(reference.learned_code, name), err_msg=name
        )
    np.testing.assert_array_equal(brain.brain.efficacy, reference.brain.efficacy)
    assert region.work == reference.learned_code.work
