"""Continuing device solves reuse bounded storage without freezing numeric inputs."""

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tiled", [True, False])
def test_changing_temperature_reuses_buffers_and_matches_the_host(device, tiled):
    anatomy = cd.layered(2, 4, 4, seed=4)
    model = cd.learning_neuron_model()
    host = cd.NeuralGraph(anatomy, model)
    dev = cd.NeuralGraph(anatomy, model, backend="torch", device=device)
    n = anatomy.n
    mask = np.zeros(n)
    selected = np.array([n - 4, n - 3, n - 2, n - 1]) if tiled else np.array([0, 2, n - 2, n - 1])
    mask[selected] = 1.0
    groups = np.zeros(n, dtype=int)
    groups[selected[2:]] = 1
    rng = np.random.default_rng(8)
    drive, target = rng.normal(size=(3, n)), rng.normal(size=(3, n))
    for k in range(130):
        temperatures = np.full(n, 0.11 + k * 0.001)
        temperatures[selected[2:]] = 0.27 + k * 0.002
        nudge = cd.Nudge(target, mask, 0.2, groups=groups, softmax_temperature=temperatures)
        actual = dev.settle_batch(drive, steps=3, nudge=nudge)
        if k in (0, 64, 129):
            expected = host.settle_batch(drive, steps=3, nudge=nudge)
            np.testing.assert_allclose(actual.activation, expected.activation,
                                       atol=3e-6 if device == "mps" else 1e-12)
    ws = dev._torch._workspaces.held[3]
    assert len(ws.nudges) == 1 and len(ws.configs) == 1
    assert len(dev._torch._nudge_specs) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_tiny_temperature_branch_changes_remain_numerically_correct(device):
    anatomy = cd.layered(2, 4, 3, seed=1)
    host = cd.NeuralGraph(anatomy, cd.learning_neuron_model())
    dev = cd.NeuralGraph(anatomy, host.neuron_model, backend="torch", device=device)
    n = anatomy.n
    mask, target = np.zeros(n), np.zeros((2, n))
    mask[-3:], target[:, -3] = 1.0, 1.0
    drive = np.random.default_rng(1).uniform(size=(2, n))
    for temperature in (0.25, 1e-40, 1e-310, 5e-324, 0.3, 1e-320):
        nudge = cd.Nudge(target, mask, 0.1, softmax_temperature=temperature)
        a = host.settle_batch(drive, steps=3, nudge=nudge)
        b = dev.settle_batch(drive, steps=3, nudge=nudge)
        np.testing.assert_allclose(a.activation, b.activation,
                                   atol=3e-6 if device == "mps" else 1e-12)


def test_distinct_topologies_and_tolerances_have_bounded_capture_caches():
    anatomy = cd.layered(4, 8, 4, seed=5)
    graph = cd.NeuralGraph(anatomy, cd.learning_neuron_model(), backend="torch", device="cpu")
    for k in range(100):
        mask = np.zeros(anatomy.n)
        mask[:8] = [(k >> bit) & 1 for bit in range(8)]
        nudge = cd.Nudge(np.ones((2, anatomy.n)), mask, 0.1, softmax_temperature=0.2)
        graph.settle_batch(np.zeros((2, anatomy.n)), steps=1, nudge=nudge,
                           tolerance=1e-5 + 1e-8 * k)
        ws = graph._torch._workspaces.held[2]
        assert len(ws.nudges) <= 16 and len(ws.configs) <= 32
        assert len(ws.seen) <= 32 and len(ws.graphs) <= 24


@pytest.mark.parametrize("device", DEVICES)
def test_cpu_and_mps_charge_every_executed_tolerance_sweep(device, monkeypatch):
    anatomy = cd.layered(3, 8, 3, seed=8)
    graph = cd.NeuralGraph(anatomy, cd.learning_neuron_model(), backend="torch", device=device)
    sweep, calls = graph._torch._sweep, []

    def counted(*args):
        calls.append(1)
        return sweep(*args)

    monkeypatch.setattr(graph._torch, "_sweep", counted)
    state = graph.settle_batch(np.zeros((2, anatomy.n)), steps=20, tolerance=1e-5)
    assert state.steps == state.attempted_steps == len(calls) == 1


def test_changing_heat_live_reuses_the_same_temperature_buffers():
    brain = cd.Brain.compose(1, 2, modules=(8,), seed=9, arousal=True,
                             arousal_youth=1000, arousal_need=1.0,
                             backend="torch", device="cpu", recall=False)
    for k in range(60):
        brain.live([[1.0]], reward=None if k == 0 else [0.1 + 0.002 * k])
    ws = brain.brain._torch._workspaces.held[1]
    assert len(ws.nudges) <= 2 and len(ws.configs) <= 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_captured_temperature_inputs_stay_live_and_charge_lookahead():
    anatomy = cd.layered(2, 6, 3, seed=3)
    graph = cd.NeuralGraph(anatomy, cd.learning_neuron_model(), backend="torch", device="cuda")
    mask, target = np.zeros(anatomy.n), np.zeros((2, anatomy.n))
    mask[-3:], target[:, -3] = 1.0, 1.0
    drive = np.random.default_rng(1).uniform(size=(2, anatomy.n))
    for temperature in (0.3, 0.2, 0.15, 0.1, 0.4):
        nudge = cd.Nudge(target, mask, 0.2, softmax_temperature=temperature)
        actual = graph.settle_batch(drive, steps=20, tolerance=1e-5, nudge=nudge)
        reference = graph.settle_batch(drive, steps=20, tolerance=1e-5, nudge=nudge,
                                       trajectory=True)
        np.testing.assert_array_equal(actual.activation, reference.activation)
        assert actual.attempted_steps == actual.steps + int(actual.steps < 20)
        assert reference.attempted_steps == reference.steps
    assert len(graph._torch._workspaces.held[2].nudges) == 1
