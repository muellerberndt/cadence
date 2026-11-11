"""Audit 2026-09-15, class 3 and 9: settling, residuals, tolerances, precision, device state."""

import numpy as np
import pytest

import cadence as cd
from cadence.brain import Nudge

torch = pytest.importorskip("torch")


def _mps() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def _problem() -> tuple[cd.Connectome, np.ndarray, Nudge, np.ndarray]:
    rng = np.random.default_rng(0)
    c, _ = cd.embedded(12, 3, 6, 24, 7, seed=0)
    d = np.zeros((8, c.n))
    for p in range(3):
        d[np.arange(8), p * 12 + rng.integers(0, 12, 8)] = 1.0
    out = np.asarray(c.populations["output"])
    mask = np.zeros(c.n)
    mask[out] = 1.0
    groups = np.full(c.n, -1)
    groups[out[:3]] = 0
    groups[out[3:]] = 1  # unequal slots: three and four choices
    target = np.zeros((8, c.n))
    target[np.arange(8), out[rng.integers(0, 3, 8)]] = 1.0
    target[np.arange(8), out[3 + rng.integers(0, 4, 8)]] = 1.0
    nudge = Nudge(
        target, mask, 0.05, softmax_temperature=0.2, weight=rng.normal(size=8), groups=groups
    )
    keep = np.ones((8, c.n))
    keep[0, out[0]] = 0.0
    keep[3, 40] = 0.5
    return c, d, nudge, keep


def _brains(c: cd.Connectome, model: cd.NeuronModel) -> dict[str, cd.NeuralGraph]:
    brains = {
        "cpu": cd.NeuralGraph(c, model),
        "cpu-unblocked": cd.NeuralGraph(c, model, dense_limit=1),
        "torch-cpu": cd.NeuralGraph(c, model, backend="torch", device="cpu"),
        "torch-cpu-scatter": cd.NeuralGraph(c, model, backend="torch", device="cpu", dense_limit=1),
    }
    if _mps():
        brains["torch-mps"] = cd.NeuralGraph(c, model, backend="torch", device="mps")
    if "mlx" in cd.available_backends():
        brains["mlx"] = cd.NeuralGraph(c, model, backend="mlx")
    return brains


@pytest.mark.parametrize("adaptation", [None, cd.Adaptation(tau_steps=5.0, strength=0.3)])
def test_residual_on_device_matches_host_residual_of_the_same_state(
    adaptation: cd.Adaptation | None,
) -> None:
    c, d, nudge, keep = _problem()
    model = cd.learning_neuron_model(dt=0.5).replace(adaptation=adaptation)
    host = cd.NeuralGraph(c, model)
    for name, brain in _brains(c, model).items():
        for nu, mk in ((None, None), (nudge, keep)):
            state = brain.settle_batch(d, steps=120, nudge=nu, mask=mk, tolerance=1e-6)
            on_device = brain.residual(d, state, nudge=nu, mask=mk)
            copy = cd.BrainState(
                np.asarray(state.v), np.asarray(state.activation), np.asarray(state.adaptation), 1
            )
            on_host = host.residual(d, copy, nudge=nu, mask=mk)
            tol = 1e-6 if "mps" in name or name == "mlx" else 1e-12
            assert np.allclose(on_device, on_host, atol=tol, rtol=1e-3), (name, nu is None)
            assert on_device.max() < 1e-4, name  # a movement tolerance of 1e-6 got this close
            # the same state settled with the same tolerance takes the same number of steps
            steps = host.settle_batch(d, steps=120, nudge=nu, mask=mk, tolerance=1e-6).steps
            assert abs(state.steps - steps) <= (2 if tol > 1e-9 else 0), name


def test_equilibrate_residual_is_brain_residual_on_every_backend() -> None:
    c, d, nudge, keep = _problem()
    model = cd.learning_neuron_model(dt=0.5)
    for name, brain in _brains(c, model).items():
        eq = brain.equilibrate(d, budget=400, chunk=32, tolerance=1e-6, nudge=nudge, mask=keep)
        again = brain.residual(d, eq.state, nudge=nudge, mask=keep)
        assert np.array_equal(eq.residual, again), name
        assert eq.converged.all(), name
        assert eq.steps % 32 == 0 and eq.steps <= 400


def test_stop_step_agrees_between_fused_numpy_and_torch_kernels() -> None:
    c, d, nudge, keep = _problem()
    model = cd.learning_neuron_model(dt=1.0)
    cpu = cd.NeuralGraph(c, model)
    dev = cd.NeuralGraph(c, model, backend="torch", device="cpu")
    for tolerance in (1e-3, 1e-5, 1e-7):
        fused = cpu.settle_batch(d, steps=500, tolerance=tolerance)
        loop = cpu.settle_batch(d, steps=500, tolerance=tolerance, trajectory=True)
        kernel = dev.settle_batch(d, steps=500, tolerance=tolerance)
        assert fused.steps == loop.steps == kernel.steps, tolerance
        assert np.allclose(fused.activation, np.asarray(kernel.activation), atol=1e-12)
        assert loop.trajectory is not None and len(loop.trajectory) == loop.steps


def test_float32_precision_on_mps_is_named_and_float64_is_refused() -> None:
    c, _, _, _ = _problem()
    if not _mps():
        pytest.skip("no MPS device")
    with pytest.raises(ValueError, match="float64"):
        cd.NeuralGraph(c, cd.learning_neuron_model(), backend="torch", device="mps", precision="float64")
    brain = cd.NeuralGraph(c, cd.learning_neuron_model(), backend="torch", device="mps")
    assert brain._torch is not None and brain._torch.dtype == torch.float32


def test_float32_equilibrate_does_not_pretend_to_reach_float64_tolerance() -> None:
    c, d, _, _ = _problem()
    brain = cd.NeuralGraph(c, cd.learning_neuron_model(), backend="torch", device="cpu", precision="float32")
    eq = brain.equilibrate(d, budget=256, chunk=32, tolerance=1e-10)
    assert not eq.converged.any() and eq.steps == 256  # the budget is spent, the flag says so


def test_empty_connectome_settles_on_every_backend() -> None:
    e = cd.Connectome.from_synapses(5, pre=[], post=[], populations={"a": range(2), "b": range(2, 5)})
    model = cd.learning_neuron_model()
    for kw in ({}, {"backend": "torch", "device": "cpu"}):
        brain = cd.NeuralGraph(e, model, **kw)
        state = brain.settle_batch(np.ones((2, 5)), steps=50, tolerance=1e-9)
        assert np.allclose(np.asarray(state.activation), model.activation(np.ones((2, 5))))
        assert brain.residual(np.ones((2, 5)), state).max() < 1e-8
        learner = cd.Learner(brain, [3, 4])
        learner.step(np.ones((2, 5)), np.array([0, 1]))
        assert learner.parameters() == 5


def test_batched_and_actor_batch_checks_read_the_device_shape_without_a_fetch() -> None:
    c, d, _, _ = _problem()
    brain = cd.NeuralGraph(c, cd.learning_neuron_model(), backend="torch", device="cpu")
    state = brain.settle_batch(d, steps=5)
    assert state.batched
    assert state.__dict__.get("v") is None  # the shape came from the device tensor
    learner = cd.Learner(brain, c.populations["output"], slots=1)
    agent = cd.ActorCritic(learner, c.populations["hidden"])
    agent.act(d)
    first = agent.state
    assert first is not None
    agent.act(d * 0.5)  # a fresh drive: settle() compares the batch size
    assert first.__dict__.get("v") is None


def test_done_rows_reset_on_the_device_and_match_the_host() -> None:
    c, d, _, _ = _problem()
    model = cd.learning_neuron_model(dt=1.0)
    config = cd.LearnerConfig(tolerance=1e-6, free_steps=200, nudged_steps=100)
    reward = cd.ActorCriticConfig(gamma=0.9, lam=0.8, eta=0.3)
    host = cd.ActorCritic(cd.Learner(cd.NeuralGraph(c, model), c.populations["output"], config), c.populations["hidden"], reward)
    dev = cd.ActorCritic(
        cd.Learner(cd.NeuralGraph(c, model, backend="torch", device="cpu"), c.populations["output"], config),
        c.populations["hidden"],
        reward,
    )
    done = np.array([True, False, True, False, False, False, False, True])
    rng = np.random.default_rng(3)
    next_drive = d[rng.permutation(8)]
    cold = host.learner.free(next_drive[done])  # before any parameter moves
    for agent in (host, dev):
        agent.act(d)
        agent.learn(np.ones(8), done, next_drive)
    assert dev.state is not None and dev.state.device is not None
    assert dev.state.device["holder"] is dev.learner.brain._torch  # never left the device
    assert host.state is not None
    assert np.allclose(np.asarray(dev.state.v), host.state.v, atol=1e-10)
    assert np.allclose(host.state.v[done], cold.v, atol=1e-10)  # a finished row starts from rest
