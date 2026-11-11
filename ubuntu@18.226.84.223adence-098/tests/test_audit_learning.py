"""Audit 2026-09-15, classes 2, 9 and 10: the update, its history, bounds, and its cost."""

import time

import numpy as np
import pytest

import cadence as cd
from cadence.learning import SCALE_CAP


def _net(seed: int = 0) -> tuple[cd.Connectome, np.ndarray]:
    return cd.embedded(12, 3, 6, 24, 5, seed=seed)


def _drive(c: cd.Connectome, rng: np.random.Generator, batch: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.zeros((batch, c.n))
    for p in range(3):
        d[np.arange(batch), p * 12 + rng.integers(0, 12, batch)] = 1.0
    return d, rng.integers(0, 5, batch)


def _reference_parameters(learner: cd.Learner) -> int:
    """The count as ``np.unique`` over the keys gave it before the audit."""
    assert learner.plastic_synapses is not None and learner.plastic_neurons is not None
    keys = np.arange(learner.brain.connectome.synapses)
    paired = learner.reverse >= 0
    keys[paired] = np.minimum(keys[paired], learner.reverse[paired])
    if len(learner._members):
        keys[learner._members] = learner.brain.connectome.synapses + learner._member_groups
    return len(np.unique(keys[learner.plastic_synapses])) + int(learner.plastic_neurons.sum())


def test_parameters_counts_pairs_ties_and_masks_like_the_sorted_reference() -> None:
    rng = np.random.default_rng(1)
    c, tie = _net()
    for reciprocal in (True, False):
        for use_tie in (True, False):
            mask = rng.random(c.synapses) < 0.7
            neurons = rng.random(c.n) < 0.5
            learner = cd.Learner(
                cd.NeuralGraph(c, cd.learning_neuron_model()),
                c.populations["output"],
                plastic_synapses=mask,
                plastic_neurons=neurons,
                reciprocal=reciprocal,
                tie_groups=tie if use_tie else None,
            )
            assert learner.parameters() == _reference_parameters(learner)


def test_parameters_is_linear_in_the_synapses() -> None:
    c = cd.layered(1200, 2000, 8, density=1.0, seed=0)  # 2.4 million synapses
    learner = cd.Learner(cd.NeuralGraph(c, cd.learning_neuron_model(), dense_limit=1), c.populations["output"])
    t0 = time.perf_counter()
    count = learner.parameters()
    elapsed = time.perf_counter() - t0
    assert count == _reference_parameters(learner)
    assert elapsed < 0.6, elapsed  # the sort took about a second here; 68M synapses took 23 s


@pytest.mark.parametrize("use_normalize", [False, True])
def test_normalized_steps_move_every_synapse_by_about_eta(use_normalize: bool) -> None:
    """With ``normalize`` the step is ``eta * m / (rms + floor)``: about ``eta`` per synapse
    whatever the contrast's size, so ``eta`` is a fraction of ``SCALE_CAP`` per update."""
    c = cd.layered(20, 30, 5, seed=0)
    rng = np.random.default_rng(0)
    config = cd.LearnerConfig(eta=0.2, normalize=0.9 if use_normalize else 0.0, tolerance=1e-5)
    learner = cd.Learner(
        cd.NeuralGraph(c, cd.learning_neuron_model()), c.populations["output"], config, reciprocal=False
    )
    d = np.zeros((8, c.n))
    d[:, :20] = rng.random((8, 20))
    before = learner.brain.efficacy.copy()
    learned, _ = learner.step(d, rng.integers(0, 5, 8))
    raw, _ = learner.contrast(learned.free, learned.nudged, learned.opposite)
    moved = np.abs(learner.brain.efficacy - before)
    if use_normalize:
        # the first update's bias-corrected RMS is |raw| itself: the step is eta * |raw| / (|raw| + floor)
        expected = config.eta * np.abs(raw) / (np.abs(raw) + config.normalize_floor)
        assert np.allclose(moved, expected, atol=1e-12)
        loud = np.abs(raw) > 10 * config.normalize_floor
        assert loud.any() and (moved[loud] > 0.9 * config.eta).all()  # about eta, whatever the contrast
        assert np.median(moved[np.abs(raw) > 0]) > 5 * np.median(config.eta * np.abs(raw[np.abs(raw) > 0]))
    else:
        assert np.allclose(moved, config.eta * np.abs(raw), atol=1e-12)
        assert np.median(moved) < 0.02


def test_bias_correction_continues_after_a_checkpoint_and_ignores_external_updates(tmp_path) -> None:
    c, tie = _net()
    rng = np.random.default_rng(0)
    config = cd.LearnerConfig(momentum=0.9, normalize=0.99, tolerance=1e-6)

    def make() -> cd.Learner:
        return cd.Learner(cd.NeuralGraph(c, cd.learning_neuron_model(dt=1.0)), c.populations["output"], config, tie_groups=tie)

    steady, resumed = make(), make()
    batches = [_drive(c, rng, 8) for _ in range(6)]
    for d, labels in batches[:3]:
        steady.step(d, labels)
        resumed.step(d, labels)
    # an external update (a reward step) moves the parameters but not the optimizer's history
    external = np.zeros(c.synapses)
    external[::3] = 0.01
    for learner in (steady, resumed):
        learner.apply(external, np.zeros(c.n))
    assert steady.updates == 4 and steady.contrast_updates == 3
    assert np.array_equal(steady.velocity, make().velocity) is False
    path = resumed.save(tmp_path / "resumed")
    resumed = cd.Learner.load(path)
    assert resumed.updates == 4 and resumed.contrast_updates == 3
    for d, labels in batches[3:]:
        steady.step(d, labels)
        resumed.step(d, labels)
    assert np.array_equal(steady.brain.efficacy, resumed.brain.efficacy)
    assert np.array_equal(steady.velocity, resumed.velocity)
    assert np.array_equal(steady.second_moment_bias, resumed.second_moment_bias)


def test_the_cap_never_touches_a_frozen_synapse_on_host_or_device() -> None:
    torch = pytest.importorskip("torch")
    c, tie = _net()
    rng = np.random.default_rng(2)
    config = cd.LearnerConfig(eta=50.0, eta_bias=5.0, tolerance=1e-6)
    frozen = np.zeros(c.synapses, dtype=bool)
    frozen[::4] = True
    frozen_neurons = np.zeros(c.n, dtype=bool)
    frozen_neurons[::3] = True
    start = rng.normal(size=c.synapses) * 3
    start[frozen] = 9.0  # beyond the cap: a frozen synapse keeps whatever it holds
    learners = {}
    for backend in ("cpu", "torch"):
        brain = cd.NeuralGraph(
            c,
            cd.learning_neuron_model(dt=1.0),
            backend=backend,
            device="cpu" if backend == "torch" else None,
            efficacy=start,
        )
        learners[backend] = cd.Learner(
            brain,
            c.populations["output"],
            config,
            plastic_synapses=~frozen,
            plastic_neurons=~frozen_neurons,
            tie_groups=tie,
        )
    capped = {backend: False for backend in learners}
    for _ in range(3):
        d, labels = _drive(c, rng, 8)
        for backend, learner in learners.items():
            learner.step(d, labels)
            efficacy = learner.brain.efficacy
            assert np.array_equal(efficacy[frozen], start[frozen]), backend  # no clip
            assert np.abs(efficacy[~frozen]).max() <= SCALE_CAP, backend
            capped[backend] |= bool((np.abs(efficacy[~frozen]) == SCALE_CAP).any())
            assert not learner.brain.bias[frozen_neurons].any(), backend
    assert all(capped.values())  # eta=50 drives plastic synapses into the cap
    assert np.allclose(learners["cpu"].brain.efficacy, learners["torch"].brain.efficacy, atol=1e-9)
    assert np.allclose(learners["cpu"].brain.bias, learners["torch"].brain.bias, atol=1e-9)
    del torch


def test_host_apply_keeps_the_torch_kernel_and_the_settled_states_on_it() -> None:
    pytest.importorskip("torch")
    c, tie = _net()
    rng = np.random.default_rng(0)
    brain = cd.NeuralGraph(c, cd.learning_neuron_model(dt=1.0), backend="torch", device="cpu")
    learner = cd.Learner(brain, c.populations["output"], cd.LearnerConfig(tolerance=1e-5), tie_groups=tie)
    agent = cd.ActorCritic(learner, c.populations["hidden"], cd.ActorCriticConfig())
    kernel = brain._torch
    d, _ = _drive(c, rng, 4)
    agent.act(d)
    learner._device_kernel = lambda plus, minus: None  # the update takes the host path
    agent.learn(np.ones(4), np.zeros(4, dtype=bool), d)
    assert learner.brain._torch is kernel  # not rebuilt
    assert agent.state is not None and agent.state.device is not None
    assert agent.state.device["holder"] is kernel  # the next free phase still rests there
    # and the parameters the kernel holds are the ones the host computed
    assert np.array_equal(learner.brain.efficacy, kernel.host_scale())


def test_free_phase_reports_the_step_cap_only_through_its_step_count() -> None:
    c, _ = _net()
    rng = np.random.default_rng(0)
    learner = cd.Learner(
        cd.NeuralGraph(c, cd.learning_neuron_model(dt=1.0)),
        c.populations["output"],
        cd.LearnerConfig(free_steps=5, tolerance=1e-9),
    )
    d, labels = _drive(c, rng, 4)
    free = learner.free(d)
    assert free.steps == learner.config.free_steps  # the cap was hit
    assert learner.brain.residual(d, free).max() > 1e-3  # and the state is far from equilibrium
    _, report = learner.step(d, labels)
    assert report["free_steps"] == 5.0
