"""Audit 2026-09-15, class 1: the rectified neuron is exactly zero at rest.

By default, every builder starts the biases at zero. Under a sign-symmetric random projection about
half of the free hidden neurons then sit at or below rest, publish at most ``-leak``, and
carry an order of magnitude less contrast than the active half. The strict xfail below
records the defect: it turns into a failure the day a builder or ``NeuralGraph`` handles it, so
the test is then updated rather than forgotten.
"""

import numpy as np
import pytest

import cadence as cd


def _drive_embedded(connectome: cd.Connectome, rng: np.random.Generator, batch: int) -> np.ndarray:
    d = np.zeros((batch, connectome.n))
    for p in range(3):
        d[np.arange(batch), p * 12 + rng.integers(0, 12, batch)] = 1.0
    return d


def _silent_fraction(brain: cd.NeuralGraph, drive: np.ndarray, members: list[int]) -> float:
    state = brain.settle_batch(drive, steps=300, tolerance=1e-6)
    return float((np.asarray(state.activation)[:, members] <= 0.0).mean())


def _builders() -> list[tuple[str, cd.NeuralGraph, np.ndarray, list[int]]]:
    rng = np.random.default_rng(0)
    out = []
    c = cd.layered(20, 40, 5, seed=0)
    d = np.zeros((16, c.n))
    d[:, :20] = rng.random((16, 20))
    out.append(("layered", cd.NeuralGraph(c, cd.learning_neuron_model()), d, list(c.populations["hidden"])))
    c, _ = cd.embedded(12, 3, 6, 24, 5, seed=0)
    brain = cd.NeuralGraph(c, cd.learning_neuron_model(dt=1.0))
    out.append(("embedded", brain, _drive_embedded(c, rng, 16), list(c.populations["hidden"])))
    g = cd.Brain.compose(10, 4, modules=(64,), seed=0, recall=False)
    drive = g.stimulus(rng.random((16, 10)), memory=False)
    out.append(("generic", g.brain, drive, list(g.association_index)))
    g = cd.Brain.compose((8, 8), 4, modules=(64,), seed=0, recall=False)
    drive = g.stimulus(rng.random((16, 8, 8)), memory=False)
    out.append(("generic-image", g.brain, drive, list(g.connectome.populations["visual/output"])))
    return out


def test_every_builder_starts_with_zero_bias() -> None:
    for name, brain, _, _ in _builders():
        assert not brain.bias.any(), name


@pytest.mark.xfail(
    reason="builders start biases at zero: about half of the free hidden neurons sit at or "
    "below rest and publish nothing (audit 2026-09-15, open)",
    strict=True,
)
def test_free_hidden_neurons_are_mostly_responsive() -> None:
    for name, brain, drive, members in _builders():
        assert _silent_fraction(brain, drive, members) < 0.25, name


def test_silent_hidden_neurons_carry_little_contrast_and_a_resting_bias_repairs_it() -> None:
    rng = np.random.default_rng(0)
    c = cd.layered(20, 40, 5, seed=0)
    d = np.zeros((16, c.n))
    d[:, :20] = rng.random((16, 20))
    labels = rng.integers(0, 5, 16)
    hidden = np.asarray(c.populations["hidden"])
    config = cd.LearnerConfig(tolerance=1e-6, free_steps=300, nudged_steps=300)

    def contrast(bias: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        brain = cd.NeuralGraph(c, cd.learning_neuron_model(), bias=np.full(c.n, bias))
        learner = cd.Learner(brain, c.populations["output"], config)
        target = learner.targets(labels)
        free = learner.free(d)
        plus = learner.nudged(d, free, target)
        minus = learner.nudged(d, free, target, sign=-1.0)
        _, neurons = learner.contrast(free, plus, minus)
        return free.activation[:, hidden], np.abs(neurons[hidden]), np.asarray(free.v)[:, hidden]

    activation, contrast_zero, _ = contrast(0.0)
    silent = activation.max(axis=0) <= 0.0
    assert 0.2 < silent.mean() < 0.65  # silent in every row; about half are silent per row
    assert contrast_zero[silent].mean() < 0.25 * contrast_zero[~silent].mean()
    activation_biased, contrast_biased, _ = contrast(0.5)
    assert (activation_biased.max(axis=0) <= 0.0).mean() < 0.2
    assert contrast_biased.mean() > 2.0 * contrast_zero.mean()
