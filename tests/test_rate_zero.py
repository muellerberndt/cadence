"""A synapse rate of zero holds the synapse, its optimizer moments and, where every inbound
synapse of a neuron steps at zero, that neuron's bias (``Learner.held``)."""

import warnings

import numpy as np
import pytest

import cadence as cd


def held_motor(backend="cpu", **place):
    """A composed brain whose motor neurons step at rate zero on every synapse they touch."""
    brain = cd.Brain.compose(6, 3, modules=(12,), seed=1, backend=backend, recall=False,
                             teaching="deep", **place)
    w = brain.connectome
    motor = np.zeros(w.n, dtype=bool)
    motor[np.asarray(w.populations["motor"])] = True
    rate = np.ones(w.synapses)
    rate[motor[w.post] | motor[w.pre]] = 0.0
    brain.learner.synapse_rate = rate
    return brain, rate, motor


def teach(brain, lessons=3):
    """Lessons, and sampled acts paid by reward: both reach the held synapses' step."""
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(lessons):
            brain.lesson(rng.random((4, 6)), rng.integers(3, size=4))
            brain.step(rng.random((4, 6)))
            brain.step(rng.random((4, 6)), reward=rng.random(4) - 0.5)
            brain.learn(rng.random(4) - 0.5, np.zeros(4, dtype=bool), rng.random((4, 6)))


def check_held(brain, rate, motor, efficacy, bias):
    learner = brain.learner
    moved_efficacy = np.abs(np.asarray(brain.brain.efficacy, dtype=float) - efficacy)
    moved_bias = np.abs(np.asarray(brain.brain.bias, dtype=float) - bias)
    assert moved_efficacy[rate == 0].max() == 0.0
    assert moved_bias[motor].max() == 0.0
    assert moved_efficacy[rate > 0].max() > 0 and moved_bias[~motor].max() > 0
    for name in ("velocity", "second_moment"):
        assert np.abs(getattr(learner, name)[rate == 0]).max() == 0.0
        assert np.abs(getattr(learner, name)[rate > 0]).max() > 0
    for name in ("velocity_bias", "second_moment_bias"):
        assert np.abs(getattr(learner, name)[motor]).max() == 0.0


def test_held_names_the_synapses_and_the_neurons_they_stop():
    brain, rate, motor = held_motor()
    synapses, neurons = brain.learner.held()
    assert np.array_equal(synapses, rate == 0) and np.array_equal(neurons, motor)
    brain.learner.synapse_rate = np.where(rate == 0, 0.5, 1.0)
    assert brain.learner.held() == (None, None)
    brain.learner.synapse_rate = None
    assert brain.learner.held() == (None, None)


def test_a_neuron_with_one_moving_inbound_synapse_steps_its_bias():
    brain, rate, motor = held_motor()
    w = brain.connectome
    first = int(np.flatnonzero(motor)[0])
    inbound = np.flatnonzero(w.post == first)
    rate = rate.copy()
    rate[inbound[0]] = 1.0
    brain.learner.synapse_rate = rate
    neurons = brain.learner.held()[1]
    assert not neurons[first] and neurons[motor].sum() == motor.sum() - 1


def test_rate_zero_holds_synapses_biases_and_moments_on_the_cpu():
    brain, rate, motor = held_motor()
    efficacy, bias = np.array(brain.brain.efficacy), np.array(brain.brain.bias)
    teach(brain)
    check_held(brain, rate, motor, efficacy, bias)


def test_rate_zero_holds_on_the_numpy_path(monkeypatch):
    from cadence import learning

    monkeypatch.setattr(learning, "_compiled_passes", lambda: False)
    brain, rate, motor = held_motor()
    efficacy, bias = np.array(brain.brain.efficacy), np.array(brain.brain.bias)
    teach(brain)
    check_held(brain, rate, motor, efficacy, bias)


def test_rate_zero_holds_on_torch():
    pytest.importorskip("torch")
    brain, rate, motor = held_motor("torch", device="cpu", precision="float64")
    efficacy, bias = np.array(brain.brain.efficacy), np.array(brain.brain.bias)
    teach(brain)
    check_held(brain, rate, motor, efficacy, bias)


def test_rate_zero_holds_on_mlx():
    pytest.importorskip("mlx.core")
    brain, rate, motor = held_motor("mlx")
    efficacy, bias = np.array(brain.brain.efficacy), np.array(brain.brain.bias)
    teach(brain)
    check_held(brain, rate, motor, efficacy, bias)


def test_rate_zero_holds_in_lives_on_torch():
    """``Lives`` on a torch device keep each life's held synapses, biases and moments."""
    pytest.importorskip("torch")
    lives = cd.Lives(6, 3, 2, modules=(12,), backend="torch", device="cpu", precision="float64",
                     recall=False, teaching="deep")
    learner, w = lives.brain.learner, lives.brain.connectome
    motor = np.zeros(w.n, dtype=bool)
    motor[np.asarray(w.populations["motor"])] = True
    rate = np.ones(w.synapses)
    rate[motor[w.post] | motor[w.pre]] = 0.0
    learner.synapse_rate = rate
    before = [lives.parameters(i) for i in range(2)]
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            lives.lesson(rng.random((2, 6)), rng.integers(3, size=2))
            lives.step(rng.random((2, 6)))
            lives.learn(rng.random(2) - 0.5, np.zeros(2, dtype=bool), rng.random((2, 6)))
    for i in range(2):
        after = lives.parameters(i)
        assert np.abs(after["efficacy"] - before[i]["efficacy"])[rate == 0].max() == 0.0
        assert np.abs(after["efficacy"] - before[i]["efficacy"])[rate > 0].max() > 0
        assert np.abs(after["bias"] - before[i]["bias"])[motor].max() == 0.0
        for name in ("velocity", "second_moment"):
            assert np.abs(after[name][rate == 0]).max() == 0.0
            assert np.abs(after[name][rate > 0]).max() > 0
        for name in ("velocity_bias", "second_moment_bias"):
            assert np.abs(after[name][motor]).max() == 0.0


def test_rates_above_zero_teach_as_before():
    """Without a rate of zero nothing is held, and lessons teach to the bit as with no rate."""
    plain, _, _ = held_motor()
    rated, _, _ = held_motor()
    plain.learner.synapse_rate = None
    rated.learner.synapse_rate = np.ones(rated.connectome.synapses)
    teach(plain)
    teach(rated)
    assert np.array_equal(plain.brain.efficacy, rated.brain.efficacy)
    assert np.array_equal(plain.brain.bias, rated.brain.bias)
    assert np.array_equal(plain.learner.second_moment, rated.learner.second_moment)
