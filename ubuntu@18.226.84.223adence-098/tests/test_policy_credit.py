"""Reward eligibility differentiates the sampled policy, including unequal slots."""

import numpy as np
import pytest

import cadence as cd


def actor(nudge="cross_entropy", slots=1, backend="cpu"):
    graph = cd.layered(2, 5, 5, density=1.0, seed=4)
    config = cd.LearnerConfig(
        nudge=nudge, beta=0.001, temperature=0.2, free_steps=160, nudged_steps=160, tolerance=1e-9
    )
    brain = cd.NeuralGraph(
        graph, cd.learning_neuron_model(dt=1.0), backend=backend, device="cpu", precision="float64"
    )
    learner = cd.Learner(brain, graph.populations["output"], config, slots=slots)
    ac = cd.ActorCritic(
        learner,
        graph.populations["hidden"],
        cd.ActorCriticConfig(eta=0.05, gamma=0.8, lam=0.9),
        seed=12,
    )
    drive = np.zeros((16, graph.n))
    drive[:, :2] = np.random.default_rng(2).uniform(0.0, 3.0, (16, 2))
    return ac, drive


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_actor_credit_is_independent_of_the_imitation_loss(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    a, drive = actor("quadratic", backend=backend)
    b, _ = actor("cross_entropy", backend=backend)
    np.testing.assert_array_equal(a.act(drive), b.act(drive))
    for index in (1, 2):
        np.testing.assert_allclose(
            a._pending[index].activation, b._pending[index].activation, atol=1e-12
        )
    reward = np.linspace(-0.5, 0.5, len(drive))
    done = np.ones(len(drive), bool)
    a.learn(reward, done, drive)
    b.learn(reward, done, drive)
    np.testing.assert_allclose(a.learner.brain.efficacy, b.learner.brain.efficacy, atol=1e-12)
    assert a.learner.config.nudge == "quadratic"  # imitation configuration is retained


def test_unequal_categorical_slots_never_sample_padding_and_learn_together():
    ac, drive = actor(slots=[2, 3])
    chosen = ac.act(drive)
    assert chosen.shape == (len(drive), 2) and np.issubdtype(chosen.dtype, np.integer)
    assert (chosen[:, 0] < 2).all() and (chosen[:, 1] < 3).all()
    p = ac.probabilities(ac.state)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0)
    assert not p[:, 0, 2].any()
    before = ac.learner.brain.efficacy.copy()
    ac.learn(np.ones(len(drive)), np.ones(len(drive), bool), drive)
    assert np.any(before != ac.learner.brain.efficacy)
    assert not ac.trace.any()


@pytest.mark.parametrize("slots", [1, [2, 3]])
def test_reward_nudge_matches_the_log_policy_derivative(slots):
    ac, drive = actor("quadratic", slots=slots)
    drive = drive[:1]
    choice = ac.act(drive)[0]
    _, plus, minus, _ = ac._pending
    # Motor bias derivative has no reciprocal-weight multiplicity ambiguity.
    contrast = (plus.activation[0] - minus.activation[0]) / (2 * ac.learner.config.beta)
    original = ac.learner.brain
    numeric = []
    for i in ac.learner.output_index:
        logs = []
        for sign in (-1, 1):
            bias = original.bias.copy()
            bias[i] += sign * 1e-5
            ac.learner.brain = original.with_parameters(bias=bias)
            state = ac.learner.free(drive)
            probabilities = ac.probabilities(state)[0]
            selected = (
                probabilities[choice]
                if probabilities.ndim == 1
                else probabilities[np.arange(len(choice)), choice]
            )
            logs.append(np.log(selected).sum())
        numeric.append((logs[1] - logs[0]) / 2e-5)
    ac.learner.brain = original
    # Cadence's nudge uses target-p, hence differentiates -T*log(p).
    # Temperature is a fixed scalar and can be absorbed in the actor rate.
    np.testing.assert_allclose(
        contrast[ac.learner.output_index],
        ac.learner.config.temperature * np.array(numeric),
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_calibrated_critic_padding_does_not_change_real_transitions(backend):
    from dataclasses import replace

    if backend == "torch":
        pytest.importorskip("torch")
    outcomes = []
    for padded in (False, True):
        ac, drive = actor(backend=backend)
        drive = drive[:2]
        if padded:
            drive = np.concatenate([drive, np.zeros_like(drive[:1])])
        # Fix phase work: a padding row may otherwise change batchwise stopping.
        ac.learner.config = replace(ac.learner.config, tolerance=None)
        for step in range(6):
            ac.rng = np.random.default_rng(step)
            ac.act(drive)
            reward = np.array([10.0, -3.0, 999.0]) if padded else np.array([10.0, -3.0])
            observed = np.array([True, True, False]) if padded else None
            ac.learn(reward, np.full(len(drive), step == 5), drive, observed=observed)
        outcomes.append(
            (ac.learner.brain.efficacy, ac.learner.brain.bias, ac.w_critic, np.array([ac.b_critic]))
        )
    for a, b in zip(*outcomes, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-8)
