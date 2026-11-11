"""Audit 2026-09-15, class 8: the actor-critic's masks, terminals, truncation and cache."""

import numpy as np
import pytest

import cadence as cd


def _setup() -> tuple[cd.Connectome, cd.NeuronModel, cd.LearnerConfig]:
    c, _ = cd.embedded(8, 2, 4, 12, 4, seed=0)
    return c, cd.learning_neuron_model(dt=1.0), cd.LearnerConfig(tolerance=1e-5, free_steps=200, nudged_steps=100)


def _drive(c: cd.Connectome, rng: np.random.Generator, batch: int) -> np.ndarray:
    d = np.zeros((batch, c.n))
    for p in range(2):
        d[np.arange(batch), p * 8 + rng.integers(0, 8, batch)] = 1.0
    return d


def _agent(backend: str = "cpu", config: cd.ActorCriticConfig | None = None) -> cd.ActorCritic:
    c, model, learning = _setup()
    brain = cd.NeuralGraph(c, model, backend=backend, device="cpu" if backend == "torch" else None)
    learner = cd.Learner(brain, c.populations["output"], learning)
    return cd.ActorCritic(learner, c.populations["hidden"], config or cd.ActorCriticConfig(gamma=0.9, lam=0.8, eta=0.5))


def test_device_learn_matches_host_learn_with_observed_done_and_bootstrap() -> None:
    pytest.importorskip("torch")
    c, _, _ = _setup()
    rng = np.random.default_rng(0)
    host, dev = _agent("cpu"), _agent("torch")
    d0, d1, d2 = _drive(c, rng, 4), _drive(c, rng, 4), _drive(c, rng, 4)
    assert np.array_equal(host.act(d0), dev.act(d0))
    reward = rng.normal(size=4)
    done = np.array([False, True, False, False])
    observed = np.array([True, True, False, True])
    reports = [a.learn(reward, done, d1, observed=observed) for a in (host, dev)]
    assert reports[0] == pytest.approx(reports[1], abs=1e-9)
    assert np.allclose(host.learner.brain.efficacy, dev.learner.brain.efficacy, atol=1e-12)
    assert host.trace is not None and dev._trace_device is not None
    device_trace = dev._trace_device[0].cpu().numpy()
    for row, cleared in enumerate([False, True, True, False]):  # done or padding rows forget
        assert (not host.trace[row].any()) is cleared and (not device_trace[row].any()) is cleared
    assert np.array_equal(host.act(d1), dev.act(d1))
    bootstrap = np.array([0.7, 0.0, 0.0, 0.0])
    truncated = np.array([True, False, False, False])
    reports = [a.learn(reward, truncated, d2, bootstrap=bootstrap) for a in (host, dev)]
    assert reports[0] == pytest.approx(reports[1], abs=1e-9)
    assert np.allclose(host.learner.brain.efficacy, dev.learner.brain.efficacy, atol=1e-12)


def test_bootstrap_value_enters_the_dopamine_of_a_truncated_row() -> None:
    c, _, _ = _setup()
    rng = np.random.default_rng(1)
    # eta_critic zero: the value read after learn is the value the dopamine was computed with
    agent = _agent(
        "cpu", cd.ActorCriticConfig(gamma=0.9, lam=0.8, eta=0.5, eta_critic=0.0, dopamine_cap=0.0)
    )
    d0, d1 = _drive(c, rng, 3), _drive(c, rng, 3)
    agent.act(d0)
    assert agent._pending is not None
    value = agent._pending[3].copy()
    reward = np.array([1.0, 2.0, 3.0])
    report = agent.learn(reward, np.array([False, True, True]), d1, bootstrap=np.array([0.0, 5.0, 0.0]))
    assert agent.state is not None
    next_value = agent.value(agent.state)
    expected = reward + 0.9 * np.array([next_value[0], 5.0, 0.0]) - value
    assert report["dopamine"] == pytest.approx(expected.mean())
    assert report["delta"] == pytest.approx(np.abs(expected).mean())


def test_observed_padding_is_excluded_and_a_batch_without_a_real_row_is_refused() -> None:
    c, _, _ = _setup()
    rng = np.random.default_rng(2)
    agent = _agent("cpu")
    agent.act(_drive(c, rng, 2))
    with pytest.raises(ValueError, match="real transition"):
        agent.learn(np.ones(2), np.zeros(2, dtype=bool), _drive(c, rng, 2), observed=np.zeros(2, dtype=bool))
    agent.learn(np.ones(2), np.zeros(2, dtype=bool), _drive(c, rng, 2), observed=np.array([True, False]))
    assert agent.trace is not None  # the padding row forgets its trace, the real row keeps it
    assert not agent.trace[1].any() and agent.trace[0].any()


def test_act_refreshes_after_learning_and_a_new_batch_size_drops_the_traces() -> None:
    c, _, _ = _setup()
    rng = np.random.default_rng(3)
    agent = _agent("cpu")
    d0, d1 = _drive(c, rng, 3), _drive(c, rng, 3)
    agent.act(d0)
    agent.learn(np.ones(3), np.zeros(3, dtype=bool), d1)
    settled = agent.state
    agent.act(d1)
    assert agent.state is not settled  # the update invalidates the bootstrap phase
    refreshed = agent.state
    agent.act(d1)
    assert agent.state is refreshed  # the drive and parameters have not changed
    agent.act(_drive(c, rng, 2))
    assert agent.trace is None and agent.state is not None and agent.state.v.shape[0] == 2
