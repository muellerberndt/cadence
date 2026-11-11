"""``actor_entropy`` (cadence#296): the policy's entropy in the actor's step, opt-in."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import ActorCritic, ActorCriticConfig, Brain


def life(brain: Brain, moments: int = 24, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, (1, len(brain.connectome.populations["sensory"])))
    actions = [int(brain.step(x)[0])]
    for i in range(moments):
        x = rng.uniform(0.0, 1.0, x.shape)
        reward = [1.0 if actions[-1] == 0 else -0.1]
        actions.append(int(brain.step(x, reward=reward, done=[i % 9 == 8])[0]))
    return actions


def test_a_brain_without_the_entropy_keeps_its_bits() -> None:
    plain = Brain.compose(5, 3, seed=2, backend="cpu")
    named = Brain.compose(5, 3, seed=2, backend="cpu", actor_entropy=0.0)
    assert life(plain) == life(named)
    assert np.array_equal(plain.brain.efficacy, named.brain.efficacy)
    assert np.array_equal(plain.brain.bias, named.brain.bias)
    assert "entropy" not in ActorCriticConfig().to_dict()
    assert "actor_entropy" not in plain.describe()["genes"]
    assert "policy_entropy" not in (plain.last_learning or {})


def test_the_surprise_is_minus_log_p_minus_the_entropy_and_averages_to_zero() -> None:
    p = np.array([[0.7, 0.2, 0.1], [0.5, 0.5, 0.0]])
    surprise, entropy = ActorCritic._surprise(p, np.array([0, 1]))
    h = [-(0.7 * np.log(0.7) + 0.2 * np.log(0.2) + 0.1 * np.log(0.1)), np.log(2.0)]
    assert np.allclose(entropy, h)
    assert np.allclose(surprise, [-np.log(0.7) - h[0], -np.log(0.5) - h[1]])
    # over the policy's own actions the surprise has mean zero: the entropy's gradient, sampled
    for row in p:
        values = [ActorCritic._surprise(row[None], np.array([a]))[0][0] for a in range(3)]
        assert np.dot(row, np.nan_to_num(values) * (row > 0)) == pytest.approx(0.0, abs=1e-12)
    # several motor slots: the slots' surprises and entropies add
    slots = np.array([[[0.6, 0.4, 0.0], [0.1, 0.3, 0.6]]])
    s, e = ActorCritic._surprise(slots, np.array([[1, 2]]))
    one = ActorCritic._surprise(slots[:, 0, :2], np.array([1]))
    two = ActorCritic._surprise(slots[:, 1], np.array([2]))
    assert s[0] == pytest.approx(one[0][0] + two[0][0])
    assert e[0] == pytest.approx(one[1][0] + two[1][0])


def test_the_entropy_step_is_the_acts_own_contrast_at_the_weight_of_its_surprise() -> None:
    weight = 0.5
    plain = Brain.compose(4, 3, seed=5, backend="cpu", episodic=False, recall=False)
    explored = Brain.compose(4, 3, seed=5, backend="cpu", episodic=False, recall=False,
                             actor_entropy=weight)
    x = np.array([[0.2, 0.9, 0.4, 0.1]])
    assert plain.step(x)[0] == explored.step(x)[0]  # the entropy leaves the act as it is
    agent = explored.basal_ganglia
    surprise, entropy = agent._pending_entropy
    _, plus, minus, _ = agent._pending
    span = 2.0 * explored.learner.config.beta
    w = explored.connectome
    a_plus, a_minus = plus.activation, minus.activation
    contrast = (a_plus[:, w.pre] * (a_plus[:, w.post] - a_minus[:, w.post])
                + (a_plus[:, w.pre] - a_minus[:, w.pre]) * a_minus[:, w.post]) / span
    contrast_bias = (a_plus - a_minus) / span
    before = (plain.brain.efficacy.copy(), plain.brain.bias.copy())
    plain.step(x, reward=[0.3], done=[False])
    explored.step(x, reward=[0.3], done=[False])
    report = explored.last_learning
    assert report["policy_entropy"] == pytest.approx(entropy[0])
    eta = agent.config.eta
    eta_bias = agent.config.eta_bias
    moved = explored.brain.efficacy - plain.brain.efficacy
    expected = eta * weight * surprise[0] * contrast[0]
    plastic = explored.learner.plastic_synapses
    assert np.allclose(moved[plastic], expected[plastic], atol=1e-12)
    assert np.allclose(explored.brain.bias - plain.brain.bias,
                       np.where(explored.learner.plastic_neurons,
                                eta_bias * weight * surprise[0] * contrast_bias[0], 0.0),
                       atol=1e-12)
    assert not np.array_equal(before[0], plain.brain.efficacy)


def test_a_pending_acts_surprise_saves_and_loads(tmp_path) -> None:
    brain = Brain.compose(4, 3, seed=1, backend="cpu", actor_entropy=0.05)
    life(brain, 7)
    held = brain.basal_ganglia._pending_entropy
    assert held is not None
    path = tmp_path / "brain.npz"
    brain.save(path)
    loaded = Brain.load(path)
    assert loaded.describe()["genes"]["actor_entropy"] == 0.05
    for kept, read in zip(held, loaded.basal_ganglia._pending_entropy, strict=True):
        assert np.array_equal(kept, read)
    x = np.full((1, 4), 0.5)
    assert brain.step(x, reward=[1.0], done=[False]) == loaded.step(x, reward=[1.0], done=[False])
    assert np.allclose(brain.brain.efficacy, loaded.brain.efficacy, atol=1e-12)


def test_the_entropy_weight_is_finite_and_nonnegative_and_retunes() -> None:
    with pytest.raises(ValueError):
        ActorCriticConfig(entropy=-0.1)
    with pytest.raises(ValueError):
        ActorCriticConfig(entropy=float("nan"))
    brain = Brain.compose(4, 2, seed=0, backend="cpu")
    brain.retune(actor_entropy=0.02)
    assert brain.describe()["genes"]["actor_entropy"] == 0.02
    life(brain, 5)
    assert "policy_entropy" in brain.last_learning


def test_many_lives_refuse_the_entropy() -> None:
    pytest.importorskip("torch")
    from cadence import Lives

    with pytest.raises(ValueError, match="actor entropy"):
        Lives(4, 2, lives=2, backend="torch", actor_entropy=0.05)
