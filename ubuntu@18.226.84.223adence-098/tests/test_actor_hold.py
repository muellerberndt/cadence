"""The actor's eligibility held apart from the critic's (``actor_hold``, cadence#283)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd
from cadence import ActorCriticConfig


def _lived(seed: int = 0, moments: int = 12, **settings) -> cd.Brain:
    warnings.simplefilter("ignore")
    brain = cd.Brain.compose(5, 2, seed=seed, arousal=True, backend="cpu", **settings)
    eye = np.eye(5)
    rng = np.random.default_rng(seed)
    brain.step(eye[[0]])
    for moment in range(1, moments):
        reward = [float(rng.choice([-1.0, 0.0, 1.0]))]
        brain.step(eye[[int(rng.integers(5))]], reward=reward, done=[moment % 5 == 0])
    return brain


def test_without_a_hold_every_trace_decays_by_gamma_lam_and_saves_without_the_field():
    config = ActorCriticConfig(gamma=0.9, lam=0.8)
    assert config.hold is None and config.decay == 0.9 * 0.8
    assert "hold" not in config.to_dict()
    assert ActorCriticConfig(**config.to_dict()) == config
    held = ActorCriticConfig(gamma=0.9, lam=0.8, hold=1.0)
    assert held.decay == 1.0 and held.to_dict()["hold"] == 1.0
    for bad in (1.5, -0.1, True, "1"):
        with pytest.raises(ValueError):
            ActorCriticConfig(hold=bad)


def test_a_composed_brain_without_the_hold_is_the_founders_to_the_bit():
    founder = _lived()
    unset = _lived(actor_hold=None)
    np.testing.assert_array_equal(founder.learner.brain.efficacy, unset.learner.brain.efficacy)
    np.testing.assert_array_equal(founder.basal_ganglia.w_critic, unset.basal_ganglia.w_critic)
    assert "actor_hold" not in unset.describe()["genes"]
    # a hold equal to the founder's decay is the founder's arithmetic
    config = founder.basal_ganglia.config
    same = _lived(actor_hold=config.gamma * config.lam)
    np.testing.assert_array_equal(founder.learner.brain.efficacy, same.learner.brain.efficacy)
    assert same.describe()["genes"]["actor_hold"] == config.gamma * config.lam


def test_a_held_actor_trace_keeps_its_credit_while_the_critics_fades():
    brain = _lived(moments=3, actor_hold=1.0)
    agent = brain.basal_ganglia
    eye = np.eye(5)
    brain.step(eye[[1]], reward=[0.0])
    trace, critic = agent.trace.copy(), agent.trace_critic.copy()
    bias = agent.trace_bias.copy()
    agent.fade()
    np.testing.assert_array_equal(agent.trace, trace)
    np.testing.assert_array_equal(agent.trace_bias, bias)
    decay = agent.config.gamma * agent.config.lam
    np.testing.assert_allclose(agent.trace_critic, decay * critic, rtol=1e-12)
    agent.fade(np.array([True]))
    assert not agent.trace.any() and not agent.trace_critic.any()


def test_a_held_trace_sums_the_contrasts_of_an_episodes_acts():
    """``learn``: e <- hold * e + contrast, the critic's trace by gamma * lam."""
    brain = _lived(moments=3, actor_hold=0.5)
    agent = brain.basal_ganglia
    eye = np.eye(5)
    brain.step(eye[[2]], reward=[0.0])
    before, critic_before = agent.trace.copy(), agent.trace_critic.copy()
    free = agent._free
    brain.step(eye[[3]], reward=[0.0])
    contrast = agent.trace - 0.5 * before
    # the contrast added is the act's own: the bias trace moved by the same rule
    assert np.abs(contrast).max() > 0
    decay = agent.config.gamma * agent.config.lam
    expected = decay * critic_before
    expected[:, :-1] += free.activation[:, agent.critic_index]
    expected[:, -1] += 1.0
    np.testing.assert_allclose(agent.trace_critic, expected, rtol=1e-12)


def test_the_hold_is_retuned_saved_and_loaded(tmp_path):
    warnings.simplefilter("ignore")
    brain = cd.Brain.compose(5, 2, seed=0, actor_hold=1.0)
    assert brain.basal_ganglia.config.hold == 1.0
    path = brain.save(tmp_path / "held.npz")
    assert cd.Brain.load(path).basal_ganglia.config.hold == 1.0
    brain.retune(actor_hold=None)
    assert brain.basal_ganglia.config.hold is None
    assert brain.basal_ganglia.config.decay == brain.basal_ganglia.config.gamma * 0.8
    with pytest.raises(ValueError):
        brain.retune(actor_hold=2.0)


def test_a_hold_on_a_torch_device_follows_the_host():
    pytest.importorskip("torch")
    warnings.simplefilter("ignore")
    eye = np.eye(5)

    def lived(backend: str, **settings) -> list[np.ndarray]:
        extra = {"device": "cpu", "precision": "float64"} if backend == "torch" else {}
        lives = cd.Lives(5, 2, 2, backend=backend, **extra, **settings)
        rng = np.random.default_rng(0)
        lives.step(eye[[0, 1]])
        for moment in range(1, 15):
            lives.step(eye[rng.integers(5, size=2)], reward=rng.choice([-1.0, 0.0, 1.0], size=2),
                       done=np.array([moment % 5 == 0] * 2))
        return [np.array(lives.life(i).learner.brain.efficacy) for i in range(2)]

    host, device = lived("cpu", actor_hold=1.0), lived("torch", actor_hold=1.0)
    unheld = lived("cpu")
    for held, there, founder in zip(host, device, unheld, strict=True):
        assert np.abs(held - there).max() < 1e-3  # the solver's tolerance, not the hold's
        assert np.abs(held - founder).max() > 1e-2  # the hold moves what the lives learn
