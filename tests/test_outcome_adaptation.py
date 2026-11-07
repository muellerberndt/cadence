"""A learner and a composed brain adapt to outcomes, and a brain built without its associative
memory keeps that choice through a checkpoint."""

from __future__ import annotations

from typing import Any

import numpy as np

import cadence as cd


class Evaluator:
    """A tiny brain learns whether a predicted terminal observation is good or bad."""

    def __init__(self, seed: int = 0) -> None:
        connectome = cd.layered(2, 8, 2, density=1, seed=seed)
        config = cd.LearnerConfig(eta=3, eta_bias=0.03, temperature=0.1, tolerance=1e-4)
        self.learner = cd.Learner(
            cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1)),
            connectome.populations["output"],
            config,
        )
        self.n = connectome.n

    def drive(self, features: np.ndarray) -> np.ndarray:
        return np.pad(features, ((0, 0), (0, self.n - 2)))

    def teach(self, features: np.ndarray, outcomes: np.ndarray) -> None:
        self.learner.step(self.drive(features), (outcomes < 0).astype(int))

    def value(self, state: dict[str, Any]) -> float:
        free = self.learner.free(self.drive(state["features"][None]))
        logits = free.activation[0, self.learner.output_index] / 0.1
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return float(p[0] - p[1])

    def observe_outcome(self, state: dict[str, Any], actual: float) -> float:
        """Report a signed prediction error and teach the observed terminal outcome."""
        error = actual - self.value(state)
        self.teach(state["features"][None], np.array([actual]))
        return error


def test_unexpected_bad_outcome_reduces_its_predicted_value():
    brain = Evaluator()
    state = {"features": np.array([0.8, 0.0])}
    for _ in range(20):
        brain.observe_outcome(state, 1.0)
    before = brain.value(state)
    error = brain.observe_outcome(state, -1.0)
    assert error < 0
    assert brain.value(state) < before


def _changing_rewards(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    agent = cd.Brain.compose(4, 4, modules=(16,), seed=seed, recall=False)
    context = rng.integers(4, size=32)
    action = agent.step(np.eye(4)[context])
    rewards = []
    for moment in range(600):
        # The world changes its reward rule halfway through the same learning life.
        rewarded_action = (context + int(moment >= 300)) % 4
        reward = (action == rewarded_action).astype(float)
        rewards.append(float(reward.mean()))
        context = rng.integers(4, size=32)
        action = agent.step(np.eye(4)[context], reward=reward, done=np.ones(32, bool))
    return {
        "before_change": float(np.mean(rewards[250:300])),
        "just_after_change": float(np.mean(rewards[300:310])),
        "after_adapting": float(np.mean(rewards[-50:])),
    }


def test_reward_stream_adapts_without_restarting_its_brain():
    result = _changing_rewards()
    assert result["before_change"] > 0.9
    assert result["after_adapting"] > 0.9
    assert result["just_after_change"] < result["after_adapting"]


def test_explicit_memory_opt_out_and_checkpoint_preserve_configuration(tmp_path):
    brain = cd.Brain.compose(2, 2, modules=(4,), episodic=False, recall=False)
    assert brain.hippocampus is None
    assert cd.Brain.load(brain.save(tmp_path / "without.npz")).hippocampus is None
    assert isinstance(cd.Brain(brain.connectome).hippocampus, cd.SynapticMemory)
