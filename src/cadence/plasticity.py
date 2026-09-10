"""Three factors: a trace at every seam, a critic, and a dopamine owner.

Learning from reward in a body is not a nudge at the end of a rollout. Every
seam keeps an eligibility trace, the recent history of what it would have
moved by for the actions that were taken; a critic population reads the
settled state and learns to predict return; and a dopamine owner broadcasts
the temporal-difference error of that prediction. The three multiply, at
every seam, every step:

    e[e]     <- gamma * lam * e[e] + contrast[e]        (the action's eligibility)
    delta    =  r + gamma * V(s') - V(s)                 (the dopamine signal)
    scale[e] += eta * delta * e[e]

There is no buffer and no epoch: one life, one pass. The contrast is the
free/nudged difference the learner already computes, per row, with the
nudge's target the action that was taken, so the eligibility is the score of
that action and the goal still enters only through the nudge. The critic is a
linear reading of the owners named for it, trained by its own trace and the
same dopamine, so every update reads a seam's two endpoints and one broadcast
number. ``normalize`` divides each seam's step by the running RMS of its own
steps, the local counterpart of an adaptive optimiser.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .learning import Learner
from .settle import SettledState

__all__ = ["ActorCritic", "ActorCriticConfig", "Population"]


@dataclass(frozen=True)
class Population:
    """A continuous action as a bump over ``size`` owners per dimension.

    Owner ``k`` of a dimension stands for the level ``centres[k]`` in ``[-1, 1]``; the
    dimension's value is read out as the activation-weighted mean of the centres, and a
    value is written as a Gaussian bump of ``width`` around it. Exploration is Gaussian
    noise of ``sigma`` on the read-out value; the taken value's bump is the nudge's target.
    """

    dims: int
    size: int = 9
    width: float = 0.25
    sigma: float = 0.3
    temperature: float = 0.1  # the readout is the softmax-weighted mean of the centres: a soft winner

    @property
    def centres(self) -> np.ndarray:
        return np.linspace(-1.0, 1.0, self.size)

    def read(self, activation: np.ndarray) -> np.ndarray:
        """``(batch, dims * size)`` activations to ``(batch, dims)`` values in ``[-1, 1]``."""
        s = activation.reshape(len(activation), self.dims, self.size) / self.temperature
        s = s - s.max(axis=2, keepdims=True)
        w = np.exp(s)
        w /= w.sum(axis=2, keepdims=True)
        return (w * self.centres[None, None, :]).sum(axis=2)

    def write(self, value: np.ndarray) -> np.ndarray:
        """``(batch, dims)`` values to ``(batch, dims * size)`` bumps."""
        gap = value[:, :, None] - self.centres[None, None, :]
        return np.exp(-(gap**2) / (2.0 * self.width**2)).reshape(len(value), self.dims * self.size)



@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    gamma: float = 0.99  # discount
    lam: float = 0.9  # trace decay of the actor's eligibility
    lam_critic: float = 0.9  # trace decay of the critic's eligibility
    eta: float = 0.5  # actor step per unit dopamine per unit trace
    eta_bias: float = 0.05
    eta_critic: float = 0.05
    normalize: float = 0.0  # >0: forgetting factor of the per-seam RMS that divides its step
    normalize_floor: float = 1e-3
    momentum: float = 0.0  # >0: each seam steps on a running average of its own steps, so sign noise cancels before the RMS divides it
    entropy: float = 0.0  # >0: a standing nudge toward the uniform policy, a taste for variety
    critic_init: float = 0.0
    dopamine_cap: float = 1.0  # the broadcast saturates: |delta| is clipped here (0: no cap)
    dopamine_center: float = 0.0  # >0: forgetting factor of a running mean and scale of delta; the phasic signal is the deviation from the tonic level
    step_cap: float = 0.0  # >0: no seam moves by more than this in one update, a bound on synaptic change per event
    critic_normalize: bool = True  # the critic's step is divided by its trace's energy, so its step size is scale-free

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class ActorCritic:
    """A learner that acts, keeps eligibility, and learns from dopamine.

    Use::

        ac = ActorCritic(learner, critic=wiring.sets["hidden"])
        action = ac.act(drive)                 # settle, sample, keep eligibility
        obs, reward, done = env.step(action)
        ac.learn(reward, done, next_drive)     # settle the next state, dopamine, update

    ``act`` reuses the settlement ``learn`` already made for the next state, so a
    step costs one free settlement and two nudged ones.
    """

    def __init__(
        self,
        learner: Learner,
        critic: Sequence[int],
        config: ActorCriticConfig | None = None,
        seed: int = 0,
        population: Population | None = None,
    ) -> None:
        self.learner = learner
        self.config = config or ActorCriticConfig()
        self.population = population
        if population is not None and len(learner.output_index) != population.dims * population.size:
            raise ValueError("the output set must hold dims * size owners for a population code")
        self.critic_index = np.asarray(list(critic), dtype=np.int64)
        self.w_critic = np.full(len(self.critic_index), self.config.critic_init)
        self.b_critic = 0.0
        self.rng = np.random.default_rng(seed)
        w = learner.engine.wiring
        self.edges, self.n = w.edges, w.n
        self.trace: np.ndarray | None = None
        self.trace_bias: np.ndarray | None = None
        self.trace_critic: np.ndarray | None = None
        self.second_moment = np.zeros(self.edges)
        self.second_moment_bias = np.zeros(self.n)
        self.velocity = np.zeros(self.edges)
        self.velocity_bias = np.zeros(self.n)
        self.delta_mean = 0.0
        self.delta_var = 1.0
        self._free: SettledState | None = None
        self._pending: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.updates = 0

    # -- readings

    def value(self, state: SettledState) -> np.ndarray:
        return state.activation[:, self.critic_index] @ self.w_critic + self.b_critic

    def probabilities(self, state: SettledState) -> np.ndarray:
        s = state.activation[:, self.learner.output_index]
        z = s / self.learner.config.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def settle(self, drive: np.ndarray) -> SettledState:
        """The free settlement for ``drive``, warm from the last one; cached for ``act``."""
        if self._free is not None and self._free.v.shape[0] != len(drive):
            self._free = None
        self._free = self.learner.free(drive, warm=self._free)
        return self._free

    # -- acting

    def act(self, drive: np.ndarray, greedy: bool = False) -> np.ndarray:
        """Settle (or reuse the cached settlement), sample an action per row, keep its eligibility.

        Discrete: an action index per row from the softmax over the output owners.
        With a ``Population``: a value per dimension in ``[-1, 1]``, Gaussian exploration
        around the read-out, the taken value's bump as the nudge's target.
        """
        free = self._free if self._free is not None and self._free.v.shape[0] == len(drive) else self.settle(drive)
        if self.population is not None:
            return self._act_continuous(drive, free, greedy)
        p = self.probabilities(free)
        if greedy:
            action = np.argmax(p, axis=1)
        else:
            u = self.rng.random(len(p))
            action = (p.cumsum(axis=1) < u[:, None]).sum(axis=1)
            action = np.minimum(action, p.shape[1] - 1)
        if not greedy:
            target = self.learner.targets(action)
            plus = self.learner.nudged(drive, free, target)
            minus = self.learner.nudged(drive, free, target, sign=-1.0)
            contrast, contrast_bias = self.learner.contrast_rows(free, plus, minus)
            if self.config.entropy > 0:  # a standing pull toward the uniform policy
                uniform = np.zeros_like(target)
                uniform[:, self.learner.output_index] = 1.0 / len(self.learner.output_index)
                up = self.learner.nudged(drive, free, uniform)
                down = self.learner.nudged(drive, free, uniform, sign=-1.0)
                c2, cb2 = self.learner.contrast_rows(free, up, down)
                contrast = contrast + self.config.entropy * c2
                contrast_bias = contrast_bias + self.config.entropy * cb2
            self._pending = (contrast, contrast_bias, self.value(free))
        return np.asarray(action, dtype=np.int64)

    def _act_continuous(self, drive: np.ndarray, free: SettledState, greedy: bool) -> np.ndarray:
        pop = self.population
        assert pop is not None
        mean = pop.read(free.activation[:, self.learner.output_index])
        if greedy:
            return mean
        action = np.clip(mean + pop.sigma * self.rng.standard_normal(mean.shape), -1.0, 1.0)
        target = np.zeros((len(drive), self.n))
        target[:, self.learner.output_index] = pop.write(action)
        plus = self.learner.nudged(drive, free, target)
        minus = self.learner.nudged(drive, free, target, sign=-1.0)
        contrast, contrast_bias = self.learner.contrast_rows(free, plus, minus)
        self._pending = (contrast, contrast_bias, self.value(free))
        return action

    # -- learning

    def learn(self, reward: np.ndarray, done: np.ndarray, next_drive: np.ndarray, bootstrap: np.ndarray | None = None) -> dict[str, float]:
        """Dopamine from the reward and the next state's value; every seam moves on its trace.

        ``done`` rows start their next life from rest and, unless ``bootstrap`` gives them
        a value, bootstrap from zero. A time limit is not a terminal state: pass the
        value of the last observation (``value_of``) as ``bootstrap`` for a truncated row.
        """
        cfg = self.config
        if self._pending is None:
            raise RuntimeError("learn needs an act first")
        contrast, contrast_bias, value = self._pending
        free = self._free
        assert free is not None
        reward = np.asarray(reward, dtype=float)
        done = np.asarray(done, dtype=bool)
        batch = len(reward)
        decay = cfg.gamma * cfg.lam
        if self.trace is None or self.trace.shape[0] != batch:
            self.trace = np.zeros((batch, self.edges))
            self.trace_bias = np.zeros((batch, self.n))
            self.trace_critic = np.zeros((batch, len(self.critic_index) + 1))
        assert self.trace_bias is not None and self.trace_critic is not None
        self.trace *= decay
        self.trace += contrast
        self.trace_bias *= decay
        self.trace_bias += contrast_bias
        self.trace_critic *= cfg.gamma * cfg.lam_critic
        self.trace_critic[:, :-1] += free.activation[:, self.critic_index]
        self.trace_critic[:, -1] += 1.0
        # the next state, warm from this one; a finished row starts its next life from rest
        next_state = self.learner.free(next_drive, warm=free)
        if done.any():
            v = next_state.v.copy()
            a = next_state.adaptation.copy()
            v[done] = 0.0
            a[done] = 0.0
            next_state = self.learner.free(next_drive, warm=SettledState(v, next_state.activation, a, next_state.steps))
        next_value = np.where(done, 0.0 if bootstrap is None else np.asarray(bootstrap, dtype=float), self.value(next_state))
        delta = reward + cfg.gamma * next_value - value
        if cfg.dopamine_center > 0:
            rho = cfg.dopamine_center
            self.delta_mean = rho * self.delta_mean + (1 - rho) * float(delta.mean())
            self.delta_var = rho * self.delta_var + (1 - rho) * float(((delta - self.delta_mean) ** 2).mean())
            delta = (delta - self.delta_mean) / (np.sqrt(self.delta_var) + 1e-6)
        if cfg.dopamine_cap > 0:
            delta = np.clip(delta, -cfg.dopamine_cap, cfg.dopamine_cap)
        # three factors
        step_scale = (delta[:, None] * self.trace).mean(axis=0)
        step_bias = (delta[:, None] * self.trace_bias).mean(axis=0)
        if cfg.momentum > 0:
            self.velocity = cfg.momentum * self.velocity + (1 - cfg.momentum) * step_scale
            self.velocity_bias = cfg.momentum * self.velocity_bias + (1 - cfg.momentum) * step_bias
            step_scale, step_bias = self.velocity, self.velocity_bias
        if cfg.normalize > 0:
            rho = cfg.normalize
            self.second_moment = rho * self.second_moment + (1 - rho) * step_scale**2
            self.second_moment_bias = rho * self.second_moment_bias + (1 - rho) * step_bias**2
            step_scale = step_scale / (np.sqrt(self.second_moment) + cfg.normalize_floor)
            step_bias = step_bias / (np.sqrt(self.second_moment_bias) + cfg.normalize_floor)
        step_scale = cfg.eta * step_scale
        step_bias = cfg.eta_bias * step_bias
        if cfg.step_cap > 0:
            step_scale = np.clip(step_scale, -cfg.step_cap, cfg.step_cap)
            step_bias = np.clip(step_bias, -cfg.step_cap, cfg.step_cap)
        report = self.learner.apply(step_scale, step_bias)
        critic_trace = self.trace_critic
        if cfg.critic_normalize:
            critic_trace = critic_trace / (1.0 + (critic_trace**2).sum(axis=1, keepdims=True))
        critic_step = cfg.eta_critic * (delta[:, None] * critic_trace).mean(axis=0)
        self.w_critic += critic_step[:-1]
        self.b_critic += float(critic_step[-1])
        # a finished row forgets its traces
        if done.any():
            self.trace[done] = 0.0
            self.trace_bias[done] = 0.0
            self.trace_critic[done] = 0.0
        self._free = next_state
        self._pending = None
        self.updates += 1
        report["delta"] = float(np.abs(delta).mean())
        report["value"] = float(value.mean())
        report["free_steps"] = float(next_state.steps)
        return report

    def value_of(self, drive: np.ndarray) -> np.ndarray:
        """The critic's value of a drive, settled cold, without touching the cached state."""
        return self.value(self.learner.free(drive))

    def reset(self) -> None:
        self._free = None
        self._pending = None
        self.trace = self.trace_bias = self.trace_critic = None

    def parameters(self) -> int:
        return self.learner.parameters() + len(self.w_critic) + 1

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "critic": [int(i) for i in self.critic_index], "updates": self.updates, "learner": self.learner.to_dict()}
