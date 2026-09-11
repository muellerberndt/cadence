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
from .rules import GradedRule
from .settle import _FUSED as _FUSED_TRACE
from .settle import Nudge, SettledState, Settlement

__all__ = [
    "ActorCritic",
    "ActorCriticConfig",
    "Population",
    "Bins",
    "ValueNet",
    "ValueConfig",
    "Rehearsal",
    "RehearsalConfig",
]


@dataclass(frozen=True, slots=True)
class ValueConfig:
    scale: float = 100.0  # the value is scale times (the value owner's activation minus offset)
    offset: float = 0.2
    beta: float = 0.1
    eta: float = 1.0
    eta_bias: float = 0.02
    consolidate: float = (
        0.005  # the slow copy, which reads the target, follows by this fraction per update
    )
    free_steps: int = 100
    nudged_steps: int = 12
    tolerance: float = 3e-3

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class ValueNet:
    """A critic with its own hidden owners: a settlement net whose value owner is nudged
    toward the target.

    Built on a ``layered(inputs, hidden, 1)`` wiring. ``value(drive)`` settles the net under
    the actor's input levels (the first ``inputs`` columns of the actor's drive) and reads the
    value owner; ``learn(drive, target)`` nudges that owner toward the target's activation
    and moves every seam on its own contrast. A slow copy of the strengths, following by
    ``consolidate`` per update, reads the bootstrap target.
    """

    def __init__(
        self,
        inputs: int,
        hidden: int,
        rule: GradedRule,
        config: ValueConfig | None = None,
        seed: int = 0,
    ) -> None:
        from .learning import LearnerConfig, layered

        self.config = config or ValueConfig()
        self.inputs = inputs
        self.wiring = layered(inputs, hidden, 1, density=1.0, seed=seed)
        engine = Settlement(self.wiring, rule)
        cfg = self.config
        self.learner = Learner(
            engine,
            self.wiring.sets["output"],
            LearnerConfig(
                beta=cfg.beta,
                eta=cfg.eta,
                eta_bias=cfg.eta_bias,
                nudge="quadratic",
                tolerance=cfg.tolerance,
                free_steps=cfg.free_steps,
                nudged_steps=cfg.nudged_steps,
            ),
        )
        self.value_index = int(self.wiring.sets["output"][0])
        self.mask = np.zeros(self.wiring.n)
        self.mask[self.value_index] = 1.0
        self.slow_scale = engine.edge_scale.copy()
        self.slow_bias = engine.bias.copy()

    def drive_from(self, actor_drive: np.ndarray) -> np.ndarray:
        out = np.zeros((len(actor_drive), self.wiring.n))
        out[:, : self.inputs] = actor_drive[:, : self.inputs]
        return out

    def _read(self, state: SettledState) -> np.ndarray:
        return (state.activation[:, self.value_index] - self.config.offset) * self.config.scale

    def value(self, actor_drive: np.ndarray, slow: bool = False) -> np.ndarray:
        cfg = self.config
        engine = (
            self.learner.engine.with_parameters(edge_scale=self.slow_scale, bias=self.slow_bias)
            if slow
            else self.learner.engine
        )
        return self._read(
            engine.settle_batch(
                self.drive_from(actor_drive), steps=cfg.free_steps, tolerance=cfg.tolerance
            )
        )

    def learn(self, actor_drive: np.ndarray, target: np.ndarray) -> dict[str, float]:
        cfg = self.config
        drive = self.drive_from(actor_drive)
        free = self.learner.free(drive)
        full = np.zeros((len(drive), self.wiring.n))
        full[:, self.value_index] = np.clip(np.asarray(target) / cfg.scale + cfg.offset, 0.02, 0.98)
        plus = self.learner.engine.settle_batch(
            drive,
            steps=cfg.nudged_steps,
            state=free,
            nudge=Nudge(full, self.mask, cfg.beta),
            tolerance=cfg.tolerance,
        )
        minus = self.learner.engine.settle_batch(
            drive,
            steps=cfg.nudged_steps,
            state=free,
            nudge=Nudge(full, self.mask, -cfg.beta),
            tolerance=cfg.tolerance,
        )
        report = self.learner.update(free, plus, minus)
        self.slow_scale += cfg.consolidate * (self.learner.engine.edge_scale - self.slow_scale)
        self.slow_bias += cfg.consolidate * (self.learner.engine.bias - self.slow_bias)
        report["value"] = float(self._read(free).mean())
        return report

    def parameters(self) -> int:
        return self.learner.parameters()


@dataclass(frozen=True)
class Bins:
    """A continuous action as one softmax choice per dimension over ``size`` levels.

    Each dimension owns ``size`` output owners standing for levels in ``[-1, 1]``; an
    action is a draw from the softmax over each dimension's owners (a discrete choice
    per dimension), and learning is the cross-entropy nudge of that group toward the
    level taken, the same rule that learns a discrete action. Discretised actions match
    Gaussian ones on continuous control (Tang and Agrawal 2020); here they let one rule
    serve every kind of action.
    """

    dims: int
    size: int = 9

    @property
    def centres(self) -> np.ndarray:
        return np.linspace(-1.0, 1.0, self.size)

    def groups(self, output_index: np.ndarray, n: int) -> np.ndarray:
        gid = np.full(n, -1, dtype=np.int64)
        gid[output_index] = np.repeat(np.arange(self.dims), self.size)
        return gid

    def read(self, activation: np.ndarray, temperature: float) -> np.ndarray:
        """The most likely level per dimension: ``(batch, dims)`` values in ``[-1, 1]``."""
        s = activation.reshape(len(activation), self.dims, self.size)
        return np.asarray(self.centres[np.argmax(s, axis=2)])


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
    temperature: float = (
        0.1  # the readout is the softmax-weighted mean of the centres: a soft winner
    )

    @property
    def centres(self) -> np.ndarray:
        return np.linspace(-1.0, 1.0, self.size)

    def read(self, activation: np.ndarray) -> np.ndarray:
        """``(batch, dims * size)`` activations to ``(batch, dims)`` values in ``[-1, 1]``."""
        s = activation.reshape(len(activation), self.dims, self.size) / self.temperature
        s = s - s.max(axis=2, keepdims=True)
        w = np.exp(s)
        w /= w.sum(axis=2, keepdims=True)
        return np.asarray((w * self.centres[None, None, :]).sum(axis=2))

    def write(self, value: np.ndarray) -> np.ndarray:
        """``(batch, dims)`` values to ``(batch, dims * size)`` bumps."""
        gap = value[:, :, None] - self.centres[None, None, :]
        bumps = np.exp(-(gap**2) / (2.0 * self.width**2))
        return np.asarray(bumps.reshape(len(value), self.dims * self.size))


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
    # >0: each seam steps on a running average of its own steps, so sign noise cancels before
    # the RMS divides it
    momentum: float = 0.0
    critic_init: float = 0.0
    dopamine_cap: float = 1.0  # the broadcast saturates: |delta| is clipped here (0: no cap)
    # >0: forgetting factor of a running mean and scale of delta; the phasic signal is the
    # deviation from the tonic level
    dopamine_center: float = 0.0
    critic_normalize: bool = (
        True  # the critic's step is divided by its trace's energy, so its step size is scale-free
    )

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
        self.critic_net: ValueNet | None = critic if isinstance(critic, ValueNet) else None
        if self.critic_net is not None:
            critic = ()
        if (
            population is not None
            and len(learner.output_index) != population.dims * population.size
        ):
            raise ValueError("the output set must hold dims * size owners for a population code")
        self.bins = population if isinstance(population, Bins) else None
        if self.bins is not None:
            self.group_id: np.ndarray | None = self.bins.groups(
                learner.output_index, learner.engine.wiring.n
            )
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
        self._drive: np.ndarray | None = None
        self._free: SettledState | None = None
        self._pending: tuple[str, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.updates = 0

    # -- readings

    def value(self, state: SettledState) -> np.ndarray:
        if self.critic_net is not None:
            assert self._drive is not None
            return self.critic_net.value(self._drive)
        return np.asarray(state.activation[:, self.critic_index] @ self.w_critic + self.b_critic)

    def probabilities(self, state: SettledState) -> np.ndarray:
        s = state.activation[:, self.learner.output_index]
        z = s / self.learner.config.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return np.asarray(p / p.sum(axis=1, keepdims=True))

    def settle(self, drive: np.ndarray) -> SettledState:
        """The free settlement for ``drive``, warm from the last one; cached for ``act``."""
        if self._free is not None and self._free.v.shape[0] != len(drive):
            self._free = None
        self._free = self.learner.free(drive, warm=self._free)
        self._drive = drive
        return self._free

    # -- acting

    def act(self, drive: np.ndarray, greedy: bool = False) -> np.ndarray:
        """Settle (or reuse the cached settlement), sample an action per row, keep its eligibility.

        Discrete: an action index per row from the softmax over the output owners.
        With a ``Population``: a value per dimension in ``[-1, 1]``, Gaussian exploration
        around the read-out, the taken value's bump as the nudge's target.
        """
        free = (
            self._free
            if self._free is not None and self._free.v.shape[0] == len(drive)
            else self.settle(drive)
        )
        self._drive = drive
        if self.bins is not None:
            return self._act_bins(drive, free, greedy)
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
            self._pending = ("phases", plus.activation, minus.activation, self.value(free))
        return np.asarray(action, dtype=np.int64)

    def _act_bins(self, drive: np.ndarray, free: SettledState, greedy: bool) -> np.ndarray:
        """One softmax draw per dimension; the taken levels' one-hots are the nudge's target,
        per group."""
        bins = self.bins
        assert bins is not None
        cfg = self.learner.config
        batch = len(drive)
        s = free.activation[:, self.learner.output_index].reshape(batch, bins.dims, bins.size)
        z = s / cfg.temperature
        z = z - z.max(axis=2, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=2, keepdims=True)
        if greedy:
            choice = np.argmax(p, axis=2)
        else:
            u = self.rng.random((batch, bins.dims))
            choice = np.minimum((p.cumsum(axis=2) < u[:, :, None]).sum(axis=2), bins.size - 1)
        action = bins.centres[choice]
        if not greedy:
            onehot = np.zeros((batch, bins.dims, bins.size))
            np.put_along_axis(onehot, choice[:, :, None], 1.0, axis=2)
            target = np.zeros((batch, self.n))
            target[:, self.learner.output_index] = onehot.reshape(batch, -1)
            plus = self._nudged_groups(drive, free, target, cfg.beta)
            minus = self._nudged_groups(drive, free, target, -cfg.beta)
            self._pending = ("phases", plus.activation, minus.activation, self.value(free))
        return action

    def _nudged_groups(
        self, drive: np.ndarray, free: SettledState, target: np.ndarray, beta: float
    ) -> SettledState:
        cfg = self.learner.config
        nudge = Nudge(
            target,
            self.learner.output_mask,
            beta,
            softmax_temperature=cfg.temperature,
            groups=self.group_id,
        )
        return self.learner.engine.settle_batch(
            drive, steps=cfg.nudged_steps, state=free, nudge=nudge, tolerance=cfg.tolerance
        )

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
        self._pending = ("phases", plus.activation, minus.activation, self.value(free))
        return action

    # -- learning

    def learn(
        self,
        reward: np.ndarray,
        done: np.ndarray,
        next_drive: np.ndarray,
        bootstrap: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Dopamine from the reward and the next state's value; every seam moves on its trace.

        ``done`` rows start their next life from rest and, unless ``bootstrap`` gives them
        a value, bootstrap from zero. A time limit is not a terminal state: pass the
        value of the last observation (``value_of``) as ``bootstrap`` for a truncated row.
        """
        cfg = self.config
        if self._pending is None:
            raise RuntimeError("learn needs an act first")
        kind, first, second, value = self._pending
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
        fused = None
        if kind == "phases" and _FUSED_TRACE:
            fused = (
                first,
                second,
            )  # the contrast, trace, and dopamine-weighted sum are one fused pass in learn
        else:
            if kind == "phases":
                w = self.learner.engine.wiring
                span = 2.0 * self.learner.config.beta
                contrast = (
                    first[:, w.pre] * first[:, w.post] - second[:, w.pre] * second[:, w.post]
                ) / span
                contrast_bias = (first - second) / span
            else:
                contrast, contrast_bias = first, second
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
            next_state = self.learner.free(
                next_drive, warm=SettledState(v, next_state.activation, a, next_state.steps)
            )
        if self.critic_net is not None:
            next_value_raw = self.critic_net.value(next_drive, slow=True)
        else:
            next_value_raw = self.value(next_state)
        next_value = np.where(
            done, 0.0 if bootstrap is None else np.asarray(bootstrap, dtype=float), next_value_raw
        )
        raw_target = reward + cfg.gamma * next_value
        delta = raw_target - value
        if cfg.dopamine_center > 0:
            rho = cfg.dopamine_center
            self.delta_mean = rho * self.delta_mean + (1 - rho) * float(delta.mean())
            self.delta_var = rho * self.delta_var + (1 - rho) * float(
                ((delta - self.delta_mean) ** 2).mean()
            )
            delta = (delta - self.delta_mean) / (np.sqrt(self.delta_var) + 1e-6)
        if cfg.dopamine_cap > 0:
            delta = np.clip(delta, -cfg.dopamine_cap, cfg.dopamine_cap)
        # three factors
        if fused is not None:
            from .fused import trace_step

            w = self.learner.engine.wiring
            step_scale, step_bias = trace_step(
                self.trace,
                self.trace_bias,
                decay,
                fused[0],
                fused[1],
                w.pre,
                w.post,
                2.0 * self.learner.config.beta,
                delta,
            )
        else:
            step_scale = (delta[:, None] * self.trace).mean(axis=0)
            step_bias = (delta[:, None] * self.trace_bias).mean(axis=0)
        self.updates += 1
        raw_scale, raw_bias = step_scale, step_bias
        if (
            cfg.momentum > 0
        ):  # a running average of each seam's own steps, corrected for its short history
            self.velocity = cfg.momentum * self.velocity + (1 - cfg.momentum) * step_scale
            self.velocity_bias = cfg.momentum * self.velocity_bias + (1 - cfg.momentum) * step_bias
            correction = 1.0 - cfg.momentum**self.updates
            step_scale, step_bias = self.velocity / correction, self.velocity_bias / correction
        if (
            cfg.normalize > 0
        ):  # divided by the running RMS of each seam's own raw steps, corrected likewise
            rho = cfg.normalize
            self.second_moment = rho * self.second_moment + (1 - rho) * raw_scale**2
            self.second_moment_bias = rho * self.second_moment_bias + (1 - rho) * raw_bias**2
            correction = 1.0 - rho**self.updates
            step_scale = step_scale / (
                np.sqrt(self.second_moment / correction) + cfg.normalize_floor
            )
            step_bias = step_bias / (
                np.sqrt(self.second_moment_bias / correction) + cfg.normalize_floor
            )
        step_scale = cfg.eta * step_scale
        step_bias = cfg.eta_bias * step_bias
        report = self.learner.apply(step_scale, step_bias)
        if self.critic_net is not None:
            assert self._drive is not None
            self.critic_net.learn(self._drive, raw_target)
        else:
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
        self._drive = next_drive
        self._pending = None
        report["delta"] = float(np.abs(delta).mean())
        report["value"] = float(value.mean())
        report["free_steps"] = float(next_state.steps)
        return report

    def value_of(self, drive: np.ndarray) -> np.ndarray:
        """The critic's value of a drive, settled cold, without touching the cached state."""
        if self.critic_net is not None:
            return self.critic_net.value(drive)
        return self.value(self.learner.free(drive))

    def reset(self) -> None:
        self._free = None
        self._pending = None
        self.trace = self.trace_bias = self.trace_critic = None

    def parameters(self) -> int:
        if self.critic_net is not None:
            return self.learner.parameters() + self.critic_net.parameters()
        return self.learner.parameters() + len(self.w_critic) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "critic": [int(i) for i in self.critic_index],
            "updates": self.updates,
            "learner": self.learner.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RehearsalConfig:
    """A window of recent experience, rehearsed a few times."""

    gamma: float = 0.99
    lam: float = 0.95
    window: int = 64  # batched steps kept before a rehearsal
    epochs: int = 4  # passes over the window
    minibatch: int = 256  # rows per update
    clip: float = 0.2  # the ratio of new to old probability beyond which a row stops pushing
    eta: float = 1.0
    eta_bias: float = 0.02
    eta_critic: float = 1.0
    advantage_scale: bool = True  # advantages centred and scaled over the window
    normalize: float = 0.0  # >0: the adaptive local step (per-seam RMS)
    normalize_floor: float = 1e-8
    momentum: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class Rehearsal:
    """Learn from a window of recent experience, rehearsed: the local contrast as the score of
    each action taken, weighted by its advantage and stopped by the ratio clip.

    This is the data flow of proximal policy optimisation with the settlement's contrast in
    place of the backward pass: collect ``window`` steps from every environment, compute
    generalised advantages with the critic, then rehearse the window ``epochs`` times in
    minibatches. Every seam still moves on its own two endpoints and one broadcast number
    per row (the clipped, scaled advantage). The critic is a ``ValueNet`` trained on the
    same window toward the lambda-returns. Discrete actions (a softmax over the outputs) or
    ``Bins`` (one softmax per action dimension).
    """

    def __init__(
        self,
        learner: Learner,
        critic: ValueNet,
        config: RehearsalConfig | None = None,
        seed: int = 0,
        bins: Bins | None = None,
    ) -> None:
        self.learner = learner
        self.critic = critic
        self.config = config or RehearsalConfig()
        self.bins = bins
        self.rng = np.random.default_rng(seed)
        self.n = learner.engine.wiring.n
        self.group_id = bins.groups(learner.output_index, self.n) if bins is not None else None
        self.rows: list[dict[str, np.ndarray]] = []
        self._free: SettledState | None = None
        self._drive: np.ndarray | None = None
        self._pending: dict[str, np.ndarray] | None = None
        self.updates = 0
        self.rehearsals = 0
        self.second_moment = np.zeros(learner.engine.wiring.edges)
        self.second_moment_bias = np.zeros(self.n)
        self.velocity = np.zeros(learner.engine.wiring.edges)
        self.velocity_bias = np.zeros(self.n)

    # -- the policy

    def _probabilities(self, activation: np.ndarray) -> np.ndarray:
        s = activation[:, self.learner.output_index]
        if self.bins is not None:
            s = s.reshape(len(s), self.bins.dims, self.bins.size)
        z = s / self.learner.config.temperature
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        return np.asarray(p / p.sum(axis=-1, keepdims=True))

    def settle(self, drive: np.ndarray, warm: bool = True) -> SettledState:
        if self._free is not None and self._free.v.shape[0] != len(drive):
            self._free = None
        self._free = self.learner.free(drive, warm=self._free if warm else None)
        self._drive = drive
        return self._free

    def act(self, drive: np.ndarray, greedy: bool = False) -> np.ndarray:
        free = (
            self._free
            if self._free is not None and self._free.v.shape[0] == len(drive)
            else self.settle(drive)
        )
        self._drive = drive
        p = self._probabilities(free.activation)
        if greedy:
            choice = np.argmax(p, axis=-1)
        else:
            u = self.rng.random(p.shape[:-1])
            choice = np.minimum((p.cumsum(axis=-1) < u[..., None]).sum(axis=-1), p.shape[-1] - 1)
        taken = np.take_along_axis(p, choice[..., None], axis=-1)[..., 0]
        logp = np.log(np.maximum(taken, 1e-12))
        if logp.ndim == 2:
            logp = logp.sum(axis=1)
        if not greedy:
            self._pending = {
                "drive": drive,
                "choice": choice,
                "logp": logp,
                "value": self.critic.value(drive),
            }
        if self.bins is not None:
            return np.asarray(self.bins.centres[choice])
        return np.asarray(choice, dtype=np.int64)

    # -- the window

    def learn(
        self,
        reward: np.ndarray,
        done: np.ndarray,
        next_drive: np.ndarray,
        bootstrap: np.ndarray | None = None,
    ) -> dict[str, float] | None:
        """Keep the step; when the window is full, rehearse it and return the report."""
        cfg = self.config
        assert self._pending is not None
        row = dict(self._pending)
        row["reward"] = np.asarray(reward, dtype=float)
        row["done"] = np.asarray(done, dtype=bool)
        row["bootstrap"] = (
            np.zeros(len(reward)) if bootstrap is None else np.asarray(bootstrap, dtype=float)
        )
        self.rows.append(row)
        self._pending = None
        # the next state, warm; finished rows start their next life from rest
        next_state = self.learner.free(next_drive, warm=self._free)
        if row["done"].any():
            v = next_state.v.copy()
            a = next_state.adaptation.copy()
            v[row["done"]] = 0.0
            a[row["done"]] = 0.0
            next_state = self.learner.free(
                next_drive, warm=SettledState(v, next_state.activation, a, next_state.steps)
            )
        self._free = next_state
        self._drive = next_drive
        if len(self.rows) < cfg.window:
            return None
        return self._rehearse(next_drive)

    def _rehearse(self, last_drive: np.ndarray) -> dict[str, float]:
        cfg = self.config
        rows = self.rows
        self.rows = []
        T, B = len(rows), len(rows[0]["reward"])
        values = np.stack([r["value"] for r in rows])
        rewards = np.stack([r["reward"] for r in rows])
        dones = np.stack([r["done"] for r in rows])
        boots = np.stack([r["bootstrap"] for r in rows])
        next_value = self.critic.value(last_drive)
        adv = np.zeros((T, B))
        last = np.zeros(B)
        for t in reversed(range(T)):
            v_next = next_value if t == T - 1 else values[t + 1]
            v_next = np.where(dones[t], boots[t], v_next)
            delta = rewards[t] + cfg.gamma * v_next - values[t]
            last = delta + cfg.gamma * cfg.lam * np.where(dones[t], 0.0, last)
            adv[t] = last
        returns = adv + values
        if cfg.advantage_scale:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        drive = np.concatenate([r["drive"] for r in rows])
        choice = np.concatenate([r["choice"] for r in rows])
        logp_old = np.concatenate([r["logp"] for r in rows])
        adv = adv.reshape(-1)  # type: ignore[assignment]
        returns = returns.reshape(-1)
        n_rows = len(drive)
        index = np.arange(n_rows)
        clipped = 0
        for _ in range(cfg.epochs):
            self.rng.shuffle(index)
            for start in range(0, n_rows, cfg.minibatch):
                idx = index[start : start + cfg.minibatch]
                free = self.learner.free(drive[idx])
                p = self._probabilities(free.activation)
                taken = np.take_along_axis(p, choice[idx][..., None], axis=-1)[..., 0]
                logp = np.log(np.maximum(taken, 1e-12))
                if logp.ndim == 2:
                    logp = logp.sum(axis=1)
                ratio = np.exp(logp - logp_old[idx])
                a = adv[idx]
                # the clipped objective's gradient: a row pushes only while its ratio is inside
                # the clip, or pushes back
                push = np.where(
                    ((ratio > 1 + cfg.clip) & (a > 0)) | ((ratio < 1 - cfg.clip) & (a < 0)),
                    0.0,
                    1.0,
                )
                clipped += int((push == 0).sum())
                weight = a * ratio * push
                target = self._targets(choice[idx])
                plus = self._nudged(drive[idx], free, target, self.learner.config.beta, weight)
                minus = self._nudged(drive[idx], free, target, -self.learner.config.beta, weight)
                overlap_term, owner_term = self.learner.contrast(free, plus, minus)
                self._apply(overlap_term, owner_term)
                self.critic.learn(drive[idx], returns[idx])
        self.rehearsals += 1
        return {
            "rows": float(n_rows),
            "clipped": float(clipped / max(n_rows * cfg.epochs, 1)),
            "advantage": float(np.abs(adv).mean()),
            "value": float(values.mean()),
        }

    def _targets(self, choice: np.ndarray) -> np.ndarray:
        target = np.zeros((len(choice), self.n))
        out = self.learner.output_index
        if self.bins is not None:
            onehot = np.zeros((len(choice), self.bins.dims, self.bins.size))
            np.put_along_axis(onehot, choice[:, :, None], 1.0, axis=2)
            target[:, out] = onehot.reshape(len(choice), -1)
        else:
            target[np.arange(len(choice)), out[choice]] = 1.0
        return target

    def _nudged(
        self,
        drive: np.ndarray,
        free: SettledState,
        target: np.ndarray,
        beta: float,
        weight: np.ndarray,
    ) -> SettledState:
        cfg = self.learner.config
        nudge = Nudge(
            target,
            self.learner.output_mask,
            beta,
            softmax_temperature=cfg.temperature,
            weight=weight,
            groups=self.group_id,
        )
        return self.learner.engine.settle_batch(
            drive, steps=cfg.nudged_steps, state=free, nudge=nudge, tolerance=cfg.tolerance
        )

    def _apply(self, overlap_term: np.ndarray, owner_term: np.ndarray) -> None:
        cfg = self.config
        self.updates += 1
        step_scale, step_bias = overlap_term, owner_term
        if cfg.momentum > 0:
            self.velocity = cfg.momentum * self.velocity + (1 - cfg.momentum) * step_scale
            self.velocity_bias = cfg.momentum * self.velocity_bias + (1 - cfg.momentum) * step_bias
            c = 1.0 - cfg.momentum**self.updates
            step_scale, step_bias = self.velocity / c, self.velocity_bias / c
        if cfg.normalize > 0:
            rho = cfg.normalize
            self.second_moment = rho * self.second_moment + (1 - rho) * overlap_term**2
            self.second_moment_bias = rho * self.second_moment_bias + (1 - rho) * owner_term**2
            c = 1.0 - rho**self.updates
            step_scale = step_scale / (np.sqrt(self.second_moment / c) + cfg.normalize_floor)
            step_bias = step_bias / (np.sqrt(self.second_moment_bias / c) + cfg.normalize_floor)
        self.learner.apply(cfg.eta * step_scale, cfg.eta_bias * step_bias)

    def reset(self) -> None:
        self._free = None
        self._pending = None

    def parameters(self) -> int:
        return self.learner.parameters() + self.critic.parameters()
