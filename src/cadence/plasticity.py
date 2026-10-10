"""Three factors: a trace at every synapse, a critic, and a dopamine neuron.

Learning from reward in a body is not a nudge at the end of a rollout. Every
synapse keeps an eligibility trace, the recent history of what it would have
moved by for the actions that were taken; a critic population reads the
settled state and learns to predict return; and a dopamine neuron broadcasts
the temporal-difference error of that prediction. The three multiply, at
every synapse, every step:

    e[e]     <- gamma * lam * e[e] + contrast[e]        (the action's eligibility)
    delta    =  r + gamma * V(s') - V(s)                 (the dopamine signal)
    scale[e] += eta * delta * e[e]

There is no buffer and no epoch: one life, one pass. The contrast is the
free/nudged difference the learner already computes, per row, with the
nudge's target the action that was taken, so the eligibility is the score of
that action and the goal still enters only through the nudge. The critic is a
linear reading of the neurons named for it, trained by its own trace and the
prediction error selected by ``critic_signal``. Use ``"td"`` to fit return in
reward units; ``"modulated"`` retains the bounded legacy critic update.
``normalize`` divides each synapse's step by the running RMS of its own
steps, the local counterpart of an adaptive optimiser.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .brain import _FUSED as _FUSED_TRACE
from .brain import Brain, BrainState, Nudge
from .learning import Learner

__all__ = [
    "ActorCritic",
    "ActorCriticConfig",
    "Bins",
]


def _batch_of(state: BrainState) -> int:
    """The rows of a settled state, read from the device shape when the state rests there."""
    handle = state.device
    if handle is not None and "s" in handle and state.__dict__.get("v") is None:
        return int(handle["s"].shape[0])
    return int(np.atleast_2d(state.v).shape[0])


@dataclass(frozen=True)
class Bins:
    """A continuous action as one softmax choice per dimension over ``size`` levels.

    Each dimension owns ``size`` output neurons standing for levels in ``[-1, 1]``; an
    action is a draw from the softmax over each dimension's neurons (a discrete choice
    per dimension), and learning is the cross-entropy nudge of that group toward the
    level taken, the same rule that learns a discrete action. Discretised actions match
    Gaussian ones on continuous control (Tang and Agrawal 2020); here they let one rule
    serve every kind of action.
    """

    dims: int
    size: int = 9

    def __post_init__(self) -> None:
        for name, minimum in (("dims", 1), ("size", 2)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")

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


@dataclass
class Valence:
    """The reward less its expectation, made into the signal that moves the synapses: the dopamine.

    ``delta`` is what came less what was expected (a critic's prediction error, or the
    reward alone). The valence takes it less its running level per stream (``level`` is the
    forgetting factor of that level, 0 for no centring), in the reward's own units
    (``units``) or over its running scale; nothing within ``floor`` scales of the level
    (quiet while the reward is what it usually is); capped at ``cap`` (0 for no cap).
    """

    level: float = 0.0
    floor: float = 0.0
    cap: float = 1.0
    units: bool = True
    per_stream: bool = True
    mean: Any = 0.0
    var: Any = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.level < 1:
            raise ValueError("level must lie in [0, 1)")
        if not np.isfinite([self.floor, self.cap]).all() or min(self.floor, self.cap) < 0:
            raise ValueError("floor and cap must be finite and nonnegative")

    def __call__(self, delta: np.ndarray, *, observed: np.ndarray | None = None) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if delta.ndim != 1 or not delta.size or not np.isfinite(delta).all():
            raise ValueError("delta must be a nonempty finite vector, one value per stream")
        known = np.ones(len(delta), dtype=bool) if observed is None else np.asarray(observed)
        if known.shape != delta.shape or known.dtype != np.bool_:
            raise ValueError("observed must be a boolean vector matching delta")
        if not known.any():
            return np.zeros_like(delta)
        if self.level > 0:
            rho = self.level
            mean, var = self.mean, self.var
            if self.per_stream:
                if not isinstance(mean, np.ndarray) or len(mean) != len(delta):
                    mean, var = np.zeros(len(delta)), np.zeros(len(delta))
                mean = np.where(known, rho * mean + (1 - rho) * delta, mean)
                var = np.where(known, rho * var + (1 - rho) * (delta - mean) ** 2, var)
            else:
                mean = rho * mean + (1 - rho) * float(delta[known].mean())
                var = rho * var + (1 - rho) * float(((delta[known] - mean) ** 2).mean())
            if not np.isfinite(mean).all() or not np.isfinite(var).all():
                raise ValueError("valence moments must remain finite")
            self.mean, self.var = mean, var
            scale = np.sqrt(self.var) + 1e-6
            centred = delta - self.mean if self.units else (delta - self.mean) / scale
            if self.floor > 0:
                within = np.abs(centred) < self.floor * (scale if self.units else 1.0)
                centred = np.where(within, 0.0, centred)
            delta = centred
        if self.cap > 0:
            delta = np.clip(delta, -self.cap, self.cap)
        out: np.ndarray = np.asarray(np.where(known, delta, 0.0), dtype=float)
        return out

    def reset(self) -> None:
        self.mean, self.var = 0.0, 1.0


SATURATION_BAND = 0.02  # an output within this of 0 or 1 has no slope left for a nudge


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    gamma: float = 0.99  # discount
    lam: float = 0.9  # trace decay of the actor's eligibility
    eta: float = 0.5  # actor step per unit dopamine per unit trace
    eta_bias: float | None = None  # bias step; None derives eta / 10 at construction (issue 143)
    eta_critic: float = 0.05
    normalize: float = 0.0  # >0: forgetting factor of the per-synapse RMS that divides its step
    # >0: each synapse steps on a running average of its own steps, so sign noise cancels before
    # the RMS divides it
    momentum: float = 0.0
    dopamine_cap: float = 1.0  # actor modulation saturates here (0: no cap)
    # >0: forgetting factor of a running mean and scale of delta; the phasic signal is the
    # deviation from the tonic level
    dopamine_center: float = 0.0
    # >0: the centred signal within this many scales of its mean is nothing; the dopamine is
    # quiet while the reward is what it usually is and speaks only for a surprise
    dopamine_floor: float = 0.0
    # the centred signal divided by its running scale (a unit signal whatever the reward's
    # size) or left in the reward's own units, so a stage cleared is ten coins, not one
    center_scale: bool = True
    critic_normalize: bool = (
        True  # the critic's step is divided by its trace's energy, so its step size is scale-free
    )

    # Legacy modulation can stabilize control, but does not generally fit mean return.
    # "auto": the raw error ("td") whenever the dopamine is centred, the modulated one otherwise;
    # a critic fed the centred signal chases a moving target and its value runs away.
    critic_signal: Literal["modulated", "td", "auto"] = "auto"
    # Reward credit uses finite nudged phases, independently of a supervised
    # learner's qualification budget. None keeps standalone learner inheritance.
    eligibility_steps: int | None = None

    @property
    def critic_target(self) -> str:
        """The signal the critic learns from: ``"td"`` or ``"modulated"``, ``"auto"`` resolved."""
        if self.critic_signal == "auto":
            return "td" if self.dopamine_center > 0 else "modulated"
        return self.critic_signal

    def __post_init__(self) -> None:
        if self.critic_signal not in ("modulated", "td", "auto"):
            raise ValueError("critic_signal must be modulated, td or auto")
        # Use the same scalar arithmetic before and after JSON checkpointing.
        for name in self.__slots__:
            value = getattr(self, name)
            if isinstance(value, np.integer):
                object.__setattr__(self, name, int(value))
            elif isinstance(value, np.floating):
                object.__setattr__(self, name, float(value))
        if self.eligibility_steps is not None and (
            isinstance(self.eligibility_steps, (bool, np.bool_))
            or not isinstance(self.eligibility_steps, (int, np.integer))
            or self.eligibility_steps < 0
        ):
            raise ValueError("eligibility_steps must be a nonnegative integer, or None")
        if self.eta_bias is None:
            # The bias rate follows the synapse rate unless chosen: issue 143, the rule of
            # issue 126 for the actor. Under RMS normalization both are absolute steps.
            if not np.isfinite(self.eta) or self.eta < 0:
                raise ValueError("eta must be finite and nonnegative")
            object.__setattr__(self, "eta_bias", self.eta / 10.0)
        eta_bias = self.eta_bias
        assert eta_bias is not None
        for name in ("gamma", "lam"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("normalize", "momentum", "dopamine_center"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in [0, 1)")
        for name in ("eta", "eta_bias", "eta_critic", "dopamine_cap", "dopamine_floor"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.normalize > 0 and max(self.eta, eta_bias) > 0.05:
            warnings.warn(
                "ActorCriticConfig with normalize > 0 and eta or eta_bias above 0.05: "
                "RMS normalization can make parameter steps much larger than raw contrasts "
                "suggest and may saturate the policy. Consider smaller actor rates; "
                "see docs/reward.md#rates-under-normalization.",
                RuntimeWarning,
                stacklevel=3,
            )
        if self.eta > 0 and eta_bias > self.eta:
            warnings.warn(
                f"ActorCriticConfig with eta_bias ({eta_bias:g}) above eta ({self.eta:g}): "
                "the actor's bias step dominates its synapse step and the greedy policy tends "
                "to one action whatever the observation. The default couples eta_bias to "
                "eta / 10; when lowering eta through dataclasses.replace, set eta_bias too, or "
                "pass eta_bias=None to re-derive it. See docs/reward.md#rates-under-normalization.",
                RuntimeWarning,
                stacklevel=3,
            )

    def to_dict(self) -> dict[str, Any]:
        values = {k: getattr(self, k) for k in self.__slots__}
        if self.eligibility_steps is not None:
            values["eligibility_steps"] = int(self.eligibility_steps)
        return values


class ActorCritic:
    """A learner that acts, keeps eligibility, and learns from dopamine.

    Use::

        ac = ActorCritic(learner, critic=connectome.populations["hidden"])
        action = ac.act(drive)                 # settle, sample, keep eligibility
        obs, reward, done = env.step(action)
        ac.learn(reward, done, next_drive)     # settle the next state, dopamine, update

    ``learn`` settles the next state for bootstrapping before changing weights.
    ``act`` refreshes that warm state under the updated weights before sampling.
    """

    def __init__(
        self,
        learner: Learner,
        critic: Sequence[int],
        config: ActorCriticConfig | None = None,
        seed: int = 0,
        population: Bins | None = None,
    ) -> None:
        self.learner = learner
        self.config = config or ActorCriticConfig()
        if (
            population is not None
            and len(learner.output_index) != population.dims * population.size
        ):
            raise ValueError("the output set must hold dims * size neurons for a population code")
        self.bins = population
        if self.bins is not None:
            self.group_id: np.ndarray | None = self.bins.groups(
                learner.output_index, learner.brain.connectome.n
            )
        else:
            self.group_id = learner.output_groups
        index = np.asarray(list(critic))
        if (
            index.ndim != 1
            or not index.size
            or not np.issubdtype(index.dtype, np.integer)
            or (index < 0).any()
            or (index >= learner.brain.connectome.n).any()
            or np.unique(index).size != index.size
        ):
            raise ValueError(
                "critic must contain distinct integer neuron indices in the connectome"
            )
        self.critic_index = index.astype(np.int64)
        self.w_critic = np.zeros(len(self.critic_index))
        self.b_critic = 0.0
        self.rng = np.random.default_rng(seed)
        w = learner.brain.connectome
        self.synapses, self.n = w.synapses, w.n
        self.trace: np.ndarray | None = None
        self.trace_bias: np.ndarray | None = None
        self.trace_critic: np.ndarray | None = None
        # one stream's traces on the torch device, when it settles there
        self._trace_device: Any = None
        self.second_moment = np.zeros(self.synapses)
        self.second_moment_bias = np.zeros(self.n)
        self.velocity = np.zeros(self.synapses)
        self.velocity_bias = np.zeros(self.n)
        self._valence: Valence | None = None
        # (batch, neurons): each synapse's eligibility is weighted by its pre neuron's salience,
        # when
        # set before ``learn`` (a ``Trace.ringing``: what is still ringing is what gets written)
        self.salience: np.ndarray | None = None
        self._drive: np.ndarray | None = None
        self._free: BrainState | None = None
        self._free_brain: Brain | None = None  # parameters that produced the cached phase
        # ("phases", plus, minus, value) as activations, or ("states", plus, minus, value) as states
        self._pending: tuple[str, Any, Any, np.ndarray] | None = None
        self.updates = 0

    # -- readings

    @property
    def state(self) -> BrainState | None:
        """The free phase of the latest moment: the state ``act`` read, or ``learn`` settled."""
        return self._free

    def value(self, state: BrainState) -> np.ndarray:
        return np.asarray(state.activation[:, self.critic_index] @ self.w_critic + self.b_critic)

    def probabilities(
        self, state: BrainState, temperature: float | Sequence[float] | np.ndarray | None = None
    ) -> np.ndarray:
        """Action probabilities, normalized separately for each motor slot.

        Multiple categorical slots return ``(batch, slots, max_size)`` with
        zero padding for shorter slots. Bins keeps its uniform population shape.
        ``temperature`` reads the same settled outputs at another softmax
        temperature, or one temperature per motor slot; left unset it is the learner's.
        """
        temperature = self._temperature(temperature)
        s = state.activation[:, self.learner.output_index]
        if self.bins is not None:
            s = s.reshape(len(s), self.bins.dims, self.bins.size)
            if isinstance(temperature, np.ndarray):
                temperature = temperature[None, :, None]
        elif self.learner.slot_count > 1:
            p = np.zeros((len(s), self.learner.slot_count, int(self.learner.slot_sizes.max())))
            for j, (start, size) in enumerate(
                zip(self.learner.slot_offsets, self.learner.slot_sizes, strict=True)
            ):
                slot_temperature = (
                    temperature[j] if isinstance(temperature, np.ndarray) else temperature
                )
                p[:, j, :size] = self._softmax(s[:, start : start + size], slot_temperature)
            return p
        elif isinstance(temperature, np.ndarray):
            temperature = float(temperature[0])
        return self._softmax(s, temperature)

    @staticmethod
    def _softmax(s: np.ndarray, temperature: float | np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z = s / temperature
            if not np.isfinite(z).all():
                # At a subnormal temperature, subtract before dividing: negative infinity
                # has zero probability, while each largest logit stays exactly zero.
                # Float32 inputs must not round the positive denominator down to zero.
                logits = np.asarray(s, dtype=float)
                z = (logits - logits.max(axis=-1, keepdims=True)) / temperature
            else:
                z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        return np.asarray(p / p.sum(axis=-1, keepdims=True))

    def _temperature(
        self, temperature: float | Sequence[float] | np.ndarray | None
    ) -> float | np.ndarray:
        """A scalar or one temperature per motor slot, validated before state changes."""
        if temperature is None:
            return float(self.learner.config.temperature)
        if isinstance(temperature, (Sequence, np.ndarray)) and not isinstance(temperature, str):
            values = np.asarray(temperature)
            slots = self.bins.dims if self.bins is not None else self.learner.slot_count
            if values.shape != (slots,) or values.dtype.kind not in "iuf":
                raise ValueError("temperature must be finite and positive, one per motor slot")
            with np.errstate(over="ignore", under="ignore"):
                values = values.astype(float, copy=True)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("temperature must be finite and positive, one per motor slot")
            return values
        if (
            isinstance(temperature, (bool, np.bool_))
            or not isinstance(temperature, (int, float, np.integer, np.floating))
            or not np.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive")
        converted = float(temperature)
        if not np.isfinite(converted) or converted <= 0:
            raise ValueError("temperature must remain finite and positive in float64")
        return converted

    def settle(self, drive: np.ndarray) -> BrainState:
        """The free phase for ``drive``, warm from the last one; cached for ``act``."""
        drive = self._validated_drive(drive)
        if self._free is not None and _batch_of(self._free) != len(drive):
            self.reset()
        self._free = self.learner.free(drive, warm=self._free)
        self._free_brain = self.learner.brain
        self._drive = drive.copy()
        self._pending = None
        return self._free

    def _validated_drive(self, drive: np.ndarray) -> np.ndarray:
        drive = np.asarray(drive, dtype=float)
        if (
            drive.ndim != 2
            or drive.shape[1] != self.n
            or not len(drive)
            or not np.isfinite(drive).all()
        ):
            raise ValueError(f"drive must be a nonempty finite (batch, {self.n}) array")
        return drive

    # -- acting

    def act(
        self,
        drive: np.ndarray,
        greedy: bool = False,
        temperature: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Settle (or reuse the cached free phase), sample an action per row, keep its eligibility.

        Discrete: an action index per row, or one index per categorical motor slot.
        With ``Bins``: one softmax draw per dimension of a population code.
        ``temperature`` samples the same settled outputs at another softmax
        temperature, or one per motor slot. Eligibility uses those same temperatures
        for the sampled action's policy score. A greedy read ignores them.
        """
        drive = self._validated_drive(drive)
        if temperature is not None:
            temperature = self._temperature(temperature)  # before any state changes
        free = (
            self._free
            if self._free is not None
            and self._drive is not None
            and self._free_brain is self.learner.brain
            and np.array_equal(self._drive, drive)
            else self.settle(drive)
        )
        self._pending = None  # eligibility always belongs to this decision, including greedy reads
        if self.bins is not None:
            return self._act_bins(drive, free, greedy, temperature)
        p = self.probabilities(free, None if greedy else temperature)
        if greedy:
            action = np.argmax(p, axis=-1)
        else:
            u = self.rng.random(p.shape[:-1])
            action = (p.cumsum(axis=-1) < u[..., None]).sum(axis=-1)
            limit = self.learner.slot_sizes - 1 if p.ndim == 3 else p.shape[-1] - 1
            action = np.minimum(action, limit)
        if not greedy:
            target = self.learner.targets(action)
            # Credit differentiates the policy that actually sampled the action.
            # A learner may use quadratic imitation; its loss must not silently
            # replace the categorical log-policy score in reward eligibility.
            beta = self.learner.config.beta
            plus = self._nudged_groups(drive, free, target, beta, temperature)
            minus = self._nudged_groups(drive, free, target, -beta, temperature)
            self._pending = ("states", plus, minus, self.value(free))
        return np.asarray(action, dtype=np.int64)

    def _act_bins(
        self,
        drive: np.ndarray,
        free: BrainState,
        greedy: bool,
        temperature: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """One softmax draw per dimension; the taken levels' one-hots are the nudge's target,
        per group."""
        bins = self.bins
        assert bins is not None
        cfg = self.learner.config
        batch = len(drive)
        p = self.probabilities(free, None if greedy else temperature)
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
            plus = self._nudged_groups(drive, free, target, cfg.beta, temperature)
            minus = self._nudged_groups(drive, free, target, -cfg.beta, temperature)
            self._pending = ("states", plus, minus, self.value(free))
        out: np.ndarray = np.asarray(action, dtype=float)
        return out

    def _nudged_groups(
        self,
        drive: np.ndarray,
        free: BrainState,
        target: np.ndarray,
        beta: float,
        temperature: float | Sequence[float] | np.ndarray | None = None,
    ) -> BrainState:
        cfg = self.learner.config
        actual = self._temperature(temperature)
        # One common scale times the joint log-policy score. The bounded mask
        # keeps colder, even subnormal, temperatures representable; the base
        # policy retains its existing unit mask and learning rate.
        scale = min(float(cfg.temperature), float(np.min(actual)))
        if isinstance(actual, np.ndarray):
            sizes = (
                self.learner.slot_sizes
                if self.bins is None
                else np.full(self.bins.dims, self.bins.size)
            )
            expanded = np.full(self.n, cfg.temperature)
            expanded[self.learner.output_index] = np.repeat(actual, sizes)
            actual = expanded
        with np.errstate(under="ignore"):
            mask = self.learner.output_mask * (scale / actual)
        nudge = Nudge(
            target,
            mask,
            beta,
            softmax_temperature=actual,
            groups=self.group_id,
        )
        return self.learner.brain.settle_batch(
            drive,
            steps=cfg.nudged_steps
            if self.config.eligibility_steps is None
            else self.config.eligibility_steps,
            state=free,
            nudge=nudge,
            tolerance=cfg.tolerance,
        )

    @property
    def valence(self) -> Valence:
        """The dopamine as a Valence built from the config; its running level is the agent's."""
        v = self._valence
        if v is None:
            cfg = self.config
            v = self._valence = Valence(
                level=cfg.dopamine_center,
                floor=cfg.dopamine_floor,
                cap=0.0,  # the cap is applied by learn, after the centring
                units=not cfg.center_scale,
                per_stream=True,
            )
        return v

    @property
    def delta_mean(self) -> Any:
        return self.valence.mean

    @delta_mean.setter
    def delta_mean(self, value: Any) -> None:
        self.valence.mean = value

    @property
    def delta_var(self) -> Any:
        return self.valence.var

    @delta_var.setter
    def delta_var(self, value: Any) -> None:
        self.valence.var = value

    def _centre(self, delta: np.ndarray, observed: np.ndarray | None = None) -> np.ndarray:
        """The phasic signal through the Valence (the cap is learn's, applied after)."""
        return self.valence(delta, observed=observed)

    def learn(
        self,
        reward: np.ndarray,
        done: np.ndarray,
        next_drive: np.ndarray,
        bootstrap: np.ndarray | None = None,
        *,
        observed: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Dopamine from the reward and the next state's value; every synapse moves on its trace.

        ``done`` rows start their next life from rest and, unless ``bootstrap`` gives them
        a value, bootstrap from zero. A time limit is not a terminal state: pass the
        value of the last observation (``value_of``) as ``bootstrap`` for a truncated row.
        ``observed`` excludes padding rows from plasticity and reward statistics;
        their traces reset. At least one real transition is required. Updates are
        averaged over the observed rows, independent of padding.
        """
        arrays = ("trace", "trace_bias", "trace_critic")
        saved = {
            name: None if getattr(self, name) is None else getattr(self, name).copy()
            for name in arrays
        }
        saved.update(
            {
                name: getattr(self, name)
                for name in (
                    "_trace_device",
                    "velocity",
                    "velocity_bias",
                    "second_moment",
                    "second_moment_bias",
                    "updates",
                    "_valence",
                )
            }
        )
        valence = None if self._valence is None else (self._valence.mean, self._valence.var)
        try:
            return self._learn(reward, done, next_drive, bootstrap, observed=observed)
        except Exception:
            self.__dict__.update(saved)
            if valence is not None:
                assert self._valence is not None
                self._valence.mean, self._valence.var = valence
            raise

    def _learn(
        self,
        reward: np.ndarray,
        done: np.ndarray,
        next_drive: np.ndarray,
        bootstrap: np.ndarray | None = None,
        *,
        observed: np.ndarray | None = None,
    ) -> dict[str, float]:
        reward, done, next_drive, bootstrap = self._validated_transition(
            reward, done, next_drive, bootstrap
        )
        observed = np.ones(len(reward), dtype=bool) if observed is None else np.asarray(observed)
        if observed.shape != reward.shape or observed.dtype != np.bool_ or not observed.any():
            raise ValueError("observed must match the batch and include a real transition")
        done = done | ~observed
        cfg = self.config
        assert self._pending is not None
        kind, first, second, value = self._pending
        free = self._free
        assert free is not None
        batch = len(reward)
        decay = cfg.gamma * cfg.lam
        device_kernel = None
        if kind == "states":  # the two phases as states: on the device when they settled there
            if cfg.momentum == 0 and cfg.normalize == 0:
                device_kernel = self.learner._device_kernel(first, second)
            if device_kernel is None:
                first, second = first.activation, second.activation
                kind = "phases"
        if device_kernel is not None:
            return self._learn_device(
                device_kernel, first, second, value, reward, done, next_drive, bootstrap, observed
            )
        if self._trace_device is not None:
            self.trace, self.trace_bias = (
                tensor.detach().cpu().double().numpy().copy() for tensor in self._trace_device
            )
            self._trace_device = None
        if self.trace is None or self.trace.shape[0] != batch:
            self.trace = np.zeros((batch, self.synapses))
            self.trace_bias = np.zeros((batch, self.n))
            self.trace_critic = np.zeros((batch, len(self.critic_index) + 1))
        assert self.trace_bias is not None and self.trace_critic is not None
        fused = None
        if kind == "phases" and _FUSED_TRACE and self.salience is None:
            fused = (
                first,
                second,
            )  # the contrast, trace, and dopamine-weighted sum are one fused pass in learn
        else:
            if kind == "phases":
                w = self.learner.brain.connectome
                span = 2.0 * self.learner.config.beta
                a_plus, a_minus = first[:, w.pre], second[:, w.pre]
                b_plus, b_minus = first[:, w.post], second[:, w.post]
                contrast = (a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus) / span
                contrast_bias = (first - second) / span
            else:
                contrast, contrast_bias = first, second
            if self.salience is not None:
                contrast = contrast * self.salience[:, self.learner.brain.connectome.pre]
            self.trace *= decay
            self.trace += contrast
            self.trace_bias *= decay
            self.trace_bias += contrast_bias
        self.trace_critic *= cfg.gamma * cfg.lam
        self.trace_critic[:, :-1] += free.activation[:, self.critic_index]
        self.trace_critic[:, -1] += 1.0
        # the next state, warm from this one; a finished row starts its next life from rest
        next_state = self._next_state(next_drive, done)
        next_value_raw = self.value(next_state)
        next_value = np.where(
            done, 0.0 if bootstrap is None else np.asarray(bootstrap, dtype=float), next_value_raw
        )
        raw_target = reward + cfg.gamma * next_value
        td_error = raw_target - value
        if not np.isfinite(td_error).all():
            raise ValueError("TD errors must remain finite")
        delta = td_error
        if cfg.dopamine_center > 0:
            delta = self._centre(delta, observed)
        capped = self._capped(delta, observed)
        if cfg.dopamine_cap > 0:
            delta = np.clip(delta, -cfg.dopamine_cap, cfg.dopamine_cap)
        delta = np.where(observed, delta / observed.mean(), 0.0)
        # three factors
        if fused is not None:
            from .fused import trace_step

            w = self.learner.brain.connectome
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
        ):  # a running average of each synapse's own steps, corrected for its short history
            self.velocity = cfg.momentum * self.velocity + (1 - cfg.momentum) * step_scale
            self.velocity_bias = cfg.momentum * self.velocity_bias + (1 - cfg.momentum) * step_bias
            correction = 1.0 - cfg.momentum**self.updates
            step_scale, step_bias = self.velocity / correction, self.velocity_bias / correction
        if (
            cfg.normalize > 0
        ):  # divided by the running RMS of each synapse's own raw steps, corrected likewise
            rho = cfg.normalize
            self.second_moment = rho * self.second_moment + (1 - rho) * raw_scale**2
            self.second_moment_bias = rho * self.second_moment_bias + (1 - rho) * raw_bias**2
            correction = 1.0 - rho**self.updates
            step_scale = step_scale / (np.sqrt(self.second_moment / correction) + 1e-3)
            step_bias = step_bias / (np.sqrt(self.second_moment_bias / correction) + 1e-3)
        step_scale = cfg.eta * step_scale
        step_bias = cfg.eta_bias * step_bias
        if not all(
            np.isfinite(getattr(self, name)).all()
            for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
        ):
            raise ValueError("optimizer moments must remain finite")
        critic_weights, critic_bias = self._critic_candidate(td_error, delta, observed)
        next_brain = self.learner.brain
        report = self.learner.apply(step_scale, step_bias, activation=free.activation)
        self.w_critic, self.b_critic = critic_weights, critic_bias
        # a finished row forgets its traces
        if done.any():
            self.trace[done] = 0.0
            self.trace_bias[done] = 0.0
            self.trace_critic[done] = 0.0
        self._free = next_state
        self._free_brain = next_brain
        self._drive = next_drive.copy()
        self._pending = None
        report["delta"] = float(np.abs(delta).mean())
        report["dopamine"] = float(delta.mean())
        report["td_error"] = float(np.abs(td_error[observed]).mean())
        report["capped"] = capped
        report["value"] = float(value[observed].mean())
        report["free_steps"] = float(next_state.steps)
        report["saturation"] = self._saturation(free)
        plastic = self.learner.plastic_synapses
        assert plastic is not None
        traces = self.trace if plastic.all() else self.trace[:, plastic]
        report["trace"] = float(np.abs(traces).mean())
        return report

    def _critic_candidate(
        self, td_error: np.ndarray, delta: np.ndarray, observed: np.ndarray
    ) -> tuple[np.ndarray, float]:
        cfg = self.config
        critic_trace = self.trace_critic
        assert critic_trace is not None
        if cfg.critic_normalize:
            critic_trace = critic_trace / (1.0 + (critic_trace**2).sum(axis=1, keepdims=True))
        # Calibrated TD retains reward units; legacy mode uses the same bounded
        # signal as the actor. Modulation can change the critic's fixed point.
        critic_delta = (
            np.where(observed, td_error / observed.mean(), 0.0)
            if cfg.critic_target == "td"
            else delta
        )
        critic_step = cfg.eta_critic * (critic_delta[:, None] * critic_trace).mean(axis=0)
        weights = self.w_critic + critic_step[:-1]
        bias = self.b_critic + float(critic_step[-1])
        if not np.isfinite(weights).all() or not np.isfinite(bias):
            raise ValueError("critic parameters must remain finite")
        return weights, bias

    def _capped(self, delta: np.ndarray, observed: np.ndarray) -> float:
        """The share of observed rows whose dopamine exceeded ``dopamine_cap`` before the clip.
        Near one, the actor learns from the sign of each outcome alone: a reward of ten and
        a reward of one move every synapse by the same amount, and a trace's sign noise is
        not averaged out by small outcomes. Zero without a cap."""
        cap = self.config.dopamine_cap
        if cap <= 0:
            return 0.0
        return float(np.mean(np.abs(np.asarray(delta)[observed]) > cap))

    def _saturation(self, free: BrainState) -> float:
        """The fraction of output activations within ``SATURATION_BAND`` of 0 or 1, where the
        activation has no slope and a nudge cannot move it: a latch shows here before it shows
        in the reward."""
        s = np.asarray(free.activation)[:, self.learner.output_index]
        return float(((s < SATURATION_BAND) | (s > 1.0 - SATURATION_BAND)).mean())

    def _validated_transition(
        self,
        reward: np.ndarray,
        done: np.ndarray,
        next_drive: np.ndarray,
        bootstrap: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Reject a malformed transition before any trace, critic or parameter changes."""
        if self._pending is None:
            raise RuntimeError("learn needs a non-greedy act first")
        batch = len(self._pending[3])
        reward, done = np.asarray(reward, dtype=float), np.asarray(done)
        if reward.shape != (batch,) or not np.isfinite(reward).all():
            raise ValueError("reward must be a finite vector matching the action batch")
        if done.shape != (batch,) or done.dtype != np.bool_:
            raise ValueError("done must be a boolean vector matching the action batch")
        next_drive = self._validated_drive(next_drive)
        if len(next_drive) != batch:
            raise ValueError("next_drive must match the action batch")
        if bootstrap is not None:
            bootstrap = np.asarray(bootstrap, dtype=float)
            if bootstrap.shape != (batch,) or not np.isfinite(bootstrap).all():
                raise ValueError("bootstrap must be a finite vector matching the action batch")
        if self.salience is not None:
            salience = np.asarray(self.salience)
            if salience.shape != (batch, self.n) or not np.isfinite(salience).all():
                raise ValueError("salience must be a finite (batch, neurons) array")
        return reward, done, next_drive, bootstrap

    def _next_state(self, drive: np.ndarray, done: np.ndarray) -> BrainState:
        warm = self._free
        assert warm is not None
        if done.any():
            kernel = self.learner.brain._torch
            handle = warm.device
            if kernel is not None and handle is not None and handle.get("holder") is kernel:
                # the finished rows return to rest on the device; nothing comes to the host
                warm = BrainState(
                    None,  # type: ignore[arg-type]
                    None,  # type: ignore[arg-type]
                    None,  # type: ignore[arg-type]
                    warm.steps,
                    device=kernel.keep_rows(handle, ~done),
                )
            else:
                v, a = warm.v.copy(), warm.adaptation.copy()
                v[done], a[done] = 0.0, 0.0
                warm = BrainState(v, warm.activation, a, warm.steps)
        return self.learner.free(drive, warm=warm)

    def _learn_device(
        self,
        kernel: Any,
        plus: BrainState,
        minus: BrainState,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_drive: np.ndarray,
        bootstrap: np.ndarray | None,
        observed: np.ndarray,
    ) -> dict[str, float]:
        """``learn`` for streams whose phases rest on the torch device: each stream's contrast,
        eligibility trace and the mean step never come to the host; the critic and the
        dopamine do (a value per stream and a row of the critic's neurons)."""
        cfg = self.config
        torch = kernel.torch
        span = 2.0 * self.learner.config.beta
        decay = cfg.gamma * cfg.lam
        batch = len(reward)
        edges, neurons = kernel.contrast_rows(plus.device["s"], minus.device["s"])
        if self.salience is not None:
            salience = torch.as_tensor(
                np.asarray(self.salience), dtype=edges.dtype, device=edges.device
            )
            edges = edges * salience[:, kernel._row_index[0]]
        if self._trace_device is None or self._trace_device[0].shape != edges.shape:
            if self.trace is not None and self.trace.shape == tuple(edges.shape):
                self._trace_device = (
                    torch.as_tensor(self.trace, dtype=edges.dtype, device=edges.device).clone(),
                    torch.as_tensor(
                        self.trace_bias, dtype=neurons.dtype, device=neurons.device
                    ).clone(),
                )
            else:
                self._trace_device = (torch.zeros_like(edges), torch.zeros_like(neurons))
            self.trace = self.trace_bias = None
        trace, trace_bias = self._trace_device
        trace = decay * trace + edges / span
        trace_bias = decay * trace_bias + neurons / span
        if self.trace_critic is None or self.trace_critic.shape[0] != batch:
            self.trace_critic = np.zeros((batch, len(self.critic_index) + 1))
        free = self._free
        assert free is not None
        self.trace_critic *= cfg.gamma * cfg.lam
        self.trace_critic[:, :-1] += free.activation[:, self.critic_index]
        self.trace_critic[:, -1] += 1.0
        next_state = self._next_state(next_drive, done)
        next_value_raw = self.value(next_state)
        next_value = np.where(
            done, 0.0 if bootstrap is None else np.asarray(bootstrap, dtype=float), next_value_raw
        )
        raw_target = reward + cfg.gamma * next_value
        td_error = raw_target - value
        if not np.isfinite(td_error).all():
            raise ValueError("TD errors must remain finite")
        delta = td_error
        if cfg.dopamine_center > 0:
            delta = self._centre(delta, observed)
        capped = self._capped(delta, observed)
        if cfg.dopamine_cap > 0:
            delta = np.clip(delta, -cfg.dopamine_cap, cfg.dopamine_cap)
        delta = np.where(observed, delta / observed.mean(), 0.0)
        self.updates += 1
        critic_weights, critic_bias = self._critic_candidate(td_error, delta, observed)
        d = torch.as_tensor(np.asarray(delta, dtype=float), dtype=trace.dtype, device=trace.device)
        d = d[:, None]
        next_brain = self.learner.brain
        report = self.learner._apply_device(
            kernel,
            cfg.eta * (d * trace).mean(dim=0),
            cfg.eta_bias * (d * trace_bias).mean(dim=0),
            state=free,
        )
        self.w_critic, self.b_critic = critic_weights, critic_bias
        if done.any():  # a finished stream forgets its traces
            keep = torch.as_tensor(~done, dtype=trace.dtype, device=trace.device)[:, None]
            trace = trace * keep
            trace_bias = trace_bias * keep
            self.trace_critic[done] = 0.0
        self._trace_device = (trace, trace_bias)
        self._free = next_state
        self._free_brain = next_brain
        self._drive = next_drive.copy()
        self._pending = None
        report["delta"] = float(np.abs(delta).mean())
        report["dopamine"] = float(delta.mean())
        report["td_error"] = float(np.abs(td_error[observed]).mean())
        report["capped"] = capped
        report["value"] = float(value[observed].mean())
        report["free_steps"] = float(next_state.steps)
        report["saturation"] = self._saturation(free)
        plastic = self.learner.plastic_synapses
        assert plastic is not None
        if plastic.all():
            report["trace"] = float(trace.abs().mean())
        else:
            index = torch.as_tensor(np.flatnonzero(plastic), device=trace.device)
            report["trace"] = float(trace[:, index].abs().mean())
        return report

    def value_of(self, drive: np.ndarray) -> np.ndarray:
        """The critic's value of a drive, settled cold, without touching the cached state."""
        return self.value(self.learner.free(drive))

    def fade(self, done: np.ndarray | None = None) -> None:
        """One moment passes that adds no eligibility: its action was answered greedily.

        Every eligibility trace decays by ``gamma * lam``, the step ``learn`` applies
        between two sampled actions, so the credit of earlier actions keeps fading with
        time while nothing is learned. ``done`` rows forget their traces, as in
        ``learn``. Parameters, the critic and optimizer history are unchanged.
        """
        decay = self.config.gamma * self.config.lam
        ended = None if done is None else np.asarray(done)
        if ended is not None and (ended.ndim != 1 or ended.dtype != np.bool_):
            raise ValueError("done must be a boolean vector")
        present = [
            name
            for name in ("trace", "trace_bias", "trace_critic")
            if getattr(self, name) is not None
        ]
        rows = {len(getattr(self, name)) for name in present}
        if self._trace_device is not None:
            rows.add(int(self._trace_device[0].shape[0]))
        if ended is not None and any(count != len(ended) for count in rows):
            raise ValueError("done must match the streams of the eligibility traces")
        if self._trace_device is not None:
            trace, trace_bias = (decay * tensor for tensor in self._trace_device)
            if ended is not None and ended.any():
                keep = trace.new_tensor((~ended).astype(float))[:, None]
                trace, trace_bias = trace * keep, trace_bias * keep
            self._trace_device = (trace, trace_bias)
        for name in present:
            value = getattr(self, name)
            value *= decay
            if ended is not None and ended.any():
                value[ended] = 0.0

    def reset(self) -> None:
        """Clear stream state and eligibility; keep learned parameters and optimizer history."""
        self._free = None
        self._free_brain = None
        self._drive = None
        self._pending = None
        self.trace = self.trace_bias = self.trace_critic = None
        self._trace_device = None
        self.salience = None
        if self._valence is not None:
            self._valence.reset()

    def parameters(self) -> int:
        return self.learner.parameters() + len(self.w_critic) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "critic": [int(i) for i in self.critic_index],
            "updates": self.updates,
            "learner": self.learner.to_dict(),
        }
