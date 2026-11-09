"""Three factors: a trace at every synapse, a critic, and a dopamine neuron.

Learning from reward happens online, at every step of one life. Every
synapse keeps an eligibility trace, the recent history of what it would have
moved by for the actions that were taken; a critic population reads the
settled state and learns to predict return; and a dopamine neuron broadcasts
the temporal-difference error of that prediction. The trace and the dopamine
multiply at every synapse, every step:

    e[e]     <- gamma * lam * e[e] + contrast[e]        (the action's eligibility)
    delta    =  r + gamma * V(s') - V(s)                 (the dopamine signal)
    scale[e] += eta * delta * e[e]

There is no buffer and no epoch. The contrast is the difference between the
two nudged phases (at +beta and -beta), per row, with the nudge's target the
action that was taken, so the eligibility is the score of that
action and the goal enters only through the nudge. The critic is a linear
reading of the neurons named for it, trained by its own trace, divided by the
trace's energy, on the actor's dopamine: the prediction error capped at
``dopamine_cap``. A ``memory`` with learned
parameters of its own (``contrast_rows``, ``propose`` and ``accept``, as ``Recall`` has)
keeps their eligibility in a trace of its own, ``trace_memory``: each sampled act's
per-row contrast joins it, it decays as every synapse's trace does, and the same dopamine
steps the parameters on it.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .brain import _FUSED as _FUSED_TRACE
from .brain import Brain, BrainState, Nudge, _host_copy, _resident
from .learning import Learner
from .resident import _CriticOnDevice, _forgotten

__all__ = [
    "ActorCritic",
    "ActorCriticConfig",
    "Maturity",
]


def _kept(value: Any) -> Any:
    """An eligibility as a refused outcome restores it: the host's in-place steps need a copy; a
    resident memory's steps make new tensors."""
    return value.copy() if isinstance(value, np.ndarray) else value


def _held_drive(drive: Any) -> Any:
    """A drive as the actor keeps it for the next read: the host's a caller may write, a
    resident one nothing writes once it settled."""
    return drive if _resident(drive) else drive.copy()


def _batch_of(state: BrainState) -> int:
    """The rows of a settled state, read from the device shape when the state rests there."""
    handle = state.device
    if handle is not None and "s" in handle and state.__dict__.get("v") is None:
        return int(handle["s"].shape[0])
    return int(np.atleast_2d(state.v).shape[0])


SATURATION_BAND = 0.02  # an output within this of 0 or 1 has no slope left for a nudge


@dataclass(slots=True)
class Maturity:
    """A brain's maturity (#253): one slow variable, the use of its plasticity.

    Each actor update adds its applied magnitude in the brain's own step units per synapse,
    ``use += sum |dw| / (eta N)``; every accepted moment of life keeps ``1 - 1 / recovery`` of
    ``use``. The step factor ``m = 1 / (1 + (use / scale) ** ORDER)`` multiplies the actor's
    synapse and bias steps and engages the store's surprise-gated writes with strength ``1 - m``.
    No count is kept: a busy brain matures and one that lives without learning returns toward
    its youth.
    """

    scale: float
    recovery: float
    use: float = 0.0

    ORDER = 4.0
    # the largest use a brain holds: at the smallest scale (1e-12) the factor's power stays
    # finite, and no life comes near it
    LIMIT = 1e64

    def factor(self) -> float:
        try:
            return float(1.0 / (1.0 + (self.use / self.scale) ** self.ORDER))
        except OverflowError:  # a use set beyond the limit: no step is left
            return 0.0

    def relax(self) -> None:
        self.use *= 1.0 - 1.0 / self.recovery

    def to_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "recovery": self.recovery, "use": self.use}


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    gamma: float = 0.99  # discount
    lam: float = 0.9  # trace decay of the actor's eligibility
    eta: float = 0.5  # actor step per unit dopamine per unit trace
    eta_bias: float | None = None  # bias step; None derives eta / 10 at construction
    eta_critic: float = 0.05
    dopamine_cap: float = 1.0  # actor modulation saturates here (0: no cap)
    # settle steps of the two nudged phases that score a sampled action; None uses the
    # learner's nudged_steps
    eligibility_steps: int | None = None

    def __post_init__(self) -> None:
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
            # The bias rate follows the synapse rate unless chosen.
            if not np.isfinite(self.eta) or self.eta < 0:
                raise ValueError("eta must be finite and nonnegative")
            object.__setattr__(self, "eta_bias", self.eta / 10.0)
        eta_bias = self.eta_bias
        assert eta_bias is not None
        for name in ("gamma", "lam"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("eta", "eta_bias", "eta_critic", "dopamine_cap"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.eta > 0 and eta_bias > self.eta:
            warnings.warn(
                f"ActorCriticConfig with eta_bias ({eta_bias:g}) above eta ({self.eta:g}): "
                "the actor's bias step dominates its synapse step and the greedy policy tends "
                "to one action whatever the observation. The default couples eta_bias to "
                "eta / 10; when lowering eta through dataclasses.replace, set eta_bias too, or "
                "pass eta_bias=None to re-derive it. See docs/reward.md#actor-rates.",
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
    ) -> None:
        self.learner = learner
        self.config = config or ActorCriticConfig()
        self.group_id: np.ndarray | None = learner.output_groups
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
        # the critic's weights and eligibility when they rest on the device of the brain; then the
        # two attributes are absent, and reading one brings both back
        self._critic_device: _CriticOnDevice | None = None
        # a memory whose learned parameters reward trains beside the synapses (the brain that
        # owns it sets it), their eligibility per row, and the contrast of the action that
        # awaits its outcome
        self.memory: Any = None
        self.trace_memory: np.ndarray | None = None
        self._pending_memory: np.ndarray | None = None
        # the carried routes' learned gains (``Hearing``), stepped by the dopamine on their own
        # eligibility; the owner sets them
        self.hearing: Any = None
        self.trace_hearing: np.ndarray | None = None
        self._pending_hearing: np.ndarray | None = None
        # one stream's traces on the torch device, when it settles there
        self._trace_device: Any = None
        # the eligibility on the MLX device: by block, over the neurons, and each row's factor on
        # the host
        self._trace_mlx: tuple[list[Any], Any, np.ndarray] | None = None
        # (batch, neurons): when set before ``learn``, each synapse's eligibility is weighted by
        # its pre neuron's salience (a ``Trace.ringing``: what is still ringing gets written)
        self.salience: np.ndarray | None = None
        self._drive: np.ndarray | None = None
        self._free: BrainState | None = None
        self._free_brain: Brain | None = None  # parameters that produced the cached phase
        # ("phases", plus, minus, value) as activations, or ("states", plus, minus, value) as states
        self._pending: tuple[str, Any, Any, np.ndarray] | None = None
        self.updates = 0
        # the brain's maturity (``Maturity``), which scales the steps; None: the founder's steps
        self.maturity: Maturity | None = None

    # -- the critic on the device

    def __getattr__(self, name: str) -> Any:
        """The critic's weights or eligibility resting on the device, read on the host: both come
        back, and the next moment places them there again."""
        if self.__dict__.get("_critic_device") is not None and name in _CriticOnDevice.OWNED:
            self.leave_device()
            return self.__dict__[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getstate__(self) -> dict[str, Any]:
        self.leave_device()
        return self.__dict__

    def reside(self, kernel: Any) -> None:
        """The critic's weights and eligibility on the device of a brain's ``kernel``, read and
        written there each moment. An array written on the host since brings the other back
        first."""
        if kernel is None:
            self.leave_device()
            return
        if self._critic_device is not None:
            if not any(name in self.__dict__ for name in _CriticOnDevice.OWNED):
                return
            self.leave_device()
        self._critic_device = _CriticOnDevice(self, kernel)

    def leave_device(self) -> None:
        held = self.__dict__.get("_critic_device")
        if held is not None:
            written = {
                name: self.__dict__.pop(name) for name in _CriticOnDevice.OWNED
                if name in self.__dict__
            }
            held.give_back()
            self.__dict__.update(written)
            self._critic_device = None

    # -- readings

    @property
    def state(self) -> BrainState | None:
        """The free phase of the latest moment: the state ``act`` read, or ``learn`` settled."""
        return self._free

    def value(self, state: BrainState) -> np.ndarray:
        if self._critic_device is not None:
            return self._critic_device.value(state)
        return np.asarray(state.part(self.critic_index) @ self.w_critic + self.b_critic)

    def probabilities(
        self,
        state: BrainState,
        temperature: float | Sequence[float] | np.ndarray | None = None,
        *,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """Action probabilities, normalized separately for each motor slot.

        Multiple categorical slots return ``(batch, slots, max_size)`` with
        zero padding for shorter slots.
        ``temperature`` reads the same settled outputs at another softmax
        temperature, or one temperature per motor slot; left unset it is the learner's.
        ``bias`` adds a ``(batch, motor)`` value per motor neuron to the settled outputs
        before each slot's softmax: a vote from outside the readout, such as the records
        of an episodic memory.
        """
        temperature = self._temperature(temperature)
        s = state.part(self.learner.output_index)
        if bias is not None:
            bias = np.asarray(bias, dtype=float)
            if bias.shape != s.shape or not np.isfinite(bias).all():
                raise ValueError("bias must be a finite (batch, motor) array")
            s = s + bias
        if self.learner.slot_count > 1:
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

    def _clear_greedy(self, state: BrainState) -> np.ndarray | None:
        """The greedy action of one output slot without its softmax, where every row's most active
        motor neuron stands clear of the others (``fused.clear_choice``); None otherwise."""
        try:
            from .fused import clear_choice, compiled
        except ImportError:  # pragma: no cover
            return None
        if not compiled():
            return None
        return clear_choice(
            state.part(self.learner.output_index), float(self.learner.config.temperature)
        )

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
            slots = self.learner.slot_count
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
        self._drive = _held_drive(drive)
        self._pending = None
        self._pending_memory = None
        return self._free

    def _validated_drive(self, drive: np.ndarray) -> np.ndarray:
        if _resident(drive):  # a resident drive reads finite memories at finite observations
            if drive.ndim != 2 or drive.shape[1] != self.n or not len(drive):
                raise ValueError(f"drive must be a nonempty finite (batch, {self.n}) array")
            return drive
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
        *,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """Settle (or reuse the cached free phase), sample an action per row, keep its eligibility.

        Discrete: an action index per row, or one index per categorical motor slot.
        ``temperature`` samples the same settled outputs at another softmax
        temperature, or one per motor slot. Eligibility uses those same temperatures
        for the sampled action's policy score. A greedy read ignores them.
        ``bias`` joins the choice as in ``probabilities``, greedy or sampled; the
        eligibility of a sampled action still scores the readout's own policy, so
        reward teaches the readout the actions that the bias helped choose. With a
        ``memory`` whose parameters keep an eligibility, its contrast between the two
        nudged phases is kept for the outcome too.
        """
        drive = self._validated_drive(drive)
        if temperature is not None:
            temperature = self._temperature(temperature)  # before any state changes
        if bias is not None:
            bias = np.asarray(bias, dtype=float)
            if bias.shape != (len(drive), len(self.learner.output_index)) or not np.isfinite(
                bias
            ).all():
                raise ValueError("bias must be a finite (batch, motor) array")
        free = (
            self._free
            if self._free is not None
            and self._drive is not None
            and self._free_brain is self.learner.brain
            and (self._drive is drive or (
                not _resident(drive) and not _resident(self._drive)
                and np.array_equal(self._drive, drive)
            ))
            else self.settle(drive)
        )
        self._pending = None  # eligibility always belongs to this decision, including greedy reads
        self._pending_memory = None
        self._pending_hearing = None
        if greedy and bias is None and self.learner.slot_count == 1:
            action = self._clear_greedy(free)
            if action is not None:
                return np.asarray(action, dtype=np.int64)
        p = self.probabilities(free, None if greedy else temperature, bias=bias)
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
            self._pending_memory = self._memory_contrast(plus, minus)
            if self.hearing is not None:
                self._pending_hearing = self.hearing.contrast_rows(
                    self.learner.brain, plus, minus, beta
                )
        return np.asarray(action, dtype=np.int64)

    def _memory_contrast(self, plus: BrainState, minus: BrainState) -> np.ndarray | None:
        """The per-row contrast of the memory's learned parameters between a sampled act's
        two nudged phases, read through the current graph; None without such a memory, or
        when it keeps no eligibility."""
        rows = None if self.memory is None else getattr(self.memory, "contrast_rows", None)
        if rows is None:
            return None
        contrast = rows(self.learner.brain, plus, minus)
        if contrast is None or not isinstance(contrast, (np.ndarray, list)):
            return contrast  # None, or a resident memory's on the device
        return np.asarray(contrast, dtype=float)

    def _memory_eligibility(self, batch: int, decay: float) -> None:
        """The memory's trace after an outcome, as every synapse's: ``e <- gamma lam e +
        contrast``. A pending action that kept no contrast belongs to a memory that learns
        nothing by reward, and the trace is dropped, as is a trace of another batch."""
        contrast, trace = self._pending_memory, self.trace_memory
        if contrast is None or len(contrast) != batch:
            self.trace_memory = None
            return
        if getattr(self.memory, "_device", None) is not None:  # never in place
            self.trace_memory = self.memory.joined_eligibility(trace, contrast, decay)
            return
        if not isinstance(contrast, np.ndarray):
            contrast = _host_copy(contrast)
        if trace is not None and not isinstance(trace, np.ndarray):
            trace = _host_copy(trace)
        if trace is None or trace.shape != contrast.shape:
            trace = np.zeros_like(contrast)
        trace *= decay
        trace += contrast
        self.trace_memory = trace

    def _hearing_eligibility(self, batch: int, decay: float) -> None:
        """The carried routes' gains' trace after an outcome, as every synapse's."""
        contrast, trace = self._pending_hearing, self.trace_hearing
        if contrast is None or len(contrast) != batch:
            self.trace_hearing = None
            return
        if trace is None or trace.shape != contrast.shape:
            trace = np.zeros_like(contrast)
        trace *= decay
        trace += contrast
        self.trace_hearing = trace

    def _hearing_step(self, delta: np.ndarray) -> None:
        """The carried routes' gains stepped on their trace by the dopamine."""
        if self.trace_hearing is not None and self.hearing is not None:
            self.hearing.step((delta[:, None] * self.trace_hearing).mean(axis=0))

    def _memory_proposal(self, delta: Any) -> Any:
        """The memory's parameters stepped on their trace by the dopamine, as every synapse
        steps: proposed and checked by the memory, not installed. None without a trace, or
        when the memory takes no step."""
        trace = self.trace_memory
        if trace is None or self.memory is None:
            return None
        if not isinstance(trace, np.ndarray):  # the memory's on the device
            return self.memory.propose(self.memory._device.weighted(trace, delta))
        if not isinstance(delta, np.ndarray):  # a dopamine made on the device, to the host's trace
            delta = np.asarray(delta.detach().cpu().numpy(), dtype=float)
        weighted = delta.reshape(-1, *([1] * (trace.ndim - 1))) * trace
        return self.memory.propose(weighted.mean(axis=0))

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
            sizes = self.learner.slot_sizes
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
            # an apical compartment passes the nudge's change at the gain the free phase set
            apical_gain=self.learner.brain.apical_gain(drive, free, own_precision=True),
        )
        return self.learner.brain.settle_batch(
            drive,
            steps=cfg.nudged_steps
            if self.config.eligibility_steps is None
            else self.config.eligibility_steps,
            state=free,
            nudge=nudge,
            tolerance=None if cfg.nudged_fixed else cfg.tolerance,
        )

    def learn(
        self,
        reward: np.ndarray,
        done: np.ndarray,
        next_drive: np.ndarray,
        bootstrap: np.ndarray | None = None,
        *,
        observed: np.ndarray | None = None,
        warm: BrainState | None = None,
    ) -> dict[str, float]:
        """Dopamine from the reward and the next state's value; every synapse moves on its trace.

        ``done`` rows start their next life from rest and, unless ``bootstrap`` gives them
        a value, bootstrap from zero. A time limit is not a terminal state: pass the
        value of the last observation (``value_of``) as ``bootstrap`` for a truncated row.
        ``observed`` excludes padding rows from plasticity and reward statistics;
        their traces reset. At least one real transition is required. Updates are
        averaged over the observed rows, independent of padding.
        ``warm`` is the state the next state settles from when the streams' activity
        moved on after the action (``Brain.wait``); by default it is the free phase the
        action was chosen in, which always carries the critic's eligibility. A ``memory``'s
        parameters step on their own trace by the same dopamine, checked before the graph's
        update and installed after it; a refused update leaves them and their trace as they
        were, and the report names the step (``address_reward_step``). With a ``maturity``
        the synapse and bias steps are its ``factor()`` times the config's, and the synapse
        step applied adds to its ``use``; the critic and the memory keep their steps.
        """
        arrays = ("trace", "trace_bias", "trace_memory", "trace_hearing")
        saved = {
            name: _kept(getattr(self, name))
            for name in arrays
        }
        saved.update(
            {
                name: getattr(self, name)
                for name in ("_trace_device", "_trace_mlx", "_critic_device", "updates")
            }
        )
        critic = self._critic_device
        critic_trace = None
        if critic is None:
            saved["trace_critic"] = _kept(self.trace_critic)
        else:
            critic_trace = critic.trace
        # a mature brain's use grows by the synapse step applied, measured on the torch backend
        # on the device (an update replaces the kernel's scale tensor): one number to the host
        maturity, kernel, before = self.maturity, self.learner.brain._torch, None
        if maturity is not None:
            before = kernel.scale if kernel is not None else np.array(self.learner.brain.efficacy)
        self.__dict__["_outcome"] = saved  # what a refusal restores, held for as long as it can
        try:
            report = self._learn(reward, done, next_drive, bootstrap, observed=observed, warm=warm)
        except Exception:
            if critic is not None:  # as it rested, whichever way the outcome went
                for name in _CriticOnDevice.OWNED:
                    self.__dict__.pop(name, None)
                critic.trace = critic_trace
            self.__dict__.update(saved)
            raise
        finally:
            self.__dict__.pop("_outcome", None)
        if maturity is not None and self.config.eta > 0:
            after = self.learner.brain
            if kernel is not None:
                moved = float((after._torch.scale - before).abs().sum())
            else:
                moved = float(np.abs(after.efficacy - before).sum())
            maturity.use = min(
                maturity.use + moved / (float(self.config.eta) * self.synapses), Maturity.LIMIT
            )
        return report

    def _rates(self) -> tuple[float, float]:
        """The actor's synapse and bias steps: the config's, times the maturity's factor."""
        cfg = self.config
        assert cfg.eta_bias is not None  # derived at construction
        factor = 1.0 if self.maturity is None else self.maturity.factor()
        return factor * cfg.eta, factor * cfg.eta_bias

    def _learn(
        self,
        reward: np.ndarray,
        done: np.ndarray,
        next_drive: np.ndarray,
        bootstrap: np.ndarray | None = None,
        *,
        observed: np.ndarray | None = None,
        warm: BrainState | None = None,
    ) -> dict[str, float]:
        reward, done, next_drive, bootstrap = self._validated_transition(
            reward, done, next_drive, bootstrap
        )
        observed = np.ones(len(reward), dtype=bool) if observed is None else np.asarray(observed)
        if observed.shape != reward.shape or observed.dtype != np.bool_ or not observed.any():
            raise ValueError("observed must match the batch and include a real transition")
        if warm is not None and _batch_of(warm) != len(reward):
            raise ValueError("warm must be a state of the action batch")
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
            device_kernel = self.learner._device_kernel(first, second)
            tables = None if self.salience is not None or self.maturity is not None else (
                self.learner._mlx_update_tables(first, second)
            )
            if tables is not None:
                return self._learn_mlx(
                    tables, first, second, value, reward, done, next_drive, bootstrap, observed,
                    warm,
                )
            if device_kernel is None:
                first, second = first.activation, second.activation
                kind = "phases"
        if device_kernel is not None:
            return self._learn_device(
                device_kernel,
                first,
                second,
                value,
                reward,
                done,
                next_drive,
                bootstrap,
                observed,
                warm,
            )
        self.leave_device()
        if self._trace_device is not None:
            self.trace, self.trace_bias = (
                tensor.detach().cpu().double().numpy().copy() for tensor in self._trace_device
            )
            self._trace_device = None
        self._host_trace()
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
        self._memory_eligibility(batch, decay)
        self._hearing_eligibility(batch, decay)
        # the next state, warm from this one; a finished row starts its next life from rest
        next_state = self._next_state(next_drive, done, warm)
        next_value_raw = self.value(next_state)
        next_value = np.where(
            done, 0.0 if bootstrap is None else np.asarray(bootstrap, dtype=float), next_value_raw
        )
        raw_target = reward + cfg.gamma * next_value
        td_error = raw_target - value
        if not np.isfinite(td_error).all():
            raise ValueError("TD errors must remain finite")
        delta = td_error
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
        eta, eta_bias = self._rates()
        step_scale = eta * step_scale
        step_bias = eta_bias * step_bias
        critic_weights, critic_bias = self._critic_candidate(delta)
        # the memory's parameters step on their trace by the same dopamine, checked before
        # anything is installed
        proposal = self._memory_proposal(delta)
        next_brain = self.learner.brain
        report = self.learner.apply(step_scale, step_bias)
        self.w_critic, self.b_critic = critic_weights, critic_bias
        if proposal is not None:
            self.memory.accept(proposal)
            report["address_reward_step"] = float(proposal[3])
        self._hearing_step(delta)
        # a finished row forgets its traces
        if done.any():
            self.trace[done] = 0.0
            self.trace_bias[done] = 0.0
            self.trace_critic[done] = 0.0
            if self.trace_memory is not None:
                self.trace_memory = _forgotten(self.trace_memory, done)
            if self.trace_hearing is not None:
                self.trace_hearing[done] = 0.0
        self._free = next_state
        self._free_brain = next_brain
        self._drive = _held_drive(next_drive)
        self._pending = None
        self._pending_memory = None
        self._pending_hearing = None
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

    def _critic_eligibility(self, free: BrainState, batch: int) -> None:
        """The critic's eligibility of a moment's free state: its neurons' activity and the bias."""
        decay = self.config.gamma * self.config.lam
        if self._critic_device is not None:
            self._critic_device.add(free, batch, decay)
            return
        if self.trace_critic is None or self.trace_critic.shape[0] != batch:
            self.trace_critic = np.zeros((batch, len(self.critic_index) + 1))
        self.trace_critic *= decay
        self.trace_critic[:, :-1] += free.part(self.critic_index)
        self.trace_critic[:, -1] += 1.0

    def _accept_critic(self, candidate: tuple[Any, float]) -> None:
        if self._critic_device is not None:
            self._critic_device.weights, self.b_critic = candidate
        else:
            self.w_critic, self.b_critic = candidate

    def _forget_critic(self, done: np.ndarray) -> None:
        if self._critic_device is not None:
            self._critic_device.forget(done)
        elif self.trace_critic is not None:
            self.trace_critic[done] = 0.0

    def _critic_candidate(self, delta: np.ndarray) -> tuple[Any, float]:
        if self._critic_device is not None:
            return self._critic_device.candidate(delta)
        cfg = self.config
        critic_trace = self.trace_critic
        assert critic_trace is not None
        # the step is divided by the trace's energy, so its size is free of the activity's scale
        critic_trace = critic_trace / (1.0 + (critic_trace**2).sum(axis=1, keepdims=True))
        # the critic learns from the actor's bounded dopamine, which stabilizes control and
        # need not fit the mean return in reward units
        critic_step = cfg.eta_critic * (delta[:, None] * critic_trace).mean(axis=0)
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
        s = free.part(self.learner.output_index)
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

    def _next_state(
        self, drive: np.ndarray, done: np.ndarray, warm: BrainState | None = None
    ) -> BrainState:
        if warm is None:
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
        warm: BrainState | None = None,
    ) -> dict[str, float]:
        """``learn`` for streams whose phases rest on the torch device: each stream's contrast,
        eligibility trace and the mean step never come to the host; a value per stream does, and
        the dopamine goes back. The critic rests with them from the size its memories do
        (``_CriticOnDevice``), and otherwise a row of its neurons comes to the host."""
        cfg = self.config
        torch = kernel.torch
        span = 2.0 * self.learner.config.beta
        decay = cfg.gamma * cfg.lam
        batch = len(reward)
        s_plus, s_minus = plus.device["s"], minus.device["s"]
        neurons = s_plus - s_minus  # the eligibility in the brain's precision, and the step with it
        synapses = self.synapses
        if self._trace_device is None or tuple(self._trace_device[0].shape) != (batch, synapses):
            if self.trace is not None and self.trace.shape == (batch, synapses):
                # one copy on the device: at speech size the streams' trace is 50 MB each
                self._trace_device = tuple(
                    torch.from_numpy(array).to(neurons.device, neurons.dtype, copy=True)
                    for array in (self.trace, self.trace_bias)
                )
            else:
                self._trace_device = (
                    torch.zeros((batch, synapses), dtype=neurons.dtype, device=neurons.device),
                    torch.zeros_like(neurons),
                )
            self.trace = self.trace_bias = None
        trace, trace_bias = self._trace_device
        trace_bias = decay * trace_bias + neurons / span
        free = self._free
        assert free is not None
        self._critic_eligibility(free, batch)
        self._memory_eligibility(batch, decay)
        self._hearing_eligibility(batch, decay)
        next_state = self._next_state(next_drive, done, warm)
        resting = self._critic_device
        if resting is None:  # the critic on the host reads the dopamine there
            next_value_raw = self.value(next_state)
            next_value = np.where(
                done, 0.0 if bootstrap is None else np.asarray(bootstrap, dtype=float),
                next_value_raw,
            )
            raw_target = reward + cfg.gamma * next_value
            td_error = raw_target - value
            if not np.isfinite(td_error).all():
                raise ValueError("TD errors must remain finite")
            delta = td_error
            capped = self._capped(delta, observed)
            if cfg.dopamine_cap > 0:
                delta = np.clip(delta, -cfg.dopamine_cap, cfg.dopamine_cap)
            delta = np.where(observed, delta / observed.mean(), 0.0)
            critic = self._critic_candidate(delta)
            d = torch.as_tensor(
                np.asarray(delta, dtype=float), dtype=trace.dtype, device=trace.device
            )
        else:
            # the dopamine on the device, from the value the critic made there: the outcome's
            # checks, the critic's and the update's, and the dopamine come to the host in one read
            # before anything is installed
            exact = torch.float32 if trace.device.type == "mps" else torch.float64

            def here(values: Any) -> Any:
                return torch.as_tensor(np.asarray(values, dtype=float), dtype=exact,
                                       device=trace.device)

            following = resting.value_here(next_state).to(exact)
            ends = torch.as_tensor(np.asarray(done), device=trace.device)
            boot = torch.zeros_like(following) if bootstrap is None else here(bootstrap)
            td_here = here(reward) + cfg.gamma * torch.where(ends, boot, following) - here(value)
            delta_here = td_here
            if cfg.dopamine_cap > 0:
                delta_here = torch.clamp(delta_here, -cfg.dopamine_cap, cfg.dopamine_cap)
            shown = torch.as_tensor(np.asarray(observed), device=trace.device)
            delta_here = torch.where(
                shown, delta_here / float(observed.mean()), torch.zeros_like(delta_here)
            )
            critic_weights, critic_finite, critic_step = resting.unread(delta_here)
            d = delta_here.to(trace.dtype)
        self.updates += 1
        proposal = unread = None
        traced, memory = self.trace_memory, self.memory
        on_device = getattr(memory, "_device", None) is not None
        if resting is not None and traced is not None and on_device:
            # the memory's step on the device too, its check read with the outcome's
            unread = memory._device.proposal_unread(memory._device.weighted(traced, delta_here))
        else:
            proposal = self._memory_proposal(  # checked before anything is installed
                delta if resting is None else delta_here
            )
        next_brain = self.learner.brain
        eta, eta_bias = self._rates()
        step_bias = eta_bias * (d[:, None] * trace_bias).mean(dim=0)
        salience = None if self.salience is None else torch.as_tensor(
            np.asarray(self.salience), dtype=trace.dtype, device=trace.device
        )
        # the mean over the streams of d e, with e <- decay e + contrast: a row's contrast is a
        # product of two short rows, so the step needs no array of them; the trace moves only after
        # the last refusal, in place, so the streams hold one array of it
        step = decay * torch.mv(trace.T, d)
        if salience is None:
            step += kernel.contrast_edges(s_plus, s_minus, d) / span
        else:
            for first, stop, part in kernel.contrast_parts(s_plus, s_minus):
                part = (part * salience[first:stop][:, kernel._row_index[0]]).div_(span)
                step += torch.mv(part.T, d[first:stop])
        if resting is None:
            report = self.learner._apply_device(kernel, eta * (step / batch), step_bias)
        else:
            update = self.learner._device_candidate(kernel, eta * (step / batch), step_bias)
            one = torch.ones((), dtype=exact, device=trace.device)
            mapped = (one, 0.0 * one) if unread is None else (unread[3], unread[4])
            read = torch.cat([
                torch.stack([torch.isfinite(td_here).all().to(exact), critic_finite.to(exact),
                             critic_step.to(exact), mapped[0].to(exact), mapped[1].to(exact)]
                            + [flag.to(exact) for flag in update["flags"]]
                            + [number.to(exact) for number in update["steps"]]),
                td_here, delta_here,
            ]).cpu().numpy().astype(float)
            if not read[0]:
                raise ValueError("TD errors must remain finite")
            critic_bias = self.b_critic + float(read[2])
            if not read[1] or not np.isfinite(critic_bias):
                raise ValueError("critic parameters must remain finite")
            if not read[3]:
                raise ValueError("the address map's reward step must be finite")
            if unread is not None:
                proposal = (*unread[:3], float(read[4]), *unread[6:])
            self.learner._device_refusal(read[5:9])
            td_error, delta = read[11 : 11 + batch], read[11 + batch :]
            capped = self._capped(td_error, observed)
            critic = (critic_weights, critic_bias)
            report = self.learner._install_device(update, read[9:11])
        trace.mul_(decay)
        kernel.add_contrast(trace, s_plus, s_minus, span, salience)
        self._accept_critic(critic)
        if proposal is not None:
            self.memory.accept(proposal)
            report["address_reward_step"] = float(proposal[3])
        self._hearing_step(delta)
        if done.any():  # a finished stream forgets its traces
            ended = torch.as_tensor(np.flatnonzero(done), device=trace.device)
            trace[ended] = 0.0
            trace_bias[ended] = 0.0
            self._forget_critic(done)
            if self.trace_memory is not None:
                self.trace_memory = _forgotten(self.trace_memory, done)
            if self.trace_hearing is not None:
                self.trace_hearing[done] = 0.0
        self._trace_device = (trace, trace_bias)
        self._free = next_state
        self._free_brain = next_brain
        self._drive = _held_drive(next_drive)
        self._pending = None
        self._pending_memory = None
        self._pending_hearing = None
        report["delta"] = float(np.abs(delta).mean())
        report["dopamine"] = float(delta.mean())
        report["td_error"] = float(np.abs(td_error[observed]).mean())
        report["capped"] = capped
        report["value"] = float(value[observed].mean())
        report["free_steps"] = float(next_state.steps)
        report["saturation"] = self._saturation(free)
        plastic = self.learner.plastic_synapses
        assert plastic is not None
        # the mean magnitude row by row: no array of the streams' size
        total = trace.new_zeros(())
        index = None
        if not plastic.all():
            index = torch.as_tensor(np.flatnonzero(plastic), device=trace.device)
        for row in trace:
            total += (row if index is None else row[index]).abs().sum()
        kept = trace.shape[1] if index is None else index.shape[0]
        report["trace"] = float(total / (batch * kept))
        return report

    def _host_trace(self) -> None:
        """The eligibility on the MLX device, back on the host where ``trace`` holds it."""
        held = self._trace_mlx
        if held is not None:
            kernel = self.learner.brain._mlx
            factor = held[2][:, None]
            self.trace = factor * kernel.rows_to_edges(held[0])
            self.trace_bias = factor * np.array(held[1], dtype=np.float64)
            self._trace_mlx = None

    def _learn_mlx(
        self,
        tables: dict[str, Any],
        plus: BrainState,
        minus: BrainState,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_drive: np.ndarray,
        bootstrap: np.ndarray | None,
        observed: np.ndarray,
        warm: BrainState | None = None,
    ) -> dict[str, float]:
        """``learn`` for streams whose phases rest on the MLX device, in float32: each stream's
        contrast, eligibility and the mean steps never come to the host. The dopamine is the host's,
        in float64, and the critic rests on whichever side the memories do. A moment that adds no
        eligibility changes a factor per row on the host (``fade``)."""
        cfg = self.config
        learner = self.learner
        span = 2.0 * learner.config.beta
        decay = cfg.gamma * cfg.lam
        batch = len(reward)
        kernel = learner.brain._mlx
        assert kernel is not None
        held = self._trace_mlx
        if held is not None and len(held[2]) != batch:
            held = None
        if held is None and self.trace is not None and self.trace.shape[0] == batch:
            held = (
                kernel.rows_to_blocks(self.trace),
                kernel.mx.array(np.asarray(self.trace_bias, dtype=np.float32)), np.ones(batch),
            )
        self.trace = self.trace_bias = None
        free = self._free
        assert free is not None
        self._critic_eligibility(free, batch)
        self._memory_eligibility(batch, decay)
        self._hearing_eligibility(batch, decay)
        next_state = self._next_state(next_drive, done, warm)
        next_value_raw = self.value(next_state)
        next_value = np.where(
            done, 0.0 if bootstrap is None else np.asarray(bootstrap, dtype=float), next_value_raw
        )
        raw_target = reward + cfg.gamma * next_value
        td_error = raw_target - value
        if not np.isfinite(td_error).all():
            raise ValueError("TD errors must remain finite")
        delta = td_error
        capped = self._capped(delta, observed)
        if cfg.dopamine_cap > 0:
            delta = np.clip(delta, -cfg.dopamine_cap, cfg.dopamine_cap)
        delta = np.where(observed, delta / observed.mean(), 0.0)
        self.updates += 1
        resting, traced, memory = self._critic_device, self.trace_memory, self.memory
        made: list[Any] = []  # made in the update's launch, read after it
        if resting is not None:
            unread_critic = resting.unread(delta)
            made += [unread_critic[1], unread_critic[2], unread_critic[0]]
        else:
            critic = self._critic_candidate(delta)
        unread = None
        if traced is not None and getattr(memory, "_device", None) is not None:
            unread = memory._device.proposal_unread(memory._device.weighted(traced, delta))
            if unread is not None:
                made += [unread[3], unread[4], *unread[5]]
        else:
            proposal = self._memory_proposal(delta)  # checked before anything is installed
        found: dict[str, Any] = {}

        def checked() -> None:
            """The critic's and the memory's checks, in their order, from the update's launch."""
            if resting is not None:
                bias = self.b_critic + float(unread_critic[2].item())
                if not bool(unread_critic[1].item()) or not np.isfinite(bias):
                    raise ValueError("critic parameters must remain finite")
                found["critic"] = (unread_critic[0], bias)
            if unread is not None:
                if not bool(unread[3].item()):
                    raise ValueError("the address map's reward step must be finite")
                found["proposal"] = (*unread[:3], float(unread[4].item()), *unread[6:])

        next_brain = learner.brain
        eta, eta_bias = self._rates()
        scale = kernel.scale if kernel.scale is not None else kernel.to_blocks(next_brain.efficacy)
        decays = decay * (np.ones(batch) if held is None else held[2])
        new, bias_trace, size, bias_size = kernel.rewarded(
            plus.device, minus.device, None if held is None else held[0],
            None if held is None else held[1], decays, span, delta, eta, eta_bias, scale, tables,
            learner.config.scale_cap, also=made, before=checked,
        )
        if resting is not None:
            critic = found["critic"]
        if traced is not None and getattr(memory, "_device", None) is not None:
            proposal = found.get("proposal")
        # nothing refuses from here: what alone holds the eligibility lets it be written over
        old = None if held is None else list(held[0])
        held = self._trace_mlx = None
        self.__dict__["_outcome"]["_trace_mlx"] = None
        blocks, sizes = kernel.eligible(
            plus.device, minus.device, old, decays, span, tables["synapses"]
        )
        learner.brain = next_brain._with_mlx_parameters(new)
        learner.updates += 1
        report: dict[str, float] = {"scale_step": size, "bias_step": bias_size}
        self._accept_critic(critic)
        if proposal is not None:
            self.memory.accept(proposal)
            report["address_reward_step"] = float(proposal[3])
        self._hearing_step(delta)
        factor = np.where(done, 0.0, 1.0)  # a finished stream forgets its traces
        if done.any():
            self._forget_critic(done)
            if self.trace_memory is not None:
                self.trace_memory = _forgotten(self.trace_memory, done)
            if self.trace_hearing is not None:
                self.trace_hearing[done] = 0.0
        self._trace_mlx = (blocks, bias_trace, factor)
        self._free = next_state
        self._free_brain = next_brain
        self._drive = _held_drive(next_drive)
        self._pending = None
        self._pending_memory = None
        self._pending_hearing = None
        report["delta"] = float(np.abs(delta).mean())
        report["dopamine"] = float(delta.mean())
        report["td_error"] = float(np.abs(td_error[observed]).mean())
        report["capped"] = capped
        report["value"] = float(value[observed].mean())
        report["free_steps"] = float(next_state.steps)
        report["saturation"] = self._saturation(free)
        report["trace"] = float((factor * sizes).mean())
        return report

    def value_of(self, drive: np.ndarray) -> np.ndarray:
        """The critic's value of a drive, settled cold, without touching the cached state."""
        return self.value(self.learner.free(drive))

    def fade(self, done: np.ndarray | None = None) -> None:
        """One moment passes that adds no eligibility: its action was answered greedily.

        Every eligibility trace decays by ``gamma * lam``, the step ``learn`` applies
        between two sampled actions, so the credit of earlier actions keeps fading with
        time while nothing is learned; a memory's trace decays with them. ``done`` rows
        forget their traces, as in ``learn``. Parameters and the critic are unchanged.
        """
        decay = self.config.gamma * self.config.lam
        ended = None if done is None else np.asarray(done)
        if ended is not None and (ended.ndim != 1 or ended.dtype != np.bool_):
            raise ValueError("done must be a boolean vector")
        critic = self._critic_device
        present = [
            name
            for name in ("trace", "trace_bias", "trace_memory", "trace_hearing")
            if getattr(self, name) is not None
        ]
        if critic is None and self.trace_critic is not None:
            present.append("trace_critic")
        rows = {len(getattr(self, name)) for name in present}
        if critic is not None and critic.trace is not None:
            rows.add(len(critic.trace))
        if self._trace_device is not None:
            rows.add(int(self._trace_device[0].shape[0]))
        if self._trace_mlx is not None:
            rows.add(len(self._trace_mlx[2]))
        if ended is not None and any(count != len(ended) for count in rows):
            raise ValueError("done must match the streams of the eligibility traces")
        if self._trace_mlx is not None:
            factor = decay * self._trace_mlx[2]
            if ended is not None and ended.any():
                factor[ended] = 0.0
            self._trace_mlx = (self._trace_mlx[0], self._trace_mlx[1], factor)
        if self._trace_device is not None:
            trace, trace_bias = self._trace_device
            trace.mul_(decay)
            trace_bias.mul_(decay)
            if ended is not None and ended.any():
                gone = np.flatnonzero(ended)
                trace[gone] = 0.0
                trace_bias[gone] = 0.0
        if critic is not None:
            critic.fade(decay, ended)
        for name in present:
            value = getattr(self, name)
            value *= decay
            if ended is not None and ended.any():
                setattr(self, name, _forgotten(value, ended))

    def reset(self) -> None:
        """Clear stream state and eligibility; keep learned parameters."""
        self._free = None
        self._free_brain = None
        self._drive = None
        self._pending = None
        self._pending_memory = None
        self._pending_hearing = None
        self.trace = self.trace_bias = None
        if self._critic_device is not None:
            self._critic_device.trace = None
        else:
            self.trace_critic = None
        self.trace_memory = None
        self.trace_hearing = None
        self._trace_device = None
        self._trace_mlx = None
        self.salience = None

    def parameters(self) -> int:
        return self.learner.parameters() + len(self.critic_index) + 1 + (self.maturity is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "critic": [int(i) for i in self.critic_index],
            "updates": self.updates,
            "learner": self.learner.to_dict(),
        }
