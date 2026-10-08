"""The free/nudged learning rule.

Learning in a brain is two settling phases and one local comparison. The
net settles *free*, with only its input stimulated; that settled state is its
answer, and nothing about the goal enters it. Then, starting from that
state, the net settles again *nudged*: the output neurons feel an extra
drive toward the target, and the nudge spreads back over the feedback
synapses into the rest of the net. Every synapse then moves its own scale
on the difference between what its two endpoints did in the two phases:

    scale[e] += eta / beta * ( s+[pre] * s+[post] - s0[pre] * s0[post] )
    bias[i]  += eta_b / beta * ( s+[i] - s0[i] )

That is a contrastive Hebbian rule. It reads two numbers per synapse and
one per neuron, so it is as local as settling itself; and the goal
enters through exactly one door, the nudge of the second phase. With
``centered`` set the learner runs the nudge both ways, ``+beta`` and
``-beta``, from the same free state and contrasts those two; that cancels
the leading error of a one-sided nudge on a smooth equilibrium branch.

For the rule to reach hidden neurons the connectome needs feedback: a synapse
from the output neurons back toward the hidden ones. ``layered`` builds such
a connectome, with forward and feedback synapses tied into one reciprocal pair,
optional lateral inhibition among the outputs, and named populations for the
layers.

The rule is exact in a limit. With symmetric effective recurrent weights and a
smooth stable equilibrium
branch, converged phases give a gradient in the small-nudge limit. The
raw contrast concerns effective weights; efficacy derivatives also need
the contact/gain factor. The cross-entropy nudge uses T times that loss.
See docs/learning.md for the parameter convention and finite-step limits. Two
things break that in practice, and ``learning_neuron_model`` is chosen so they do
not: a neuron below rest with a hard rectifier publishes nothing and cannot
be moved (hence a small ``leak``), and a neuron on a steep sigmoid sits
either silent or saturated (hence unit slope). The equilibrium tests check
the alignment of the rule against finite differences.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from .blocks import block_contrast
from .brain import Backend, Brain, BrainState, Equilibrium, Nudge
from .connectome import Connectome
from .neuron import NeuronModel
from .recovery import RecoveryStart

__all__ = [
    "Learner",
    "LearnerConfig",
    "LearningPhaseError",
    "layered",
    "embedded",
    "learning_neuron_model",
    "LearnedState",
    "calibrate_bias",
    "naive_efficacy",
    "seam_report",
    "preflight",
]


SCALE_CAP = 8.0  # the default of LearnerConfig.scale_cap: the magnitude a synapse may not exceed
_MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    beta: float = 0.1  # nudge strength
    eta: float = 0.2  # synapse learning rate, divided by beta in the update
    eta_bias: float | None = None  # bias learning rate; None derives eta / 10 at construction
    centered: bool = True  # contrast +beta against -beta rather than against the free state
    free_steps: int = 100  # most steps the free phase may take
    nudged_steps: int = 50  # most steps a nudged phase may take
    tolerance: float | None = 1e-4  # finite movement tolerance; qualified equation residual
    nudge: str = "cross_entropy"  # "quadratic" or "cross_entropy"
    temperature: float = 0.2  # softmax temperature of the cross-entropy nudge
    # >0: forgetting factor of the per-synapse RMS of its raw contrast that divides its step;
    # with momentum this is the adaptive local step (Adam written per synapse), bias-corrected
    normalize: float = 0.0
    normalize_floor: float = 1e-3  # added to the RMS so a quiet synapse does not blow up
    momentum: float = 0.0  # >0: each synapse steps on a running average of its own contrast
    decay: float = 0.0  # >0: every update shrinks each trainable synapse and bias by this fraction
    # the magnitude a plastic synapse's efficacy may not exceed; a smaller cap keeps a readout
    # neuron out of saturation, where a nudge has no slope and nothing can move again
    scale_cap: float = SCALE_CAP
    qualified: bool = False  # opt-in: require free and teaching equation residuals
    damping: int = 3  # candidate numerical strategy: at most this many dt halvings

    def __post_init__(self) -> None:
        if self.eta_bias is None:
            # The bias rate follows the synapse rate unless chosen: issue 126.
            if not np.isfinite(self.eta) or self.eta < 0:
                raise ValueError(
                    "beta must be finite and positive; learning rates finite and nonnegative"
                )
            object.__setattr__(self, "eta_bias", self.eta / 10.0)
        eta_bias = self.eta_bias
        assert eta_bias is not None
        if self.nudge not in ("quadratic", "cross_entropy"):
            raise ValueError("nudge must be 'quadratic' or 'cross_entropy'")
        if not np.isfinite(self.scale_cap) or self.scale_cap <= 0:
            raise ValueError("scale_cap must be finite and positive")
        if (
            not np.isfinite([self.beta, self.eta, eta_bias]).all()
            or self.beta <= 0
            or self.eta < 0
            or eta_bias < 0
        ):
            raise ValueError(
                "beta must be finite and positive; learning rates finite and nonnegative"
            )
        if not 0 <= self.normalize < 1:
            raise ValueError("normalize is a forgetting factor in [0, 1)")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum lies in [0, 1)")
        if not 0 <= self.decay < 1:
            raise ValueError("decay is a fraction in [0, 1)")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(self.normalize_floor) or self.normalize_floor <= 0:
            raise ValueError("normalize_floor must be finite and positive")
        if self.tolerance is not None and (not np.isfinite(self.tolerance) or self.tolerance < 0):
            raise ValueError("tolerance must be finite and nonnegative, or None")
        if not isinstance(self.qualified, bool):
            raise ValueError("qualified must be boolean")
        if self.qualified and self.tolerance is None:
            raise ValueError("qualified learning requires a finite residual tolerance")
        if (
            isinstance(self.damping, bool)
            or not isinstance(self.damping, (int, np.integer))
            or self.damping < 0
        ):
            raise ValueError("damping must be a nonnegative integer")
        for name in ("free_steps", "nudged_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.normalize > 0 and max(self.eta, eta_bias) > 0.05:
            warnings.warn(
                "LearnerConfig with normalize > 0 and eta or eta_bias above 0.05: "
                "RMS normalization can make parameter steps much larger than raw contrasts "
                "suggest and may saturate the readout. Consider smaller rates; "
                "see docs/learning.md#rates-under-normalization.",
                RuntimeWarning,
                stacklevel=3,
            )
        if self.eta > 0 and eta_bias > self.eta:
            warnings.warn(
                f"LearnerConfig with eta_bias ({eta_bias:g}) above eta ({self.eta:g}): "
                "the bias step dominates the synapse step. The default couples "
                "eta_bias to eta / 10; when lowering eta through dataclasses.replace, "
                "set eta_bias too, or pass eta_bias=None to re-derive it. "
                "See docs/learning.md#7-every-knob.",
                RuntimeWarning,
                stacklevel=3,
            )
        if self.qualified and self.nudged_steps < self.free_steps:
            warnings.warn(
                f"LearnerConfig with qualified=True and nudged_steps ({self.nudged_steps}) "
                f"below free_steps ({self.free_steps}): qualified teaching checks every "
                "phase against the full equations within its budget, and a nudged budget "
                "sized for finite teaching refuses lessons the free budget would settle. "
                "Set nudged_steps comparable to free_steps; "
                "see docs/learning.md#qualified-teaching-budgets.",
                RuntimeWarning,
                stacklevel=3,
            )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True)
class LearnedState:
    free: BrainState
    nudged: BrainState
    opposite: BrainState | None = None  # the -beta phase, when centered


def _phase_report(
    phases: Mapping[str, Equilibrium], *, accepted: bool, qualified: bool
) -> dict[str, float]:
    report = {
        "accepted": float(accepted),
        "qualified": float(qualified and accepted),
        "qualification_required": float(qualified),
        "attempted_presentations": float(np.size(next(iter(phases.values())).residual)),
    }
    report["accepted_presentations"] = report["attempted_presentations"] if accepted else 0.0
    for name, phase in phases.items():
        rows = np.size(phase.residual)
        report[f"{name}_steps"] = float(phase.steps)
        report[f"{name}_residual"] = float(np.max(phase.residual))
        report[f"{name}_residual_checks"] = float(phase.residual_checks)
        report[f"{name}_damping_halvings"] = float(phase.damping_halvings)
        report[f"{name}_stagnation_checks"] = float(phase.stagnation_checks)
        report[f"{name}_row_sweeps"] = float(rows * phase.steps)
        report[f"{name}_row_residual_checks"] = float(rows * phase.residual_checks)
    report["total_steps"] = float(sum(phase.steps for phase in phases.values()))
    report["total_residual_checks"] = float(sum(phase.residual_checks for phase in phases.values()))
    report["total_stagnation_checks"] = float(sum(
        phase.stagnation_checks for phase in phases.values()
    ))
    report["total_row_sweeps"] = float(sum(
        np.size(phase.residual) * phase.steps for phase in phases.values()
    ))
    report["total_row_residual_checks"] = float(sum(
        np.size(phase.residual) * phase.residual_checks for phase in phases.values()
    ))
    return report


class LearningPhaseError(RuntimeError):
    """A refused lesson, retaining every attempted phase and its computational work.

    ``phase`` names the failed solve; ``phases`` contains the earlier and failed
    equilibria. ``report`` counts attempted presentations, phase sweeps,
    residual evaluations and complete-state stagnation comparisons.
    No parameter update or optimizer-history write has occurred.
    ``hint`` names a configuration cause when one is known, such as a nudged
    budget far below the free budget.
    """

    def __init__(
        self, phase: str, phases: Mapping[str, Equilibrium], hint: str | None = None
    ) -> None:
        self.phase = phase
        self.phases = dict(phases)
        self.hint = hint
        self.report = _phase_report(phases, accepted=False, qualified=True)
        failed = phases[phase]
        super().__init__(
            f"{phase} learning phase did not settle within {failed.steps} steps: "
            f"residual={float(np.max(failed.residual)):.6g}, tolerance={failed.tolerance:g}; "
            "no learning update applied" + (f"; {hint}" if hint else "")
        )


@dataclass
class Learner:
    """Owns the plastic ``efficacy`` and ``bias`` of a brain and applies the learning rule."""

    brain: Brain
    outputs: Sequence[int]
    config: LearnerConfig = field(default_factory=LearnerConfig)
    plastic_synapses: np.ndarray | None = None  # bool per synapse; default all
    plastic_neurons: np.ndarray | None = None  # bool per neuron: whose bias moves and decays
    reciprocal: bool = True  # a synapse and its reverse share one efficacy: a reciprocal pair
    tie_groups: np.ndarray | None = None  # int per synapse (-1: none); a group shares one scale
    # float per synapse multiplying its step (default one): a fan-in scale, a per-projection
    # rate; a wide population's inbound synapses need a smaller step than a narrow one's
    synapse_rate: np.ndarray | None = None
    # the outputs as groups, each its own softmax choice, all nudged together: a count of
    # equal groups, or one size per group (a body's controller: a move of nine, a grip of two)
    slots: int | Sequence[int] = 1
    updates: int = 0
    contrast_updates: int = 0  # optimizer history excludes externally applied reward updates
    # opt-in: where qualified nudged phases start settling; None starts them from the free state
    recovery: RecoveryStart | None = None
    velocity: np.ndarray = field(init=False, repr=False)
    velocity_bias: np.ndarray = field(init=False, repr=False)
    second_moment: np.ndarray = field(init=False, repr=False)
    second_moment_bias: np.ndarray = field(init=False, repr=False)
    last_calibration: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        outputs = np.asarray(list(self.outputs))
        if outputs.ndim != 1 or outputs.size == 0:
            raise ValueError("outputs must contain at least one neuron index")
        if not np.issubdtype(outputs.dtype, np.integer):
            raise ValueError("outputs must contain integer neuron indices")
        if (outputs < 0).any() or (outputs >= self.brain.connectome.n).any():
            raise ValueError("output neuron index is outside the connectome")
        if np.unique(outputs).size != outputs.size:
            raise ValueError("outputs must contain distinct neuron indices")
        self.output_index = outputs.astype(np.int64)
        self.output_mask = np.zeros(self.brain.connectome.n)
        self.output_mask[self.output_index] = 1.0
        if isinstance(self.slots, (int, np.integer)):
            if (
                isinstance(self.slots, bool)
                or self.slots < 1
                or len(self.output_index) % self.slots
            ):
                raise ValueError("slots must divide the number of output neurons")
            sizes = [len(self.output_index) // self.slots] * self.slots
        else:
            if any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in self.slots):
                raise ValueError("slot sizes must be integers")
            sizes = [int(k) for k in self.slots]
            if not sizes or min(sizes) < 1 or sum(sizes) != len(self.output_index):
                raise ValueError("slot sizes must be positive and add up to the output neurons")
        self.slot_sizes = np.asarray(sizes, dtype=np.int64)
        self.slot_offsets = np.concatenate([[0], np.cumsum(self.slot_sizes)[:-1]]).astype(np.int64)
        if self.recovery is not None and not isinstance(self.recovery, RecoveryStart):
            raise ValueError("recovery must be a RecoveryStart or None")
        self.slot_count = len(sizes)  # ``slots`` keeps what was asked for
        self.slot_size = int(self.slot_sizes[0]) if len(set(sizes)) == 1 else 0  # equal, or not
        self.output_groups: np.ndarray | None = None
        if self.slot_count > 1:  # one softmax per slot: a whole utterance settles at once
            self.output_groups = np.full(self.brain.connectome.n, -1, dtype=np.int64)
            groups = np.repeat(np.arange(self.slot_count), self.slot_sizes)
            self.output_groups[self.output_index] = groups
        if self.plastic_synapses is None:
            self.plastic_synapses = np.ones(self.brain.connectome.synapses, dtype=bool)
        if self.plastic_neurons is None:
            self.plastic_neurons = np.ones(self.brain.connectome.n, dtype=bool)
        w = self.brain.connectome
        for name, size in (("plastic_synapses", w.synapses), ("plastic_neurons", w.n)):
            mask = np.asarray(getattr(self, name))
            if mask.shape != (size,) or mask.dtype != np.bool_:
                raise ValueError(f"{name} must be a boolean vector of length {size}")
            setattr(self, name, mask)
        if self.synapse_rate is not None:
            rate = np.asarray(self.synapse_rate, dtype=float)
            if rate.shape != (w.synapses,) or not np.isfinite(rate).all() or (rate < 0).any():
                raise ValueError(
                    f"synapse_rate must be a finite nonnegative vector of length {w.synapses}"
                )
            self.synapse_rate = rate
        self.second_moment = np.zeros(w.synapses)  # per synapse, for normalized steps
        self.second_moment_bias = np.zeros(w.n)
        self.velocity = np.zeros(w.synapses)  # per synapse, for momentum
        self.velocity_bias = np.zeros(w.n)
        self.reverse = np.full(w.synapses, -1, dtype=np.int64)
        if self.reciprocal and w.synapses:
            key = w.post * w.n + w.pre
            reverse_key = w.pre * w.n + w.post
            order = np.argsort(key)
            hit = np.searchsorted(key[order], reverse_key)
            hit = np.minimum(hit, len(order) - 1)
            found = key[order][hit] == reverse_key
            self.reverse[found] = order[hit[found]]
        # Index arrays for the update, made once: the paired synapses and the tied members,
        # so an update touches those and not every synapse of a wide net.
        self._paired = np.flatnonzero(self.reverse >= 0)
        self._paired_reverse = self.reverse[self._paired]
        if not np.array_equal(self.reverse[self._paired_reverse], self._paired):
            raise ValueError(
                "reciprocal learning needs unique pairs; merge with from_synapses first"
            )
        self._members = np.zeros(0, dtype=np.int64)
        self._member_groups = np.zeros(0, dtype=np.int64)
        self._member_count = np.zeros(0)
        if self.tie_groups is not None:
            groups = np.asarray(self.tie_groups)
            if (
                groups.shape != (w.synapses,)
                or not np.issubdtype(groups.dtype, np.integer)
                or (groups < -1).any()
            ):
                raise ValueError("tie_groups must be an integer per synapse, -1 for untied")
            self.tie_groups = groups.copy()
            members = np.flatnonzero(groups >= 0)
            partners = self.reverse[members]
            self._members = np.union1d(members, partners[partners >= 0])
            if len(self._members):
                # Close explicit ties over reciprocal partners. Compact IDs avoid allocating
                # an array up to the largest user-supplied label, and overlapping ties compose.
                parent = {int(i): int(i) for i in self._members}

                def root(i: int) -> int:
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i

                representatives: dict[int, int] = {}
                for i in self._members:
                    if self.reverse[i] >= 0:
                        parent[root(int(i))] = root(int(self.reverse[i]))
                    group = int(groups[i])
                    if group >= 0:
                        representative = representatives.setdefault(group, int(i))
                        parent[root(int(i))] = root(representative)
                _, self._member_groups = np.unique(
                    [root(int(i)) for i in self._members], return_inverse=True
                )
                self._member_count = np.bincount(self._member_groups).astype(float)

    # -- phases

    def free(self, drive: np.ndarray, warm: BrainState | None = None) -> BrainState:
        """Settle with the input stimulated and nothing else: the net's own answer.

        ``warm`` starts settling from an earlier state instead of rest,
        say the state the same inputs settled to on the previous pass. Within
        one attracting basin this can save steps; multiple attractors can
        give different answers from different starts.
        """
        cfg = self.config
        if cfg.qualified:
            phase = self._qualified_phase(drive, warm, cfg.free_steps)
            if not np.all(phase.qualified):
                raise LearningPhaseError("free", {"free": phase})
            return replace(phase.state, steps=phase.steps)
        return self.brain.settle_batch(
            drive, steps=cfg.free_steps, state=warm, tolerance=cfg.tolerance
        )

    def nudge_for(self, target: np.ndarray, beta: float, weight: np.ndarray | None = None) -> Nudge:
        cfg = self.config
        temperature = cfg.temperature if cfg.nudge == "cross_entropy" else None
        # With several slots each slot's nudge carries beta / slots, so the contrast over 2 beta is
        # the gradient of the mean loss over the slots and eta means the same at any slot count.
        return Nudge(
            target,
            self.output_mask,
            beta / self.slot_count,
            softmax_temperature=temperature,
            weight=weight,
            groups=self.output_groups,
        )

    def nudged(
        self,
        drive: np.ndarray,
        free: BrainState,
        target: np.ndarray,
        sign: float = 1.0,
        weight: np.ndarray | None = None,
    ) -> BrainState:
        """From the free state, settle with the output neurons pulled toward ``target``.

        ``sign`` of -1 pushes them away instead: the opposite phase of a centered
        contrast. ``weight`` scales the pull row by row (an advantage, when the
        target is an action that was taken).
        """
        cfg = self.config
        if cfg.qualified:
            phase = self._qualified_phase(
                drive, free, cfg.nudged_steps,
                self.nudge_for(target, sign * cfg.beta, weight),
            )
            name = "nudged" if sign >= 0 else "opposite"
            if not np.all(phase.qualified):
                raise LearningPhaseError(name, {name: phase}, self._nudged_budget_hint())
            return replace(phase.state, steps=phase.steps)
        return self.brain.settle_batch(
            drive,
            steps=cfg.nudged_steps,
            state=free,
            nudge=self.nudge_for(target, sign * cfg.beta, weight),
            tolerance=cfg.tolerance,
        )

    def _nudged_budget_hint(self) -> str | None:
        """Name the budget mismatch when a refused nudged phase had less room than the free one."""
        cfg = self.config
        if cfg.nudged_steps >= cfg.free_steps:
            return None
        return (
            f"the nudged budget ({cfg.nudged_steps}) is below the free budget "
            f"({cfg.free_steps}); qualified nudged phases settle to the same tolerance, "
            "see docs/learning.md#qualified-teaching-budgets"
        )

    def _qualified_phase(
        self, drive: np.ndarray, state: BrainState | None, budget: int,
        nudge: Nudge | None = None,
    ) -> Equilibrium:
        cfg = self.config
        if cfg.tolerance is None:
            raise ValueError("qualified learning requires a finite residual tolerance")
        return self.brain.equilibrate(
            drive, state=state, budget=budget, tolerance=cfg.tolerance,
            nudge=nudge, damping=cfg.damping,
        )

    def _labels(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        if self.slot_count > 1:
            if labels.ndim != 2 or labels.shape[1] != self.slot_count:
                raise ValueError("a slotted learner takes labels shaped (batch, slots)")
        elif labels.ndim != 1:
            raise ValueError("labels must have shape (batch,) for a single output slot")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be integer class indices")
        if (labels < 0).any() or (labels >= self.slot_sizes).any():
            raise ValueError("a label lies outside its output slot's choices")
        return labels.astype(np.int64, copy=False)

    def targets(self, labels: np.ndarray) -> np.ndarray:
        """Per-neuron target activation for class labels: ``(batch,)``, or ``(batch, slots)``."""
        labels = self._labels(labels)
        batch = len(labels)
        target = np.zeros((batch, self.brain.connectome.n))
        target[:, self.output_index] = 0.0
        if self.slot_count > 1:
            neurons = self.output_index[labels + self.slot_offsets[None, :]]
            target[np.arange(batch)[:, None], neurons] = 1.0
        else:
            target[np.arange(batch), self.output_index[labels]] = 1.0
        return target

    # -- the rule

    def _validate_phases(
        self, free: BrainState, nudged: BrainState, opposite: BrainState | None
    ) -> None:
        """A contrast pairs the same streams, never NumPy's broadcast rows.

        Read resident tensor shapes without fetching activations from a device.
        This is a shape contract, not an equilibrium or provenance certificate.
        """
        shapes = []
        for state in (free, nudged) if opposite is None else (free, nudged, opposite):
            activation = state.__dict__.get("activation")
            if activation is None and state.device is not None and "s" in state.device:
                shape = tuple(state.device["s"].shape)
            else:
                shape = np.asarray(state.activation).shape
            shapes.append(shape)
        shape = shapes[0]
        if (len(shape) != 2 or shape[0] == 0 or shape[1] != self.brain.connectome.n
                or any(other != shape for other in shapes[1:])):
            raise ValueError("all phases must have the same nonempty (batch, neurons) shape")

    def contrast(
        self, free: BrainState, nudged: BrainState, opposite: BrainState | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The per-synapse and per-neuron differences the rule reads, already divided by beta.

        One-sided: ``(nudged - free) / beta``. Centered: ``(nudged - opposite) / (2 beta)``.
        """
        self._validate_phases(free, nudged, opposite)
        w = self.brain.connectome
        beta = self.config.beta
        minus_state, span = (free, beta) if opposite is None else (opposite, 2.0 * beta)
        on_device = self.brain.contrast_on_device(nudged, minus_state)
        if on_device is not None:  # both phases still on the accelerator: read the contrast there
            edges, neurons = on_device
            assert nudged.device is not None
            batch = int(nudged.device["s"].shape[0])
            return edges / (batch * span), neurons / (batch * span)
        s_plus, s_minus = nudged.activation, minus_state.activation
        if self.brain._blocked and w.synapses > 4 * w.n:
            gram = block_contrast(self.brain.layout, s_plus, s_minus)
            return gram / (len(s_plus) * span), (s_plus - s_minus).mean(axis=0) / span
        if not self.brain._blocked:
            # Sparse graphs must not materialize the layout's potentially n-by-n matrix.
            # Bound each temporary to about one million batch/synapse entries.
            edges = np.empty(w.synapses)
            chunk = max(1, 1_000_000 // len(s_plus))
            for start in range(0, w.synapses, chunk):
                pre, post = w.pre[start : start + chunk], w.post[start : start + chunk]
                a_plus, a_minus = s_plus[:, pre], s_minus[:, pre]
                b_plus, b_minus = s_plus[:, post], s_minus[:, post]
                products = a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus
                edges[start : start + chunk] = products.mean(axis=0) / span
            return edges, (s_plus - s_minus).mean(axis=0) / span
        a_plus, a_minus = s_plus[:, w.pre], s_minus[:, w.pre]
        b_plus, b_minus = s_plus[:, w.post], s_minus[:, w.post]
        products = a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus
        return products.mean(axis=0) / span, (s_plus - s_minus).mean(axis=0) / span

    def contrast_rows(
        self, free: BrainState, nudged: BrainState, opposite: BrainState | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The same differences as ``contrast``, one row per batch element: ``(batch, edges)``
        and ``(batch, n)``. What a per-row eligibility trace reads."""
        self._validate_phases(free, nudged, opposite)
        w = self.brain.connectome
        beta = self.config.beta
        s_plus = nudged.activation
        if opposite is None:
            s_minus, span = free.activation, beta
        else:
            s_minus, span = opposite.activation, 2.0 * beta
        a_plus, a_minus = s_plus[:, w.pre], s_minus[:, w.pre]
        b_plus, b_minus = s_plus[:, w.post], s_minus[:, w.post]
        hebb = a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus
        return hebb / span, (s_plus - s_minus) / span

    def apply(self, delta_scale: np.ndarray, delta_bias: np.ndarray) -> dict[str, float]:
        """Apply a per-synapse and per-neuron step: masks, tying, decay, bounds; then set the brain.

        This is the last half of ``update``; a rule that computes its own step (a
        three-factor trace, say) hands it here so every learner shares one notion of
        which synapses move, how a tied pair moves, and what bounds hold.
        """
        cfg = self.config
        assert self.plastic_synapses is not None
        all_trainable = bool(self.plastic_synapses.all())
        delta_scale = np.array(delta_scale, dtype=float)  # this update's own copy
        delta_bias = np.asarray(delta_bias, dtype=float)
        if (
            delta_scale.shape != (self.brain.connectome.synapses,)
            or delta_bias.shape != (self.brain.connectome.n,)
            or not np.isfinite(delta_scale).all()
            or not np.isfinite(delta_bias).all()
        ):
            raise ValueError("steps must be finite vectors, one per synapse and one per neuron")
        if not all_trainable:
            delta_scale[~self.plastic_synapses] = 0.0
        if self.synapse_rate is not None:  # each synapse's own step, before tying
            delta_scale *= self.synapse_rate
        if len(self._paired):  # one synapse, one weight: both directions move by the same amount
            mean = 0.5 * (delta_scale[self._paired] + delta_scale[self._paired_reverse])
            delta_scale[self._paired] = mean
        if len(self._members):  # a group moves by the mean of its members' contrasts
            total = np.bincount(self._member_groups, weights=delta_scale[self._members])
            delta_scale[self._members] = (total / np.maximum(self._member_count, 1.0))[
                self._member_groups
            ]
        if not all_trainable:  # tying never moves a frozen synapse
            delta_scale[~self.plastic_synapses] = 0.0
        if not np.isfinite(delta_scale).all():
            raise ValueError("scaled and tied steps must remain finite")
        scale = self.brain.efficacy + delta_scale
        assert self.plastic_neurons is not None
        delta_bias = np.where(self.plastic_neurons, delta_bias, 0.0)
        bias = self.brain.bias + delta_bias
        if cfg.decay > 0:  # a leak on the synapses: what is not relearned fades away
            scale = np.where(self.plastic_synapses, scale * (1.0 - cfg.decay), scale)
            bias = np.where(self.plastic_neurons, bias * (1.0 - cfg.decay), bias)
        cap = cfg.scale_cap
        scale = np.where(self.plastic_synapses, np.clip(scale, -cap, cap), scale)
        kernel = self.brain._torch
        if kernel is None:
            self.brain = self.brain.with_parameters(efficacy=scale, bias=bias)
        else:
            # The kernel keeps the parameters on its device and is updated in place, so a
            # state that rests on it continues there; rebuilding the brain would re-upload
            # every weight and strand every settled state on a kernel no longer used.
            torch = kernel.torch
            self.brain = self.brain._with_device_parameters(
                torch.from_numpy(np.ascontiguousarray(scale)).to(kernel.device, kernel.param_dtype),
                torch.from_numpy(np.ascontiguousarray(bias)).to(kernel.device, kernel.param_dtype),
            )
        self.updates += 1
        return {
            "scale_step": float(np.abs(delta_scale).mean()) if delta_scale.size else 0.0,
            "bias_step": float(np.abs(delta_bias).mean()),
        }

    def _device_kernel(self, plus: BrainState, minus: BrainState) -> Any:
        """The torch kernel both states rest on, when the update can stay on the device."""
        kernel = self.brain._torch
        if kernel is None or kernel.layout is None:
            return None
        for state in (plus, minus):
            if state.device is None or state.device.get("holder") is not kernel:
                return None
        return kernel

    def _host_moments(self) -> None:
        """Materialise optimizer history only when a host caller reads or edits it."""
        held = self.__dict__.get("_device_moments")
        if held is not None:
            for name in _MOMENTS:
                self.__dict__["_" + name] = held[name].detach().cpu().double().numpy().copy()
            self.__dict__["_device_moments"] = None

    def _moments_on_device(self, kernel: Any) -> dict[str, Any]:
        held = self.__dict__.get("_device_moments")
        if held is None or held["holder"] is not kernel:
            self._host_moments()
            held = {"holder": kernel}
            for name in _MOMENTS:
                held[name] = (
                    kernel.torch.from_numpy(np.ascontiguousarray(self.__dict__["_" + name]))
                    .to(kernel.device, kernel.param_dtype)
                    .clone()
                )
            self.__dict__["_device_moments"] = held
        return cast(dict[str, Any], held)

    def _adaptive_device(self, kernel: Any, edges: Any, neurons: Any) -> tuple[Any, Any]:
        """The host update's bias-corrected momentum/RMS, with history on the device."""
        cfg = self.config
        if not (cfg.momentum or cfg.normalize):
            return edges, neurons
        previous = self._moments_on_device(kernel)
        held = {"holder": kernel, **{name: previous[name].clone() for name in _MOMENTS}}
        self.__dict__["_device_moments"] = held
        count = self.contrast_updates + 1
        raw_edges, raw_neurons = edges, neurons
        if cfg.momentum:
            m = cfg.momentum
            held["velocity"].mul_(m).add_(edges, alpha=1.0 - m)
            held["velocity_bias"].mul_(m).add_(neurons, alpha=1.0 - m)
            correction = 1.0 - m**count
            edges = held["velocity"] / correction
            neurons = held["velocity_bias"] / correction
        if cfg.normalize:
            rho = cfg.normalize
            held["second_moment"].mul_(rho).addcmul_(raw_edges, raw_edges, value=1.0 - rho)
            held["second_moment_bias"].mul_(rho).addcmul_(raw_neurons, raw_neurons, value=1.0 - rho)
            correction = 1.0 - rho**count
            edges = edges / ((held["second_moment"] / correction).sqrt() + cfg.normalize_floor)
            neurons = neurons / (
                (held["second_moment_bias"] / correction).sqrt() + cfg.normalize_floor
            )
        if not bool(kernel.torch.stack([
            kernel.torch.isfinite(held[name]).all() for name in _MOMENTS
        ]).all()):
            raise ValueError("optimizer moments must remain finite")
        return edges, neurons

    def _device_indices(self, kernel: Any) -> dict[str, Any]:
        """Index tensors of the update on the kernel's device, made once per kernel and masks."""
        torch = kernel.torch
        assert self.plastic_synapses is not None and self.plastic_neurons is not None
        key = (
            id(kernel),
            id(self.plastic_synapses),
            id(self.plastic_neurons),
            id(self.synapse_rate),
        )
        cache = self.__dict__.setdefault("_device_cache", {})
        if (
            cache.get("key") != key
            or not np.array_equal(cache.get("host_synapses"), self.plastic_synapses)
            or not np.array_equal(cache.get("host_neurons"), self.plastic_neurons)
            or (
                cache.get("host_rate") is not None
                if self.synapse_rate is None
                else not np.array_equal(cache.get("host_rate"), self.synapse_rate)
            )
        ):
            dev = kernel.device

            def to(x: np.ndarray, dtype: Any = None) -> Any:
                return torch.from_numpy(np.ascontiguousarray(x)).to(dev, dtype)

            all_synapses = bool(self.plastic_synapses.all())
            all_neurons = bool(self.plastic_neurons.all())
            cache.clear()
            cache.update(
                key=key,
                host_synapses=self.plastic_synapses.copy(),
                host_neurons=self.plastic_neurons.copy(),
                host_rate=None if self.synapse_rate is None else self.synapse_rate.copy(),
                rate=None
                if self.synapse_rate is None
                else to(self.synapse_rate, kernel.param_dtype),
                paired=to(self._paired) if len(self._paired) else None,
                paired_reverse=to(self._paired_reverse) if len(self._paired) else None,
                members=to(self._members) if len(self._members) else None,
                member_groups=to(self._member_groups) if len(self._members) else None,
                member_count=to(self._member_count, kernel.param_dtype)
                if len(self._members)
                else None,
                synapses=None if all_synapses else to(self.plastic_synapses, kernel.param_dtype),
                neurons=None if all_neurons else to(self.plastic_neurons, kernel.param_dtype),
            )
        return cast(dict[str, Any], cache)

    def _apply_device(self, kernel: Any, delta_scale: Any, delta_bias: Any) -> dict[str, float]:
        """``apply`` on the device: the same masks, tying, decay and bounds, no host array."""
        torch, cfg = kernel.torch, self.config
        if (
            tuple(delta_scale.shape) != (self.brain.connectome.synapses,)
            or tuple(delta_bias.shape) != (self.brain.connectome.n,)
            or not bool(torch.isfinite(delta_scale).all() & torch.isfinite(delta_bias).all())
        ):
            raise ValueError("steps must be finite vectors, one per synapse and one per neuron")
        ix = self._device_indices(kernel)
        d = delta_scale.clone()
        if ix["synapses"] is not None:
            d = d * ix["synapses"]
        if ix["rate"] is not None:  # each synapse's own step, before tying
            d = d * ix["rate"]
        if (
            ix["paired"] is not None
        ):  # one synapse, one weight: both directions move by the same amount
            d[ix["paired"]] = 0.5 * (d[ix["paired"]] + d[ix["paired_reverse"]])
        if ix["members"] is not None:  # a group moves by the mean of its members' contrasts
            total = torch.zeros(len(ix["member_count"]), dtype=d.dtype, device=d.device)
            total.index_add_(0, ix["member_groups"], d[ix["members"]])
            d[ix["members"]] = (total / torch.clamp(ix["member_count"], min=1.0))[
                ix["member_groups"]
            ]
        if ix["synapses"] is not None:  # tying never moves a frozen synapse
            d = d * ix["synapses"]
        if not bool(torch.isfinite(d).all()):
            raise ValueError("scaled and tied steps must remain finite")
        scale = kernel.scale + d
        db = delta_bias if ix["neurons"] is None else delta_bias * ix["neurons"]
        bias = kernel.bias_param + db
        if cfg.decay > 0:  # a leak on the synapses: what is not relearned fades away
            keep_scale = (
                1.0 - cfg.decay if ix["synapses"] is None else 1.0 - cfg.decay * ix["synapses"]
            )
            plastic = ix["neurons"]
            keep_bias = 1.0 - cfg.decay if plastic is None else 1.0 - cfg.decay * plastic
            scale = scale * keep_scale
            bias = bias * keep_bias
        bounded = torch.clamp(scale, -cfg.scale_cap, cfg.scale_cap)
        scale = (
            bounded
            if ix["synapses"] is None
            else torch.where(ix["synapses"].bool(), bounded, kernel.scale)
        )
        self.brain = self.brain._with_device_parameters(scale, bias)
        self.updates += 1
        return {
            "scale_step": float(d.abs().mean()) if d.numel() else 0.0,
            "bias_step": float(db.abs().mean()),
        }

    def update(
        self, free: BrainState, nudged: BrainState, opposite: BrainState | None = None
    ) -> dict[str, float]:
        """Move every trainable synapse and every neuron on its own two-phase difference."""
        # Proposed optimizer history is part of the update transaction. A bad
        # contrast or an unrepresentable parameter step must not poison a retry.
        moments = {"_" + name: self.__dict__.get("_" + name) for name in _MOMENTS}
        held = self.__dict__.get("_device_moments")
        try:
            return self._update(free, nudged, opposite)
        except Exception:
            self.__dict__.update(moments)
            self.__dict__["_device_moments"] = held
            raise

    def _update(
        self, free: BrainState, nudged: BrainState, opposite: BrainState | None = None
    ) -> dict[str, float]:
        self._validate_phases(free, nudged, opposite)
        cfg = self.config
        assert cfg.eta_bias is not None  # resolved at construction
        minus_state, span = (free, cfg.beta) if opposite is None else (opposite, 2.0 * cfg.beta)
        kernel = self._device_kernel(nudged, minus_state)
        if kernel is not None:  # both phases rest on the torch device: the whole update stays there
            edges, neurons = kernel.contrast_tensors(nudged.device["s"], minus_state.device["s"])
            norm = float(nudged.device["s"].shape[0]) * span
            edges, neurons = self._adaptive_device(kernel, edges / norm, neurons / norm)
            report = self._apply_device(kernel, cfg.eta * edges, cfg.eta_bias * neurons)
            self.contrast_updates += 1
            return report
        synapse_term, neuron_term = self.contrast(free, nudged, opposite)
        if not np.isfinite(synapse_term).all() or not np.isfinite(neuron_term).all():
            raise ValueError("phase contrasts must be finite")
        raw_synapse, raw_neuron = synapse_term, neuron_term
        count = self.contrast_updates + 1  # only this optimizer's own history
        if cfg.momentum > 0:  # still local: a synapse accumulates only its own contrast
            m = cfg.momentum
            self.velocity = m * self.velocity + (1 - m) * synapse_term
            self.velocity_bias = m * self.velocity_bias + (1 - m) * neuron_term
            correction = 1.0 - m**count  # a short history is an average over fewer updates
            synapse_term, neuron_term = self.velocity / correction, self.velocity_bias / correction
        if cfg.normalize > 0:  # still local: a synapse reads only its own history
            rho = cfg.normalize
            self.second_moment = rho * self.second_moment + (1 - rho) * raw_synapse**2
            self.second_moment_bias = rho * self.second_moment_bias + (1 - rho) * raw_neuron**2
            correction = 1.0 - rho**count
            rms = np.sqrt(self.second_moment / correction) + cfg.normalize_floor
            rms_bias = np.sqrt(self.second_moment_bias / correction) + cfg.normalize_floor
            synapse_term, neuron_term = synapse_term / rms, neuron_term / rms_bias
        delta_scale = cfg.eta * synapse_term
        delta_bias = cfg.eta_bias * neuron_term
        if not all(np.isfinite(getattr(self, name)).all() for name in _MOMENTS):
            raise ValueError("optimizer moments must remain finite")
        report = self.apply(delta_scale, delta_bias)
        self.contrast_updates += 1
        return report

    def step(
        self,
        drive: np.ndarray,
        labels: np.ndarray,
        warm: BrainState | None = None,
        weight: np.ndarray | None = None,
    ) -> tuple[LearnedState, dict[str, float]]:
        """One free phase, the nudged phase(s), one update; returns the states and step sizes.

        ``weight`` is one number per row: how hard, and in which direction, that
        row's label is pulled. With actions as labels and advantages as weights
        this is the policy-gradient step, and the goal still enters only through
        the nudge.
        """
        target = self.targets(labels)
        drive = np.asarray(drive, dtype=float)
        if drive.ndim == 1:
            drive = drive[None, :]
        if drive.ndim != 2 or drive.shape[0] != target.shape[0]:
            raise ValueError("drive and labels must have the same batch size")
        cfg = self.config
        recovery = self.recovery
        if recovery is not None and not cfg.qualified:
            raise ValueError(
                "a recovery start requires qualified=True: only qualified phases are "
                "checked against the original equations wherever they start"
            )
        if cfg.qualified:
            phases: dict[str, Equilibrium] = {}
            phases["free"] = self._qualified_phase(drive, warm, cfg.free_steps)
            if not np.all(phases["free"].qualified):
                raise LearningPhaseError("free", phases)
            free = replace(phases["free"].state, steps=phases["free"].steps)
            nudges: dict[str, Nudge] = {}
            for name, sign in (("nudged", 1.0), ("opposite", -1.0)):
                if name == "opposite" and not cfg.centered:
                    break
                nudges[name] = self.nudge_for(target, sign * cfg.beta, weight)
                start = free if recovery is None else recovery.start(self, free, nudges[name])
                phases[name] = self._qualified_phase(
                    drive, start, cfg.nudged_steps, nudges[name],
                )
                if not np.all(phases[name].qualified):
                    raise LearningPhaseError(name, phases, self._nudged_budget_hint())
            nudged = replace(phases["nudged"].state, steps=phases["nudged"].steps)
            opposite = (replace(phases["opposite"].state, steps=phases["opposite"].steps)
                        if "opposite" in phases else None)
            report = self.update(free, nudged, opposite)
            report.update(_phase_report(phases, accepted=True, qualified=True))
            if recovery is not None:
                # Fit between lessons, from accepted settled phases only.
                for name in nudges:
                    recovery.observe(self, free, phases[name].state, nudges[name])
                # One rendering and one fitting pass per phase, each crossing every seam once.
                report["recovery_row_seam_passes"] = float(
                    2 * len(nudges) * recovery.depth * np.size(phases["free"].residual)
                )
            return LearnedState(free, nudged, opposite), report
        free = self.free(drive, warm)
        nudged = self.nudged(drive, free, target, weight=weight)
        opposite = (
            self.nudged(drive, free, target, sign=-1.0, weight=weight)
            if self.config.centered
            else None
        )
        # Finite phases retain the original update law. Their post-solve
        # diagnostics count additional work and do not certify the lesson.
        phases = {
            "free": Equilibrium(free, self.brain.residual(drive, free), free.steps,
                                cfg.tolerance if cfg.tolerance is not None else 0.0,
                                residual_checks=1),
            "nudged": Equilibrium(
                nudged, self.brain.residual(drive, nudged,
                    nudge=self.nudge_for(target, cfg.beta, weight)),
                nudged.steps, cfg.tolerance if cfg.tolerance is not None else 0.0,
                residual_checks=1,
            ),
        }
        if opposite is not None:
            phases["opposite"] = Equilibrium(
                opposite, self.brain.residual(drive, opposite,
                    nudge=self.nudge_for(target, -cfg.beta, weight)),
                opposite.steps, cfg.tolerance if cfg.tolerance is not None else 0.0,
                residual_checks=1,
            )
        report = self.update(free, nudged, opposite)
        report.update(_phase_report(phases, accepted=True, qualified=False))
        # Finite teaching updates from whatever state the budget reached. A free
        # phase that used its entire budget usually did not reach the movement
        # stop; the lesson was then learned from a state that may not have
        # settled. Count it and say so instead of hiding it (issue 127).
        exhausted = cfg.tolerance is not None and free.steps >= cfg.free_steps > 0
        report["free_budget_exhausted"] = float(exhausted)
        if exhausted:
            warnings.warn(
                "finite teaching free phase used its entire free_steps budget; the "
                "lesson may have been learned from a state that did not settle "
                "(free_residual in the report holds the equation residual). Raise "
                "free_steps, lower the gain, or opt into qualified=True to refuse "
                "such lessons; "
                "see docs/learning.md#finite-free-phases-that-run-out-of-budget.",
                RuntimeWarning,
                stacklevel=2,
            )
        return LearnedState(free, nudged, opposite), report

    # -- calibration

    def calibrate(
        self, drive: np.ndarray, *, level: float | None = None, grid: Sequence[float] | None = None
    ) -> float:
        """Choose the sampled gain whose free output operating point is nearest the target.

        Left unset, the target is competitive: the mean over rows and output
        slots of the highest output activation, at 0.5 — one choice up, the
        others wherever the competition puts them. An explicit ``level``
        instead targets the mean over every output neuron, which on a readout
        of n choices pushes all n toward that level together and saturates a
        competitive readout well before the winner is useful (issue 125).

        The default grid tries the current gain first, then its powers-of-two
        multiples from 1/256 to 256. An explicit grid retains its order and
        first-on-tie rule. Qualified learners admit only fully qualified, finite
        candidate states; finite learners keep their finite-phase contract.
        ``last_calibration`` retains every candidate and its work, including
        refusals. No admissible candidate raises ``RuntimeError`` without
        changing the graph or optimizer.

        Use training inputs before label-based selection. This bounded operating-
        point heuristic need not reach the target or make every output responsive.
        """
        drive = np.asarray(drive, dtype=float)
        if drive.ndim == 1:
            drive = drive[None, :]
        if (drive.ndim != 2 or drive.shape[0] == 0
                or drive.shape[1] != self.brain.connectome.n
                or not np.isfinite(drive).all()):
            raise ValueError("calibration needs a nonempty finite drive with one column per neuron")
        target = "mean" if level is not None else "top"
        if level is None:
            level = 0.5
        if np.ndim(level) != 0 or not np.isfinite(level):
            raise ValueError("calibration needs a finite scalar target level")
        if grid is None:
            current = self.brain.neuron_model.gain
            with np.errstate(over="ignore", under="ignore"):
                multiples = np.ldexp(current, np.array([k for k in range(-8, 9) if k]))
            candidates = [float(current), *(
                float(gain) for gain in multiples if np.isfinite(gain) and gain > 0
            )]
        else:
            values = np.asarray(list(grid), dtype=float)
            if values.ndim != 1:
                raise ValueError("calibration gains must be a one-dimensional sequence")
            candidates = values.tolist()
        if (
            not candidates
            or not np.isfinite(candidates).all()
            or min(candidates) <= 0
        ):
            raise ValueError("calibration needs finite positive gains")
        cfg = self.config
        attempts: list[dict[str, Any]] = []
        report: dict[str, Any] = {
            "qualification_required": cfg.qualified, "level": float(level), "target": target,
            "selected_gain": None, "candidates": attempts,
            "attempted_candidates": 0, "admitted_candidates": 0,
            "attempted_presentations": 0, "total_steps": 0,
            "total_residual_checks": 0, "total_stagnation_checks": 0,
            "total_activation_checks": 0, "total_row_activation_checks": 0,
            "total_row_sweeps": 0, "total_row_residual_checks": 0,
        }
        self.last_calibration = report
        best_brain, best_gap = None, float("inf")
        for gain in candidates:
            brain = self._with_gain(gain)
            if cfg.qualified:
                assert cfg.tolerance is not None
                phase = brain.equilibrate(
                    drive, budget=cfg.free_steps, tolerance=cfg.tolerance, damping=cfg.damping
                )
            else:
                state = brain.settle_batch(drive, steps=cfg.free_steps, tolerance=cfg.tolerance)
                phase = Equilibrium(
                    state, brain.residual(drive, state), state.steps,
                    cfg.tolerance if cfg.tolerance is not None else 0.0, residual_checks=1,
                )
            state = phase.state
            finite = all(np.isfinite(value).all() for value in (
                state.v, state.activation, state.adaptation, phase.residual
            ))
            rounding = 8 * np.finfo(
                np.float32 if brain._mlx is not None or (brain._torch is not None
                    and brain._torch.dtype == brain._torch.torch.float32)
                else np.float64
            ).eps
            consistent = finite and np.allclose(
                state.activation, brain.neuron_model.activation(state.v),
                rtol=rounding, atol=rounding,
            )
            qualified = bool(consistent and np.all(phase.qualified))
            admitted = bool(consistent and (qualified or not cfg.qualified))
            mean_out = top_out = None
            if admitted:
                outputs = state.activation[:, self.output_index]
                mean_out = float(outputs.mean())
                # One winner per output slot: the competitive operating point.
                top_out = float(np.maximum.reduceat(outputs, self.slot_offsets, axis=1).mean())
            metric = top_out if target == "top" else mean_out
            gap = abs(metric - level) if metric is not None else None
            rows = drive.shape[0]
            attempts.append({
                "gain": gain, "admitted": admitted, "qualified": qualified,
                "finite": bool(finite), "activation_consistent": bool(consistent),
                "mean_output": mean_out, "top_output": top_out, "gap": gap,
                "residual": [float(x) if np.isfinite(x) else None for x in phase.residual],
                "steps": phase.steps, "residual_checks": phase.residual_checks,
                "damping_halvings": phase.damping_halvings,
                "stagnation_checks": phase.stagnation_checks,
                "activation_checks": int(finite),
                "row_activation_checks": rows * int(finite),
                "row_sweeps": rows * phase.steps,
                "row_residual_checks": rows * phase.residual_checks,
            })
            report["attempted_candidates"] += 1
            report["admitted_candidates"] += int(admitted)
            report["attempted_presentations"] += rows
            for name, value in (
                ("steps", phase.steps), ("residual_checks", phase.residual_checks),
                ("stagnation_checks", phase.stagnation_checks),
                ("activation_checks", int(finite)),
                ("row_activation_checks", rows * int(finite)),
                ("row_sweeps", rows * phase.steps),
                ("row_residual_checks", rows * phase.residual_checks),
            ):
                report[f"total_{name}"] += value
            if gap is not None and gap < best_gap:
                best_brain, best_gap = brain, gap
        if best_brain is None:
            raise RuntimeError("no admissible calibration candidate; graph and optimizer unchanged")
        self.brain = best_brain
        selected = float(best_brain.neuron_model.gain)
        report["selected_gain"] = selected
        return selected

    def _with_gain(self, gain: float) -> Brain:
        return Brain(
            self.brain.connectome,
            self.brain.neuron_model.replace(gain=gain),
            backend=self.brain.backend,
            device=str(self.brain._torch.device) if self.brain._torch is not None else None,
            efficacy=(self.brain._torch.host_scale() if self.brain._efficacy is None
                      and self.brain._torch is not None else self.brain.efficacy),
            log_gain=self.brain.log_gain,
            bias=(self.brain._torch.host_bias() if self.brain._bias is None
                  and self.brain._torch is not None else self.brain.bias),
            dense_limit=self.brain.dense_limit,
            layout=self.brain.layout,
            precision=self.brain.precision,
        )

    # -- readout

    def predict(self, drive: np.ndarray) -> np.ndarray:
        """Most active output neuron after the free phase; ``(batch, slots)`` when slotted."""
        free = self.free(drive)
        out = free.activation[:, self.output_index]
        if self.slot_count > 1 and self.slot_size == 0:  # unequal slots: one argmax each
            choice = [
                np.argmax(out[:, o : o + k], axis=1)
                for o, k in zip(self.slot_offsets, self.slot_sizes, strict=True)
            ]
            return np.asarray(np.stack(choice, axis=1), dtype=np.int64)
        if self.slot_count > 1:
            out = out.reshape(len(out), self.slot_count, self.slot_size)
        return np.asarray(np.argmax(out, axis=-1), dtype=np.int64)

    def accuracy(self, drive: np.ndarray, labels: np.ndarray, batch: int = 256) -> float:
        """Fraction of correct class choices, averaged over rows and output slots."""
        labels = self._labels(labels)
        if labels.size == 0:
            raise ValueError("accuracy needs at least one labeled example")
        if not isinstance(batch, (int, np.integer)) or batch < 1:
            raise ValueError("batch must be a positive integer")
        drive = np.asarray(drive, dtype=float)
        if drive.ndim == 1:
            drive = drive[None, :]
        if drive.ndim != 2 or drive.shape[0] != len(labels):
            raise ValueError("drive and labels must have the same batch size")
        hits = 0
        for start in range(0, len(labels), batch):
            hits += int(
                (self.predict(drive[start : start + batch]) == labels[start : start + batch]).sum()
            )
        return hits / labels.size

    def parameters(self) -> int:
        """Trainable numbers: one per synapse (a tied pair or group counts once) plus the biases."""
        assert self.plastic_synapses is not None and self.plastic_neurons is not None
        synapses = self.brain.connectome.synapses
        keys = np.arange(synapses)
        paired = self.reverse >= 0
        keys[paired] = np.minimum(keys[paired], self.reverse[paired])
        if len(self._members):
            keys[self._members] = synapses + self._member_groups
        # one flag per distinct key, not a sort of every key: linear in the synapses
        counted = np.zeros(synapses + len(self._member_count), dtype=bool)
        counted[keys[self.plastic_synapses]] = True
        return int(np.count_nonzero(counted)) + int(np.count_nonzero(self.plastic_neurons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "outputs": [int(i) for i in self.output_index],
            "slots": [int(k) for k in self.slot_sizes],
            "updates": self.updates,
            "contrast_updates": self.contrast_updates,
            "parameters": self.parameters(),
            "brain": self.brain.to_dict(),
        }

    # -- checkpoints

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        """Write this learner, connectome and parameters included, to one ``.npz`` file;
        ``compressed=False`` trades disk space for a fast write of a very large brain."""
        from .checkpoint import save

        return save(self, path, compressed=compressed)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backend: Backend | None = None,
        device: str | None = None,
        config: LearnerConfig | None = None,
        precision: str | None = None,
    ) -> Learner:
        """Rebuild a learner from ``save``; see ``cadence.checkpoint.load``."""
        from .checkpoint import load

        return load(path, backend=backend, device=device, config=config, precision=precision)


def _moment_property(name: str) -> property:
    """Keep the existing mutable NumPy attributes while avoiding per-step transfers."""

    def get(learner: Learner) -> np.ndarray:
        learner._host_moments()
        return cast(np.ndarray, learner.__dict__["_" + name])

    def put(learner: Learner, value: np.ndarray) -> None:
        learner._host_moments()
        learner.__dict__["_" + name] = np.asarray(value, dtype=float)

    return property(get, put)


for _moment_name in _MOMENTS:
    setattr(Learner, _moment_name, _moment_property(_moment_name))


def layered(
    inputs: int,
    hidden: int,
    outputs: int,
    *,
    density: float = 0.3,
    feedback: float = 1.0,
    lateral: float = 0.0,
    seed: int = 0,
    count: float = 1.0,
    init: float = 1.0,
    skip: bool = False,
    skip_init: float | None = None,
    excitatory_forward: bool = False,
) -> Connectome:
    """Input, hidden, and output neurons with forward, feedback, and lateral synapses.

    Forward synapses run input to hidden (random, ``density``) and hidden to
    output (dense); feedback synapses run output to hidden with scale
    ``feedback`` so the nudge can reach the hidden neurons; lateral synapses
    among the outputs carry ``lateral`` (negative: winner takes most); with
    ``skip`` every input also reaches every output directly. ``skip_init`` overrides
    the initialization scale of only those direct projections; ``None`` uses ``init``.
    Zero keeps the new projections trainable while preserving the original effective
    weights and initial predictions.
    Forward synapses start with random signs and fan-scaled magnitudes times
    ``init`` (positive only when ``excitatory_forward``); each feedback
    synapse starts equal to its forward partner scaled by ``feedback``, so a
    symmetric learner sees one synapse with one weight. Sets: ``input``,
    ``hidden``, ``output``.
    """
    for name, value in (("inputs", inputs), ("hidden", hidden), ("outputs", outputs)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= density <= 1:
        raise ValueError("density must lie in [0, 1]")
    if not np.isfinite([feedback, lateral, count, init]).all() or min(count, init) < 0:
        raise ValueError("connection settings must be finite; count and init nonnegative")
    if skip_init is not None and (not np.isfinite(skip_init) or skip_init < 0):
        raise ValueError("skip_init must be finite and nonnegative, or None")
    rng = np.random.default_rng(seed)
    n = inputs + hidden + outputs
    i0, h0, o0 = 0, inputs, inputs + hidden
    pre: list[np.ndarray] = []
    post: list[np.ndarray] = []
    sign: list[np.ndarray] = []

    def forward_signs(size: int, fan_in: float, fan_out: float, scale: float = init) -> np.ndarray:
        # Fan-scaled magnitudes (Glorot), so a fresh layer's drive is of order one.
        if size == 0:
            return np.zeros(0)
        magnitude = rng.uniform(0.0, 1.0, size=size) * np.sqrt(6.0 / (fan_in + fan_out)) * scale
        if excitatory_forward:
            return np.asarray(magnitude, dtype=float)
        return np.asarray(rng.choice([-1.0, 1.0], size=size) * magnitude, dtype=float)

    # input -> hidden, sparse
    mask = rng.random((inputs, hidden)) < density
    in_rows, hid_cols = np.nonzero(mask)
    pre.append(i0 + in_rows)
    post.append(h0 + hid_cols)
    sign.append(forward_signs(len(in_rows), inputs * density, hidden * density))
    # hidden -> output, dense
    grid_h, grid_o = np.meshgrid(np.arange(hidden), np.arange(outputs), indexing="ij")
    pre.append(h0 + grid_h.ravel())
    post.append(o0 + grid_o.ravel())
    sign.append(forward_signs(hidden * outputs, hidden, outputs))
    # output -> hidden feedback, dense, the same sign as its forward partner, scaled
    pre.append(o0 + grid_o.ravel())
    post.append(h0 + grid_h.ravel())
    sign.append(feedback * sign[-1])
    if skip:  # input -> output, dense
        grid_i, grid_o2 = np.meshgrid(np.arange(inputs), np.arange(outputs), indexing="ij")
        pre.append(i0 + grid_i.ravel())
        post.append(o0 + grid_o2.ravel())
        sign.append(
            forward_signs(
                inputs * outputs, inputs, outputs, init if skip_init is None else skip_init
            )
        )
    # output <-> output lateral
    a, b = np.meshgrid(np.arange(outputs), np.arange(outputs), indexing="ij")
    keep = a.ravel() != b.ravel()
    pre.append(o0 + a.ravel()[keep])
    post.append(o0 + b.ravel()[keep])
    sign.append(np.full(int(keep.sum()), lateral))
    return Connectome.from_synapses(
        n,
        pre=np.concatenate(pre),
        post=np.concatenate(post),
        count=np.full(sum(len(p) for p in pre), count),
        sign=np.concatenate(sign),
        populations={
            "input": range(i0, h0),
            "hidden": range(h0, o0),
            "output": range(o0, n),
        },
        label=f"layered:{inputs}x{hidden}x{outputs}",
    )


def embedded(
    vocabulary: int,
    positions: int,
    dim: int,
    hidden: int,
    outputs: int,
    *,
    seed: int = 0,
    init: float = 1.0,
) -> tuple[Connectome, np.ndarray]:
    """A window of one-hot tokens through a shared embedding: the connectome and its tie groups.

    Input neurons: ``positions`` blocks of ``vocabulary`` one-hot neurons. Embedding neurons:
    ``positions`` blocks of ``dim`` neurons; the synapse from token ``t`` at any position to
    embedding unit ``d`` of that position is one tie group, so every position reads the
    same embedding. Then a dense layer from all embedding neurons to ``hidden`` neurons, and
    from those to ``outputs``, with feedback synapses tied in pairs as in ``layered``. Pass
    the tie groups to ``Learner(tie_groups=...)``. Sets: ``input``, ``embedding``,
    ``hidden``, ``output``.
    """
    for name, value in (
        ("vocabulary", vocabulary),
        ("positions", positions),
        ("dim", dim),
        ("hidden", hidden),
        ("outputs", outputs),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(init) or init < 0:
        raise ValueError("init must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    n_in, n_emb = positions * vocabulary, positions * dim
    i0, e0, h0, o0 = 0, n_in, n_in + n_emb, n_in + n_emb + hidden
    n = o0 + outputs
    pre: list[np.ndarray] = []
    post: list[np.ndarray] = []
    sign: list[np.ndarray] = []
    tie: list[np.ndarray] = []
    # the shared embedding: the same random start for every position, one group per (token, unit)
    shape = (vocabulary, dim)
    table = rng.choice([-1.0, 1.0], size=shape) * rng.uniform(0.0, 1.0, size=shape)
    table = table * np.sqrt(6.0 / (1 + dim)) * init
    tok, unit = np.meshgrid(np.arange(vocabulary), np.arange(dim), indexing="ij")
    for p in range(positions):
        pre.append(i0 + p * vocabulary + tok.ravel())
        post.append(e0 + p * dim + unit.ravel())
        sign.append(table.ravel())
        tie.append((tok * dim + unit).ravel())

    def magnitude(size: int, fan_in: float, fan_out: float) -> np.ndarray:
        signs = rng.choice([-1.0, 1.0], size=size) * rng.uniform(0.0, 1.0, size=size)
        return np.asarray(signs * np.sqrt(6.0 / (fan_in + fan_out)) * init, dtype=float)

    emb, hid = np.meshgrid(np.arange(n_emb), np.arange(hidden), indexing="ij")
    forward = magnitude(n_emb * hidden, n_emb, hidden)
    pre += [e0 + emb.ravel(), h0 + hid.ravel()]
    post += [h0 + hid.ravel(), e0 + emb.ravel()]
    sign += [forward, forward]
    tie += [np.full(n_emb * hidden, -1), np.full(n_emb * hidden, -1)]
    hid2, out = np.meshgrid(np.arange(hidden), np.arange(outputs), indexing="ij")
    forward = magnitude(hidden * outputs, hidden, outputs)
    pre += [h0 + hid2.ravel(), o0 + out.ravel()]
    post += [o0 + out.ravel(), h0 + hid2.ravel()]
    sign += [forward, forward]
    tie += [np.full(hidden * outputs, -1), np.full(hidden * outputs, -1)]
    connectome = Connectome.from_synapses(
        n,
        pre=np.concatenate(pre),
        post=np.concatenate(post),
        sign=np.concatenate(sign),
        populations={
            "input": range(i0, e0),
            "embedding": range(e0, h0),
            "hidden": range(h0, o0),
            "output": range(o0, n),
        },
        label=f"embedded:{positions}x{vocabulary}->{dim}->{hidden}->{outputs}",
    )
    # Connectome.from_synapses sorts synapses; recover each synapse's tie group by (pre, post)
    all_pre, all_post, all_tie = np.concatenate(pre), np.concatenate(post), np.concatenate(tie)
    key: dict[tuple[int, int], int] = {
        (int(a), int(b)): int(g) for a, b, g in zip(all_pre, all_post, all_tie, strict=True)
    }
    groups = np.array(
        [key[(int(a), int(b))] for a, b in zip(connectome.pre, connectome.post, strict=True)],
        dtype=np.int64,
    )
    return connectome, groups


def learning_neuron_model(
    gain: float = 1.0,
    *,
    slope: float = 1.0,
    leak: float = 0.1,
    dt: float = 0.5,
    stimulus_amplitude: float = 1.0,
) -> NeuronModel:
    """A graded rule for nets that learn: unit slope, a zero threshold, a small leak, unit stimulus.

    At threshold zero the re-based sigmoid emits nothing at rest and rises
    smoothly for positive drive; with unit slope it saturates only around a
    drive of four, so a fan-scaled layer sits in the responsive range rather
    than silent or saturated. The leak lets a neuron below rest publish a
    small negative activation, so the rule can still move it. A unit stimulus
    amplitude makes an input neuron read its level roughly linearly, and the
    larger step converges in tens of steps rather than hundreds.
    """
    return NeuronModel(
        dt=dt,
        slope=slope,
        threshold=0.0,
        gain=gain,
        stimulus_amplitude=stimulus_amplitude,
        leak=leak,
    )


def calibrate_bias(
    brain: Brain,
    drives: np.ndarray,
    targets: Mapping[str | Sequence[int], float],
    *,
    per_neuron: bool = False,
    rounds: int = 3,
    span: tuple[float, float] = (-6.0, 6.0),
    iterations: int = 16,
    steps: int = 100,
    tolerance: float | None = 1e-4,
    qualified: bool = False,
    damping: int = 3,
    report: dict[str, Any] | None = None,
) -> np.ndarray:
    """Search for biases near declared population activity targets under a set of drives.

    ``targets`` maps a population (a name in the connectome, or neuron indices) to the mean
    activation it should have, averaged over its members and over ``drives`` (rows of
    stimulus drive as ``settle_batch`` takes them: the situations the brain will meet).
    One shared bias per population, the population's threshold, or one per member with
    ``per_neuron`` (a readout's cells are read one by one, so each is centred). Each bias is
    found by bisection within ``span``, every population in turn, ``rounds`` times, so that
    populations that feed each other settle jointly; a target no bias in the span reaches
    leaves the bias at the span's end, and the caller checks the returned means. Returns the
    bias array, the brain's own plus the calibration; the brain itself is unchanged.

    Every midpoint and final state must be finite and have consistent activation,
    including finite searches without a report. Finite bisection is the default.
    With ``qualified=True``, every midpoint and
    the final cold state must satisfy the original equations within ``steps``
    sweeps, using up to ``damping`` numerical halvings. A refused solve raises
    ``RuntimeError`` before its activity can steer the search. ``report`` retains
    every attempted solve, including refusals, its work and final target gaps.
    Numerical qualification does not guarantee that the targets are reached.

    This is the operating point a connectome does not carry: the wiring says who talks to
    whom, not how excitable each cell is. A threshold cannot tame a runaway loop (the fly's
    antennal lobe stayed ignited at a bias of -6: that needs a gain, ``Brain(log_gain=...)``
    selected by protocol). A qualified responsive candidate can put a readout in
    the range where a nudge has a slope; inspect its final means and target gaps.
    """
    for name, value, minimum in (
        ("rounds", rounds, 1), ("iterations", iterations, 1),
        ("steps", steps, 0), ("damping", damping, 0),
    ):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(per_neuron, (bool, np.bool_)) or not isinstance(qualified, (bool, np.bool_)):
        raise ValueError("per_neuron and qualified must be boolean")
    def finite_scalar(value: Any) -> bool:
        scalar = np.asarray(value)
        return scalar.ndim == 0 and scalar.dtype.kind in "iuf" and bool(np.isfinite(scalar))

    if tolerance is None:
        if qualified:
            raise ValueError("qualified bias calibration requires a finite tolerance")
    elif not finite_scalar(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    bounds = np.asarray(span, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("span must contain two finite increasing bounds")
    if report is not None and not isinstance(report, dict):
        raise ValueError("report must be a dictionary or None")
    drives = np.atleast_2d(np.asarray(drives, dtype=float))
    n = brain.connectome.n
    if (drives.ndim != 2 or drives.shape[0] == 0 or drives.shape[1] != n
            or not np.isfinite(drives).all()):
        raise ValueError(
            "bias calibration needs a nonempty finite drive with one column per neuron"
        )
    if not isinstance(targets, Mapping):
        raise ValueError("targets must map populations to activity levels")
    groups: list[tuple[np.ndarray, float]] = []
    descriptions: list[dict[str, Any]] = []
    for who, target in targets.items():
        if isinstance(who, str):
            if who not in brain.connectome.populations:
                raise ValueError(f"unknown population {who!r}")
            idx = np.asarray(brain.connectome.populations[who], dtype=np.int64)
        else:
            try:
                values = np.asarray(list(who))
            except TypeError as exc:
                raise ValueError("population indices must be a sequence of integers") from exc
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError("population indices must be a sequence of integers")
            idx = values.astype(np.int64)
        if idx.size == 0 or np.unique(idx).size != idx.size:
            raise ValueError(f"population {who!r} must contain distinct nonempty indices")
        if (idx < 0).any() or (idx >= n).any():
            raise ValueError(f"population {who!r} lies outside the connectome")
        if not finite_scalar(target) or not 0.0 <= target <= 1.0:
            raise ValueError("targets are activations in [0, 1]")
        members = [np.array([i]) for i in idx] if per_neuron else [idx]
        for member in members:
            groups.append((member, float(target)))
            descriptions.append({
                "population": who if isinstance(who, str) else None,
                "indices": member.tolist(), "target": float(target),
            })
    receipt: dict[str, Any] = report if report is not None else {}
    receipt.clear()
    receipt.update({
        "qualification_required": bool(qualified), "completed": False,
        "per_neuron": bool(per_neuron), "rounds": int(rounds), "iterations": int(iterations),
        "steps": int(steps), "tolerance": tolerance, "damping": int(damping),
        "span": bounds.tolist(),
        "targets": descriptions, "solves": [], "final_means": [], "target_gaps": [],
        "attempted_solves": 0, "admitted_solves": 0,
        "attempted_trials": 0, "admitted_trials": 0, "attempted_presentations": 0,
        "total_steps": 0, "total_residual_checks": 0, "total_stagnation_checks": 0,
        "total_activation_checks": 0, "total_row_activation_checks": 0,
        "total_row_sweeps": 0, "total_row_residual_checks": 0,
    })
    # Reading resident parameters directly keeps the source graph's lazy host
    # fields untouched, including when the first qualified trial refuses.
    efficacy = (brain._torch.host_scale() if brain._efficacy is None and brain._torch is not None
                else brain.efficacy)
    bias = np.array(
        brain._torch.host_bias()
        if brain._bias is None and brain._torch is not None else brain.bias,
        dtype=float,
    )

    def candidate(b: np.ndarray) -> Brain:
        return Brain(
            brain.connectome, brain.neuron_model, backend=brain.backend,
            efficacy=efficacy, log_gain=brain.log_gain, bias=b,
            device=str(brain._torch.device) if brain._torch is not None else None,
            dense_limit=brain.dense_limit, layout=brain.layout, precision=brain.precision,
        )

    def solve(b: np.ndarray, *, kind: str, round_index: int, group: int | None) -> BrainState:
        trial = candidate(b)
        phase: Equilibrium | None = None
        if qualified:
            assert tolerance is not None
            phase = trial.equilibrate(drives, budget=steps, tolerance=tolerance, damping=damping)
            state = phase.state
        else:
            state = trial.settle_batch(drives, steps=steps, tolerance=tolerance)
            if report is not None:
                phase = Equilibrium(
                    state, trial.residual(drives, state), state.steps,
                    tolerance if tolerance is not None else 0.0, residual_checks=1,
                )
        values = [state.v, state.activation, state.adaptation]
        if phase is not None:
            values.append(phase.residual)
        finite = all(np.isfinite(value).all() for value in values)
        rounding = 8 * np.finfo(
            np.float32 if trial._mlx is not None or (trial._torch is not None
                and trial._torch.dtype == trial._torch.torch.float32) else np.float64
        ).eps
        consistent = bool(finite and np.allclose(
            state.activation, trial.neuron_model.activation(state.v),
            rtol=rounding, atol=rounding,
        ))
        if phase is None:
            # A finite search need not meet an equation tolerance, but invalid
            # numerical states cannot decide which half of the bias span survives.
            if not consistent:
                raise RuntimeError(f"invalid bias calibration {kind}; original graph unchanged")
            return state
        passed = bool(consistent and np.all(phase.qualified))
        admitted = bool(consistent and (passed or not qualified))
        rows = len(drives)
        counts = {
            "steps": phase.steps, "residual_checks": phase.residual_checks,
            "stagnation_checks": phase.stagnation_checks, "activation_checks": int(finite),
            "row_activation_checks": rows * int(finite), "row_sweeps": rows * phase.steps,
            "row_residual_checks": rows * phase.residual_checks,
        }
        receipt["solves"].append({
            "kind": kind, "round": round_index, "group": group,
            "trial_bias": float(b[groups[group][0][0]]) if group is not None else None,
            "mean_output": (float(state.activation[:, groups[group][0]].mean())
                            if admitted and group is not None else None),
            "admitted": admitted, "qualified": passed, "finite": bool(finite),
            "activation_consistent": consistent, "damping_halvings": phase.damping_halvings,
            "residual": [float(x) if np.isfinite(x) else None for x in phase.residual],
            **counts,
        })
        receipt["attempted_solves"] += 1
        receipt["admitted_solves"] += int(admitted)
        receipt["attempted_trials"] += int(kind == "midpoint")
        receipt["admitted_trials"] += int(kind == "midpoint" and admitted)
        receipt["attempted_presentations"] += rows
        for name, value in counts.items():
            receipt[f"total_{name}"] += value
        if not admitted:
            reason = "unqualified" if qualified else "invalid"
            raise RuntimeError(f"{reason} bias calibration {kind}; original graph unchanged")
        return state

    for round_index in range(rounds):
        for group, (idx, target) in enumerate(groups):
            lo, hi = bounds
            for _ in range(iterations):
                mid = 0.5 * lo + 0.5 * hi
                trial = bias.copy()
                trial[idx] = mid
                state = solve(trial, kind="midpoint", round_index=round_index, group=group)
                if float(state.activation[:, idx].mean()) < target:
                    lo = mid
                else:
                    hi = mid
            bias[idx] = 0.5 * lo + 0.5 * hi
    final = solve(bias, kind="final", round_index=rounds, group=None)
    if qualified or report is not None:
        receipt["final_means"] = [float(final.activation[:, idx].mean()) for idx, _ in groups]
        receipt["target_gaps"] = [
            abs(mean - target)
            for mean, (_, target) in zip(receipt["final_means"], groups, strict=True)
        ]
    receipt["completed"] = True
    return bias


def naive_efficacy(connectome: Connectome, plastic: np.ndarray) -> np.ndarray:
    """Efficacies that give every plastic synapse class the same weight.

    On the plastic set the efficacy is the class's sign times the set's mean count over the
    class's own count, so ``gain * count * efficacy`` is one number for every class; the
    other synapses keep their sign. A specimen's synapse counts at a memory site are that
    specimen's memories (the fly's Kenyon-cell-to-MBON counts made one odour aversive and the
    other attractive before any lesson); a lesson that should start naive starts here.
    """
    plastic = np.asarray(plastic, dtype=bool)
    if plastic.shape != (connectome.synapses,):
        raise ValueError("plastic must have one entry per synapse")
    efficacy = np.array(connectome.sign, dtype=float)
    if plastic.any():
        counts = connectome.count[plastic]
        efficacy[plastic] = connectome.sign[plastic] * counts.mean() / counts
    return efficacy


def seam_report(
    connectome: Connectome, pre: str | Sequence[int], post: str | Sequence[int]
) -> dict[str, Any]:
    """The plastic seam from ``pre`` to ``post``: its classes and synapses, and its coverage.

    ``coverage_pre`` is the fraction of pre neurons with a class onto some post neuron,
    ``classes_per_post`` the number of classes into each post neuron (by index). A seam
    thinned by custody shows here: a synapse floor that removes noise elsewhere removes a
    distributed memory, whose synapses are many and weak (the fly's floor of five kept 231
    of 1,079 Kenyon cell classes onto one output neuron and 14 of 336 onto the other).
    """

    def members(who: str | Sequence[int]) -> np.ndarray:
        return (
            np.asarray(connectome.populations[who], dtype=np.int64)
            if isinstance(who, str)
            else np.asarray(list(who), dtype=np.int64)
        )

    a, b = members(pre), members(post)
    in_pre = np.zeros(connectome.n, dtype=bool)
    in_pre[a] = True
    in_post = np.zeros(connectome.n, dtype=bool)
    in_post[b] = True
    seam = in_pre[connectome.pre] & in_post[connectome.post]
    per_post = np.bincount(connectome.post[seam], minlength=connectome.n)
    senders = np.unique(connectome.pre[seam])
    return {
        "classes": int(seam.sum()),
        "synapses": float(connectome.count[seam].sum()),
        "coverage_pre": float(len(senders) / len(a)) if len(a) else 0.0,
        "classes_per_post": {int(i): int(per_post[i]) for i in b},
        "median_count": float(np.median(connectome.count[seam])) if seam.any() else 0.0,
    }


def preflight(
    brain: Brain,
    outputs: Sequence[int],
    plastic: np.ndarray,
    drives: np.ndarray,
    *,
    level: float = 0.5,
    saturation: float = 0.02,
    shared_max: float = 0.25,
    eligibility_min: int = 10,
    steps: int = 100,
    tolerance: float | None = 1e-4,
) -> dict[str, Any]:
    """The checks a lesson needs before its first decision, each with the block that repairs it.

    The brain settles under ``drives`` (rows of stimulus drive, the situations it will decide
    in) and three conditions are read, the ones that stopped the fruit fly from learning a
    discrimination on its measured wiring:

    - a readout in ``outputs`` within ``saturation`` of 0 or 1 under every drive has no slope
      for a nudge, so no lesson moves it: ``calibrate_bias``;
    - the code of the plastic synapses' senders (the members at or above ``level``) shared
      between two drives beyond ``shared_max`` of the union carries nothing a lesson can attach
      to one situation and not the other: a population upstream ignites at the global gain,
      and its gain is selected by protocol on the ``specific`` fact (``Brain(log_gain=...)``);
    - fewer than ``eligibility_min`` plastic synapses from active senders onto a readout under a
      drive leaves the lesson almost no eligibility there: the seam is thin, most often cut by
      a synapse floor in custody (``seam_report``);
    - two drives whose driven neurons overlap beyond ``shared_max`` of their union are the
      same situation to the brain before it settles anything: the senses, not the wiring, are
      to be told apart (the fly's two odours reached both receptor classes near saturation at
      either fruit, from one wide odour field).

    Returns ``{"outputs", "shared", "inputs", "eligibility", "warnings"}``: the readings and one
    warning per finding, in words, naming the remedy. An empty ``warnings`` list is what a
    receipt should show before the lessons start.
    """
    C = brain.connectome
    plastic = np.asarray(plastic, dtype=bool)
    if plastic.shape != (C.synapses,):
        raise ValueError("plastic must have one entry per synapse")
    drives = np.atleast_2d(np.asarray(drives, dtype=float))
    outputs = [int(i) for i in outputs]
    act = brain.settle_batch(drives, steps=steps, tolerance=tolerance).activation
    warnings: list[str] = []
    readouts: dict[int, dict[str, Any]] = {}
    for i in outputs:
        lo, hi = float(act[:, i].min()), float(act[:, i].max())
        rail = bool(np.all((act[:, i] <= saturation) | (act[:, i] >= 1.0 - saturation)))
        readouts[i] = {"min": lo, "max": hi, "saturated": rail}
        if rail:
            warnings.append(
                f"output {i} sits within {saturation} of 0 or 1 under every drive (from {lo:.2f} "
                f"to {hi:.2f}): no slope for a nudge; calibrate_bias"
            )
    senders = np.unique(C.pre[plastic])
    codes = act[:, senders] >= level
    driven = drives > 0.0
    shared: dict[tuple[int, int], float] = {}
    inputs: dict[tuple[int, int], float] = {}
    for a in range(len(drives)):
        for b in range(a + 1, len(drives)):
            union = int((driven[a] | driven[b]).sum())
            same = float((driven[a] & driven[b]).sum() / union) if union else 0.0
            inputs[(a, b)] = same
            if union and same > shared_max:
                warnings.append(
                    f"drives {a} and {b} drive {same:.2f} of the same neurons (at most"
                    f" {shared_max}): the same situation before the brain settles anything;"
                    " make the senses tell them apart"
                )
            union = int((codes[a] | codes[b]).sum())
            share = float((codes[a] & codes[b]).sum() / union) if union else 0.0
            shared[(a, b)] = share
            if union and share > shared_max:
                warnings.append(
                    f"the plastic senders' code is {share:.2f} shared between drives {a} and {b}"
                    f" (at most {shared_max}): a population upstream ignites at the global gain;"
                    " select its gain by protocol on the specific fact (Brain(log_gain=...))"
                )
    eligibility: dict[int, list[int]] = {}
    for i in outputs:
        onto = plastic & (C.post == i)
        pre = C.pre[onto]
        counts = [int((act[r, pre] >= level).sum()) for r in range(len(drives))]
        eligibility[i] = counts
        thin = [r for r, n in enumerate(counts) if n < eligibility_min]
        if thin:
            warnings.append(
                f"output {i} has {int(onto.sum())} plastic classes and under drive"
                f"{'s' if len(thin) > 1 else ''} {', '.join(map(str, thin))} fewer than"
                f" {eligibility_min} come from active senders: the seam is thin; seam_report,"
                " and keep it whole in custody"
            )
    return {
        "outputs": readouts,
        "shared": shared,
        "inputs": inputs,
        "eligibility": eligibility,
        "warnings": warnings,
    }
