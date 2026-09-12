"""The free/nudged learning rule.

Learning in a patch net is two settlements and one local comparison. The
net settles *free*, with only its input clamped; that settled state is its
answer, and nothing about the goal enters it. Then, starting from that
state, the net settles again *nudged*: the output owners feel an extra
drive toward the target, and the nudge spreads back over the feedback
overlaps into the rest of the net. Every overlap then moves its own scale
on the difference between what its two endpoints did in the two phases:

    scale[e] += eta / beta * ( s+[pre] * s+[post] - s0[pre] * s0[post] )
    bias[i]  += eta_b / beta * ( s+[i] - s0[i] )

That is a contrastive Hebbian rule. It reads two numbers per overlap and
one per owner, so it is as local as the settlement itself; and the goal
enters through exactly one door, the nudge of the second phase. With
``centered`` set the learner runs the nudge both ways, ``+beta`` and
``-beta``, from the same free state and contrasts those two; that cancels
the first-order error of a one-sided nudge and is what the digits example
uses.

For the rule to reach hidden owners the wiring needs feedback: an overlap
from the output owners back toward the hidden ones. ``layered`` builds such
a wiring, with forward and feedback overlaps tied into one seam per pair,
optional lateral inhibition among the outputs, and named sets for the
layers.

The rule is exact in a limit. When the overlaps are symmetric the
settlement descends an energy, and for a small nudge the contrast above is
the gradient of the nudge's loss with respect to the overlap scales. Two
things break that in practice, and ``learning_rule`` is chosen so they do
not: an owner below rest with a hard rectifier publishes nothing and cannot
be moved (hence a small ``leak``), and an owner on a steep sigmoid sits
either silent or saturated (hence unit slope). The digits example checks
the alignment of the rule against finite differences and records it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from .blocks import block_contrast
from .rules import GradedRule
from .settle import Backend, Nudge, SettledState, Settlement
from .wiring import Wiring

__all__ = ["Learner", "LearnerConfig", "layered", "embedded", "learning_rule", "LearnedState"]


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    beta: float = 0.1  # nudge strength
    eta: float = 0.2  # overlap learning rate, divided by beta in the update
    eta_bias: float = 0.02
    centered: bool = True  # contrast +beta against -beta rather than against the free state
    free_steps: int = 100  # most steps a free settlement may take
    nudged_steps: int = 50  # most steps a nudged settlement may take
    tolerance: float | None = 1e-4  # a settlement stops once no owner moves more than this
    scale_floor: float = 0.0  # magnitude an overlap may not fall below (0: may cross zero)
    scale_cap: float = 8.0  # magnitude an overlap may not exceed
    target_level: float = 1.0  # activation the target owner is nudged toward
    off_level: float = 0.0  # activation the other output owners are nudged toward
    nudge: str = "cross_entropy"  # "quadratic" or "cross_entropy"
    temperature: float = 0.2  # softmax temperature of the cross-entropy nudge
    # >0: forgetting factor of the per-overlap RMS of its raw contrast that divides its step;
    # with momentum this is the adaptive local step (Adam written per seam), bias-corrected
    normalize: float = 0.0
    normalize_floor: float = 1e-3  # added to the RMS so a quiet overlap does not blow up
    momentum: float = 0.0  # >0: each overlap steps on a running average of its own contrast
    decay: float = 0.0  # >0: every update shrinks each trainable overlap and bias by this fraction

    def __post_init__(self) -> None:
        if self.nudge not in ("quadratic", "cross_entropy"):
            raise ValueError("nudge must be 'quadratic' or 'cross_entropy'")
        if self.beta <= 0 or self.eta < 0 or self.eta_bias < 0:
            raise ValueError("beta must be positive and the learning rates nonnegative")
        if not 0 <= self.normalize < 1:
            raise ValueError("normalize is a forgetting factor in [0, 1)")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum lies in [0, 1)")
        if not 0 <= self.decay < 1:
            raise ValueError("decay is a fraction in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True)
class LearnedState:
    free: SettledState
    nudged: SettledState
    opposite: SettledState | None = None  # the -beta phase, when centered


@dataclass
class Learner:
    """Owns the trainable ``edge_scale`` and ``bias`` of a settlement and applies the rule."""

    engine: Settlement
    outputs: Sequence[int]
    config: LearnerConfig = field(default_factory=LearnerConfig)
    trainable_overlaps: np.ndarray | None = None  # bool per overlap; default all
    trainable_owners: np.ndarray | None = None  # bool per owner: whose bias moves and decays
    symmetric: bool = True  # an overlap and its reverse share one scale: one seam, one weight
    tie_groups: np.ndarray | None = None  # int per overlap (-1: none); a group shares one scale
    # the outputs as groups, each its own softmax choice, all nudged together: a count of
    # equal groups, or one size per group (a body's controller: a move of nine, a grip of two)
    slots: int | Sequence[int] = 1
    updates: int = 0

    def __post_init__(self) -> None:
        self.output_index = np.asarray(list(self.outputs), dtype=np.int64)
        self.output_mask = np.zeros(self.engine.wiring.n)
        self.output_mask[self.output_index] = 1.0
        if isinstance(self.slots, int):
            if self.slots < 1 or len(self.output_index) % self.slots:
                raise ValueError("slots must divide the number of output owners")
            sizes = [len(self.output_index) // self.slots] * self.slots
        else:
            sizes = [int(k) for k in self.slots]
            if not sizes or min(sizes) < 1 or sum(sizes) != len(self.output_index):
                raise ValueError("slot sizes must be positive and add up to the output owners")
        self.slot_sizes = np.asarray(sizes, dtype=np.int64)
        self.slot_offsets = np.concatenate([[0], np.cumsum(self.slot_sizes)[:-1]]).astype(np.int64)
        self.slot_count = len(sizes)  # ``slots`` keeps what was asked for
        self.slot_size = int(self.slot_sizes[0]) if len(set(sizes)) == 1 else 0  # equal, or not
        self.output_groups: np.ndarray | None = None
        if self.slot_count > 1:  # one softmax per slot: a whole utterance settles at once
            self.output_groups = np.full(self.engine.wiring.n, -1, dtype=np.int64)
            groups = np.repeat(np.arange(self.slot_count), self.slot_sizes)
            self.output_groups[self.output_index] = groups
        if self.trainable_overlaps is None:
            self.trainable_overlaps = np.ones(self.engine.wiring.edges, dtype=bool)
        if self.trainable_owners is None:
            self.trainable_owners = np.ones(self.engine.wiring.n, dtype=bool)
        w = self.engine.wiring
        self.second_moment = np.zeros(w.edges)  # per overlap, for normalized steps
        self.second_moment_bias = np.zeros(w.n)
        self.velocity = np.zeros(w.edges)  # per overlap, for momentum
        self.velocity_bias = np.zeros(w.n)
        self.reverse = np.full(w.edges, -1, dtype=np.int64)
        if self.symmetric and w.edges:
            key = w.post * w.n + w.pre
            reverse_key = w.pre * w.n + w.post
            order = np.argsort(key)
            hit = np.searchsorted(key[order], reverse_key)
            hit = np.minimum(hit, len(order) - 1)
            found = key[order][hit] == reverse_key
            self.reverse[found] = order[hit[found]]
        # Index arrays for the update, made once: the paired overlaps and the tied members,
        # so an update touches those and not every overlap of a wide net.
        self._paired = np.flatnonzero(self.reverse >= 0)
        self._paired_reverse = self.reverse[self._paired]
        self._members = np.zeros(0, dtype=np.int64)
        self._member_groups = np.zeros(0, dtype=np.int64)
        self._member_count = np.zeros(0)
        if self.tie_groups is not None:
            self._members = np.flatnonzero(np.asarray(self.tie_groups) >= 0)
            self._member_groups = np.asarray(self.tie_groups)[self._members]
            if len(self._members):
                self._member_count = np.bincount(self._member_groups).astype(float)

    # -- phases

    def free(self, drive: np.ndarray, warm: SettledState | None = None) -> SettledState:
        """Settle with the input clamped and nothing else: the net's own answer.

        ``warm`` starts the settlement from an earlier state instead of rest,
        say the state the same inputs settled to on the previous pass; the
        fixed point is the same, it is just reached in fewer steps.
        """
        cfg = self.config
        return self.engine.settle_batch(
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
        free: SettledState,
        target: np.ndarray,
        sign: float = 1.0,
        weight: np.ndarray | None = None,
    ) -> SettledState:
        """From the free state, settle with the output owners pulled toward ``target``.

        ``sign`` of -1 pushes them away instead: the opposite phase of a centered
        contrast. ``weight`` scales the pull row by row (an advantage, when the
        target is an action that was taken).
        """
        cfg = self.config
        return self.engine.settle_batch(
            drive,
            steps=cfg.nudged_steps,
            state=free,
            nudge=self.nudge_for(target, sign * cfg.beta, weight),
            tolerance=cfg.tolerance,
        )

    def targets(self, labels: np.ndarray) -> np.ndarray:
        """Per-owner target activation for class labels: ``(batch,)``, or ``(batch, slots)``."""
        labels = np.asarray(labels)
        batch = len(labels)
        target = np.zeros((batch, self.engine.wiring.n))
        target[:, self.output_index] = self.config.off_level
        if self.slot_count > 1:
            if labels.ndim != 2 or labels.shape[1] != self.slot_count:
                raise ValueError("a slotted learner takes one label per slot")
            if labels.min() < 0 or (labels >= self.slot_sizes[None, :]).any():
                raise ValueError("a slot's label lies outside its choices")
            owners = self.output_index[labels + self.slot_offsets[None, :]]
            target[np.arange(batch)[:, None], owners] = self.config.target_level
        else:
            target[np.arange(batch), self.output_index[labels]] = self.config.target_level
        return target

    # -- the rule

    def contrast(
        self, free: SettledState, nudged: SettledState, opposite: SettledState | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The per-overlap and per-owner differences the rule reads, already divided by beta.

        One-sided: ``(nudged - free) / beta``. Centered: ``(nudged - opposite) / (2 beta)``.
        """
        w = self.engine.wiring
        beta = self.config.beta
        minus_state, span = (free, beta) if opposite is None else (opposite, 2.0 * beta)
        on_device = self.engine.contrast_on_device(nudged, minus_state)
        if on_device is not None:  # both phases still on the accelerator: read the contrast there
            edges, owners = on_device
            assert nudged.device is not None
            batch = int(nudged.device["s"].shape[0])
            return edges / (batch * span), owners / (batch * span)
        s_plus, s_minus = nudged.activation, minus_state.activation
        if w.edges > 4 * w.n:  # the pairwise products as block products, read at the overlaps
            gram = block_contrast(self.engine.layout, s_plus, s_minus)
            return gram / (len(s_plus) * span), (s_plus - s_minus).mean(axis=0) / span
        hebb_plus = (s_plus[:, w.pre] * s_plus[:, w.post]).mean(axis=0)
        hebb_minus = (s_minus[:, w.pre] * s_minus[:, w.post]).mean(axis=0)
        return (hebb_plus - hebb_minus) / span, (s_plus - s_minus).mean(axis=0) / span

    def contrast_rows(
        self, free: SettledState, nudged: SettledState, opposite: SettledState | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The same differences as ``contrast``, one row per batch element: ``(batch, edges)``
        and ``(batch, n)``. What a per-row eligibility trace reads."""
        w = self.engine.wiring
        beta = self.config.beta
        s_plus = nudged.activation
        if opposite is None:
            s_minus, span = free.activation, beta
        else:
            s_minus, span = opposite.activation, 2.0 * beta
        hebb = s_plus[:, w.pre] * s_plus[:, w.post] - s_minus[:, w.pre] * s_minus[:, w.post]
        return hebb / span, (s_plus - s_minus) / span

    def apply(self, delta_scale: np.ndarray, delta_bias: np.ndarray) -> dict[str, float]:
        """Apply a per-overlap and per-owner step: masks, tying, decay, bounds; then set the engine.

        This is the last half of ``update``; a rule that computes its own step (a
        three-factor trace, say) hands it here so every learner shares one notion of
        which overlaps move, how a tied pair moves, and what bounds hold.
        """
        cfg = self.config
        assert self.trainable_overlaps is not None
        all_trainable = bool(self.trainable_overlaps.all())
        delta_scale = np.array(delta_scale, dtype=float)  # this update's own copy
        if not all_trainable:
            delta_scale[~self.trainable_overlaps] = 0.0
        if len(self._paired):  # one seam, one weight: both directions move by the same amount
            mean = 0.5 * (delta_scale[self._paired] + delta_scale[self._paired_reverse])
            delta_scale[self._paired] = mean
        if len(self._members):  # a group moves by the mean of its members' contrasts
            total = np.bincount(self._member_groups, weights=delta_scale[self._members])
            delta_scale[self._members] = (total / np.maximum(self._member_count, 1.0))[
                self._member_groups
            ]
        if not all_trainable:  # tying never moves a frozen overlap
            delta_scale[~self.trainable_overlaps] = 0.0
        scale = self.engine.edge_scale + delta_scale
        assert self.trainable_owners is not None
        delta_bias = np.where(self.trainable_owners, delta_bias, 0.0)
        bias = self.engine.bias + delta_bias
        if cfg.decay > 0:  # a leak on the seams: what is not relearned fades away
            scale = np.where(self.trainable_overlaps, scale * (1.0 - cfg.decay), scale)
            bias = np.where(self.trainable_owners, bias * (1.0 - cfg.decay), bias)
        if cfg.scale_floor > 0:
            magnitude = np.clip(np.abs(scale), cfg.scale_floor, cfg.scale_cap)
            scale = np.sign(scale) * magnitude
        else:  # the same bounds in one pass
            scale = np.clip(scale, -cfg.scale_cap, cfg.scale_cap)
        self.engine = self.engine.with_parameters(edge_scale=scale, bias=bias)
        self.updates += 1
        return {
            "scale_step": float(np.abs(delta_scale).mean()),
            "bias_step": float(np.abs(delta_bias).mean()),
        }

    def _device_kernel(self, plus: SettledState, minus: SettledState) -> Any:
        """The torch kernel both states rest on, when the update can stay on the device."""
        cfg = self.config
        kernel = self.engine._torch
        if kernel is None or kernel.layout is None or cfg.normalize > 0 or cfg.momentum > 0:
            return None
        for state in (plus, minus):
            if state.device is None or state.device.get("owner") is not kernel:
                return None
        return kernel

    def _device_indices(self, kernel: Any) -> dict[str, Any]:
        """Index tensors of the update on the kernel's device, made once per kernel and masks."""
        torch = kernel.torch
        assert self.trainable_overlaps is not None and self.trainable_owners is not None
        key = (id(kernel), id(self.trainable_overlaps), id(self.trainable_owners))
        cache = self.__dict__.setdefault("_device_cache", {})
        if cache.get("key") != key:
            dev = kernel.device

            def to(x: np.ndarray, dtype: Any = None) -> Any:
                return torch.from_numpy(np.ascontiguousarray(x)).to(dev, dtype)

            all_overlaps = bool(self.trainable_overlaps.all())
            all_owners = bool(self.trainable_owners.all())
            cache.clear()
            cache.update(
                key=key,
                paired=to(self._paired) if len(self._paired) else None,
                paired_reverse=to(self._paired_reverse) if len(self._paired) else None,
                members=to(self._members) if len(self._members) else None,
                member_groups=to(self._member_groups) if len(self._members) else None,
                member_count=to(self._member_count, kernel.param_dtype)
                if len(self._members)
                else None,
                overlaps=None if all_overlaps else to(self.trainable_overlaps, kernel.param_dtype),
                owners=None if all_owners else to(self.trainable_owners, kernel.param_dtype),
            )
        return cast(dict[str, Any], cache)

    def _apply_device(self, kernel: Any, delta_scale: Any, delta_bias: Any) -> dict[str, float]:
        """``apply`` on the device: the same masks, tying, decay and bounds, no host array."""
        torch, cfg = kernel.torch, self.config
        ix = self._device_indices(kernel)
        d = delta_scale.clone()
        if ix["overlaps"] is not None:
            d = d * ix["overlaps"]
        if (
            ix["paired"] is not None
        ):  # one seam, one weight: both directions move by the same amount
            d[ix["paired"]] = 0.5 * (d[ix["paired"]] + d[ix["paired_reverse"]])
        if ix["members"] is not None:  # a group moves by the mean of its members' contrasts
            total = torch.zeros(len(ix["member_count"]), dtype=d.dtype, device=d.device)
            total.index_add_(0, ix["member_groups"], d[ix["members"]])
            d[ix["members"]] = (total / torch.clamp(ix["member_count"], min=1.0))[
                ix["member_groups"]
            ]
        if ix["overlaps"] is not None:  # tying never moves a frozen overlap
            d = d * ix["overlaps"]
        scale = kernel.scale + d
        db = delta_bias if ix["owners"] is None else delta_bias * ix["owners"]
        bias = kernel.bias_param + db
        if cfg.decay > 0:  # a leak on the seams: what is not relearned fades away
            keep_scale = (
                1.0 - cfg.decay if ix["overlaps"] is None else 1.0 - cfg.decay * ix["overlaps"]
            )
            keep_bias = 1.0 - cfg.decay if ix["owners"] is None else 1.0 - cfg.decay * ix["owners"]
            scale = scale * keep_scale
            bias = bias * keep_bias
        if cfg.scale_floor > 0:
            magnitude = torch.clamp(scale.abs(), cfg.scale_floor, cfg.scale_cap)
            scale = torch.sign(scale) * magnitude
        else:
            scale = torch.clamp(scale, -cfg.scale_cap, cfg.scale_cap)
        self.engine = self.engine._with_device_parameters(scale, bias)
        self.updates += 1
        return {"scale_step": float(d.abs().mean()), "bias_step": float(db.abs().mean())}

    def update(
        self, free: SettledState, nudged: SettledState, opposite: SettledState | None = None
    ) -> dict[str, float]:
        """Move every trainable overlap and every owner on its own two-phase difference."""
        cfg = self.config
        minus_state, span = (free, cfg.beta) if opposite is None else (opposite, 2.0 * cfg.beta)
        kernel = self._device_kernel(nudged, minus_state)
        if kernel is not None:  # both phases rest on the torch device: the whole update stays there
            edges, owners = kernel.contrast_tensors(nudged.device["s"], minus_state.device["s"])
            norm = float(nudged.device["s"].shape[0]) * span
            return self._apply_device(kernel, cfg.eta * edges / norm, cfg.eta_bias * owners / norm)
        overlap_term, owner_term = self.contrast(free, nudged, opposite)
        raw_overlap, raw_owner = overlap_term, owner_term
        count = self.updates + 1  # this update's place in the history, for the corrections
        if cfg.momentum > 0:  # still local: an overlap accumulates only its own contrast
            m = cfg.momentum
            self.velocity = m * self.velocity + (1 - m) * overlap_term
            self.velocity_bias = m * self.velocity_bias + (1 - m) * owner_term
            correction = 1.0 - m**count  # a short history is an average over fewer updates
            overlap_term, owner_term = self.velocity / correction, self.velocity_bias / correction
        if cfg.normalize > 0:  # still local: an overlap reads only its own history
            rho = cfg.normalize
            self.second_moment = rho * self.second_moment + (1 - rho) * raw_overlap**2
            self.second_moment_bias = rho * self.second_moment_bias + (1 - rho) * raw_owner**2
            correction = 1.0 - rho**count
            rms = np.sqrt(self.second_moment / correction) + cfg.normalize_floor
            rms_bias = np.sqrt(self.second_moment_bias / correction) + cfg.normalize_floor
            overlap_term, owner_term = overlap_term / rms, owner_term / rms_bias
        delta_scale = cfg.eta * overlap_term
        delta_bias = cfg.eta_bias * owner_term
        return self.apply(delta_scale, delta_bias)

    def step(
        self,
        drive: np.ndarray,
        labels: np.ndarray,
        warm: SettledState | None = None,
        weight: np.ndarray | None = None,
    ) -> tuple[LearnedState, dict[str, float]]:
        """One free phase, the nudged phase(s), one update; returns the states and step sizes.

        ``weight`` is one number per row: how hard, and in which direction, that
        row's label is pulled. With actions as labels and advantages as weights
        this is the policy-gradient step, and the goal still enters only through
        the nudge.
        """
        target = self.targets(labels)
        free = self.free(drive, warm)
        nudged = self.nudged(drive, free, target, weight=weight)
        opposite = (
            self.nudged(drive, free, target, sign=-1.0, weight=weight)
            if self.config.centered
            else None
        )
        report = self.update(free, nudged, opposite)
        report["free_steps"] = float(free.steps)
        report["nudged_steps"] = float(nudged.steps)
        return LearnedState(free, nudged, opposite), report

    # -- calibration

    def calibrate(
        self, drive: np.ndarray, *, level: float = 0.5, grid: Sequence[float] | None = None
    ) -> float:
        """Pick the rule's gain so the free output activation on ``drive`` sits near ``level``.

        A net that is silent or saturated gives the rule nothing to compare.
        The gain is chosen on training inputs only, before any label is seen,
        and the chosen value is returned for the receipt.
        """
        candidates = list(grid) if grid is not None else [0.02 * 1.3**k for k in range(16)]
        best_gain, best_gap = candidates[0], float("inf")
        for gain in candidates:
            engine = self._with_gain(gain)
            mean_out = float(
                engine.settle_batch(
                    drive, steps=self.config.free_steps, tolerance=self.config.tolerance
                )
                .activation[:, self.output_index]
                .mean()
            )
            gap = abs(mean_out - level)
            if gap < best_gap:
                best_gain, best_gap = gain, gap
        self.engine = self._with_gain(best_gain)
        return best_gain

    def _with_gain(self, gain: float) -> Settlement:
        return Settlement(
            self.engine.wiring,
            self.engine.rule.replace(gain=gain),
            backend=self.engine.backend,
            edge_scale=self.engine.edge_scale,
            log_gain=self.engine.log_gain,
            bias=self.engine.bias,
            dense_limit=self.engine.dense_limit,
            layout=self.engine.layout,
            precision=self.engine.precision,
        )

    # -- readout

    def predict(self, drive: np.ndarray) -> np.ndarray:
        """Most active output owner after a free settlement; ``(batch, slots)`` when slotted."""
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
        hits = 0
        for start in range(0, len(labels), batch):
            hits += int(
                (self.predict(drive[start : start + batch]) == labels[start : start + batch]).sum()
            )
        return hits / len(labels)

    def parameters(self) -> int:
        """Trainable numbers: one per seam (a tied pair or group counts once) plus the biases."""
        assert self.trainable_overlaps is not None
        tied_twice = (self.reverse >= 0) & self.trainable_overlaps
        seams = int(self.trainable_overlaps.sum() - tied_twice.sum() // 2)
        if self.tie_groups is not None:
            member = (self.tie_groups >= 0) & self.trainable_overlaps
            seams -= int(member.sum()) - len(np.unique(self.tie_groups[member]))
        return seams + self.engine.wiring.n

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "outputs": [int(i) for i in self.output_index],
            "slots": [int(k) for k in self.slot_sizes],
            "updates": self.updates,
            "parameters": self.parameters(),
            "engine": self.engine.to_dict(),
        }

    # -- checkpoints

    def save(self, path: str | Path) -> Path:
        """Write this learner, wiring and parameters included, to one ``.npz`` file."""
        from .checkpoint import save

        return save(self, path)

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
    excitatory_forward: bool = False,
) -> Wiring:
    """Input, hidden, and output owners with forward, feedback, and lateral overlaps.

    Forward overlaps run input to hidden (random, ``density``) and hidden to
    output (dense); feedback overlaps run output to hidden with scale
    ``feedback`` so the nudge can reach the hidden owners; lateral overlaps
    among the outputs carry ``lateral`` (negative: winner takes most); with
    ``skip`` every input also reaches every output directly.
    Forward overlaps start with random signs and fan-scaled magnitudes times
    ``init`` (positive only when ``excitatory_forward``); each feedback
    overlap starts equal to its forward partner scaled by ``feedback``, so a
    symmetric learner sees one seam with one weight. Sets: ``input``,
    ``hidden``, ``output``.
    """
    rng = np.random.default_rng(seed)
    n = inputs + hidden + outputs
    i0, h0, o0 = 0, inputs, inputs + hidden
    pre: list[np.ndarray] = []
    post: list[np.ndarray] = []
    sign: list[np.ndarray] = []

    def forward_signs(size: int, fan_in: float, fan_out: float) -> np.ndarray:
        # Fan-scaled magnitudes (Glorot), so a fresh layer's drive is of order one.
        magnitude = rng.uniform(0.0, 1.0, size=size) * np.sqrt(6.0 / (fan_in + fan_out)) * init
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
        sign.append(forward_signs(inputs * outputs, inputs, outputs))
    # output <-> output lateral
    a, b = np.meshgrid(np.arange(outputs), np.arange(outputs), indexing="ij")
    keep = a.ravel() != b.ravel()
    pre.append(o0 + a.ravel()[keep])
    post.append(o0 + b.ravel()[keep])
    sign.append(np.full(int(keep.sum()), lateral))
    return Wiring.from_edges(
        n,
        pre=np.concatenate(pre),
        post=np.concatenate(post),
        count=np.full(sum(len(p) for p in pre), count),
        sign=np.concatenate(sign),
        sets={
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
) -> tuple[Wiring, np.ndarray]:
    """A window of one-hot tokens through a shared embedding: the wiring and its tie groups.

    Input owners: ``positions`` blocks of ``vocabulary`` one-hot owners. Embedding owners:
    ``positions`` blocks of ``dim`` owners; the seam from token ``t`` at any position to
    embedding unit ``d`` of that position is one tie group, so every position reads the
    same embedding. Then a dense layer from all embedding owners to ``hidden`` owners, and
    from those to ``outputs``, with feedback seams tied in pairs as in ``layered``. Pass
    the tie groups to ``Learner(tie_groups=...)``. Sets: ``input``, ``embedding``,
    ``hidden``, ``output``.
    """
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
    wiring = Wiring.from_edges(
        n,
        pre=np.concatenate(pre),
        post=np.concatenate(post),
        sign=np.concatenate(sign),
        sets={
            "input": range(i0, e0),
            "embedding": range(e0, h0),
            "hidden": range(h0, o0),
            "output": range(o0, n),
        },
        label=f"embedded:{positions}x{vocabulary}->{dim}->{hidden}->{outputs}",
    )
    # Wiring.from_edges sorts overlaps; recover each overlap's tie group by (pre, post)
    all_pre, all_post, all_tie = np.concatenate(pre), np.concatenate(post), np.concatenate(tie)
    key: dict[tuple[int, int], int] = {
        (int(a), int(b)): int(g) for a, b, g in zip(all_pre, all_post, all_tie, strict=True)
    }
    groups = np.array(
        [key[(int(a), int(b))] for a, b in zip(wiring.pre, wiring.post, strict=True)],
        dtype=np.int64,
    )
    return wiring, groups


def learning_rule(
    gain: float = 1.0,
    *,
    slope: float = 1.0,
    leak: float = 0.1,
    dt: float = 0.5,
    clamp_amplitude: float = 1.0,
) -> GradedRule:
    """A graded rule for nets that learn: unit slope, a zero threshold, a small leak, unit clamp.

    At threshold zero the re-based sigmoid emits nothing at rest and rises
    smoothly for positive drive; with unit slope it saturates only around a
    drive of four, so a fan-scaled layer sits in the responsive range rather
    than silent or saturated. The leak lets an owner below rest publish a
    small negative activation, so the rule can still move it. A unit clamp
    amplitude makes an input owner read its level roughly linearly, and the
    larger step converges in tens of steps rather than hundreds.
    """
    return GradedRule(
        dt=dt, slope=slope, threshold=0.0, gain=gain, clamp_amplitude=clamp_amplitude, leak=leak
    )
