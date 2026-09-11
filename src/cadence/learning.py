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
from typing import Any

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
    normalize: float = 0.0  # >0: forgetting factor of the per-overlap RMS that divides its step
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
    updates: int = 0

    def __post_init__(self) -> None:
        self.output_index = np.asarray(list(self.outputs), dtype=np.int64)
        self.output_mask = np.zeros(self.engine.wiring.n)
        self.output_mask[self.output_index] = 1.0
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
        return Nudge(target, self.output_mask, beta, softmax_temperature=temperature, weight=weight)

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
        """Per-owner target activation for a batch of class labels."""
        batch = len(labels)
        target = np.zeros((batch, self.engine.wiring.n))
        target[:, self.output_index] = self.config.off_level
        target[np.arange(batch), self.output_index[np.asarray(labels)]] = self.config.target_level
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
        s_plus = nudged.activation
        if opposite is None:
            s_minus, span = free.activation, beta
        else:
            s_minus, span = opposite.activation, 2.0 * beta
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
        delta_scale = np.where(self.trainable_overlaps, delta_scale, 0.0)
        paired = self.reverse >= 0
        if paired.any():  # one seam, one weight: both directions move by the same amount
            delta_scale = np.where(
                paired, 0.5 * (delta_scale + delta_scale[np.maximum(self.reverse, 0)]), delta_scale
            )
        if self.tie_groups is not None:  # a group moves by the mean of its members' contrasts
            groups = self.tie_groups
            member = groups >= 0
            if member.any():
                count = np.bincount(groups[member])
                total = np.bincount(groups[member], weights=delta_scale[member])
                index = np.maximum(groups, 0)
                shared = total[index] / np.maximum(count[index], 1)
                delta_scale = np.where(member, shared, delta_scale)
        # tying never moves a frozen overlap
        delta_scale = np.where(self.trainable_overlaps, delta_scale, 0.0)
        scale = self.engine.edge_scale + delta_scale
        assert self.trainable_owners is not None
        delta_bias = np.where(self.trainable_owners, delta_bias, 0.0)
        bias = self.engine.bias + delta_bias
        if cfg.decay > 0:  # a leak on the seams: what is not relearned fades away
            scale = np.where(self.trainable_overlaps, scale * (1.0 - cfg.decay), scale)
            bias = np.where(self.trainable_owners, bias * (1.0 - cfg.decay), bias)
        magnitude = np.clip(np.abs(scale), cfg.scale_floor, cfg.scale_cap)
        scale = np.sign(scale) * magnitude
        self.engine = self.engine.with_parameters(edge_scale=scale, bias=bias)
        self.updates += 1
        return {
            "scale_step": float(np.abs(delta_scale).mean()),
            "bias_step": float(np.abs(delta_bias).mean()),
        }

    def update(
        self, free: SettledState, nudged: SettledState, opposite: SettledState | None = None
    ) -> dict[str, float]:
        """Move every trainable overlap and every owner on its own two-phase difference."""
        cfg = self.config
        overlap_term, owner_term = self.contrast(free, nudged, opposite)
        if cfg.normalize > 0:  # still local: an overlap reads only its own history
            rho = cfg.normalize
            self.second_moment = rho * self.second_moment + (1 - rho) * overlap_term**2
            self.second_moment_bias = rho * self.second_moment_bias + (1 - rho) * owner_term**2
            overlap_term = overlap_term / (np.sqrt(self.second_moment) + cfg.normalize_floor)
            owner_term = owner_term / (np.sqrt(self.second_moment_bias) + cfg.normalize_floor)
        if cfg.momentum > 0:  # still local: an overlap accumulates only its own contrast
            self.velocity = cfg.momentum * self.velocity + (1 - cfg.momentum) * overlap_term
            self.velocity_bias = cfg.momentum * self.velocity_bias + (1 - cfg.momentum) * owner_term
            overlap_term, owner_term = self.velocity, self.velocity_bias
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
        )

    # -- readout

    def predict(self, drive: np.ndarray) -> np.ndarray:
        """Class index of the most active output owner after a free settlement."""
        free = self.free(drive)
        return np.asarray(np.argmax(free.activation[:, self.output_index], axis=1), dtype=np.int64)

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
    ) -> Learner:
        """Rebuild a learner from ``save``; see ``cadence.checkpoint.load``."""
        from .checkpoint import load

        return load(path, backend=backend, device=device, config=config)


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
