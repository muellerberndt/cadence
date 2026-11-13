"""Persistent synaptic weights plus a fading, per-stream residual.

No experience buffer is stored. Repetition and salience change the persistent
key-to-value synapses using the actual observed value's local prediction error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .resident import _StoreOnDevice
from .stream import FastSynapses

if TYPE_CHECKING:
    from .plasticity import Maturity


@dataclass
class SynapticMemory(FastSynapses):
    """Two timescales of normalized delta plasticity on declared key/value ports.

    ``consolidated`` is one persistent matrix shared across streams. ``strength``
    is that matrix plus each stream's transient residual. On each observation the
    residual fades by ``decay``; a local slow update consolidates the observed
    association, and a fast correction restores the current observed value.
    Repetition accumulates in the slow matrix, even when the fast read is correct.
    Salience increases the slow write rate, capped at one. Unknown value components
    never become teaching targets. ``reset`` clears transient memory only; ``clear``
    explicitly erases persistent weights too.

    With a surprise gate (``gate_noise``, each record's noise scale, and ``maturity``, the
    brain's ``Maturity``; ``Brain.compose(maturity=True)`` sets both on its place records), a
    write's change of the shared matrix and of each writing stream's residual is multiplied,
    per action, by ``1 - (1 - m)(1 - g)``: ``m`` the maturity's factor, ``g`` the mean over the
    writing rows that observe the action of ``max(0, 1 - GATE_WIDTH sqrt(R) / |e|)``, ``e`` the
    row's error on the action against its stream's read before the write and ``R`` the noise
    of the records along the row's cue. A mature store (``m`` near 0) moves a record only by
    the part of its error beyond ``GATE_WIDTH`` noise scales; a young one (``m`` 1) writes as
    without the gate. Every other stream keeps its residual. Each writing row moves the noise
    of the records its cue reads toward its squared error, at ``GATE_RATE`` times the cue's
    squared weight on the record relative to its largest.

    With a fade (``fade`` above 0, ``Brain.compose(memory_fade=...)``), records nobody uses fade
    (#253). Each key coordinate ``k`` keeps a depth, the sum of the unit cue's magnitude
    ``|c_k|`` over every moment that cues it and every write that uses it, and a trace of its
    recent cueing, the moments' sum kept at ``1 - 1 / FADE_IDLE`` a moment. Every moment of life
    (``fade_moment``) the records along coordinate ``k`` keep
    ``1 - fade * i_k / (1 + (depth_k / FADE_SCALE) ** FADE_ORDER)`` of themselves,
    ``i_k = 1 / (1 + (recent_k / FADE_FLOOR) ** 2)`` the coordinate's idleness: a record in use
    does not fade, one idle for some fifteen thousand moments starts to, and a deeply used one
    fades far more slowly. Each stream keeps its residual around the shared records. No count is
    kept: the depth and the trace are sums of activity, as maturity's use is. With ``fade_keep``
    (cadence#336) the records along a coordinate whose depth has reached it are kept for good:
    they stop fading (the depth only grows, so a kept coordinate stays kept).
    """

    decay: float = 0.9
    rule: str = "delta"
    consolidation: float = 0.05
    consolidated: np.ndarray = field(init=False)
    gate_noise: np.ndarray | None = field(default=None, repr=False)
    maturity: Maturity | None = field(default=None, repr=False, compare=False)
    fade: float = 0.0
    fade_keep: float = 0.0
    fade_depth: np.ndarray | None = field(default=None, repr=False)
    fade_recent: np.ndarray | None = field(default=None, repr=False)

    GATE_WIDTH = 2.0
    GATE_RATE = 0.05
    # the fade's constants (``fade``): the depth at which a record fades at half the rate, the
    # law's order, the moments the recent trace keeps ``1 - 1 / FADE_IDLE`` over, and the trace
    # below which a coordinate counts as idle
    FADE_SCALE = 2000.0
    FADE_ORDER = 4.0
    FADE_IDLE = 2000.0
    FADE_FLOOR = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rule != "delta":
            raise ValueError("SynapticMemory uses normalized delta plasticity")
        if not np.isfinite(self.consolidation) or not 0 <= self.consolidation <= 1:
            raise ValueError("consolidation must lie in [0, 1]")
        if self.separator is not None and self.separator.center != 0:
            raise ValueError(
                "persistent memory requires a fixed separator coordinate system; "
                "set separator.center=0 and apply any fixed centering to the input"
            )
        self.consolidated = np.zeros((self.key_width, len(self.post)))
        if not np.isfinite(self.fade) or not 0 <= self.fade < 1:
            raise ValueError("memory_fade must lie in [0, 1)")
        self.fade = float(self.fade)
        if isinstance(self.fade_keep, bool) or not np.isfinite(self.fade_keep) \
                or self.fade_keep < 0:
            raise ValueError("fade_keep is the depth past which records are kept: 0 or more")
        self.fade_keep = float(self.fade_keep)
        if self.fade_keep and not self.fade:
            raise ValueError("fade_keep keeps records from the fade: it needs fade above 0")
        if self.fade and self.fade_depth is None:
            self.fade_depth = np.zeros(self.key_width)
            self.fade_recent = np.zeros(self.key_width)

    def fade_moment(self, key: np.ndarray | None) -> None:
        """One moment of life of a fading store (``fade``): the coordinates ``key`` cues
        (``(batch, pre)``, the moment's place key; None for a moment that cues none) deepen and
        are marked recent, then every idle record fades by its depth's law and the recent trace
        decays one moment."""
        if not self.fade:
            return
        assert self.fade_depth is not None and self.fade_recent is not None
        if key is not None:
            k = np.atleast_2d(np.asarray(key, dtype=float))
            if k.shape[1] == len(self.pre):
                k = k[np.any(k != 0, axis=1)]
                if len(k):
                    cue = self.separator.code(k) if self.separator is not None else k
                    used = np.abs(self._delta_unit(cue)).sum(axis=0)
                    self.fade_depth = self.fade_depth + used
                    self.fade_recent = self.fade_recent + used
        idle = 1.0 / (1.0 + (self.fade_recent / self.FADE_FLOOR) ** 2)
        lost = self.fade * idle / (1.0 + (self.fade_depth / self.FADE_SCALE) ** self.FADE_ORDER)
        if self.fade_keep:  # coordinates used to the kept depth are kept for good
            lost = np.where(self.fade_depth >= self.fade_keep, 0.0, lost)
        if self._device is not None:  # a fading store rests on the host (``reside``)
            self.leave_device()
        shed = self.consolidated * lost[:, None]
        self.consolidated = self.consolidated - shed
        self.strength = self.strength - shed  # each stream keeps its residual
        self.fade_recent = self.fade_recent * (1.0 - 1.0 / self.FADE_IDLE)

    def reside(self, kernel: Any) -> None:
        """The strengths on the device of a brain's ``kernel``, unless a separator, a surprise
        gate or a fade keeps them on the host."""
        if (
            kernel is None or self.separator is not None or self.gate_noise is not None
            or self.fade
        ):
            self.leave_device()
            return
        held = self._device
        if held is not None:
            if any(name in self.__dict__ for name in _StoreOnDevice.OWNED):
                written = {
                    name: self.__dict__.pop(name) for name in _StoreOnDevice.OWNED
                    if name in self.__dict__
                }
                self.leave_device()
                self.__dict__.update(written)
            else:
                held.kernel = kernel
                return
        self._device = _StoreOnDevice(self, kernel)

    def held_values(self) -> dict[str, Any]:
        """``FastSynapses.held_values``, the shared matrix and the noise scales with them."""
        held: Any = self._device or self
        return {
            **super().held_values(), "consolidated": held.consolidated,
            "gate_noise": self.gate_noise,  # with maturity, replaced as well
        }

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Forget transient residuals; persistent synapses survive episode/batch changes."""
        all_rows = rows is None or len(self.strength) != batch
        super().reset(batch, rows)
        if all_rows:
            self.strength[:] = self.consolidated
        else:
            self.strength[rows] = self.consolidated

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps, the consolidated records among them (and the
        records' noise scales, with a surprise gate)."""
        kept = {} if self.gate_noise is None else {"gate_noise": self.gate_noise}
        if self.fade:
            assert self.fade_depth is not None and self.fade_recent is not None
            kept.update(fade_depth=self.fade_depth, fade_recent=self.fade_recent)
        return {**super().arrays(), "consolidated": self.consolidated, **kept}

    def clear(self) -> None:
        """Explicitly erase both persistent and transient associations."""
        self.consolidated.fill(0)
        if self.gate_noise is not None:
            self.gate_noise.fill(0)
        if self.fade_depth is not None and self.fade_recent is not None:
            self.fade_depth = np.zeros_like(self.fade_depth)
            self.fade_recent = np.zeros_like(self.fade_recent)
        self.reset(len(self.strength))

    def _baseline(self) -> np.ndarray:
        return self.consolidated

    @staticmethod
    def salience_vector(salience: np.ndarray | None, batch: int) -> np.ndarray:
        out = np.zeros(batch) if salience is None else np.asarray(salience, dtype=float)
        if out.shape != (batch,) or not np.isfinite(out).all() or (out < 0).any():
            raise ValueError("salience must be a finite nonnegative vector matching the batch")
        return out

    def observe(
        self,
        key: np.ndarray,
        value: np.ndarray,
        write: np.ndarray | None = None,
        *,
        salience: np.ndarray | None = None,
        value_mask: np.ndarray | None = None,
    ) -> None:
        """Consolidate real observations, with optional row salience and observed-value mask.

        Slow updates average over writing rows (one shared set of synapses); fast
        residuals remain per stream. ``salience=0`` is ordinary experience, not a
        disabled write. Reads never advance either timescale.
        """
        key = self._port(key, len(self.pre), "key")
        value = self._port(value, len(self.post), "value")
        batch = len(key)
        if len(value) != batch:
            raise ValueError("key and value batches must match")
        salience = self.salience_vector(salience, batch)
        gate = np.ones(batch, bool) if write is None else np.asarray(write)
        if gate.shape != (batch,) or gate.dtype != np.bool_:
            raise ValueError("write must be a boolean vector with one entry per stream")
        observed = np.ones(value.shape, bool) if value_mask is None else np.asarray(value_mask)
        if observed.shape != value.shape or observed.dtype != np.bool_:
            raise ValueError("value_mask must be a boolean array matching the values")
        gate = gate & np.any(key != 0, axis=1) & observed.any(axis=1)
        if self._device is not None:
            self._device.observe(key, value, gate, salience, observed)
            return
        same_batch = len(self.strength) == batch
        previous = (
            self.strength
            if same_batch
            else np.broadcast_to(self.consolidated, (batch, *self.consolidated.shape))
        )
        # W = C + F. Decay F, never C; neural-state resets also leave C intact.
        strength = self.consolidated + self.decay * (previous - self.consolidated)
        mass = self.mass * self.decay if same_batch else np.zeros(batch)
        rows = np.flatnonzero(gate)
        if not len(rows):
            self.strength, self.mass = strength, mass
            return
        cue = key[rows]
        if self.separator is not None:
            cue = self.separator.code(cue)
        cue = self._delta_unit(cue)
        mask = observed[rows]
        # Clip before multiplying to avoid overflow for large finite salience.
        rate = self.consolidation + np.minimum(1.0, self.consolidation * salience[rows])
        rate = np.minimum(1.0, rate)
        error = (value[rows] - cue @ self.consolidated) * mask
        change = np.einsum("b,bi,bj->ij", rate, cue, error) / len(rows)
        gated = self.gate_noise is not None and self.maturity is not None
        base = strength.copy() if gated else strength
        strength += change  # preserve each stream's residual around the shared matrix
        prediction = (cue[:, None, :] @ strength[rows])[:, 0, :]
        correction = (value[rows] - prediction) * mask
        fast = self.rate * cue[:, :, None] * correction[:, None, :]
        noise = None
        if not gated:
            consolidated = self.consolidated + change
            strength[rows] += fast
        else:  # the write above, multiplied per action by the surprise gate's gain
            gain, noise = self._gate(cue, value[rows], previous[rows], mask)
            # one gain per action, the mean over the rows that observe it, on the shared matrix
            # and on every writing stream's residual alike: each stream keeps its residual
            # around the shared matrix
            seen = mask.sum(axis=0)
            shared = np.where(seen > 0, (gain * mask).sum(axis=0) / np.maximum(seen, 1), 1.0)
            moved = np.broadcast_to(change, strength.shape).copy()
            moved[rows] += fast
            consolidated = self.consolidated + np.einsum(
                "b,bi,bj->ij", rate, cue, error * shared
            ) / len(rows)
            strength = base + shared * moved
        if (
            not np.isfinite(strength).all()
            or not np.isfinite(consolidated).all()
            or (noise is not None and not np.isfinite(noise).all())
        ):
            raise ValueError("synaptic update overflowed; scale the observed values")
        mass[rows] += self.rate
        self.consolidated, self.strength, self.mass = consolidated, strength, mass
        if noise is not None:
            self.gate_noise = noise
        self.writes += len(rows)
        if self.fade:  # a write uses the records its cue reads: the writing rows' unit cues
            assert self.fade_depth is not None
            self.fade_depth = self.fade_depth + np.abs(cue).sum(axis=0)

    def _gate(
        self, cue: np.ndarray, value: np.ndarray, previous: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each writing row's gain per action (the surprise gate engaged by maturity), and
        the records' noise scales moved toward the rows' squared errors: a new array, which
        the write installs with the records it accepts."""
        assert self.gate_noise is not None and self.maturity is not None
        noise = self.gate_noise.copy()
        engaged = 1.0 - self.maturity.factor()  # the gate's strength, 1 - m
        error = value - np.einsum("bi,bij->bj", cue, previous)
        square = cue * cue
        spread = np.sqrt(square @ noise)
        size = np.abs(error)
        with np.errstate(over="ignore"):  # an error far within the noise: a ratio past the range
            ratio = self.GATE_WIDTH * spread / np.where(size > 0, size, 1.0)
        surprise = np.where(size > 0, np.maximum(0.0, 1.0 - ratio), 0.0)
        gain = np.where(mask, 1.0 - engaged * (1.0 - surprise), 1.0)
        weight = square / np.maximum(square.max(axis=1, keepdims=True), 1e-30)
        with np.errstate(over="ignore", invalid="ignore"):  # an overflow is refused (observe)
            for row in range(len(cue)):  # the noise scale of each observed action's record
                seen = mask[row]
                noise[:, seen] += self.GATE_RATE * weight[row][:, None] * (
                    error[row, seen][None, :] ** 2 - noise[:, seen]
                )
        return gain, noise

    def to_dict(self) -> dict[str, float | int | str]:
        gate: dict[str, Any] = {} if self.gate_noise is None else {"surprise_gate": True}
        if self.fade:
            gate["fade"] = self.fade
        if self.fade_keep:
            gate["fade_keep"] = self.fade_keep
        return {
            **super().to_dict(),
            "kind": "consolidating",
            "consolidation": self.consolidation,
            "persistent_parameters": int(self.consolidated.size),
            **gate,
        }
