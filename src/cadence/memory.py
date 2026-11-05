"""Persistent synaptic weights plus a fading, per-stream residual.

No experience buffer is stored. Repetition and salience change the persistent
key-to-value synapses using the actual observed value's local prediction error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .stream import FastSynapses


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
    brain's ``Maturity``; ``Brain.compose(maturity=True)``) a write moves a record only by the
    part of its error beyond ``GATE_WIDTH`` noise scales of the record, engaged by the brain's
    maturity: the write's change of the record, the shared matrix and the stream's residual
    alike, is multiplied by ``1 - (1 - m)(1 - g)``, ``g = max(0, 1 - GATE_WIDTH sqrt(R) /
    |e|)``, ``e`` the write's error on the action against the stream's read before the write,
    ``R`` the record's noise along the cue. Each write moves the noise scale toward the squared
    error at ``GATE_RATE``. A young brain (``m`` 1) writes as without the gate.
    """

    decay: float = 0.9
    rule: str = "delta"
    consolidation: float = 0.05
    consolidated: np.ndarray = field(init=False)
    gate_noise: np.ndarray | None = field(default=None, repr=False)
    maturity: Any = field(default=None, repr=False, compare=False)

    GATE_WIDTH = 2.0
    GATE_RATE = 0.05

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
        return {**super().arrays(), "consolidated": self.consolidated, **kept}

    def clear(self) -> None:
        """Explicitly erase both persistent and transient associations."""
        self.consolidated.fill(0)
        if self.gate_noise is not None:
            self.gate_noise.fill(0)
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
        noise = None
        if self.gate_noise is not None and self.maturity is not None:
            gain, noise = self._gate(cue, value[rows], previous[rows], mask)
            base = strength
            change = np.einsum("b,bi,bj->ij", rate, cue, error) / len(rows)
            strength = strength + change
            prediction = (cue[:, None, :] @ strength[rows])[:, 0, :]
            correction = (value[rows] - prediction) * mask
            moved = np.broadcast_to(change, strength.shape).copy()
            moved[rows] += self.rate * cue[:, :, None] * correction[:, None, :]
            shared = gain.mean(axis=0)  # the rows' gain on the shared matrix, per action
            weight = np.broadcast_to(shared, (len(strength), len(shared))).copy()
            weight[rows] = gain
            gated = np.einsum("b,bi,bj->ij", rate, cue, error * gain) / len(rows)
            consolidated = self.consolidated + gated
            strength = base + weight[:, None, :] * moved
        else:
            change = np.einsum("b,bi,bj->ij", rate, cue, error) / len(rows)
            consolidated = self.consolidated + change
            strength += change  # preserve each stream's residual around the shared matrix
            prediction = (cue[:, None, :] @ strength[rows])[:, 0, :]
            correction = (value[rows] - prediction) * mask
            strength[rows] += self.rate * cue[:, :, None] * correction[:, None, :]
        if not np.isfinite(strength).all() or not np.isfinite(consolidated).all():
            raise ValueError("synaptic update overflowed; scale the observed values")
        mass[rows] += self.rate
        self.consolidated, self.strength, self.mass = consolidated, strength, mass
        if noise is not None:
            self.gate_noise = noise
        self.writes += len(rows)

    def _gate(
        self, cue: np.ndarray, value: np.ndarray, previous: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each writing row's gain per action (the surprise gate engaged by maturity), and
        the records' noise scales moved toward the rows' squared errors: a new array, which
        the write installs with the records it accepts."""
        assert self.gate_noise is not None
        noise = self.gate_noise.copy()
        strength = 1.0 - self.maturity.factor()
        error = value - np.einsum("bi,bij->bj", cue, previous)
        square = cue * cue
        spread = np.sqrt(square @ noise)
        size = np.abs(error)
        safe = np.where(size > 0, size, 1.0)
        surprise = np.where(size > 0, np.maximum(0.0, 1.0 - self.GATE_WIDTH * spread / safe), 0.0)
        gain = np.where(mask, 1.0 - strength * (1.0 - surprise), 1.0)
        weight = square / np.maximum(square.max(axis=1, keepdims=True), 1e-30)
        for row in range(len(cue)):  # the noise scale of each observed action's record
            seen = mask[row]
            noise[:, seen] += self.GATE_RATE * weight[row][:, None] * (
                error[row, seen][None, :] ** 2 - noise[:, seen]
            )
        return gain, noise

    def to_dict(self) -> dict[str, float | int | str]:
        gate = {} if self.gate_noise is None else {"surprise_gate": True}
        return {
            **super().to_dict(),
            "kind": "consolidating",
            "consolidation": self.consolidation,
            "persistent_parameters": int(self.consolidated.size),
            **gate,
        }
