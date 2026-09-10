"""A life for the seams: fast and slow strengths, tags, sleep, pruning, sprouting.

A seam's strength is two numbers. The fast one moves with every update the
learner makes and fades unless it is consolidated; the slow one is what sleep
keeps of it, and it is the net's long-term memory. A tag accumulates on a seam
whenever it moves, so sleep can tell what was repeated and reinforced from
what merely happened. Sleep consolidates tagged fast strength into slow
strength, downscales the rest, prunes seams whose total fell below the floor,
and sprouts seams, among the wiring's silent overlaps, between owners that
kept firing together without one. Each owner holds a fixed total of incoming
strength per block, so the strengths a learner writes are relative.

The wiring is immutable and holds every overlap that could ever carry a seam;
``alive`` marks the ones that do. A pruned overlap is silent (zero strength,
frozen) until it sprouts again. The learner's ``trainable_overlaps`` mask is
kept equal to ``alive``, so nothing it does touches a silent overlap, and the
conformance reference sees exactly the live seams.

Every operation here reads one seam's two endpoints, its own state, and the
owner's incoming total; the co-activation that decides sprouting is the
product of the two endpoints' activations, accumulated where a seam is absent.
The composer of cadence-examples rung 10 is where this was built and where
its traps were paid for; the tests carry them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .learning import Learner

__all__ = ["Seams", "SleepConfig"]


@dataclass(frozen=True, slots=True)
class SleepConfig:
    consolidate: float = 0.5  # fraction of a fully tagged fast strength that becomes slow in one sleep
    downscale: float = 0.5  # what remains of the fast strength after sleep
    tag_saturation: float = 1.0  # accumulated |step| at which a seam counts as fully tagged
    tag_decay: float = 0.0  # >0: forgetting factor of the tag between sleeps (per observe)
    prune_below: float = 0.01  # a seam whose total magnitude falls below this is pruned
    sprout_above: float = 0.5  # co-activation a silent overlap must accumulate to sprout
    sprout_strength: float = 0.02  # magnitude a new seam is born with (sign from co-activation)
    budget: int = 0  # >0: most live incoming seams per owner; sprouting respects it
    strength: float = 0.0  # >0: conserved total of incoming |strength| per owner
    fast_decay: float = 0.0  # >0: forgetting factor of the fast strength per observe

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass
class Seams:
    """The life of a learner's seams. Call ``observe`` after every learner update and
    ``sleep`` at the end of a day."""

    learner: Learner
    config: SleepConfig = field(default_factory=SleepConfig)
    slow: np.ndarray = field(init=False)
    fast: np.ndarray = field(init=False)
    tag: np.ndarray = field(init=False)
    coact: np.ndarray = field(init=False)
    alive: np.ndarray = field(init=False)
    born: np.ndarray = field(init=False)
    day: int = 0
    _last: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        w = self.learner.engine.wiring
        scale = self.learner.engine.edge_scale
        self.alive = np.asarray(self.learner.trainable_overlaps, dtype=bool).copy()
        self.slow = np.zeros(w.edges)
        self.fast = np.where(self.alive, scale, 0.0)
        self.tag = np.zeros(w.edges)
        self.coact = np.zeros(w.edges)
        self.born = np.where(self.alive, 0, -1)
        self._last = scale.copy()
        self._enforce()

    # -- the day

    def observe(self, activation: np.ndarray | None = None) -> None:
        """Account for what the learner just did: fast strength moved, tags accumulate;
        with a batch of activations, silent overlaps accumulate their endpoints' co-activation."""
        cfg = self.config
        w = self.learner.engine.wiring
        scale = self.learner.engine.edge_scale
        step = np.where(self.alive, scale - self._last, 0.0)
        self.fast = np.where(self.alive, scale - self.slow, 0.0)
        self.tag += np.abs(step)
        if cfg.tag_decay > 0:
            self.tag *= 1.0 - cfg.tag_decay
        if cfg.fast_decay > 0:
            self.fast *= 1.0 - cfg.fast_decay
        if activation is not None:
            s = np.atleast_2d(activation)
            product = (s[:, w.pre] * s[:, w.post]).mean(axis=0)
            self.coact += np.where(self.alive, 0.0, product)
        self._enforce()

    # -- the night

    def sleep(self, activation_sign: np.ndarray | None = None) -> dict[str, Any]:
        """Consolidate, downscale, prune, sprout. Returns what changed."""
        cfg = self.config
        w = self.learner.engine.wiring
        self.day += 1
        capture = np.clip(self.tag / cfg.tag_saturation, 0.0, 1.0)
        moved = np.where(self.alive, cfg.consolidate * capture * self.fast, 0.0)  # transferred, not added
        self.slow += moved
        self.fast -= moved
        self.fast *= cfg.downscale  # what was not consolidated fades
        self.tag[:] = 0.0
        total = self.slow + self.fast
        weak = self.alive & (np.abs(total) < cfg.prune_below)
        self.alive[weak] = False
        self.slow[weak] = 0.0
        self.fast[weak] = 0.0
        self.born[weak] = -1
        sprouted = 0
        candidates = np.where(self.alive, -np.inf, self.coact)
        order = np.argsort(-candidates, kind="stable")
        in_degree = np.bincount(w.post[self.alive], minlength=w.n)
        for e in order:
            if not np.isfinite(candidates[e]) or candidates[e] < cfg.sprout_above:
                break
            if cfg.budget > 0 and in_degree[w.post[e]] >= cfg.budget:
                continue
            self.alive[e] = True
            sign = 1.0 if activation_sign is None else float(np.sign(activation_sign[e]) or 1.0)
            self.fast[e] = cfg.sprout_strength * sign
            self.slow[e] = 0.0
            self.born[e] = self.day
            in_degree[w.post[e]] += 1
            sprouted += 1
        self.coact[:] = 0.0
        self._enforce()
        return {"day": self.day, "pruned": int(weak.sum()), "sprouted": sprouted, "alive": int(self.alive.sum()), "slow_fraction": self.slow_fraction(), "digest": self.digest()}

    # -- invariants

    def _enforce(self) -> None:
        """Conserve each owner's incoming total, write the strengths back, keep the learner's mask equal to ``alive``."""
        cfg = self.config
        w = self.learner.engine.wiring
        if cfg.strength > 0:
            magnitude = np.where(self.alive, np.abs(self.slow + self.fast), 0.0)
            total = np.bincount(w.post, weights=magnitude, minlength=w.n)
            factor = np.where(total > cfg.strength, cfg.strength / np.maximum(total, 1e-12), 1.0)[w.post]
            self.slow *= factor
            self.fast *= factor
        scale = np.where(self.alive, self.slow + self.fast, 0.0)
        self.learner.engine = self.learner.engine.with_parameters(edge_scale=scale, bias=self.learner.engine.bias)
        self.learner.trainable_overlaps = self.alive.copy()
        self._last = scale.copy()

    def slow_fraction(self) -> float:
        total = np.abs(self.slow[self.alive]).sum() + np.abs(self.fast[self.alive]).sum()
        return float(np.abs(self.slow[self.alive]).sum() / total) if total > 0 else 0.0

    def digest(self) -> str:
        """SHA-256 of which overlaps are alive: the wiring as a state variable."""
        import hashlib

        return hashlib.sha256(np.packbits(self.alive).tobytes()).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {"alive": int(self.alive.sum()), "overlaps": int(len(self.alive)), "slow_fraction": self.slow_fraction(), "day": self.day, "digest": self.digest()}
