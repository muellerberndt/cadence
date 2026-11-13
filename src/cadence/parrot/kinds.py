"""What a thing is, learned: categories formed from how things look.

A figure's look is a population code: its hue around the hue circle (weighted by how saturated
it is), its saturation and its lightness. A layer of category neurons settles on it, each joined
to every look neuron by a reciprocal pair of synapses that starts at zero. The most active
category is the thing's kind. A look whose best category settles below ``vigilance`` takes the
next free category: a kind enters when a new sort of thing is first seen. A category's ``n``-th
lesson closes ``1 / n`` of its error and no less than ``rate``, so its synapses hold the mean of
what it was seen as. Nothing tells the brain what a thing is.
"""

from __future__ import annotations

import colorsys
import dataclasses
import math
from typing import Any

import numpy as np

HUES, SATURATIONS, LIGHTS = 12, 4, 4


def look(rgb: np.ndarray) -> np.ndarray:
    """A colour's look code (``HUES + SATURATIONS + LIGHTS`` units), at unit length: its hue
    around the circle, as strong as the colour is saturated; a colour with little saturation
    is told by its lightness. Shading changes a thing's lightness more than its hue, so a
    saturated thing's lightness weighs little."""
    r, g, b = (float(v) for v in np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0))
    hue, light, saturation = colorsys.rgb_to_hls(r, g, b)
    angles = 2.0 * math.pi * np.arange(HUES) / HUES
    hue_code = saturation * np.exp(4.0 * (np.cos(2.0 * math.pi * hue - angles) - 1.0))
    sat_code = 0.3 * np.exp(-0.5 * ((saturation - np.linspace(0, 1, SATURATIONS)) / 0.2) ** 2)
    light_code = (1.0 - saturation) * np.exp(
        -0.5 * ((light - np.linspace(0, 1, LIGHTS)) / 0.2) ** 2)
    code = np.concatenate([hue_code, sat_code, light_code])
    return np.asarray(code / max(float(np.linalg.norm(code)), 1e-12))


class Kinds:
    """``capacity`` category neurons learned over the look code."""

    features = HUES + SATURATIONS + LIGHTS

    def __init__(self, capacity: int = 12, *, vigilance: float = 0.2, rate: float = 0.05,
                 steps: int = 6, level: float = 0.5, drive: float = 2.0,
                 threshold: float = 0.15, backend: Any = "cpu"):
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model

        self.capacity = int(capacity)
        self.vigilance, self.rate, self.steps = float(vigilance), float(rate), int(steps)
        self.level, self.drive, self.threshold = float(level), float(drive), float(threshold)
        self.category = np.arange(self.capacity)
        self.feature = self.capacity + np.arange(self.features)
        self.neurons = n = self.capacity + self.features
        pairs = np.stack(np.meshgrid(self.category, self.feature, indexing="ij"),
                         axis=-1).reshape(-1, 2)
        self.connectome = Connectome.from_synapses(
            n, label="kinds", pre=np.concatenate([pairs[:, 0], pairs[:, 1]]),
            post=np.concatenate([pairs[:, 1], pairs[:, 0]]),
            populations={"kind": self.category.tolist(), "look": self.feature.tolist()})
        self._model = learning_neuron_model(dt=1.0)
        self._bias = np.zeros(n)
        self._bias[self.category] = -self.threshold
        self.graph = Graph(self.connectome, self._model,
                           efficacy=np.zeros(self.connectome.synapses), bias=self._bias,
                           backend=backend)
        pre, post = np.asarray(self.connectome.pre), np.asarray(self.connectome.post)
        owner = np.where(np.isin(pre, self.category), pre, post)
        order = np.argsort(owner, kind="stable")
        self._touch = np.split(order, np.searchsorted(owner[order], self.category[1:]))
        self._learner: Any = None
        self.counts = np.zeros(self.capacity, dtype=np.int64)
        self.used = 0
        self.mask = np.ones(n)
        self.work = {"settles": 0, "macs": 0}

    def parameters(self) -> int:
        return int(self.connectome.synapses // 2)

    def _count(self, rows: int) -> None:
        self.work["settles"] += rows
        self.work["macs"] += rows * self.steps * int(self.connectome.synapses)

    def settle(self, codes: np.ndarray) -> np.ndarray:
        """Every neuron settled on ``(rows, features)`` look codes, ``(rows, neurons)``."""
        given = np.asarray(codes, dtype=np.float64).reshape(-1, self.features)
        drive = np.zeros((len(given), self.neurons))
        drive[:, self.feature] = self.drive * given
        state = self.graph.settle_batch(drive * self.mask, steps=self.steps, tolerance=None,
                                        mask=self.mask)
        self._count(len(given))
        return np.asarray(state.activation, dtype=np.float64)

    def see(self, codes: np.ndarray, *, learn: bool = True) -> tuple[list[int | None], np.ndarray]:
        """The kind of each look code (None where no kind has entered and none may) and every
        neuron's settled activity; with ``learn`` each kind takes its code as a lesson."""
        given = np.asarray(codes, dtype=np.float64).reshape(-1, self.features)
        activity = self.settle(given) if len(given) else np.zeros((0, self.neurons))
        out: list[int | None] = []
        for row, code in enumerate(given):
            best: int | None = None
            if self.used:
                active = activity[row, self.category[: self.used]]
                best = int(np.argmax(active))
                if active[best] < self.vigilance and learn and self.used < self.capacity:
                    best = None
            if best is None and learn and self.used < self.capacity:
                best, self.used = self.used, self.used + 1
            if best is not None and learn:
                self.teach(best, code)
            out.append(best)
        return out, activity

    def teach(self, category: int, code: np.ndarray) -> None:
        from ..learning import Learner, LearnerConfig

        if self._learner is None:
            self._learner = Learner(
                self.graph, list(self.feature),
                LearnerConfig(nudge="quadratic", beta=0.1, eta=1.0, eta_bias=0.0,
                              free_steps=self.steps, nudged_steps=self.steps, tolerance=None),
                reciprocal=True, plastic_neurons=np.zeros(self.neurons, dtype=bool))
        learner = self._learner
        k = int(category)
        drive = np.zeros((1, self.neurons))
        drive[0, self.category[k]] = self.drive
        lit = np.tanh(drive[:, self.category] / 2.0)
        shared = float((lit.sum(axis=0) ** 2).sum())
        share = max(1.0 / (self.counts[k] + 1), self.rate)
        steps = np.zeros(self.connectome.synapses)
        steps[self._touch[k]] = 1.0
        learner.synapse_rate = steps
        learner.config = dataclasses.replace(
            learner.config, eta=share / (0.25 * max(shared, 1e-12)))
        target = np.zeros_like(drive)
        target[0, self.feature] = self.level * np.asarray(code, dtype=np.float64)
        free = learner.free(drive)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        learner.update(free, plus, minus)
        self.graph = learner.brain
        self.counts[k] += 1
        self._count(3)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"efficacy": np.asarray(self.graph.efficacy, dtype=np.float64).copy(),
                "counts": self.counts.copy(), "used": np.array(self.used)}

    def take(self, arrays: dict[str, Any]) -> None:
        from ..brain import Brain as Graph

        self.graph = Graph(self.connectome, self._model,
                           efficacy=np.asarray(arrays["efficacy"], dtype=np.float64).copy(),
                           bias=self._bias, backend="cpu")
        self._learner = None
        self.counts = np.asarray(arrays["counts"], dtype=np.int64).copy()
        self.used = int(arrays["used"])


__all__ = ["HUES", "LIGHTS", "SATURATIONS", "Kinds", "look"]
