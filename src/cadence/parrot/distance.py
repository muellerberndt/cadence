"""The sense of distance, learned: how far a thing is, from what one moment's two images show.

A layer of distance neurons, each preferring one distance (log-spaced over ``span``), settles
with the cues of a figure, each a layer of tuned input neurons: the disparity units' answer
(``cadence.parrot.vision``) for a thing both eyes see, and the figure's size in the field
(degrees, log scale) and its elevation. Size and elevation have two layers each: one for a
thing one eye sees and one for a thing both eyes see, so that what one eye alone tells and what
two eyes tell are learned apart. For a thing one eye sees, the parallax units compare where that
eye saw it a few moments before, from where the eye was then (the head's own movement moved
it): each unit is tuned to a shift of the thing in the eye and to how far the eye moved across
the line of sight (the Wulst's two-view units, docs/parrot-brain.md).
Each wired pair of an input neuron and a distance neuron is a reciprocal pair of synapses that
starts at zero, so a newborn brain has no sense of distance at all. Lessons are the library's
free and nudged phases with a quadratic nudge (as ``cadence.space.map.Sense``): the target is a
bump on the distance neurons at the distance a lesson gives. No lesson holds a distance the world
told: the parrot's own teacher (``cadence.parrot.brain``) gives the distance at which two
sightings of one thing, before and after its head turned, agree.

The distance read is the activity-weighted mean of the preferred distances (on a log scale)
over the bump around the most active neuron; a layer with no neuron above ``floor`` has no
sense of distance for the figure. ``mask`` silences neurons inside the settle.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

FAMILIES = ("disparity", "parallax", "size_one", "elevation_one", "size_both", "elevation_both")
WIRED = ("disparity", "parallax", "size_one", "elevation_one")  # the hand-set wiring


def tuned(values: np.ndarray, preferred: np.ndarray, width: float) -> np.ndarray:
    """``(rows,)`` values as Gaussian tuned responses ``(rows, len(preferred))``, ``width`` in
    units of the preferred values' spacing."""
    v = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    spacing = float(np.mean(np.diff(preferred))) if len(preferred) > 1 else 1.0
    return np.asarray(np.exp(-0.5 * ((v - preferred[None, :]) / (width * spacing)) ** 2))


class Distance:
    """Distance neurons learned from a figure's cues.

    ``disparity`` is the count of disparity units (the vision's channels); ``sizes`` and
    ``heights`` the tuned inputs of the figure's size and elevation (each for one eye and for
    both, ``FAMILIES``); ``units`` the distance neurons over ``span`` metres (the distance on
    the table's plane). ``wiring`` gives the wired ``(input, distance neuron)`` pairs; by
    default every input neuron of the families in ``families`` is wired to every distance
    neuron. ``rate`` is the share of a lesson's error one lesson closes."""

    def __init__(self, disparity: int, *, sizes: int = 8, heights: int = 8, units: int = 24,
                 shifts: int = 12, baselines: int = 6,
                 span: tuple[float, float] = (0.12, 1.2), width: float = 1.5,
                 rate: float = 0.3, steps: int = 6, level: float = 0.5, floor: float = 0.02,
                 families: tuple[str, ...] = WIRED, wiring: np.ndarray | None = None,
                 backend: Any = "cpu"):
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model

        self.counts = {"disparity": int(disparity), "parallax": int(shifts) * int(baselines),
                       "size_one": int(sizes),
                       "elevation_one": int(heights), "size_both": int(sizes),
                       "elevation_both": int(heights)}
        starts = np.cumsum([0] + [self.counts[f] for f in FAMILIES])
        self.inputs = {f: np.arange(int(starts[i]), int(starts[i + 1]))
                       for i, f in enumerate(FAMILIES)}
        self.width_in = int(starts[-1])
        self.units = int(units)
        self.span = (float(span[0]), float(span[1]))
        self.preferred = np.exp(np.linspace(math.log(self.span[0]), math.log(self.span[1]),
                                            self.units))
        self.width, self.rate, self.steps = float(width), float(rate), int(steps)
        self.level, self.floor = float(level), float(floor)
        self.families = tuple(families)
        self.sizes = np.linspace(math.log(1.0), math.log(40.0), int(sizes))
        self.heights = np.linspace(-50.0, 50.0, int(heights))
        self.shifts = np.linspace(0.0, 12.0, int(shifts))  # degrees
        self.baselines = np.linspace(0.0, 0.05, int(baselines))  # metres
        self.output = self.width_in + np.arange(self.units)
        self.neurons = n = self.width_in + self.units
        if wiring is None:
            wired = np.concatenate([self.inputs[f] for f in self.families])
            pairs = np.stack(np.meshgrid(wired, self.output, indexing="ij"),
                             axis=-1).reshape(-1, 2)
        else:
            pairs = np.asarray(wiring, dtype=np.int64).reshape(-1, 2)
            if pairs.size and (pairs[:, 0].min() < 0 or pairs[:, 0].max() >= self.width_in
                               or pairs[:, 1].min() < self.width_in
                               or pairs[:, 1].max() >= n):
                raise ValueError("wiring pairs an input neuron with a distance neuron")
        self.pairs = pairs
        self.connectome = Connectome.from_synapses(
            n, label="distance", pre=np.concatenate([pairs[:, 0], pairs[:, 1]]),
            post=np.concatenate([pairs[:, 1], pairs[:, 0]]),
            populations={"cues": list(range(self.width_in)),
                         "distance": self.output.tolist()})
        self._model = learning_neuron_model(dt=1.0)
        self.graph = Graph(self.connectome, self._model,
                           efficacy=np.zeros(self.connectome.synapses), backend=backend)
        self._learner: Any = None
        self.lessons = 0
        self.mask = np.ones(n)
        self.work = {"settles": 0, "macs": 0}

    def parameters(self) -> int:
        """Learned parameters: one efficacy a reciprocal pair."""
        return int(len(self.pairs))

    def code(self, disparity: np.ndarray | None, size: float, elevation: float,
             parallax: tuple[float, float] | None = None) -> np.ndarray:
        """A figure's cues as the input neurons' drive: the disparity units' answer (None for a
        figure one eye sees), its size in degrees, its elevation in degrees and, for a figure
        one eye sees, its parallax (the shift in degrees, the eye's movement across the line of
        sight in metres) or None."""
        out = np.zeros(self.width_in)
        seen = "one"
        if disparity is None and parallax is not None:
            shift = tuned(np.array([parallax[0]]), self.shifts, 1.0)[0]
            moved = tuned(np.array([parallax[1]]), self.baselines, 1.0)[0]
            out[self.inputs["parallax"]] = np.outer(shift, moved).reshape(-1)
        if disparity is not None:
            out[self.inputs["disparity"]] = np.asarray(disparity, dtype=np.float64)
            seen = "both"
        out[self.inputs["size_" + seen]] = tuned(np.array([math.log(max(size, 0.5))]),
                                                 self.sizes, 1.0)[0]
        out[self.inputs["elevation_" + seen]] = tuned(np.array([elevation]), self.heights,
                                                      1.0)[0]
        return out

    def target(self, distances: np.ndarray) -> np.ndarray:
        """The bump a lesson nudges the distance neurons toward, ``(rows, units)``."""
        logs = np.log(np.clip(np.asarray(distances, dtype=np.float64), 1e-3, None))
        bump = tuned(logs, np.log(self.preferred), self.width)
        return np.asarray(self.level * bump / np.linalg.norm(bump, axis=1, keepdims=True))

    def _count(self, rows: int, phases: int) -> None:
        self.work["settles"] += rows * phases
        self.work["macs"] += rows * phases * self.steps * int(self.connectome.synapses)

    def settle(self, codes: np.ndarray) -> np.ndarray:
        """Every neuron's activity settled on ``(rows, inputs)`` codes, ``(rows, neurons)``,
        with the silenced neurons (``mask``) held at rest."""
        given = np.asarray(codes, dtype=np.float64).reshape(-1, self.width_in)
        drive = np.zeros((len(given), self.neurons))
        drive[:, : self.width_in] = given
        state = self.graph.settle_batch(drive * self.mask, steps=self.steps, tolerance=None,
                                        mask=self.mask)
        self._count(len(given), 1)
        return np.asarray(state.activation, dtype=np.float64)

    def read(self, activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The distances ``(rows,)`` (NaN where no distance neuron is above ``floor``) and the
        summed activity of the bump each was read from, from ``(rows, neurons)`` activity."""
        a = np.maximum(np.asarray(activity)[:, self.output], 0.0)
        rows = len(a)
        out = np.full(rows, np.nan)
        strength = np.zeros(rows)
        logs = np.log(self.preferred)
        reach = max(1, int(round(2 * self.width)))
        for r in range(rows):
            peak = int(a[r].argmax())
            if a[r, peak] <= self.floor:
                continue
            lo, hi = max(0, peak - reach), min(self.units, peak + reach + 1)
            w = a[r, lo:hi]
            out[r] = math.exp(float((w * logs[lo:hi]).sum() / w.sum()))
            strength[r] = float(w.sum())
        return out, strength

    def teach(self, codes: np.ndarray, distances: np.ndarray) -> float:
        """Lessons: ``(rows, inputs)`` codes and the ``(rows,)`` distances the teacher gives.
        Gives the mean error of the free answer, as a share of the target's sum."""
        from ..learning import Learner, LearnerConfig

        code = np.asarray(codes, dtype=np.float64).reshape(-1, self.width_in)
        rows = len(code)
        drive = np.zeros((rows, self.neurons))
        drive[:, : self.width_in] = code
        target = np.zeros_like(drive)
        target[:, self.output] = self.target(np.asarray(distances).reshape(rows))
        if self._learner is None:
            self._learner = Learner(
                self.graph, list(self.output),
                LearnerConfig(nudge="quadratic", beta=0.1, eta=1.0, eta_bias=0.0,
                              free_steps=self.steps, nudged_steps=self.steps, tolerance=None),
                reciprocal=True, plastic_neurons=np.zeros(self.neurons, dtype=bool))
        learner = self._learner
        lit = np.tanh(code / 2.0)
        shared = float((lit.sum(axis=0) ** 2).sum()) / rows
        learner.config = dataclasses.replace(
            learner.config, eta=self.rate / (0.25 * max(shared, 1e-12)))
        free = learner.free(drive)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        learner.update(free, plus, minus)
        self.graph = learner.brain
        self.lessons += rows
        self._count(rows, 3)
        answer = np.asarray(free.activation)[:, self.output]
        wanted = target[:, self.output]
        return float(np.abs(answer - wanted).sum() / np.abs(wanted).sum())

    def arrays(self) -> dict[str, np.ndarray]:
        return {"efficacy": np.asarray(self.graph.efficacy, dtype=np.float64).copy(),
                "pairs": self.pairs.copy(), "lessons": np.array(self.lessons)}

    def take(self, arrays: dict[str, Any]) -> None:
        from ..brain import Brain as Graph

        if "pairs" in arrays and not np.array_equal(np.asarray(arrays["pairs"]), self.pairs):
            raise ValueError("the arrays were learned on another wiring")
        self.graph = Graph(self.connectome, self._model,
                           efficacy=np.asarray(arrays["efficacy"], dtype=np.float64).copy(),
                           backend="cpu")
        self._learner = None
        self.lessons = int(arrays["lessons"])


__all__ = ["FAMILIES", "WIRED", "Distance", "tuned"]
