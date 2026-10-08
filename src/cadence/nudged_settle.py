"""An opt-in numerical strategy for the qualified nudged phases.

Qualified teaching settles each nudged phase with ``NeuralGraph.equilibrate``
and its bounded ``damping`` schedule: an attempt at the model's ``dt``, then at
successively halved steps. When the full step cannot settle a graph (for
example a period-two orbit under strong lateral inhibition), its whole share of
the budget is spent before the first halved attempt starts. ``NudgedSettle``
starts the nudged phases further down that same schedule with
``first_halving``; the equations, tolerance, smallest step, qualification and
contrast rule are unchanged.

A different integration path can, on a graph with several stationary points,
end at a different one. The optional probe solves each nudged phase a second
time from another start, a one-pass copy of the nudge down the seams from the
outputs, and accepts the result only if both solves qualify and agree. Otherwise
the phase is solved again exactly as the default learner would solve it. This
guard detects disagreement between two starts; it does not prove uniqueness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .brain import BrainState, Equilibrium, Nudge

if TYPE_CHECKING:
    from .learning import Learner

__all__ = ["NudgedSettle"]


@dataclass(frozen=True)
class NudgedSettle:
    """Where in the damping schedule qualified nudged phases start, and an agreement probe.

    ``first_halving`` must not exceed the learner's ``damping``. ``probe`` is the
    largest accepted activation difference between the two solves, in multiples
    of the phase tolerance; ``None`` disables the probe and its fallback.
    """

    first_halving: int = 2
    probe: float | None = 10.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.first_halving, bool)
            or not isinstance(self.first_halving, (int, np.integer))
            or self.first_halving < 0
        ):
            raise ValueError("first_halving must be a nonnegative integer")
        if self.probe is not None and (not np.isfinite(self.probe) or self.probe < 0):
            raise ValueError("probe must be finite and nonnegative, or None")

    def solve(
        self, learner: Learner, drive: np.ndarray, free: BrainState, nudge: Nudge
    ) -> tuple[Equilibrium, dict[str, float]]:
        """One nudged phase: the shortened schedule, the probe, and the default fallback."""
        cfg = learner.config
        tolerance = cfg.tolerance
        if tolerance is None:
            raise ValueError("qualified learning requires a finite residual tolerance")
        if self.first_halving > cfg.damping:
            raise ValueError("NudgedSettle.first_halving cannot exceed LearnerConfig.damping")

        def schedule(start: BrainState, first: int) -> Equilibrium:
            return learner.brain.equilibrate(
                drive, state=start, budget=cfg.nudged_steps, tolerance=tolerance,
                nudge=nudge, damping=cfg.damping, first_halving=first,
            )

        work = {"probe_sweeps": 0.0, "discarded_sweeps": 0.0, "fallback": 0.0}
        phase = schedule(free, self.first_halving)
        if self.probe is None or not np.all(phase.qualified):
            if not np.all(phase.qualified):
                work["discarded_sweeps"] += phase.steps
                work["fallback"] = 1.0
                phase = learner._qualified_phase(drive, free, cfg.nudged_steps, nudge)
            return phase, work
        check = schedule(seam_copy(learner, free, nudge), self.first_halving)
        work["probe_sweeps"] += check.steps
        gap = float(np.max(np.abs(
            np.asarray(check.state.activation) - np.asarray(phase.state.activation)
        )))
        work["probe_gap"] = gap
        if np.all(check.qualified) and gap <= self.probe * tolerance:
            return phase, work
        work["discarded_sweeps"] += phase.steps
        work["fallback"] = 1.0
        return learner._qualified_phase(drive, free, cfg.nudged_steps, nudge), work


def seam_copy(learner: Learner, free: BrainState, nudge: Nudge) -> BrainState:
    """The free state plus the nudge copied once down the seams from the outputs.

    Shell 0 is the output group; shell ``k`` holds neurons first reached from shell
    ``k - 1`` by a synapse. Outputs add the nudge drive at the free state; deeper
    neurons add the synaptic input received from the shell above's change.
    Neurons the nudge never reaches keep their free state.
    """
    connectome = learner.brain.connectome
    activation = learner.brain.neuron_model.activation
    weights = learner.brain.weights
    free_v = np.atleast_2d(np.asarray(free.v, dtype=float))
    free_s = np.atleast_2d(np.asarray(free.activation, dtype=float))
    v, s = free_v.copy(), free_s.copy()
    reached = np.zeros(connectome.n, dtype=bool)
    shell = np.zeros(connectome.n, dtype=bool)
    shell[learner.output_index] = True
    reached |= shell
    v[:, shell] += nudge.drive(free_s)[:, shell]
    s[:, shell] = activation(v[:, shell])
    while True:
        edges = np.flatnonzero(shell[connectome.pre] & ~reached[connectome.post])
        if edges.size == 0:
            break
        received = np.zeros_like(v)
        np.add.at(
            received.T, connectome.post[edges],
            ((s - free_s)[:, connectome.pre[edges]] * weights[edges]).T,
        )
        shell = np.zeros(connectome.n, dtype=bool)
        shell[connectome.post[edges]] = True
        reached |= shell
        v[:, shell] += received[:, shell]
        s[:, shell] = activation(v[:, shell])
    if not (np.isfinite(v).all() and np.isfinite(s).all()):
        return free
    adaptation = np.atleast_2d(np.asarray(free.adaptation, dtype=float)) + (s - free_s)
    return BrainState(v=v, activation=s, adaptation=adaptation, steps=0)
