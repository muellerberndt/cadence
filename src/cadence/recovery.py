"""An opt-in starting state for the qualified nudged phases.

A nudged phase normally starts from the free state and settles under the
nudge. ``RecoveryStart`` renders a different starting state: it estimates how
far each neuron's potential will move from the free state, using only what the
neuron receives across one seam, then lets the ordinary local settle continue
from there. The settle, its equations, its tolerance and the contrast rule are
unchanged; only the point where settling starts moves.

Seams follow hop distance from the nudged output neurons. Shell 0 holds the
outputs; shell ``k`` holds the neurons first reached from shell ``k - 1``
through a synapse. Each neuron keeps one gain. Its starting displacement is
that gain times its feature: the nudge drive at the free state for an output,
or for a deeper neuron the synaptic input it receives from shell ``k - 1``'s
rendered change. With every gain at one (``fit=False``) this is a plain
one-pass copy of the nudge down the seams.

Gains are refitted between lessons from accepted, settled phases only. The
target is the neuron's own settled potential change; its feature is computed
from the settled change of the shell above, never from this object's rendered
start. Each gain is a ridge fit pulled toward one, so a neuron with little
evidence keeps the plain copy.

The statistics are a numerical cache. They are not part of a learner
checkpoint: a reloaded learner restarts them at one. The start can change the
qualified state only within the phase tolerance, and on a nonlinear graph a
different start could in principle reach a different stationary point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .brain import BrainState, Nudge

if TYPE_CHECKING:
    from .learning import Learner

__all__ = ["RecoveryStart"]


@dataclass
class RecoveryStart:
    """Render a nudged phase's starting state from the free state, one seam at a time.

    ``decay`` forgets older lessons' statistics; ``ridge`` pulls every gain toward one
    in proportion to its own evidence. ``fit=False`` keeps every gain at one: the
    unfitted copy, a control for the fitted gains.
    """

    decay: float = 0.9
    ridge: float = 0.3
    fit: bool = True
    observed: int = field(default=0, init=False)
    _shell: np.ndarray | None = field(default=None, init=False, repr=False)
    _seams: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _n: int = field(default=-1, init=False, repr=False)
    _moments: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.decay) or not 0 <= self.decay < 1:
            raise ValueError("decay is a forgetting factor in [0, 1)")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if not isinstance(self.fit, bool):
            raise ValueError("fit must be boolean")

    # -- structure

    def _bind(self, learner: Learner) -> None:
        connectome = learner.brain.connectome
        if self._shell is not None:
            if connectome.n != self._n:
                raise ValueError("a RecoveryStart serves one connectome; make a new one")
            return
        n = connectome.n
        shell = np.full(n, -1, dtype=np.int64)
        shell[learner.output_index] = 0
        seams: list[np.ndarray] = [np.zeros(0, dtype=np.int64)]
        depth = 0
        while True:
            edges = np.flatnonzero(shell[connectome.pre] == depth)
            reached = np.unique(connectome.post[edges])
            reached = reached[shell[reached] < 0]
            if reached.size == 0:
                break
            depth += 1
            shell[reached] = depth
            seams.append(edges[shell[connectome.post[edges]] == depth])
        self._shell, self._seams, self._n = shell, seams, n
        self._moments = {
            name: np.zeros(n) for name in ("ff", "fy", "yy", "unexplained", "copy")
        }

    @property
    def depth(self) -> int:
        """The deepest shell reached from the outputs; zero before first use."""
        return max(0, len(self._seams) - 1)

    def shells(self) -> np.ndarray:
        """Hop distance from the outputs per neuron, -1 where a nudge never arrives."""
        if self._shell is None:
            raise RuntimeError("RecoveryStart has not served a learner yet")
        return self._shell.copy()

    def gains(self) -> np.ndarray:
        """One gain per neuron; one where no evidence has arrived or ``fit`` is off."""
        if self._shell is None:
            raise RuntimeError("RecoveryStart has not served a learner yet")
        ff, fy = self._moments["ff"], self._moments["fy"]
        gain = np.ones(self._n)
        if self.fit:
            seen = ff > 0
            gain[seen] = (fy[seen] + self.ridge * ff[seen]) / ((1.0 + self.ridge) * ff[seen])
        return gain

    def _seam_input(self, learner: Learner, depth: int, change: np.ndarray) -> np.ndarray:
        """Synaptic input that shell ``depth`` receives from shell ``depth - 1``'s change."""
        connectome = learner.brain.connectome
        edges = self._seams[depth]
        received = np.zeros_like(change)
        # Accumulate along the neuron axis of the transposed view: rows share the seam.
        np.add.at(
            received.T,
            connectome.post[edges],
            (change[:, connectome.pre[edges]] * learner.brain.weights[edges]).T,
        )
        return received

    # -- the start and the fit

    def start(self, learner: Learner, free: BrainState, nudge: Nudge) -> BrainState:
        """The state a nudged phase starts from: the free state plus rendered changes."""
        self._bind(learner)
        assert self._shell is not None
        gain = self.gains()
        activation = learner.brain.neuron_model.activation
        free_v = np.atleast_2d(np.asarray(free.v, dtype=float))
        free_s = np.atleast_2d(np.asarray(free.activation, dtype=float))
        v, s = free_v.copy(), free_s.copy()
        outputs = self._shell == 0
        v[:, outputs] += gain[outputs] * nudge.drive(free_s)[:, outputs]
        s[:, outputs] = activation(v[:, outputs])
        for depth in range(1, self.depth + 1):
            members = self._shell == depth
            received = self._seam_input(learner, depth, s - free_s)
            v[:, members] += gain[members] * received[:, members]
            s[:, members] = activation(v[:, members])
        if not (np.isfinite(v).all() and np.isfinite(s).all()):
            return free
        adaptation = np.atleast_2d(np.asarray(free.adaptation, dtype=float)) + (s - free_s)
        return BrainState(v=v, activation=s, adaptation=adaptation, steps=0)

    def observe(
        self, learner: Learner, free: BrainState, settled: BrainState, nudge: Nudge
    ) -> None:
        """Fold one accepted, settled phase into the gains' statistics."""
        self._bind(learner)
        assert self._shell is not None
        free_v = np.atleast_2d(np.asarray(free.v, dtype=float))
        free_s = np.atleast_2d(np.asarray(free.activation, dtype=float))
        change = np.atleast_2d(np.asarray(settled.activation, dtype=float)) - free_s
        target = np.atleast_2d(np.asarray(settled.v, dtype=float)) - free_v
        feature = np.zeros_like(target)
        outputs = self._shell == 0
        feature[:, outputs] = nudge.drive(free_s)[:, outputs]
        for depth in range(1, self.depth + 1):
            members = self._shell == depth
            feature[:, members] = self._seam_input(learner, depth, change)[:, members]
        if not (np.isfinite(feature).all() and np.isfinite(target).all()):
            return
        gain = self.gains()  # before this lesson: the fidelity is a prediction, not a fit
        m, keep = self._moments, self.decay
        m["ff"] = keep * m["ff"] + (feature * feature).sum(axis=0)
        m["fy"] = keep * m["fy"] + (feature * target).sum(axis=0)
        m["yy"] = keep * m["yy"] + (target * target).sum(axis=0)
        m["unexplained"] = keep * m["unexplained"] + ((target - gain * feature) ** 2).sum(axis=0)
        m["copy"] = keep * m["copy"] + ((target - feature) ** 2).sum(axis=0)
        self.observed += 1

    def fidelity(self) -> list[dict[str, float]]:
        """Per shell: the fraction of the settled change the start left unexplained.

        ``gains`` uses the gains held before each lesson; ``copy`` uses every gain at one.
        Both are decayed sums over observed lessons, divided by the settled change.
        """
        if self._shell is None:
            return []
        out: list[dict[str, float]] = []
        m = self._moments
        for depth in range(self.depth + 1):
            members = self._shell == depth
            total = float(m["yy"][members].sum())
            row: dict[str, Any] = {"shell": depth, "neurons": int(members.sum())}
            for name in ("unexplained", "copy"):
                part = float(m[name][members].sum())
                row["gains" if name == "unexplained" else "copy"] = (
                    part / total if total > 0 else float("nan")
                )
            out.append(row)
        return out
