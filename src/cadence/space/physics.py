"""Learned physics on a map: how things move, learned from the things the brain watches move.

The map (``cadence.space.map.Map``) holds where things are. Physics gives each of its cells
velocity cells: a thing's mass at a cell is shared among velocities on a grid of ``per`` by
``per`` velocities over ``(-top, top)`` metres a moment on each axis. The still velocity is the
map's own sheet, carried exactly as the map carries it. Every other velocity is a region of its
own, ``went`` (the mass the moment before) and ``moving`` (where it is now).

- **Sight.** A report writes the thing's place, as the map does. Its velocity is what the brain
  sees of its motion: the report less the place the map held for that thing the moment before,
  as direction-selective cells give it. The seen motion is written as weights on the four
  velocities around it and blended into the velocity the thing carried, at the share ``blend``.
- **Carry.** The mass on each moving velocity is the next moment's drive on ``went``, and one
  kernel of three by three cells for each velocity, tied across every cell and every kind, drives
  ``moving``. The kernels are born as "stay": a brain that has watched nothing answers as the map
  does.
- **Lessons.** Whenever a thing is seen at two moments in a row, the mass it carried the moment
  before (its place and velocities) is the drive and the same mass moved by the motion seen is
  the target. The library's free and nudged phases with a quadratic nudge step the kernels. Each
  velocity's kernel learns where the mass of things seen moving at that velocity is found a moment
  later: going on, stopping, or turning.
- **Out of view.** The kernels move the mass, and the cells the camera sees empty what was moved
  into them, so a thing that would have come into view and did not is held where it can be.

Nothing about movement is given: the kernels hold what the brain learned.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PASS = 2.0  # the efficacy that passes activity one for one: the inverse of the slope at rest
OFFSETS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))  # index 4 is no offset


class Physics:
    """Velocity cells for a map of ``count`` cells (``cells``), learned kernels and their lessons.

    ``top`` is the fastest velocity in metres a moment, ``per`` the velocities on each axis
    (odd, so that one of them is still), ``blend`` the share of a moment's seen motion in a
    thing's velocity and ``rate`` the share of a lesson's error one lesson closes. ``kernel``
    holds taught efficacies, one a velocity and offset, ``(velocities, 9)``.

    ``mass`` holds the carried potentials ``(cells, units, velocities)`` of the moving
    velocities; the still velocity is the map's sheet less their sum."""

    def __init__(self, cells: tuple[int, int], size: np.ndarray, units: int, *,
                 top: float = 0.024, per: int = 5, blend: float = 0.1, rate: float = 0.1,
                 kernel: np.ndarray | None = None, backend: Any = "cpu"):
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model

        nx, ny = int(cells[0]), int(cells[1])
        if per < 3 or per % 2 == 0 or not top > 0 or not 0 < blend <= 1:
            raise ValueError("physics takes an odd per of 3 or more, a top above 0 and a blend "
                             "in (0, 1]")
        self.cells, self.size, self.units = (nx, ny), np.asarray(size, dtype=np.float64), units
        self.top, self.per, self.blend, self.rate = float(top), int(per), float(blend), float(rate)
        self.count = count = nx * ny
        axis = np.linspace(-self.top, self.top, self.per)
        grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        self._still = (self.per * self.per) // 2
        keep = np.arange(len(grid)) != self._still
        self.velocities = grid[keep]  # the moving velocities, metres a moment
        self.speeds = len(self.velocities)
        b = self.speeds
        went = np.arange(b * count, dtype=np.int64).reshape(b, count)
        moving = went + b * count
        self.neurons = 2 * b * count
        self._went, self._moving = went, moving
        cell = np.arange(count, dtype=np.int64).reshape(nx, ny)
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        pre, post, group = [], [], []
        for v in range(b):
            for o, (ox, oy) in enumerate(OFFSETS):  # a synapse that would leave the plane is none
                inside = ((ix + ox >= 0) & (ix + ox < nx) & (iy + oy >= 0)
                          & (iy + oy < ny)).ravel()
                target = cell[np.clip(ix + ox, 0, nx - 1), np.clip(iy + oy, 0, ny - 1)].ravel()
                pre.append(went[v][inside])
                post.append(moving[v][target[inside]])
                group.append(np.full(int(inside.sum()), v * 9 + o, dtype=np.int64))
        pre_all, post_all, group_all = (np.concatenate(x) for x in (pre, post, group))
        self.connectome = Connectome.from_synapses(
            self.neurons, pre=pre_all, post=post_all, label="physics",
            populations={"went": went.ravel().tolist(), "moving": moving.ravel().tolist()})
        order = np.argsort(post_all * self.neurons + pre_all, kind="stable")
        found = np.asarray(self.connectome.post) * self.neurons + np.asarray(self.connectome.pre)
        if not np.array_equal(found, (post_all * self.neurons + pre_all)[order]):
            raise RuntimeError("the physics synapses were merged or reordered")
        self.groups = group_all[order]
        # one velocity's synapses alone (its went onto its moving region): the graph a moment
        # settles, a velocity at a time, for the velocities that carry mass
        one = self.groups < 9
        self._one = Connectome.from_synapses(
            2 * count, pre=np.asarray(self.connectome.pre)[one],
            post=np.asarray(self.connectome.post)[one] - (b - 1) * count, label="velocity",
            populations={"went": list(range(count)), "moving": list(range(count, 2 * count))})
        self._one_groups = self.groups[one] % 9
        born = np.zeros((b, 9))
        born[:, 4] = PASS  # stay
        self.kernel = born if kernel is None else np.asarray(kernel, dtype=np.float64).reshape(
            b, 9).copy()
        self._model = learning_neuron_model(dt=1.0)
        self._graph_class, self._backend = Graph, backend
        self._learner: Any = None
        self._build()
        self.lessons = 0
        self.reset()

    def _build(self) -> None:
        self.graph = self._graph_class(
            self.connectome, self._model, efficacy=self.kernel.ravel()[self.groups],
            backend=self._backend)
        self._graphs = [self._graph_class(self._one, self._model,
                                          efficacy=self.kernel[v][self._one_groups],
                                          backend=self._backend)
                        for v in range(self.speeds)]

    def reset(self) -> None:
        """Empty the moving velocities."""
        self.mass = np.zeros((self.count, self.units, self.speeds))

    # -- velocities as weights

    def code(self, motion: np.ndarray) -> np.ndarray:
        """A seen motion, metres a moment, as weights on the four velocities around it, summing
        to one, whose mean is the motion (beyond ``top`` it is read at ``top``): ``(per *
        per,)`` over the whole grid, the still velocity included."""
        at = (np.clip(np.asarray(motion, dtype=np.float64).reshape(2), -self.top, self.top)
              + self.top) / (2.0 * self.top) * (self.per - 1)
        lo = np.minimum(at.astype(np.intp), self.per - 2)
        a = at - lo
        out = np.zeros((self.per, self.per))
        out[lo[0], lo[1]] = (1 - a[0]) * (1 - a[1])
        out[lo[0] + 1, lo[1]] = a[0] * (1 - a[1])
        out[lo[0], lo[1] + 1] = (1 - a[0]) * a[1]
        out[lo[0] + 1, lo[1] + 1] = a[0] * a[1]
        return out.ravel()

    def shift(self, mass: np.ndarray, by: np.ndarray) -> np.ndarray:
        """``(cells, ...)`` mass moved by ``by`` metres, each cell's share split between the
        four cells around where it lands (mass that would leave the plane is lost): the mass a
        thing carried, where the seen motion took it."""
        nx, ny = self.cells
        grid = np.asarray(mass, dtype=np.float64).reshape(nx, ny, -1)
        at = np.asarray(by, dtype=np.float64).reshape(2) / self.size
        lo = np.floor(at).astype(np.intp)
        a = at - lo
        out = np.zeros_like(grid)
        for dx, wx in ((lo[0], 1.0 - a[0]), (lo[0] + 1, a[0])):
            for dy, wy in ((lo[1], 1.0 - a[1]), (lo[1] + 1, a[1])):
                if wx * wy <= 0.0:
                    continue
                xs = slice(max(dx, 0), nx + min(dx, 0))
                ys = slice(max(dy, 0), ny + min(dy, 0))
                out[xs, ys] += wx * wy * grid[max(-dx, 0) : nx - max(dx, 0),
                                              max(-dy, 0) : ny - max(dy, 0)]
        return out.reshape(np.shape(mass))

    def attribute(self, drive: np.ndarray, seen: np.ndarray, total: float) -> np.ndarray:
        """A lesson's target: the mass a thing carried on each velocity, ``(cells,
        velocities)``, moved by that velocity and by the share of the motion seen that the
        thing's own velocity did not hold. The thing's velocity is the mean over its whole mass
        (``total``, the still share at no motion); the motion seen less it is what the thing did
        that its velocity did not say, and every velocity's mass is taught to have done it.
        Where things keep their motion the kernels learn to move each velocity's mass by that
        velocity; where they stop or turn, by less or aside."""
        mass = np.asarray(drive, dtype=np.float64)
        own = (mass.sum(axis=0) @ self.velocities) / max(total, float(mass.sum()), 1e-300)
        out = np.zeros_like(mass)
        for v in np.flatnonzero(mass.any(axis=0)):
            out[:, v] = self.shift(mass[:, v], self.velocities[v] + np.asarray(seen) - own)
        return out

    def moving_share(self, whole: np.ndarray) -> np.ndarray:
        """Weights over the whole grid as the moving velocities' shares ``(velocities,)``."""
        return np.delete(np.asarray(whole, dtype=np.float64), self._still)

    # -- a moment

    def carry(self) -> np.ndarray:
        """Settle the kernels on the carried mass: the moving mass of the next moment,
        ``(cells, units, velocities)``. Each velocity's synapses are a graph of their own, and
        a velocity is settled on the code units that carry mass on it; the rest carry none (a
        neuron with no drive rests). Before any lesson the kernels stay, and the mass is
        carried as it is."""
        if not self.lessons or not self.mass.any():
            return np.array(self.mass, dtype=np.float64)
        from ..brain import BrainState

        out = np.zeros_like(self.mass)
        count = self.count
        for v in np.flatnonzero(self.mass.any(axis=(0, 1))):
            units = np.flatnonzero(self.mass[:, :, v].any(axis=0))
            drive = np.zeros((len(units), 2 * count))
            drive[:, :count] = self.mass[:, units, v].T
            state = BrainState(v=drive.copy(), activation=self._model.activation(drive),
                               adaptation=np.zeros_like(drive), steps=0)
            state = self._graphs[v].settle_batch(drive, steps=1, tolerance=None, state=state)
            out[:, units, v] = np.asarray(state.v, dtype=np.float64)[:, count:].T
        return out

    # -- lessons

    def teach(self, drives: np.ndarray, targets: np.ndarray) -> float:
        """Lessons: ``drives`` and ``targets``, ``(rows, cells, velocities)`` potentials, the
        mass a thing carried into a moment and the same mass where it was seen. Gives the mean
        distance between the free answer and the target as a share of the target."""
        import dataclasses

        from ..learning import Learner, LearnerConfig

        rows = len(drives)
        drive = np.zeros((rows, self.neurons))
        target = np.zeros((rows, self.neurons))
        for row in range(rows):
            drive[row, self._went.ravel()] = np.asarray(drives[row]).T.ravel()
            target[row, self._moving.ravel()] = np.tanh(np.asarray(targets[row]).T.ravel() / 2.0)
        if self._learner is None:
            self._learner = Learner(
                self.graph, self._moving.ravel().tolist(),
                LearnerConfig(nudge="quadratic", beta=0.1, eta=1.0, eta_bias=0.0, free_steps=2,
                              nudged_steps=4, tolerance=None),
                reciprocal=False, tie_groups=self.groups,
                plastic_neurons=np.zeros(self.neurons, dtype=bool))
        learner = self._learner
        lit = np.tanh(np.maximum(drive, 0.0) / 2.0)
        # a tie group steps by the mean of its ``count`` members' contrasts, and a member's
        # contrast is its sender's activity times about 0.275 of its receiver's error (the
        # slopes above and below rest, 0.5 and 0.05, over the two phases); a receiver moves by
        # half its drive's change. The rate that closes ``rate`` of a lesson's error:
        energy = float((lit**2).sum() / max(rows, 1))
        eta = self.rate * self.count / (0.5 * 0.275 * max(energy, 1e-30))
        learner.config = dataclasses.replace(learner.config, eta=eta)
        free = learner.free(drive)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        learner.update(free, plus, minus)
        efficacy = np.asarray(learner.brain.efficacy, dtype=np.float64)
        flat = np.zeros(self.speeds * 9)
        np.maximum.at(flat, self.groups, efficacy)  # a group's members hold one efficacy
        kernel = np.maximum(flat, 0.0).reshape(self.speeds, 9)  # mass moves, never less
        whole = kernel.sum(axis=1, keepdims=True)  # and a thing does not multiply
        self.kernel = kernel * np.minimum(1.0, PASS / np.maximum(whole, 1e-300))
        if (efficacy < 0.0).any() or (whole > PASS).any():
            learner.brain = learner.brain.with_parameters(efficacy=self.kernel.ravel()[self.groups])
        self.graph = learner.brain
        self.lessons += rows
        answer = np.asarray(free.activation)[:, self._moving.ravel()]
        wanted = target[:, self._moving.ravel()]
        return float(np.abs(answer - wanted).sum() / max(np.abs(wanted).sum(), 1e-300))

    # -- what was learned

    def shares(self) -> dict[str, np.ndarray]:
        """The kernels as shares of a velocity's mass: ``kept`` (the share that stays on the
        map), ``went`` (the mean offset of that share, in metres a moment, ``(velocities,
        2)``) and ``stayed`` (the share left in its own cell)."""
        share = self.kernel / PASS
        kept = share.sum(axis=1)
        offsets = np.array(OFFSETS, dtype=np.float64) * self.size[None, :]
        went = (share @ offsets) / np.maximum(kept, 1e-300)[:, None]
        return {"kept": kept, "went": went, "stayed": share[:, 4] / np.maximum(kept, 1e-300)}

    def settings(self) -> dict[str, Any]:
        return {"top": self.top, "per": self.per, "blend": self.blend, "rate": self.rate,
                "lessons": int(self.lessons)}


__all__ = ["OFFSETS", "PASS", "Physics"]
