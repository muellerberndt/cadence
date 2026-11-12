"""A belief over places held in a sheet's own settled activity.

A thing's place is activity over the cells of a plane, on two sheets: ``moving`` (mass that goes
on along the thing's seen movement) and ``rest`` (mass that has come to rest). A cell's activity
is the share of belief that the thing is there. The sheets are neurons of a ``NeuralGraph`` and
every answer is a settled state of the library's patch rule:

- **Carry.** The potentials the sheets settled to are the next moment's drive onto input
  neurons, one for one (``went`` for the moving sheet, ``stayed`` for the sheet at rest), so
  those neurons publish the activity the sheets held. Nothing is stored beside the potentials.
- **Predict.** ``went`` is nine regions, one a shift of a whole cell (none, or one cell along
  either axis or both). Each projects onto ``moving`` through a kernel of three by three cells
  and onto ``rest`` cell to cell. A kernel is one set of synapses shared by every position
  (tie groups), so a lesson at one place teaches every place. ``stayed`` projects onto ``rest``
  cell to cell at an efficacy of 2, the inverse of the neuron's slope at rest, which no lesson
  moves: mass at rest stays in its cell.
- **Normalisation inside the settle.** The two sheets and one reference neuron form one pool of
  an apical compartment (``cadence.brain.Apical``). In every sweep the pool's synaptic input is
  scaled, one gain a row, so that the sheets' potentials keep a fixed length (``level``). A peak
  neither fades nor floods, and the shares between the cells and between the two sheets are
  kept, since the gain is one number for the pool.
- **Evidence.** Cells the caller says are in sight, at a moment with no report, are held at
  zero by the settle's mask: a multiplication of the cell's potential and activity by zero in
  every sweep. A report writes the sheets: the drive is the report's place as weights on the
  four cells around it.

The program beside the sheet (``Space``) chooses, at each moment, which of the nine ``went``
regions takes the carried moving mass: it adds the thing's seen movement to a remainder it
keeps, in cells, and the whole cells of the sum are the moment's shift. It reads the sheets
(the mean or the peak, and the smallest set of cells that holds ``RATE``) and makes each
lesson's target from a position it is given.

Lessons are the library's free and nudged phases with a quadratic nudge
(``Learner(nudge="quadratic", tie_groups=...)``) on the same synapses with the pool's gain held
at one. A lesson's drive is the mass the sheets carried into a moment and its target is the
same mass at the place the thing is found to be at that moment: on ``moving`` where it went on,
on ``rest`` where it stopped. Activity is linear in mass at the sheet's level, so the efficacies
that satisfy the lessons are the shares of mass that go to each offset: the share that goes on
(``keep``), the share that stops, and the spread.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RATE = 0.90  # the share of belief an area is stated at
OFFSETS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))  # index 4 is no shift
PASS = 2.0  # the efficacy that passes activity one for one: the inverse of the slope at rest


class Sheet:
    """The settled belief sheet of ``rows`` things over ``cells`` of a plane.

    ``plane`` is ``(x low, x high, y low, y high)`` in metres and ``cells`` the cells an axis.
    ``level`` is the length the sheets' potentials are held at; at 0.02 the neuron's curve
    bends a thing's place by 0.01 cm over 1,000 moments at rest. ``rate`` is the share of a
    lesson's error a kernel closes in one lesson whose every row drives it. ``born`` says what
    the kernels hold before any lesson: ``"shift"`` (each ``went`` region passes its mass one
    cell along its shift, and none comes to rest) or ``"even"`` (each passes its mass evenly
    onto the nine cells around). ``reach`` is the width, in metres, of the peak a row is
    driven by when every cell that held its mass is seen empty.

    ``begin`` makes the rows, ``write`` writes a report, ``step`` settles one moment with no
    report, ``where`` and ``area`` read the sheets, ``taken`` keeps what a moment's settle was
    driven by and ``teach`` gives a lesson for it. ``kernel`` holds the taught efficacies, one
    a tie group, and ``arrays`` what a file keeps."""

    def __init__(self, plane: tuple[float, float, float, float],
                 cells: tuple[int, int] = (32, 26), *, level: float = 0.02, rate: float = 0.05,
                 born: str = "shift", reach: float = 0.3, kernel: np.ndarray | None = None,
                 backend: Any = "cpu"):
        from ..brain import Apical
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model

        x_low, x_high, y_low, y_high = (float(v) for v in plane)
        nx, ny = int(cells[0]), int(cells[1])
        if not (x_high > x_low and y_high > y_low) or min(nx, ny) < 3:
            raise ValueError("a sheet takes a plane with low below high and 3 cells or more "
                             "an axis")
        if not 0.0 < level <= 0.2 or born not in ("shift", "even"):
            raise ValueError("level lies in (0, 0.2] and born is 'shift' or 'even'")
        self.plane, self.cells = (x_low, x_high, y_low, y_high), (nx, ny)
        self.level, self.rate, self.born = float(level), float(rate), born
        self.reach = float(reach)
        self.count = count = nx * ny
        self._low = np.array([x_low, y_low])
        self.size = np.array([(x_high - x_low) / nx, (y_high - y_low) / ny])  # a cell, in metres
        xs = x_low + (np.arange(nx) + 0.5) * self.size[0]
        ys = y_low + (np.arange(ny) + 0.5) * self.size[1]
        self.centers = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        # neurons: the pool's reference, went (nine regions), stayed, moving, rest
        went = 1 + np.arange(9 * count, dtype=np.int64).reshape(9, count)
        stayed = 1 + 9 * count + np.arange(count, dtype=np.int64)
        moving = stayed + count
        rest = moving + count
        self.neurons = n = 1 + 12 * count
        self._went, self._stayed, self._moving, self._rest = went, stayed, moving, rest
        cell = np.arange(count, dtype=np.int64).reshape(nx, ny)
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        pre: list[np.ndarray] = []
        post: list[np.ndarray] = []
        group: list[np.ndarray] = []
        for d in range(9):
            for o, (ox, oy) in enumerate(OFFSETS):  # a synapse that would leave the plane is none
                inside = ((ix + ox >= 0) & (ix + ox < nx) & (iy + oy >= 0) & (iy + oy < ny)).ravel()
                target = cell[np.clip(ix + ox, 0, nx - 1), np.clip(iy + oy, 0, ny - 1)].ravel()
                pre.append(went[d][inside])
                post.append(moving[target[inside]])
                group.append(np.full(int(inside.sum()), d * 9 + o, dtype=np.int64))
            pre.append(went[d])
            post.append(rest)
            group.append(np.full(count, 81 + d, dtype=np.int64))
        pre.append(stayed)
        post.append(rest)
        group.append(np.full(count, 90, dtype=np.int64))
        pre_all, post_all, group_all = (np.concatenate(x) for x in (pre, post, group))
        self.connectome = Connectome.from_synapses(
            n, pre=pre_all, post=post_all, label="sheet", populations={
                "reference": [0], "went": went.ravel().tolist(), "stayed": stayed.tolist(),
                "moving": moving.tolist(), "rest": rest.tolist()})
        # the connectome orders its synapses by their post neuron: each one's tie group again
        order = np.argsort(post_all * n + pre_all, kind="stable")
        found = np.asarray(self.connectome.post) * n + np.asarray(self.connectome.pre)
        if len(found) != len(order) or not np.array_equal(found, (post_all * n + pre_all)[order]):
            raise RuntimeError("the sheet's synapses were merged or reordered")
        self.groups = group_all[order]
        start = np.zeros(91)
        for d in range(9):
            start[d * 9 : d * 9 + 9] = PASS / 9.0 if born == "even" else 0.0
            if born == "shift":
                start[d * 9 + d] = PASS
        start[90] = PASS
        if kernel is not None:
            start = np.asarray(kernel, dtype=np.float64).reshape(91).copy()
        pools = np.full(n, -1, dtype=np.int64)
        pools[0], pools[moving], pools[rest] = 0, 0, 0
        # the pool's potentials keep the length ``level``: the reference's bias is 1, so the
        # pool's own drive has a length of 1, and the compartment scales the synaptic input to
        # ``fraction`` times that (its floor far below any mass the sheets hold)
        self._apical = Apical(np.ones(self.connectome.synapses, dtype=bool), pools,
                              fraction=self.level, floor=1e-6)
        self._bias = np.zeros(n)
        self._bias[0] = 1.0
        self._model = learning_neuron_model(dt=1.0)
        self._graph_class, self._backend = Graph, backend
        self._plastic = self.groups < 90  # the carry of mass at rest is no lesson's to move
        self._outputs = np.concatenate([moving, rest])
        self._learner: Any = None
        self.kernel = start
        self._build()
        self.lessons = 0
        self.rows = 0
        self.work = {"moments": 0, "sweeps_rows": 0.0, "lesson_rows": 0.0}

    # -- the graph

    def _build(self) -> None:
        """The settling graph of the present kernel: the same synapses under the pool."""
        self.graph = self._graph_class(
            self.connectome, self._model, efficacy=self.kernel[self.groups], bias=self._bias,
            apical=self._apical, backend=self._backend)

    def _teacher(self) -> Any:
        """The learner of the kernels: the library's free and nudged phases with a quadratic
        nudge on the sheets, the synapses tied by group, the pool's gain held at one."""
        if self._learner is None:
            from ..learning import Learner, LearnerConfig

            plain = self._graph_class(self.connectome, self._model,
                                      efficacy=self.kernel[self.groups], bias=self._bias,
                                      backend=self._backend)
            # a tie group steps by the mean of its members' contrasts, a few lit cells among
            # ``count``, at activities of ``level``: the rate that closes ``rate`` of an error
            eta = 2.2 * self.rate * self.count / self.level**2
            config = LearnerConfig(nudge="quadratic", beta=0.1, eta=eta, eta_bias=0.0,
                                   free_steps=2, nudged_steps=4, tolerance=None)
            self._learner = Learner(
                plain, self._outputs.tolist(), config, reciprocal=False, tie_groups=self.groups,
                plastic_synapses=self._plastic, plastic_neurons=np.zeros(self.neurons, dtype=bool))
        return self._learner

    # -- places as weights on cells

    def code(self, points: np.ndarray) -> np.ndarray:
        """``(rows, 2)`` places as ``(rows, cells)`` weights on the four cells around each,
        summing to one, whose mean is the place. A place beyond the outer cells' middles is
        read at them."""
        nx, ny = self.cells
        at = (np.asarray(points, dtype=np.float64).reshape(-1, 2) - self._low) / self.size - 0.5
        at = np.clip(at, 0.0, [nx - 1.0, ny - 1.0])
        ix = np.minimum(at[:, 0].astype(np.intp), nx - 2)
        iy = np.minimum(at[:, 1].astype(np.intp), ny - 2)
        ax, ay = at[:, 0] - ix, at[:, 1] - iy
        out = np.zeros((len(at), self.count))
        rows = np.arange(len(at), dtype=np.intp)
        out[rows, ix * ny + iy] = (1 - ax) * (1 - ay)
        out[rows, (ix + 1) * ny + iy] = ax * (1 - ay)
        out[rows, ix * ny + iy + 1] = (1 - ax) * ay
        out[rows, (ix + 1) * ny + iy + 1] = ax * ay
        return out

    # -- a scene

    def begin(self, rows: int) -> None:
        """``rows`` things, their sheets empty until written."""
        self.rows = int(rows)
        self.potential = np.zeros((self.rows, 2, self.count))  # moving, rest: what is carried
        self.remainder = np.zeros((self.rows, 2))  # the movement not yet moved, in cells
        self.gone = np.zeros(self.rows, dtype=bool)  # every cell that held mass was in sight
        self._taken: dict[str, np.ndarray] = {}

    def write(self, rows: np.ndarray, points: np.ndarray, moving: np.ndarray) -> None:
        """A report writes the sheets of ``rows`` (a mask): the place as weights on the four
        cells around it, at the sheet's level, on ``moving`` where ``moving`` says so and on
        ``rest`` otherwise. ``points`` is ``(rows, 2)`` and ``moving`` one flag a row, both for
        every row of the sheet."""
        rows = np.asarray(rows, dtype=bool).reshape(self.rows)
        if not rows.any():
            return
        weights = self.code(np.asarray(points, dtype=np.float64).reshape(self.rows, 2)[rows])
        weights *= self.level / np.linalg.norm(weights, axis=1, keepdims=True)
        goes = np.asarray(moving, dtype=bool).reshape(self.rows)[rows]
        self.potential[rows] = np.stack([weights * goes[:, None], weights * ~goes[:, None]], axis=1)
        self.remainder[rows] = 0.0
        self.gone[rows] = False

    def step(self, rows: np.ndarray, movement: np.ndarray, lit: np.ndarray | None = None) -> None:
        """One moment with no report for ``rows`` (a mask). ``movement`` is each thing's seen
        movement a step, ``(rows, 2)`` in metres: the program adds it to the row's remainder and
        the whole cells of the sum name the ``went`` region that takes the carried moving mass.
        ``lit`` says which cells the camera sees at this moment, ``(cells,)`` or ``(rows,
        cells)``; those cells are held at zero through the settle. Where that leaves a row no
        mass, the thing moved unseen: the row is settled again on a peak ``reach`` wide around
        where it was held, with the same cells at zero, so its mass lies on the cells out of
        sight near that place. Where that too leaves none, the row keeps the belief it had
        and is marked ``gone``."""
        rows = np.asarray(rows, dtype=bool).reshape(self.rows)
        index = np.flatnonzero(rows)
        if not len(index):
            self._taken = {}
            return
        asked = np.asarray(movement, dtype=np.float64).reshape(self.rows, 2)[index] / self.size
        total = self.remainder[index] + asked
        whole = np.clip(np.rint(total), -1, 1)
        self.remainder[index] = np.clip(total - whole, -1.0, 1.0)
        shift = ((whole[:, 0] + 1) * 3 + (whole[:, 1] + 1)).astype(np.intp)
        drive = np.zeros((len(index), self.neurons))
        for k, row in enumerate(index):
            drive[k, self._went[shift[k]]] = self.potential[row, 0]
        drive[:, self._stayed] = self.potential[index, 1]
        mask = None
        if lit is not None:
            seen = np.asarray(lit, dtype=bool)
            seen = np.broadcast_to(seen if seen.ndim == 1 else seen[index],
                                   (len(index), self.count))
            if seen.any():
                mask = np.ones((len(index), self.neurons))
                mask[:, self._moving] = ~seen
                mask[:, self._rest] = ~seen
        v = self._settle(drive, mask)
        empty = ~(np.maximum(v, 0.0).sum(axis=(1, 2)) > 0.0)
        if mask is not None and empty.any():
            # every cell that held mass is in sight and no report came: the thing moved
            # unseen. Its sheet at rest is driven by a wide peak around where it was held
            # (``reach`` metres), and the cells in sight are held at zero as before
            was = np.maximum(np.tanh(self.potential[index[empty]] / 2.0), 0.0).sum(axis=1)
            at = (was @ self.centers) / np.maximum(was.sum(axis=1, keepdims=True), 1e-300)
            apart = np.linalg.norm(self.centers[None] - at[:, None], axis=2)
            wide = np.exp(-0.5 * (apart / self.reach) ** 2)
            again = np.zeros((int(empty.sum()), self.neurons))
            again[:, self._stayed] = wide * (self.level / np.linalg.norm(wide, axis=1,
                                                                         keepdims=True))
            v[empty] = self._settle(again, mask[empty])
            still = ~(np.maximum(v[empty], 0.0).sum(axis=(1, 2)) > 0.0)
            if still.any():  # no cell within reach is out of sight: the belief is kept
                lost = np.flatnonzero(empty)[still]
                v[lost] = self._settle(drive[lost], None)
                self.gone[index[lost]] = True
            self.remainder[index[empty]] = 0.0
        self.potential[index] = v
        self._taken = {"rows": index, "drive": drive, "shift": shift,
                       "remainder": self.remainder[index].copy()}

    def _settle(self, drive: np.ndarray, mask: np.ndarray | None, sweeps: int = 1) -> np.ndarray:
        """Settle the graph on ``drive``: the sheets' potentials ``(rows, 2, cells)``. The
        synapses run from the carried inputs to the sheets and no synapse leads back. The
        settle begins with every input neuron at the potential its drive gives it, which is
        its fixed point, and one sweep then brings the sheets to theirs; further ``sweeps``
        move nothing."""
        from ..brain import BrainState

        begun = drive.copy()
        begun[:, 0] = 1.0  # the reference rests at its bias
        state = BrainState(v=begun, activation=self._model.activation(begun),
                           adaptation=np.zeros_like(begun), steps=0)
        state = self.graph.settle_batch(drive, steps=sweeps, tolerance=None, mask=mask,
                                        state=state)
        v = np.asarray(state.v, dtype=np.float64)
        self.work["moments"] += len(drive)
        self.work["sweeps_rows"] += float(sweeps) * len(drive)
        return np.stack([v[:, self._moving], v[:, self._rest]], axis=1)

    # -- answers

    def mass(self) -> np.ndarray:
        """Each row's belief ``(rows, 2, cells)``: the activity above rest of ``moving`` and
        ``rest``, as shares that sum to one a row (zero for a row never written)."""
        held = np.maximum(np.tanh(self.potential / 2.0), 0.0)
        total = held.sum(axis=(1, 2), keepdims=True)
        return np.asarray(np.divide(held, total, out=np.zeros_like(held), where=total > 0))

    def where(self, read: str = "mean") -> np.ndarray:
        """Each row's place ``(rows, 2)``: the mean of its belief over the cells' middles, or
        with ``read="peak"`` the mean within one cell of its most believed cell. The moving
        share is read ahead by the part of a cell its movement has yet to cross (the
        remainder)."""
        mass = self.mass()
        both = mass.sum(axis=1)
        if read == "peak":
            nx, ny = self.cells
            top = both.argmax(axis=1)
            tx, ty = np.divmod(top, ny)
            cx, cy = np.divmod(np.arange(self.count), ny)
            near = (np.abs(cx[None] - tx[:, None]) <= 1) & (np.abs(cy[None] - ty[:, None]) <= 1)
            both = both * near
            both = both / np.maximum(both.sum(axis=1, keepdims=True), 1e-300)
            going = (mass[:, 0] * near).sum(axis=1) / np.maximum(
                (mass.sum(axis=1) * near).sum(axis=1), 1e-300)
        else:
            going = mass[:, 0].sum(axis=1)
        return np.asarray(both @ self.centers + going[:, None] * self.remainder * self.size)

    def area(self, rate: float = RATE) -> np.ndarray:
        """Each row's area ``(rows, cells x, cells y)``: the smallest set of cells that holds
        ``rate`` of its belief, as flags."""
        both = self.mass().sum(axis=1)
        order = np.argsort(-both, axis=1)
        ranked = np.take_along_axis(both, order, axis=1)
        total = ranked.sum(axis=1, keepdims=True)
        need = (np.cumsum(ranked, axis=1) < rate * total).sum(axis=1) + 1
        need = np.where(ranked.sum(axis=1) > 0, need, 0)  # a row with no mass has no cell
        chosen = np.arange(self.count)[None, :] < need[:, None]
        out = np.zeros((self.rows, self.count), dtype=bool)
        np.put_along_axis(out, order, chosen, axis=1)
        return out.reshape(self.rows, *self.cells)

    def holds(self, points: np.ndarray, rate: float = RATE) -> np.ndarray:
        """Whether each row's area holds its point, ``(rows, 2)``: one flag a row."""
        nx, ny = self.cells
        at = np.floor((np.asarray(points, dtype=np.float64).reshape(self.rows, 2) - self._low)
                      / self.size).astype(np.intp)
        ix, iy = np.clip(at[:, 0], 0, nx - 1), np.clip(at[:, 1], 0, ny - 1)
        return np.asarray(self.area(rate)[np.arange(self.rows), ix, iy])

    # -- lessons

    def taken(self) -> dict[str, np.ndarray]:
        """What the latest ``step`` settled on: its rows, their drive, each row's shift and its
        remainder after the step. ``teach`` takes it, at once or some moments later."""
        return self._taken

    def teach(self, taken: dict[str, np.ndarray], rows: np.ndarray, points: np.ndarray,
              kept: np.ndarray) -> float:
        """A lesson for a settle that was ``taken``: the rows in ``rows`` (a mask over the
        sheet's rows) are taught that the thing was at ``points``, ``(rows, 2)``, at that
        moment, going on (``kept``) or at rest. The target is the mass the settle was driven by
        at that place: on ``moving``, behind the place by the row's remainder, or on ``rest``.
        Gives the mean distance between the free answer and the target, as a share of the
        target."""
        if not taken:
            return 0.0
        rows = np.asarray(rows, dtype=bool).reshape(self.rows)
        chosen = rows[taken["rows"]]
        if not chosen.any():
            return 0.0
        index = taken["rows"][chosen]
        drive = taken["drive"][chosen]
        at = np.asarray(points, dtype=np.float64).reshape(self.rows, 2)[index]
        goes = np.asarray(kept, dtype=bool).reshape(self.rows)[index]
        carried = np.maximum(np.tanh(drive / 2.0), 0.0)
        total = carried[:, self._went.ravel()].sum(axis=1) + carried[:, self._stayed].sum(axis=1)
        behind = at - taken["remainder"][chosen] * self.size
        weights = self.code(np.where(goes[:, None], behind, at)) * total[:, None]
        target = np.zeros_like(drive)
        target[:, self._moving] = weights * goes[:, None]
        target[:, self._rest] = weights * ~goes[:, None]
        learner = self._teacher()
        free = learner.free(drive)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        learner.update(free, plus, minus)
        efficacy = np.asarray(learner.brain.efficacy, dtype=np.float64)
        kernel = np.zeros(91)
        np.maximum.at(kernel, self.groups, efficacy)  # a group's members hold one efficacy
        kernel = np.maximum(kernel, 0.0)  # the kernels are excitatory: none goes below zero
        kernel[90] = PASS
        if (efficacy < 0.0).any():
            learner.brain = learner.brain.with_parameters(efficacy=kernel[self.groups])
        self.kernel = kernel
        self._build()
        self.lessons += 1
        self.work["lesson_rows"] += float(len(index))
        answer = np.asarray(free.activation)[:, self._outputs]
        wanted = target[:, self._outputs]
        return float(np.abs(answer - wanted).sum() / max(np.abs(wanted).sum(), 1e-300))

    # -- what a kernel holds

    def shares(self) -> dict[str, np.ndarray]:
        """The kernels as shares of a ``went`` region's mass: ``keep`` (the share that goes on,
        one a shift), ``stop`` (the share that comes to rest), ``lost`` (the share that goes
        nowhere) and ``spread`` (of the share that goes on, the part that lands off the
        shift's own cell)."""
        on = self.kernel[:81].reshape(9, 9) / PASS
        stop = self.kernel[81:90] / PASS
        keep = on.sum(axis=1)
        own = on[np.arange(9), np.arange(9)]
        return {"keep": keep, "stop": stop, "lost": 1.0 - keep - stop,
                "spread": 1.0 - own / np.maximum(keep, 1e-300)}

    def arrays(self) -> dict[str, np.ndarray]:
        """What a file keeps of a sheet: its kernels."""
        return {"kernel": self.kernel.copy()}

    def to_dict(self) -> dict[str, Any]:
        return {"plane": list(self.plane), "cells": list(self.cells), "level": self.level,
                "rate": self.rate, "born": self.born, "reach": self.reach,
                "lessons": int(self.lessons)}


__all__ = ["OFFSETS", "PASS", "RATE", "Sheet"]
