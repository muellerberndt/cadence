"""A map of things: what is where, held in a sheet's own settled activity.

The map is one sheet of neurons, ``cells`` of a plane by ``units`` of a kind's code. A unit at a
cell is active where a thing whose kind's code holds that unit is at that cell. The sheet's
activity is the whole state: no list of things is given to the map and none is kept beside it.

- **Carry.** The potentials the sheet settled to are the next moment's drive onto the sheet's
  own neurons, one for one. A neuron under a drive and no synapse settles at that drive, so
  with no sensory information the activity stays exactly as it is, and a thing's last known
  place stays, for any number of moments. Nothing sets a gain or a height in the carry.
- **A thing enters when it is sensed.** A report is a kind and a place. It drives the four
  cells around the place on the units of the kind's code. Two things of one kind are two peaks
  on the same units.
- **Sight of an empty place.** A cell the camera sees is held at zero by the settle's mask (a
  multiplication in every sweep), unless a report writes the cell at this moment. ``empty``
  says what becomes of a thing whose place is seen empty with no report of its kind nearby:
  ``"near"`` drives it onto the cells out of sight around where it was held, ``"clear"`` leaves
  it known by its kind with no place, ``"keep"`` leaves it where it was.
- **Reading by settling, both ways.** For a kind as a cue the carried potentials drive the
  input region ``held``, which projects onto the sheet cell to cell. Each code unit's cells and
  one reference neuron form a pool of an apical compartment (``cadence.brain.Apical``), and
  the kind's code drives the references. A pool's gain is a multiplication by its reference,
  so a pool whose reference is silent passes nothing, and ``where`` (a neuron a cell) settles
  on the places of that kind. A place as a cue keeps the cells around it (the settle's mask on
  the rest) and ``what`` (a neuron a code unit) settles on the code of the thing there.
- **Time and doubt.** ``clock`` (a neuron a cell) carries its own potential and a bias of
  one, so it counts the moments since a sense last wrote its cell. ``doubt`` (a neuron a cell)
  carries a one where the cell holds a thing that was not seen to be there: the cells a thing
  is driven onto when its place is seen empty.

The map's graph has no taught synapse: it is wiring and carried activity. The program beside
it turns a report into the drive on four cells, passes the camera's seen cells as the mask,
carries the potentials forward, sets a question's cue and reads the peaks.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

PASS = 2.0  # the efficacy that passes activity one for one: the inverse of the slope at rest
FORMAT = "cadence-space-map/1"


def sparse_codes(kinds: int, units: int, weight: int = 3, seed: int = 0) -> np.ndarray:
    """``kinds`` codes of ``weight`` units among ``units``, any two sharing one unit at most:
    ``(kinds, units)`` of zeros and ones."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 320, int(units)]))
    every = list(itertools.combinations(range(units), weight))
    chosen: list[tuple[int, ...]] = []
    used: set[tuple[int, int]] = set()
    for index in rng.permutation(len(every)):
        pairs = set(itertools.combinations(every[index], 2))
        if not pairs & used:
            chosen.append(every[index])
            used |= pairs
            if len(chosen) == kinds:
                break
    if len(chosen) < kinds:
        raise ValueError(f"{units} units hold {len(chosen)} codes of {weight} units that share "
                         f"one unit at most, not {kinds}: more units")
    out = np.zeros((kinds, units))
    for row, held in enumerate(chosen):
        out[row, list(held)] = 1.0
    return out


class Map:
    """What is where, over ``cells`` of a plane, for kinds given as codes of ``units`` units.

    ``plane`` is ``(x low, x high, y low, y high)`` in metres. A kind is a number: a row of
    ``codes``. Without ``kinds`` the codes are one unit a kind (``units`` kinds, one sheet a
    kind); with ``kinds`` above ``units`` each kind takes ``weight`` units and any two kinds
    share one unit at most; ``codes`` gives a table of the caller's own. ``level`` is the
    length a report's potentials are written at on each of its units.

    Settings of the rules, each stated with its reading in the guide: ``empty`` (above),
    ``reach`` (the width in metres of the peak a thing is driven by when its place is seen
    empty; cells farther than two reaches take none), ``step`` (a report within this distance
    of a piece of its kind's activity is that thing, seen again) and ``stride`` (a report of a
    kind with none of it within ``step``: the nearest piece of the kind within ``stride`` is
    that thing, moved unseen; with none, it is a new thing) and ``margin`` (a place a sense wrote
    the moment before counts as seen where the camera sees a place this near it).

    With ``trace`` every settle begins with the sheet at rest, runs two sweeps and is kept
    by sweep in ``trajectory``: ``asked`` (``"moment"``, ``"where_is"`` or ``"what_is_at"``),
    ``sweeps`` (2) and every region's activity above rest by sweep: ``map`` and ``held``
    ``(sweeps + 1, cells x, cells y, units)``; ``where``, ``clock`` and ``doubt``
    ``(sweeps + 1, cells x, cells y)``; ``what``, ``reference`` and ``lost`` ``(sweeps + 1,
    units)``. Row 0 is the state the settle begins from, row 1 the sheet settled and row 2
    ``where`` and ``what`` settled on it. The moments a cell's clock counts are its potential
    (``Map.clock``), which its activity saturates over.

    ``moment`` takes one moment's reports and the cells in sight. ``where_is`` and
    ``what_is_at`` ask by settling, ``things`` reads the inventory off the map and ``reset``
    empties it. ``arrays`` is the carried state, ``save`` and ``load`` write and read it."""

    def __init__(self, plane: tuple[float, float, float, float],
                 cells: tuple[int, int] = (32, 26), *, units: int = 16, kinds: int | None = None,
                 weight: int = 3, codes: np.ndarray | None = None, level: float = 0.02,
                 empty: str = "near", reach: float = 0.15, step: float = 0.15,
                 stride: float = 0.5, margin: float = 0.02, seed: int = 0,
                 trace: bool = False, backend: Any = "cpu"):
        from ..brain import Apical
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model

        x_low, x_high, y_low, y_high = (float(v) for v in plane)
        nx, ny = int(cells[0]), int(cells[1])
        if not (x_high > x_low and y_high > y_low) or min(nx, ny) < 3:
            raise ValueError("a map takes a plane with low below high and 3 cells or more an axis")
        if empty not in ("near", "clear", "keep"):
            raise ValueError("empty is 'near', 'clear' or 'keep'")
        if codes is not None:
            table = np.asarray(codes, dtype=np.float64)
            if table.ndim != 2 or not np.isin(table, (0.0, 1.0)).all() or not (
                    table.any(axis=1).all()):
                raise ValueError("codes is (kinds, units) of zeros and ones, a one in every row")
            units = table.shape[1]
        elif kinds is None or int(kinds) <= int(units):
            table = np.eye(int(units))[: int(units) if kinds is None else int(kinds)]
        else:
            table = sparse_codes(int(kinds), int(units), int(weight), seed)
        self.codes = table
        self.plane, self.cells, self.units = (x_low, x_high, y_low, y_high), (nx, ny), int(units)
        self.level, self.empty = float(level), empty
        self.reach, self.step, self.stride = float(reach), float(step), float(stride)
        self.margin = float(margin)
        self.settings = {"plane": list(self.plane), "cells": [nx, ny], "units": self.units,
                         "level": self.level, "empty": empty, "reach": self.reach,
                         "step": self.step, "stride": self.stride, "margin": self.margin}
        self.count = count = nx * ny
        d = self.units
        self._low = np.array([x_low, y_low])
        self.size = np.array([(x_high - x_low) / nx, (y_high - y_low) / ny])  # a cell, in metres
        xs = x_low + (np.arange(nx) + 0.5) * self.size[0]
        ys = y_low + (np.arange(ny) + 0.5) * self.size[1]
        self.centers = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        # neurons, each region one run:
        # reference | held | map | where | what | clock | lost | doubt
        ends = np.cumsum([d, count * d, count * d, count, d, count, d, count])
        self._runs = {name: slice(int(a), int(b)) for name, a, b in zip(
            ("reference", "held", "map", "where", "what", "clock", "lost", "doubt"),
            np.concatenate([[0], ends[:-1]]), ends, strict=True)}
        self.neurons = n = int(ends[-1])
        held = np.arange(ends[0], ends[1], dtype=np.int64)
        sheet = held + count * d
        where = np.arange(ends[2], ends[3], dtype=np.int64)
        what = np.arange(ends[3], ends[4], dtype=np.int64)
        self.connectome = Connectome.from_synapses(
            n, label="map",
            pre=np.concatenate([held, sheet, sheet]),
            post=np.concatenate([sheet, np.repeat(where, d), np.tile(what, count)]),
            populations={name: list(range(run.start, run.stop))
                         for name, run in self._runs.items()})
        pools = np.full(n, -1, dtype=np.int64)
        pools[self._runs["reference"]] = np.arange(d)
        pools[self._runs["map"]] = np.tile(np.arange(d), count)
        bias = np.zeros(n)
        bias[self._runs["clock"]] = 1.0  # a moment a settle
        self._model = learning_neuron_model(dt=1.0)
        self.graph = Graph(
            self.connectome, self._model, efficacy=np.full(self.connectome.synapses, PASS),
            bias=bias, backend=backend, apical=Apical(
                np.isin(np.asarray(self.connectome.post), sheet), pools, fraction=self.level,
                floor=1e-6))
        # the same synapses without the compartment: a moment and a question by place use no
        # pool, and settle without one
        self._plain = Graph(self.connectome, self._model,
                            efficacy=np.full(self.connectome.synapses, PASS), bias=bias,
                            backend=backend)
        self._floor = 1e-3 * self.level  # activity under this is no thing
        self.trace = bool(trace)
        self.trajectory: dict[str, Any] = {}  # the latest settle by sweep, with ``trace``
        self.work = {"moments": 0, "questions": 0}
        self.reset()

    # -- the carried state

    def reset(self) -> None:
        """Empty the carried activity: the map then holds no thing."""
        self.potential = np.zeros((self.count, self.units))  # the sheet's carried potentials
        self.clock = np.zeros(self.count)  # the moments since a sense wrote each cell
        self.lost = np.zeros(self.units)  # the code units of things known with no place
        self.doubt = np.zeros(self.count)  # one where a cell holds a thing not seen to be there
        self._sensed: list[tuple[int, str]] = []
        self.reading = np.zeros(self.count + self.units)  # ``where`` and ``what``, settled
        # what the camera saw around each report of the moment before
        self._views: list[tuple[np.ndarray, np.ndarray]] = []

    def arrays(self) -> dict[str, np.ndarray]:
        """The carried state: the potentials of ``map``, ``clock``, ``lost`` and ``doubt``."""
        return {"potential": self.potential.copy(), "clock": self.clock.copy(),
                "lost": self.lost.copy(), "doubt": self.doubt.copy()}

    def parts(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """The map as a record (its settings) and arrays (its codes and carried state): what a
        map file and a brain that holds a map keep."""
        return {"format": FORMAT, **self.settings}, {"codes": self.codes, **self.arrays()}

    @classmethod
    def from_parts(cls, record: dict[str, Any], arrays: Any, **settings: Any) -> Map:
        """A map from ``parts``, with the state it carried."""
        told = dict(record)
        if told.pop("format", None) != FORMAT:
            raise ValueError(f"a map's record states {FORMAT!r}")
        told.update(settings)
        out = cls(tuple(told.pop("plane")), tuple(told.pop("cells")),
                  codes=np.asarray(arrays["codes"], dtype=np.float64),
                  **{k: v for k, v in told.items() if k != "units"})
        for name in ("potential", "clock", "lost", "doubt"):
            value = np.asarray(arrays[name], dtype=np.float64)
            if value.shape != getattr(out, name).shape:
                raise ValueError(f"the map's {name!r} does not fit its cells")
            setattr(out, name, value.copy())
        return out

    def save(self, path: str | Path, *, card: dict[str, Any] | None = None) -> Path:
        """Write the map as a file: its settings, codes and carried state as arrays and one JSON
        record, with a checksum of every array and a ``card`` (what it was raised on and its
        readings). No pickled object is written."""
        from ..modules import _checksum

        path = Path(path)
        record, arrays = self.parts()
        stored = {k.replace("/", "_"): np.asarray(v) for k, v in arrays.items()}
        record["checksums"] = {k: _checksum(v) for k, v in sorted(stored.items())}
        record["card"] = dict(card or {})
        from .. import __version__

        record["library"] = __version__
        written: dict[str, Any] = {"record": np.array(json.dumps(record)), **stored}
        with path.open("wb") as handle:
            np.savez_compressed(handle, **written)
        return path

    @classmethod
    def load(cls, path: str | Path, **settings: Any) -> Map:
        """A map from a file ``save`` wrote, with the state it carried. Nothing in the file is
        run: the archive is read without pickled objects, every array is checked against its
        checksum, and a file that fails is refused by what is wrong with it."""
        from ..modules import _checksum

        name = Path(path).name
        try:
            with np.load(Path(path), allow_pickle=False) as held:
                stored = {key: held[key] for key in held.files}
        except Exception:  # a text file, a pickled object, a broken archive: nothing of it is run
            raise ValueError(f"{name} is no map file: a map file is a NumPy archive of arrays "
                             "and one JSON record, with no pickled object") from None
        try:
            record = json.loads(str(stored.pop("record")))
        except (KeyError, json.JSONDecodeError):
            raise ValueError(f"{name} is no map file: it holds no record") from None
        sums = record.pop("checksums", None)
        record.pop("card", None)
        record.pop("library", None)
        if sums is not None:
            if set(sums) != set(stored):
                odd = sorted(set(sums) ^ set(stored))[0]
                raise ValueError(f"{name}: array {odd!r} is in the file or in the record alone")
            for key, value in stored.items():
                if _checksum(value) != sums[key]:
                    raise ValueError(f"{name}: array {key!r} differs from its checksum")
        arrays = stored
        try:
            return cls.from_parts(record, arrays, **settings)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} is no map file this version reads: {error}") from None

    @staticmethod
    def card_of(path: str | Path) -> dict[str, Any]:
        """A map file's card and the library version that wrote it, read without loading it."""
        with np.load(Path(path), allow_pickle=False) as held:
            record = json.loads(str(held["record"]))
        return {**dict(record.get("card") or {}), "library": record.get("library")}

    def observation(self) -> np.ndarray:
        """What a brain that holds the map reads of it: the activity of ``where`` (one value a
        cell) and of ``what`` (one a code unit) as the latest moment settled them, side by
        side, ``(cells + units,)``."""
        return self.reading.copy()

    # -- places as weights on cells

    def weights(self, point: np.ndarray) -> np.ndarray:
        """A place as weights on the four cells around it, ``(cells,)`` summing to one, whose
        mean is the place. A place beyond the outer cells' middles is read at them."""
        nx, ny = self.cells
        at = np.clip((np.asarray(point, dtype=np.float64).reshape(2) - self._low) / self.size
                     - 0.5, 0.0, [nx - 1.0, ny - 1.0])
        ix, iy = min(int(at[0]), nx - 2), min(int(at[1]), ny - 2)
        ax, ay = at[0] - ix, at[1] - iy
        out = np.zeros(self.count)
        out[ix * ny + iy] = (1 - ax) * (1 - ay)
        out[(ix + 1) * ny + iy] = ax * (1 - ay)
        out[ix * ny + iy + 1] = (1 - ax) * ay
        out[(ix + 1) * ny + iy + 1] = ax * ay
        return out

    def _pieces(self, active: np.ndarray) -> list[np.ndarray]:
        """The connected pieces of ``active`` cells (cells that touch, corners too), each as
        its cells' numbers."""
        nx, ny = self.cells
        left = set(np.flatnonzero(active).tolist())
        out: list[np.ndarray] = []
        while left:
            stack, piece = [left.pop()], []
            while stack:
                cell = stack.pop()
                piece.append(cell)
                i, j = divmod(cell, ny)
                for a in (i - 1, i, i + 1):
                    for b in (j - 1, j, j + 1):
                        other = a * ny + b
                        if 0 <= a < nx and 0 <= b < ny and other in left:
                            left.remove(other)
                            stack.append(other)
            out.append(np.array(sorted(piece), dtype=np.intp))
        return out

    def _around(self, piece: np.ndarray) -> np.ndarray:
        """A piece's cells and every cell that touches one of them."""
        nx, ny = self.cells
        i, j = np.divmod(piece, ny)
        a = (i[:, None] + np.array([-1, 0, 1])[None, :]).clip(0, nx - 1)
        b = (j[:, None] + np.array([-1, 0, 1])[None, :]).clip(0, ny - 1)
        return np.unique((a[:, :, None] * ny + b[:, None, :]).ravel()).astype(np.intp)

    def _activity(self) -> np.ndarray:
        """The sheet's carried activity above rest, ``(cells, units)``."""
        return np.asarray(np.maximum(np.tanh(self.potential / 2.0), 0.0))

    def _of_kind(self, kind: int, activity: np.ndarray) -> np.ndarray:
        """The activity of a kind's own units at every cell: the least of them, ``(cells,)``."""
        units = np.flatnonzero(self.codes[int(kind)])
        return np.asarray(activity[:, units].min(axis=1))

    def kind_of(self, code: np.ndarray) -> int | None:
        """The kind whose code a pattern over the units matches best, or None for no
        activity."""
        held = np.maximum(np.asarray(code, dtype=np.float64).reshape(self.units), 0.0)
        if not held.max() > 0:
            return None
        held = held / held.max()
        score = (self.codes @ held - (1.0 - self.codes) @ held) / self.codes.sum(axis=1)
        return int(np.argmax(score))

    # -- one moment

    def _settle(self, *, write: np.ndarray | None = None, carried: np.ndarray | None = None,
                clock: np.ndarray | None = None, fresh: np.ndarray | None = None,
                kept: np.ndarray | None = None, doubt: np.ndarray | None = None,
                sure: np.ndarray | None = None, cue: np.ndarray | None = None,
                asked: str = "moment") -> np.ndarray:
        """Settle the graph: every neuron's potential.

        Without ``cue`` the sheet's own neurons are driven by the potentials they carried
        (those ``carried`` keeps, flags ``(cells, units)``) and by ``write``; a unit that keeps
        nothing and is written nothing is held at zero by the settle's mask, and so is every
        cell ``kept`` (flags on cells) leaves out. With ``cue``, a kind's code, the carried
        potentials drive ``held`` in place of the sheet, the code drives the pools' references
        and the sheet settles on the pools the code opens. ``clock`` and ``doubt`` drive those
        regions in place of what they carried; ``fresh`` and ``sure`` (flags on cells) hold
        them at zero.

        The settle begins with every driven neuron at the potential its drive gives it and
        takes one sweep for ``where`` and ``what`` (two with a cue: the sheet, then the two
        regions that read it). With ``trace`` it begins with the sheet at rest and takes two
        sweeps, kept by sweep."""
        from ..brain import BrainState

        run = self._runs
        drive = np.zeros((1, self.neurons))
        held = self.potential if carried is None else self.potential * carried
        mask = np.ones((1, self.neurons))
        if cue is not None:
            drive[0, run["reference"]] = cue
            drive[0, run["held"]] = held.ravel()
        else:
            drive[0, run["map"]] = (held if write is None else held + write).ravel()
            if carried is not None:
                alive = carried if write is None else carried | (write != 0)
                mask[0, run["map"]] = alive.ravel()
        if kept is not None:
            mask[0, run["map"]] *= np.repeat(kept, self.units)
        drive[0, run["clock"]] = self.clock if clock is None else clock
        drive[0, run["lost"]] = self.lost
        drive[0, run["doubt"]] = self.doubt if doubt is None else doubt
        if fresh is not None:
            mask[0, run["clock"]] = ~fresh
        if sure is not None:
            mask[0, run["doubt"]] = ~sure
        begun = drive.copy()
        begun[0, run["clock"]] += 1.0
        sweeps = 2 if cue is not None else 1
        if self.trace or cue is not None:
            begun[0, run["map"]] = 0.0  # the sheet at rest: the settle is watched from there
            sweeps = 2
        begun *= mask
        state = BrainState(v=begun, activation=self._model.activation(begun),
                           adaptation=np.zeros_like(begun), steps=0)
        first = np.maximum(np.asarray(state.activation)[0], 0.0)
        graph = self.graph if cue is not None else self._plain
        state = graph.settle_batch(drive, steps=sweeps, tolerance=None, mask=mask, state=state,
                                   trajectory=self.trace)
        if self.trace:  # the settle by sweep, for a reader who watches it
            rows = np.maximum(np.concatenate(
                [first[None], np.asarray(state.trajectory)[:, 0]], axis=0), 0.0)
            nx, ny = self.cells
            self.trajectory = {
                "asked": asked, "sweeps": sweeps,
                "map": rows[:, run["map"]].reshape(sweeps + 1, nx, ny, self.units),
                "held": rows[:, run["held"]].reshape(sweeps + 1, nx, ny, self.units),
                "where": rows[:, run["where"]].reshape(sweeps + 1, nx, ny),
                "what": rows[:, run["what"]].copy(),
                "reference": rows[:, run["reference"]].copy(),
                "clock": rows[:, run["clock"]].reshape(sweeps + 1, nx, ny),
                "doubt": rows[:, run["doubt"]].reshape(sweeps + 1, nx, ny),
                "lost": rows[:, run["lost"]].copy()}
        return np.asarray(np.asarray(state.v, dtype=np.float64)[0])

    def moment(self, seen: Any = (), heard: Any = (), felt: Any = (), lit: Any = None) -> None:
        """One moment. ``seen``, ``heard`` and ``felt`` hold this moment's reports by sense,
        each a kind and a place ``(kind, (x, y))``: what the camera sees, what a sound names
        and what the body's own readings name. ``lit`` says which cells the camera sees: flags
        ``(cells,)`` or a function of ``(points, 2)`` places; None where nothing says. A thing
        is seen empty where the camera sees its held place (the mean of its piece) and no
        report claims it: a function judges the place itself, flags the cell that holds it. A
        place a sense wrote the moment before is a report, as far off as the camera's error: it
        is seen empty where the camera sees around it, within ``margin``, everything it saw
        there the moment before, and something. A hand last seen at a shadow's edge may have
        been reported just past it, and the camera that did not move sees it empty; a camera
        turning away from a thing it reported no longer sees some of what it saw around it, and
        the thing stays. Any other held place is seen empty where the camera sees all of it
        within ``margin``: the camera's error never decides it.

        A report writes its place on its kind's units, in place of what its cells carried, and
        the pieces of its kind's activity within ``step`` of it are that thing: they are
        cleared. With none there, the nearest piece of the kind within ``stride`` is the thing,
        moved unseen, and is cleared; with none, the report is a new thing. The pieces of the
        kind that lie on cells of doubt are cleared as well: a thing of the kind is found. A
        cell in sight loses what it carried. A piece seen empty with no report of its kind is
        treated as ``empty`` says."""
        reports = [(int(kind), np.asarray(place, dtype=np.float64).reshape(2), sense)
                   for sense, given in (("seen", seen), ("heard", heard), ("felt", felt))
                   for kind, place in given]
        sight = np.zeros(self.count, dtype=bool)
        if callable(lit):
            sight = np.asarray(lit(self.centers), dtype=bool).reshape(self.count)
        elif lit is not None:
            sight = np.asarray(lit, dtype=bool).reshape(self.count)
        looks = self._looks(lit, sight)
        before = self.potential.copy()  # the map as the moment before left it
        sensed = self.clock < 0.5  # the cells a sense wrote the moment before
        activity = self._activity()
        carried = np.ones((self.count, self.units), dtype=bool)
        write = np.zeros((self.count, self.units))
        clock = self.clock.copy()
        fresh = np.zeros(self.count, dtype=bool)
        doubt = self.doubt.copy()
        lost = self.lost.copy()
        claimed = np.zeros((self.count, self.units), dtype=bool)
        for kind, place, _ in reports:
            units = np.flatnonzero(self.codes[kind])
            apart = np.linalg.norm(self.centers - place, axis=1)
            pieces = self._pieces(np.asarray(self._of_kind(kind, activity) > self._floor))
            near = [p for p in pieces if apart[p].min() <= self.step]
            if not near:  # none of its kind here: the nearest piece within a stride moved unseen
                within = [p for p in pieces if apart[p].min() <= self.stride]
                near = [min(within, key=lambda p: float(apart[p].min()))] if within else []
            # a thing of the kind is sensed: the doubt about where one of it went is over
            near += [p for p in pieces if self.doubt[p].mean() > 0.5]
            for piece in near:  # its cells, and around them those its kind alone holds
                around = self._around(piece)
                alone = around[(activity[around][:, units] > 0).any(axis=1)
                               & ~np.isin(around, self._other_kinds(kind, activity))]
                claimed[np.ix_(np.union1d(piece, alone), units)] = True
            weights = self.weights(place)
            cells = weights > 0
            claimed[np.ix_(np.flatnonzero(cells), units)] = True  # the report takes its cells
            write[:, units] += (self.level * weights / np.linalg.norm(weights))[:, None]
            fresh |= cells
            lost[units] = 0.0  # a thing of the kind with no place is found
        carried &= ~claimed
        # a thing whose held place the camera sees, with no report that claims it, is not there
        empty = np.zeros(self.count, dtype=bool)
        for kind in self._present(activity):
            units = np.flatnonzero(self.codes[kind])
            own = self._of_kind(kind, activity) * ~claimed[:, units].any(axis=1)
            for piece in self._pieces(np.asarray(own > self._floor)):
                at = (own[piece] @ self.centers[piece]) / own[piece].sum()
                # a place a sense wrote the moment before is a report, as far off as the
                # camera's error: it is seen empty where the camera sees around it, within
                # ``margin``, all it saw there when it reported it (where the field moved off
                # it, the thing stays); any other held place where the camera sees all of it
                # within ``margin``
                if self.doubt[piece].mean() > 0.5 and self.empty == "near":
                    # a thing in doubt was placed out of sight once; sight only takes away the
                    # cells it sees, and with none left the thing is not known where
                    seen_here = piece[sight[piece]]
                    if len(seen_here):
                        carried[np.ix_(seen_here, units)] = False
                        empty[seen_here] = True
                        if len(seen_here) == len(piece) and not (
                                own[~sight] > self._floor).any():
                            lost[units] = 1.0
                    continue
                views = [(p, f) for p, f in self._views if np.linalg.norm(p - at) <= self.step]
                if sensed[piece].any() and views:
                    place, then = min(views, key=lambda v: float(np.linalg.norm(v[0] - at)))
                    now = looks(place, self.margin)
                    seen_empty = bool(now.any()) and not (then & ~now).any()
                else:  # a held place: the camera sees all of it, as far as its own error
                    seen_empty = bool(looks(at, self.margin).all())
                seen_empty = seen_empty and self.empty != "keep"
                if not seen_empty:
                    continue  # the thing stays where the map held it
                carried[np.ix_(piece, units)] = False
                empty[piece] = True
                if self.empty == "near":
                    apart = np.linalg.norm(self.centers - at, axis=1)
                    wide = np.exp(-0.5 * (apart / self.reach) ** 2) * (
                        ~sight & (apart <= 2.0 * self.reach))
                    if wide.any():  # onto the cells out of sight around where it was held
                        length = float(np.linalg.norm(before[piece][:, units].max(axis=1)))
                        write[:, units] += (length * wide / np.linalg.norm(wide))[:, None]
                        clock = np.where(wide > 0, float(self.clock[piece].max()), clock)
                        doubt = np.where(wide > 0, 1.0, doubt)
                        continue
                lost[units] = 1.0  # known by its kind, with no place
        self.lost = lost
        # a cell a sense writes, or one seen empty, holds no doubt
        sure = fresh | claimed.any(axis=1) | empty | (sight & ~(activity > self._floor).any(axis=1))
        v = self._settle(write=write if write.any() else None, carried=carried, clock=clock,
                         fresh=fresh, doubt=doubt, sure=sure)
        run = self._runs
        self.potential = v[run["map"]].reshape(self.count, self.units).copy()
        self.clock = v[run["clock"]].copy()
        self.lost = v[run["lost"]].copy()
        self.doubt = v[run["doubt"]].copy()
        self.reading = np.maximum(np.tanh(np.concatenate([v[run["where"]], v[run["what"]]])
                                          / 2.0), 0.0)
        self._sensed = [(kind, sense) for kind, _, sense in reports]
        # what the camera saw around each report, for the next moment's judgment
        self._views = [(place, looks(place, self.margin)) for _, place, _ in reports]
        self.work["moments"] += 1

    def _looks(self, lit: Any, sight: np.ndarray) -> Any:
        """What the camera sees of a place: whether it sees the place, or with a distance which
        of the 81 places on a grid of that half-width around it it sees. ``lit`` judges a
        place where it is a function of places; with flags on cells, the flag of the cell that
        holds it; nothing is seen where nothing says. Flags judge a place by its
        cell's middle, so a thing just outside the field whose cell's middle is inside it reads
        as seen: give a function where the field's edge crosses cells."""
        nx, ny = self.cells
        grid = np.linspace(-1.0, 1.0, 9)
        around = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)

        def seen(at: np.ndarray, within: float = 0.0) -> np.ndarray:
            points = np.asarray(at, dtype=np.float64)[None] + (
                within * around if within else np.zeros((1, 2)))
            if callable(lit):
                return np.asarray(lit(points), dtype=bool).reshape(len(points))
            if lit is None:
                return np.zeros(len(points), dtype=bool)
            i = np.clip(np.floor((points - self._low) / self.size).astype(np.intp), 0,
                        [nx - 1, ny - 1])
            return np.asarray(sight[i[:, 0] * ny + i[:, 1]])

        return seen

    def _other_kinds(self, kind: int, activity: np.ndarray) -> np.ndarray:
        """The cells where a kind other than ``kind`` holds a thing."""
        others = np.zeros(self.count, dtype=bool)
        for other in self._present(activity):
            if other != kind:
                others |= self._of_kind(other, activity) > self._floor
        return np.flatnonzero(others)

    def _present(self, activity: np.ndarray) -> list[int]:
        """The kinds whose every code unit holds activity somewhere."""
        held = np.asarray((activity > self._floor).any(axis=0))
        return [kind for kind in range(len(self.codes)) if held[self.codes[kind] > 0].all()]

    # -- questions, answered by settling

    def where_is(self, kind: int) -> np.ndarray:
        """Where the things of a kind are, ``(things, 2)``: the kind's code drives the pools'
        references, so the sheet settles on that kind's units alone. For a code of one unit
        ``where`` then holds the kind's places. For a code of several units a place is a cell
        where every one of them is active, and the least of them is read there. Each connected
        piece is one place, the mean of its activity."""
        code = self.codes[int(kind)]
        v = self._settle(cue=code, asked="where_is")
        self.work["questions"] += 1
        if code.sum() == 1:
            found = np.maximum(np.tanh(v[self._runs["where"]] / 2.0), 0.0)
        else:
            settled = v[self._runs["map"]].reshape(self.count, self.units)
            found = self._of_kind(int(kind), np.maximum(np.tanh(settled / 2.0), 0.0))
        return np.array([(found[p] @ self.centers[p]) / found[p].sum()
                         for p in self._pieces(found > self._floor)]).reshape(-1, 2)

    def what_is_at(self, place: np.ndarray, within: float | None = None) -> int | None:
        """What is at a place: the settle keeps the sheet's cells ``within`` metres of it (a
        cell and a half unless given) and holds the others at zero, ``what`` settles on the
        code there, and the kind whose code it matches is the answer; None where those cells
        hold no thing."""
        reach = 1.5 * float(self.size.max()) if within is None else float(within)
        near = np.linalg.norm(self.centers - np.asarray(place, dtype=np.float64), axis=1) <= reach
        v = self._settle(kept=near, asked="what_is_at")
        self.work["questions"] += 1
        code = np.maximum(np.tanh(v[self._runs["what"]] / 2.0), 0.0)
        return self.kind_of(code) if code.max() > self._floor else None

    # -- the inventory, read off the map

    def things(self) -> list[dict[str, Any]]:
        """Every thing the map holds, read off its carried activity: ``kind``, ``place`` (the
        mean of its activity), ``cells`` (its cells, numbers into ``centers``), ``pieces``
        (the cells by connected piece), ``since`` (the moments since a sense wrote it) and
        ``state``: ``"seen"``, ``"heard"`` or ``"felt"`` at the moment a sense wrote it,
        ``"held"`` at its last known place, ``"near"`` on the cells out of sight around where it
        was seen to be missing. A thing in doubt that lies in several pieces, sensed last at one
        moment, is one entry with every piece. A kind known with no place is listed with
        ``place`` None and ``"not known where"``."""
        activity = self._activity()
        out: list[dict[str, Any]] = []
        for kind in range(len(self.codes)):
            own = self._of_kind(kind, activity)
            groups: dict[int, list[np.ndarray]] = {}
            for piece in self._pieces(np.asarray(own > self._floor)):
                since = int(round(float(self.clock[piece[int(np.argmax(own[piece]))]])))
                groups.setdefault(since, []).append(piece)
            for since, pieces in groups.items():
                cells = np.concatenate(pieces)
                doubt = bool(self.doubt[cells].mean() > 0.5)
                # the pieces a thing in doubt left, sensed last at one moment, are one thing;
                # things at rest are a thing a piece
                for group in ([pieces] if doubt else [[p] for p in pieces]):
                    held = np.concatenate(group)
                    state = "near" if doubt else "held"
                    if since == 0:
                        state = next((sense for k, sense in self._sensed if k == kind), "seen")
                    entry = {"kind": kind, "place": (own[held] @ self.centers[held])
                             / own[held].sum(), "cells": held, "pieces": group, "since": since,
                             "state": state}
                    out.append(entry)
            if self.lost[np.flatnonzero(self.codes[kind])].min() > 0.5 and not (
                    own > self._floor).any():
                out.append({"kind": kind, "place": None, "cells": np.zeros(0, dtype=np.intp),
                            "pieces": [], "since": None, "state": "not known where"})
        return out

    def area(self, kind: int, rate: float = 0.9) -> np.ndarray:
        """The smallest set of cells that holds ``rate`` of a kind's activity, as flags
        ``(cells,)``: where the things of that kind are, at that rate. No cell for a kind the
        map does not hold."""
        own = self._of_kind(int(kind), self._activity())
        out = np.zeros(self.count, dtype=bool)
        if not own.sum() > 0:
            return out
        order = np.argsort(-own)
        need = int(np.searchsorted(np.cumsum(own[order]), rate * own.sum())) + 1
        out[order[:need]] = True
        return out

    def sheet(self, kind: int | None = None) -> np.ndarray:
        """The map's activity as a picture ``(cells x, cells y)``: of one kind, or of every
        unit summed."""
        activity = self._activity()
        held = activity.sum(axis=1) if kind is None else self._of_kind(int(kind), activity)
        return np.asarray(held.reshape(self.cells))


class Sense:
    """A sense's projection onto a map's cells: a reading that names a place, settled into it.

    The reading (a sound's direction and distance, the arm's joint readings) lights a sheet of
    tuned inputs, one unit a combination of the readings' tuned values, and every unit projects
    onto ``cells``, one neuron a cell of the map. The settled activity of ``cells`` is the place
    the reading names, read as the mean within a cell and a half of its peak. The synapses
    start at zero and are taught by the library's free and nudged phases with a quadratic
    nudge, at the moments a thing is heard or felt while the camera sees it: the target is the
    seen place as weights on its four cells. ``Map.moment(heard=...)`` and ``felt=`` take the
    place ``named`` gives.

    ``ranges`` holds each reading's ``(low, high)`` and ``per`` the tuned inputs of each;
    ``rate`` is the share of a lesson's error one lesson closes."""

    def __init__(self, held: Map, ranges: Any, per: Any, *, rate: float = 0.3,
                 efficacy: np.ndarray | None = None, backend: Any = "cpu"):
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model
        from ..tuning import PopulationCode

        self.map, self.rate = held, float(rate)
        self.ranges = [(float(a), float(b)) for a, b in ranges]
        self.per = [int(n) for n in per]
        if len(self.ranges) != len(self.per) or not self.per or min(self.per) < 2:
            raise ValueError("a sense takes a (low, high) range and 2 tuned inputs or more a "
                             "reading")
        self._codes = [PopulationCode([r], per=n)
                       for r, n in zip(self.ranges, self.per, strict=True)]
        self.units = units = int(np.prod(self.per))
        count = held.count
        self.neurons = units + count
        self.connectome = Connectome.from_synapses(
            self.neurons, label="sense", pre=np.repeat(np.arange(units), count),
            post=np.tile(units + np.arange(count), units),
            populations={"sheet": list(range(units)), "cells": list(range(units, self.neurons))})
        self._model = learning_neuron_model(dt=1.0)
        start = np.zeros(self.connectome.synapses) if efficacy is None else np.asarray(
            efficacy, dtype=np.float64).reshape(self.connectome.synapses)
        self.graph = Graph(self.connectome, self._model, efficacy=start, backend=backend)
        self._learner: Any = None
        self.lessons = 0

    def code(self, readings: np.ndarray) -> np.ndarray:
        """``(rows, readings)`` readings as the sheet's drive ``(rows, units)``: each unit the
        product of its tuned inputs' responses."""
        given = np.asarray(readings, dtype=np.float64).reshape(-1, len(self.per))
        out = np.ones((len(given), 1))
        for k, tuned in enumerate(self._codes):
            out = (out[:, :, None] * tuned(given[:, k : k + 1])[:, None, :]).reshape(len(given), -1)
        return out

    def named(self, reading: np.ndarray) -> np.ndarray | None:
        """The place a reading names: ``cells`` settled on the sheet, read as the mean within
        a cell and a half of the most active cell; None where no cell is above rest."""
        drive = np.zeros((1, self.neurons))
        drive[0, : self.units] = self.code(reading)[0]
        state = self._learner.free(drive) if self._learner is not None else (
            self.graph.settle_batch(drive, steps=2, tolerance=None))
        found = np.maximum(np.asarray(state.activation)[0, self.units :], 0.0)
        if not found.max() > 1e-6:
            return None
        centers = self.map.centers
        near = np.linalg.norm(centers - centers[int(found.argmax())], axis=1) <= (
            1.5 * float(self.map.size.max()))
        return np.asarray((found * near) @ centers / (found * near).sum())

    def teach(self, readings: np.ndarray, places: np.ndarray) -> float:
        """Lessons: ``(rows, readings)`` readings taken while the camera saw the thing at
        ``(rows, 2)`` places. Gives the mean distance between the free answer and the target,
        as a share of the target."""
        import dataclasses

        from ..learning import Learner, LearnerConfig

        code = self.code(readings)
        at = np.asarray(places, dtype=np.float64).reshape(len(code), 2)
        drive = np.zeros((len(code), self.neurons))
        drive[:, : self.units] = code
        target = np.zeros_like(drive)
        for row, place in enumerate(at):
            weights = self.map.weights(place)
            target[row, self.units :] = 0.5 * weights / np.linalg.norm(weights)
        if self._learner is None:
            self._learner = Learner(
                self.graph, list(range(self.units, self.neurons)),
                LearnerConfig(nudge="quadratic", beta=0.1, eta=1.0, eta_bias=0.0, free_steps=2,
                              nudged_steps=4, tolerance=None),
                reciprocal=False, plastic_neurons=np.zeros(self.neurons, dtype=bool))
        learner = self._learner
        lit = np.tanh(code / 2.0)
        # a synapse steps by its batch-mean contrast. The step that closes ``rate`` of the
        # batch's error: the rows' summed activity over the sheet squared, a row
        shared = float((lit.sum(axis=0) ** 2).sum()) / len(code)
        learner.config = dataclasses.replace(
            learner.config, eta=self.rate / (0.25 * max(shared, 1e-12)))
        free = learner.free(drive)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        learner.update(free, plus, minus)
        self.graph = learner.brain
        self.lessons += len(code)
        answer = np.asarray(free.activation)[:, self.units :]
        wanted = target[:, self.units :]
        return float(np.abs(answer - wanted).sum() / np.abs(wanted).sum())

    def arrays(self) -> dict[str, np.ndarray]:
        """What a file keeps of a sense: its taught efficacies."""
        return {"efficacy": np.asarray(self.graph.efficacy, dtype=np.float32).copy()}


__all__ = ["FORMAT", "PASS", "Map", "Sense", "sparse_codes"]
