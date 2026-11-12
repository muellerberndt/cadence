"""A place held while out of view: the regions of a place brain and the space that holds places.

A thing's place is a population code, a peak over tuned inputs on each axis of a plane, held from
moment to moment without decay. It changes only when information about it arrives:

- **Seen.** A camera's report of the thing writes the place.
- **Moved while unseen.** The brain settles on what it holds (the place, the copy of the arm's
  command for the arm's own hand, the thing's last seen movement, the moments since it was seen)
  and answers how far the place moves in this step, and the place moves by that answer. The
  lessons are the brain's own: whenever the thing is seen, the place predicted from the moment
  before is taught toward the place seen. No lesson holds a position the camera did not give.
- **Heard or felt.** A cue from another sense (a direction and a distance from an ear) is settled
  into a place, taught at the moments the thing is heard and seen at once. The arm's own joint
  readings, where they are given, are settled into the place of its end the same way.
- **Nothing.** With no report, no command, no movement and no cue, the place stays as it is.

The brain also answers how far off the place may be, taught by how far off its predictions were
whenever the thing came back into view; ``Space.where`` gives each place with the radius of the
area that holds the thing at ``RATE``. A map of where things were seen says where a thing that is
out of view cannot be: a place does not move into a cell where things are seen.

``Interface`` is the record of the regions and their sizes, ``anatomy`` the connectome of those
regions, ``compose`` a brain of them, and ``Space`` holds the places of several things, one row
each.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..tuning import PopulationCode

RATE = 0.90  # the rate a place's area is stated at
AREA = float(np.sqrt(-2.0 * np.log(1.0 - RATE)))  # that area's radius, in standard deviations
INPUTS = ("place", "moved", "motion", "unseen", "heard", "felt")
OUTPUTS = ("shift", "spread", "found")
# The movement a step that six reports in a row give, by least squares: their slope.
SLOPE = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]) / 35.0
KEY = 64  # the width of a record's key: a code of what a thing looks like


class Tuned:
    """One output slot of ``units`` values evenly spaced over ``(low, high)``, or over their
    logarithms with ``log``. ``targets`` makes a lesson of a value, ``mean`` and ``quantile`` read
    a settled slot."""

    def __init__(self, low: float, high: float, units: int, log: bool = False):
        if units < 2 or not high > low or (log and low <= 0):
            raise ValueError("a tuned slot takes 2 or more units over a range with low below high")
        self.low, self.high, self.units, self.log = float(low), float(high), int(units), log
        ends = np.log([low, high]) if log else np.array([low, high], dtype=np.float64)
        self.centers = np.linspace(ends[0], ends[1], units)
        self.spacing = float(self.centers[1] - self.centers[0])

    def targets(self, values: np.ndarray, width: float = 0.0) -> np.ndarray:
        """``(rows,)`` values as ``(rows, units)`` lesson targets: each value's weight on the two
        units it lies between, so the slot's mean is the value. A value past the range is read at
        the nearer end. With ``width`` the target is a peak that many units wide around the
        value: a slot taught such peaks adds the peaks of its inputs into a peak between them,
        where a slot taught two units picks one of them."""
        v = np.clip(np.asarray(values, dtype=np.float64), self.low, self.high)
        at = ((np.log(v) if self.log else v) - self.centers[0]) / self.spacing
        if width > 0:
            peak = np.exp(-0.5 * ((np.arange(self.units)[None, :] - at[:, None]) / width) ** 2)
            return np.asarray(peak / peak.sum(axis=1, keepdims=True))
        lower = np.clip(np.floor(at).astype(np.intp), 0, self.units - 2)
        share = np.clip(at - lower, 0.0, 1.0)
        out = np.zeros((len(v), self.units))
        rows = np.arange(len(v), dtype=np.intp)
        out[rows, lower] = 1.0 - share
        out[rows, lower + 1] = share
        return out

    def beliefs(self, activity: np.ndarray, temperature: float) -> np.ndarray:
        """The softmax a lesson pulls on: ``(rows, units)`` shares that sum to one."""
        s = np.asarray(activity, dtype=np.float64) / temperature
        p = np.exp(s - s.max(axis=-1, keepdims=True))
        return np.asarray(p / p.sum(axis=-1, keepdims=True))

    def mean(self, activity: np.ndarray, temperature: float) -> np.ndarray:
        """The value a settled slot names: its mean under the softmax."""
        value = self.beliefs(activity, temperature) @ self.centers
        return np.asarray(np.exp(value) if self.log else value)

    def peak(self, activity: np.ndarray, temperature: float, reach: int = 2) -> np.ndarray:
        """The value at a settled slot's highest peak: the mean of its belief within ``reach``
        units of its most believed unit. Where a slot believes in two values, ``mean`` reads
        between them and ``peak`` reads the likelier one."""
        belief = self.beliefs(activity, temperature)
        top = belief.argmax(axis=-1)
        near = np.abs(np.arange(self.units)[None, :] - top[:, None]) <= reach
        local = belief * near
        value = (local @ self.centers) / local.sum(axis=-1)
        return np.asarray(np.exp(value) if self.log else value)

    def quantile(self, activity: np.ndarray, temperature: float, rate: float) -> np.ndarray:
        """The value a share ``rate`` of a settled slot's belief lies under."""
        total = np.cumsum(self.beliefs(activity, temperature), axis=-1)
        above = np.minimum((total < rate).sum(axis=-1), self.units - 1)
        rows = np.arange(len(total), dtype=np.intp)
        below = np.where(above > 0, total[rows, np.maximum(above - 1, 0)], 0.0)
        share = (rate - below) / np.maximum(total[rows, above] - below, 1e-12)
        value = self.centers[0] + (above - 1 + np.clip(share, 0.0, 1.0)) * self.spacing
        value = np.clip(value, self.centers[0], self.centers[-1])
        return np.asarray(np.exp(value) if self.log else value)


@dataclass(frozen=True)
class Interface:
    """The regions of a place brain and what their units stand for.

    Inputs, which their own senses drive: ``place`` (the held place, a peak on each axis),
    ``moved`` (the copy of the arm's command, a peak a joint; dark for a thing that is no arm's
    end and for a command of zero), ``motion`` (the thing's last seen movement, a peak an axis,
    and how many reports it was seen over; dark until three reports in a row), ``unseen`` (the
    moments since it was seen), ``heard`` (a cue's direction and distance, one unit a pair of
    them; dark while silent) and ``felt`` (the arm's joint readings, one unit a posture; dark
    for a thing that is no arm's end and where no readings are given). ``hidden`` reads them
    all. Outputs, one softmax each: ``shift`` (how far the place moves in this step, an axis a
    slot), ``spread`` (how far off a place predicted here is found to be) and ``found`` (the
    place a cue or the joint readings name, an axis a slot)."""

    plane: tuple[float, float, float, float]  # x low, x high, y low, y high, in metres
    joints: int = 2
    place: int = 16  # tuned inputs an axis of the held place
    moved: int = 9  # tuned inputs a joint's command
    command: float = 0.1  # the largest command a step, in radians
    memory: int = 60  # the most reports in a row a seen movement is fitted over
    check: int = 12  # the latest reports whose own movement is held against the whole fit's
    counted: int = 6  # tuned inputs of how many reports the movement was seen over
    motion: int = 25  # tuned inputs an axis of the seen movement
    speed: float = 0.024  # the fastest seen movement a step, in metres
    unseen: int = 8  # tuned inputs of the moments since seen
    patience: int = 200  # the moments since seen that the unseen inputs tell apart
    heard: tuple[int, int] = (20, 16)  # tuned directions and distances of a cue: a sheet
    ear: tuple[float, float] = (0.95, 1.35)
    hearing: float = 3.0  # the farthest cue, in metres
    felt: int = 0  # tuned readings a joint; the region holds one unit a posture. Where 0:
    # the most, up to 16, that keep the region at 600 units or under
    limits: tuple[tuple[float, float], ...] = ()  # each joint's range; a whole turn if omitted
    hidden: int = 128
    shift: int = 32  # tuned outputs an axis of a step's movement
    reach: float = 0.06  # the largest movement a step, in metres
    spread: int = 16  # tuned outputs of how far off, from 2 mm to 2 m
    found: int = 24  # tuned outputs an axis of a place another sense names
    feedback: bool = False  # the outputs also drive the hidden region

    def __post_init__(self) -> None:
        x_low, x_high, y_low, y_high = self.plane
        if not (x_high > x_low and y_high > y_low):
            raise ValueError("a plane is (x low, x high, y low, y high) with low below high")
        if not self.felt:
            most = max(f for f in range(2, 17) if f**self.joints <= 600 or f == 2)
            object.__setattr__(self, "felt", most)
        counts = (self.place, self.moved, self.motion, self.unseen, self.hidden, self.shift,
                  self.spread, self.found, self.felt, *self.heard)
        if not self.limits:
            object.__setattr__(self, "limits", ((-np.pi, np.pi),) * self.joints)
        if len(self.limits) != self.joints:
            raise ValueError("limits holds one (low, high) range a joint")
        object.__setattr__(self, "limits", tuple((float(a), float(b)) for a, b in self.limits))
        if min(counts) < 2 or self.joints < 1 or self.counted < 2:
            raise ValueError("every region of an interface holds 2 or more units")
        if not 6 <= self.check < self.memory:
            raise ValueError("check is 6 reports or more and under memory")
        if self.felt**self.joints > 4096:
            raise ValueError("felt ** joints is 4,096 postures at most: fewer readings a joint")

    @property
    def regions(self) -> dict[str, int]:
        """Every region's size, the inputs first, in the order a brain's observation holds
        them, then ``hidden``, then the outputs in the order of its motor slots."""
        return {
            "place": 2 * self.place, "moved": self.joints * self.moved,
            "motion": 2 * self.motion + self.counted, "unseen": self.unseen,
            "heard": self.heard[0] * self.heard[1], "felt": self.felt**self.joints,
            "hidden": self.hidden,
            "shift": 2 * self.shift, "spread": self.spread, "found": 2 * self.found,
        }

    @property
    def inputs(self) -> int:
        return sum(self.regions[name] for name in INPUTS)

    @property
    def outputs(self) -> int:
        return sum(self.regions[name] for name in OUTPUTS)

    @property
    def slots(self) -> tuple[int, ...]:
        return (self.shift, self.shift, self.spread, self.found, self.found)

    def to_dict(self) -> dict[str, Any]:
        return {k: [list(x) if isinstance(x, tuple) else x for x in v]
                if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Interface:
        held: dict[str, Any] = {
            k: tuple(tuple(x) if isinstance(x, list) else x for x in v)
            if isinstance(v, list) else v for k, v in data.items()}
        return cls(**held)


def anatomy(interface: Interface, seed: int = 0) -> Any:
    """The connectome of a place brain: the interface's regions by name, every input read by
    ``hidden`` through one-way synapses, every output reading ``hidden`` (and driving it too
    with ``feedback``), and one-way synapses that start at zero from ``motion`` and ``moved`` to
    ``shift``, from ``unseen`` and ``motion`` to ``spread`` and from ``heard`` and ``felt`` to
    ``found``. ``sensory`` names the inputs, ``association`` the hidden region and ``motor`` the
    outputs. Without ``feedback`` every synapse runs from the inputs toward the outputs, so a
    settle has one answer whatever it began from."""
    from ..connectome import Connectome
    from ..genome import Genome, Projection, develop
    from ..regions import Region

    sizes = interface.regions
    regions = tuple(Region(name, size) for name, size in sizes.items())
    projections = [Projection(name, "hidden", scale=1.2, reciprocal=False) for name in INPUTS]
    projections += [Projection("hidden", name, reciprocal=interface.feedback)
                    for name in OUTPUTS]
    projections += [Projection(pre, post, scale=0.0, reciprocal=False)
                    for pre, post in (("motion", "shift"), ("moved", "shift"),
                                      ("unseen", "spread"), ("motion", "spread"),
                                      ("heard", "found"), ("felt", "found"))]
    grown: Connectome = develop(Genome(regions, tuple(projections), label="place"), seed=seed)
    held = grown.populations
    return grown.with_populations(
        sensory=[i for name in INPUTS for i in held[name]], association=held["hidden"],
        motor=[i for name in OUTPUTS for i in held[name]],
    )


def compose(interface: Interface, seed: int = 0, **settings: Any) -> Any:
    """A brain of the interface's regions: its observation is the inputs side by side and its
    motor slots are the outputs. ``settings`` are ``Brain.compose``'s; unless they say
    otherwise the brain is taught at the founder point and its slots are read at a softmax
    temperature of 0.05. At the founder's 0.2 a slot's units saturate before its peak is sharp,
    and a slot's mean then reads short of the value taught (a slope of .91 against 1.00)."""
    from ..generic import Brain

    settings.setdefault("episodic", False)
    settings.setdefault("teaching", "founder")
    settings.setdefault("temperature", 0.05)
    return Brain.compose(interface.inputs, interface.outputs,
                         connectome=anatomy(interface, seed), slots=interface.slots, seed=seed,
                         **settings)


class Space:
    """The places of several things, one row each, held between moments.

    ``begin`` takes the list of starting positions. ``advance`` takes one step: the reports of
    the rows in view, the arm's command and the cues, and gives every row's place. With
    ``learn`` it also teaches the brain from the reports, and at times it leaves a report unread
    for a span of steps (``practice``, the rate a span begins at a step in view; ``span``, its
    mean length), so that the places it predicts there are taught by the reports it did not
    read. ``where`` gives the places and the radius of each one's area at ``RATE``; ``state``
    gives what the regions hold.

    ``sight`` (off unless given) turns on the map of where things are seen, for a camera and a
    blind spot that stay where they are: a place that is out of view does not move into a cell
    whose share of reports among its visits is above ``sight``.

    A thing's seen movement is the slope of the line through its reports in a row, up to
    ``Interface.memory`` of them, by least squares. Where the latest ``Interface.check`` reports
    move otherwise than the whole line by three times their noise, or a report lies five times
    the camera's noise from where the movement carries the place, the movement has changed and
    the line begins anew.

    The movement a lesson teaches is the slope of six reports in a row. For an arm's own end
    they lie around the step taught. For a thing that moves by itself they are the six from the
    step taught on, which share no report with what the answer read.

    Reports carry no names. ``begin`` takes what each row looks like (``looks``), ``assign``
    gives each report to a row of its look, the nearer predicted places first, and ``hear``
    gives each cue to a row likewise. ``records`` (off unless given) keeps the places of things
    at rest in a ``cadence.LongTermMemory`` of that many records, keyed by what a thing looks
    like: a thing with a look of its own is filed once its place is a mean of four reports, and
    its record is revised as the mean sharpens and when it is seen elsewhere. The records stay
    over ``begin``, and ``find`` reads one.

    ``smooth`` (on unless refused): a thing in steady movement takes, in view, the place its
    line through six reports or more gives for the latest one, as a thing at rest takes the mean
    of its reports. Without it a moving thing in view takes the report itself.

    ``read`` says how a step's movement is read from the brain's ``shift``: ``"mean"``, the mean
    of its belief, or ``"peak"``, the value at its highest peak. A brain that has seen things
    stop and slow while unseen believes in both while a thing is out of view: the mean moves the
    place part of the way, the peak moves it as the likelier of the two. ``"seen"`` is the
    control a brain's answer is read against: a thing that moves by itself is moved by its last
    seen movement as it is, whatever the brain answers. It can be set at any step."""

    def __init__(self, interface: Interface, brain: Any = None, *, seed: int = 0,
                 practice: float = 0.04, span: int = 20, sight: float | None = None,
                 cells: int = 32, records: int = 0, smooth: bool = True, read: str = "mean",
                 **settings: Any):
        if read not in ("mean", "peak", "seen"):
            raise ValueError("read is 'mean', 'peak' or 'seen'")
        self.interface = interface
        self.brain = compose(interface, seed, **settings) if brain is None else brain
        self.practice, self.span, self.sight = practice, int(span), sight
        self.smooth, self.read = bool(smooth), read
        self.records: Any = None
        if records:
            from ..longterm import LongTermMemory

            self.records = LongTermMemory(width=KEY, value_width=3, capacity=int(records),
                                          cosine=True, fade=False)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 320]))
        x_low, x_high, y_low, y_high = interface.plane
        f = interface
        self._low = np.array([x_low, y_low])
        self._size = np.array([x_high - x_low, y_high - y_low])
        self._centers = np.stack([np.linspace(x_low, x_high, f.place),
                                  np.linspace(y_low, y_high, f.place)])
        self._spacing = self._size / (f.place - 1)
        self._moved = PopulationCode([(-f.command, f.command)] * f.joints, per=f.moved)
        self._motion = PopulationCode([(-f.speed, f.speed)] * 2, per=f.motion)
        self._unseen = PopulationCode([(0.0, float(np.log1p(f.patience)))], per=f.unseen)
        self._counted = PopulationCode([(float(np.log(3.0)), float(np.log(f.memory)))],
                                       per=f.counted)
        # a cue's direction is coded from the direction to the plane's middle, over the
        # directions and distances of the plane's corners from the ear
        corners = np.array([[x, y] for x in (x_low, x_high) for y in (y_low, y_high)])
        away = corners - np.array(f.ear)
        self._bearing = float(np.arctan2(*(corners.mean(0) - np.array(f.ear))[::-1]))
        turns = (np.arctan2(away[:, 1], away[:, 0]) - self._bearing + np.pi) % (2 * np.pi) - np.pi
        wide = float(min(np.abs(turns).max() + 0.1, np.pi))
        far = float(min(np.linalg.norm(away, axis=1).max() * 1.1, f.hearing))
        self._heard = (PopulationCode([(-wide, wide)], per=f.heard[0]),
                       PopulationCode([(0.0, far)], per=f.heard[1]))
        self._felt = [PopulationCode([limit], per=f.felt) for limit in f.limits]
        self.shift = Tuned(-f.reach, f.reach, f.shift)
        self.spread = Tuned(0.002, 2.0, f.spread, log=True)
        self.found = (Tuned(x_low, x_high, f.found), Tuned(y_low, y_high, f.found))
        populations = self.brain.connectome.populations
        self._index = {name: np.asarray(populations[name], dtype=np.intp)
                       for name in (*INPUTS, "hidden", *OUTPUTS)}
        self.cells = int(cells)
        self.seen_map = np.zeros((self.cells, self.cells))  # reports of things, by cell
        self.visit_map = np.zeros((self.cells, self.cells))  # reports and unseen places, by cell
        self.noise = 1e-4  # the camera's variance on an axis, learned from seen steps
        self.work = {"lesson_row_sweeps": 0.0, "act_sweeps_rows": 0.0, "moments": 0}
        self.rows = 0

    # -- a scene

    def begin(self, starts: np.ndarray, own: np.ndarray | bool = False,
              looks: np.ndarray | None = None) -> None:
        """New things at their starting positions, ``(rows, 2)``. ``own`` says which rows are an
        arm's own end, moved by the commands, and ``looks`` what each row looks like (one number
        a row; all alike where omitted). A row whose start is NaN begins where the records hold
        a thing of its look, or at the plane's middle."""
        f = self.interface
        self.place = np.asarray(starts, dtype=np.float64).reshape(-1, 2).copy()
        self.rows = rows = len(self.place)
        self.own = np.broadcast_to(np.asarray(own, dtype=bool), (rows,)).copy()
        self.looks = (np.zeros(rows, dtype=np.intp) if looks is None
                      else np.asarray(looks, dtype=np.intp).reshape(rows))
        alike = np.array([(self.looks == look).sum() for look in self.looks])
        self._single = (alike == 1) & ~self.own  # a look of its own: a record can hold it
        for row in np.flatnonzero(~np.isfinite(self.place).all(axis=1)):
            known = self.find(int(self.looks[row]))
            self.place[row] = self._low + self._size / 2 if known is None else known[0]
        self.radius = np.full(rows, AREA * np.sqrt(self.noise))
        self.since = np.zeros(rows, dtype=np.intp)
        self._trail = np.zeros((rows, f.memory, 2))  # the latest reports read, the last last
        self._count = np.zeros(rows, dtype=np.intp)  # reports in a row on the row's line
        self._movement = np.zeros((rows, 2))  # the seen movement a step, held while unseen
        self._known = np.zeros(rows, dtype=bool)  # a movement was seen
        self._fitted = np.zeros(rows, dtype=np.intp)  # the reports that movement was seen over
        self._line = self.place.copy()  # the place the line gives for the latest report
        self._practising = np.zeros(rows, dtype=np.intp)
        self._moving = np.zeros(rows, dtype=bool)  # a seen movement above its window's noise
        self._smoothed = np.zeros(rows, dtype=bool)  # the place is a mean of carried reports
        self._rest = np.ones(rows, dtype=np.intp)  # reports averaged into a place at rest
        self._named = np.zeros(rows, dtype=np.intp)  # cues averaged into a place heard at rest
        self._pending: deque[dict[str, np.ndarray]] = deque(maxlen=len(SLOPE))
        self._push(np.ones(rows, dtype=bool), self.place)
        self._last: dict[str, np.ndarray] = {}
        self._step: dict[str, np.ndarray] = {}
        self.brain.reset()

    def sense(self, commands: np.ndarray | None = None, cues: np.ndarray | None = None,
              joints: np.ndarray | None = None) -> np.ndarray:
        """What the brain's input regions hold for this step: ``(rows, inputs)``, the regions
        side by side. ``commands`` is the arm's command, ``(joints,)`` or ``(rows, joints)``;
        ``cues`` is ``(rows, 2)`` directions and distances, NaN for a silent row; ``joints`` is
        the arm's joint readings, shaped as the commands."""
        f, rows = self.interface, self.rows
        offset = (self.place[:, :, None] - self._centers[None]) / self._spacing[None, :, None]
        peak = np.exp(-0.5 * offset**2)
        peak[peak < 0.05] = 0.0
        asked = np.zeros((rows, f.joints)) if commands is None else np.broadcast_to(
            np.asarray(commands, dtype=np.float64), (rows, f.joints))
        moving = self.own[:, None] & (asked != 0).any(axis=1, keepdims=True)
        moved = self._moved(asked) * moving
        over = self._counted(np.log(np.clip(self._fitted, 3, f.memory))[:, None])
        motion = np.hstack([self._motion(self._movement), over]) * self._known[:, None]
        unseen = self._unseen(np.log1p(np.minimum(self.since, f.patience))[:, None])
        heard = np.zeros((rows, f.heard[0] * f.heard[1]))
        if cues is not None:
            cue = np.asarray(cues, dtype=np.float64).reshape(rows, 2)
            sounding = np.isfinite(cue).all(axis=1)
            turn = (np.nan_to_num(cue[:, 0]) - self._bearing + np.pi) % (2 * np.pi) - np.pi
            sheet = (self._heard[0](turn[:, None])[:, :, None]
                     * self._heard[1](np.nan_to_num(cue[:, 1:2]))[:, None, :])
            heard = sheet.reshape(rows, -1) * sounding[:, None]
        felt = np.zeros((rows, f.felt**f.joints))
        if joints is not None:
            read = np.broadcast_to(np.asarray(joints, dtype=np.float64), (rows, f.joints))
            sheet = np.ones((rows, 1))
            for k, code in enumerate(self._felt):  # one unit a posture: every joint's reading
                sheet = (sheet[:, :, None] * code(read[:, k : k + 1])[:, None, :]).reshape(
                    rows, -1)
            felt = sheet * self.own[:, None]
        return np.hstack([peak.reshape(rows, -1), moved, motion, unseen, heard, felt])

    def predict(self, commands: np.ndarray | None = None, cues: np.ndarray | None = None,
                joints: np.ndarray | None = None, *, learn: bool = False) -> np.ndarray:
        """The first half of a step: where every row is predicted to be, ``(rows, 2)``, before
        the camera's reports of the step are read. The brain settles on the rows that hold
        information: an arm's own end with a command or with joint readings, a thing with a
        seen movement, a thing heard. Every other row keeps its place, and with none the brain
        is not asked. With ``learn`` every row is settled, to be taught. A row that is heard
        or felt is predicted where ``found`` names; any other live row where it was, moved by
        ``shift``."""
        f, rows = self.interface, self.rows
        asked = np.zeros((rows, f.joints)) if commands is None else np.broadcast_to(
            np.asarray(commands, dtype=np.float64), (rows, f.joints))
        sounding = np.zeros(rows, dtype=bool) if cues is None else np.isfinite(
            np.asarray(cues, dtype=np.float64).reshape(rows, 2)).all(axis=1)
        felt = self.own & (joints is not None)
        sounding = sounding | felt  # another sense names the place
        quiet = np.where(self.own, ~(asked != 0).any(axis=1), ~self._moving)
        live = np.ones(rows, dtype=bool) if learn else (~quiet | sounding)
        x = self.sense(asked, cues, joints)
        predicted, off = self.place.copy(), self.radius.copy()
        motor: dict[str, np.ndarray] = {}
        if live.any():
            motor = self._settle(x, live)
            t = float(self.brain.learner.config.temperature)
            d, b = f.shift, f.found
            reading = self.shift.peak if self.read == "peak" else self.shift.mean
            shift = np.stack([reading(motor["shift"][:, :d], t),
                              reading(motor["shift"][:, d:], t)], axis=1)
            named = np.stack([self.found[0].mean(motor["found"][:, :b], t),
                              self.found[1].mean(motor["found"][:, b:], t)], axis=1)
            if self.read == "seen":  # the control: the seen movement itself
                shift = np.where(self.own[live, None], shift,
                                 (self._movement * self._known[:, None])[live])
            moved = np.where(sounding[live, None], named, self.place[live] + shift)
            predicted[live] = np.where((quiet & ~sounding)[live, None], self.place[live], moved)
            off[live] = self.spread.quantile(motor["spread"], t, RATE)
        self._step = {"x": x, "live": live, "quiet": quiet, "sounding": sounding,
                      "predicted": predicted, "off": off, **motor}
        return predicted.copy()

    def observe(self, seen: np.ndarray, reports: np.ndarray, *, learn: bool = False
                ) -> np.ndarray:
        """The second half of a step. ``seen`` says which rows the camera reports and
        ``reports`` holds their positions, ``(rows, 2)``. A report read writes its row's place:
        a thing at rest takes the mean of its reports, and any other thing the report itself.
        A row with no report read takes its predicted place, unless things are seen there (the
        map of where things were seen), in which case it stays. Gives every row's place."""
        rows, step = self.rows, self._step
        seen = np.asarray(seen, dtype=bool).reshape(rows)
        z = np.where(seen[:, None], np.asarray(reports, dtype=np.float64).reshape(rows, 2), 0.0)
        live, quiet, sounding = step["live"], step["quiet"], step["sounding"]
        predicted = step["predicted"]
        read = seen & ~(self._practising > 0)  # the rows whose report is read
        refused = ~read & live & ~quiet & ~sounding & self._in_sight(predicted)
        predicted = np.where(refused[:, None], self.place, predicted)
        if learn:
            self._teach(step, seen, z, predicted, read)
        deviation = np.sqrt(self.noise)
        near = np.linalg.norm(z - self.place, axis=1) < 4.0 * deviation * np.sqrt(
            1.0 + 1.0 / self._rest)
        resting = read & quiet & near
        self._rest = np.where(resting, np.minimum(self._rest + 1, 500), 1)
        mean = self.place + (z - self.place) / self._rest[:, None]
        seen_place = np.where(resting[:, None], mean, z)
        wide = np.sqrt(np.maximum((step["off"] / AREA) ** 2 - self.noise, 0.25 * self.noise))
        moved = live & ~quiet | sounding
        self.radius = np.where(read, AREA * deviation / np.sqrt(self._rest),
                               np.where(moved, AREA * wide, self.radius))
        elsewhere = read & quiet & ~near  # a thing at rest, seen elsewhere
        astray = np.linalg.norm(z - self.place - self._movement * self._known[:, None],
                                axis=1) > 5.0 * deviation
        # a row that comes back, or is seen away from where its movement carries it, begins
        # its line anew
        back = read & ((self.since > 0) | elsewhere | (~self.own & ~quiet & astray))
        self._count[back] = 0
        self._known[back & (self.since == 0)] = False
        self._moving[elsewhere] = False
        self._push(read, z)
        self._smoothed = np.zeros(rows, dtype=bool)
        if self.smooth:  # a thing in steady movement: the place its line gives
            steady = read & ~resting & ~self.own & self._moving & (self._count >= 6)
            seen_place[steady] = self._line[steady]
            self.radius[steady] = AREA * deviation * 2.0 / np.sqrt(self._count[steady])
            self._smoothed = steady
        # a place another sense names, near the place held: the mean of the two grows sharper
        named = ~read & sounding & (np.linalg.norm(predicted - self.place, axis=1) < self.radius)
        self._named = np.where(named, np.minimum(self._named + 1, 50), np.where(sounding, 1, 0))
        predicted = np.where(named[:, None],
                             self.place + (predicted - self.place)
                             / np.maximum(self._named, 1)[:, None],
                             predicted)
        self.radius = np.where(named, self.radius / np.sqrt(np.maximum(self._named, 1)),
                               self.radius)
        self.place = np.where(read[:, None], seen_place, predicted)
        if self.records is not None:
            filed = self._single & ((resting & np.isin(self._rest, (4, 16, 64, 256)))
                                    | elsewhere)
            for row in np.flatnonzero(filed):
                self.records.write(_key(int(self.looks[row])),
                                   np.array([*self.place[row], float(self._rest[row])]))
        self.since = np.where(read, 0, self.since + 1)
        if learn:
            begin = read & (self.rng.random(rows) < self.practice)
            length = np.minimum(self.rng.geometric(1.0 / self.span, rows), 3 * self.span)
            self._practising = np.where(seen, np.maximum(self._practising - 1, 0), 0)
            self._practising = np.where(begin, length, self._practising)
        self._last = {**step, "refused": refused, "read": read}
        return self.place.copy()

    def advance(self, seen: np.ndarray, reports: np.ndarray,
                commands: np.ndarray | None = None, cues: np.ndarray | None = None,
                joints: np.ndarray | None = None, *, learn: bool = False) -> np.ndarray:
        """One step, ``predict`` and then ``observe``: every row's place after it."""
        self.predict(commands, cues, joints, learn=learn)
        return self.observe(seen, reports, learn=learn)

    def assign(self, reports: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Give the camera's reports of a step to rows, after ``predict``. ``reports`` holds one
        row a report: x, y and what the thing looks like. Reports of one look go to the rows of
        that look at the least summed distance from the rows' predicted places; a report with no
        row of its look is left out. Gives ``seen`` and ``reports`` as ``observe`` takes
        them."""
        reports = np.asarray(reports, dtype=np.float64).reshape(-1, 3)
        seen, z = np.zeros(self.rows, dtype=bool), np.zeros((self.rows, 2))
        predicted = self._step["predicted"]
        for look in np.unique(reports[:, 2]):
            members = np.flatnonzero(self.looks == int(look))
            points = reports[reports[:, 2] == look, :2]
            for i, j in _pairs(predicted[members], points):
                seen[members[i]], z[members[i]] = True, points[j]
        return seen, z

    def hear(self, cues: np.ndarray) -> np.ndarray:
        """Give the ear's cues of a step to rows, before ``predict``. ``cues`` holds one row a
        cue: direction, distance and what the thing looks like. A cue goes to the row of its
        look whose place is nearest the point the cue names. Gives ``(rows, 2)`` directions and
        distances, NaN for a silent row, as ``predict`` takes them."""
        cues = np.asarray(cues, dtype=np.float64).reshape(-1, 3)
        out = np.full((self.rows, 2), np.nan)
        ear = np.asarray(self.interface.ear)
        for direction, distance, look in cues:
            members = np.flatnonzero(self.looks == int(look))
            if len(members):
                point = ear + distance * np.array([np.cos(direction), np.sin(direction)])
                nearest = members[int(np.argmin(np.linalg.norm(self.place[members] - point,
                                                               axis=1)))]
                out[nearest] = direction, distance
        return out

    def find(self, look: int) -> tuple[np.ndarray, int] | None:
        """The record of a thing by what it looks like: its place and the reports its place is
        a mean of, or None where the records hold no such thing."""
        if self.records is None or not self.records.count:
            return None
        key = _key(look)
        slot = self.records.read(key, credit=False).first()
        if slot < 0 or float(self.records.keys[slot] @ key) < 0.9:
            return None
        value = self.records.values[slot]
        return np.array(value[:2], dtype=np.float64), int(value[2])

    def where(self) -> tuple[np.ndarray, np.ndarray]:
        """Every row's place and the radius of the area that holds the thing at ``RATE``."""
        return self.place.copy(), self.radius.copy()

    def state(self) -> dict[str, Any]:
        """What the regions hold after the latest step, as plain arrays: ``regions`` (each
        region's size), ``at`` (every row's place), ``radius`` (of its area at ``RATE``),
        ``since`` (the moments since it was seen), the camera's learned ``noise``, every input
        region's activity by the region's name (one row a thing), ``live`` (the rows the brain
        settled on) and for those rows the hidden activity and every output's belief;
        ``refused`` marks the rows held back from where things are seen."""
        f = self.interface
        out: dict[str, Any] = {"regions": dict(f.regions), "at": self.place.copy(),
                               "radius": self.radius.copy(), "since": self.since.copy(),
                               "noise": float(np.sqrt(self.noise))}
        if not self._last:
            return out
        at = 0
        for name in INPUTS:
            out[name] = self._last["x"][:, at : at + f.regions[name]].copy()
            at += f.regions[name]
        out["live"] = self._last["live"].copy()
        out["refused"] = self._last["refused"].copy()
        if "hidden" in self._last:
            t = float(self.brain.learner.config.temperature)
            d, b = f.shift, f.found
            out["hidden"] = self._last["hidden"].copy()
            out["shift"] = np.stack([self.shift.beliefs(self._last["shift"][:, :d], t),
                                     self.shift.beliefs(self._last["shift"][:, d:], t)], axis=1)
            out["spread"] = self.spread.beliefs(self._last["spread"], t)
            out["found"] = np.stack([self.found[0].beliefs(self._last["found"][:, :b], t),
                                     self.found[1].beliefs(self._last["found"][:, b:], t)],
                                    axis=1)
        return out

    # -- inside

    def _settle(self, x: np.ndarray, live: np.ndarray) -> dict[str, np.ndarray]:
        """Settle the brain on the live rows, from rest: each region's activity, one row a live
        row. A place brain answers from what its inputs hold alone. Its hidden region and its
        outputs read each other, so a settle that began from the step before could keep that
        step's answer; from rest an answer is a function of the inputs, in a lesson as in an
        answer."""
        self.brain.reset()
        self.brain.act(x[live], greedy=True)
        settled = self.brain.last_settlement or {}
        count = int(live.sum())
        self.work["act_sweeps_rows"] += float(settled.get("steps", 0)) * count
        self.work["moments"] += count
        state = self.brain.basal_ganglia.state
        return {name: np.asarray(state.part(self._index[name]), dtype=np.float64)
                for name in ("hidden", *OUTPUTS)}

    def _push(self, rows: np.ndarray, reports: np.ndarray) -> None:
        """A report read joins its row's trail, and the row's seen movement is the slope of the
        line through its reports in a row (three or more, ``Interface.memory`` at most). Where
        the latest ``Interface.check`` reports move otherwise than the whole line by more than
        three times their noise, the movement has changed and the line keeps those reports
        alone. A row is moving where its movement is above three times the line's noise; a line
        of under six reports says moving."""
        if not rows.any():
            return
        f = self.interface
        self._trail[rows] = np.roll(self._trail[rows], -1, axis=1)
        self._trail[rows, -1] = reports[rows]
        self._count[rows] = np.minimum(self._count[rows] + 1, f.memory)
        slope, last, noise = self._fit(self._count)
        late = rows & (self._count >= 2 * f.check)
        if late.any():
            recent, _, recent_noise = self._fit(np.full(self.rows, f.check, dtype=np.intp))
            changed = late & (np.linalg.norm(recent - slope, axis=1)
                              > 3.0 * np.sqrt(2.0 * (noise**2 + recent_noise**2)))
            if changed.any():
                self._count[changed] = f.check
                slope, last, noise = self._fit(self._count)
        fitted = rows & (self._count >= 3)
        self._movement[fitted] = slope[fitted]
        self._fitted[fitted] = self._count[fitted]
        self._known |= fitted
        self._line = np.where(fitted[:, None], last, self._line)
        loud = np.linalg.norm(slope, axis=1) > 3.0 * np.sqrt(2.0) * noise
        self._moving = np.where(fitted, np.where(self._count >= 6, loud, True), self._moving)

    def _fit(self, count: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The line through each row's latest ``count`` reports by least squares: its slope a
        step ``(rows, 2)``, its value at the latest report ``(rows, 2)`` and the slope's noise
        on an axis ``(rows,)``. A row of under two reports has a slope of zero."""
        length = self._trail.shape[1]
        steps = np.arange(length, dtype=np.float64)
        n = np.maximum(count, 1).astype(np.float64)
        within = (steps[None, :] >= length - n[:, None]).astype(np.float64)
        middle = (within * steps).sum(axis=1) / n
        lever = within * (steps[None, :] - middle[:, None])
        power = np.maximum((lever**2).sum(axis=1), 1e-12)
        slope = np.einsum("rl,rlk->rk", lever, self._trail) / power[:, None]
        slope = np.where((count >= 2)[:, None], slope, 0.0)
        mean = np.einsum("rl,rlk->rk", within, self._trail) / n[:, None]
        last = mean + slope * (length - 1 - middle)[:, None]
        return slope, last, np.sqrt(self.noise / power)

    def _cell(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        at = np.clip(((points - self._low) / self._size * self.cells).astype(np.intp), 0,
                     self.cells - 1)
        return at[:, 0], at[:, 1]

    def _in_sight(self, points: np.ndarray) -> np.ndarray:
        """Whether things at these points are seen: the share of reports among the visits of
        each point's cell is above ``sight``, on 20 visits or more."""
        if self.sight is None:
            return np.zeros(len(points), dtype=bool)
        i, j = self._cell(points)
        visits = self.visit_map[i, j]
        return np.asarray((visits >= 20) & (self.seen_map[i, j] > self.sight * visits))

    def _teach(self, step: dict[str, np.ndarray], seen: np.ndarray, z: np.ndarray,
               predicted: np.ndarray, read: np.ndarray) -> None:
        """The step's lessons, each for an answer of a few steps before. A row that had a report
        teaches ``spread`` how far off its predicted place was and, heard as well, teaches
        ``found`` the place. ``shift`` is taught the slope of six reports in a row: for an arm's
        own end the six around the answer, for a thing that moves by itself the six from the
        answer on."""
        f, t = self.interface, float(self.brain.learner.config.temperature)
        sounding = step["sounding"]
        off = np.linalg.norm(z - predicted, axis=1)
        steady = read & (self.since == 0) & ~step["quiet"] & ~sounding & ~self._smoothed
        if steady.any():  # a report against a prediction from a report: twice the camera's noise
            self.noise += 0.01 * (float(np.mean(off[steady] ** 2)) / 4.0 - self.noise)
        i, j = self._cell(np.where(seen[:, None], z, predicted))
        np.add.at(self.visit_map, (i, j), 1.0)
        np.add.at(self.seen_map, (i[seen], j[seen]), 1.0)
        self._pending.append({"x": step["x"], "seen": seen.copy(), "z": z.copy(), "off": off,
                              "sounding": sounding.copy(),
                              **{name: step[name] for name in OUTPUTS}})
        if len(self._pending) < len(SLOPE):
            return
        held = list(self._pending)
        around, ahead = held[3], held[0]  # the answers taught: an arm's own end, another thing
        entry = {key: np.where(_wide(self.own, around[key]), around[key], ahead[key])
                 for key in around}
        whole = np.all([e["seen"] & ~e["sounding"] for e in held], axis=0)
        slope = np.tensordot(SLOPE, np.stack([e["z"] for e in held]), axes=1)
        d, b = f.shift, f.found
        targets = np.hstack([self.shift.beliefs(entry["shift"][:, :d], t),
                             self.shift.beliefs(entry["shift"][:, d:], t),
                             self.spread.beliefs(entry["spread"], t),
                             self.found[0].beliefs(entry["found"][:, :b], t),
                             self.found[1].beliefs(entry["found"][:, b:], t)])
        if whole.any():
            targets[whole, :d] = self.shift.targets(slope[whole, 0])
            targets[whole, d : 2 * d] = self.shift.targets(slope[whole, 1])
        wide = entry["seen"]
        if wide.any():
            targets[wide, 2 * d : 2 * d + f.spread] = self.spread.targets(entry["off"][wide])
        both = entry["seen"] & entry["sounding"]
        if both.any():
            at = 2 * d + f.spread
            targets[both, at : at + b] = self.found[0].targets(entry["z"][both, 0], width=1.0)
            targets[both, at + b :] = self.found[1].targets(entry["z"][both, 1], width=1.0)
        taught = whole | wide
        if taught.any():
            self.brain.reset()  # the lesson's free phase from rest, as the answer it teaches
            report = self.brain.lesson(entry["x"], targets=targets,
                                       weights=taught.astype(np.float64))
            self.work["lesson_row_sweeps"] += float(report.get("total_row_sweeps", 0.0))


def _key(look: int) -> np.ndarray:
    """The key of a look: a fixed code of unit length, one for one look."""
    code = np.random.default_rng(np.random.SeedSequence([int(look), KEY])).standard_normal(KEY)
    return np.asarray(code / np.linalg.norm(code))


def _pairs(places: np.ndarray, points: np.ndarray) -> list[tuple[int, int]]:
    """Pair places with points at the least summed distance: ``(place, point)`` pairs, as many
    as the fewer of the two. Up to five of either are tried in every order; more are paired
    nearest first."""
    m, n = len(places), len(points)
    if m == 0 or n == 0:
        return []
    cost = np.linalg.norm(places[:, None] - points[None], axis=-1)
    if max(m, n) > 5:
        pairs: list[tuple[int, int]] = []
        for flat in np.argsort(cost, axis=None):
            i, j = divmod(int(flat), n)
            if all(i != a and j != b for a, b in pairs):
                pairs.append((i, j))
        return pairs
    if m <= n:
        best = min(itertools.permutations(range(n), m),
                   key=lambda order: sum(cost[i, j] for i, j in enumerate(order)))
        return list(enumerate(best))
    chosen = min(itertools.permutations(range(m), n),
                 key=lambda order: sum(cost[i, j] for j, i in enumerate(order)))
    return [(i, j) for j, i in enumerate(chosen)]


def _wide(mask: np.ndarray, like: np.ndarray) -> np.ndarray:
    """A row mask shaped to choose whole rows of ``like``."""
    return np.asarray(mask.reshape((-1,) + (1,) * (like.ndim - 1)))
