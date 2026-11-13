"""Time in words: where things were, from the brain's records, and where they will be, from its
forecast, with the words for time learned from answered questions (cadence#326, W8 to W10).

``Episodes`` is a part of ``cadence.words.Words`` (``Words(map, parts=[Episodes(map)])``). A
question is answered from three sources, and the words of the question choose among them:

- **the records**, the places things left. Each moment the brain compares where its map holds
  each thing with the place the thing rested at; a thing that leaves its place by more than
  ``leave`` is written as a record (the place it left, and what it is) in a long-term store
  (``cadence.longterm.LongTermMemory``) keyed by the thing and the moment. A thing that holds its
  place again (it moved less than ``still`` in a moment) rests there. The map holds the latest
  place alone: the place before comes from the records only.
- **the map**, what is where now (what ``Words`` finds).
- **the forecast**, the map ``ahead`` moments on as its learned physics predicts
  (``Map.forecast(moments)``, a copy that leaves the map as it is). A map with no physics
  forecasts itself: what is at rest stays.

**When.** A drifting context (``Clock``) is the brain's sense of when: a population of units
whose activity drifts a little every moment, so two moments near in time hold near contexts. A
record's key is the thing's code and the context of the moment it was written; a read is cued by
the thing and the context now, so among a thing's records the latest scores first.

**The words for time.** The part adds one concept a source to the lexicon (``when``: the
records, the map, the forecast), joined to every word by synapses that start at zero. A
question's words settle on them as on every other concept. In practice the brain hears the
teacher's answer and finds which of its sources give that answer for the thing asked about;
where its sources disagree, the question's words of asking are taught those sources. No word is
wired to a source, and the words for time are nonce words like every other.

**An answer.** The settled ``when`` activities compete for the sources' gains as the reads
compete for theirs (a softmax at the brain's ``sharpness``); a question whose words settle on no
source above the floor asks the map alone. Each source whose gain is above the brain's
``floor`` finds the things the question names, and a thing's concepts are each source's at its
gain; ``Words`` then reads and says them. With the records barred (``Records.barred``), a
question about the past finds nothing at the records' gain, and the brain says it does not know.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SOURCES = ("records", "map", "forecast")  # where an answer comes from: the past, now, ahead


def forecast(held: Any, moments: int) -> Any:
    """The map ``moments`` moments on: ``held.forecast(moments)`` where the map has learned
    physics (a copy; the map is left as it is), else the map itself: with no physics nothing is
    predicted to move."""
    if getattr(held, "physics", None) is not None:
        return held.forecast(int(moments))
    return held


class Clock:
    """A drifting context: ``units`` units at unit length. Each moment (``tick``) the context
    becomes ``drift * c + sqrt(1 - drift ** 2) * n`` at unit length, ``n`` a draw of the
    founder's ``seed``, so contexts ``t`` moments apart hold a cosine near ``drift ** t``."""

    def __init__(self, units: int = 64, drift: float = 0.99, seed: int = 0):
        if int(units) < 1 or not 0.0 < float(drift) < 1.0:
            raise ValueError("a clock takes 1 unit or more and a drift in (0, 1)")
        self.units, self.drift = int(units), float(drift)
        self._draw = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 8]))
        start = self._draw.standard_normal(self.units)
        self.now = start / np.linalg.norm(start)
        self.moments = 0

    def tick(self) -> None:
        """One moment: the context drifts."""
        noise = self._draw.standard_normal(self.units) / math.sqrt(self.units)
        moved = self.drift * self.now + math.sqrt(1.0 - self.drift ** 2) * noise
        self.now = moved / np.linalg.norm(moved)
        self.moments += 1


class Records:
    """The places things left, keyed by the thing and the time: a ``LongTermMemory`` of
    ``capacity`` slots whose key is a thing's code (``width`` units: the map's code of its kind
    and its values, each unit at ``thing``) beside the clock's context at unit length, and whose
    value is the place the thing left and its code. The read is the memory's: the dot product of
    the cue with every key, the first choice whole, so a record scores ``thing ** 2`` for each
    unit of the cue's code it holds (a unit it holds beyond them costs nothing) and the cosine
    of its moment's context with the context now. A record is the thing asked about where its
    code holds every unit of the cue's code (``sure`` of it).

    ``barred`` bars every read: a recall then finds nothing."""

    def __init__(self, width: int, *, capacity: int = 256, clock: Clock | None = None,
                 thing: float = 2.0, sure: float = 0.99, seed: int = 0):
        from ..longterm import LongTermMemory

        if int(width) < 1 or not float(thing) > 0.0:
            raise ValueError("records take a code of 1 unit or more and a thing's length above 0")
        self.width, self.thing, self.sure = int(width), float(thing), float(sure)
        self.clock = clock if clock is not None else Clock(seed=seed)
        self.store = LongTermMemory(self.width + self.clock.units, 2 + self.width,
                                    int(capacity), temperature=0.0, revise=0.0, seed=int(seed))
        self.barred = False
        self.work = {"writes": 0, "reads": 0, "macs": 0}

    def key(self, code: np.ndarray) -> np.ndarray:
        """A thing's code and the context now, as a key and as a cue."""
        code = np.asarray(code, dtype=np.float64).reshape(self.width)
        if not code.any():
            raise ValueError("a thing's code holds a unit")
        return np.concatenate([self.thing * code, self.clock.now])

    def write(self, code: np.ndarray, place: Any) -> None:
        """A record: the thing of ``code`` left ``place`` at this moment."""
        value = np.concatenate([np.asarray(place, dtype=np.float64).reshape(2),
                                np.asarray(code, dtype=np.float64).reshape(self.width)])
        self.store.write(self.key(code), value, recognised=-1)
        self.work["writes"] += 1

    def recall(self, code: np.ndarray, *, probe: bool = False
               ) -> tuple[np.ndarray, np.ndarray] | None:
        """The latest place the thing of ``code`` left and the code it had there: the memory
        read with the thing's code and the context now. None where the records are barred or
        hold no record of the thing. A ``probe`` (a reader looking in) leaves the records and
        the work as they were."""
        if self.barred or not self.store.count:
            return None
        cue = self.key(code)
        read = self.store.read(cue, credit=not probe)
        if not probe:
            self.work["reads"] += 1
            self.work["macs"] += self.store.count * len(cue)
        slot = read.first()
        if slot < 0:
            return None
        value = np.asarray(self.store.values[slot], dtype=np.float64)
        held, asked = value[2:], np.asarray(code, dtype=np.float64).reshape(self.width)
        if float(held @ asked) < self.sure * float(asked @ asked):
            return None  # the best record is of another thing
        return value[:2].copy(), held.copy()

    def tick(self) -> None:
        """One moment: the clock drifts and the store's records fade by their use."""
        self.clock.tick()
        self.store.moment()
        self.work["macs"] += self.clock.units


class Episodes:
    """A part of ``cadence.words.Words`` that answers in time: its records (``Records`` sized to
    ``map``'s sheet, made here unless given) and its forecast ``ahead`` moments on, with one
    concept a source (``when``) in the lexicon. ``leave`` is the distance in metres a thing goes
    from where it rested for the place to be written as a record; ``still`` the distance under
    which a thing that moved rests again.

    The part keeps up with the map: before each moment the brain looks (``Words.look``) and
    before each question, every map moment not yet witnessed is witnessed (the clock drifts and
    the records are written). A map that holds nothing (a new room) holds no resting place.
    ``last`` holds what each source found for the latest question and the sources' gains."""

    def __init__(self, map: Any, *, records: Records | None = None, ahead: int = 8,
                 leave: float = 0.3, still: float = 0.02, seed: int = 0):
        if int(ahead) < 0 or not float(leave) > float(still) >= 0.0:
            raise ValueError("ahead is 0 moments or more, and leave exceeds still")
        self.groups = {"when": len(SOURCES)}
        self.records = records if records is not None else Records(map.width, seed=seed)
        if self.records.width != map.width:
            raise ValueError("the records take a code as wide as the map's sheet")
        self.ahead, self.leave, self.still = int(ahead), float(leave), float(still)
        self._rested: dict[int, np.ndarray | None] = {}  # where each thing rests; None: moving
        self._last: dict[int, np.ndarray] = {}  # where the map held each thing, last witnessed
        self._witnessed = int(map.work["moments"])
        self.last: dict[str, Any] = {}
        # the forecasts made, their moments and the questions asked of them (a forecast is a
        # copy: its work is counted here)
        self.work = {"forecasts": 0, "forecast_moments": 0, "forecast_questions": 0}

    # -- living: the records written as the map's moments pass

    @staticmethod
    def code(map: Any, kind: int, values: dict[str, int | None] | None = None) -> np.ndarray:
        """A thing's code on the map's sheet: its kind's units and its values' units."""
        out = np.zeros(map.width)
        out[: map.units] = map.codes[int(kind)]
        for name, value in (values or {}).items():
            if value is not None and name in map.values:
                out[map.values[name].start + int(value)] = 1.0
        return out

    def sense(self, brain: Any, seen: Any, lit: Any) -> None:
        """Before a moment of the senses: the moments the map took since the last witness."""
        self.witness(brain.map)

    def witness(self, map: Any) -> None:
        """For each map moment not yet witnessed the clock drifts; then each thing the map holds
        is compared with where it rested: a thing that left its resting place by more than
        ``leave`` is written as a record of that place and is moving; a moving thing that moved
        less than ``still`` since rests where it is. A thing the map holds in several places is
        where it was sensed last."""
        moments = int(map.work["moments"])
        for _ in range(max(moments - self._witnessed, 0)):
            self.records.tick()
        fresh = moments != self._witnessed
        self._witnessed = moments
        if not map.potential.any():  # a map that holds nothing: a new room
            self._rested.clear()
            self._last.clear()
            return
        if not fresh:
            return
        latest: dict[int, dict[str, Any]] = {}
        for entry in map.things():
            if entry["place"] is None:
                continue
            kind = int(entry["kind"])
            if kind not in latest or entry["since"] < latest[kind]["since"]:
                latest[kind] = entry
        for kind, entry in latest.items():
            place = np.asarray(entry["place"], dtype=np.float64)
            if kind not in self._last:  # first held: it rests where it is
                self._rested[kind] = place
            else:
                rested = self._rested.get(kind)
                if rested is not None and float(np.linalg.norm(place - rested)) > self.leave:
                    self.records.write(self.code(map, kind, entry.get("values")), rested)
                    self._rested[kind] = None
                elif rested is None and float(
                        np.linalg.norm(place - self._last[kind])) <= self.still:
                    self._rested[kind] = place
            self._last[kind] = place

    # -- what each source finds

    def gains(self, brain: Any, heard: Any) -> np.ndarray:
        """The sources' gains from a question's settle: the ``when`` activities compete, a
        softmax at the brain's ``sharpness``; where no source is named above the floor, the
        question asks the map alone: the present."""
        when = np.maximum(np.asarray(heard.concepts[brain.lexicon.groups["when"]],
                                     dtype=np.float64), 0.0)
        if not when.max() > brain.floor:
            return np.asarray(np.eye(len(SOURCES))[SOURCES.index("map")])
        odds = np.exp((when - when.max()) / max(float(brain.sharpness), 1e-9))
        return np.asarray(odds / odds.sum())

    @staticmethod
    def _on(brain: Any, held: Any, about: tuple[str, np.ndarray]
            ) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
        """The things a map holds that the question names, each a place and what the senses
        give of it there."""
        from .room import Senses

        senses = Senses(held, brain.body)
        out = []
        for place in senses.find(*about):
            described = senses.describe(place)
            if described is not None:
                out.append((np.asarray(place), described))
        return out

    def _recalled(self, brain: Any, about: tuple[str, np.ndarray], probe: bool = False
                  ) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
        """What the records hold of the thing a kind names: the latest place it left and what
        it was there. The records are read by a thing: a value or a place around the body
        finds nothing there."""
        group, pattern = about
        if group != "kind":
            return []
        kind = int(np.argmax(pattern))
        recalled = self.records.recall(self.code(brain.map, kind), probe=probe)
        if recalled is None:
            return []
        place, code = recalled
        described: dict[str, np.ndarray] = {"kind": np.eye(len(brain.map.codes))[kind]}
        for name, units in brain.map.values.items():
            described[name] = np.asarray(code[units], dtype=np.float64).copy()
        described["place"] = np.eye(int(brain.senses.groups["place"]))[brain.body.place(place)]
        return [(place, described)]

    def sources(self, brain: Any, about: tuple[str, np.ndarray] | None, things: list[Any],
                needed: Any = None, probe: bool = False) -> dict[str, list[Any]]:
        """What each source finds of what a question names, ``{source: [(place, concepts)]}``:
        the map's are ``things`` (what ``Words`` found); a source whose entry of ``needed`` is
        False is not asked. A ``probe`` leaves the records and the work as they were."""
        out: dict[str, list[Any]] = {"records": [], "map": list(things), "forecast": []}
        if about is None:
            return out
        if needed is None or needed[0]:
            out["records"] = self._recalled(brain, about, probe)
        if needed is None or needed[2]:
            ahead = forecast(brain.map, self.ahead)
            if ahead is brain.map:
                out["forecast"] = list(things)
            else:
                asked = int(ahead.work["questions"])
                out["forecast"] = self._on(brain, ahead, about)
                if not probe:
                    self.work["forecasts"] += 1
                    self.work["forecast_moments"] += self.ahead
                    self.work["forecast_questions"] += int(ahead.work["questions"]) - asked
        return out

    def find(self, brain: Any, heard: Any, things: list[Any]) -> list[Any]:
        """The things found in time: each source whose gain is above the brain's floor finds
        the things the question names, and a thing's concepts (by its kind) are each source's
        at its gain; its place is the one of the source of the most gain."""
        self.witness(brain.map)
        gains = self.gains(brain, heard)
        found = self.sources(brain, heard.about, things, needed=gains > brain.floor)
        combined: dict[int, tuple[float, np.ndarray, dict[str, np.ndarray]]] = {}
        for source, gain in zip(SOURCES, gains, strict=True):
            if not gain > brain.floor:
                continue
            for place, described in found[source]:
                kind = int(np.argmax(described["kind"]))
                top, at, held = combined.get(kind, (0.0, place, {}))
                for group, pattern in described.items():
                    held[group] = held.get(group, 0.0) + float(gain) * np.asarray(pattern)
                combined[kind] = (max(top, float(gain)), place if gain > top else at, held)
        self.last = {"found": found, "gains": gains, "map": list(things)}
        return [(at, held) for _, at, held in combined.values()]

    # -- practice: the lesson of time

    def practice(self, brain: Any, words: list[str], answer: list[str], heard: Any) -> None:
        """The lesson of time: where the answer is one word naming a concept, the sources that
        find that concept for a thing are found; where some source finds the thing with another
        concept, the question's words of asking are taught the sources that gave the answer, at
        equal shares."""
        if len(answer) != 1 or heard.about is None:
            return
        groups = brain.lexicon.groups
        told = brain.lexicon.hear(answer)
        named = max(brain.senses.groups, key=lambda g: float(told[groups[g]].max()))
        if not told[groups[named]].max() > brain.floor:
            return
        concept = int(np.argmax(told[groups[named]]))
        found = self.sources(brain, heard.about, self.last.get("map", []))
        gave, other = [], []
        for source in SOURCES:
            held = [int(np.argmax(d[named])) for _, d in found[source]
                    if named in d and np.asarray(d[named]).max() > 0]
            gave.append(concept in held)
            other.append(any(h != concept for h in held))
        asking = brain.asking(words)
        if asking and any(gave) and any(o and not g for o, g in zip(other, gave, strict=True)):
            share = 1.0 / sum(gave)
            brain.lexicon.name(asking, {("when", i): share for i, g in enumerate(gave) if g},
                               groups=("when",))
