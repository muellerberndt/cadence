"""Time in words: where things were, from the brain's records, and where they will be, from its
forecast, with the words for time learned from answered questions (cadence#326, W8 to W10).

``Words`` answers from its map: what is where now. ``Times`` answers from three sources, and the
words of a question choose among them:

- **the records**, the places things left. Each moment the brain compares where its map holds
  each thing with the place the thing rested at; a thing that leaves its place by more than
  ``leave`` is written as a record (the place it left, and what it is) in a long-term store
  (``cadence.longterm.LongTermMemory``) keyed by the thing and the moment. A thing that holds its
  place again (it moved less than ``still`` in a moment) rests there. The map holds the latest
  place alone: the place before comes from the records only.
- **the map**, what is where now (``Words``'s reads).
- **the forecast**, the map ``ahead`` moments on as its learned physics predicts
  (``Map.forecast(moments)``, a copy that leaves the map as it is). A map with no physics
  forecasts itself: what is at rest stays.

**When.** A drifting context (``Clock``) is the brain's sense of when: a population of units
whose activity drifts a little every moment, so two moments near in time hold near contexts. A
record's key is the thing's code and the context of the moment it was written; a read is cued by
the thing and the context now, so among a thing's records the latest scores first.

**The words for time.** The lexicon holds one more concept a source (``when``: the records, the
map, the forecast), joined to every word by synapses that start at zero. A question's words
settle on them as on every other concept. In practice the brain hears the teacher's answer and
finds which of its sources give that answer for the thing asked about; where its sources
disagree, the question's words are taught those sources. No word is wired to a source, and the
words for time are nonce words like every other.

**An answer.** Each source finds the things the question names and reads each one (its place,
its colour or its kind) as ``Words`` does on the map. The settled ``when`` activities are gains
on what the sources found: a thing's drive on the concepts is each source's concept at its gain
(its activity over the most active source's), and the lexicon's motor says the most active word
of that drive. A drive too weak to raise a word over its threshold says nothing; where no thing
is said, the brain says the word for the read that finds nothing. A question whose words settle
on no source above the floor asks the map alone, as ``Words`` does. With the records barred
(``Records.barred``), a question about the past finds nothing at the records' gain, and the
brain says it does not know.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .lexicon import Lexicon
from .room import PLACES, READS, Words

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
    and its values) at length ``thing`` beside the clock's context at unit length, and whose
    value is the place the thing left and its code. The read is the memory's: the cosine of the
    cue with every key, the first choice whole. A record is the thing asked about where its code
    holds every unit of the cue's code (``sure`` of it).

    ``barred`` bars every read: a recall then finds nothing."""

    def __init__(self, width: int, *, capacity: int = 256, clock: Clock | None = None,
                 thing: float = 2.0, sure: float = 0.99, seed: int = 0):
        from ..longterm import LongTermMemory

        if int(width) < 1 or not float(thing) > 0.0:
            raise ValueError("records take a code of 1 unit or more and a thing's length above 0")
        self.width, self.thing, self.sure = int(width), float(thing), float(sure)
        self.clock = clock if clock is not None else Clock(seed=seed)
        self.store = LongTermMemory(self.width + self.clock.units, 2 + self.width,
                                    int(capacity), temperature=0.0, cosine=True, revise=0.0,
                                    seed=int(seed))
        self.barred = False
        self.work = {"writes": 0, "reads": 0, "macs": 0}

    def key(self, code: np.ndarray) -> np.ndarray:
        """A thing's code and the context now, as a key and as a cue."""
        code = np.asarray(code, dtype=np.float64).reshape(self.width)
        length = float(np.linalg.norm(code))
        if not length > 0.0:
            raise ValueError("a thing's code holds a unit")
        return np.concatenate([self.thing * code / length, self.clock.now])

    def write(self, code: np.ndarray, place: Any) -> None:
        """A record: the thing of ``code`` left ``place`` at this moment."""
        value = np.concatenate([np.asarray(place, dtype=np.float64).reshape(2),
                                np.asarray(code, dtype=np.float64).reshape(self.width)])
        self.store.write(self.key(code), value, recognised=-1)
        self.work["writes"] += 1

    def recall(self, code: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """The latest place the thing of ``code`` left and the code it had there: the memory
        read with the thing's code and the context now. None where the records are barred or
        hold no record of the thing."""
        if self.barred or not self.store.count:
            return None
        cue = self.key(code)
        read = self.store.read(cue)
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

    def moment(self) -> None:
        """One moment: the clock drifts and the store's records fade by their use."""
        self.clock.tick()
        self.store.moment()
        self.work["macs"] += self.clock.units


class Times(Words):
    """A brain that answers about its room in time: ``Words`` with its records (``Records``,
    made here unless given), its forecast ``ahead`` moments on, and a lexicon whose concepts add
    the three sources (``when``). ``leave`` is the distance in metres a thing goes from where it
    rested for the place to be written as a record; ``still`` the distance under which a thing
    that moved rests again. Every other setting is ``Words``'s.

    ``moment(...)`` is one moment of the map's (its reports and sight) followed by the records'
    moment; ``enter()`` begins a new room (the map is emptied, the records stay); ``ask`` and
    ``practice`` are ``Words``'s, in time."""

    def __init__(self, map: Any, *, records: Records | None = None, ahead: int = 8,
                 leave: float = 0.3, still: float = 0.02, lexicon: Lexicon | None = None,
                 rate: float = 0.05, capacity: int = 256, backend: Any = "cpu",
                 seed: int = 0, **settings: Any):
        group = str(settings.get("group", "colour"))
        if group not in map.values:
            raise ValueError(f"the map holds no values of {group!r}: Map(values={{{group!r}: n}})")
        values = map.values[group].stop - map.values[group].start
        if lexicon is None:
            lexicon = Lexicon({"kind": len(map.codes), "colour": values, "place": len(PLACES),
                               "read": len(READS), "none": 1, "when": len(SOURCES)},
                              rate=rate, capacity=capacity, backend=backend)
        if "when" not in lexicon.groups:
            raise ValueError("a brain in time takes a lexicon with the concepts 'when'")
        super().__init__(map, lexicon=lexicon, rate=rate, capacity=capacity, backend=backend,
                         **settings)
        self.records = records if records is not None else Records(map.width, seed=seed)
        if self.records.width != map.width:
            raise ValueError("the records take a code as wide as the map's sheet")
        if int(ahead) < 0 or not float(leave) > float(still) >= 0.0:
            raise ValueError("ahead is 0 moments or more, and leave exceeds still")
        self.ahead, self.leave, self.still = int(ahead), float(leave), float(still)
        self._rested: dict[int, np.ndarray | None] = {}  # where each thing rests; None: moving
        self._last: dict[int, np.ndarray] = {}  # where the map held each thing the moment before
        self.work = {"forecasts": 0, "forecast_moments": 0}

    # -- living: the map's moments and the records written

    def enter(self) -> None:
        """A new room: the map holds nothing and no thing rests anywhere. The records stay."""
        self.map.reset()
        self._rested.clear()
        self._last.clear()

    def moment(self, **reports: Any) -> None:
        """One moment: the map's (``seen``, ``heard``, ``felt``, ``lit``), then the records'."""
        self.map.moment(**reports)
        self.witness()

    def code(self, kind: int, values: dict[str, int | None] | None = None) -> np.ndarray:
        """A thing's code on the map's sheet: its kind's units and its values' units."""
        out = np.zeros(self.map.width)
        out[: self.map.units] = self.map.codes[int(kind)]
        for name, value in (values or {}).items():
            if value is not None and name in self.map.values:
                out[self.map.values[name].start + int(value)] = 1.0
        return out

    def witness(self) -> None:
        """The records' moment: the clock drifts, and each thing the map holds is compared with
        where it rested. A thing that left its resting place by more than ``leave`` is written
        as a record of that place and is moving; a moving thing that moved less than ``still``
        in the moment rests where it is. A thing the map holds in several places is where it
        was sensed last."""
        self.records.moment()
        latest: dict[int, dict[str, Any]] = {}
        for entry in self.map.things():
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
                    self.records.write(self.code(kind, entry.get("values")), rested)
                    self._rested[kind] = None
                elif rested is None and float(
                        np.linalg.norm(place - self._last[kind])) <= self.still:
                    self._rested[kind] = place
            self._last[kind] = place

    # -- what each source finds

    def _read(self, held: Any, place: np.ndarray, read: int, kind: int) -> tuple[str, int] | None:
        """The read's concept of the thing of ``kind`` at ``place`` on a map."""
        if READS[read] == "kind":
            return ("kind", int(kind))
        if READS[read] == "colour":
            value = held.values_at(place)[self.group]
            return None if value is None else ("colour", int(value))
        return ("place", self.body.place(place))

    def _on_map(self, held: Any, about: tuple[str, int], read: int) -> list[tuple[int, Any]]:
        """The things a map holds that a concept names, each with the read's concept of it:
        ``(kind, concept)``."""
        group, value = about
        mine, self.map = self.map, held
        try:
            if group == "kind":
                found = [(int(value), p) for p in held.where_is(value)]
            else:
                places = (list(held.where_is_value(self.group, value)) if group == "colour"
                          else self._things_in(value))
                found = [(int(k), p) for p in places
                         for k in [held.what_is_at(p)] if k is not None]
            out = []
            for kind, place in found:
                concept = self._read(held, place, read, kind)
                if concept is not None:
                    out.append((kind, concept))
            return out
        finally:
            self.map = mine

    def _in_records(self, about: tuple[str, int], read: int) -> list[tuple[int, Any]]:
        """What the records hold of the thing a kind names: the latest place it left, read for
        the read's concept. The records are read by a thing: a colour or a place around the
        body finds nothing there."""
        group, value = about
        if group != "kind":
            return []
        recalled = self.records.recall(self.code(int(value)))
        if recalled is None:
            return []
        place, code = recalled
        if READS[read] == "kind":
            return [(int(value), ("kind", int(value)))]
        if READS[read] == "colour":
            units = code[self.map.values[self.group]]
            return [(int(value), ("colour", int(np.argmax(units))))] if units.max() > 0 else []
        return [(int(value), ("place", self.body.place(place)))]

    def sources(self, about: tuple[str, int] | None,
                read: int | None) -> dict[str, list[tuple[int, Any]]]:
        """What each source finds for a question: ``{source: [(kind, concept), ...]}``."""
        if about is None or read is None:
            return {s: [] for s in SOURCES}
        ahead = forecast(self.map, self.ahead)
        if ahead is not self.map:
            self.work["forecasts"] += 1
            self.work["forecast_moments"] += self.ahead
        return {"records": self._in_records(about, read),
                "map": self._on_map(self.map, about, read),
                "forecast": self._on_map(ahead, about, read)}

    # -- a question in time

    def gains(self) -> np.ndarray:
        """The sources' gains from the latest question's settle: each ``when`` concept's
        activity over the most active one's, none below zero. Where no source is named above
        the floor, the question asks the map alone: the present."""
        heard = np.maximum(np.asarray(self.last["heard"][self.lexicon.groups["when"]],
                                      dtype=np.float64), 0.0)
        if not heard.max() > self.floor:
            return np.asarray(np.eye(len(SOURCES))[SOURCES.index("map")])
        return np.asarray(heard / heard.max())

    def ask(self, question: str) -> list[str]:
        """The words the brain says to a question: one a thing its sources found, said by the
        lexicon's motor from the sum of each source's concept at its gain; the word for the read
        that finds nothing where no thing is said."""
        about, read = self._heard(self._words(question))
        found = self.sources(about, read)
        gains = self.gains()
        drives: dict[int, np.ndarray] = {}
        for source, gain in zip(SOURCES, gains, strict=True):
            for kind, concept in found[source]:
                drive = drives.setdefault(kind, np.zeros(self.lexicon.concepts))
                drive[self.lexicon.index(*concept)] += gain
        said = [word for drive in drives.values() for word in [self.lexicon.say(drive)] if word]
        if not said:
            word = self.lexicon.say(("none", 0))
            said = [word if word is not None else ""]
        self.last.update({"found": found, "gains": gains, "said": said})
        return said

    def practice(self, question: str, answer: str | list[str]) -> None:
        """``Words.practice``, and the lesson of time: where the answer is one word naming a
        concept, the sources that find that concept for a thing are found; where some source
        finds the thing with another concept, the question's words are taught the sources that
        gave the answer, at equal shares."""
        super().practice(question, answer)
        said = [answer] if isinstance(answer, str) else list(answer)
        if len(said) != 1:
            return
        words = self._words(question)
        about, read = self._heard(words)
        heard = self.lexicon.hear(said)
        groups = self.lexicon.groups
        named = max(("kind", "colour", "place"), key=lambda g: float(heard[groups[g]].max()))
        if not heard[groups[named]].max() > self.floor:
            return
        concept = (named, int(np.argmax(heard[groups[named]])))
        found = self.sources(about, read)
        gave = [any(c == concept for _, c in found[s]) for s in SOURCES]
        other = [any(c != concept for _, c in found[s]) for s in SOURCES]
        if any(gave) and any(o and not g for o, g in zip(other, gave, strict=True)):
            share = 1.0 / sum(gave)
            self.lexicon.name(words, {("when", i): share for i, g in enumerate(gave) if g},
                              groups=("when",))
