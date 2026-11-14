"""The parrot's meaning brain on a page (cadence#334): load a trained brain and talk to it.

A page holds two files: the brain (``Brain.save``) and its ``spec`` (JSON): the senses' layout,
each word's heard code, the voice's words in unit order, and the world's feature prototypes (a
kind's shape and width, a colour's hue, saturation and lightness, a material's surface), so the
page reports a thing as the chamber's eyes do.

Every moment (about 300 ms):

1. ``see(things)``: what the parrot holds, each thing a dict with its ``kind``, ``colour``,
   ``material``, ``direction`` (radians from where the head points, left positive),
   ``distance`` (metres) and ``focus`` (clicked or held up). Only its features enter the brain;
   the thing in focus is also seen in a place of its own (the glimpse).
2. ``hear(word)``: the word a trainer says this moment (or None).
3. ``moment()``: the brain settles and acts; it returns the word the voice says (None: silence
   or held). ``more=True`` holds the voice: pass it on every moment the trainer speaks. The
   parrot's turn is the gap after the trainer stops: three moments with no word heard (for a
   question about the whole scene, one moment a thing, ``see(things, glimpse=thing)``).
4. ``pistachio()``: a pistachio for the last reply: the three-factor rule, one lesson on the
   moments of the reply with the parrot's own words as the labels.
5. ``correct(word)``: after a wrong reply the trainer says the right word: where the brain keeps
   word records (a ``records`` file beside the brain), the word is kept under the association
   region's pattern of the moment the question ended. At the parrot's turn the records its
   pattern then recalls (the nearest records, where they agree) are replayed to its ear.

The voice says a word by replaying the take of the trainer's word its memory keeps (the page's
audio); this module returns which word.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..generic import Brain
from .senses import Eyes, Seen


class MeaningParrot:
    """A trained meaning brain and its senses, driven one moment at a time."""

    def __init__(self, brain: str | Path, spec: str | Path, *, seed: int = 0,
                 records: str | Path | None = None):
        self.brain = Brain.load(brain)
        self.spec = json.loads(Path(spec).read_text())
        eyes = self.spec["eyes"]
        self.eyes = Eyes(**eyes)
        self.codes = {w: np.asarray(c, dtype=np.float64) for w, c in self.spec["codes"].items()}
        self.width = int(self.spec["ear_width"])
        self.voice: list[str] = list(self.spec["voice"])
        self.silence = len(self.voice)
        self.rng = np.random.default_rng(seed)
        self.things: list[Seen] = []
        self.glimpsed: Seen | None = None
        self.heard: str | None = None
        self.own: str | None = None
        self.reply: list[tuple[np.ndarray, int]] = []
        self.rehearsal: list[dict[str, Any]] = []
        # the sparse mixed code of the glimpse and the words heard (``mixed`` in the spec)
        mixed = self.spec.get("mixed")
        self.mixed_w = None if mixed is None else np.asarray(mixed["w"], dtype=np.float64)
        self.mixed_on = 0 if mixed is None else int(mixed["on"])
        self.mixed_decay = 0.0 if mixed is None else float(mixed["decay"])
        self.trace = np.zeros(self.width)
        self.brain.reset()
        # the word records: kept under the association's pattern, replayed to the ear at the turn
        self.records: Any = None
        self.cue: np.ndarray | None = None  # the pattern of the last moment a word was heard
        self.recalled = np.zeros(self.width)
        self.in_turn = False
        given = self.spec.get("records")
        path = Path(records) if records is not None else Path(brain).with_suffix(".records.npz")
        if given is not None and path.exists():
            from ..longterm import LongTermMemory

            self.records = LongTermMemory(**given["settings"])
            with np.load(path) as f:
                self.records.restore({k: f[k] for k in f.files})
            self.association = np.asarray(list(self.brain.connectome.populations["association"]))

    def code(self, word: str | None) -> np.ndarray:
        if word is None:
            return np.zeros(self.width)
        return self.codes.get(word, np.zeros(self.width)).copy()

    def seen(self, thing: dict[str, Any]) -> Seen:
        kind = self.spec["kinds"][thing["kind"]]
        hue, sat, light = self.spec["colours"][thing["colour"]]
        surface = self.spec["materials"][thing["material"]]
        return Seen(hue=math.radians(hue if hue is not None else 0.0), saturation=sat,
                    lightness=light, shape=np.asarray(kind["shape"]), width=kind["width"],
                    direction=float(thing["direction"]), distance=float(thing["distance"]),
                    salient=bool(thing.get("focus", False)), surface=np.asarray(surface))

    def see(self, things: list[dict[str, Any]], glimpse: dict[str, Any] | None = None) -> None:
        """What the parrot holds; ``glimpse`` (a thing) is where its focus is this moment, by
        default the thing in focus (during a scan the focus visits each thing in turn)."""
        self.things = [self.seen(t) for t in things][: self.eyes.slots]
        self.glimpsed = self.seen(glimpse) if glimpse is not None else None

    def hear(self, word: str | None) -> None:
        self.heard = word

    def observation(self) -> np.ndarray:
        order = self.rng.permutation(len(self.things))
        scene = self.eyes.scene([self.things[int(i)] for i in order])
        focus = self.glimpsed if self.glimpsed is not None else next(
            (t for t in self.things if t.salient), None)
        glimpse = [self.eyes.slot(focus)] if self.spec.get("glimpse") else []
        ear = self.code(self.heard) + self.code(self.own) + self.recalled
        parts = glimpse + [scene]
        if self.mixed_w is not None:  # the glimpse and the heard trace, expanded and sparse
            drive = np.concatenate([self.eyes.slot(focus), self.trace]) @ self.mixed_w
            code = np.zeros(self.mixed_w.shape[1])
            code[np.argpartition(drive, -self.mixed_on)[-self.mixed_on:]] = 1.0
            parts.append(code)
        return np.concatenate(parts + [ear])[None]

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """The word the records hold for ``cue``: the nearest records' value where the
        ``agree`` nearest hold one word and the nearest lies at cosine ``match`` or above."""
        assert self.records is not None
        if not self.records.count or not np.any(cue):
            return np.zeros(self.width)
        heads = min(int(self.spec["records"]["agree"]), int(self.records.count))
        got = self.records.read(cue, heads=heads, credit=True)
        slot = got.first()
        if slot < 0:
            return np.zeros(self.width)
        key = np.asarray(self.records.keys[slot], dtype=np.float64)
        cos = float(key @ cue) / (np.linalg.norm(key) * np.linalg.norm(cue) + 1e-12)
        value = self.records.value(got)
        if cos >= float(self.spec["records"].get("exact", 0.95)):
            return value  # the situation itself, recognised: its record alone is enough
        for h in range(1, heads):
            if got.first(h) < 0 or not np.array_equal(self.records.value(got, h), value):
                return np.zeros(self.width)
        return value if cos >= float(self.spec["records"]["match"]) else np.zeros(self.width)

    def correct(self, word: str) -> bool:
        """The trainer's right word after a reply: kept as a record under the pattern of the
        moment the question ended. Returns whether a record was written."""
        if self.records is None or self.cue is None or word not in self.codes:
            return False
        self.records.write(self.cue, self.code(word))
        return True

    def moment(self, *, more: bool = False, greedy: bool = True) -> str | None:
        """One moment: settle on what is seen and heard, act; the word said, or None."""
        if self.records is not None:
            self.records.moment()  # a moment of the store: unused, shallow records fade first
            if self.heard is not None:  # the trainer speaks: no turn, nothing recalled
                self.in_turn, self.recalled = False, np.zeros(self.width)
            elif not more and not self.in_turn and self.cue is not None:
                self.in_turn = True  # the turn opens in the gap: the records are read once
                self.recalled = self.recall(self.cue)
        heard_now = self.heard is not None
        if self.mixed_w is not None:  # the trace of the words heard in this exchange
            self.trace = self.mixed_decay * self.trace + self.code(self.heard)
        x = self.observation()
        action = np.asarray(self.brain.act(x, greedy=greedy)).reshape(1, -1)
        if self.records is not None and heard_now:
            self.cue = np.asarray(self.brain.basal_ganglia.state.activation, dtype=np.float64)[
                0, self.association].copy()
        unit = int(action[0, 0])
        said = None if unit >= self.silence or more else self.voice[unit]
        if said is not None:
            self.reply.append((x[0].copy(), unit))
        self.own, self.heard = said, None
        return said

    def pistachio(self, amount: float = 1.0, *, first: bool = False,
                  rehearse: int = 0) -> int:
        """A pistachio for the last reply: its moments taught with its own words. Returns the
        lessons taken. ``first`` teaches each word of the reply once (its first moment), and
        ``rehearse`` replays that many of the night's exchanges (``rehearsal.json`` beside the
        brain, ``load_rehearsal``) after the lesson, so a paid answer does not overwrite the
        other answers the same focus gives: the training's revisits, on the page."""
        taken = 0
        seen: set[int] = set()
        for x, unit in self.reply:
            if first and unit in seen:
                continue
            seen.add(unit)
            self.brain.lesson(x[None], np.array([[unit, 0]]), weights=np.array([amount]))
            taken += 1
        self.reply = []
        self.trace = np.zeros(self.width)
        for _ in range(int(rehearse) if self.rehearsal else 0):
            self.rehearse_one(amount)
        return taken

    def load_rehearsal(self, path: str | Path) -> int:
        """The night's exchanges to revisit after a paid lesson: a list of {things, words,
        answer}, each a room as ``see`` takes it, the trainer's words and the right answer."""
        self.rehearsal = json.loads(Path(path).read_text())
        return len(self.rehearsal)

    def rehearse_one(self, amount: float = 1.0) -> None:
        """One revisit: the exchange's words heard (the voice held), then the right answer's
        first word taught on the moment the turn opens (the gap)."""
        item = self.rehearsal[int(self.rng.integers(len(self.rehearsal)))]
        held_reply, held_own = self.reply, self.own
        self.brain.reset()
        self.own, self.reply = None, []
        self.see(item["things"])
        for w in item["words"]:
            self.hear(w)
            self.moment(more=True)
        unit = self.voice.index(item["answer"][0])
        self.brain.lesson(self.observation(), np.array([[unit, 0]]), weights=np.array([amount]))
        self.brain.reset()
        self.reply, self.own = held_reply, held_own

    def turn_over(self) -> None:
        """The exchange ends: the reply's moments are forgotten (no pistachio came)."""
        self.reply = []
        self.trace = np.zeros(self.width)
        self.in_turn, self.recalled = False, np.zeros(self.width)
