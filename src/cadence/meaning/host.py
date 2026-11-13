"""The parrot's meaning brain on a page (cadence#334): load a trained brain and talk to it.

A page holds two files: the brain (``Brain.save``) and its ``spec`` (JSON): the senses' layout,
each word's heard code, the voice's words in unit order, and the world's feature prototypes (a
kind's shape and width, a colour's hue, saturation and lightness, a material's surface), so the
page reports a thing as the chamber's eyes do.

Every moment (about 300 ms):

1. ``see(things)``: what the parrot holds, each thing a dict with its ``kind``, ``colour``,
   ``material``, ``direction`` (radians from where the head points, left positive),
   ``distance`` (metres) and ``focus`` (clicked or held up). Only its features enter the brain.
2. ``hear(word)``: the word a trainer says this moment (or None).
3. ``moment()``: the brain settles and acts; it returns the word the voice says (None: silence
   or held while the trainer has more to say). ``more=True`` holds the voice.
4. ``pistachio()``: a pistachio for the last reply: the three-factor rule, one lesson on the
   moments of the reply with the parrot's own words as the labels.

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

    def __init__(self, brain: str | Path, spec: str | Path, *, seed: int = 0):
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
        self.heard: str | None = None
        self.own: str | None = None
        self.reply: list[tuple[np.ndarray, int]] = []
        self.brain.reset()

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

    def see(self, things: list[dict[str, Any]]) -> None:
        self.things = [self.seen(t) for t in things][: self.eyes.slots]

    def hear(self, word: str | None) -> None:
        self.heard = word

    def observation(self) -> np.ndarray:
        order = self.rng.permutation(len(self.things))
        scene = self.eyes.scene([self.things[int(i)] for i in order])
        return np.concatenate([scene, self.code(self.heard) + self.code(self.own)])[None]

    def moment(self, *, more: bool = False, greedy: bool = True) -> str | None:
        """One moment: settle on what is seen and heard, act; the word said, or None."""
        x = self.observation()
        action = np.asarray(self.brain.act(x, greedy=greedy)).reshape(1, -1)
        unit = int(action[0, 0])
        said = None if unit >= self.silence or more else self.voice[unit]
        if said is not None:
            self.reply.append((x[0].copy(), unit))
        self.own, self.heard = said, None
        return said

    def pistachio(self, amount: float = 1.0) -> int:
        """A pistachio for the last reply: its moments taught with its own words. Returns the
        lessons taken."""
        taken = 0
        for x, unit in self.reply:
            self.brain.lesson(x[None], np.array([[unit, 0]]), weights=np.array([amount]))
            taken += 1
        self.reply = []
        return taken

    def turn_over(self) -> None:
        """The exchange ends: the reply's moments are forgotten (no pistachio came)."""
        self.reply = []
