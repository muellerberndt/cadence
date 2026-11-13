"""The parrot's focus: one thing attended among the things on its map (#333).

In the bird brain the optic tectum, with the isthmic nuclei, lets one place win: local feedback
holds the winner and a global inhibition silences the rest, and the winner turns the head. Here
the candidates are the things the map holds, in view or out of it. Each candidate's drive is how
much it stands out:

- **it appeared**: a thing that entered the map in the last moments;
- **it moved**: a thing whose place changed;
- **it is new to the eyes**: the longer since it was last in focus, the more it draws the eye
  (the tectum's return inhibited);
- **it holds**: the thing in focus keeps a share of drive while nothing changes, which falls
  as the parrot habituates to it;
- **it was heard**: a cue another part gives (a heard word's thing), at its own weight.

The candidates settle in one layer of neurons with mutual inhibition (the library's neuron rule);
the most active above ``floor`` is the focus, and the head turns so that the focus lies in front
of the beak, as far as the neck allows. With no thing on the map, the head looks to a new
direction now and then. Every weight here is a gene with this hand-set value as its control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(eq=False)
class Candidate:
    """A thing the focus remembers between moments: its kind and place, when it entered, when
    it moved and when it was last in focus."""

    kind: int
    place: np.ndarray
    entered: int
    moved: int = -1_000_000
    focused: int = -1_000_000
    held: int = 0  # moments in focus in a row
    heard: float = 0.0


@dataclass
class Choice:
    """The focus at one moment: the candidate's ``place`` and ``kind``, ``why`` it won (the
    largest term of its drive) and every candidate's settled ``activity``."""

    place: np.ndarray
    kind: int
    why: str
    index: int
    activity: np.ndarray = field(default_factory=lambda: np.zeros(0))


class Focus:
    """One winner among up to ``slots`` things on the map."""

    def __init__(self, slots: int = 16, *, appear: float = 1.0, move: float = 0.8,
                 novel: float = 0.6, hold: float = 1.2, habituate: float = 6.0,
                 recent: int = 4, return_time: float = 30.0, heard: float = 1.5,
                 inhibition: float = 2.0, floor: float = 0.05, same: float = 0.10,
                 backend: Any = "cpu"):
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model

        self.slots = int(slots)
        self.appear, self.move, self.novel, self.hold = (float(appear), float(move),
                                                          float(novel), float(hold))
        self.habituate, self.recent = float(habituate), int(recent)
        self.return_time, self.heard_weight = float(return_time), float(heard)
        self.inhibition, self.floor, self.same = float(inhibition), float(floor), float(same)
        n = self.slots
        pre, post = np.nonzero(~np.eye(n, dtype=bool))
        self.connectome = Connectome.from_synapses(n, label="focus", pre=pre, post=post,
                                                   populations={"focus": list(range(n))})
        self.graph = Graph(self.connectome, learning_neuron_model(dt=1.0),
                           efficacy=np.full(len(pre), -self.inhibition), backend=backend)
        self.candidates: list[Candidate] = []
        self.current: Candidate | None = None
        self.mask = np.ones(n)  # neurons held at rest inside the settle
        self.work = {"settles": 0, "macs": 0}

    def reset(self) -> None:
        self.candidates = []
        self.current = None

    def _match(self, things: list[dict[str, Any]], moment: int) -> list[Candidate]:
        """The map's things as candidates, each matched with the one remembered of its kind
        nearest it (within ``same``), or entered new."""
        out: list[Candidate] = []
        free = list(self.candidates)
        for t in things:
            if t["place"] is None or t["kind"] is None:
                continue
            place = np.asarray(t["place"], dtype=np.float64)
            best, gap = None, self.same
            for c in free:
                if c.kind != t["kind"]:
                    continue
                d = float(np.linalg.norm(c.place - place))
                if d <= gap:
                    best, gap = c, d
            if best is None:
                out.append(Candidate(int(t["kind"]), place, moment))
                continue
            free.remove(best)
            if gap > 0.5 * self.same:
                best.moved = moment
            best.place = place
            out.append(best)
        return out

    def choose(self, things: list[dict[str, Any]], moment: int,
               heard: dict[int, float] | None = None) -> Choice | None:
        """The focus among the map's ``things`` (``Map.things()``); ``heard`` gives a cue's
        weight by kind (a heard word's thing)."""
        self.candidates = self._match(things, moment)
        if self.current is not None and self.current not in self.candidates:
            self.current = None
        if not self.candidates:
            return None
        terms = np.zeros((len(self.candidates), 5))
        for i, c in enumerate(self.candidates):
            age = moment - c.entered
            terms[i, 0] = self.appear if 0 <= age < self.recent else 0.0
            terms[i, 1] = self.move if 0 <= moment - c.moved < self.recent else 0.0
            since = moment - c.focused
            terms[i, 2] = self.novel * (1.0 - math.exp(-since / self.return_time))
            if c is self.current:
                terms[i, 3] = self.hold * math.exp(-c.held / self.habituate)
            terms[i, 4] = self.heard_weight * float((heard or {}).get(c.kind, 0.0))
        drive = terms.sum(axis=1)
        order = np.argsort(-drive)[: self.slots]
        given = np.zeros((1, self.slots))
        given[0, : len(order)] = drive[order]
        state = self.graph.settle_batch(given * self.mask, steps=8, tolerance=None,
                                        mask=self.mask)
        self.work["settles"] += 1
        self.work["macs"] += 8 * int(self.connectome.synapses)
        activity = np.asarray(state.activation)[0, : len(order)]
        win = int(np.argmax(activity))
        if activity[win] <= self.floor:
            self.current = None
            return None
        index = int(order[win])
        chosen = self.candidates[index]
        if chosen is self.current:
            chosen.held += 1
        else:
            if self.current is not None:
                self.current.held = 0
            chosen.held = 1
        chosen.focused = moment
        self.current = chosen
        names = ("appeared", "moved", "new to the eyes", "held", "heard")
        return Choice(chosen.place.copy(), chosen.kind, names[int(terms[index].argmax())],
                      index, activity)

    def settings(self) -> dict[str, Any]:
        return {"slots": self.slots, "appear": self.appear, "move": self.move,
                "novel": self.novel, "hold": self.hold, "habituate": self.habituate,
                "recent": self.recent, "return_time": self.return_time,
                "heard": self.heard_weight, "inhibition": self.inhibition, "floor": self.floor,
                "same": self.same}


__all__ = ["Candidate", "Choice", "Focus"]
