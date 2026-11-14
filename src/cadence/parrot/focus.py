"""The parrot's focus: one thing attended among the things on its map (#333).

In the bird brain the optic tectum, with the isthmic nuclei, lets one place win: local feedback
holds the winner and a global inhibition silences the rest, and the winner turns the head. Here
the candidates are the things the map holds, in view or out of it. Each candidate's drive is how
much it stands out:

- **it is in view**: a thing the eyes report at this moment;
- **it is pointed at**: the visitor points (clicks) toward a thing, and the focus goes to the
  thing nearest that direction at once (joint attention);
- **it appeared**: a thing that entered the map in the last moments where none of its kind was
  held for a while (a thing out of the map for a few moments keeps its identity);
- **it moved**: how far its place moved this moment against its own usual move (the surprise
  rule of #337: a thing at rest that starts to move draws the eye; one that keeps moving
  habituates); a change across the brain's own head movement is no surprise (the efference
  copy, as saccades are not seen as the world moving);
- **it is new to the eyes**: the longer since it was last in focus, the more it draws the eye
  (the tectum's return inhibited);
- **it holds**: the thing in focus keeps a share of drive while nothing changes, which falls
  as the parrot habituates to it;
- **it was heard**: a cue another part gives (a heard word's thing), at its own weight.

The candidates settle in one layer of neurons with mutual inhibition (the library's neuron rule;
the inhibition a pair is the global inhibition shared among the rivals, so the layer settles
with the most driven candidate most active whatever their number); the most active above
``floor`` is the focus, and the head turns so that the focus lies in front
of the beak, as far as the neck allows. With no thing on the map, the head looks to a new
direction at times. Every weight here is a gene with this hand-set value as its control.
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
    usual: float = 0.0  # the thing's usual move a moment, learned (metres)
    surprise: float = 0.0  # this moment's move against its usual move
    seen: bool = False
    seen_at: int = 0  # the last moment the map held it
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

    def __init__(self, slots: int = 16, *, appear: float = 1.0, move: float = 2.0,
                 novel: float = 0.6, hold: float = 1.2, habituate: float = 6.0,
                 recent: int = 4, return_time: float = 30.0, heard: float = 1.5,
                 inhibition: float = 2.0, floor: float = 0.05, same: float = 0.10,
                 view: float = 0.5, point: float = 3.0, point_angle: float = 12.0,
                 noise: float = 0.01, usual_rate: float = 0.3, absence: int = 30,
                 return_reach: float = 0.25, still_turn: float = 1.0, after_turn: int = 3,
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
        self.view, self.point = float(view), float(point)
        self.point_angle = math.radians(float(point_angle))
        self.noise, self.usual_rate = float(noise), float(usual_rate)
        self.absence, self.return_reach = int(absence), float(return_reach)
        self.still_turn, self.after_turn = float(still_turn), int(after_turn)
        self.turned_at = -1_000_000
        n = self.slots
        pre, post = np.nonzero(~np.eye(n, dtype=bool))
        self.connectome = Connectome.from_synapses(n, label="focus", pre=pre, post=post,
                                                   populations={"focus": list(range(n))})
        # the global inhibition shared among the rivals: a pair inhibits at its share
        self.graph = Graph(self.connectome, learning_neuron_model(dt=0.5),
                           efficacy=np.full(len(pre), -self.inhibition / max(n - 1, 1)),
                           backend=backend)
        self.steps = 24
        self.candidates: list[Candidate] = []  # the things the map holds at this moment
        self.remembered: list[Candidate] = []  # things out of the map for a short while
        self.current: Candidate | None = None
        self.mask = np.ones(n)  # neurons held at rest inside the settle
        self.drive = np.zeros((1, n))
        self.work = {"settles": 0, "macs": 0}

    def reset(self) -> None:
        self.candidates = []
        self.remembered = []
        self.current = None
        self.turned_at = -1_000_000

    def _match(self, things: list[dict[str, Any]], moment: int,
               turned: float = 0.0) -> list[Candidate]:
        """The map's things as candidates, each matched with the one remembered of its kind
        nearest it, or entered new. A surprise is a change the brain did not cause and could
        not predict:

        - the nearest remembered thing of its kind within ``return_reach`` is the same thing,
          also through an absence of up to ``absence`` moments, so only a thing never held
          near there appears;
        - a move measured across the brain's own head movement (``turned``, degrees in the
          moment before: the efference copy) or in the ``after_turn`` moments after it, while
          the eyes' estimate settles on the new view, is no surprise, nor is a return after an
          absence;
        - any other move is read against the thing's usual move, which learns its own jitter
          (``usual_rate``) above a born floor (``noise``)."""
        out: list[Candidate] = []
        known = list(self.candidates) + list(self.remembered)
        free = list(known)
        if abs(float(turned)) > self.still_turn:
            self.turned_at = moment
        # the brain's own movement, and the few moments after it while the eyes' estimate of a
        # place settles on the new view (the efference copy)
        own = moment - self.turned_at <= self.after_turn
        for t in things:
            if t["place"] is None or t["kind"] is None:
                continue
            place = np.asarray(t["place"], dtype=np.float64)
            seen = t.get("state") == "seen"
            best, gap = None, math.inf
            for c in free:
                if c.kind != t["kind"]:
                    continue
                d = float(np.linalg.norm(c.place - place))
                # the nearest of its kind within ``return_reach`` is the same thing: it may have
                # moved, or the map may hold it a little elsewhere after an absence or a turn
                if d <= self.return_reach and d < gap:
                    best, gap = c, d
            if best is None:
                # a place beside a thing of its kind the brain knows is no new thing
                beside = any(c.kind == t["kind"] and float(np.linalg.norm(c.place - place))
                             <= self.return_reach for c in known)
                out.append(Candidate(int(t["kind"]), place, -1_000_000 if beside else moment,
                                     seen=seen, seen_at=moment))
                continue
            free.remove(best)
            if own or best.seen_at < moment - 1:
                best.surprise = 0.0
            else:
                # the move against the thing's usual move (#337's surprise rule)
                best.surprise = gap / (best.usual + self.noise)
                best.usual += self.usual_rate * (gap - best.usual)
                if best.surprise > 3.0 and gap > 0.5 * self.same:
                    best.moved = moment
            best.place, best.seen, best.seen_at = place, seen, moment
            out.append(best)
        self.remembered = [c for c in free if moment - c.seen_at <= self.absence]
        return out

    def choose(self, things: list[dict[str, Any]], moment: int,
               heard: dict[int, float] | None = None,
               pointed: tuple[np.ndarray, np.ndarray] | None = None,
               turned: float = 0.0) -> Choice | None:
        """The focus among the map's ``things`` (``Map.things()``); ``heard`` gives a cue's
        weight by kind (a heard word's thing); ``pointed`` is a ray on the table (an origin
        and a direction) the visitor points along: the thing nearest it, within
        ``point_angle``, is drawn at once; ``turned`` is how far the head turned in the moment
        before (degrees, the brain's own movement)."""
        self.candidates = self._match(things, moment, turned)
        if self.current is not None and self.current not in self.candidates:
            self.current = None
        if not self.candidates:
            return None
        target = None
        if pointed is not None:
            origin, ray = (np.asarray(v, dtype=np.float64) for v in pointed)
            ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
            best_angle = self.point_angle
            for i, c in enumerate(self.candidates):
                to = c.place - origin
                angle = math.atan2(abs(float(to[0] * ray[1] - to[1] * ray[0])),
                                   float(to @ ray))
                if angle < best_angle:
                    target, best_angle = i, angle
        terms = np.zeros((len(self.candidates), 7))
        for i, c in enumerate(self.candidates):
            age = moment - c.entered
            terms[i, 0] = self.appear if 0 <= age < self.recent else 0.0
            terms[i, 1] = self.move if 0 <= moment - c.moved < self.recent else 0.0
            since = moment - c.focused
            terms[i, 2] = self.novel * (1.0 - math.exp(-since / self.return_time))
            if c is self.current:
                terms[i, 3] = self.hold * math.exp(-c.held / self.habituate)
            terms[i, 4] = self.heard_weight * float((heard or {}).get(c.kind, 0.0))
            terms[i, 5] = self.view if c.seen else 0.0
            terms[i, 6] = self.point if i == target else 0.0
        drive = terms.sum(axis=1)
        order = np.argsort(-drive)[: self.slots]
        given = np.zeros((1, self.slots))
        given[0, : len(order)] = drive[order]
        self.drive = given * self.mask  # the latest settle's drive, for a reader
        state = self.graph.settle_batch(given * self.mask, steps=self.steps, tolerance=None,
                                        mask=self.mask)
        self.work["settles"] += 1
        self.work["macs"] += self.steps * int(self.connectome.synapses)
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
        names = ("appeared", "moved", "new to the eyes", "held", "heard", "in view",
                 "pointed at")
        return Choice(chosen.place.copy(), chosen.kind, names[int(terms[index].argmax())],
                      index, activity)

    def settings(self) -> dict[str, Any]:
        return {"slots": self.slots, "appear": self.appear, "move": self.move,
                "novel": self.novel, "hold": self.hold, "habituate": self.habituate,
                "recent": self.recent, "return_time": self.return_time,
                "heard": self.heard_weight, "inhibition": self.inhibition, "floor": self.floor,
                "same": self.same, "view": self.view, "point": self.point,
                "point_angle": math.degrees(self.point_angle), "noise": self.noise,
                "usual_rate": self.usual_rate, "absence": self.absence,
                "return_reach": self.return_reach, "still_turn": self.still_turn,
                "after_turn": self.after_turn}


__all__ = ["Candidate", "Choice", "Focus"]
