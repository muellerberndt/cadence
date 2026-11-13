"""The lexicon's place taken by a table (cadence#326 audit): how much of the readings is the
settling lexicon's, and how much the program around it?

Three arms live the founder's script on one map and one body, each with its own lexicon:

- ``brain``: ``cadence.words.Words`` as the chamber makes it;
- ``table``: the same ``Words`` with the lexicon replaced by a count table (a control by the
  lane's rules: a lookup): a word's row is the running mean of the concept shares of its
  lessons, hearing sums the heard words' rows, saying is the entered word whose row is largest
  at the concept. Same lessons, same shares, same motor letters; no settle, no synapse;
- ``table_heard_back``: the table, saying as the head's lexicon says (a word is heard before it
  is said, and gives way where its hearing names another concept than the said vector's
  largest);
- ``oracle``: the same ``Words`` with a table written by hand from the world's own lexicon (each
  kind, colour and place word on its concept, "where" on the place read, "colour" on the colour
  read, "what" and "you" on the kind read, the nothing word on nothing) and no lesson at all:
  the program's ceiling.

``same`` is the share of questions the arm answers with the brain's very words.

    python lookup.py <founder> <out> [short]
"""

from __future__ import annotations

import sys
import time
from typing import Any

import common
import numpy as np
import world
from common import scored, tally


class Table:
    """A count table in the lexicon's place, with the lexicon's calls."""

    def __init__(self, concepts: dict[str, int], *, capacity: int = 256, rate: float = 0.05,
                 level: float = 0.5):
        self.groups: dict[str, slice] = {}
        start = 0
        for name, size in concepts.items():
            self.groups[name] = slice(start, start + int(size))
            start += int(size)
        self.concepts, self.capacity = start, int(capacity)
        self.rate, self.level = float(rate), float(level)
        self.rows = np.zeros((self.capacity, self.concepts))
        self.counts: dict[tuple[str, ...], np.ndarray] = {}
        self.forms: list[str] = []
        self._units: dict[str, int] = {}
        self.lessons = 0
        self.work = {"settles": 0, "macs": 0}
        self.fixed = False  # a hand-written table takes no lesson
        self.heard_back = False  # say as the lexicon of the head says: hear the word first

    def index(self, group: str, value: int) -> int:
        return int(self.groups[group].start) + int(value)

    def unit(self, word: str) -> int | None:
        return self._units.get(" ".join(str(word).lower().split()))

    def _enter(self, word: str) -> int:
        form = " ".join(str(word).lower().split())
        if form not in self._units:
            self._units[form] = len(self.forms)
            self.forms.append(form)
        return self._units[form]

    def _concepts(self, concepts: Any) -> np.ndarray:
        if not isinstance(concepts, dict):
            return np.asarray(concepts, dtype=np.float64).reshape(self.concepts).copy()
        out = np.zeros(self.concepts)
        for (group, value), share in concepts.items():
            out[self.index(group, value)] += float(share)
        return out

    def name(self, words: Any, concepts: Any, *, groups: Any = None) -> float:
        heard = [words] if isinstance(words, str) else list(words)
        units = sorted({self._enter(w) for w in heard})
        if self.fixed:
            return 0.0
        groups = tuple(self.groups) if groups is None else tuple(groups)
        cols = np.concatenate([np.arange(self.groups[g].start, self.groups[g].stop)
                               for g in groups])
        target = self.level * self._concepts(concepts)
        counts = self.counts.setdefault(groups, np.zeros(self.capacity, dtype=np.int64))
        for u in units:
            counts[u] += 1
            share = max(1.0 / counts[u], self.rate)
            self.rows[u, cols] += share * (target[cols] - self.rows[u, cols])
        self.lessons += 1
        return 0.0

    def hear(self, words: Any) -> np.ndarray:
        heard = [words] if isinstance(words, str) else list(words)
        out = np.zeros(self.concepts)
        for w in heard:
            u = self.unit(w)
            if u is not None:
                out += self.rows[u]
        return out

    def words_for(self, concept: Any) -> np.ndarray:
        if isinstance(concept, tuple):
            concept = self.index(*concept)
        if isinstance(concept, (int, np.integer)):
            return self.rows[: len(self.forms), int(concept)].copy()
        return self.rows[: len(self.forms)] @ np.asarray(concept, dtype=np.float64)

    def say(self, concept: Any, floor: float = 1e-3, tries: int = 3) -> str | None:
        active = self.words_for(concept)
        if not len(active) or active.max() <= floor:
            return None
        chosen = int(np.argmax(active))
        if self.heard_back:  # the lexicon's rule at the head: a word is heard before it is said
            vector = self._vector(concept)
            if vector.max() > 0:
                meant = int(np.argmax(vector))
                among = np.concatenate([np.arange(g.start, g.stop) for g in self.groups.values()
                                        if vector[g].max() > 0])
                for unit in np.argsort(-active)[: max(int(tries), 1)]:
                    if active[unit] <= floor:
                        break
                    heard = self.hear(self.forms[int(unit)])
                    if int(among[np.argmax(heard[among])]) == meant:
                        chosen = int(unit)
                        break
        return self.forms[chosen]

    def _vector(self, concept: Any) -> np.ndarray:
        if isinstance(concept, tuple):
            concept = self.index(*concept)
        if isinstance(concept, (int, np.integer)):
            return np.asarray(np.eye(self.concepts)[int(concept)])
        return np.asarray(concept, dtype=np.float64).reshape(self.concepts)


def heard_back(table: Table) -> Table:
    table.heard_back = True
    return table


def oracle(lex: world.Lexicon, concepts: dict[str, int]) -> Table:
    """The hand-written table of the world's lexicon."""
    out = Table(concepts)
    rows: list[tuple[str, tuple[str, int], float]] = []
    rows += [(w, ("kind", k), 1.0) for k, w in enumerate(lex.kind)]
    rows += [(w, ("colour", c), 1.0) for c, w in enumerate(lex.colour)]
    rows += [(w, ("place", p), 1.0) for p, w in enumerate(lex.place)]
    q = lex.question
    rows += [(q["where"], ("read", 2), 1.0), (q["colour"], ("read", 1), 2.0),
             (q["what"], ("read", 0), 1.0), (q["you"], ("read", 0), 1.0),
             (lex.none, ("none", 0), 1.0)]
    for w in (q["is"], q["the"]):
        out._enter(w)
    for word, concept, value in rows:
        out.rows[out._enter(word), out.index(*concept)] = value
    out.fixed = True
    return out


def life(seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    import words

    from cadence.words import Words

    settings = {**words.DEFAULTS, **settings}
    events = words.script(seed, **settings)
    brain = words.make_brain(seed)
    concepts = {g: s.stop - s.start for g, s in brain.lexicon.groups.items()}
    lex = world.lexicon(seed)
    arms = {"brain": brain,
            "table": Words(brain.map, lexicon=Table(concepts), body=brain.body),  # type: ignore[arg-type]
            "table_heard_back": Words(brain.map, lexicon=heard_back(Table(concepts)),  # type: ignore[arg-type]
                                      body=brain.body),
            "oracle": Words(brain.map, lexicon=oracle(lex, concepts),  # type: ignore[arg-type]
                            body=brain.body)}
    h = common.Harness(seed, brain, settings)
    books: dict[str, dict[str, Any]] = {name: {} for name in arms}
    began = time.time()
    r = None
    for event in events:
        kind = event[0]
        if kind == "room":
            r = event[1]
            words.look_around(brain, r)
        elif kind == "turn":
            words.turn(brain, r, event[1])
        elif kind == "name":
            for arm in arms.values():
                arm.name(event[2], event[1])
        elif kind == "practice":
            q = event[1]
            for arm in arms.values():
                arm.practice(q.text, q.answer if len(q.answer) > 1 else q.answer[0])
        else:
            gate, q = event[1], event[2]
            said = {name: arm.ask(q.text) for name, arm in arms.items()}
            for name in arms:
                tally(books[name], gate, scored(h, q, said[name], base=said["brain"]))
    return {"seed": int(seed), "settings": settings, "seconds": round(time.time() - began, 1),
            "parameters": {"brain": int(brain.lexicon.connectome.synapses // 2),
                           "table": int(arms["table"].lexicon.rows.size),
                           "table_heard_back": int(arms["table"].lexicon.rows.size),
                           "oracle": 0},
            "arms": {name: {"shares": common.shares(book), "gates": book}
                     for name, book in books.items()}}


def main(argv: list[str]) -> None:
    seed, out, settings = common.arguments(argv)
    print(common.write(out, "lookup", seed, life(seed, settings)))


if __name__ == "__main__":
    main(sys.argv)
