"""The grammar chamber's language at any vocabulary, by class.

``Language(nouns, verbs)`` holds ``nouns`` noun stems in the singular and the plural, ``verbs``
verbs in the singular and the plural and the chamber's six function words (``near``, four
brackets, ``end``). Its input holds one coordinate per word and the concepts of the meaning: one
per noun stem and verb, and the chamber's eleven flags and bracket concepts. ``Language(4, 2)`` is
the chamber of ``benchmarks/grammar/grammar.py`` to the bit: the same coordinates and, from the
same seeds, the same sentences (a test holds it).

Moments are kept as tuples of the coordinates that are on, because at thousands of words a dense
copy of every moment of every panel holds gigabytes; ``dense`` gives a batch's array.

``fresh`` stems stay out of training and out of the ordinary panels: the one-hearing reading
(``hearing.py``) names them. ``verb_flag`` adds a ``VERB`` flag to the meaning's verb moment, the
class flag the meaning's nouns carry through their roles.

Sizes by class, ``by_size``: the words are the chamber's mix, two noun stems to a verb.

=====  =====  =====  ==========  ======  =========================
words  nouns  verbs  function    inputs  actions (one per word)
=====  =====  =====  ==========  ======  =========================
18     4      2      6           35      18
200    65     32     6           308     200
1000   331    166    6           1508    1000
5000   1665   832    6           7508    5000
=====  =====  =====  ==========  ======  =========================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BATCH = 16
SEEDS = {"train": 11, "test": 33, "split": 7}
TRAIN_L, TEST_L = (0, 1, 2), (3, 4)  # distractor phrases in a clause
TRAIN_DEPTH, TEST_DEPTH = 3, (4, 5)  # the deepest nesting of a Dyck sentence
TRAIN_HOPS, TEST_HOPS = (1, 2), (3,)  # facts chained by a path sentence
FAMILIES = {"clause": 0.5, "dyck": 0.25, "path": 0.25}

Moment = tuple[int, ...]


def by_size(words: int) -> tuple[int, int]:
    """The noun stems and verbs of a language of ``words`` words: two stems to a verb."""
    if words == 18:
        return 4, 2
    pairs = (words - 6) // 2
    verbs = pairs // 3
    return pairs - verbs, verbs


def mix(*parts: int) -> int:
    """A deterministic integer hash, for the split's membership at any size."""
    h = 1469598103934665603
    for part in parts:
        h ^= int(part) & 0xFFFFFFFF
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 29
    return h


@dataclass(frozen=True)
class Split:
    """What one founder's training never shows. ``Language(4, 2)``: the chamber's two
    agent-number-verb triples and one noun pair, which hold about one triple in eight and one
    pair in six. Any larger language holds the same shares by a hash of the founder's salt."""

    triples: tuple[tuple[int, int, int], ...] = ()
    pair: tuple[int, int] = (0, 0)
    salt: int | None = None  # None: the chamber's explicit form

    def held_triple(self, agent: tuple[int, int], v: int) -> bool:
        if self.salt is None:
            return (agent[0], agent[1], v) in self.triples
        return mix(self.salt, 1, agent[0], agent[1], v) % 8 == 0

    def held_pair(self, p: int, q: int) -> bool:
        if p == q:
            return False
        if self.salt is None:
            return {p, q} == set(self.pair)
        return mix(self.salt, 2, min(p, q), max(p, q)) % 6 == 0

    def held(self, agent: tuple[int, int], patient: tuple[int, int], v: int) -> bool:
        return self.held_triple(agent, v) or self.held_pair(agent[0], patient[0])


@dataclass
class Row:
    """One stream's episode: heard moments, then the sentence from ``GO``."""

    family: str
    pre: list[Moment]
    target: list[int]
    meta: dict[str, Any] = field(default_factory=dict)
    force: int = 0  # the target's first words given as heard (the read starts after them)


class Language:
    """The chamber's three families and two grammars over a vocabulary of this size."""

    def __init__(self, nouns: int = 4, verbs: int = 2, fresh: int = 0,
                 verb_flag: bool = False) -> None:
        if not 0 <= fresh < nouns - 1:
            raise ValueError("a language trains at least two noun stems")
        self.nouns, self.verbs, self.fresh = int(nouns), int(verbs), int(fresh)
        self.trained = self.nouns - self.fresh  # the stems training and the panels draw from
        self.spoken_verbs = self.verbs  # the verbs they draw from (``grown``)
        n, v = self.nouns, self.verbs
        self.verb0 = 2 * n
        self.near = 2 * n + 2 * v
        self.open_round, self.close_round = self.near + 1, self.near + 2
        self.open_square, self.close_square = self.near + 3, self.near + 4
        self.end = self.near + 5
        self.words = self.end + 1
        self.opens = (self.open_round, self.open_square)
        self.match = {self.open_round: self.close_round, self.open_square: self.close_square}
        base = self.words
        self.noun_concept0 = base
        self.plural = base + n
        self.agent, self.patient, self.mod = base + n + 1, base + n + 2, base + n + 3
        self.verb_concept0 = base + n + 4
        after = base + n + 4 + v
        self.c_open_round, self.c_open_square, self.c_close = after, after + 1, after + 2
        self.c_from, self.dialect, self.go, self.speak = after + 3, after + 4, after + 5, after + 6
        self.inputs = after + 7
        self.verb_flag = None
        if verb_flag:
            self.verb_flag = self.inputs
            self.inputs += 1
        self.actions = self.words
        self.c_opens = (self.c_open_round, self.c_open_square)

    def grown(self, nouns: int, verbs: int) -> None:
        """The stems and verbs a sentence draws from from here on: a vocabulary that grows in a
        life, its words on their coordinates from the start."""
        self.trained = min(int(nouns), self.nouns - self.fresh)
        self.spoken_verbs = min(int(verbs), self.verbs)

    # ------------------------------------------------------------------------------ moments

    def noun(self, i: int, plural: int) -> int:
        return int(i) + self.nouns * int(plural)

    def verb(self, v: int, plural: int) -> int:
        return self.verb0 + 2 * int(v) + int(plural)

    def is_noun(self, w: int) -> bool:
        return 0 <= w < self.verb0

    def is_verb(self, w: int) -> bool:
        return self.verb0 <= w < self.near

    def heard(self, word: int) -> Moment:
        return (int(word),)

    def spoken(self, word: int) -> Moment:
        return (int(word), self.speak)

    def noun_moment(self, i: int, plural: int, *flags: int) -> Moment:
        return (self.noun_concept0 + int(i), *((self.plural,) if plural else ()), *flags)

    def dense(self, moments: list[Moment]) -> np.ndarray:
        x = np.zeros((len(moments), self.inputs))
        for row, moment in enumerate(moments):
            x[row, list(moment)] = 1.0
        return x

    def say(self, words: list[int]) -> str:
        return " ".join(self.word_name(w) for w in words)

    def word_name(self, w: int) -> str:
        if self.is_noun(w):
            return f"n{w % self.nouns}{'s' if w >= self.nouns else ''}"
        if self.is_verb(w):
            k = w - self.verb0
            return f"v{k // 2}{'p' if k % 2 else ''}"
        return {self.near: "near", self.open_round: "(", self.close_round: ")",
                self.open_square: "[", self.close_square: "]", self.end: "end"}.get(w, "?")

    # -------------------------------------------------------------------------------- split

    def split(self, founder: int) -> Split:
        rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["split"]]))
        if self.trained == 4 and self.verbs == 2:
            a, b = (int(i) for i in rng.permutation(4)[:2])
            triples = ((a, 1, 0), (b, 0, 1))
            p, q = (int(i) for i in rng.permutation(4)[:2])
            return Split(triples, (min(p, q), max(p, q)))
        return Split(salt=int(rng.integers(2**31)))

    def held_pair(self, rng: np.random.Generator, split: Split) -> tuple[int, int]:
        """A noun pair training never shows together: the chamber's one, or a draw."""
        if split.salt is None:
            return split.pair
        while True:
            p, q = (int(i) for i in rng.integers(self.trained, size=2))
            if split.held_pair(p, q):
                return min(p, q), max(p, q)

    # ------------------------------------------------------------------------------ families

    def clause(self, rng: np.random.Generator, L: int, *, agent: tuple[int, int] | None = None,
               patient: tuple[int, int] | None = None, v: int | None = None,
               mods: list[tuple[int, int]] | None = None, inverted: bool | None = None,
               dialect: bool = False) -> Row:
        agent = agent or (int(rng.integers(self.trained)), int(rng.integers(2)))
        patient = patient or (int(rng.integers(self.trained)), int(rng.integers(2)))
        v = int(rng.integers(self.spoken_verbs)) if v is None else v
        mods = mods if mods is not None else [(int(rng.integers(self.trained)),
                                               int(rng.integers(2))) for _ in range(L)]
        inverted = bool(rng.integers(2)) if inverted is None else inverted
        act: Moment = (self.verb_concept0 + v, *((self.dialect,) if dialect else ()),
                       *(() if self.verb_flag is None else (self.verb_flag,)))
        a = self.noun_moment(*agent, self.agent)
        p = self.noun_moment(*patient, self.patient)
        m = [self.noun_moment(*mod, self.mod) for mod in mods]
        pre = [act, p, a, *m] if inverted else [act, a, *m, p]
        words = [self.noun(*agent)]
        for mod in mods:
            words += [self.near, self.noun(*mod)]
        tail = ([self.noun(*patient), self.verb(v, agent[1])] if dialect
                else [self.verb(v, agent[1]), self.noun(*patient)])
        meta = {"agent": list(agent), "patient": list(patient), "verb": v,
                "mods": [list(x) for x in mods], "L": len(mods), "inverted": inverted,
                "dialect": dialect,
                "attract": any(x[1] != agent[1] for x in mods),
                "patient_attract": patient[1] != agent[1]}
        return Row("clause", pre, words + tail + [self.end], meta)

    def trained_clause(self, rng: np.random.Generator, split: Split, L: int,
                       dialect: bool) -> Row:
        while True:
            row = self.clause(rng, L, dialect=dialect)
            m = row.meta
            if not split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"]):
                return row

    @staticmethod
    def dyck_pattern(rng: np.random.Generator, length: int, max_depth: int,
                     min_depth: int = 1, nested: bool | None = None) -> list[int]:
        """A prefix of opens (1) and closes (0) whose depth stays within ``max_depth``, ends
        open, and reaches ``min_depth``."""
        while True:
            if nested:
                return [1] * length
            depth, seq, deepest = 0, [], 0
            for _ in range(length):
                can_open, can_close = depth < max_depth, depth > 0
                step = 1 if not can_close else 0 if not can_open else int(rng.random() < 0.6)
                depth += 1 if step else -1
                deepest = max(deepest, depth)
                seq.append(step)
            if depth > 0 and deepest >= min_depth and (nested is None or (0 in seq) != nested):
                return seq

    def dyck(self, rng: np.random.Generator, pattern: list[int],
             types: list[int] | None = None) -> Row:
        types = types if types is not None else [int(rng.integers(2)) for _ in pattern]
        pre: list[Moment] = []
        words: list[int] = []
        stack: list[int] = []
        for step, t in zip(pattern, types, strict=True):
            if step:
                pre.append((self.c_opens[t],))
                words.append(self.opens[t])
                stack.append(self.opens[t])
            else:
                pre.append((self.c_close,))
                words.append(self.match[stack.pop()])
        depth, deepest = 0, 0
        for step in pattern:
            depth += 1 if step else -1
            deepest = max(deepest, depth)
        meta = {"prefix": len(pattern), "depth": deepest, "open": len(stack),
                "nested": 0 not in pattern, "types": [int(t) for t in types]}
        return Row("dyck", pre, words + [self.match[o] for o in reversed(stack)] + [self.end],
                   meta)

    def path(self, rng: np.random.Generator, hops: int, distractors: int = 2) -> Row:
        """Facts ``X near Y end`` with one chain of ``hops`` from the start to a sink."""
        entities = 2 * self.trained
        order = [int(i) for i in rng.permutation(entities)]
        order = [self._entity(e) for e in order]
        chain, rest = order[:hops + 1], order[hops + 1:]
        facts = [(chain[i], chain[i + 1]) for i in range(hops)]
        firsts = [int(i) for i in rng.permutation(rest)[:distractors]]
        for f in firsts:
            others = [self._entity(e) for e in range(entities) if self._entity(e) != f]
            facts.append((f, int(rng.choice(others))))
        pre: list[Moment] = []
        for i in rng.permutation(len(facts)):
            x, y = facts[int(i)]
            pre += [self.heard(x), self.heard(self.near), self.heard(y), self.heard(self.end)]
        start = chain[0]
        pre.append(self.noun_moment(start % self.nouns, start // self.nouns, self.c_from))
        words = [chain[0]]
        for e in chain[1:]:
            words += [self.near, e]
        return Row("path", pre, words + [self.end],
                   {"hops": hops, "chain": chain, "facts": len(facts)})

    def _entity(self, e: int) -> int:
        """The word of the ``e``-th trained noun form: singulars, then plurals."""
        return e if e < self.trained else e - self.trained + self.nouns

    def train_batch(self, rng: np.random.Generator, split: Split, grammar: str = "g1",
                    size: int = BATCH) -> list[Row]:
        """One training batch: one family and one shape, streams of their own content."""
        if grammar == "g2":
            L = int(rng.choice(TRAIN_L))
            return [self.trained_clause(rng, split, L, True) for _ in range(size)]
        family = str(rng.choice(list(FAMILIES), p=list(FAMILIES.values())))
        if family == "clause":
            L = int(rng.choice(TRAIN_L))
            return [self.trained_clause(rng, split, L, False) for _ in range(size)]
        if family == "dyck":
            pattern = self.dyck_pattern(rng, int(rng.integers(1, 7)), TRAIN_DEPTH)
            return [self.dyck(rng, pattern) for _ in range(size)]
        hops = int(rng.choice(TRAIN_HOPS))
        return [self.path(rng, hops) for _ in range(size)]

    def solve(self, pre: list[Moment]) -> list[int]:
        """The sentence an exact solver says from the heard moments alone: the instrument's
        check that every sentence is answerable from what the brain hears."""
        on = [set(moment) for moment in pre]

        def named(moment: set[int]) -> int:
            stem = next(c - self.noun_concept0 for c in moment
                        if self.noun_concept0 <= c < self.plural)
            return self.noun(stem, self.plural in moment)

        if any(self.c_from in o for o in on):
            here = named(on[-1])
            words = [next(iter(o)) for o in on[:-1]]
            facts = {words[i]: words[i + 2] for i in range(0, len(words), 4)}
            said = [here]
            while here in facts:
                here = facts[here]
                said += [self.near, here]
            return said + [self.end]
        if any(o & {*self.c_opens, self.c_close} for o in on):
            said, stack = [], []
            for o in on:
                if self.c_close in o:
                    said.append(self.match[stack.pop()])
                else:
                    stack.append(self.open_round if self.c_open_round in o else self.open_square)
                    said.append(stack[-1])
            return said + [self.match[o] for o in reversed(stack)] + [self.end]
        v = next(c - self.verb_concept0 for c in on[0]
                 if self.verb_concept0 <= c < self.verb_concept0 + self.verbs)
        nouns, mods = {}, []
        for o in on[1:]:
            word = named(o)
            if self.mod in o:
                mods.append(word)
            else:
                nouns["agent" if self.agent in o else "patient"] = word
        agent, patient = nouns["agent"], nouns["patient"]
        said = [agent]
        for m in mods:
            said += [self.near, m]
        form = self.verb(v, agent >= self.nouns)
        tail = [patient, form] if self.dialect in on[0] else [form, patient]
        return said + tail + [self.end]

    # -------------------------------------------------------------------------------- panels

    def twin(self, row: Row, **change: Any) -> Row:
        """The clause of ``row`` with one part of its meaning changed."""
        m = row.meta
        spec = {"agent": tuple(m["agent"]), "patient": tuple(m["patient"]), "v": m["verb"],
                "mods": [tuple(x) for x in m["mods"]], "inverted": m["inverted"],
                "dialect": m["dialect"]}
        spec.update(change)
        return self.clause(np.random.default_rng(0), len(spec["mods"]), **spec)

    @staticmethod
    def removed(row: Row, keep: int) -> Row:
        """The context removed: no meaning, ``GO``, then the recent words as heard."""
        recent = row.target[row.force - keep:]
        return Row(row.family, [], recent, {**row.meta, "removed": True}, force=keep)

    def panels(self, founder: int, batches: int = 4) -> dict[str, list[list[Row]]]:
        """The founder's frozen test panels, each ``batches`` batches of one shape."""
        split = self.split(founder)
        rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["test"]]))
        out: dict[str, list[list[Row]]] = {}

        def held_clause(L: int) -> Row:
            while True:
                row = self.clause(rng, L)
                m = row.meta
                if split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"]):
                    return row

        def train_shape(i: int) -> int:
            return TRAIN_L[i % len(TRAIN_L)]

        out["clause"] = [[self.trained_clause(rng, split, train_shape(i), False)
                          for _ in range(BATCH)] for i in range(batches)]
        out["clause_held"] = [[held_clause(train_shape(i)) for _ in range(BATCH)]
                              for i in range(batches)]
        out["clause_long"] = [[self.trained_clause(rng, split, TEST_L[i % len(TEST_L)], False)
                               for _ in range(BATCH)] for i in range(batches)]
        roles = []  # a held pair in both roles, side by side: the same words, the roles swapped
        for i in range(batches):
            rows = []
            for _ in range(BATCH // 2):
                p, q = self.held_pair(rng, split)
                base = self.clause(rng, train_shape(i), agent=(p, int(rng.integers(2))),
                                   patient=(q, int(rng.integers(2))))
                if rng.random() < 0.5:
                    base = self.twin(base, agent=tuple(base.meta["patient"]),
                                     patient=tuple(base.meta["agent"]))
                rows += [base, self.twin(base, agent=tuple(base.meta["patient"]),
                                         patient=tuple(base.meta["agent"]))]
            roles.append(rows)
        out["roles"] = roles
        out["g2"] = [[self.trained_clause(rng, split, train_shape(i), True)
                      for _ in range(BATCH)] for i in range(batches)]
        out["g2_long"] = [[self.trained_clause(rng, split, TEST_L[i % len(TEST_L)], True)
                           for _ in range(BATCH)] for i in range(batches)]
        out["dyck"] = []
        out["dyck_deep"] = []
        for i in range(batches):
            nested = i % 2 == 0
            length = int(rng.integers(1 if nested else 3, 4 if nested else 7))
            pattern = self.dyck_pattern(rng, length, TRAIN_DEPTH, nested=nested)
            out["dyck"].append([self.dyck(rng, pattern) for _ in range(BATCH)])
            depth = TEST_DEPTH[(i // 2) % len(TEST_DEPTH)]
            length = depth if nested else int(rng.integers(depth + 2, depth + 5))
            pattern = self.dyck_pattern(rng, length, depth, min_depth=depth, nested=nested)
            out["dyck_deep"].append([self.dyck(rng, pattern) for _ in range(BATCH)])
        out["path"] = [[self.path(rng, TRAIN_HOPS[i % len(TRAIN_HOPS)]) for _ in range(BATCH)]
                       for i in range(batches)]
        out["path_long"] = [[self.path(rng, TEST_HOPS[0]) for _ in range(BATCH)]
                            for _ in range(batches)]

        # the context: twins share their recent words and differ in an earlier part
        for name, change in (("ctx_agent", "agent"), ("ctx_verb", "verb")):
            rows_all = []
            for i in range(batches):
                L = 1 + i % 2
                rows = []
                for _ in range(BATCH // 2):
                    base = self.trained_clause(rng, split, L, False)
                    m = base.meta
                    for _ in range(64):
                        other = (self.twin(base, agent=(m["agent"][0], 1 - m["agent"][1]))
                                 if change == "agent" else
                                 self.twin(base, v=self._other_verb(m["verb"])))
                        om = other.meta
                        if not split.held(tuple(om["agent"]), tuple(om["patient"]), om["verb"]):
                            break
                        base = self.trained_clause(rng, split, L, False)
                        m = base.meta
                    for r in (base, other):
                        r.force = 1 + 2 * L  # the agent and the distractors are given
                    rows += [base, other]
                rows_all.append(rows)
            out[name] = rows_all
            out[name + "_removed"] = [[self.removed(r, 2 * (1 + i % 2)) for r in rows]
                                      for i, rows in enumerate(rows_all)]
        rows_all = []
        for i in range(batches):
            depth = 2 + i % 2
            rows = []
            for _ in range(BATCH // 2):
                types = [int(t) for t in rng.integers(2, size=depth)]
                for first in (types[0], 1 - types[0]):
                    r = self.dyck(rng, [1] * depth, [first, *types[1:]])
                    r.force = depth  # the opens are given; the closes are the brain's
                    rows.append(r)
            rows_all.append(rows)
        out["ctx_dyck"] = rows_all
        out["ctx_dyck_removed"] = [[self.removed(r, r.force - 1) for r in rows]
                                   for rows in rows_all]
        return out

    def _other_verb(self, v: int) -> int:
        """The verb a context twin says instead: the chamber's other verb; at any size the
        next one (the twins differ by one meaning coordinate, whichever verbs they are)."""
        return 1 - v if self.verbs == 2 else (v + 1) % self.verbs

    # ------------------------------------------------------------------------------- scoring

    def score(self, row: Row, said: list[int]) -> dict[str, Any]:
        """The whole sentence said, from its first word through ``end``."""
        target = row.target[row.force:]
        out: dict[str, Any] = {"exact": said == target}
        if row.meta.get("removed"):
            return out
        end = self.end
        words = said[:-1] if said and said[-1] == end else said
        ended = bool(said) and said[-1] == end
        v0, n = self.verb0, self.nouns
        if row.family == "clause" and row.force == 0:
            m = row.meta
            verbs = [i for i, w in enumerate(words) if self.is_verb(w)]
            one = len(verbs) == 1
            k = verbs[0] if one else -1
            out["agree"] = one and (words[k] - v0) % 2 == m["agent"][1]
            out["verb"] = one and (words[k] - v0) // 2 == m["verb"]
            subject = words[0] if words and self.is_noun(words[0]) else None
            out["agree_own"] = (one and subject is not None
                                and (words[k] - v0) % 2 == subject // n)
            before = words[k - 1] if one and k > 0 and self.is_noun(words[k - 1]) else None
            out["attracted"] = (None if before is None or before // n == m["agent"][1]
                                else (words[k] - v0) % 2 == before // n)
            out["subject"] = subject == self.noun(*m["agent"])
            patient_at = (k - 1) if m["dialect"] else (k + 1)
            out["patient"] = one and 0 <= patient_at < len(words) and words[patient_at] == (
                self.noun(*m["patient"]))
            out["roles"] = out["subject"] and out["patient"]
            mods = [w for pair in m["mods"] for w in (self.near, self.noun(*pair))]
            out["mods"] = words[1:1 + len(mods)] == mods if mods else None
            out["ended"] = ended
        elif row.family == "clause":  # a context continuation: the verb, the patient, end
            out["agree"] = (bool(said) and self.is_verb(said[0])
                            and (said[0] - v0) % 2 == row.meta["agent"][1])
            out["verb"] = (bool(said) and self.is_verb(said[0])
                           and (said[0] - v0) // 2 == row.meta["verb"])
        elif row.family == "dyck":
            stack: list[int] = []
            valid = ended
            for w in row.target[:row.force] + words:
                if w in self.opens:
                    stack.append(w)
                elif w in (self.close_round, self.close_square) and stack and (
                        self.match[stack[-1]] == w):
                    stack.pop()
                else:
                    valid = False
            out["valid"] = valid and not stack and len(words) == len(target) - 1
            if row.force == 0:
                out["prefix"] = words[:row.meta["prefix"]] == row.target[:row.meta["prefix"]]
        elif row.family == "path":
            chain = row.meta["chain"]
            out["hops"] = sum(1 for h in range(1, len(chain))
                              if len(words) > 2 * h and words[2 * h] == chain[h]
                              and words[2 * h - 1] == self.near and all(
                                  words[2 * g] == chain[g] for g in range(h)))
            out["first_hop"] = len(words) > 2 and words[2] == chain[1]
        return out

    def tally(self, rows: list[Row], said: list[list[int]]) -> dict[str, Any]:
        scores = [self.score(r, s) for r, s in zip(rows, said, strict=True)]
        out: dict[str, Any] = {"n": len(scores)}
        for key in scores[0]:
            kept = [s[key] for s in scores if s.get(key) is not None]
            out[key] = float(np.mean(kept)) if kept else None
        return out

    @staticmethod
    def conditions(r: Row) -> list[str]:
        m = r.meta
        if r.family == "clause" and "L" in m:
            keys = [f"L{m['L']}"]
            if r.force == 0:
                keys += ["attract" if m["attract"] else "plain",
                         "inverted" if m["inverted"] else "canonical"]
            return keys
        if r.family == "dyck":
            return [f"depth{m['depth']}", "nested" if m["nested"] else "mixed"]
        if r.family == "path":
            return [f"hops{m['hops']}"]
        return []

    def read_speaker(self, speaker: Any, next_words: Any, batches: list[list[Row]], *,
                     twins: bool = False) -> dict[str, Any]:
        """The panel's scores for any speaker: ``speaker(rows)`` returns the words each row
        says from its ``force`` on, ``next_words(rows)`` (or None) the teacher-forced hits."""
        rows_all: list[Row] = []
        said_all: list[list[int]] = []
        hits: list[float] = []
        conds: dict[str, list[int]] = {}
        for rows in batches:
            said = speaker(rows)
            rows_all += rows
            said_all += said
            if next_words is not None:
                hits += next_words(rows)
        out = self.tally(rows_all, said_all)
        if next_words is not None:
            out["next_word"] = float(np.mean(hits))
        agree: dict[str, list[int]] = {}
        for r, s in zip(rows_all, said_all, strict=True):
            scored = self.score(r, s)
            for key in self.conditions(r):
                c = conds.setdefault(key, [0, 0])
                c[0] += int(scored["exact"])
                c[1] += 1
                if scored.get("agree") is not None:
                    a = agree.setdefault(key, [0, 0])
                    a[0] += int(scored["agree"])
                    a[1] += 1
        out["conditions"] = {k: round(v[0] / v[1], 4) for k, v in sorted(conds.items())}
        if agree:
            out["agree_conditions"] = {k: round(v[0] / v[1], 4)
                                       for k, v in sorted(agree.items())}
        if twins:
            pairs = list(zip(rows_all[0::2], said_all[0::2], rows_all[1::2], said_all[1::2],
                             strict=True))
            out["both"] = float(np.mean([self.score(a, sa)["exact"]
                                         and self.score(b, sb)["exact"]
                                         for a, sa, b, sb in pairs]))
            out["differ"] = float(np.mean([sa != sb for _, sa, _, sb in pairs]))
        out["examples"] = [[self.say(r.target[r.force:]), self.say(s)] for r, s in
                           list(zip(rows_all, said_all, strict=True))[:4]]
        return out


TWINS = ("roles", "ctx_agent", "ctx_agent_removed", "ctx_verb", "ctx_verb_removed",
         "ctx_dyck", "ctx_dyck_removed")
G1_PANELS = ("clause", "clause_held", "clause_long", "roles", "dyck", "dyck_deep", "path",
             "path_long")
CONTEXT_PANELS = ("ctx_agent", "ctx_agent_removed", "ctx_verb", "ctx_verb_removed", "ctx_dyck",
                  "ctx_dyck_removed")
EVERYTHING = G1_PANELS + CONTEXT_PANELS + ("g2", "g2_long")
LEAN = ("clause", "clause_held", "clause_long", "roles", "dyck", "path", "path_long")
