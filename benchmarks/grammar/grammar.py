"""A bounded compositional language for the Grammar stage of cadence#173.

Eighteen words, each an input and an action of the brain. A meaning arrives as a sequence of
non-word concept moments; at ``GO`` the brain says the sentence word by word, hearing its own
previous word with the ``SPEAK`` flag, until ``end``. Three families of sentences share the
words:

- ``clause``: ``agent (near mod)* verb patient end``. The verb agrees in number with the agent
  across the distractor phrases, whose nouns may have the other number; the meaning names the
  verb first and gives the agent and the patient by role flags, in either order, so the same
  noun is the agent of one sentence and the patient of another.
- ``dyck``: the meaning is a prefix of open-round, open-square and close concepts; the brain
  says each bracket (a close is the match of the innermost open bracket) and then closes every
  bracket still open, innermost first.
- ``path``: the world says facts ``X near Y end`` (heard words); the meaning names the start;
  the brain says the chain ``start near b near c end``, each hop addressed by the word it said
  last (two-hop composition), and ends where no fact leads on.

A second grammar over the same words (``g2``, the meaning's verb moment flagged ``DIALECT``)
puts the patient before the verb: ``agent (near mod)* patient verb end``.

Held out per founder (``Split``): two agent-number-verb triples and one pair of nouns that are
never agent and patient of one trained sentence, in either order; lengths and depths beyond
training. Scores read the whole sentence the brain says, from its first word to ``end``,
closed loop; next-word accuracy under teacher forcing is component evidence beside them.

    python benchmarks/grammar/grammar.py --arms recipe --founders 0 1 2 --out DIR/dev.jsonl
    python benchmarks/grammar/grammar.py --summary DIR/dev.jsonl
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

# ------------------------------------------------------------------------------- language


def _lexicon() -> tuple[int, int]:
    """The lexicon's nouns and verbs: four and two, or what the environment variable
    ``CADENCE_GRAMMAR_LEXICON`` names (``"16x8"``), read once when the chamber is imported. A
    scaled lexicon keeps the grammar and names its words ``n0 n1 ... v0s v0 ...``."""
    text = os.environ.get("CADENCE_GRAMMAR_LEXICON", "4x2")
    try:
        nouns, verbs = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise ValueError("CADENCE_GRAMMAR_LEXICON names nouns and verbs as '16x8'") from None
    if nouns < 2 or verbs < 2:
        raise ValueError("a lexicon holds at least two nouns and two verbs")
    return nouns, verbs


NOUNS, VERBS = _lexicon()  # noun i: singular word i, plural word i + NOUNS
SCALED = (NOUNS, VERBS) != (4, 2)
if SCALED:
    _NOUN_NAMES = tuple(f"n{i}" for i in range(NOUNS))
    _VERB_NAMES = tuple(f"v{i}" for i in range(VERBS))
    WORDS: tuple[str, ...] = (
        *_NOUN_NAMES, *(name + "s" for name in _NOUN_NAMES),
        *(form for name in _VERB_NAMES for form in (name + "s", name)),
        "near", "(", ")", "[", "]", "end",
    )
    CONCEPTS: tuple[str, ...] = (
        *(name.upper() for name in _NOUN_NAMES), "PLURAL", "AGENT", "PATIENT", "MOD",
        *(name.upper() for name in _VERB_NAMES),
        "OPEN_ROUND", "OPEN_SQUARE", "CLOSE", "FROM", "DIALECT", "GO", "SPEAK",
    )
else:
    WORDS = (
        "dog", "cat", "fox", "owl", "dogs", "cats", "foxes", "owls",
        "sees", "see", "chases", "chase", "near", "(", ")", "[", "]", "end",
    )
    CONCEPTS = (
        "DOG", "CAT", "FOX", "OWL", "PLURAL", "AGENT", "PATIENT", "MOD", "SEE", "CHASE",
        "OPEN_ROUND", "OPEN_SQUARE", "CLOSE", "FROM", "DIALECT", "GO", "SPEAK",
    )
VERB = 2 * NOUNS  # verb v: its singular form word VERB + 2 v, its plural form the word after
NEAR, OPEN_ROUND, CLOSE_ROUND, OPEN_SQUARE, CLOSE_SQUARE, END = range(
    VERB + 2 * VERBS, VERB + 2 * VERBS + 6)
VERB_CONCEPTS = CONCEPTS[NOUNS + 4:NOUNS + 4 + VERBS]
C = {name: len(WORDS) + i for i, name in enumerate(CONCEPTS)}
INPUTS, ACTIONS = len(WORDS) + len(CONCEPTS), len(WORDS)
OPENS = (OPEN_ROUND, OPEN_SQUARE)
MATCH = {OPEN_ROUND: CLOSE_ROUND, OPEN_SQUARE: CLOSE_SQUARE}

BATCH = 16
FOUNDERS = (0, 1, 2)
SEEDS = {"train": 11, "test": 33, "split": 7}
TAUGHT_SHARE = 0.25  # of the verbs, taught with each agent form at a scaled lexicon
TRAIN_L, TEST_L = (0, 1, 2), (3, 4)  # distractor phrases in a clause
TRAIN_DEPTH, TEST_DEPTH = 3, (4, 5)  # the deepest nesting of a Dyck sentence
TRAIN_HOPS, TEST_HOPS = (1, 2), (3,)  # facts chained by a path sentence
FAMILIES = {"clause": 0.5, "dyck": 0.25, "path": 0.25}  # the first grammar's mix


def noun(i: int, plural: int) -> int:
    return int(i) + NOUNS * int(plural)


def verb(v: int, plural: int) -> int:
    return VERB + 2 * int(v) + int(plural)


def is_verb(word: int) -> bool:
    return VERB <= word < NEAR


def say(words: list[int]) -> str:
    return " ".join(WORDS[w] for w in words)


def concept(*names: str) -> np.ndarray:
    x = np.zeros(INPUTS)
    for name in names:
        x[C[name]] = 1.0
    return x


def heard(word: int) -> np.ndarray:
    x = np.zeros(INPUTS)
    x[word] = 1.0
    return x


def spoken(word: int) -> np.ndarray:
    """The brain's own word, as it hears it."""
    x = heard(word)
    x[C["SPEAK"]] = 1.0
    return x


def noun_concept(i: int, plural: int, *flags: str) -> np.ndarray:
    return concept(CONCEPTS[i], *(("PLURAL",) if plural else ()), *flags)


# --------------------------------------------------------------------------------- split


@dataclass(frozen=True)
class Split:
    """What one founder's training never shows: agent-number-verb triples and a noun pair."""

    triples: tuple[tuple[int, int, int], ...]  # (agent noun, plural, verb)
    pair: tuple[int, int]  # never agent and patient of one trained clause, either order
    # a scaled lexicon: the triples training shows, every other one held out (``triples`` empty)
    trained: frozenset[tuple[int, int, int]] | None = None

    @classmethod
    def of(cls, founder: int) -> Split:
        rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["split"]]))
        if SCALED:
            return cls.scaled(rng)
        a, b = (int(i) for i in rng.permutation(NOUNS)[:2])
        triples = ((a, 1, 0), (b, 0, 1))
        p, q = (int(i) for i in rng.permutation(NOUNS)[:2])
        return cls(triples, (min(p, q), max(p, q)))

    @classmethod
    def scaled(cls, rng: np.random.Generator) -> Split:
        """A scaled lexicon's split: each agent in each number is taught with ``TAUGHT_SHARE``
        of the verbs (at least one) and every other agent-number-verb triple is held out, so
        most of the language is combinations no lesson shows. The taught verbs go round the
        verbs in a random order, so every verb form is taught with the same number of agents
        where the verbs divide the taught pairs."""
        taught = max(1, round(TAUGHT_SHARE * VERBS))
        nouns, verbs = rng.permutation(NOUNS), rng.permutation(VERBS)
        trained = set()
        for plural in (0, 1):
            offset = int(rng.integers(VERBS))
            for i, agent in enumerate(nouns):
                for j in range(taught):
                    v = int(verbs[(i * taught + offset + j) % VERBS])
                    trained.add((int(agent), plural, v))
        p, q = (int(i) for i in rng.permutation(NOUNS)[:2])
        return cls((), (min(p, q), max(p, q)), frozenset(trained))

    def held(self, agent: tuple[int, int], patient: tuple[int, int], v: int) -> bool:
        if self.trained is not None:
            return ((agent[0], agent[1], v) not in self.trained
                    or {agent[0], patient[0]} == set(self.pair))
        return (agent[0], agent[1], v) in self.triples or {agent[0], patient[0]} == set(self.pair)

    def held_triple(self, agent: tuple[int, int], v: int) -> bool:
        """Whether the agent, its number and the verb are a combination no lesson shows."""
        if self.trained is not None:
            return (agent[0], agent[1], v) not in self.trained
        return (agent[0], agent[1], v) in self.triples


# ------------------------------------------------------------------------------- episodes


@dataclass
class Row:
    """One stream's episode: heard moments, then the sentence from ``GO``."""

    family: str
    pre: list[np.ndarray]
    target: list[int]
    meta: dict[str, Any] = field(default_factory=dict)
    force: int = 0  # the target's first words given as heard (the read starts after them)


def clause(rng: np.random.Generator, L: int, *, agent: tuple[int, int] | None = None,
           patient: tuple[int, int] | None = None, v: int | None = None,
           mods: list[tuple[int, int]] | None = None, inverted: bool | None = None,
           dialect: bool = False) -> Row:
    agent = agent or (int(rng.integers(NOUNS)), int(rng.integers(2)))
    patient = patient or (int(rng.integers(NOUNS)), int(rng.integers(2)))
    v = int(rng.integers(VERBS)) if v is None else v
    mods = mods if mods is not None else [(int(rng.integers(NOUNS)), int(rng.integers(2)))
                                          for _ in range(L)]
    inverted = bool(rng.integers(2)) if inverted is None else inverted
    act = concept(VERB_CONCEPTS[v], *(("DIALECT",) if dialect else ()))
    a = noun_concept(*agent, "AGENT")
    p = noun_concept(*patient, "PATIENT")
    m = [noun_concept(*mod, "MOD") for mod in mods]
    pre = [act, p, a, *m] if inverted else [act, a, *m, p]
    words = [noun(*agent)]
    for mod in mods:
        words += [NEAR, noun(*mod)]
    tail = [noun(*patient), verb(v, agent[1])] if dialect else [verb(v, agent[1]), noun(*patient)]
    meta = {"agent": list(agent), "patient": list(patient), "verb": v,
            "mods": [list(x) for x in mods], "L": len(mods), "inverted": inverted,
            "dialect": dialect,
            "attract": any(x[1] != agent[1] for x in mods),
            "patient_attract": patient[1] != agent[1]}
    return Row("clause", pre, words + tail + [END], meta)


def trained_clause(rng: np.random.Generator, split: Split, L: int, dialect: bool) -> Row:
    while True:
        row = clause(rng, L, dialect=dialect)
        m = row.meta
        if not split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"]):
            return row


def dyck_pattern(rng: np.random.Generator, length: int, max_depth: int,
                 min_depth: int = 1, nested: bool | None = None) -> list[int]:
    """A prefix of opens (1) and closes (0) whose depth stays within ``max_depth``, ends open,
    and reaches ``min_depth``."""
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


def dyck(rng: np.random.Generator, pattern: list[int], types: list[int] | None = None) -> Row:
    types = types if types is not None else [int(rng.integers(2)) for _ in pattern]
    pre, words, stack = [], [], []
    for step, t in zip(pattern, types, strict=True):
        if step:
            pre.append(concept(("OPEN_ROUND", "OPEN_SQUARE")[t]))
            words.append(OPENS[t])
            stack.append(OPENS[t])
        else:
            pre.append(concept("CLOSE"))
            words.append(MATCH[stack.pop()])
    depth, deepest = 0, 0
    for step in pattern:
        depth += 1 if step else -1
        deepest = max(deepest, depth)
    meta = {"prefix": len(pattern), "depth": deepest, "open": len(stack),
            "nested": 0 not in pattern, "types": [int(t) for t in types]}
    return Row("dyck", pre, words + [MATCH[o] for o in reversed(stack)] + [END], meta)


def path(rng: np.random.Generator, hops: int, distractors: int = 2) -> Row:
    """Facts ``X near Y end`` with one chain of ``hops`` from the start to a sink."""
    entities = 2 * NOUNS
    order = [int(i) for i in rng.permutation(entities)]
    chain, rest = order[:hops + 1], order[hops + 1:]
    facts = [(chain[i], chain[i + 1]) for i in range(hops)]
    firsts = [int(i) for i in rng.permutation(rest)[:distractors]]
    for f in firsts:
        facts.append((f, int(rng.choice([e for e in range(entities) if e != f]))))
    pre = []
    for i in rng.permutation(len(facts)):
        x, y = facts[int(i)]
        pre += [heard(x), heard(NEAR), heard(y), heard(END)]
    start = chain[0]
    pre.append(noun_concept(start % NOUNS, start // NOUNS, "FROM"))
    words = [chain[0]]
    for e in chain[1:]:
        words += [NEAR, e]
    return Row("path", pre, words + [END], {"hops": hops, "chain": chain, "facts": len(facts)})


def train_batch(rng: np.random.Generator, split: Split, grammar: str) -> list[Row]:
    """One training batch: one family and one shape, ``BATCH`` streams of their own content."""
    if grammar == "g2":
        L = int(rng.choice(TRAIN_L))
        return [trained_clause(rng, split, L, True) for _ in range(BATCH)]
    family = str(rng.choice(list(FAMILIES), p=list(FAMILIES.values())))
    if family == "clause":
        L = int(rng.choice(TRAIN_L))
        return [trained_clause(rng, split, L, False) for _ in range(BATCH)]
    if family == "dyck":
        pattern = dyck_pattern(rng, int(rng.integers(1, 7)), TRAIN_DEPTH)
        return [dyck(rng, pattern) for _ in range(BATCH)]
    hops = int(rng.choice(TRAIN_HOPS))
    return [path(rng, hops) for _ in range(BATCH)]


def solve(pre: list[np.ndarray]) -> list[int]:
    """The sentence an exact solver says from the heard moments alone (the instrument's check
    that every sentence is answerable from what the brain hears)."""
    on = [{name for name, i in C.items() if x[i]} for x in pre]
    words = [int(np.argmax(x[:ACTIONS])) if x[:ACTIONS].any() else -1 for x in pre]
    if any("FROM" in o for o in on):
        start = on[-1]
        here = noun(CONCEPTS.index(next(n for n in start if n in CONCEPTS[:NOUNS])),
                    "PLURAL" in start)
        facts = {words[i]: words[i + 2] for i in range(0, len(pre) - 1, 4)}
        said = [here]
        while here in facts:
            here = facts[here]
            said += [NEAR, here]
        return said + [END]
    if any(o & {"OPEN_ROUND", "OPEN_SQUARE", "CLOSE"} for o in on):
        said, stack = [], []
        for o in on:
            if "CLOSE" in o:
                said.append(MATCH[stack.pop()])
            else:
                stack.append(OPEN_ROUND if "OPEN_ROUND" in o else OPEN_SQUARE)
                said.append(stack[-1])
        return said + [MATCH[o] for o in reversed(stack)] + [END]
    v = next(i for i, name in enumerate(VERB_CONCEPTS) if name in on[0])
    nouns = {}
    mods = []
    for o in on[1:]:
        word = noun(CONCEPTS.index(next(n for n in o if n in CONCEPTS[:NOUNS])), "PLURAL" in o)
        if "MOD" in o:
            mods.append(word)
        else:
            nouns["AGENT" if "AGENT" in o else "PATIENT"] = word
    agent, patient = nouns["AGENT"], nouns["PATIENT"]
    said = [agent]
    for m in mods:
        said += [NEAR, m]
    form = verb(v, agent >= NOUNS)
    return said + ([patient, form] if "DIALECT" in on[0] else [form, patient]) + [END]


# --------------------------------------------------------------------------------- panels


def twin(row: Row, **change: Any) -> Row:
    """The clause of ``row`` with one part of its meaning changed."""
    m = row.meta
    spec = {"agent": tuple(m["agent"]), "patient": tuple(m["patient"]), "v": m["verb"],
            "mods": [tuple(x) for x in m["mods"]], "inverted": m["inverted"],
            "dialect": m["dialect"]}
    spec.update(change)
    return clause(np.random.default_rng(0), len(spec["mods"]), **spec)


def removed(row: Row, keep: int) -> Row:
    """The context removed: no meaning, ``GO``, then the recent words as heard."""
    recent = row.target[row.force - keep:]
    return Row(row.family, [], recent, {**row.meta, "removed": True}, force=keep)


def panels(founder: int, batches: int = 4) -> dict[str, list[list[Row]]]:
    """The founder's frozen test panels, each ``batches`` batches of one shape."""
    split = Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["test"]]))
    out: dict[str, list[list[Row]]] = {}

    def held_clause(L: int) -> Row:
        while True:
            row = clause(rng, L)
            m = row.meta
            if split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"]):
                return row

    def train_shape(i: int) -> int:
        return TRAIN_L[i % len(TRAIN_L)]

    out["clause"] = [[trained_clause(rng, split, train_shape(i), False) for _ in range(BATCH)]
                     for i in range(batches)]
    out["clause_held"] = [[held_clause(train_shape(i)) for _ in range(BATCH)]
                          for i in range(batches)]
    out["clause_long"] = [[trained_clause(rng, split, TEST_L[i % len(TEST_L)], False)
                           for _ in range(BATCH)] for i in range(batches)]
    roles = []  # the held pair in both roles, side by side: the same words, the roles swapped
    p, q = split.pair
    for i in range(batches):
        rows = []
        for _ in range(BATCH // 2):
            base = clause(rng, train_shape(i), agent=(p, int(rng.integers(2))),
                          patient=(q, int(rng.integers(2))))
            if rng.random() < 0.5:
                base = twin(base, agent=tuple(base.meta["patient"]),
                            patient=tuple(base.meta["agent"]))
            rows += [base, twin(base, agent=tuple(base.meta["patient"]),
                                patient=tuple(base.meta["agent"]))]
        roles.append(rows)
    out["roles"] = roles
    out["g2"] = [[trained_clause(rng, split, train_shape(i), True) for _ in range(BATCH)]
                 for i in range(batches)]
    out["g2_long"] = [[trained_clause(rng, split, TEST_L[i % len(TEST_L)], True)
                       for _ in range(BATCH)] for i in range(batches)]
    out["dyck"] = []
    out["dyck_deep"] = []
    for i in range(batches):
        nested = i % 2 == 0
        length = int(rng.integers(1 if nested else 3, 4 if nested else 7))
        pattern = dyck_pattern(rng, length, TRAIN_DEPTH, nested=nested)
        out["dyck"].append([dyck(rng, pattern) for _ in range(BATCH)])
        depth = TEST_DEPTH[(i // 2) % len(TEST_DEPTH)]
        length = depth if nested else int(rng.integers(depth + 2, depth + 5))
        pattern = dyck_pattern(rng, length, depth, min_depth=depth, nested=nested)
        out["dyck_deep"].append([dyck(rng, pattern) for _ in range(BATCH)])
    out["path"] = [[path(rng, TRAIN_HOPS[i % len(TRAIN_HOPS)]) for _ in range(BATCH)]
                   for i in range(batches)]
    out["path_long"] = [[path(rng, TEST_HOPS[0]) for _ in range(BATCH)] for _ in range(batches)]

    # the context: twins share their recent words and differ in an earlier part
    for name, change in (("ctx_agent", "agent"), ("ctx_verb", "verb")):
        rows_all = []
        for i in range(batches):
            L = 1 + i % 2
            rows = []
            for _ in range(BATCH // 2):
                base = trained_clause(rng, split, L, False)
                m = base.meta
                for _ in range(64):
                    other = (twin(base, agent=(m["agent"][0], 1 - m["agent"][1]))
                             if change == "agent" else twin(base, v=(m["verb"] + 1) % VERBS))
                    om = other.meta
                    if not split.held(tuple(om["agent"]), tuple(om["patient"]), om["verb"]):
                        break
                    base = trained_clause(rng, split, L, False)
                    m = base.meta
                for r in (base, other):
                    r.force = 1 + 2 * L  # the agent and the distractors are given
                rows += [base, other]
            rows_all.append(rows)
        out[name] = rows_all
        out[name + "_removed"] = [[removed(r, 2 * (1 + i % 2)) for r in rows]
                                  for i, rows in enumerate(rows_all)]
    rows_all = []
    for i in range(batches):
        depth = 2 + i % 2
        rows = []
        for _ in range(BATCH // 2):
            types = [int(t) for t in rng.integers(2, size=depth)]
            for first in (types[0], 1 - types[0]):
                r = dyck(rng, [1] * depth, [first, *types[1:]])
                r.force = depth  # the opens are given; the closes are the brain's
                rows.append(r)
        rows_all.append(rows)
    out["ctx_dyck"] = rows_all
    out["ctx_dyck_removed"] = [[removed(r, r.force - 1) for r in rows] for rows in rows_all]
    return out


# ---------------------------------------------------------------------------------- arms

HERE = Path(__file__).resolve().parent


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BATTERY = _module("sequence_battery", HERE.parent / "sequence" / "battery.py")
RECIPE = BATTERY.ARMS["recipe_relative"]  # the recall guide's recipe for unseen lengths, copying
SEQUENCE = {"episodic": False, "teaching": "normalized"}  # the battery's composition
ARMS: dict[str, dict[str, Any]] = {
    "founder": {},  # Brain.compose at every founder setting
    "recipe": {**SEQUENCE, **RECIPE},
    "maturity": {**SEQUENCE, **RECIPE, "maturity": True},
    "frozen": {**SEQUENCE, **RECIPE},  # the newborn, never taught
    "default": {"recall": True},  # what Brain.compose gives a new brain, nothing named
}


def make_brain(arm: str, founder: int, settings: dict[str, Any] | None = None,
               module: int = 64) -> Any:
    import cadence as cd

    return cd.Brain.compose(INPUTS, ACTIONS, modules=(module,), seed=int(founder),
                            **{"backend": "cpu", "recall": False,
                               **{**ARMS[arm], **(settings or {})}})


# ---------------------------------------------------------------------------------- life


def new_work() -> dict[str, float]:
    return {"lessons": 0, "rows": 0, "agreed_rows": 0, "lesson_row_sweeps": 0, "acts": 0,
            "act_sweeps": 0, "refused": 0, "exhausted": 0, "free_budget_exhausted": 0,
            "rehearsal_moments": 0, "rehearsal_rows": 0}


def speak(brain: Any, rows: list[Row], *, teach: bool, closed: bool = True,
          work: dict[str, float] | None = None, cap: int = 4,
          intervene: Any = None) -> list[list[int]]:
    """One batch on fresh streams. Heard moments are acted on greedily and never taught. From
    ``GO`` each moment hears the previous word: the target's while teaching, within a row's
    ``force`` and when ``closed`` is off; else the brain's own. With ``teach`` a lesson on the
    target word comes before every act from ``GO`` on. ``intervene(brain, rows)`` runs once,
    before the act of the first word the brain says freely (the rows share their ``force``).
    Returns each row's words from its ``force`` on, through its first ``end`` (or ``cap``
    words past its target)."""
    work = work if work is not None else new_work()
    n = len(rows)
    lengths = {len(r.pre) for r in rows}
    if len(lengths) != 1:
        raise ValueError("a batch's rows hear the same number of moments")
    if teach and len({len(r.target) for r in rows}) != 1:
        raise ValueError("a taught batch's sentences have one length")
    brain.reset()
    for t in range(lengths.pop()):
        _act(brain, np.stack([r.pre[t] for r in rows]), work)
    said: list[list[int]] = [[] for _ in rows]
    done = [False] * n
    longest = max(len(r.target) for r in rows) + (0 if teach else cap)
    for j in range(longest):
        if j == 0:
            x = np.tile(concept("GO"), (n, 1))
        else:
            words = []
            for i, r in enumerate(rows):
                given = teach or not closed or j <= r.force
                words.append(r.target[j - 1] if given and j - 1 < len(r.target)
                             else said[i][j - 1])
            x = np.stack([spoken(w) if w >= 0 else concept("SPEAK") for w in words])
        if intervene is not None and j == rows[0].force:
            intervene(brain, rows)
        if teach:
            labels = np.array([r.target[j] for r in rows])
            try:
                report = brain.lesson(x, labels)
                work["lessons"] += 1
                work["rows"] += report.get("attempted_presentations", 0.0)
                work["agreed_rows"] += report.get("agreed_presentations", 0.0)
                work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
                work["free_budget_exhausted"] += report.get("free_budget_exhausted", 0.0)
                replayed = brain.last_learning or {}  # with ``learning_rehearsal``
                work["rehearsal_moments"] += replayed.get("rehearsal_moments", 0.0)
                work["rehearsal_rows"] += replayed.get("rehearsal_accepted_presentations", 0.0)
            except RuntimeError:
                work["refused"] += 1
        action = _act(brain, x, work)
        for i, r in enumerate(rows):
            said[i].append(int(action[i]))
            if j >= r.force and action[i] == END:
                done[i] = True
        if not teach and all(done):
            break
    out = []
    for i, r in enumerate(rows):
        words = said[i][r.force:]
        out.append(words[:words.index(END) + 1] if END in words else words)
    return out


def _act(brain: Any, x: np.ndarray, work: dict[str, float]) -> np.ndarray:
    try:
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
        settled = brain.last_settlement or {}
        work["act_sweeps"] += float(settled.get("steps", 0))
    except RuntimeError:
        work["exhausted"] += 1
        action = np.full(len(x), -1)
    work["acts"] += 1
    return action


def practise(brain: Any, rows: list[Row], work: dict[str, float] | None = None) -> float:
    """One batch of practice paid by reward alone: ``Brain.step`` samples every word, hears the
    target's word next (as a tutor repeats the right word) and is paid 1 for a right word and 0
    for a wrong one; the last word's outcome ends the episode through ``Brain.learn``. Returns
    the share of words said right."""
    work = work if work is not None else new_work()
    n = len(rows)
    brain.reset()
    reward = None
    for t in range(len(rows[0].pre)):
        brain.step(np.stack([r.pre[t] for r in rows]), reward=reward)
        reward = np.zeros(n)
        work["steps"] = work.get("steps", 0) + 1
    hits = []
    for j in range(len(rows[0].target)):
        x = (np.tile(concept("GO"), (n, 1)) if j == 0
             else np.stack([spoken(r.target[j - 1]) for r in rows]))
        action = np.asarray(brain.step(x, reward=reward)).reshape(n, -1)[:, 0]
        settled = brain.last_settlement or {}
        work["act_sweeps"] += float(settled.get("steps", 0))
        work["steps"] = work.get("steps", 0) + 1
        reward = (action == np.array([r.target[j] for r in rows])).astype(float)
        hits.append(float(reward.mean()))
    following = np.stack([spoken(r.target[-1]) for r in rows])
    brain.learn(reward, np.ones(n, dtype=bool), following)
    work["steps"] = work.get("steps", 0) + 1
    return float(np.mean(hits))


def erase(where: str) -> Any:
    """An intervention at the continuation's start. ``recall`` empties the slots of the earlier
    context (keys, values and nexts set to zero): the meaning's moments, ``GO`` and the first
    given word, the one the twins of ``ctx_agent`` and ``ctx_dyck`` differ in; the slots of the
    recent words stay. ``trace`` sets the working trace to zero."""

    def act(brain: Any, rows: list[Row]) -> None:
        if where == "trace":
            brain.working_memory.trace = np.zeros_like(brain.working_memory.trace)
            return
        recall = brain.recall
        if recall is None:
            return
        written = len(rows[0].pre) + rows[0].force  # the meaning, GO and the given words
        ages = np.asarray(recall.ages())
        meaning = ages >= written - len(rows[0].pre) - 2  # moments 0 to len(pre) + 1
        recall.keys = np.where(meaning[None, :, None], 0.0, recall.keys)
        recall.values = np.where(meaning[None, :, None], 0.0, recall.values)
        recall.nexts = np.where(meaning[None, :, None], 0.0, recall.nexts)

    return act


def forced(brain: Any, rows: list[Row]) -> list[float]:
    """Next-word accuracy with every previous word given (component evidence)."""
    brain.reset()
    for t in range(len(rows[0].pre)):
        brain.act(np.stack([r.pre[t] for r in rows]), greedy=True)
    hits = []
    for j in range(len(rows[0].target)):
        x = (np.tile(concept("GO"), (len(rows), 1)) if j == 0
             else np.stack([spoken(r.target[j - 1]) for r in rows]))
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
        if j >= rows[0].force:
            hits += [float(action[i] == r.target[j]) for i, r in enumerate(rows)]
    return hits


def sources(row: Row) -> list[int | None]:
    """For each target word, the heard moment that holds its content (an index into ``pre``),
    or None for a word the grammar alone decides (``near``, ``end``)."""
    if row.family == "clause":
        m = row.meta
        L = m["L"]
        agent, mods, patient = ((2, list(range(3, 3 + L)), 1) if m["inverted"]
                                else (1, list(range(2, 2 + L)), 2 + L))
        out: list[int | None] = [agent]
        for i in mods:
            out += [None, i]
        tail: list[int | None] = [patient, 0] if m["dialect"] else [0, patient]
        return out + tail + [None]
    if row.family == "dyck":
        out, stack = [], []
        for i, x in enumerate(row.pre):
            if x[C["CLOSE"]]:
                out.append(stack.pop())
            else:
                out.append(i)
                stack.append(i)
        return out + list(reversed(stack)) + [None]
    out = [len(row.pre) - 1]  # the start, named by the meaning
    heard_words = [int(np.argmax(x)) for x in row.pre[:-1]]
    for a in row.meta["chain"][:-1]:
        at = next(i for i in range(0, len(heard_words), 4) if heard_words[i] == a)
        out += [None, at + 2]  # the fact's second word as heard
    return out + [None]


def locate(brain: Any, batches: list[list[Row]]) -> dict[str, Any]:
    """Where the read lands and what is said, teacher-forced, on a fork: for every target word
    whose content a heard moment holds, whether the recall memory's first head attends most to
    that moment or to the moment before it, whose next it is (``delivered``: the read's value
    or next port holds it), and whether the brain says the word. Words are grouped by family
    and kind (the clause's agent, distractor, verb and patient; the Dyck opens and closes; the
    path's start and hops)."""
    fork = copy.deepcopy(brain)
    groups: dict[str, list[list[int]]] = {}
    for rows in batches:
        fork.reset()
        written = 0
        for t in range(len(rows[0].pre)):
            fork.act(np.stack([r.pre[t] for r in rows]), greedy=True)
            written += 1
        expected = [sources(r) for r in rows]
        for j in range(len(rows[0].target)):
            x = (np.tile(concept("GO"), (len(rows), 1)) if j == 0
                 else np.stack([spoken(r.target[j - 1]) for r in rows]))
            # the slots' ages as the read sees them, before the act writes this moment's slot
            ages = None if fork.recall is None else np.asarray(fork.recall.ages())
            action = np.asarray(fork.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
            heads = fork.recall.attended_heads if fork.recall is not None else None
            p = heads[0] if heads else None
            if p is not None and ages is not None and p.shape[1] != len(ages):
                raise RuntimeError("the read's slots and their ages differ")
            for i, r in enumerate(rows):
                source = expected[i][j]
                if source is None:
                    continue
                landed = -2
                if p is not None and ages is not None:
                    landed = written - 1 - int(ages[int(np.argmax(p[i]))])
                kind = _kind(r, j)
                g = groups.setdefault(kind, [0, 0, 0, 0])  # delivered, said, said|delivered, n
                found = landed in (source, source - 1)  # its value, or the next of the one before
                g[0] += int(found)
                g[1] += int(action[i] == r.target[j])
                g[2] += int(found and action[i] == r.target[j])
                g[3] += 1
            written += 1
    return {k: {"n": v[3], "delivered": round(v[0] / v[3], 4), "said": round(v[1] / v[3], 4),
                "said_when_delivered": round(v[2] / v[0], 4) if v[0] else None}
            for k, v in sorted(groups.items())}


def _kind(r: Row, j: int) -> str:
    if r.family == "clause":
        m = r.meta
        k = next(i for i, w in enumerate(r.target) if is_verb(w))
        if j == 0:
            name = "agent"
        elif j == k:
            name = "verb"
        elif r.target[j] == noun(*m["patient"]) and (j == k + 1 or (m["dialect"] and j == k - 1)):
            name = "patient"
        else:
            name = f"mod{(j - 1) // 2 + 1}"
        return f"clause_L{m['L']}_{name}"
    if r.family == "dyck":
        kind = "open" if r.target[j] in OPENS else "close"
        return f"dyck_d{r.meta['depth']}_{kind}"
    return f"path_h{r.meta['hops']}_" + ("start" if j == 0 else f"hop{j // 2}")


# -------------------------------------------------------------------------------- scoring


def score(row: Row, said: list[int]) -> dict[str, Any]:
    """The whole sentence said, from its first word through ``end``."""
    target = row.target[row.force:]
    out: dict[str, Any] = {"exact": said == target}
    if row.meta.get("removed"):
        return out
    words = said[:-1] if said and said[-1] == END else said
    ended = bool(said) and said[-1] == END
    if row.family == "clause" and row.force == 0:
        m = row.meta
        verbs = [i for i, w in enumerate(words) if is_verb(w)]
        one = len(verbs) == 1
        k = verbs[0] if one else -1
        out["agree"] = one and (words[k] - VERB) % 2 == m["agent"][1]
        out["verb"] = one and (words[k] - VERB) // 2 == m["verb"]
        subject = words[0] if words and words[0] < VERB else None
        out["agree_own"] = (one and subject is not None
                            and (words[k] - VERB) % 2 == subject // NOUNS)
        # an attracted verb: it agrees with the noun said just before it, not with the agent
        before = words[k - 1] if one and k > 0 and words[k - 1] < VERB else None
        out["attracted"] = (None if before is None or before // NOUNS == m["agent"][1]
                            else (words[k] - VERB) % 2 == before // NOUNS)
        out["subject"] = subject == noun(*m["agent"])
        patient_at = (k - 1) if m["dialect"] else (k + 1)
        out["patient"] = one and 0 <= patient_at < len(words) and words[patient_at] == noun(
            *m["patient"])
        out["roles"] = out["subject"] and out["patient"]
        mods = [w for pair in m["mods"] for w in (NEAR, noun(*pair))]
        out["mods"] = words[1:1 + len(mods)] == mods if mods else None
        out["ended"] = ended
    elif row.family == "clause":  # a context continuation: the verb, the patient, end
        out["agree"] = bool(said) and is_verb(said[0]) and (said[0] - VERB) % 2 == row.meta[
            "agent"][1]
        out["verb"] = (bool(said) and is_verb(said[0])
                       and (said[0] - VERB) // 2 == row.meta["verb"])
    elif row.family == "dyck":
        stack, valid = [], ended
        for w in row.target[:row.force] + words:
            if w in OPENS:
                stack.append(w)
            elif w in (CLOSE_ROUND, CLOSE_SQUARE) and stack and MATCH[stack[-1]] == w:
                stack.pop()
            else:
                valid = False
        # balanced, every close matching, and as long as the sentence the meaning asks for
        out["valid"] = valid and not stack and len(words) == len(target) - 1
        if row.force == 0:
            out["prefix"] = words[:row.meta["prefix"]] == row.target[:row.meta["prefix"]]
    elif row.family == "path":
        chain = row.meta["chain"]
        out["hops"] = sum(1 for h in range(1, len(chain))
                          if len(words) > 2 * h and words[2 * h] == chain[h]
                          and words[2 * h - 1] == NEAR and all(
                              words[2 * g] == chain[g] for g in range(h)))
        out["first_hop"] = len(words) > 2 and words[2] == chain[1]
    return out


def tally(rows: list[Row], said: list[list[int]]) -> dict[str, Any]:
    scores = [score(r, s) for r, s in zip(rows, said, strict=True)]
    out: dict[str, Any] = {"n": len(scores)}
    for key in scores[0]:
        kept = [s[key] for s in scores if s.get(key) is not None]
        out[key] = float(np.mean(kept)) if kept else None
    return out


def read_panel(brain: Any, batches: list[list[Row]], *, twins: bool = False,
               components: bool = True, intervene: Any = None) -> dict[str, Any]:
    """Closed-loop sentences (and teacher-forced next words) on a fork of ``brain``."""
    fork = copy.deepcopy(brain)
    return read_speaker(lambda rows: speak(fork, rows, teach=False, intervene=intervene),
                        (lambda rows: forced(fork, rows)) if components else None,
                        batches, twins=twins)


def read_speaker(speaker: Any, next_words: Any, batches: list[list[Row]], *,
                 twins: bool = False) -> dict[str, Any]:
    """The panel's scores for any speaker: ``speaker(rows)`` returns the words each row says
    from its ``force`` on, ``next_words(rows)`` (or None) the teacher-forced hits."""
    rows_all, said_all, hits, conds = [], [], [], {}
    for rows in batches:
        said = speaker(rows)
        rows_all += rows
        said_all += said
        if next_words is not None:
            hits += next_words(rows)
    components = next_words is not None
    out = tally(rows_all, said_all)
    if components:
        out["next_word"] = float(np.mean(hits))
    agree: dict[str, list[int]] = {}
    for r, s in zip(rows_all, said_all, strict=True):
        scored = score(r, s)
        for key in _conditions(r):
            c = conds.setdefault(key, [0, 0])
            c[0] += int(scored["exact"])
            c[1] += 1
            if scored.get("agree") is not None:
                a = agree.setdefault(key, [0, 0])
                a[0] += int(scored["agree"])
                a[1] += 1
    out["conditions"] = {k: round(v[0] / v[1], 4) for k, v in sorted(conds.items())}
    if agree:
        out["agree_conditions"] = {k: round(v[0] / v[1], 4) for k, v in sorted(agree.items())}
    if twins:
        pairs = list(zip(rows_all[0::2], said_all[0::2], rows_all[1::2], said_all[1::2],
                         strict=True))
        out["both"] = float(np.mean([score(a, sa)["exact"] and score(b, sb)["exact"]
                                     for a, sa, b, sb in pairs]))
        out["differ"] = float(np.mean([sa != sb for _, sa, _, sb in pairs]))
    out["examples"] = [[say(r.target[r.force:]), say(s)] for r, s in
                       list(zip(rows_all, said_all, strict=True))[:4]]
    return out


def _conditions(r: Row) -> list[str]:
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


TWINS = ("roles", "ctx_agent", "ctx_agent_removed", "ctx_verb", "ctx_verb_removed",
         "ctx_dyck", "ctx_dyck_removed")
G1_PANELS = ("clause", "clause_held", "clause_long", "roles", "dyck", "dyck_deep", "path",
             "path_long")
CONTEXT_PANELS = ("ctx_agent", "ctx_agent_removed", "ctx_verb", "ctx_verb_removed", "ctx_dyck",
                  "ctx_dyck_removed")


ERASED = ("ctx_agent", "ctx_verb", "ctx_dyck")


def reading(brain: Any, tests: dict[str, list[list[Row]]], names: tuple[str, ...],
            components: bool = True) -> dict[str, Any]:
    """Each named panel; with the context panels, the same twins with the context erased from
    the recall memory or from the working trace at the continuation's start."""
    out = {name: read_panel(brain, tests[name], twins=name in TWINS, components=components)
           for name in names}
    for name in ERASED:
        if name in names:
            for where in ("recall", "trace"):
                out[f"{name}_erased_{where}"] = read_panel(
                    brain, tests[name], twins=True, components=False, intervene=erase(where))
    return out


EVERYTHING = G1_PANELS + CONTEXT_PANELS + ("g2", "g2_long")
LOCATE = ("clause", "clause_long", "g2", "dyck", "dyck_deep", "path", "path_long")


def live_phases(brain: Any, arm: str, rng: np.random.Generator, split: Split,
                tests: dict[str, list[list[Row]]], phases: list[tuple[str, int]],
                checkpoints: set[int], first: int = 0,
                save: Path | None = None) -> list[dict[str, Any]]:
    """Teach ``phases`` in turn (``g1`` or ``g2`` by lessons, ``g1:reward`` or ``g2:reward`` by
    paid practice), each followed by a full reading on forks; with ``save`` the brain is saved
    after each phase as ``<save>-ph<index>``."""
    readings = []
    for index, (grammar, count) in enumerate(phases, start=first):
        work = new_work()
        phase_started = time.monotonic()
        language, _, how = grammar.partition(":")
        paid = []
        for b in range(1, count + 1):
            batch = train_batch(rng, split, language)
            if arm == "frozen":
                continue
            if how == "reward":
                paid.append(practise(brain, batch, work))
            else:
                speak(brain, batch, teach=True, work=work)
            if b in checkpoints and b != count:
                readings.append({
                    "phase": index, "grammar": grammar, "batches": b, "work": dict(work),
                    "panels": reading(brain, tests, ("clause", "dyck", "path", "g2"),
                                      components=False)})
        work["seconds"] = round(time.monotonic() - phase_started, 1)
        if paid:
            work["paid_first"] = float(np.mean(paid[:50]))
            work["paid_last"] = float(np.mean(paid[-50:]))
        entry = {"phase": index, "grammar": grammar, "batches": count, "work": work,
                 "panels": reading(brain, tests, EVERYTHING),
                 "located": {name: locate(brain, tests[name]) for name in LOCATE}
                 if brain.recall is not None else None}
        maturity = getattr(brain.basal_ganglia, "maturity", None)
        if maturity is not None:
            entry["maturity"] = {**maturity.to_dict(), "factor": float(maturity.factor())}
        if save is not None:
            entry["saved"] = str(brain.save(Path(f"{save}-ph{index}")))
        readings.append(entry)
    return readings


def continuation(brain: Any, rng: np.random.Generator, split: Split,
                 rows: list[Row]) -> dict[str, Any]:
    """Save and load a fork; both copies take the same three lessons and say the same panel."""
    import tempfile

    import cadence as cd

    live = copy.deepcopy(brain)
    with tempfile.TemporaryDirectory() as folder:
        loaded = cd.Brain.load(live.save(Path(folder) / "brain"))
    stream = copy.deepcopy(rng)  # the life's own stream is not advanced
    batches = [train_batch(stream, split, "g1") for _ in range(3)]
    for b in batches:
        speak(live, b, teach=True)
        speak(loaded, b, teach=True)
    said = [speak(x, rows, teach=False) for x in (live, loaded)]
    return {"identical": said[0] == said[1], "sentences": len(rows)}


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life. ``spec``: ``arm``, ``founder``, ``phases`` (the trunk: a list of
    ``[grammar, batches]``, read after each), ``checkpoints`` (batches within a phase at which
    a lean reading is taken), ``forks`` (each ``name``, ``after`` (the trunk phase it follows),
    ``retune`` (settings), ``scale_from_use`` (``maturity_scale`` set to the use at the fork)
    and ``phases``, lived on a copy of the brain and of the training stream), ``settings`` over
    the arm's, ``module``, ``panel_batches``, ``continuation`` (the saved-continuation check
    after the first phase) and ``save`` (a folder: the brain saved after every phase)."""
    warnings.simplefilter("ignore")
    import cadence as cd

    arm, founder = spec["arm"], int(spec["founder"])
    phases = [(str(g), int(n)) for g, n in spec.get("phases", [["g1", 2000]])]
    forks = spec.get("forks", [])
    started = time.monotonic()
    brain = make_brain(arm, founder, spec.get("settings"), int(spec.get("module", 64)))
    split = Split.of(founder)
    tests = panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, SEEDS["train"]]))
    row: dict[str, Any] = {
        "arm": arm, "founder": founder, "tag": spec.get("tag"),
        "settings": {**ARMS[arm], **spec.get("settings", {})},
        "phases": [list(p) for p in phases], "forks": forks,
        "module": int(spec.get("module", 64)), "parameters": int(brain.parameters()),
        "cadence": cd.__version__,
        "split": {"triples": [list(t) for t in split.triples], "pair": list(split.pair)},
        "readings": [{"phase": -1, "batches": 0, "work": new_work(),
                      "panels": reading(brain, tests, EVERYTHING)}],
        "branches": {},
    }
    checkpoints = set(int(c) for c in spec.get("checkpoints", []))
    folder = Path(spec["save"]) if spec.get("save") else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)

    def saved(branch: str) -> Path | None:
        return None if folder is None else folder / f"{arm}-f{founder}-{branch}"

    for index, phase in enumerate(phases):
        row["readings"] += live_phases(brain, arm, rng, split, tests, [phase],
                                       checkpoints if index == 0 else set(), first=index,
                                       save=saved("trunk"))
        if index == 0 and spec.get("continuation") and arm != "frozen":
            row["continuation"] = continuation(brain, rng, split, tests["clause"][0])
        for fork in forks:
            if int(fork["after"]) != index:
                continue
            twin_brain, twin_rng = copy.deepcopy(brain), copy.deepcopy(rng)
            retune = dict(fork.get("retune", {}))
            if fork.get("scale_from_use"):
                retune["maturity_scale"] = max(twin_brain.basal_ganglia.maturity.use, 1e-6)
            if retune:
                twin_brain.retune(**retune)
            fork_phases = [(str(g), int(n)) for g, n in fork["phases"]]
            row["branches"][fork["name"]] = {
                "retune": retune,
                "readings": live_phases(twin_brain, arm, twin_rng, split, tests, fork_phases,
                                        set(), first=index + 1, save=saved(fork["name"]))}
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


# -------------------------------------------------------------------------------- summary


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...] = ("exact",)) -> list[str]:
    lines = []
    for r in sorted(rows, key=lambda r: (r["arm"], r["founder"])):
        for rd in r["readings"]:
            if rd["phase"] < 0 and r["arm"] != "frozen":
                continue
            cells = []
            for name, p in rd["panels"].items():
                bits = [f"{p[k]:.2f}" for k in keys if k in p]
                if "both" in p:
                    bits.append(f"both {p['both']:.2f}")
                cells.append(f"{name} {'/'.join(bits)}")
            lines.append(f"{r['arm']:9s} f{r['founder']} ph{rd['phase']} b{rd['batches']}: "
                         + ", ".join(cells))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arms", nargs="*", default=["recipe"], choices=list(ARMS))
    parser.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--phases", type=json.loads, default=[["g1", 2000]])
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[])
    parser.add_argument("--settings", type=json.loads, default={}, help="over the arms'")
    parser.add_argument("--panel-batches", type=int, default=4)
    parser.add_argument("--save", type=Path, help="a folder for the brain after every phase")
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs in place of the grid")
    parser.add_argument("--out", type=Path, help="JSON lines, one row per life, appended")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary", type=Path, help="print the summary of a results file")
    args = parser.parse_args(argv)
    if args.summary is not None:
        rows = [json.loads(line) for line in args.summary.read_text().splitlines() if line]
        print("\n".join(summarize(rows)))
        return 0
    if args.out is None:
        parser.error("--out is required")
    if args.plan is not None:
        specs = json.loads(args.plan.read_text())
    else:
        specs = [{"arm": arm, "founder": f, "phases": args.phases, "settings": args.settings,
                  "checkpoints": args.checkpoints, "panel_batches": args.panel_batches,
                  "continuation": True, "save": str(args.save) if args.save else None}
                 for arm in args.arms for f in args.founders]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for result in map(run_life, specs):
                sink.write(json.dumps(result) + "\n")
                sink.flush()
        else:
            with get_context("spawn").Pool(min(args.workers, len(specs))) as pool:
                for result in pool.imap_unordered(run_life, specs, chunksize=1):
                    sink.write(json.dumps(result) + "\n")
                    sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
