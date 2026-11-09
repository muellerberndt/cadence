"""Composed attention functions (cadence#269): duplicates, suppression and indirect objects.

Interpretability work on transformers finds functions composed across heads: one head detects a
duplicate, another suppresses attention to it, another moves the remaining name to the output.
This chamber reads that composition in one brain, and each part alone first, at tokens no lesson
used. The named task: ``when mary and john went out john gave a drink to`` answers ``mary``.

Twenty words, each an input and an action of the brain: twelve names and the frame ``when and
went out gave a drink to``. Per founder (``Split``) eight names are taught and four are never
heard and never answered in training, and four pairs of taught names are never heard together.
The single functions (``TASKS``) use the same twelve tokens, the same split and a cue flag of
their own, each in a life of its own:

- ``previous``: a stream, its last token flagged ``ASK``; the token before it.
- ``duplicate``: distinct tokens, then a token flagged ``ASK``; ``yes`` if it was told, else ``no``.
- ``successor``: distinct tokens told once, then one flagged ``ASK``; the token told after it.
- ``successor_fixed``: the same, with one fixed order of the taught tokens in training.
- ``induction``: ``... A B ... A``, the query unflagged; ``B``.
- ``twohop``: pairs told, then ``a``, then ``THINK``; ``b`` and ``c``: a read on what a read found.
- ``copying``: ``START``, distinct tokens, ``GO``, each answer heard (closed loop at test); each
  token in turn.
- ``suppression``: ``A B S GO``, ``S`` one of the two; the one not repeated.
- ``which``: the same; ``first`` or ``second``, the place of the one not repeated.
- ``ioi``: ``when A and B (went (out)) S gave (a drink) to``; the one not repeated.
- ``bare``: the same with no ``when`` and no ``and``: no fixed word before the names.
- ``mover``: the frame without ``S``, ``to`` flagged ``FIRST`` or ``SECOND``; the name at that
  place.
- ``dupbit``: the sentence of ``ioi``, ``to`` flagged ``WHICH``; ``first`` or ``second``, where the
  duplicate's earlier mention stood.

``mover`` and ``dupbit`` are the indirect-object task's two parts inside its own frame. Lessons
fall on answer moments only, before the act on the same observation; every other moment is heard
and acted on greedily. A panel's rows carry conditions: ``taught`` (every token taught),
``key_new`` (the token the read is addressed by is never-taught, the answer taught),
``answer_new`` and ``new`` (both). Beside the answer each row reads the read's own first choice
(``first``: the slot head 0 attends most holds the right token as its value or its next), so a read
that finds a token the motor cannot say is told apart from a read that fails.

Causality (``CAUSAL``): the read silenced at every moment or at one, the attended slot erased at
the answer or at the duplicate, the answer's own slot lesioned on both routes, a slot neither
attended nor holding the answer erased (a control), the carriers across moments cleared at the
answer (the working trace, the cortex the address reads, the previous read's focus), and on twin
panels each put in a row's place from its twin. ``Support`` names a term by what restores the
composition when the harness gives or takes it from a life; it is no option of the brain.

    python benchmarks/heads/heads.py --task ioi --arms Fd Fd2 FdT --founders 0 1 2 \\
        --out DIR/ioi.jsonl --workers 9
    python benchmarks/heads/heads.py --table DIR/ioi.jsonl
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

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


gr = _module("grammar", HERE.parent / "grammar" / "grammar.py")

# ----------------------------------------------------------------------------- the language

BATCH = 16  # the grammar chamber's
FOUNDERS = (0, 1, 2)
SEEDS = {"train": 11, "test": 33, "split": 7}  # the grammar chamber's
NAMES = ("mary", "john", "anna", "tom", "lisa", "paul", "sara", "mark", "nina", "owen", "ruth",
         "sean")
FRAME = ("when", "and", "went", "out", "gave", "a", "drink", "to")
WORDS = NAMES + FRAME
TOKENS = len(NAMES)
NEW_TOKENS, HELD_PAIRS = 4, 4
WHEN, AND, WENT, OUT, GAVE, A, DRINK, TO = range(TOKENS, TOKENS + len(FRAME))
PRE = ((), (WENT,), (WENT, OUT))  # the words between the pair and the duplicate
POST = {1: (GAVE,), 3: (GAVE, A, DRINK)}  # the words between the duplicate and ``to``
TRAIN_SHAPES = ((2, 3), (1, 3), (0, 3), (2, 1), (0, 1))
HELD_SHAPE = (1, 1)  # a duplicate position no lesson reached
KINDS = ("taught", "key_new", "answer_new", "new")
# one pass at the grammar chamber's batches
LESSONS = {"ioi": 4000, "bare": 4000, "mover": 4000, "dupbit": 4000}
SINGLE_LESSONS = 1500  # the sequence battery's


def onehot(index: Any, width: int) -> np.ndarray:
    index = np.asarray(index)
    out = np.zeros((len(index), width))
    known = index >= 0  # a moment that hears nothing is silent
    out[np.flatnonzero(known), index[known]] = 1.0
    return out


@dataclass(frozen=True)
class Split:
    """What one founder's training never shows: four tokens and four pairs of taught tokens."""

    taught: tuple[int, ...]
    new: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]

    @classmethod
    def of(cls, founder: int) -> Split:
        rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["split"]]))
        order = [int(i) for i in rng.permutation(TOKENS)]
        new, taught = sorted(order[:NEW_TOKENS]), sorted(order[NEW_TOKENS:])
        every = [(a, b) for i, a in enumerate(taught) for b in taught[i + 1:]]
        chosen = rng.permutation(len(every))[:HELD_PAIRS]
        return cls(tuple(taught), tuple(new), tuple(every[int(i)] for i in chosen))

    def held(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.pairs


@dataclass
class Moment:
    """One moment of one batch. ``label`` is taught and scored when given; ``source`` the token a
    right read delivers (the label by default, -1 where no slot holds one); ``avoid`` the token a
    suppressing read leaves alone; ``slots`` the moment each named token was heard at; ``mark``
    names a moment other than an answer (``dup``: the duplicate's); ``feedback``: the previous
    answer is heard in closed loop; ``tags``: each row's conditions."""

    x: np.ndarray
    label: np.ndarray | None = None
    source: np.ndarray | None = None
    avoid: np.ndarray | None = None
    slots: dict[str, np.ndarray] = field(default_factory=dict)
    mark: str = ""
    feedback: bool = False
    tags: list[tuple[str, ...]] | None = None


def _class_of(key: int, answer: int, new: set[int]) -> str:
    return KINDS[int(key in new) + 2 * int(answer in new)]


class Task:
    """A family of batches on one founder's split: ``train(rng)`` a training batch, ``panels(n)``
    the frozen test panels of about ``n`` batches each."""

    name = ""
    flags: tuple[str, ...] = ()  # cue flags, input coordinates after the words
    extra_actions: tuple[str, ...] = ()  # actions that are no token
    words = TOKENS
    twins = False  # every panel's rows are pairs differing in what the task asks

    def __init__(self, founder: int) -> None:
        self.founder = int(founder)
        self.split = Split.of(founder)
        self.taught = np.array(self.split.taught)
        self.new = np.array(self.split.new)
        self.new_set = set(self.split.new)
        self.inputs = self.words + len(self.flags)
        self.actions = self.words + len(self.extra_actions)
        self.flag = {name: self.words + i for i, name in enumerate(self.flags)}

    def tokens(self) -> list[int]:
        """Per action the observation coordinate of the same token (``recall_tokens``)."""
        return [*range(self.words), *([-1] * len(self.extra_actions))]

    def pools(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """The tokens of a row's key and of its answer, by the row's kind."""
        return (self.new if kind in ("key_new", "new") else self.taught,
                self.new if kind in ("answer_new", "new") else self.taught)

    def x(self, tokens: Any, *flags: str) -> np.ndarray:
        out = onehot(tokens, self.inputs)
        for name in flags:
            out[:, self.flag[name]] = 1.0
        return out

    def test_rng(self) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.founder, SEEDS["test"]]))

    def kinds_of_a_batch(self) -> list[str]:
        return [KINDS[i % len(KINDS)] for i in range(BATCH)]

    def train(self, rng: np.random.Generator) -> list[Moment]:
        return self.batch(rng, ["taught"] * BATCH)

    def batch(self, rng: np.random.Generator, kinds: list[str]) -> list[Moment]:
        raise NotImplementedError

    def panels(self, batches: int) -> dict[str, list[list[Moment]]]:
        rng = self.test_rng()
        return {"test": [self.batch(rng, self.kinds_of_a_batch()) for _ in range(batches)]}


def _pick(rng: np.random.Generator, pool: Any, avoid: tuple[int, ...] = ()) -> int:
    choices = [int(t) for t in pool if int(t) not in avoid]
    return choices[int(rng.integers(len(choices)))]


class Previous(Task):
    """A stream of tokens, its last flagged ``ASK``: the token before it, by position against
    content (earlier tokens may be the present one)."""

    name = "previous"
    flags = ("ASK",)
    lengths = (3, 8)

    def batch(self, rng: np.random.Generator, kinds: list[str]) -> list[Moment]:
        length = int(rng.integers(self.lengths[0], self.lengths[1] + 1))
        rows = []
        for kind in kinds:
            keys, answers = self.pools(kind)
            key = _pick(rng, keys)
            answer = _pick(rng, answers, (key,))
            rows.append([*(_pick(rng, self.taught) for _ in range(length - 2)), answer, key])
        seq = np.array(rows)
        moments = [Moment(self.x(seq[:, t])) for t in range(length - 1)]
        moments.append(Moment(self.x(seq[:, -1], "ASK"), label=seq[:, -2],
                              slots={"answer": np.full(BATCH, length - 2)},
                              tags=[(k, f"L{length}") for k in kinds]))
        return moments


class Duplicate(Task):
    """Distinct tokens, then one flagged ``ASK``: was it told? The answer names no token; half the
    rows of a batch say ``yes``."""

    name = "duplicate"
    flags = ("ASK",)
    extra_actions = ("yes", "no")
    lengths = (3, 6)

    def kinds_of_a_batch(self) -> list[str]:
        return [("taught", "key_new")[i % 2] for i in range(BATCH)]

    def batch(self, rng: np.random.Generator, kinds: list[str]) -> list[Moment]:
        length = int(rng.integers(self.lengths[0], self.lengths[1] + 1))
        rows, queries, where, tags = [], [], [], []
        yes = rng.permutation(np.arange(BATCH) % 2 == 0)
        # the two kinds of a panel's batch are balanced apart: a yes and a no of each
        if len(set(kinds)) > 1:
            for kind in set(kinds):
                at = [i for i, k in enumerate(kinds) if k == kind]
                yes[at] = rng.permutation(np.arange(len(at)) % 2 == 0)
        for i, kind in enumerate(kinds):
            keys, _ = self.pools(kind)
            query = _pick(rng, keys)
            others = [int(t) for t in rng.permutation(self.taught) if int(t) != query][:length]
            place = int(rng.integers(length))
            if yes[i]:
                others[place] = query
            rows.append(others)
            queries.append(query)
            where.append(place if yes[i] else -1)
            tags.append((kind, "yes" if yes[i] else "no", f"L{length}"))
        seq = np.array(rows)
        moments = [Moment(self.x(seq[:, t])) for t in range(length)]
        label = np.where(yes, self.words, self.words + 1)
        queries_a = np.array(queries)
        moments.append(Moment(self.x(queries_a, "ASK"), label=label,
                              source=np.where(yes, queries_a, -1),
                              slots={"answer": np.array(where)}, tags=tags))
        return moments


class Successor(Task):
    """Distinct tokens told once, each episode in its own order, then one flagged ``ASK``: the
    token told after it."""

    name = "successor"
    flags = ("ASK",)
    lengths = (3, 6)

    def told(self, rng: np.random.Generator, kinds: list[str], length: int) -> list[Moment]:
        rows, keys_at, tags = [], [], []
        for kind in kinds:
            keys, answers = self.pools(kind)
            key = _pick(rng, keys)
            answer = _pick(rng, answers, (key,))
            place = int(rng.integers(length - 1))
            rest = [int(t) for t in rng.permutation(self.taught) if int(t) not in (key, answer)]
            seq = rest[:length - 2]
            seq[place:place] = [key, answer]
            rows.append(seq)
            keys_at.append(place)
            tags.append((kind, f"L{length}"))
        return self.moments(np.array(rows), np.array(keys_at), tags)

    def moments(self, seq: np.ndarray, keys_at: np.ndarray, tags: list[tuple[str, ...]]
                ) -> list[Moment]:
        at = np.arange(len(seq))
        moments = [Moment(self.x(seq[:, t])) for t in range(seq.shape[1])]
        moments.append(Moment(self.x(seq[at, keys_at], "ASK"), label=seq[at, keys_at + 1],
                              slots={"answer": keys_at + 1}, tags=tags))
        return moments

    def batch(self, rng: np.random.Generator, kinds: list[str]) -> list[Moment]:
        return self.told(rng, kinds, int(rng.integers(self.lengths[0], self.lengths[1] + 1)))


class SuccessorFixed(Successor):
    """The successor with one fixed order of the taught tokens told in every training episode, so
    that weights alone can answer it; read on that order and, as ``successor``, on told orders of
    their own."""

    name = "successor_fixed"

    def __init__(self, founder: int) -> None:
        super().__init__(founder)
        rng = np.random.default_rng(np.random.SeedSequence([self.founder, SEEDS["split"], 1]))
        self.order = rng.permutation(self.taught)

    def fixed(self, rng: np.random.Generator, tags: list[tuple[str, ...]]) -> list[Moment]:
        seq = np.tile(self.order, (BATCH, 1))
        return self.moments(seq, rng.integers(seq.shape[1] - 1, size=BATCH), tags)

    def train(self, rng: np.random.Generator) -> list[Moment]:
        return self.fixed(rng, [("fixed",)] * BATCH)

    def panels(self, batches: int) -> dict[str, list[list[Moment]]]:
        rng = self.test_rng()
        return {"test": [self.batch(rng, self.kinds_of_a_batch()) for _ in range(batches)],
                "fixed": [self.fixed(rng, [("fixed",)] * BATCH) for _ in range(batches)]}


class Induction(Task):
    """``... A B ... A``: the token that followed the query's earlier occurrence, the sequence
    battery's task."""

    name = "induction"
    lengths = (5, 10)
    gap_min = 2

    def batch(self, rng: np.random.Generator, kinds: list[str]) -> list[Moment]:
        length = int(rng.integers(self.lengths[0], self.lengths[1] + 1))
        rows, places, tags = [], [], []
        for kind in kinds:
            keys, answers = self.pools(kind)
            key = _pick(rng, keys)
            answer = _pick(rng, answers, (key,))
            seq = [_pick(rng, self.taught, (key,)) for _ in range(length - 1)]
            place = int(rng.integers(0, length - 1 - self.gap_min))
            seq[place], seq[place + 1] = key, answer
            rows.append([*seq, key])
            places.append(place)
            tags.append((kind, f"L{length}"))
        seq_a = np.array(rows)
        at = np.array(places) + 1
        moments = [Moment(self.x(seq_a[:, t])) for t in range(length - 1)]
        moments.append(Moment(self.x(seq_a[:, -1]), label=seq_a[np.arange(BATCH), at],
                              slots={"answer": at}, tags=tags))
        return moments


class TwoHops(Task):
    """Pairs ``first second`` told as consecutive moments, among them ``a b`` and ``b c``; then
    ``a`` answers ``b``, and ``THINK`` answers ``c``: a read addressed by what the read before
    found, the sequence battery's task. A row's kind names ``b`` as the second hop's key and ``c``
    as its answer."""

    name = "twohop"
    flags = ("THINK",)
    pairs = 4

    def batch(self, rng: np.random.Generator, kinds: list[str]) -> list[Moment]:
        rows, firsts, seconds = [], [], []
        for kind in kinds:
            keys, answers = self.pools(kind)
            a = _pick(rng, self.taught)
            b = _pick(rng, keys, (a,))
            c = _pick(rng, answers, (a, b))
            used = (a, b, c)
            others = [int(t) for t in rng.permutation(self.taught) if int(t) not in used]
            pairs = [(a, b), (b, c)]
            pairs += [(others[2 * i], others[2 * i + 1]) for i in range(self.pairs - 2)]
            order = rng.permutation(self.pairs)
            firsts.append([pairs[int(i)][0] for i in order])
            seconds.append([pairs[int(i)][1] for i in order])
            rows.append((a, b, c))
        a, b, c = (np.array(col) for col in zip(*rows, strict=True))
        first, second = np.array(firsts), np.array(seconds)
        moments = []
        for i in range(self.pairs):
            moments.append(Moment(self.x(first[:, i])))
            moments.append(Moment(self.x(second[:, i])))
        for keys, answers, hop, x in ((a, b, "hop1", self.x(a)),
                                      (b, c, "hop2", self.x(np.full(len(a), -1), "THINK"))):
            kinds_here = [_class_of(int(k), int(v), self.new_set)
                          for k, v in zip(keys, answers, strict=True)]
            moments.append(Moment(x, label=answers,
                                  tags=[(kind, hop, f"{hop}_{kind}") for kind in kinds_here]))
        return moments


class Copying(Task):
    """``START``, distinct tokens, ``GO``; then each token in turn, the brain hearing the token
    just answered (its own answer in closed loop at test). A panel holds rows of taught tokens,
    of never-taught tokens and of the two mixed; each answer's condition reads the token heard
    and the token asked."""

    name = "copying"
    flags = ("START", "GO")
    lengths = (3, 6)

    def kinds_of_a_batch(self) -> list[str]:
        return [("taught", "mixed", "new", "mixed")[i % 4] for i in range(BATCH)]

    def batch(self, rng: np.random.Generator, kinds: list[str]) -> list[Moment]:
        length = int(rng.integers(self.lengths[0], self.lengths[1] + 1))
        if "new" in kinds:  # as many as the never-taught tokens allow
            length = min(length, NEW_TOKENS)
        rows = []
        for kind in kinds:
            pool = (self.taught if kind == "taught" else self.new if kind == "new"
                    else np.arange(TOKENS))
            rows.append([int(t) for t in rng.permutation(pool)[:length]])
        seq = np.array(rows)
        n = len(seq)
        moments = [Moment(self.x(np.full(n, -1), "START"))]
        moments += [Moment(self.x(seq[:, t])) for t in range(length)]

        def tags(keys: np.ndarray | None, answers: np.ndarray, j: int) -> list[tuple[str, ...]]:
            out = []
            for r in range(n):
                key = int(answers[r]) if keys is None else int(keys[r])
                kind = _class_of(key, int(answers[r]), self.new_set)
                out.append((kind, f"p{j}", kinds[r]))
            return out

        moments.append(Moment(self.x(np.full(n, -1), "GO"), label=seq[:, 0],
                              slots={"answer": np.full(n, 1)}, tags=tags(None, seq[:, 0], 0)))
        for j in range(length - 1):
            moments.append(Moment(self.x(seq[:, j]), label=seq[:, j + 1], feedback=True,
                                  slots={"answer": np.full(n, j + 2)},
                                  tags=tags(seq[:, j], seq[:, j + 1], j + 1)))
        return moments

    def panels(self, batches: int) -> dict[str, list[list[Moment]]]:
        """``test`` in closed loop, where a token the brain cannot say leaves the next moment
        hearing another, and ``forced``, the same rows hearing the token told: only there does a
        moment's key name what was heard."""
        closed = super().panels(batches)["test"]
        forced = [[replace(m, feedback=False) for m in moments] for moments in closed]
        return {"test": closed, "forced": forced}


def _twins(rng: np.random.Generator, firsts: Any, seconds: Any, count: int,
           keep: Any = None) -> list[tuple[int, int, int]]:
    """``count`` pairs of names in rows ``(a, b, duplicate)``, each pair twice: its first name
    repeated and its second, so that a pair's two rows ask for different names."""
    rows: list[tuple[int, int, int]] = []
    while len(rows) < 2 * count:
        a, b = _pick(rng, firsts), _pick(rng, seconds)
        if a == b or (keep is not None and not keep(a, b)):
            continue
        rows += [(a, b, a), (a, b, b)]
    return rows


class Suppression(Task):
    """``A B S GO``, ``S`` one of the two: the one not repeated. The parts of the indirect object
    without its frame; ``A B A`` holds an induction's answer at the duplicate's moment and
    ``A B B`` does not."""

    name = "suppression"
    flags = ("GO",)
    twins = True

    def moments(self, rows: list[tuple[int, int, int]], tags: list[tuple[str, ...]]
                ) -> list[Moment]:
        a, b, s = (np.array(c) for c in zip(*rows, strict=True))
        n = len(rows)
        answer = np.where(s == a, b, a)
        earlier = np.where(s == a, 0, 1)
        return [
            Moment(self.x(a)), Moment(self.x(b)),
            Moment(self.x(s), mark="dup", source=s, slots={"dup": earlier}),
            Moment(self.x(np.full(n, -1), "GO"), label=answer, avoid=s,
                   slots={"answer": np.where(s == a, 1, 0), "dup": earlier,
                          "dup2": np.full(n, 2)}, tags=tags),
        ]

    def tagged(self, rows: list[tuple[int, int, int]]) -> list[tuple[str, ...]]:
        out = []
        for a, b, s in rows:
            answer = b if s == a else a
            order = "abb" if s == b else "bab"  # abb: the second name repeats, the first answers
            out.append((_class_of(s, answer, self.new_set), order))
        return out

    def pair_rows(self, rng: np.random.Generator, panel: str) -> list[tuple[int, int, int]]:
        if panel == "taught":
            return _twins(rng, self.taught, self.taught, BATCH // 2)
        if panel == "new":
            return _twins(rng, self.new, self.new, BATCH // 2)
        rows: list[tuple[int, int, int]] = []  # one of each pair never-taught, in either place
        for i in range(BATCH // 2):
            pools = (self.new, self.taught) if i % 2 else (self.taught, self.new)
            rows += _twins(rng, *pools, 1)
        return rows

    def train(self, rng: np.random.Generator) -> list[Moment]:
        rows = []
        for _ in range(BATCH):
            a, b = (int(t) for t in rng.permutation(self.taught)[:2])
            rows.append((a, b, a if rng.random() < 0.5 else b))
        return self.moments(rows, self.tagged(rows))

    def panels(self, batches: int) -> dict[str, list[list[Moment]]]:
        rng = self.test_rng()
        out = {}
        for panel in ("taught", "half", "new"):
            batch = []
            for _ in range(batches):
                rows = self.pair_rows(rng, panel)
                batch.append(self.moments(rows, self.tagged(rows)))
            out[panel] = batch
        return out


class Which(Suppression):
    """``A B S GO``: ``first`` or ``second``, the place of the one not repeated. Suppression whose
    answer needs no route: a token never taught as an answer can be the one selected, and a brain
    without a route to it can say where it stood. ``key_new`` and ``answer_new`` of a suppression
    panel read here apart from "say the taught name"."""

    name = "which"
    extra_actions = ("first", "second")

    def moments(self, rows: list[tuple[int, int, int]], tags: list[tuple[str, ...]]
                ) -> list[Moment]:
        moments = super().moments(rows, tags)
        asked = moments[-1]
        at_first = asked.slots["answer"] == 0
        moments[-1] = replace(asked, label=np.where(at_first, self.words, self.words + 1),
                              source=asked.label)
        return moments


class IndirectObject(Suppression):
    """``when A and B (went (out)) S gave (a drink) to``: the one not repeated. A batch is one
    shape, its rows pairs of names each asked twice. Panels: ``ioi`` (pairs training shows, its
    shapes), ``ioi_pair`` (taught names never heard together), ``ioi_shape`` (a duplicate position
    no lesson reached), ``ioi_half`` (one name of each pair never-taught), ``ioi_new`` (both, the
    named task)."""

    name = "ioi"
    flags = ()
    words = len(WORDS)
    twins = True

    def x(self, tokens: Any, *flags: str) -> np.ndarray:
        return onehot(tokens, self.inputs)

    def frame(self, rows: list[tuple[int, int, int]], shape: tuple[int, int],
              tags: list[tuple[str, ...]]) -> list[Moment]:
        a, b, s = (np.array(c) for c in zip(*rows, strict=True))
        n = len(rows)
        pre, post = PRE[shape[0]], POST[shape[1]]
        stream = [np.full(n, WHEN), a, np.full(n, AND), b, *(np.full(n, w) for w in pre), s,
                  *(np.full(n, w) for w in post)]
        at = 4 + len(pre)  # the duplicate's moment
        earlier = np.where(s == a, 1, 3)
        moments = [Moment(self.x(w)) for w in stream]
        moments[at] = Moment(self.x(s), mark="dup", source=s, slots={"dup": earlier})
        moments.append(self.asked(a, b, s, earlier, at, tags))
        return moments

    def asked(self, a: np.ndarray, b: np.ndarray, s: np.ndarray, earlier: np.ndarray, at: int,
              tags: list[tuple[str, ...]]) -> Moment:
        return Moment(self.x(np.full(len(a), TO)), label=np.where(s == a, b, a), avoid=s,
                      slots={"answer": np.where(s == a, 3, 1), "dup": earlier,
                             "dup2": np.full(len(a), at)}, tags=tags)

    def tagged(self, rows: list[tuple[int, int, int]], shape: tuple[int, int] = (2, 3)
               ) -> list[tuple[str, ...]]:
        tag = (f"pre{shape[0]}", f"post{shape[1]}")
        return [(*row, *tag) for row in super().tagged(rows)]

    def train(self, rng: np.random.Generator) -> list[Moment]:
        shape = TRAIN_SHAPES[int(rng.integers(len(TRAIN_SHAPES)))]
        rows: list[tuple[int, int, int]] = []
        while len(rows) < BATCH:
            a, b = (int(t) for t in rng.permutation(self.taught)[:2])
            if not self.split.held(a, b):
                rows.append((a, b, a if rng.random() < 0.5 else b))
        return self.frame(rows, shape, self.tagged(rows, shape))

    def panels(self, batches: int) -> dict[str, list[list[Moment]]]:
        rng = self.test_rng()
        held = self.split.held

        def seen(a: int, b: int) -> bool:
            return not held(a, b)

        out: dict[str, list[list[Moment]]] = {}

        def built(draw: Any, shapes: tuple[tuple[int, int], ...]) -> list[list[Moment]]:
            batch = []
            for i in range(batches):
                shape = shapes[i % len(shapes)]
                rows = draw()
                batch.append(self.frame(rows, shape, self.tagged(rows, shape)))
            return batch

        out["ioi"] = built(lambda: _twins(rng, self.taught, self.taught, BATCH // 2, seen),
                           TRAIN_SHAPES)
        out["ioi_pair"] = built(lambda: _twins(rng, self.taught, self.taught, BATCH // 2, held),
                                TRAIN_SHAPES)
        out["ioi_shape"] = built(lambda: _twins(rng, self.taught, self.taught, BATCH // 2, seen),
                                 (HELD_SHAPE,))
        out["ioi_half"] = built(lambda: self.pair_rows(rng, "half"), TRAIN_SHAPES)
        out["ioi_new"] = built(lambda: self.pair_rows(rng, "new"), TRAIN_SHAPES)
        return out


class Bare(IndirectObject):
    """``A B (went (out)) S gave (a drink) to``: the indirect object with no fixed word before the
    names. Every slot that holds a name, or whose next does, then differs from sentence to sentence
    but for the boundary, so the address has no constant key to learn for the names' slots."""

    name = "bare"

    def frame(self, rows: list[tuple[int, int, int]], shape: tuple[int, int],
              tags: list[tuple[str, ...]]) -> list[Moment]:
        a, b, s = (np.array(c) for c in zip(*rows, strict=True))
        n = len(rows)
        pre, post = PRE[shape[0]], POST[shape[1]]
        stream = [a, b, *(np.full(n, w) for w in pre), s, *(np.full(n, w) for w in post)]
        at = 2 + len(pre)
        earlier = np.where(s == a, 0, 1)
        moments = [Moment(self.x(w)) for w in stream]
        moments[at] = Moment(self.x(s), mark="dup", source=s, slots={"dup": earlier})
        asked = Moment(self.x(np.full(n, TO)), label=np.where(s == a, b, a), avoid=s,
                       slots={"answer": np.where(s == a, 1, 0), "dup": earlier,
                              "dup2": np.full(n, at)}, tags=tags)
        moments.append(asked)
        return moments


class Mover(IndirectObject):
    """The frame without the duplicate, ``to`` flagged ``FIRST`` or ``SECOND``: the name at that
    place. The name mover alone, its selection told by the cue."""

    name = "mover"
    flags = ("FIRST", "SECOND")

    def x(self, tokens: Any, *flags: str) -> np.ndarray:
        return Task.x(self, tokens, *flags)

    def frame(self, rows: list[tuple[int, int, int]], shape: tuple[int, int],
              tags: list[tuple[str, ...]]) -> list[Moment]:
        a, b, s = (np.array(c) for c in zip(*rows, strict=True))
        n = len(rows)
        pre, post = PRE[shape[0]], POST[shape[1]]
        stream = [np.full(n, WHEN), a, np.full(n, AND), b, *(np.full(n, w) for w in pre),
                  *(np.full(n, w) for w in post)]
        moments = [Moment(self.x(w)) for w in stream]
        first = s == b  # the name asked is the one a sentence's duplicate leaves
        x = self.x(np.full(n, TO))
        x[first, self.flag["FIRST"]] = 1.0
        x[~first, self.flag["SECOND"]] = 1.0
        moments.append(Moment(x, label=np.where(first, a, b), avoid=np.where(first, b, a),
                              slots={"answer": np.where(first, 1, 3),
                                     "dup": np.where(first, 3, 1)}, tags=tags))
        return moments


class DuplicateBit(IndirectObject):
    """The sentence of ``ioi``, ``to`` flagged ``WHICH``: ``first`` or ``second``, where the
    duplicate's earlier mention stood. No slot holds the answer, so the reading is the duplicate
    signal through the frame, apart from moving a name."""

    name = "dupbit"
    flags = ("WHICH",)
    extra_actions = ("first", "second")

    def x(self, tokens: Any, *flags: str) -> np.ndarray:
        return Task.x(self, tokens, *flags)

    def asked(self, a: np.ndarray, b: np.ndarray, s: np.ndarray, earlier: np.ndarray, at: int,
              tags: list[tuple[str, ...]]) -> Moment:
        n = len(a)
        label = np.where(s == a, self.words, self.words + 1)
        return Moment(self.x(np.full(n, TO), "WHICH"), label=label, source=s,
                      slots={"answer": earlier, "dup": earlier, "dup2": np.full(n, at)}, tags=tags)


TASKS: dict[str, type[Task]] = {
    "previous": Previous, "duplicate": Duplicate, "successor": Successor,
    "successor_fixed": SuccessorFixed, "induction": Induction, "twohop": TwoHops,
    "copying": Copying,
    "suppression": Suppression, "which": Which, "ioi": IndirectObject, "bare": Bare,
    "mover": Mover, "dupbit": DuplicateBit,
}
FUNCTIONS = ("previous", "duplicate", "successor", "induction", "copying", "suppression", "which")


def chance(task: Task) -> dict[str, float]:
    """What the panel's conditions read without any function: a uniform guess among the tokens,
    and for a two-way task or a pair of names the one a fixed place gives."""
    if task.name in ("duplicate", "dupbit", "suppression", "which", "ioi", "bare", "mover"):
        return {"guess": 0.5}
    return {"guess": 1.0 / len(task.split.taught)}


# ------------------------------------------------------------------------------- the arms

RECIPE = gr.ARMS["recipe"]  # the sequence battery's recipe_relative at the normalized point
FD = {**RECIPE, "teaching": "deep"}  # the base composition
ARMS: dict[str, dict[str, Any]] = {
    "Fd": FD,
    "Fd2": {**FD, "recall_heads": 2},
    "FdT": FD,  # the trace silent at every lesson and act
    "Fdg": {**FD, "recall_gain": 30.0},  # recall_tokens set per task
    "Fd2g": {**FD, "recall_heads": 2, "recall_gain": 30.0},
    "FdTg": {**FD, "recall_gain": 30.0},
    "frozen": FD,  # the newborn, never taught
}
CUTS = {"FdT": ["trace"], "FdTg": ["trace"]}
MODULE = 64


def make_brain(task: Task, arm: str, founder: int, settings: dict[str, Any] | None = None,
               module: int = MODULE, replica: int = 0) -> Any:
    import cadence as cd

    gx = _module("gradients", HERE.parent / "grammar" / "gradients.py")
    options = {**ARMS[arm], **(settings or {})}
    if options.get("recall_tokens") == "task":  # the task's tokens declared without a gain
        options["recall_tokens"] = task.tokens()
    if options.get("recall_gain"):
        options.setdefault("recall_tokens", task.tokens())
    options.setdefault("backend", "cpu")  # one backend for every life, as the grammar chamber's
    brain = cd.Brain.compose(task.inputs, task.actions, modules=(module,), seed=int(founder),
                             **options)
    gx.replicate(brain, int(replica))
    if arm in CUTS:
        gx.install_cut(brain, CUTS[arm])
    return brain


# ------------------------------------------------------------------------------- the life


def new_work() -> dict[str, float]:
    return {"lessons": 0, "rows": 0, "agreed_rows": 0, "lesson_row_sweeps": 0, "acts": 0,
            "act_sweeps": 0, "refused": 0, "exhausted": 0, "free_budget_exhausted": 0}


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


class Support:
    """What the harness gives a life by oracle, to name the term that fails by what restores the
    composition when it is given. It is no option of the brain and no candidate for one: it knows
    which moment is the duplicate's. ``raw_dup``: the read at the duplicate's moment scores the
    event against the keys alone, every map at zero. ``hold_focus``, ``hold_cortex`` and
    ``hold_trace``: what the duplicate's moment left of the previous read's focus (aging as the
    slot does), of the cortex the address reads and of the working trace stands in place of what
    the words after it would leave, up to the answer. ``no_address_cortex``: the address map's
    cortex columns stay at zero through the life, every head's (no oracle, a cut). ``no_cortex``:
    the cortex of the moment before reads zero at every lesson and act, so that no map reads it
    (no oracle, a cut, the grammar harness's). ``gain_at_answer``: the gains' drive reaches the
    motor at answer moments alone. ``gain_late``: the gains stay at zero through the life's first
    ``LATE`` lessons (a schedule the harness keeps). ``center_cortex``: the maps read the cortex
    less its running mean over the life's moments, so that what moments share is not read (no
    oracle). ``cortex_unit``: the maps read the cortex of the moment before at unit norm per
    stream, the event's energy (no oracle, an emulation of a balanced read). ``cortex_scale``
    (a number, ``Support(names, scale)``): the maps read that cortex times the number (no oracle;
    0 is ``no_cortex``)."""

    NAMES = ("raw_dup", "hold_focus", "hold_cortex", "hold_trace", "no_address_cortex",
             "no_cortex", "gain_at_answer", "gain_late", "center_cortex", "cortex_unit")
    LATE = 1000
    CENTER_DECAY = 0.999
    MAPS = ("head_maps", "position_maps", "anchor_maps", "relative_maps", "slope_maps")

    def __init__(self, names: Any = (), scale: float | None = None) -> None:
        self.names = tuple(names)
        self.scale = None if scale is None else float(scale)
        # with ``gain_late``: the gains set to this value (one, or one per port) once released
        self.late_value: Any = None
        if not set(self.names) <= set(self.NAMES):
            raise ValueError(f"unknown support: {self.names}")
        self.lessons = 0
        self.mean: np.ndarray | None = None
        self.reset()

    def reset(self) -> None:
        self.focus: np.ndarray | None = None
        self.cortex: np.ndarray | None = None
        self.trace: np.ndarray | None = None

    def before(self, brain: Any, teaching: bool = False) -> None:
        """Before the moment's lesson and act: what is held stands in place. A running mean moves
        on moments of a life's lessons, and a reading leaves it as it is."""
        recall = brain.recall
        if "no_cortex" in self.names:
            recall.previous = np.zeros_like(recall.previous)
        if "cortex_unit" in self.names:
            raw = np.atleast_2d(np.asarray(recall.previous, dtype=float))
            norm = np.linalg.norm(raw, axis=1, keepdims=True)
            recall.previous = np.where(norm > 0, raw / np.where(norm > 0, norm, 1.0), raw)
        if self.scale is not None:
            recall.previous = self.scale * np.asarray(recall.previous)
        if "center_cortex" in self.names:
            raw = np.atleast_2d(np.asarray(recall.previous))
            active = raw.any(axis=1)
            if active.any():
                if teaching:
                    seen = raw[active].mean(axis=0)
                    self.mean = (seen if self.mean is None else self.CENTER_DECAY * self.mean
                                 + (1.0 - self.CENTER_DECAY) * seen)
                if self.mean is not None:
                    recall.previous = np.where(active[:, None], raw - self.mean, raw)
        if self.focus is not None:
            recall.focus = self.focus.copy()
        if self.cortex is not None:
            recall.previous = self.cortex.copy()
        if self.trace is not None:
            brain.working_memory.trace = self.trace.copy()

    def act(self, brain: Any, m: Moment, x: np.ndarray, work: dict[str, float]) -> np.ndarray:
        recall = brain.recall
        gains = None
        if "gain_at_answer" in self.names and m.label is None:
            gains = recall.gains
            recall.gains = np.zeros_like(gains)
        try:
            if m.mark != "dup" or "raw_dup" not in self.names:
                return _act(brain, x, work)
            saved = {name: getattr(recall, name) for name in self.MAPS}
            for name, maps in saved.items():
                setattr(recall, name, [tuple(np.zeros_like(a) for a in head) for head in maps])
            try:
                return _act(brain, x, work)
            finally:
                for name, maps in saved.items():
                    setattr(recall, name, maps)
        finally:
            if gains is not None:
                recall.gains = gains

    def taught(self, brain: Any) -> None:
        """After a lesson: what a cut keeps at zero."""
        if "no_address_cortex" in self.names:
            recall = brain.recall
            for h in range(recall.heads):
                w_x, w_c, b = recall.head_map(h)
                recall._set_map(h, w_x, np.zeros_like(w_c), b)
        self.lessons += 1
        if "gain_late" in self.names and self.lessons <= self.LATE:
            brain.recall.gains = np.zeros_like(brain.recall.gains)
        if self.late_value is not None and self.lessons == self.LATE:
            brain.recall.gains = np.broadcast_to(
                np.asarray(self.late_value, dtype=float), np.shape(brain.recall.gains)).copy()

    def after(self, brain: Any, m: Moment) -> None:
        """After the moment's act: the duplicate's moment gives what is held, and a held focus
        ages with the slot it points to."""
        if m.mark == "dup":
            if "hold_focus" in self.names:
                self.focus = np.array(brain.recall.focus)
            if "hold_cortex" in self.names:
                self.cortex = np.array(brain.recall.previous)
            if "hold_trace" in self.names:
                self.trace = np.array(brain.working_memory.trace)
        elif self.focus is not None:
            self.focus = np.where(self.focus >= 0, self.focus + 1, -1)


CORTEX_MAPS = ("head_maps", "position_maps", "anchor_maps", "relative_maps", "slope_maps",
               "grid_maps")


def _cortex_columns(recall: Any) -> dict[str, list[np.ndarray]]:
    """Every map's cortex columns, per kind and head (``w_c`` of ``(w_x, w_c, b)``)."""
    return {name: [np.array(head[1]) for head in getattr(recall, name, None) or []]
            for name in CORTEX_MAPS}


def _scale_cortex_step(recall: Any, before: dict[str, list[np.ndarray]], rate: float) -> None:
    """A lesson's step on the maps' cortex columns, times ``rate``: an emulation of a slower
    rate on what every moment shares (``cortex_rate`` of a life's spec)."""
    for name, columns in before.items():
        maps = getattr(recall, name, None) or []
        setattr(recall, name, [
            (head[0], old + rate * (np.asarray(head[1]) - old), head[2])
            for head, old in zip(maps, columns, strict=True)])


def teach(brain: Any, moments: list[Moment], work: dict[str, float],
          support: Support | None = None, cortex_rate: float | None = None) -> None:
    """One training batch on fresh streams: a lesson before the act at every answer moment.
    ``cortex_rate`` scales each lesson's step on the maps' cortex columns."""
    support = support or Support()
    brain.reset()
    support.reset()
    for m in moments:
        support.before(brain, teaching=True)
        if m.label is not None:
            before = None if cortex_rate is None else _cortex_columns(brain.recall)
            try:
                report = brain.lesson(m.x, m.label)
                work["lessons"] += 1
                work["rows"] += report.get("attempted_presentations", 0.0)
                work["agreed_rows"] += report.get("agreed_presentations", 0.0)
                work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
                work["free_budget_exhausted"] += report.get("free_budget_exhausted", 0.0)
            except RuntimeError:
                work["refused"] += 1
            if before is not None:
                _scale_cortex_step(brain.recall, before, float(cortex_rate))
            support.taught(brain)
        support.act(brain, m, m.x, work)
        support.after(brain, m)


# ----------------------------------------------------------------------------- the reading


class Tally:
    """The answers and the reads of a panel, per condition."""

    def __init__(self) -> None:
        self.cells: dict[str, dict[str, Any]] = {}
        self.hits: list[np.ndarray] = []
        self.said: list[np.ndarray] = []

    def cell(self, name: str) -> dict[str, Any]:
        return self.cells.setdefault(name, {"n": 0, "answer": 0.0, "says_avoid": 0.0, "pair": 0.0,
                                            "reads": 0, "first": 0.0, "first_any": 0.0,
                                            "first_avoid": 0.0, "mass": 0.0, "mass_avoid": 0.0,
                                            "right_wrong": 0.0, "ports": {}})

    def add(self, m: Moment, action: np.ndarray, read: dict[str, Any] | None) -> None:
        assert m.label is not None
        n = len(action)
        hit = action == m.label
        self.hits.append(hit)
        self.said.append(action)
        for r in range(n):
            for name in ("all", *(m.tags[r] if m.tags else ())):
                c = self.cell(name)
                c["n"] += 1
                c["answer"] += float(hit[r])
                if m.avoid is not None:
                    c["says_avoid"] += float(action[r] == m.avoid[r])
                    c["pair"] += float(hit[r] or action[r] == m.avoid[r])
                if read is not None and read["held"][r]:
                    c["reads"] += 1
                    c["first"] += float(read["first"][r])
                    c["first_any"] += float(read["first_any"][r])
                    c["right_wrong"] += float(read["first"][r] and not hit[r])
                    c["mass"] += float(read["mass"][r])
                    if m.avoid is not None:
                        c["first_avoid"] += float(read["first_avoid"][r])
                        c["mass_avoid"] += float(read["mass_avoid"][r])
                    for port, named in read["ports"].items():
                        c["ports"][port] = c["ports"].get(port, 0.0) + float(named[r])

    def summary(self, twins: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"conditions": {}}
        for name, c in sorted(self.cells.items()):
            n = max(c["n"], 1)
            cell = {"n": c["n"], "answer": round(c["answer"] / n, 4)}
            if c["says_avoid"] or c["pair"]:
                cell["says_avoid"] = round(c["says_avoid"] / n, 4)
                cell["pair"] = round(c["pair"] / n, 4)
            if c["reads"]:
                reads = c["reads"]
                for key in ("first", "first_any", "first_avoid", "mass", "mass_avoid",
                            "right_wrong"):
                    cell[key] = round(c[key] / reads, 4)
                cell["reads"] = reads
                ports = {k: round(v / reads, 4) for k, v in c["ports"].items()}
                cell["ports"] = ports
                if ports:
                    cell["port_best"] = max(ports.values())
            out["conditions"][name] = cell
        if twins and self.hits:
            hits = np.concatenate(self.hits)
            said = np.concatenate(self.said)
            out["both"] = round(float(np.mean(hits[0::2] & hits[1::2])), 4)
            out["differ"] = round(float(np.mean(said[0::2] != said[1::2])), 4)
        return out


def _delivered(rows: np.ndarray, top: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """The token a slot holds on a port (its largest coordinate), -1 where it holds none."""
    held = bank[rows, top]
    return np.where(held.max(axis=1) > 0, held.argmax(axis=1), -1)


def read_of(heads: tuple[np.ndarray | None, ...], values: np.ndarray, nexts: np.ndarray,
            source: np.ndarray, avoid: np.ndarray | None) -> dict[str, Any] | None:
    """What the read found at a moment, before that moment's write: per row whether head 0's first
    choice holds the right token on its value or its next (``first``), whether any head's does,
    whether it holds the token to avoid, the attention on slots that hold each, and per port
    whether the token most driven is the right one."""
    p0 = heads[0] if heads else None
    if p0 is None or p0.shape[1] != values.shape[1]:
        return None
    rows = np.arange(len(p0))
    held = source >= 0
    token = np.where(held, source, 0)
    out: dict[str, Any] = {"held": held, "ports": {}}
    first_any = np.zeros(len(p0), dtype=bool)
    mass = np.zeros(len(p0))
    for h, p in enumerate(heads):
        if p is None:
            continue
        top = np.argmax(p, axis=1)
        on_value, on_next = _delivered(rows, top, values), _delivered(rows, top, nexts)
        right = (on_value == source) | (on_next == source)
        first_any |= right
        slots_hold = (values[rows, :, token] > 0) | (nexts[rows, :, token] > 0)
        mass = np.maximum(mass, np.sum(p * slots_hold, axis=1))
        if h == 0:
            out["first"] = right
            if avoid is not None:
                out["first_avoid"] = (on_value == avoid) | (on_next == avoid)
                hold_avoid = (values[rows, :, avoid] > 0) | (nexts[rows, :, avoid] > 0)
                out["mass_avoid"] = np.sum(p * hold_avoid, axis=1)
        out["ports"][f"recall{h}"] = np.argmax(np.einsum("bs,bsw->bw", p, values), axis=1) == source
        out["ports"][f"next{h}"] = np.argmax(np.einsum("bs,bsw->bw", p, nexts), axis=1) == source
    out["first_any"] = first_any
    out["mass"] = mass
    return out


def motor_inputs(brain: Any, tokens: dict[str, np.ndarray], actions: int) -> dict[str, Any]:
    """What drives each named token's motor neuron at the act just settled, per row: the
    association's synapses, the gains' drive and everything else (the motor's own synapses, the
    bias). Read after the act; a token below zero reads nothing."""
    graph = brain.connectome
    pre, post = np.asarray(graph.pre), np.asarray(graph.post)
    motor = np.asarray(brain.motor_index)[:actions]
    association = np.asarray(graph.populations["association"])
    weights = np.asarray(brain.brain.weights)
    activation = np.atleast_2d(np.asarray(brain.basal_ganglia.state.activation))
    drive = np.atleast_2d(np.asarray(brain.basal_ganglia._drive))
    bias = np.asarray(brain.brain.bias)
    from_association = np.isin(pre, association)
    rows = np.arange(len(activation))
    out: dict[str, Any] = {}
    for name, token in tokens.items():
        known = token >= 0
        neuron = motor[np.where(known, token, 0)]
        cortex, other = np.zeros(len(rows)), np.zeros(len(rows))
        for r in rows[known]:
            into = post == neuron[r]
            parts = weights[into] * activation[r, pre[into]]
            cortex[r] = parts[from_association[into]].sum()
            other[r] = parts[~from_association[into]].sum() + bias[neuron[r]]
        out[name] = {"association": cortex, "gain": drive[rows, neuron], "other": other,
                     "known": known}
    return out


def _attended(brain: Any, x: np.ndarray) -> list[np.ndarray]:
    """Every head's attention at the moment ``x``, before its settle. ``stimulus`` addresses the
    moment and ``act`` addresses it again, so the act that follows is unchanged (a test reads
    it)."""
    brain.stimulus(x)
    heads = brain.recall.attended_heads
    return [] if heads is None else [p for p in heads if p is not None]


def logit_terms(recall: Any, m: Moment, t: int) -> dict[str, dict[str, np.ndarray]]:
    """Head 0's logits at a moment by what they come from, on the slots a suppressing read tells
    apart: the slot of each named token and the slot before it (whose next holds it). Read after
    ``stimulus``, before the act. ``key`` is the event heard against a slot's key, ``event``,
    ``cortex`` and ``bias`` the address map's rows against it, and ``position_*`` every position
    map's same rows at the slot's buckets; with ``content`` and ``position`` their sums."""
    moment = recall.last
    rows = np.arange(len(m.x))
    x, c = moment.inputs, moment.cortex
    w_x, w_c, b = recall.head_map(0)
    temperature = recall.temperature
    keys = recall.keys
    parts: dict[str, np.ndarray] = {
        "key": np.einsum("bk,bmk->bm", moment.key, keys) / temperature,
        "event": np.einsum("bk,bmk->bm", x @ w_x, keys) / temperature,
        "cortex": np.einsum("bk,bmk->bm", c @ w_c, keys) / temperature,
        "bias": np.einsum("k,bmk->bm", b, keys) / temperature,
    }
    position = {"event": 0.0, "cortex": 0.0, "bias": 0.0}
    tables = [(recall.position_maps[0], recall.buckets()[None, :].repeat(len(x), axis=0))]
    if recall.anchor_maps:
        tables.append((recall.anchor_maps[0], recall.anchored_buckets()))
    if recall.relative_maps:
        tables.append((recall.relative_maps[0], recall.relative_buckets()))
    for maps, buckets in tables:
        for name, values in zip(("event", "cortex", "bias"),
                                (x @ maps[0], c @ maps[1], np.tile(maps[2], (len(x), 1))),
                                strict=True):
            position[name] = position[name] + np.take_along_axis(values, buckets, axis=1)
    if recall.slope_maps:
        lean = (recall.ages() / recall.SLOPE)[None, :]
        maps = recall.slope_maps[0]
        for name, value in zip(("event", "cortex", "bias"),
                               (x @ maps[0], c @ maps[1], np.full(len(x), float(maps[2][0]))),
                               strict=True):
            position[name] = position[name] + np.reshape(value, (len(x), 1)) * lean
    parts.update({f"position_{name}": value for name, value in position.items()})
    parts["content"] = parts["key"] + parts["event"] + parts["cortex"] + parts["bias"]
    parts["position"] = sum(position.values())
    slots = keys.shape[1]
    out: dict[str, dict[str, np.ndarray]] = {term: {} for term in parts}
    for name, at in m.slots.items():
        place = slots - (t - at)
        for suffix, offset in (("", 0), ("_before", -1)):
            where = np.clip(place + offset, 0, slots - 1)
            for term, values in parts.items():
                out[term][name + suffix] = values[rows, where]
    return out


def _lesion(recall: Any, rows: np.ndarray, slots: np.ndarray) -> None:
    """Zero the key, the value and the next of a slot per row."""
    keep = (slots >= 0) & (slots < recall.keys.shape[1])
    rows, slots = rows[keep], slots[keep]
    for name in ("keys", "values", "nexts"):
        bank = np.array(getattr(recall, name), copy=True)
        bank[rows, slots] = 0.0
        setattr(recall, name, bank)


def intervene(brain: Any, m: Moment, plan: dict[str, tuple[str, ...]], t: int,
              rng: np.random.Generator) -> bool:
    """A causal intervention before the act of a moment named in ``plan``; returns whether the
    read is silent at this act."""
    name = "answer" if m.label is not None else m.mark
    recall = brain.recall
    if recall is None:
        return False
    if name in plan.get("trace", ()) and brain.working_memory is not None:
        brain.working_memory.trace = np.zeros_like(brain.working_memory.trace)
    if name in plan.get("cortex", ()):
        recall.previous = np.zeros_like(recall.previous)
    if name in plan.get("focus", ()):
        recall.focus = np.full_like(recall.focus, -1)
    rows = np.arange(len(m.x))
    if name in plan.get("erase", ()) and recall.slots:
        for p in _attended(brain, m.x):
            _lesion(recall, rows, np.argmax(p, axis=1))
    if name in plan.get("erase_other", ()) and recall.slots and m.source is not None:
        heads = _attended(brain, m.x)
        tokens = np.where(m.source >= 0, m.source, 0)
        slots = np.full(len(rows), -1)
        for r in rows:
            holds = (recall.values[r, :, tokens[r]] > 0) | (recall.nexts[r, :, tokens[r]] > 0)
            tops = {int(np.argmax(p[r])) for p in heads}
            free = [s for s in range(recall.keys.shape[1]) if not holds[s] and s not in tops]
            if free:
                slots[r] = free[int(rng.integers(len(free)))]
        _lesion(recall, rows, slots)
    if name in plan.get("erase_source", ()) and "answer" in m.slots:
        at = recall.keys.shape[1] - (t - m.slots["answer"])  # the slot of the answer's moment
        at = np.where(m.slots["answer"] >= 0, at, -1)
        _lesion(recall, rows, at)
        before = np.array(recall.nexts, copy=True)  # the next route: the slot before it
        known = at >= 1
        before[rows[known], at[known] - 1] = 0.0
        recall.nexts = before
    twin = rows ^ 1  # a twin panel's rows come in pairs
    if name in plan.get("swap_cortex", ()):
        recall.previous = recall.previous[twin]
    if name in plan.get("swap_trace", ()) and brain.working_memory is not None:
        brain.working_memory.trace = brain.working_memory.trace[twin]
    if name in plan.get("swap_focus", ()):
        recall.focus = recall.focus[twin]
    if name in plan.get("swap_slot", ()) and "dup2" in m.slots:
        at = recall.keys.shape[1] - (t - m.slots["dup2"])  # the duplicate's slot, and the next of
        for bank_name, offset in (("keys", 0), ("values", 0), ("nexts", 0), ("nexts", -1)):
            bank = np.array(getattr(recall, bank_name), copy=True)  # the slot before it
            bank[rows, at + offset] = bank[twin, at + offset]
            setattr(recall, bank_name, bank)
    silent = plan.get("silence", ())
    return name in silent or "all" in silent


def read_episode(brain: Any, task: Task, moments: list[Moment], tally: Tally,
                 work: dict[str, float], plan: dict[str, tuple[str, ...]] | None,
                 rng: np.random.Generator, dup: dict[str, list[float]] | None,
                 terms: dict[str, Any] | None = None, support: Support | None = None,
                 vote: dict[str, Any] | None = None) -> list[np.ndarray]:
    """One batch on fresh streams, greedy, in closed loop where the task feeds answers back; returns
    each answer moment's hits. ``terms`` keeps the logit's terms per moment and row kind, ``vote``
    what drives the answer's, the avoided token's and the winner's motor neurons per row kind."""
    support = support or Support()
    brain.reset()
    support.reset()
    previous = None
    hits = []
    recall = brain.recall
    for t, m in enumerate(moments):
        x = m.x
        if m.feedback and previous is not None:
            x = onehot(previous, task.inputs)
            m = Moment(x, m.label, m.source, m.avoid, m.slots, m.mark, m.feedback, m.tags)
        support.before(brain)
        silent = bool(plan) and intervene(brain, m, plan or {}, t, rng)
        saved = None
        if silent:
            saved = recall.amplitude
            recall.amplitude = 0.0
        values = nexts = None
        scored = m.label is not None or m.mark == "dup"
        if scored and recall is not None:
            values, nexts = np.array(recall.values, copy=True), np.array(recall.nexts, copy=True)
            if plan is None and terms is not None and "dup" in m.slots and recall.slots:
                brain.stimulus(x)
                found = logit_terms(recall, m, t)
                answer = moments[-1].tags or [("all",)] * len(x)
                for r, row_tags in enumerate(answer):
                    kind = terms.setdefault("answer" if m.label is not None else m.mark, {})
                    kind = kind.setdefault(row_tags[0], {})
                    for term, named in found.items():
                        cell = kind.setdefault(term, {})
                        for name, at in named.items():
                            cell.setdefault(name, []).append(float(at[r]))
        action = support.act(brain, m, x, work)
        support.after(brain, m)
        if saved is not None:
            recall.amplitude = saved
        if (m.mark == "dup" and plan is None and dup is not None and values is not None
                and m.source is not None):
            heads = recall.attended_heads
            if heads and heads[0] is not None and heads[0].shape[1] == values.shape[1]:
                rows = np.arange(len(x))
                top = np.argmax(heads[0], axis=1)
                dup["first_value"] += list(_delivered(rows, top, values) == m.source)
                dup["first_next"] += list(_delivered(rows, top, nexts) == m.source)
                at = values.shape[1] - (t - m.slots["dup"])
                dup["mass"] += list(heads[0][rows, at])
        if m.label is not None:
            if plan is None and vote is not None and recall is not None and m.tags:
                named = {"answer": m.label, "winner": action}
                if m.avoid is not None:
                    named["avoid"] = m.avoid
                found = motor_inputs(brain, named, task.actions)
                for r, row_tags in enumerate(m.tags):
                    kind = vote.setdefault(row_tags[0], {})
                    for name, parts in found.items():
                        if parts["known"][r]:
                            cell = kind.setdefault(name, {})
                            for part in ("association", "gain", "other"):
                                cell.setdefault(part, []).append(float(parts[part][r]))
            read = None
            if values is not None:
                source = m.label if m.source is None else m.source
                heads = recall.attended_heads
                if heads:
                    read = read_of(tuple(heads), values, nexts, source, m.avoid)
            tally.add(m, action, read)
            hits.append(action == m.label)
            previous = action
    return hits


def read_panel(brain: Any, task: Task, batches: list[list[Moment]],
               plan: dict[str, tuple[str, ...]] | None = None, seed: int = 0,
               support: Support | None = None, votes: bool = False) -> dict[str, Any]:
    """A panel on a fork of ``brain``: answers and, without an intervention, the read's own; with
    ``votes`` what drives the motor neurons."""
    fork = copy.deepcopy(brain)
    tally = Tally()
    work = new_work()
    rng = np.random.default_rng(np.random.SeedSequence([task.founder, 77, seed]))
    dup: dict[str, list[float]] = {"first_value": [], "first_next": [], "mass": []}
    terms: dict[str, Any] = {}
    vote: dict[str, Any] | None = {} if votes else None
    exact: list[np.ndarray] = []
    for moments in batches:
        hits = read_episode(fork, task, moments, tally, work, plan, rng, dup, terms, support, vote)
        exact.append(np.all(np.stack(hits), axis=0))
    out = tally.summary(twins=task.twins)
    if vote:
        out["vote"] = {kind: {name: {part: round(float(np.mean(v)), 3)
                                     for part, v in cell.items()}
                              for name, cell in named.items()}
                       for kind, named in vote.items()}
    if plan is None and terms:
        out["terms"] = {moment: {kind: {term: {name: round(float(np.mean(v)), 3)
                                               for name, v in cell.items()}
                                        for term, cell in named.items()}
                                 for kind, named in kinds.items()}
                        for moment, kinds in terms.items()}
    out["exact"] = round(float(np.mean(np.concatenate(exact))), 4)
    if dup["first_value"]:
        out["duplicate_read"] = {k: round(float(np.mean(v)), 4) for k, v in dup.items()}
    if work["exhausted"]:
        out["exhausted"] = work["exhausted"]
    return out


# a read or a carrier cut at test; ``erase_other`` is the control, a slot neither read nor holding
# the answer
CAUSAL: dict[str, dict[str, tuple[str, ...]]] = {
    "silence": {"silence": ("all",)},
    "silence_answer": {"silence": ("answer",)},
    "erase": {"erase": ("answer",)},
    "erase_source": {"erase_source": ("answer",)},
    "erase_other": {"erase_other": ("answer",)},
    "trace": {"trace": ("answer",)},
    "cortex": {"cortex": ("answer",)},
    "focus": {"focus": ("answer",)},
    "silence_dup": {"silence": ("dup",)},
    "erase_dup": {"erase": ("dup",)},
    # a twin's carrier in the row's place at the answer: an answer that follows the twin's
    # (``says_avoid``) names the carrier
    "swap_cortex": {"swap_cortex": ("answer",)},
    "swap_trace": {"swap_trace": ("answer",)},
    "swap_focus": {"swap_focus": ("answer",)},
    "swap_slot": {"swap_slot": ("answer",)},
    "swap_all": {"swap_cortex": ("answer",), "swap_trace": ("answer",),
                 "swap_focus": ("answer",), "swap_slot": ("answer",)},
}
DUP_CAUSAL = ("silence_dup", "erase_dup")
TWIN_CAUSAL = ("swap_cortex", "swap_trace", "swap_focus", "swap_slot", "swap_all")


def _ridge(x: np.ndarray, y: np.ndarray, reads: dict[str, tuple[np.ndarray, np.ndarray]],
           penalty: float = 1.0) -> dict[str, float]:
    """A ridge decoder of the sign of ``y`` fit on ``x``, read on each of ``reads``."""
    mean = x.mean(axis=0)
    a = x - mean
    w = np.linalg.solve(a.T @ a + penalty * np.eye(a.shape[1]), a.T @ y)
    return {name: round(float(np.mean(((tx - mean) @ w > 0) == (ty > 0))), 4)
            for name, (tx, ty) in reads.items()}


def _bit_states(brain: Any, task: Task, batches: list[list[Moment]],
                support: Support | None = None) -> dict[str, Any]:
    """Which name repeated, against what each carrier holds: the cortex and the trace right after
    the duplicate's moment and just before the answer's, and the cortex the answer settles to."""
    fork = copy.deepcopy(brain)
    support = support or Support()
    work = new_work()
    states: dict[str, list[np.ndarray]] = {}
    bits: list[np.ndarray] = []
    for moments in batches:
        fork.reset()
        support.reset()
        for i, m in enumerate(moments):
            support.before(fork)
            support.act(fork, m, m.x, work)
            support.after(fork, m)
            names = ["after_dup"] if m.mark == "dup" else []
            if i == len(moments) - 2:
                names.append("before_answer")
            if m.label is not None:
                names.append("after_answer")
            for name in names:
                states.setdefault(f"cortex_{name}", []).append(np.array(fork.recall.previous))
                if fork.working_memory is not None:
                    states.setdefault(f"trace_{name}", []).append(
                        np.array(fork.working_memory.trace))
        asked = moments[-1]
        bits.append(np.where(asked.slots["dup"] == min(asked.slots["dup"]), 1.0, -1.0))
    return {"x": {k: np.concatenate(v) for k, v in states.items()}, "y": np.concatenate(bits)}


def decoding(brain: Any, task: Task, batches: int = 40, support: Support | None = None
             ) -> dict[str, Any]:
    """Whether the cortex and the trace hold which name repeated, by a read-only ridge decoder fit
    on taught sentences and read on others: the information a learned readout could use. Above
    chance it says the bit is there, and says nothing of whether the brain's own weights read it."""
    tests = task.panels(batches)
    fit_on, read_on = tests["ioi"][: batches - 10], tests["ioi"][batches - 10:]
    fit = _bit_states(brain, task, fit_on, support)
    reads = {"taught": _bit_states(brain, task, read_on, support),
             "new": _bit_states(brain, task, tests["ioi_new"][:10], support),
             "half": _bit_states(brain, task, tests["ioi_half"][:10], support)}
    out = {}
    for name, x in fit["x"].items():
        if x.std() == 0:
            continue
        out[name] = _ridge(x, fit["y"], {k: (v["x"][name], v["y"]) for k, v in reads.items()})
    return out


def reading(brain: Any, task: Task, tests: dict[str, list[list[Moment]]], causal: bool,
            support: Support | None = None, votes: bool = False) -> dict[str, Any]:
    out = {}
    for name, batches in tests.items():
        panel = read_panel(brain, task, batches, support=support, votes=votes)
        if causal and brain.recall is not None:
            has_dup = any(m.mark == "dup" for m in batches[0])
            panel["causal"] = {}
            for i, (cut, plan) in enumerate(CAUSAL.items()):
                if cut in DUP_CAUSAL and not has_dup:
                    continue
                if cut in TWIN_CAUSAL and not task.twins:
                    continue
                read = read_panel(brain, task, batches, plan, seed=i, support=support)
                keep = ("answer", "says_avoid", "first", "first_avoid", "port_best", "ports")
                panel["causal"][cut] = {
                    "conditions": {k: {key: v[key] for key in keep if key in v}
                                   for k, v in read["conditions"].items()},
                    **{k: read[k] for k in ("both", "differ", "exact") if k in read}}
        out[name] = panel
    return out


def make_task(name: str, founder: int) -> Task:
    return TASKS[name](founder)


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``task``, ``arm``, ``founder``, with ``lessons`` (batches), ``panel_batches``,
    ``module``, ``init_seed`` (the brain's seed, on the founder's split, lessons and panels),
    ``replica``, ``settings`` over the arm's, ``checkpoints`` (batches with a lean reading),
    ``causal``, ``decode``, ``vote`` (what drives the motor neurons at the answers) and ``support``
    (the oracle a diagnostic life is given), ``cortex_rate`` (each lesson's step on the maps' cortex
    columns scaled, an emulation), ``gain_start`` (every port's gain at birth, an emulation)."""
    warnings.simplefilter("ignore")
    import cadence as cd

    arm, founder = spec["arm"], int(spec["founder"])
    task = make_task(spec["task"], founder)
    lessons = int(spec.get("lessons", LESSONS.get(task.name, SINGLE_LESSONS)))
    if arm == "frozen":
        lessons = 0
    module = int(spec.get("module", MODULE))
    init_seed = int(spec.get("init_seed", founder))
    started = time.monotonic()
    brain = make_brain(task, arm, init_seed, spec.get("settings"), module,
                       int(spec.get("replica", 0)))
    if spec.get("gain_start") is not None:  # an emulation: every port's gain at birth
        brain.recall.gains = np.full_like(np.asarray(brain.recall.gains),
                                          float(spec["gain_start"]))
    tests = task.panels(int(spec.get("panel_batches", 10 if task.twins else 24)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, SEEDS["train"]]))
    checkpoints = {int(c) for c in spec.get("checkpoints", [])}
    work = new_work()
    curve = []
    support = Support(spec.get("support", ()), spec.get("cortex_scale"))
    if spec.get("gain_late_value") is not None:
        support.late_value = spec["gain_late_value"]
    for b in range(1, lessons + 1):
        teach(brain, task.train(rng), work, support, spec.get("cortex_rate"))
        if b in checkpoints and b != lessons:
            curve.append({"batches": b, "panels": {
                name: panel["conditions"]["all"]
                for name, panel in reading(brain, task, tests, False, support).items()}})
    seconds_taught = time.monotonic() - started
    row: dict[str, Any] = {
        "task": task.name, "arm": arm, "founder": founder, "init_seed": init_seed,
        "replica": int(spec.get("replica", 0)), "tag": spec.get("tag"),
        "settings": {**ARMS[arm], **spec.get("settings", {})}, "cut": CUTS.get(arm, []),
        "support": list(support.names), "cortex_rate": spec.get("cortex_rate"),
        "gain_start": spec.get("gain_start"),
        "cortex_scale": support.scale,
        "lessons": lessons, "module": module, "inputs": task.inputs, "actions": task.actions,
        "parameters": int(brain.parameters()), "cadence": cd.__version__,
        "split": {"taught": list(task.split.taught), "new": list(task.split.new),
                  "pairs": [list(p) for p in task.split.pairs]},
        "chance": chance(task), "work": work, "curve": curve,
        "seconds_taught": round(seconds_taught, 1),
        "panels": reading(brain, task, tests, bool(spec.get("causal", True)), support,
                          bool(spec.get("vote"))),
    }
    if spec.get("decode"):
        row["decode"] = decoding(brain, task, support=support)
    if brain.recall is not None and getattr(brain.recall, "gain", 0):
        row["gains"] = [round(float(g), 4) for g in brain.recall.gains]
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


# -------------------------------------------------------------------------------- summary


def table(rows: list[dict[str, Any]]) -> list[str]:
    """One line per life and panel condition: the answer, the read's first choice, the mass on what
    it should read, and the answer under the read silenced and the attended slot erased."""
    lines = []
    for r in sorted(rows, key=lambda r: (r["task"], r["arm"], r["founder"], r.get("replica", 0))):
        for name, panel in r["panels"].items():
            for cond in ("taught", "key_new", "answer_new", "new", "fixed", "all"):
                c = panel["conditions"].get(cond)
                if c is None or (cond == "all" and len(panel["conditions"]) > 1 and any(
                        k in panel["conditions"] for k in KINDS)):
                    continue
                causal = panel.get("causal", {})
                cuts = " ".join(
                    f"{k[:12]} {v['conditions'][cond]['answer']:.2f}" for k, v in causal.items()
                    if cond in v["conditions"])
                lines.append(
                    f"{r['task']:12s} {r['arm']:5s} f{r['founder']}.{r.get('replica', 0)} "
                    f"{name:9s} {cond:10s} n{c['n']:4d} answer {c['answer']:.2f} "
                    f"first {c.get('first', float('nan')):.2f} "
                    f"mass {c.get('mass', float('nan')):.2f} "
                    f"avoid {c.get('says_avoid', float('nan')):.2f} | {cuts}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--arms", nargs="*", default=["Fd"], choices=list(ARMS))
    parser.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs in place of the grid")
    parser.add_argument("--out", type=Path, help="JSON lines, one row per life, appended")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--table", type=Path, help="print the table of a results file")
    args = parser.parse_args(argv)
    if args.table is not None:
        rows = [json.loads(line) for line in args.table.read_text().splitlines() if line]
        print("\n".join(table(rows)))
        return 0
    if args.out is None:
        parser.error("--out is required")
    if args.plan is not None:
        specs = json.loads(args.plan.read_text())
    else:
        if args.task is None:
            parser.error("--task or --plan is required")
        specs = [{"task": args.task, "arm": arm, "founder": f,
                  "tag": f"{args.task}/{arm}/{f}"} for arm in args.arms for f in args.founders]
    done = set()
    if args.out.exists():
        done = {json.loads(line).get("tag") for line in args.out.read_text().splitlines() if line}
    specs = [s for s in specs if s.get("tag") not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        # one process per life: nothing a life patches reaches the next
        with get_context("spawn").Pool(max(args.workers, 1), maxtasksperchild=1) as pool:
            for row in pool.imap_unordered(run_life, specs, chunksize=1):
                sink.write(json.dumps(row) + "\n")
                sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
