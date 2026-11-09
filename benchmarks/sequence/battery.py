"""The sequence battery: reads at lengths no lesson reached (cadence#232, #234, #237, #238).

Each task is a continuing stream of one-hot moments, ``BATCH`` independent streams in
lockstep. Lessons fall only on answer moments and come before the act on the same observation
(``Brain.lesson`` then ``Brain.act``); every other moment is heard without a lesson. Every act
is greedy. After the lessons a frozen test on fresh episodes follows, without lessons. Where a
task's answers are fed back (copying), the test runs in closed loop, the brain hearing its own
previous answer, and once more teacher-forced.

The brain is ``Brain.compose(inputs, actions, modules=(64,), seed=founder, episodic=False,
teaching="normalized")`` with the arm's settings beside it (``ARMS``); the arm ``composed`` is
what a new brain composes, ``Brain.compose(inputs, actions, modules=(64,), seed=founder,
episodic=False)`` with the recall memory and its own teaching point. The training episodes
come from ``SeedSequence([founder, 11])`` and the test episodes from ``SeedSequence([founder,
33])``, so every arm of a founder is taught and read on the same episodes.

Two readings beside the test: ``erase`` sets every head's anchored maps to zero after the test
and reads the same test again (does the anchor carry the read?), and ``phases`` teaches look
back's cues one group after another, the lessons split evenly, instead of mixed (what do
earlier lessons keep?).

    python benchmarks/sequence/battery.py --task lookback \\
        --task-args '{"lengths": [5, 12], "test_lengths": [24, 60]}' \\
        --arms table recipe --founders 0 1 2 --out results/lookback.jsonl --workers 3
    python benchmarks/sequence/battery.py --summary results/lookback.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

BATCH = 16
LESSONS, TESTS, MODULE = 1500, 100, 64
FOUNDERS = (0, 1, 2)
SEEDS = {"train": 11, "test": 33}

TABLE = {"recall": True, "recall_founder": True, "recall_recency": 0.0, "learning_surprise": True}
BOUNDARY = {**TABLE, "recall_boundary": True}
RECIPE = {**BOUNDARY, "recall_anchor": True, "recall_slope": 0.1, "recall_position_rate": 0.01}
HEADS = {**TABLE, "recall_heads": 2}
COPY = {"recall_copy": 2.0, "recall_tokens": list(range(16))}  # two hops' 16 tokens
WHEN = {"recall": True, "recall_founder": True, "recall_recency": 0.0}  # every-row lessons
ARMS: dict[str, dict[str, Any]] = {
    "table": TABLE,
    "boundary": BOUNDARY,
    "slow": {**BOUNDARY, "recall_position_rate": 0.01},
    "slope": {**BOUNDARY, "recall_slope": 0.1},
    "slope_slow": {**BOUNDARY, "recall_slope": 0.1, "recall_position_rate": 0.01},
    "anchor": {**BOUNDARY, "recall_anchor": True},
    "anchor_relative": {**BOUNDARY, "recall_anchor": True, "recall_relative": True},
    "recipe": RECIPE,
    "recipe64": {**RECIPE, "recall_position": 64},
    "recipe_relative": {**RECIPE, "recall_relative": True},
    "heads": HEADS,
    "heads_copy": {**HEADS, **COPY},
    "every_row": WHEN,
    "when": {**WHEN, "recall_when": True},
    "when_anchor": {**WHEN, "recall_when": True, "recall_anchor": True},
    # the when read that stays on the item heard: a sharp softmax, the position term not taught
    "when_sharp": {**WHEN, "recall_when": True, "recall_position_rate": 0.0,
                   "recall_temperature": 0.1},
    "rehearsal": {**BOUNDARY, "recall_anchor": True, "learning_rehearsal": 1.0},
    # what a new brain composes: the memory at its composition and the composition's teaching
    "composed": {"recall": True},
}
COMPOSED = ("composed",)  # the arms that take the composition's own teaching point


@dataclass
class Moment:
    x: np.ndarray  # (batch, inputs); replaced by the previous answer where ``feedback``
    label: np.ndarray | None = None  # (batch,) taught and scored when given
    age: np.ndarray | None = None  # (batch,) age of the slot whose event is the answer
    condition: list[str] | None = None  # per-row condition for the scores
    feedback: bool = False  # the observation is the previous answer (closed loop in tests)


@dataclass
class Task:
    """A family of episodes; ``episode(rng, train)`` returns the moments of one batch."""

    name: str
    inputs: int
    actions: int
    settings: dict[str, Any] = field(default_factory=dict)

    def episode(self, rng: np.random.Generator, train: bool) -> list[Moment]:
        raise NotImplementedError

    def chance(self) -> float:
        return 1.0 / self.actions


def onehot(index: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((len(index), width))
    out[np.arange(len(index)), index] = 1.0
    return out


# -------------------------------------------------------------------------------- tasks


class Induction(Task):
    """``A B ... A -> B``: the token that followed the query's earlier occurrence."""

    def __init__(self, vocab: int = 8, lengths=(8, 16), test_lengths=None, gap_min: int = 2):
        super().__init__("induction", vocab, vocab, dict(vocab=vocab, lengths=list(lengths),
                                                         test_lengths=test_lengths,
                                                         gap_min=gap_min))
        self.vocab, self.lengths = vocab, tuple(lengths)
        self.test_lengths = tuple(test_lengths) if test_lengths else self.lengths
        self.gap_min = gap_min

    def episode(self, rng, train):
        lo, hi = self.lengths if train else self.test_lengths
        length = int(rng.integers(lo, hi + 1))
        rows = []
        ages = np.zeros(BATCH, dtype=int)
        for r in range(BATCH):
            a = int(rng.integers(self.vocab))
            others = [t for t in range(self.vocab) if t != a]
            seq = list(rng.choice(others, size=length - 1))
            i = int(rng.integers(0, length - 1 - self.gap_min))  # A at i, B at i + 1
            seq[i] = a
            seq[i + 1] = int(rng.choice(others))  # B may be anything but A; nothing else is A
            seq.append(a)  # the query: the last moment
            rows.append(seq)
            ages[r] = (length - 1) - 1 - i
        rows = np.array(rows)
        moments = [Moment(onehot(rows[:, t], self.inputs)) for t in range(length - 1)]
        label = np.array([rows[r][np.where(rows[r][:-1] == rows[r][-1])[0][0] + 1]
                          for r in range(BATCH)])
        moments.append(Moment(onehot(rows[:, -1], self.inputs), label, ages - 1,
                              [f"L{length}"] * BATCH))
        return moments


class NBack(Task):
    """A stream of tokens, then a cue ``k``: the token ``k`` moments back."""

    def __init__(self, vocab: int = 8, ks=(1, 2, 3, 4), lengths=(6, 12)):
        super().__init__("nback", vocab + len(ks), vocab,
                         dict(vocab=vocab, ks=list(ks), lengths=list(lengths)))
        self.vocab, self.ks, self.lengths = vocab, tuple(ks), tuple(lengths)

    def episode(self, rng, train):
        length = int(rng.integers(self.lengths[0], self.lengths[1] + 1))
        seq = rng.integers(self.vocab, size=(BATCH, length))
        moments = [Moment(onehot(seq[:, t], self.inputs)) for t in range(length)]
        k_index = rng.integers(len(self.ks), size=BATCH)
        ks = np.array(self.ks)[k_index]
        label = seq[np.arange(BATCH), length - ks]
        moments.append(Moment(onehot(self.vocab + k_index, self.inputs), label, ks - 1,
                              [f"k{k}" for k in ks]))
        return moments


class LookBack(Task):
    """A stream of symbols, then a cue, ``first``, ``last`` or ``second``: the symbol at the
    cued place. ``ask`` (the cues asked, every cue keeping its input) is set by phases."""

    CUES = ("first", "last", "second")

    def __init__(self, vocab: int = 4, lengths=(5, 12), test_lengths=None, cues=CUES):
        super().__init__("lookback", vocab + len(cues), vocab,
                         dict(vocab=vocab, lengths=list(lengths), test_lengths=test_lengths,
                              cues=list(cues)))
        self.vocab, self.lengths = vocab, tuple(lengths)
        self.test_lengths = tuple(test_lengths) if test_lengths else self.lengths
        self.cues = tuple(cues)
        self.ask = self.cues

    def episode(self, rng, train):
        lo, hi = self.lengths if train else self.test_lengths
        length = int(rng.integers(lo, hi + 1))
        seq = rng.integers(self.vocab, size=(BATCH, length))
        moments = [Moment(onehot(seq[:, t], self.inputs)) for t in range(length)]
        if self.ask == self.cues:
            c = rng.integers(len(self.cues), size=BATCH)
        else:
            allowed = np.array([self.cues.index(cue) for cue in self.ask])
            c = allowed[rng.integers(len(allowed), size=BATCH)]
        where = {"first": 0, "last": length - 1, "second": 1}
        pos = np.array([where[self.cues[i]] for i in c])
        label = seq[np.arange(BATCH), pos]
        moments.append(Moment(onehot(self.vocab + c, self.inputs), label, length - 1 - pos,
                              [f"{self.cues[i]}" for i in c]))
        return moments


class Repeat(Task):
    """Copying: ``START``, the tokens, ``GO``; then each token in turn, the brain hearing the
    token just answered (its own answer in closed loop)."""

    def __init__(self, vocab: int = 16, lengths=(3, 6), test_lengths=None, unique: bool = True):
        super().__init__("repeat", vocab + 2, vocab,
                         dict(vocab=vocab, lengths=list(lengths), test_lengths=test_lengths,
                              unique=unique))
        self.vocab, self.lengths, self.unique = vocab, tuple(lengths), unique
        self.test_lengths = tuple(test_lengths) if test_lengths else self.lengths
        self.START, self.GO = vocab, vocab + 1

    def episode(self, rng, train):
        lo, hi = self.lengths if train else self.test_lengths
        length = int(rng.integers(lo, hi + 1))
        if self.unique:
            seq = np.stack([rng.permutation(self.vocab)[:length] for _ in range(BATCH)])
        else:
            seq = rng.integers(self.vocab, size=(BATCH, length))
        moments = [Moment(onehot(np.full(BATCH, self.START), self.inputs))]
        moments += [Moment(onehot(seq[:, t], self.inputs)) for t in range(length)]
        # GO answers the first token, heard length - 1 moments before it
        moments.append(Moment(onehot(np.full(BATCH, self.GO), self.inputs), seq[:, 0],
                              np.full(BATCH, length - 1), [f"L{length}:p0"] * BATCH))
        # hearing token j (the answer just given) answers token j + 1, whose slot is always
        # ``length - 1`` moments back (the rest of the sequence, GO and the answers since)
        for j in range(length - 1):
            moments.append(Moment(onehot(seq[:, j], self.inputs), seq[:, j + 1],
                                  np.full(BATCH, length - 1), [f"L{length}:p{j + 1}"] * BATCH,
                                  feedback=True))
        return moments


class Periodic(Task):
    """A motif of period ``p`` repeated with a little variation: from the second period on,
    every moment answers the motif's next symbol."""

    def __init__(self, vocab: int = 6, periods=(8,), repeats: int = 4, variation: float = 0.1):
        super().__init__("periodic", vocab, vocab,
                         dict(vocab=vocab, periods=list(periods), repeats=repeats,
                              variation=variation))
        self.vocab, self.periods = vocab, tuple(periods)
        self.repeats, self.variation = repeats, variation

    def episode(self, rng, train):
        p = int(rng.choice(self.periods))
        motif = rng.integers(self.vocab, size=(BATCH, p))
        length = p * self.repeats
        seq = np.tile(motif, (1, self.repeats))
        vary = rng.random((BATCH, length)) < self.variation
        vary[:, :p] = False  # the first period is the motif itself
        seq = np.where(vary, rng.integers(self.vocab, size=(BATCH, length)), seq)
        target = np.tile(motif, (1, self.repeats + 1))
        moments = []
        for t in range(length):
            if t < p:  # the first period is only heard
                moments.append(Moment(onehot(seq[:, t], self.inputs)))
                continue
            moments.append(Moment(onehot(seq[:, t], self.inputs), target[:, t + 1],
                                  np.full(BATCH, p - 2), [f"P{p}"] * BATCH))
        return moments


class Associative(Task):
    """Key, value pairs, then a key: its value."""

    def __init__(self, keys: int = 32, values: int = 8, pairs=(4,), distractors: int = 0):
        super().__init__("associative", keys + values + 1, values,
                         dict(keys=keys, values=values, pairs=list(pairs),
                              distractors=distractors))
        self.keys, self.values = keys, values
        self.pairs, self.distractors = tuple(pairs), distractors

    def episode(self, rng, train):
        n = int(rng.choice(self.pairs))
        ks = np.stack([rng.permutation(self.keys)[:n] for _ in range(BATCH)])
        vs = rng.integers(self.values, size=(BATCH, n))
        moments, t_of_key = [], np.zeros((BATCH, n), dtype=int)
        t = 0
        for i in range(n):
            moments.append(Moment(onehot(ks[:, i], self.inputs)))
            t_of_key[:, i] = t
            t += 1
            moments.append(Moment(onehot(self.keys + vs[:, i], self.inputs)))
            t += 1
            for _ in range(self.distractors):
                blank = np.full(BATCH, self.keys + self.values)
                moments.append(Moment(onehot(blank, self.inputs)))
                t += 1
        q = rng.integers(n, size=BATCH)
        label = vs[np.arange(BATCH), q]
        ages = (t - 1) - t_of_key[np.arange(BATCH), q]
        moments.append(Moment(onehot(ks[np.arange(BATCH), q], self.inputs), label, ages - 1,
                              [f"N{n}"] * BATCH))
        return moments


class TwoHop(Task):
    """Pairs (first, second) as consecutive moments, among them a chain ``a -> b`` and
    ``b -> c``. The query hears ``a``; with ``think`` a THINK moment follows and answers ``c``
    while the query answers ``b`` (taught too); without it the query moment answers ``c``."""

    def __init__(self, vocab: int = 16, pairs: int = 4, think: bool = True):
        super().__init__("twohop", vocab + 1, vocab, dict(vocab=vocab, pairs=pairs, think=think))
        self.vocab, self.pairs, self.think, self.THINK = vocab, pairs, think, vocab

    def episode(self, rng, train):
        n = self.pairs
        firsts = np.zeros((BATCH, n), dtype=int)
        seconds = np.zeros((BATCH, n), dtype=int)
        a = np.zeros(BATCH, dtype=int)
        b = np.zeros(BATCH, dtype=int)
        c = np.zeros(BATCH, dtype=int)
        for r in range(BATCH):
            tokens = rng.permutation(self.vocab)
            a[r], b[r], c[r] = tokens[:3]
            fs = [a[r], b[r]] + list(tokens[3:3 + n - 2])  # distinct firsts
            ss = [b[r], c[r]] + list(rng.integers(self.vocab, size=n - 2))
            order = rng.permutation(n)
            firsts[r], seconds[r] = np.array(fs)[order], np.array(ss)[order]
        moments = []
        for i in range(n):
            moments.append(Moment(onehot(firsts[:, i], self.inputs)))
            moments.append(Moment(onehot(seconds[:, i], self.inputs)))
        if self.think:
            moments.append(Moment(onehot(a, self.inputs), b, None, ["hop1"] * BATCH))
            moments.append(Moment(onehot(np.full(BATCH, self.THINK), self.inputs), c, None,
                                  ["hop2"] * BATCH))
        else:
            moments.append(Moment(onehot(a, self.inputs), c, None, ["hop2_direct"] * BATCH))
        return moments


STORY = 24  # the stories' vocabulary in ``ago`` and ``before``
BUCKETS = (2, 4, 8)  # ``ago``'s answers: 0-1, 2-3, 4-7, 8 or more moments before the question


class Ago(Task):
    """A story of distinct tokens, then X with a question flag: how long ago was X told?"""

    def __init__(self, lengths=(8, 16)):
        super().__init__("ago", STORY + 2, len(BUCKETS) + 1, dict(vocab=STORY,
                                                                  lengths=list(lengths)))
        self.lengths = tuple(lengths)

    def episode(self, rng, train):
        length = int(rng.integers(self.lengths[0], self.lengths[1] + 1))
        story = np.stack([rng.permutation(STORY)[:length] for _ in range(BATCH)])
        moments = [Moment(onehot(story[:, t], self.inputs)) for t in range(length)]
        place = rng.integers(length, size=BATCH)
        x = onehot(story[np.arange(BATCH), place], self.inputs)
        x[:, STORY] = 1.0
        ages = length - 1 - place
        label = np.searchsorted(BUCKETS, ages, side="right")
        moments.append(Moment(x, label, ages, [f"b{b}" for b in label]))
        return moments


class Before(Task):
    """A story of distinct tokens, then A with one flag and B with another: was A told
    before B? The answer falls at B; the cortex carries what it made of A."""

    def __init__(self, lengths=(8, 16)):
        super().__init__("before", STORY + 2, 2, dict(vocab=STORY, lengths=list(lengths)))
        self.lengths = tuple(lengths)

    def episode(self, rng, train):
        length = int(rng.integers(self.lengths[0], self.lengths[1] + 1))
        story = np.stack([rng.permutation(STORY)[:length] for _ in range(BATCH)])
        moments = [Moment(onehot(story[:, t], self.inputs)) for t in range(length)]
        pair = np.stack([rng.permutation(length)[:2] for _ in range(BATCH)])
        rows = np.arange(BATCH)
        first = onehot(story[rows, pair[:, 0]], self.inputs)
        first[:, STORY] = 1.0
        second = onehot(story[rows, pair[:, 1]], self.inputs)
        second[:, STORY + 1] = 1.0
        label = (pair[:, 0] < pair[:, 1]).astype(np.int64)  # 1: A was told before B
        moments.append(Moment(first))
        moments.append(Moment(second, label, None, ["yes" if v else "no" for v in label]))
        return moments


TASKS: dict[str, type[Task]] = {
    "induction": Induction, "nback": NBack, "lookback": LookBack, "repeat": Repeat,
    "periodic": Periodic, "associative": Associative, "twohop": TwoHop, "ago": Ago,
    "before": Before,
}


# --------------------------------------------------------------------------------- life


def make_brain(task: Task, arm: str, founder: int, settings: dict[str, Any] | None = None,
               module: int = MODULE) -> Any:
    """``Brain.compose`` for ``task`` with the arm's settings, and any ``settings`` over them."""
    import cadence as cd

    point = {} if arm in COMPOSED else {"teaching": "normalized"}
    return cd.Brain.compose(task.inputs, task.actions, modules=(module,), seed=int(founder),
                            episodic=False,
                            **{"backend": "cpu", "recall": False, **point,
                               **{**ARMS[arm], **(settings or {})}})


def new_stats() -> dict[str, Any]:
    return {"conditions": {}, "mass": [], "refused": 0, "exhausted": 0, "episode_hits": []}


def run_episode(brain: Any, task: Task, moments: list[Moment], teach: bool, closed: bool,
                stats: dict[str, Any]) -> None:
    """One episode on a fresh stream: a lesson before the act at every answer moment when
    ``teach``, else a score of the act; in closed loop a fed-back moment hears the answer."""
    brain.reset()
    previous = None
    for m in moments:
        x = m.x
        if m.feedback and closed and previous is not None:
            x = onehot(previous, task.inputs)
        if teach and m.label is not None:
            try:
                brain.lesson(x, m.label)
            except RuntimeError:
                stats["refused"] += 1
        try:
            action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
        except RuntimeError:
            stats["exhausted"] += 1
            action = np.full(len(x), -1)
        if not teach and m.label is not None:
            hit = action == m.label
            for r, cond in enumerate(m.condition or ["all"] * len(x)):
                c = stats["conditions"].setdefault(cond, [0, 0])
                c[0] += int(hit[r])
                c[1] += 1
            heads = brain.recall.attended_heads if brain.recall is not None else []
            if m.age is not None and heads:
                masses = []
                for p in heads:
                    if p is None:
                        continue
                    slots = p.shape[1]
                    idx = slots - 1 - m.age
                    ok = (idx >= 0) & (idx < slots)
                    at = p[np.arange(len(x)), np.clip(idx, 0, slots - 1)]
                    masses.append(np.where(ok, at, 0.0))
                if masses:
                    stats["mass"].extend(np.max(np.stack(masses), axis=0).tolist())
            stats["episode_hits"].append(hit)
        previous = action if m.label is not None else previous


def evaluate(brain: Any, task: Task, rng: np.random.Generator, episodes: int,
             closed: bool) -> dict[str, Any]:
    """A frozen test: accuracy, per condition, the share of episodes answered without a
    miss and the read's attention on the answer's slot (the most any head gave it)."""
    stats = new_stats()
    exact = []
    for _ in range(episodes):
        start = len(stats["episode_hits"])
        run_episode(brain, task, task.episode(rng, train=False), teach=False, closed=closed,
                    stats=stats)
        hits = stats["episode_hits"][start:]
        if hits:
            exact.extend(np.all(np.stack(hits), axis=0).tolist())
    total = sum(v[0] for v in stats["conditions"].values())
    count = sum(v[1] for v in stats["conditions"].values())
    return {
        "accuracy": total / max(count, 1),
        "conditions": {k: v[0] / v[1] for k, v in sorted(stats["conditions"].items())},
        "exact_episode": float(np.mean(exact)) if exact else None,
        "mass": float(np.mean(stats["mass"])) if stats["mass"] else None,
        "exhausted": stats["exhausted"],
    }


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``spec`` names ``task``, ``arm`` and ``founder``, with ``task_args``,
    ``settings`` over the arm's, ``lessons``, ``tests``, ``module`` (the width), ``phases``
    (look back's cue groups, taught one after another) and ``erase`` (read again with the
    anchored maps at zero)."""
    warnings.simplefilter("ignore")
    import cadence as cd

    task = TASKS[spec["task"]](**spec.get("task_args", {}))
    arm, founder = spec["arm"], int(spec["founder"])
    lessons, tests = int(spec.get("lessons", LESSONS)), int(spec.get("tests", TESTS))
    phases = [tuple(p) for p in spec.get("phases") or []]
    if phases and not isinstance(task, LookBack):
        raise ValueError("phases are look back's cue groups")
    started = time.monotonic()
    brain = make_brain(task, arm, founder, spec.get("settings"), int(spec.get("module", MODULE)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, SEEDS["train"]]))
    stats = new_stats()
    for ask in phases or [None]:
        if ask is not None:
            task.ask = ask
        for _ in range(lessons // max(len(phases), 1)):
            run_episode(brain, task, task.episode(rng, train=True), teach=True, closed=False,
                        stats=stats)
    if phases:
        task.ask = task.cues
    closed = task.name == "repeat"

    def test() -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([founder, SEEDS["test"]]))

    row: dict[str, Any] = {
        "task": spec["task"], "task_settings": task.settings, "arm": arm, "founder": founder,
        "settings": {**ARMS[arm], **spec.get("settings", {})}, "lessons": lessons,
        "tests": tests, "module": int(spec.get("module", MODULE)),
        "phases": [list(p) for p in phases], "batch": BATCH,
        "chance": task.chance(), "cadence": cd.__version__,
        "refused_lessons": stats["refused"], "exhausted_train": stats["exhausted"],
        "test": evaluate(brain, task, test(), tests, closed),
    }
    if closed:
        row["teacher_forced"] = evaluate(brain, task, test(), tests, False)
    if spec.get("erase"):
        recall = brain.recall
        recall.anchor_maps = [tuple(np.zeros_like(a) for a in maps) for maps in recall.anchor_maps]
        row["erased"] = evaluate(brain, task, test(), tests, closed)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


# ------------------------------------------------------------------------------ summary


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per task, its settings, arm and phases: the founders' accuracy, per condition, its
    least and greatest (closed loop where the task feeds answers back)."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = json.dumps([r["task"], r["task_settings"], r["arm"], r["phases"]], sort_keys=True)
        cells.setdefault(key, []).append(r)
    out = []
    for key, group in sorted(cells.items()):
        task, settings, arm, phases = json.loads(key)
        cell: dict[str, Any] = {"task": task, "task_settings": settings, "arm": arm,
                                "phases": phases, "founders": sorted(r["founder"] for r in group)}
        for reading in ("test", "teacher_forced", "erased"):
            readings = [r[reading] for r in group if reading in r]
            if not readings:
                continue
            names = sorted({c for x in readings for c in x["conditions"]})
            span = {"accuracy": _span([x["accuracy"] for x in readings])}
            if len(names) <= 4:
                span.update({c: _span([x["conditions"].get(c) for x in readings])
                             for c in names})
            cell[reading] = span
        cell["refused_lessons"] = sum(r["refused_lessons"] for r in group)
        out.append(cell)
    return out


def _span(values: list[float | None]) -> list[float]:
    kept = [v for v in values if v is not None]
    return [round(min(kept), 3), round(max(kept), 3)] if kept else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--task-args", type=json.loads, default={})
    parser.add_argument("--arms", nargs="*", default=["table"], choices=list(ARMS))
    parser.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--lessons", type=int, default=LESSONS)
    parser.add_argument("--tests", type=int, default=TESTS)
    parser.add_argument("--settings", type=json.loads, default={}, help="over the arms'")
    parser.add_argument("--phases", nargs="*", default=[],
                        help="look back's cue groups taught in turn, e.g. first last second")
    parser.add_argument("--erase", action="store_true", help="read again without the anchor")
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs in place of the grid")
    parser.add_argument("--out", type=Path, help="JSON lines, one row per life, appended")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary", type=Path, help="print the summary of a results file")
    args = parser.parse_args(argv)
    if args.summary is not None:
        rows = [json.loads(line) for line in args.summary.read_text().splitlines() if line]
        for cell in summarize(rows):
            print(json.dumps(cell))
        return 0
    if args.out is None:
        parser.error("--out is required")
    if args.plan is not None:
        specs = json.loads(args.plan.read_text())
    else:
        if args.task is None:
            parser.error("--task or --plan is required")
        specs = [
            {"task": args.task, "task_args": args.task_args, "arm": arm, "founder": founder,
             "lessons": args.lessons, "tests": args.tests, "settings": args.settings,
             "phases": [p.split(",") for p in args.phases], "erase": args.erase}
            for arm in args.arms for founder in args.founders
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for row in map(run_life, specs):
                sink.write(json.dumps(row) + "\n")
                sink.flush()
        else:
            with get_context("spawn").Pool(min(args.workers, len(specs))) as pool:
                for row in pool.imap_unordered(run_life, specs, chunksize=1):
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
