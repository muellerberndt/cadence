"""Matched controls for the encyclopedia chamber (cadence#311): four models hear the facts and
questions the brain hears, one word a moment through a linear input layer on the instrument's
moments (``facts.Lexicon``: 257 channels, the turn at channel 256), and say each answer at the
question's silent moment as one of the closed answers, trained by cross-entropy at the labelled
moments. Each model's width is the one whose parameter count is nearest the brain's
(``target``: the parameters of ``chamber.make_brain`` for the plan's ``target_arm`` and
``module``). They are scored controls, never part of the brain's answer path.

- ``closed_lstm``, ``closed_transformer`` (closed-book): trained on every told fact of the
  reading corpus and of the lesson sessions, each its own sequence ("the {r} of {s} is", the turn
  and a silent moment labelled with the told object), and on the lesson questions, each its own
  sequence, in shuffled epochs until the training loss stops falling (patience 3), ``epochs``
  pass or the time budget ends. A question is asked as its own sequence with nothing before it.
  After the first four groups are asked, the revisions are taught by the same training on the
  corpus with each revised fact's telling replaced by its revision (``revision_budget``).
- ``context_transformer`` (in-context): a causal transformer reading a window of ``window``
  moments. It is trained on the lesson sessions' turns as one stream, whole sessions packed
  into windows, at the answer moments. In the reading the corpus's told facts are one stream
  (the revisions told after it) and each question follows the latest whole facts that fit in
  the window with it; ``in window`` reads whether every fact the question needs is inside it.
- ``retrieval``: BM25 (``rank_bm25.BM25Okapi``) indexes the told sentences as words; the
  question's words fetch the ``k`` best sentences, heard in their told order as told turns
  before the question by a causal transformer reader. The reader is trained on the lesson
  questions, fetched from an index of the lesson facts' sentences with each lesson revision
  appended once told; ``found`` reads whether the answer's sentence is among those fetched.

The questions are the chamber's draws, in its order and score groups. ``erased`` asks the
trained questions with the model's store emptied: the context window or the fetched sentences.
The closed-book models keep no store apart from their weights, so their ``erased`` are the
trained questions asked after the revisions are taught.

Adam (rate ``rate``, gradient norm clipped at 1) on padded batches; each answer is said alone
(``answer_batch`` 1). Work is counted as seconds and analytic multiply-accumulates (a backward
pass at twice the forward; BM25's read one accumulate per posting a query reaches). Memory is
the parameters at 4 bytes, with the index and the told sentences' text for ``retrieval`` and the
window's keys and values for ``context_transformer``.

    PYTHONPATH=src python benchmarks/encyclopedia/controls.py --plan plan.json --out rows.jsonl \\
        --data facts.tsv --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import chamber as ch  # noqa: E402
import facts as fx  # noqa: E402

MODELS = ("closed_lstm", "closed_transformer", "context_transformer", "retrieval")
WINDOW, K = 1024, 3
CLIP, PATIENCE, RATE = 1.0, 3, 0.001
BUDGET = 45.0  # minutes of training in a cell
EPOCHS = {"closed": 1000, "context": 100, "retrieval": 100}
BATCH = {"closed": 128, "context": 8, "retrieval": 64}
MAX_LENGTH = 4096
SILENT, TURN = 0, 1  # the word ids of the silent moment and the turn
IGNORE = -100


# ------------------------------------------------------------------------------ the moments


class Moments:
    """Turns as word ids, and batches of them as the moments ``facts.Lexicon`` makes: a word's
    ``ON`` form channels, the turn's channel ``WIDTH``, the silent moment empty."""

    def __init__(self, lexicon: fx.Lexicon) -> None:
        self.lexicon, self.inputs = lexicon, lexicon.inputs
        self.ids: dict[str, int] = {}
        self.codes = np.full((4096, fx.ON), self.inputs, dtype=np.int64)  # ``inputs``: unused
        self.codes[TURN, 0] = fx.WIDTH
        self.count = 2

    def id(self, word: str) -> int:
        i = self.ids.get(word)
        if i is None:
            i = self.ids[word] = self.count
            if i >= len(self.codes):
                grown = np.full((2 * len(self.codes), fx.ON), self.inputs, dtype=np.int64)
                grown[: len(self.codes)] = self.codes
                self.codes = grown
            self.codes[i] = fx.form(word)
            self.count += 1
        return i

    def turn(self, turn: fx.Turn, told: bool = False) -> tuple[list[int], list[int]]:
        """A turn's word ids and the action taught at each (``IGNORE`` where none), in the order
        of ``Lexicon.moments``. ``told`` asks a told turn for its object: its words before the
        object, the turn, and a silent moment labelled with the object."""
        if told and not turn.asked:
            at = int(turn.turn_at)  # type: ignore[arg-type]
            turn = fx.Turn(turn.said[:at], turn.said[at], None, turn.tags)
        ids: list[int] = []
        labels: list[int] = []
        for i, w in enumerate(turn.said):
            if turn.turn_at == i and not turn.asked:
                ids.append(TURN)
                labels.append(IGNORE)
            ids.append(self.id(w))
            labels.append(IGNORE)
        if turn.asked:
            ids += [TURN, SILENT]
            labels += [IGNORE, self.lexicon.action[str(turn.answer)]]
        return ids, labels

    def batch(self, rows: list[tuple[list[int], list[int]]]) -> tuple[np.ndarray, np.ndarray]:
        """``(rows, moments, inputs)`` moments and ``(rows, moments)`` labels, padded at the end
        with silent moments."""
        length = max(len(r[0]) for r in rows)
        ids = np.zeros((len(rows), length), dtype=np.int64)
        y = np.full((len(rows), length), IGNORE, dtype=np.int64)
        for i, (a, b) in enumerate(rows):
            ids[i, : len(a)], y[i, : len(b)] = a, b
        x = np.zeros((len(rows), length, self.inputs + 1), dtype=np.float32)
        np.put_along_axis(x, self.codes[ids], 1.0, axis=2)
        return np.ascontiguousarray(x[..., : self.inputs]), y


# ------------------------------------------------------------------------------- the models


def build(kind: str, width: int, inputs: int, actions: int, seed: int,
          window: int | None = None, length: int = MAX_LENGTH) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class LSTM(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTM(inputs, width, batch_first=True)
            self.out = nn.Linear(width, actions)

        def forward(self, x: Any) -> Any:
            h, _ = self.cell(x)
            return self.out(h)

        def step_macs(self, seen: int) -> int:
            return 4 * width * (inputs + width) + width * actions

        def macs(self, length: int) -> int:
            return length * self.step_macs(length)

    class Transformer(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 2

        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(inputs, width)
            layer = nn.TransformerEncoderLayer(width, self.HEADS, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(width, actions)
            position = torch.arange(length)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(length, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            i = torch.arange(n)
            blocked = i[None, :] > i[:, None]
            if window is not None:
                blocked = blocked | (i[:, None] - i[None, :] >= window)
            mask = torch.zeros(n, n).masked_fill(blocked, float("-inf"))
            h = self.inp(x) + self.position[:n]
            return self.out(self.body(h, mask=mask))

        def step_macs(self, seen: int) -> int:
            """One moment with the keys and values of ``seen`` moments held."""
            return (inputs * width + self.LAYERS * 8 * width * width + width * actions
                    + self.LAYERS * 2 * width * seen)

        def macs(self, length: int) -> int:
            seen = min(length, window or length)
            per = inputs * width + self.LAYERS * 8 * width * width + width * actions
            return length * per + self.LAYERS * 2 * width * length * seen

    return LSTM() if kind == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(kind: str, target: int, inputs: int, actions: int) -> int:
    """The even width whose parameter count is nearest ``target``."""

    def size(width: int) -> int:
        return count(build(kind, width, inputs, actions, 0, length=1))

    lo, hi = 1, 512
    while lo < hi:
        mid = (lo + hi) // 2
        if size(2 * mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return min({max(2 * lo - 2, 2), 2 * lo}, key=lambda w: (abs(size(w) - target), w))


class Learner:
    def __init__(self, kind: str, width: int, lexicon: fx.Lexicon, seed: int, rate: float,
                 window: int | None = None) -> None:
        import torch

        self.torch = torch
        self.net = build(kind, width, lexicon.inputs, lexicon.actions, seed, window)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "macs": 0, "seconds": 0.0}

    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        torch = self.torch
        t0 = time.perf_counter()
        self.net.train()
        logits = self.net(torch.from_numpy(x))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                 torch.from_numpy(y).reshape(-1),
                                                 ignore_index=IGNORE)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
        self.opt.step()
        self.work["updates"] += 1
        self.work["macs"] += 3 * self.net.macs(x.shape[1]) * x.shape[0]
        self.work["seconds"] += time.perf_counter() - t0
        return float(loss.item())

    def said(self, x: np.ndarray, at: list[int]) -> np.ndarray:
        """The action said at moment ``at[i]`` of each row ``i``."""
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(x))
            return np.asarray(logits[torch.arange(len(at)), torch.tensor(at)].argmax(-1))


Row = tuple[list[int], list[int]]


def fit(learner: Learner, moments: Moments, items: list[Row], batch: int,
        rng: np.random.Generator, epochs: int, budget: float) -> dict[str, Any]:
    """Shuffled epochs of padded batches until the training loss stops falling (patience 3),
    ``epochs`` pass or ``budget`` seconds of training end."""
    losses: list[float] = []
    best, waited, stopped = math.inf, 0, "epochs"
    t0 = time.perf_counter()
    while len(losses) < epochs and items:
        order = rng.permutation(len(items))
        total, n = 0.0, 0
        for k in range(0, len(order), batch):
            x, y = moments.batch([items[int(i)] for i in order[k: k + batch]])
            total += learner.update(x, y)
            n += 1
            if time.perf_counter() - t0 > budget:
                stopped = "budget"
                break
        losses.append(total / max(n, 1))
        best, waited = (losses[-1], 0) if losses[-1] < best - 1e-4 else (best, waited + 1)
        if stopped == "budget":
            break
        if waited >= PATIENCE:
            stopped = "patience"
            break
    curve = {str(e): round(losses[e - 1], 4) for e in range(1, len(losses) + 1)
             if e & (e - 1) == 0 or e == len(losses)}  # at epochs 1, 2, 4, 8, ... and the last
    return {"epochs": len(losses), "loss": round(losses[-1], 4) if losses else None,
            "curve": curve, "stopped": stopped, "items": len(items),
            "seconds": round(time.perf_counter() - t0, 2)}


def rounded(value: float) -> float:
    return float(f"{value:.4g}")


# ---------------------------------------------------------------------------- the questions


@dataclass
class Question:
    turn: fx.Turn
    groups: list[str]
    needs: list[int]  # positions in the told stream of the facts it needs, the answer's last
    stale: str | None = None
    erase: bool = False


@dataclass
class Draws:
    told: list[tuple[fx.Fact, str]]  # the told stream: the corpus, then the revisions
    first: list[Question]  # trained, held, synonym, two
    revised: list[Question]
    erased: list[Question]
    size: int = 0
    of: list[int] = field(default_factory=list)  # the corpus position each revision tells again


def reading_draws(corpus: fx.Corpus, reading: list[fx.Fact], founder: int,
                  spec: dict[str, Any]) -> Draws:
    """The chamber's reading questions from its generator, drawn in its order."""
    rng = np.random.default_rng(np.random.SeedSequence([founder, 311, 2]))
    place = {id(f): i for i, f in enumerate(reading)}
    n = int(spec.get("questions", 1000))
    first: list[Question] = []
    picks = rng.choice(len(reading), size=min(2 * n, len(reading)), replace=False)
    for i, k in enumerate(picks[:n]):
        f = reading[int(k)]
        first.append(Question(fx.ask(f, "trained", i), ["trained", f"trained {f.relation}"],
                              [int(k)]))
    for i, k in enumerate(picks[n: 2 * n]):
        f = reading[int(k)]
        first.append(Question(fx.ask(f, "held", i), ["held", f"held {f.relation}"], [int(k)]))
    m = int(spec.get("small", 300))
    for i, k in enumerate(rng.choice(len(reading), size=min(m, len(reading)), replace=False)):
        first.append(Question(fx.ask(reading[int(k)], "synonym", i), ["synonym"], [int(k)]))
    pairs = fx.chains(reading)
    for k in rng.permutation(len(pairs))[:m]:
        a, b = pairs[int(k)]
        first.append(Question(fx.ask_two(a, b), ["two", f"two {a.relation}>{b.relation}"],
                              [place[id(a)], place[id(b)]]))
    of = [int(k) for k in rng.choice(len(reading), size=min(m, len(reading)), replace=False)]
    chosen = [reading[k] for k in of]
    answer = {f: fx.revised(f, corpus, rng) for f in chosen}
    told = [(f, f.answer) for f in reading] + [(f, answer[f]) for f in chosen]
    latest = {f: len(reading) + j for j, f in enumerate(chosen)}
    revised = [Question(fx.ask(f, "trained", i, answer[f]), ["revised"], [latest[f]],
                        stale=f.answer) for i, f in enumerate(chosen)]
    erased = [Question(fx.ask(reading[int(k)], "trained", i), ["erased"], [int(k)], erase=True)
              for i, k in enumerate(rng.choice(len(reading), size=min(m, len(reading)),
                                               replace=False))]
    return Draws(told, first, revised, erased, len(reading), of)


def revision_tells(session: list[fx.Turn]) -> list[fx.Turn]:
    """A lesson session's revisions: a told turn right after the telling of the same subject and
    relation with another object."""
    out, before = [], None
    for t in session:
        if t.asked:
            continue
        at = int(t.turn_at)  # type: ignore[arg-type]
        if (before is not None and before.said[: before.turn_at] == t.said[:at]
                and before.said[int(before.turn_at)] != t.said[at]):  # type: ignore[arg-type]
            out.append(t)
        before = t
    return out


# ------------------------------------------------------------------------------- the index


class Index:
    """``rank_bm25.BM25Okapi`` over sentences of words, its scores read from postings (the
    same sums ``get_scores`` makes, over the sentences holding each query word)."""

    def __init__(self, sentences: list[list[str]]) -> None:
        from rank_bm25 import BM25Okapi

        t0 = time.perf_counter()
        bm25 = BM25Okapi(sentences)
        self.idf, self.k1 = bm25.idf, bm25.k1
        length = np.asarray(bm25.doc_len, dtype=np.float64)
        self.norm = bm25.k1 * (1 - bm25.b + bm25.b * length / bm25.avgdl)
        lists: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for d, counts in enumerate(bm25.doc_freqs):
            for word, c in counts.items():
                held = lists[word]
                held[0].append(d)
                held[1].append(c)
        self.postings = {w: (np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.float64))
                         for w, (a, b) in lists.items()}
        self.size = len(sentences)
        self.entries = int(sum(len(a) for a, _ in self.postings.values()))
        self.nbytes = int(self.norm.nbytes + sum(a.nbytes + b.nbytes
                                                 for a, b in self.postings.values()))
        self.text_bytes = int(sum(len(" ".join(s).encode()) for s in sentences))
        self.seconds = time.perf_counter() - t0

    def scores(self, query: list[str]) -> tuple[np.ndarray, int]:
        """BM25 scores of every sentence and the postings the query reaches."""
        score = np.zeros(self.size)
        reached = 0
        for word in query:
            held = self.postings.get(word)
            if held is None:
                continue
            ids, c = held
            score[ids] += (self.idf.get(word) or 0) * (c * (self.k1 + 1) / (c + self.norm[ids]))
            reached += len(ids)
        return score, reached

    def top(self, query: list[str], k: int, limit: int | None = None) -> tuple[list[int], int]:
        """The ``k`` best sentences among the first ``limit`` (best first, ties to the earlier)
        and the postings reached."""
        score, reached = self.scores(query)
        if limit is not None:
            score = score[:limit]
        k = min(k, len(score))
        if k <= 0:
            return [], reached
        part = np.argpartition(-score, k - 1)[:k]
        return [int(d) for d in sorted(part, key=lambda d: (-score[d], d))], reached


# ---------------------------------------------------------------------------------- a cell


@dataclass
class Cell:
    spec: dict[str, Any]
    corpus: fx.Corpus
    reading: list[fx.Fact]
    moments: Moments
    sessions: list[list[fx.Turn]]
    draws: Draws
    width: int
    taught: set[str]
    scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    work: dict[str, float] = field(default_factory=lambda: {
        "answers": 0, "seconds": 0.0, "macs": 0.0, "read_macs": 0.0})

    @property
    def lexicon(self) -> fx.Lexicon:
        return self.corpus.lexicon

    def setting(self, name: str, family: str, table: dict[str, Any]) -> Any:
        return self.spec.get(name, table[family])

    def record(self, q: Question, said: int, **extra: float) -> None:
        """The chamber's score groups for one answer, with ``extra`` per first group and, where
        an ``extra`` holds, the answer's score in the group ``"{group} with {extra}"``."""
        answer = str(q.turn.answer)
        right = float(said == self.lexicon.action[answer])
        for g in q.groups:
            self.scores[g].append(right)
            if answer not in self.taught:
                self.scores[f"{g} untaught"].append(right)
        if q.stale is not None:
            self.scores["stale"].append(float(said == self.lexicon.action.get(q.stale, -1)))
        for name, value in extra.items():
            self.scores[f"{name} {q.groups[0]}"].append(float(value))
            if value:
                self.scores[f"{q.groups[0]} with {name}"].append(right)

    def ask(self, learner: Learner, rows: list[Row]) -> list[int]:
        """The answers said at each row's last moment, ``answer_batch`` rows at a time."""
        batch = int(self.spec.get("answer_batch", 1))
        said: list[int] = []
        t0 = time.perf_counter()
        for k in range(0, len(rows), batch):
            part = rows[k: k + batch]
            x, _ = self.moments.batch(part)
            said += [int(s) for s in learner.said(x, [len(r[0]) - 1 for r in part])]
            self.work["macs"] += sum(learner.net.macs(len(r[0])) for r in part)
        self.work["seconds"] += time.perf_counter() - t0
        self.work["answers"] += len(rows)
        return said

    def told_turn(self, p: int) -> fx.Turn:
        fact, answer = self.draws.told[p]
        return fx.tell(fact, answer)


def closed(cell: Cell, kind: str) -> dict[str, Any]:
    spec, mo = cell.spec, cell.moments
    seed = int(spec["founder"])
    items = [mo.turn(fx.tell(f), told=True) for f in cell.reading]
    items += [mo.turn(t, told=True) for session in cell.sessions for t in session]
    learner = Learner(kind, cell.width, cell.lexicon, seed, float(spec.get("rate", RATE)))
    rng = np.random.default_rng(seed)
    batch = int(cell.setting("batch", "closed", BATCH))
    epochs = int(cell.setting("epochs", "closed", EPOCHS))
    trained = fit(learner, mo, items, batch, rng, epochs, 60 * float(spec.get("budget", BUDGET)))
    intake = dict(learner.work)
    questions = cell.draws.first
    for q, said in zip(questions, cell.ask(learner, [mo.turn(q.turn) for q in questions]),
                       strict=True):
        cell.record(q, said)
    # the revisions: the same training on the corpus with each revised fact's telling replaced
    for j, p in enumerate(cell.draws.of):
        items[p] = mo.turn(cell.told_turn(cell.draws.size + j), told=True)
    revision = fit(learner, mo, items, batch, rng, epochs,
                   60 * float(spec.get("revision_budget", 10.0)))
    questions = cell.draws.revised + cell.draws.erased
    for q, said in zip(questions, cell.ask(learner, [mo.turn(q.turn) for q in questions]),
                       strict=True):
        cell.record(q, said)
    facts = max(len(cell.reading), 1)
    return {"training": trained, "revision": revision, "train": dict(learner.work),
            "parameters": count(learner.net), "store_bytes": 0,
            "intake_seconds_per_fact": rounded(intake["seconds"] / facts),
            "intake_macs_per_fact": int(intake["macs"] / facts)}


def pack(blocks: list[Row], window: int) -> list[Row]:
    """Whole blocks packed in order into rows of at most ``window`` moments."""
    rows: list[Row] = []
    ids: list[int] = []
    labels: list[int] = []
    for a, b in blocks:
        if ids and len(ids) + len(a) > window:
            rows.append((ids, labels))
            ids, labels = [], []
        ids, labels = ids + a[-window:], labels + b[-window:]
    if ids:
        rows.append((ids, labels))
    return rows


def context(cell: Cell) -> dict[str, Any]:
    spec, mo = cell.spec, cell.moments
    seed, window = int(spec["founder"]), int(spec.get("window", WINDOW))
    assert window <= MAX_LENGTH
    blocks = []
    for session in cell.sessions:
        ids: list[int] = []
        labels: list[int] = []
        for t in session:
            a, b = mo.turn(t)
            ids += a
            labels += b
        blocks.append((ids, labels))
    learner = Learner("transformer", cell.width, cell.lexicon, seed,
                      float(spec.get("rate", RATE)), window)
    rng = np.random.default_rng(seed)
    trained = fit(learner, mo, pack(blocks, window), int(cell.setting("batch", "context", BATCH)),
                  rng, int(cell.setting("epochs", "context", EPOCHS)),
                  60 * float(spec.get("budget", BUDGET)))
    sentences: dict[int, list[int]] = {}

    def sentence(p: int) -> list[int]:
        if p not in sentences:
            sentences[p] = mo.turn(cell.told_turn(p))[0]
        return sentences[p]

    def tail(end: int, room: int) -> tuple[list[int], int]:
        """The latest whole told sentences before position ``end`` that fit in ``room``
        moments, and the first position they hold."""
        parts: list[list[int]] = []
        used, p = 0, end - 1
        while p >= 0 and used + len(sentence(p)) <= room:
            parts.append(sentence(p))
            used += len(sentence(p))
            p -= 1
        return [w for part in reversed(parts) for w in part], p + 1

    size, told = cell.draws.size, len(cell.draws.told)
    questions = cell.draws.first + cell.draws.revised + cell.draws.erased
    rows: list[Row] = []
    inside: list[float] = []
    for q in questions:
        a, b = mo.turn(q.turn)
        if q.erase:
            rows.append((a, b))
            inside.append(0.0)
            continue
        end = told if q.groups[0] == "revised" else size
        held, start = tail(end, window - len(a))
        rows.append((held + a, [IGNORE] * len(held) + b))
        inside.append(float(all(start <= p < end for p in q.needs)))
    cached = 0
    for q, said, held in zip(questions, cell.ask(learner, rows), inside, strict=True):
        cell.record(q, said, **{"in window": held})
        cached += len(mo.turn(q.turn)[0]) * learner.net.step_macs(window)
    # a fact's intake: the forward over a window of the corpus, by the facts it holds
    stream, start = tail(size, window)
    x, _ = mo.batch([(stream, [IGNORE] * len(stream))])
    t0 = time.perf_counter()
    for _ in range(3):
        learner.said(x, [len(stream) - 1])
    held_facts = max(size - start, 1)
    per_fact = len(stream) / held_facts
    width = cell.width
    return {"training": trained, "train": dict(learner.work), "parameters": count(learner.net),
            "window": window, "store_bytes": 2 * learner.net.LAYERS * window * width * 4,
            "window_facts": size - start,
            "intake_seconds_per_fact": rounded((time.perf_counter() - t0) / 3 / held_facts),
            "intake_macs_per_fact": int(per_fact * learner.net.step_macs(window)),
            "answer_macs_cached": int(cached / max(len(questions), 1))}


def retrieval(cell: Cell) -> dict[str, Any]:
    spec, mo = cell.spec, cell.moments
    seed, k = int(spec["founder"]), int(spec.get("k", K))
    pool = cell.corpus.lessons
    news = [t for session in cell.sessions for t in revision_tells(session)]
    lesson_index = Index([fx.tell(f).said for f in pool] + [t.said for t in news])

    def lesson_turn(d: int) -> fx.Turn:
        return fx.tell(pool[d]) if d < len(pool) else news[d - len(pool)]

    items: list[Row] = []
    limit = len(pool)
    for session in cell.sessions:
        limit += len(revision_tells(session))
        for t in session:
            if not t.asked:
                continue
            hits, _ = lesson_index.top(t.said, k, limit)
            ids: list[int] = []
            labels: list[int] = []
            for d in sorted(hits):
                a, b = mo.turn(lesson_turn(d))
                ids += a
                labels += b
            a, b = mo.turn(t)
            items.append((ids + a, labels + b))
    learner = Learner("transformer", cell.width, cell.lexicon, seed,
                      float(spec.get("rate", RATE)))
    rng = np.random.default_rng(seed)
    trained = fit(learner, mo, items, int(cell.setting("batch", "retrieval", BATCH)), rng,
                  int(cell.setting("epochs", "retrieval", EPOCHS)),
                  60 * float(spec.get("budget", BUDGET)))
    del lesson_index
    size = cell.draws.size
    index = Index([cell.told_turn(p).said for p in range(size)])
    revised_index = Index([cell.told_turn(p).said for p in range(len(cell.draws.told))])
    for group, use in ((cell.draws.first, index), (cell.draws.revised, revised_index),
                       (cell.draws.erased, None)):
        t0 = time.perf_counter()
        rows: list[Row] = []
        found: list[float] = []
        for q in group:
            ids, labels = [], []
            hits: list[int] = []
            if use is not None:
                hits, reached = use.top(q.turn.said, k)
                cell.work["read_macs"] += reached
                for d in sorted(hits):
                    a, b = mo.turn(cell.told_turn(d))
                    ids += a
                    labels += b
            found.append(float(q.needs[-1] in hits))
            a, b = mo.turn(q.turn)
            rows.append((ids + a, labels + b))
        cell.work["seconds"] += time.perf_counter() - t0
        for q, said, hit in zip(group, cell.ask(learner, rows), found, strict=True):
            cell.record(q, said, found=hit)
    facts = max(size, 1)
    return {"training": trained, "train": dict(learner.work), "parameters": count(learner.net),
            "k": k, "lesson_index_sentences": len(pool) + len(news),
            "index_bytes": index.nbytes, "corpus_bytes": index.text_bytes,
            "store_bytes": index.nbytes + index.text_bytes,
            "intake_seconds_per_fact": rounded(index.seconds / facts),
            "intake_macs_per_fact": round(index.entries / facts, 3)}


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    torch.set_num_threads(int(spec.get("threads", 8)))
    t0 = time.perf_counter()
    founder, model, size = int(spec["founder"]), spec["model"], int(spec["size"])
    assert model in MODELS, model
    corpus = fx.Corpus(Path(spec["data"]), keep=size)
    lexicon = corpus.lexicon
    reading = corpus.corpus(size)
    target = spec.get("target")
    if target is None:  # the parameters of the brain the plan names
        brain = ch.make_brain(lexicon, founder, spec.get("target_arm", "records"),
                              size + int(spec.get("slack", 10000)), int(spec.get("module", 64)))
        target = int(brain.parameters())
        del brain
    kind = "lstm" if model == "closed_lstm" else "transformer"
    width = matched_width(kind, int(target), lexicon.inputs, lexicon.actions)
    rng = np.random.default_rng(np.random.SeedSequence([founder, 311, 1]))
    sessions = [fx.lesson_session(corpus, rng, int(spec.get("facts", 6)),
                                  float(spec.get("two", 0.2)))
                for _ in range(int(spec.get("sessions", 1000)))]
    cell = Cell(spec, corpus, reading, Moments(lexicon), sessions,
                reading_draws(corpus, reading, founder, spec), width,
                {f.answer for f in corpus.lessons})
    if model == "closed_lstm":
        out = closed(cell, "lstm")
    elif model == "closed_transformer":
        out = closed(cell, "transformer")
    elif model == "context_transformer":
        out = context(cell)
    else:
        out = retrieval(cell)
    answers = max(int(cell.work["answers"]), 1)
    parameter_bytes = 4 * int(out["parameters"])
    return [{"founder": founder, "model": model, "arm": spec.get("label", model), "size": size,
             "width": width, "target": int(target),
             "target_arm": spec.get("target_arm", "records"), "module": int(spec.get("module", 64)),
             "actions": lexicon.actions, "sessions": len(sessions),
             "epochs": out["training"]["epochs"], "loss": out["training"]["loss"],
             **out, "scores": ch.summarise(cell.scores),
             "answer_ms": round(1e3 * cell.work["seconds"] / answers, 4),
             "answer_macs": int((cell.work["macs"] + cell.work["read_macs"]) / answers),
             "answer_read_macs": int(cell.work["read_macs"] / answers),
             "parameter_bytes": parameter_bytes,
             "memory_bytes": parameter_bytes + int(out["store_bytes"]),
             "answers_taught": round(float(np.mean([f.answer in cell.taught for f in reading])),
                                     4),
             "threads": int(spec.get("threads", 8)),
             "seconds": round(time.perf_counter() - t0)}]


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """One cell per size, founder and model; a model is a name or a dict of its settings."""
    common = {k: v for k, v in plan.items() if k not in ("founders", "models", "sizes")}
    models = [m if isinstance(m, dict) else {"model": m} for m in plan["models"]]
    return [{**common, "size": size, "founder": f, **m}
            for size in plan["sizes"] for f in plan["founders"] for m in models]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data", type=Path, required=True, help="the corpus table")
    args = parser.parse_args(argv)
    todo = [{**c, "data": str(args.data)} for c in cells(json.loads(args.plan.read_text()))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink, get_context("spawn").Pool(args.workers,
                                                               maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(run_cell, todo, chunksize=1):
            for row in rows:
                sink.write(json.dumps(row) + "\n")
            sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
