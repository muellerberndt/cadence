"""The encyclopedia chamber (cadence#311): a brain reads a corpus of facts once, one word a
moment, writing its own records, and answers questions about them long after, each answer said
whole as one action at the silent moment after the question.

**Arms.** Every arm is ``Brain.compose(inputs, actions, modules=(module,), backend="cpu")`` at
the library's defaults, with the heard history of the moment before:

- ``records``: a held message (``message=280``) and ``longterm`` records written at every turn
  (``longterm_turn``), read by what tells them apart (``longterm_distinct``) with the record
  that holds the most of the question (``longterm_best``) and a second head cued by what the
  question asks beyond the first record (``longterm_rest``);
- ``distinct``: the records of cadence#305's second declaration (``longterm_distinct=0.5``, the
  latest record above the match, the second head chained from what the first record is about);
- ``previous_word``: the default alone, with no message and no records.

``longterm`` in a spec names further records settings.

**The life.** One stream. Every told fact and every question is its own episode
(``Brain.reset()`` before it empties the message and the windows; the records continue).

1. The nursery: ``sessions`` lesson sessions (``facts.lesson_session``), each answer taught at
   its silent moment (``lesson``) and then said.
2. The reading, on the nursery's brain with its records emptied: the first ``size`` facts of the
   reading corpus told once each, in the corpus's order; then the questions, in the order
   ``trained``, ``held``, ``synonym``, ``two`` (questions of two facts whose facts the corpus
   holds); then ``revised`` facts told again with another object of their relation and asked;
   then ``erased``: trained questions with the records emptied at each answer.

Every answer is said greedily and scored exactly; ``found`` reads whether the record a head
holds says the answer (head 1 for a question of two facts, head 0 for any other). Work is
counted as seconds and multiply-accumulates: every synapse once a sweep and row of a moment's
settle, and the records' read, one accumulate per record each of the cue's entries reaches.

    PYTHONPATH=src python benchmarks/encyclopedia/chamber.py --plan plan.json --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import facts as fx  # noqa: E402

ARMS = ("records", "distinct", "previous_word")
UNITS = 280


def make_brain(lexicon: fx.Lexicon, founder: int, arm: str, capacity: int, module: int = 64,
               longterm: dict[str, Any] | None = None) -> Any:
    warnings.simplefilter("ignore")
    import cadence as cd

    settings: dict[str, Any] = {}
    if arm != "previous_word":
        settings.update(message=UNITS, message_fan_in=max(16, round(16 / 70 * 2 * lexicon.inputs)),
                        longterm=int(capacity), longterm_turn=fx.WIDTH, longterm_heads=2)
    if arm == "records":
        settings.update(longterm_distinct=0.1, longterm_match=0.5, longterm_best=True,
                        longterm_rest=True)
    if arm == "distinct":
        settings.update(longterm_distinct=0.5, longterm_match=0.5)
    settings.update({f"longterm_{k}": v for k, v in (longterm or {}).items()})
    return cd.Brain.compose(lexicon.inputs, lexicon.actions, modules=(module,), seed=int(founder),
                            backend="cpu", **settings)


def read_work(brain: Any) -> float:
    """The records' read before a moment's settle: one accumulate per record each entry of the
    cue reaches, for each head that reads."""
    kept = getattr(brain, "longterm", None)
    if kept is None or not len(kept.since) or kept.since[0] != 0 or not kept.stores:
        return 0.0
    postings = kept._postings[0] if kept._postings else {}
    reached = sum(len(postings.get(int(e), ())) for e in kept._cues[0])
    return float(reached * kept.heads)


class Life:
    """One stream's moments through ``brain``, counting work."""

    def __init__(self, brain: Any, lexicon: fx.Lexicon) -> None:
        self.brain, self.lexicon = brain, lexicon
        self.size = int(brain.describe()["layout"]["synapses"])
        self.work = {"moments": 0, "macs": 0.0, "read_macs": 0.0, "seconds": 0.0}

    def act(self, x: np.ndarray) -> int:
        brain = self.brain
        t0 = time.perf_counter()
        before = read_work(brain)
        said = int(np.asarray(brain.act(x[None], greedy=True)).reshape(-1)[0])
        report = brain.last_settlement
        sweeps = report.get("row_sweeps", int(report.get("steps", 0)))
        self.work["moments"] += 1
        self.work["macs"] += before + float(sweeps) * self.size
        self.work["read_macs"] += before
        self.work["seconds"] += time.perf_counter() - t0
        return said

    def turn(self, turn: fx.Turn, teach: bool = False, erase: bool = False) -> int | None:
        """One turn as its own episode: the said answer where it asks."""
        self.brain.reset()
        said = None
        for x, label in self.lexicon.moments(turn):
            if label is None:
                self.act(x)
                continue
            if erase and getattr(self.brain, "longterm", None) is not None:
                self.brain.longterm.clear()
            if teach:
                self.brain.lesson(x[None], np.array([label]))
            said = self.act(x)
        return said

    def found(self, turn: fx.Turn) -> float | None:
        kept = getattr(self.brain, "longterm", None)
        if kept is None or not kept.stores:
            return None
        head = 1 if turn.tags.get("wording") == "two" else 0
        slot = int(kept.read_slots[0, head])
        want = self.lexicon.moment(str(turn.answer))
        held = slot >= 0 and bool(np.array_equal(
            np.asarray(kept.stores[0].values[slot], dtype=float), want))
        return float(held)


def summarise(scores: dict[str, list[float]]) -> dict[str, Any]:
    return {k: {"right": round(float(np.mean(v)), 4), "questions": len(v)}
            for k, v in sorted(scores.items())}


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    founder, arm, size = int(spec["founder"]), spec["arm"], int(spec["size"])
    corpus = fx.Corpus(Path(spec["data"]))
    lexicon = corpus.lexicon
    reading = corpus.corpus(size)
    sessions = int(spec.get("sessions", 1000))
    capacity = size + int(spec.get("slack", 10000))
    brain = make_brain(lexicon, founder, arm, capacity, int(spec.get("module", 64)),
                       spec.get("longterm"))
    life = Life(brain, lexicon)
    rng = np.random.default_rng(np.random.SeedSequence([founder, 311, 1]))
    nursery: list[float] = []
    t0 = time.perf_counter()
    for s in range(sessions):
        for turn in fx.lesson_session(corpus, rng, int(spec.get("facts", 6))):
            said = life.turn(turn, teach=True)
            if turn.asked and s >= sessions - max(sessions // 10, 1):
                nursery.append(float(said == lexicon.action[str(turn.answer)]))
    nursery_seconds = time.perf_counter() - t0
    taught = {f.answer for f in corpus.lessons}
    base = {"founder": founder, "arm": spec.get("label", arm), "size": size,
            "parameters": int(brain.parameters()), "neurons": int(brain.connectome.n),
            "synapses": life.size, "actions": lexicon.actions,
            "nursery": round(float(np.mean(nursery)), 4) if nursery else None,
            "nursery_seconds": round(nursery_seconds), "nursery_work": dict(life.work)}
    # the reading: the corpus told once, on the nursery's brain with its records emptied
    if getattr(brain, "longterm", None) is not None:
        brain.longterm.clear()
    life.work = {"moments": 0, "macs": 0.0, "read_macs": 0.0, "seconds": 0.0}
    t0 = time.perf_counter()
    for fact in reading:
        life.turn(fx.tell(fact))
    intake = dict(life.work)
    intake_seconds = time.perf_counter() - t0
    read_rng = np.random.default_rng(np.random.SeedSequence([founder, 311, 2]))
    scores: dict[str, list[float]] = defaultdict(list)
    work = {"moments": 0, "macs": 0.0, "read_macs": 0.0, "seconds": 0.0, "answers": 0}

    def answer(turn: fx.Turn, groups: list[str], erase: bool = False,
               stale: str | None = None) -> None:
        life.work = {"moments": 0, "macs": 0.0, "read_macs": 0.0, "seconds": 0.0}
        said = life.turn(turn, erase=erase)
        right = float(said == lexicon.action[str(turn.answer)])
        for g in groups:
            scores[g].append(right)
            if str(turn.answer) not in taught:
                scores[f"{g} untaught"].append(right)
        if stale is not None:
            scores["stale"].append(float(said == lexicon.action.get(stale, -1)))
        found = life.found(turn)
        if found is not None:
            scores[f"found {groups[0]}"].append(found)
        for k in ("moments", "macs", "read_macs", "seconds"):
            work[k] += life.work[k]
        work["answers"] += 1

    n = int(spec.get("questions", 1000))
    picks = read_rng.choice(len(reading), size=min(2 * n, len(reading)), replace=False)
    for i, k in enumerate(picks[:n]):
        fact = reading[int(k)]
        answer(fx.ask(fact, "trained", i), ["trained", f"trained {fact.relation}"])
    for i, k in enumerate(picks[n: 2 * n]):
        fact = reading[int(k)]
        answer(fx.ask(fact, "held", i), ["held", f"held {fact.relation}"])
    m = int(spec.get("small", 300))
    for i, k in enumerate(read_rng.choice(len(reading), size=min(m, len(reading)),
                                          replace=False)):
        answer(fx.ask(reading[int(k)], "synonym", i), ["synonym"])
    pairs = fx.chains(reading)
    for k in read_rng.permutation(len(pairs))[:m]:
        first, second = pairs[int(k)]
        answer(fx.ask_two(first, second), ["two", f"two {first.relation}>{second.relation}"])
    # revisions: facts told again with another object, then asked
    chosen = [reading[int(k)] for k in read_rng.choice(len(reading), size=min(m, len(reading)),
                                                       replace=False)]
    now = {f: fx.revised(f, corpus, read_rng) for f in chosen}
    for f in chosen:
        life.turn(fx.tell(f, now[f]))
    for i, f in enumerate(chosen):
        answer(fx.ask(f, "trained", i, now[f]), ["revised"], stale=f.answer)
    for i, k in enumerate(read_rng.choice(len(reading), size=min(m, len(reading)),
                                          replace=False)):
        answer(fx.ask(reading[int(k)], "trained", i), ["erased"], erase=True)
    answers = max(work["answers"], 1)
    kept = getattr(brain, "longterm", None)
    return [{**base, "scores": summarise(scores),
             "intake_seconds_per_fact": round(intake_seconds / max(len(reading), 1), 6),
             "intake_macs_per_fact": int(intake["macs"] / max(len(reading), 1)),
             "answer_ms": round(1e3 * work["seconds"] / answers, 4),
             "answer_macs": int(work["macs"] / answers),
             "answer_read_macs": int(work["read_macs"] / answers),
             "records": None if kept is None else int(sum(s.count for s in kept.stores)),
             "record_bytes": None if kept is None else int(sum(
                 s.keys.nbytes + s.values.nbytes for s in kept.stores)),
             "entries": None if kept is None or kept.distinct is None else kept.entries,
             "answers_taught": round(float(np.mean([f.answer in taught for f in reading])), 4)}]


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    common = {k: v for k, v in plan.items() if k not in ("founders", "arms", "sizes")}
    return [{**common, "size": size, "founder": f, **arm}
            for size in plan["sizes"] for f in plan["founders"] for arm in plan["arms"]]


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
