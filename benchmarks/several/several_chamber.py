"""The several-records chamber (cadence#325): a brain reads facts once, one word a moment,
writing its own long-term records, and answers questions whose one answer needs several of
them, said whole by its motor slots at the silent moment after the question.

**The brain.** ``Brain.compose(inputs, actions, slots=SLOTS, modules=(module,), message=280,
longterm=records, longterm_turn=channel, longterm_heads=heads, ...)`` at the library's
defaults; an arm's ``longterm`` names its records settings. ``oracle`` in an arm hands every
head its record at each answer (the record of part j on head j, a chain from its end on head 0):
the ceiling of the answer's path with the read given, and no arm of the brain's own.

**The life.** One stream; every told fact and every question is its own episode
(``Brain.reset()`` before it; the records continue).

1. The nursery: the lesson population told once, then ``lessons`` questions about it, each
   taught at its silent moment (``lesson``) and then said. The questions mix parts of 1 to
   ``SLOTS`` subjects (k parts drawn in proportion to k) and chains of 2 and 3 hops, with
   absent parts and broken chains.
2. The reading, on the nursery's brain with its records emptied: the first ``size`` facts of
   the reading population told once; then the questions of each kind, without lessons.

**Readings,** per kind and size of a question:

- ``said``: every slot says its answer; ``part``: the share of the asked parts said;
- ``delivered``: the heads hold the records the answer needs, each once, and nothing else;
  ``ordered``: head j holds the record of part j (a chain: the heads hold it from its end);
- ``silent``: of the parts no record answers, the share whose head holds nothing;
  ``refused``: of the questions no record answers, the share the brain answers ``NOTHING``;
- seconds and multiply-accumulates of an answer, and the milliseconds of the records' read at
  the answer's moment.

    PYTHONPATH=src python benchmarks/several/several_chamber.py --plan plan.json --out rows.jsonl
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

import several_world as sw  # noqa: E402

UNITS = 280
RECORDS = {"distinct": 0.1, "match": 0.5, "best": True, "fade": False}  # the encyclopedia's read


def make_brain(lexicon: sw.Lexicon, founder: int, capacity: int, module: int = 256,
               heads: int = sw.SLOTS, longterm: dict[str, Any] | None = None,
               records: bool = True, settings: dict[str, Any] | None = None) -> Any:
    warnings.simplefilter("ignore")
    import cadence as cd

    made: dict[str, Any] = {}
    if records:
        made.update(message=UNITS, message_fan_in=max(16, round(16 / 70 * 2 * lexicon.inputs)),
                    longterm=int(capacity), longterm_turn=sw.WIDTH, longterm_heads=int(heads))
        made.update({f"longterm_{k}": v for k, v in {**RECORDS, **(longterm or {})}.items()
                     if v is not None})
    made.update(settings or {})
    return cd.Brain.compose(lexicon.inputs, lexicon.actions, slots=sw.SLOTS, modules=(module,),
                            seed=int(founder), backend="cpu", **made)


def reads(kept: Any) -> bool:
    """Whether the next settle's read addresses the records: the moment after a turn."""
    return bool(kept is not None and len(kept.since) and kept.since[0] == 0 and kept.stores)


def read_work(kept: Any) -> float:
    """The records' read before a moment's settle, called before it: one accumulate per record
    each entry of the cue reaches, for each head that reads."""
    if not reads(kept):
        return 0.0
    if kept.distinct is None:  # every record scored on the cue's units, once a head
        return float(kept.key[0].sum() * kept.stores[0].count * kept.heads)
    postings = kept._postings[0] if kept._postings else {}
    return float(sum(len(postings.get(int(e), ())) for e in kept._cues[0]) * kept.heads)


class Life:
    """One stream's moments through ``brain``, counting work; with ``oracle`` every head is
    handed its record at each answer."""

    def __init__(self, brain: Any, lexicon: sw.Lexicon, oracle: bool = False) -> None:
        self.brain, self.lexicon = brain, lexicon
        self.kept = getattr(brain, "longterm", None)
        self.size = int(brain.describe()["layout"]["synapses"])
        self.slot_of: list[int] = []  # each told fact's record
        self.handed: list[int] | None = None
        self.read_seconds = 0.0
        self.reset_work()
        if self.kept is not None:
            read = self.kept.stimulus

            def stimulus(drive: Any, moment: Any = None) -> np.ndarray:
                kept = self.kept
                reading = reads(kept)  # the moment after a turn: the heads read
                t0 = time.perf_counter()
                out = np.asarray(read(drive, moment))
                if reading:
                    self.read_seconds += time.perf_counter() - t0
                if oracle and self.handed is not None and len(kept.since) and kept.since[0] == 0:
                    kept.read_slots[:] = -1
                    for head, columns in enumerate(kept.columns):
                        out[0, columns] = 0.0
                        slot = self.handed[head] if head < len(self.handed) else -1
                        if slot >= 0:
                            out[0, columns] = kept.amplitude * kept.stores[0].values[slot]
                            kept.read_slots[0, head] = slot
                return out

            self.kept.stimulus = stimulus

    def reset_work(self) -> None:
        self.work = {"moments": 0, "macs": 0.0, "read_macs": 0.0, "seconds": 0.0}
        self.read_seconds = 0.0

    def act(self, x: np.ndarray) -> np.ndarray:
        brain = self.brain
        t0 = time.perf_counter()
        reading, before = reads(self.kept), read_work(self.kept)
        said = np.asarray(brain.act(x[None], greedy=True)).reshape(-1)
        counted = getattr(self.kept, "work", None)
        if reading and counted is not None:  # what the read itself counted
            before = float(counted)
        report = brain.last_settlement
        sweeps = report.get("row_sweeps", int(report.get("steps", 0)))
        self.work["moments"] += 1
        self.work["macs"] += before + float(sweeps) * self.size
        self.work["read_macs"] += before
        self.work["seconds"] += time.perf_counter() - t0
        return said

    def tell(self, fact: sw.Fact) -> None:
        """One told fact as its own episode; its record's slot is kept."""
        self.brain.reset()
        store = None if self.kept is None or not self.kept.stores else self.kept.stores[0]
        before = 0 if store is None else store.count
        for x in self.lexicon.told(fact):
            self.act(x)
        if self.kept is None:
            self.slot_of.append(-1)
            return
        store = self.kept.stores[0]
        self.slot_of.append(store.count - 1 if store.count > before else -1)

    def needed(self, question: sw.Question) -> list[int]:
        """The record each head is to hold: part j on head j; a chain from its end."""
        slots = [self.slot_of[i] if i >= 0 else -1 for i in question.needs]
        if question.kind == "chain":
            slots = slots[::-1] if all(s >= 0 for s in slots) else []
        return slots + [-1] * (sw.SLOTS - len(slots))

    def ask(self, question: sw.Question, teach: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """One question as its own episode: what the slots said and their answers."""
        self.brain.reset()
        moments, labels = self.lexicon.asked(question)
        self.handed = self.needed(question)
        for x in moments[:-1]:
            self.act(x)
        if teach:
            self.brain.lesson(moments[-1][None], labels[None])
        said = self.act(moments[-1])
        self.handed = None
        return said, labels


def score(life: Life, question: sw.Question, said: np.ndarray, labels: np.ndarray,
          scores: dict[str, list[float]]) -> None:
    """One answered question's readings, under its group."""
    group = f"{question.kind} {question.k}" + (" absent" if question.tags.get("absent") else "")
    asked = question.k if question.kind == "parts" else 1
    right = said == labels
    scores[f"said {group}"].append(float(right.all()))
    scores[f"part {group}"].append(float(right[:asked].mean()))
    nothing = life.lexicon.action[sw.NOTHING]
    if not (labels != nothing).any():
        scores[f"refused {group}"].append(float((said == nothing).all()))
    kept = life.kept
    if kept is None or not kept.stores:
        return
    want = life.needed(question)
    held = [int(s) for s in kept.read_slots[0]]
    heads = len(held)
    wanted = sorted(s for s in want if s >= 0)
    scores[f"delivered {group}"].append(float(sorted(s for s in held if s >= 0) == wanted))
    scores[f"ordered {group}"].append(float(held == want[:heads] and len(wanted) <= heads))
    if question.kind == "parts":
        for j in range(min(question.k, heads)):
            if question.needs[j] < 0:
                scores[f"silent {group}"].append(float(held[j] < 0))


def summarise(scores: dict[str, list[float]]) -> dict[str, Any]:
    return {k: {"right": round(float(np.mean(v)), 4), "questions": len(v)}
            for k, v in sorted(scores.items())}


def lesson_question(people: sw.Population, rng: np.random.Generator, spec: dict[str, Any],
                    ) -> sw.Question:
    """One lesson's question: a chain of 2 or 3 hops, or parts of 1 to ``SLOTS`` subjects, a
    question of k parts drawn in proportion to k so that every slot is asked often."""
    absent = float(spec.get("absent", 0.15))
    if rng.random() < float(spec.get("chains", 0.3)):
        return people.chain(rng, 2 + int(rng.integers(2)), broken=absent)
    sizes = np.arange(1, int(spec.get("parts", sw.SLOTS)) + 1)
    return people.parts(rng, int(rng.choice(sizes, p=sizes / sizes.sum())), absent)


def reading_population(founder: int, size: int) -> sw.Population:
    """The reading population of ``size`` facts: subjects of its own words, of whom about
    ``size`` facts are drawn, the first ``size`` told."""
    return sw.Population([int(founder), 325, 3, int(size)], int(size / 3.4) + 8, "e").keep(size)


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    founder, arm = int(spec["founder"]), spec["arm"]
    lexicon = sw.Lexicon()
    sizes = [int(n) for n in spec["sizes"]]
    taught = sw.Population([founder, 325, 1], int(spec.get("lesson_subjects", 600)), "t")
    capacity = max(max(sizes), len(taught.facts)) + 64
    brain = make_brain(lexicon, founder, capacity, int(spec.get("module", 256)),
                       int(spec.get("heads", sw.SLOTS)), spec.get("longterm"),
                       bool(spec.get("records", True)), spec.get("settings"))
    life = Life(brain, lexicon, oracle=bool(spec.get("oracle")))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 325, 2]))
    t0 = time.perf_counter()
    for fact in taught.facts:
        life.tell(fact)
    lessons = int(spec.get("lessons", 4000))
    nursery: dict[str, list[float]] = defaultdict(list)
    curve: list[float] = []
    block: list[float] = []
    for i in range(lessons):
        question = lesson_question(taught, rng, spec)
        said, labels = life.ask(question, teach=True)
        block.append(float((said == labels).all()))
        if len(block) == max(lessons // 10, 1):
            curve.append(round(float(np.mean(block)), 4))
            block = []
        if i >= lessons - max(lessons // 10, 1):
            score(life, question, said, labels, nursery)
    base = {"founder": founder, "arm": spec.get("label", arm),
            "parameters": int(brain.parameters()), "neurons": int(brain.connectome.n),
            "synapses": life.size, "heads": int(spec.get("heads", sw.SLOTS)),
            "lessons": lessons, "nursery": summarise(nursery), "nursery_curve": curve,
            "nursery_seconds": round(time.perf_counter() - t0),
            "nursery_macs": float(life.work["macs"])}
    out = []
    kept = life.kept
    for size in sizes:
        if kept is not None:
            kept.clear()
        life.slot_of = []
        world = reading_population(founder, size)
        if len(world.facts) < size:
            raise RuntimeError("the reading population holds too few facts")
        life.reset_work()
        t0 = time.perf_counter()
        for fact in world.facts:
            life.tell(fact)
        intake, intake_seconds = dict(life.work), time.perf_counter() - t0
        read_rng = np.random.default_rng(np.random.SeedSequence([founder, 325, 4, size]))
        scores: dict[str, list[float]] = defaultdict(list)
        costs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        n = int(spec.get("questions", 200))

        def answer(question: sw.Question, scores: Any = scores, costs: Any = costs) -> None:
            life.reset_work()
            said, labels = life.ask(question)
            score(life, question, said, labels, scores)
            cost = costs[f"{question.kind} {question.k}"]
            cost["answers"] += 1
            cost["seconds"] += life.work["seconds"]
            cost["macs"] += life.work["macs"]
            cost["read_macs"] += life.work["read_macs"]
            cost["read_seconds"] += life.read_seconds

        for k in spec.get("parts_read", [1, 2, 4, 8]):
            for _ in range(n):
                answer(world.parts(read_rng, int(k)))
            for _ in range(n // 2):
                answer(world.parts(read_rng, int(k), absent=0.3))
        for hops in spec.get("chains_read", [2, 3]):
            for _ in range(n):
                answer(world.chain(read_rng, int(hops)))
            for _ in range(n // 2):
                answer(world.chain(read_rng, int(hops), broken=1.0))
        out.append({
            **base, "size": size, "scores": summarise(scores),
            "records": None if kept is None else int(sum(s.count for s in kept.stores)),
            "entries": None if kept is None or kept.distinct is None else kept.entries,
            "intake_seconds_per_fact": round(intake_seconds / size, 6),
            "intake_macs_per_fact": int(intake["macs"] / size),
            "costs": {g: {"answer_ms": round(1e3 * c["seconds"] / c["answers"], 4),
                          "answer_macs": int(c["macs"] / c["answers"]),
                          "read_macs": int(c["read_macs"] / c["answers"]),
                          "read_ms": round(1e3 * c["read_seconds"] / c["answers"], 4)}
                      for g, c in sorted(costs.items())},
        })
    return out


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    common = {k: v for k, v in plan.items() if k not in ("founders", "arms")}
    return [{**common, "founder": f, **arm} for f in plan["founders"] for arm in plan["arms"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    todo = cells(json.loads(args.plan.read_text()))
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
