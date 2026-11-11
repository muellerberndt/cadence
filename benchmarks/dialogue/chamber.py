"""The conversation chamber (cadence#305): a continuing brain hears conversations one word a
moment and says each answer whole, as one action, at the silent moment after a question.

**Arms.** Every arm is ``Brain.compose(inputs, actions, modules=(64,), backend="cpu")`` at the
library's defaults (with the heard history of the moment before, ``history_decays=(0.0,)``):

- ``previous_word``: the default alone;
- ``message``: with a held message (``message=280``, the chamber's share of the channels as its
  fan-in), held over a session;
- ``records``: with the message held from the world's turn (``message_boundary``,
  ``message_boundary_holds=True``) and ``longterm`` records keyed by it, each turn its own
  episode (the records' own format, one fact an episode);
- ``records_session``: the same brain, each session one episode (several facts an episode).

**Sessions.** A session begins with ``Brain.reset()``, which empties the message and the recall
memory's slots; the records and the learned weights continue (``records`` resets at every turn
as well).

**The life.** A nursery of lesson conversations: at each question's silent moment the brain is
taught the answer (``lesson``) and then acts. Then readings on fresh conversations without
lessons, each on a copy of the nursery's brain: every answer said greedily and scored exactly,
grouped by the question's tags. ``erase`` reads with the records emptied before the reading
(``records``) or the message emptied at each answer's moment (``message``).

    PYTHONPATH=src python benchmarks/dialogue/chamber.py --plan plan.json --out rows.jsonl
"""

from __future__ import annotations

import argparse
import copy
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

import world as wd  # noqa: E402

ARMS = ("previous_word", "message", "records", "records_session")
STREAMS = 16
UNITS = 280


def make_brain(lexicon: wd.Lexicon, founder: int, arm: str, records: int = 4096,
               module: int = 64) -> Any:
    warnings.simplefilter("ignore")
    import cadence as cd

    settings: dict[str, Any] = {}
    if arm != "previous_word":
        settings.update(message=UNITS, message_fan_in=max(16, round(16 / 70 * 2 * lexicon.inputs)))
    if arm in ("records", "records_session"):
        settings.update(message_boundary=lexicon.index[wd.TURN], message_boundary_holds=True,
                        longterm=int(records))
    return cd.Brain.compose(lexicon.inputs, lexicon.actions, modules=(module,), seed=int(founder),
                            backend="cpu", **settings)


def live(brain: Any, lexicon: wd.Lexicon, batch: list[wd.Conversation], arm: str, teach: bool,
         erase: str | None = None, scores: dict[str, list[float]] | None = None,
         work: dict[str, float] | None = None) -> None:
    """The streams of ``batch`` (conversations of one shape) through ``brain``: every word a
    moment, every answer said at its silent moment, taught where ``teach``; each answer's score
    appended to ``scores`` under each of its tags."""
    n = len(batch)
    rows = np.arange(n)
    for s in range(len(batch[0])):
        brain.reset()
        for t in range(len(batch[0][s])):
            turns = [c[s][t] for c in batch]
            if arm == "records" and t:
                brain.reset()
            steps = [lexicon.moments(turn) for turn in turns]
            for m in range(len(steps[0])):
                x = np.stack([steps[i][m][0] for i in range(n)])
                label = steps[0][m][1]
                if label is None:
                    brain.act(x, greedy=True)
                    continue
                labels = np.array([steps[i][m][1] for i in range(n)])
                if erase == "message" and getattr(brain, "message", None) is not None:
                    brain.message.reset(n)
                t0 = time.perf_counter()
                if teach:
                    brain.lesson(x, labels)
                said = np.asarray(brain.act(x, greedy=True)).reshape(n, -1)[:, 0]
                if work is not None:
                    work["answers"] += n
                    work["seconds"] += time.perf_counter() - t0
                if scores is not None:
                    right = said == labels
                    for i in rows:
                        for key in tags(turns[i], lexicon, int(said[i])):
                            scores[key].append(float(right[i]))


def tags(turn: wd.Turn, lexicon: wd.Lexicon, said: int) -> list[str]:
    """The groups a question's score joins, and its stale reading."""
    t = turn.tags
    gap = int(t.get("gap", 0))
    span = "0" if gap == 0 else "1" if gap == 1 else "2-4" if gap < 5 else "5-9" if gap < 10 \
        else "10+"
    out = ["all", f"gap {span}"]
    if t.get("revised"):
        out.append("revised")
    if t.get("held"):
        out.append("held pair")
    if t.get("stale") is not None and lexicon.answers[said] == t["stale"]:
        out.append("stale")  # counted over the revised questions
    return out


def summarise(scores: dict[str, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    revised = len(scores.get("revised", []))
    for key, values in scores.items():
        if key == "stale":
            continue
        out[key] = {"right": round(float(np.mean(values)), 4), "questions": len(values)}
    out["stale"] = {"share": round(len(scores.get("stale", [])) / revised, 4) if revised else None,
                    "questions": revised}
    return out


# ------------------------------------------------------------------------------- the two lives


def synthetic_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    founder, arm = int(spec["founder"]), spec["arm"]
    world = wd.World(founder, int(spec.get("people", 12)), int(spec.get("places", 8)))
    lexicon = world.lexicon
    brain = make_brain(lexicon, founder, arm, int(spec.get("records", 4096)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 305, 2]))
    nursery: dict[str, list[float]] = defaultdict(list)
    t0 = time.perf_counter()
    sessions, facts = int(spec.get("sessions", 4)), int(spec.get("facts", 2))
    rounds = int(spec.get("rounds", 100))
    for r in range(rounds):
        batch = world.conversations(rng, STREAMS, sessions, facts, lesson=True)
        live(brain, lexicon, batch, arm, teach=True,
             scores=nursery if r >= rounds - max(rounds // 10, 1) else None)
    base = {"founder": founder, "arm": arm, "chamber": "synthetic",
            "parameters": int(brain.parameters()), "neurons": int(brain.connectome.n),
            "nursery": summarise(nursery)["all"],
            "nursery_seconds": round(time.perf_counter() - t0)}
    out = []
    for reading in spec.get("readings", [{}]):
        reader = copy.deepcopy(brain)
        erase = reading.get("erase")
        if erase == "records" and getattr(reader, "longterm", None) is not None:
            reader.longterm.clear()
        scores: dict[str, list[float]] = defaultdict(list)
        work = {"answers": 0, "seconds": 0.0}
        read_rng = np.random.default_rng(np.random.SeedSequence([founder, 305, 3]))
        for _ in range(int(reading.get("batches", 8))):
            batch = world.conversations(read_rng, STREAMS, int(reading.get("sessions", sessions)),
                                        facts, lesson=False,
                                        wording=reading.get("wording", "trained"))
            live(reader, lexicon, batch, arm, teach=False, erase=erase, scores=scores, work=work)
        out.append({**base, "reading": reading, "scores": summarise(scores),
                    "answer_ms": round(1e3 * work["seconds"] / max(work["answers"], 1), 4)})
    return out


def babi_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    founder, arm, task = int(spec["founder"]), spec["arm"], int(spec["task"])
    folder = Path(spec["data"])
    (train_path,) = folder.glob(f"qa{task}_*_train.txt")
    (test_path,) = folder.glob(f"qa{task}_*_test.txt")
    train, test = wd.babi_qa(train_path), wd.babi_qa(test_path)
    lexicon = wd.babi_lexicon(train, test)
    brain = make_brain(lexicon, founder, arm, int(spec.get("records", 4096)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 305, 4, task]))
    t0 = time.perf_counter()
    nursery: dict[str, list[float]] = defaultdict(list)
    epochs = int(spec.get("epochs", 3))
    for e in range(epochs):
        for i in rng.permutation(len(train)):
            live(brain, lexicon, [train[int(i)]], arm, teach=True,
                 scores=nursery if e == epochs - 1 else None)
    base = {"founder": founder, "arm": arm, "chamber": f"babi-{task}",
            "parameters": int(brain.parameters()), "neurons": int(brain.connectome.n),
            "nursery": summarise(nursery)["all"],
            "nursery_seconds": round(time.perf_counter() - t0)}
    out = []
    for reading in spec.get("readings", [{}]):
        reader = copy.deepcopy(brain)
        erase = reading.get("erase")
        if erase == "records" and getattr(reader, "longterm", None) is not None:
            reader.longterm.clear()
        scores: dict[str, list[float]] = defaultdict(list)
        work = {"answers": 0, "seconds": 0.0}
        for story in test[: int(reading.get("stories", len(test)))]:
            live(reader, lexicon, [story], arm, teach=False, erase=erase, scores=scores,
                 work=work)
        out.append({**base, "reading": reading, "scores": summarise(scores),
                    "answer_ms": round(1e3 * work["seconds"] / max(work["answers"], 1), 4)})
    return out


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return babi_cell(spec) if "task" in spec else synthetic_cell(spec)


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """A plan's lives: every founder by every arm (and every bAbI task in ``tasks``)."""
    common = {k: v for k, v in plan.items() if k not in ("founders", "arms", "tasks")}
    tasks = [{"task": int(t)} for t in plan["tasks"]] if "tasks" in plan else [{}]
    return [{**common, **task, "founder": f, **arm}
            for task in tasks for f in plan["founders"] for arm in plan["arms"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data", type=Path, default=Path("."),
                        help="the folder the plan's bAbI data paths are relative to")
    args = parser.parse_args(argv)
    todo = [{**c, "data": str(args.data / c["data"])} if "data" in c else c
            for c in cells(json.loads(args.plan.read_text()))]
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
