"""A planning window: the grammar chamber's sentences said k words at a time.

The owner's question for the Grammar stage of cadence#173: should the brain settle several
words at once, as a speaker plans a phrase? This reading says the chamber's sentences with a
window of ``k`` words settled jointly. The brain is ``Brain.compose(INPUTS, k * ACTIONS,
slots=k)``: ``k`` motor slots of the eighteen words, one softmax each, settled in the one
settle. The message arrives as in ``grammar.py`` (the concept moments, then ``GO``); at each
production step the brain hears the words of its previous window (each at 1.0, with the
``SPEAK`` flag; at the first step ``GO``), settles the ``k`` slots at once, says their words and
moves the window on, until a window holds ``end``. The target of a window past the sentence's
end is ``end`` in every remaining slot. With ``k = 1`` this is the chamber's word-by-word life
to the bit (the observation is the previous word with ``SPEAK``).

The library settles the slots jointly, and each slot has motor neurons and synapses of its own:
no weight is shared across the positions of the window, so a position is a coordinate the
brain learns word by word, not a role the same synapses fill at every place.

    python benchmarks/grammar/window.py --ks 1 4 8 --founders 0 1 2 --out DIR/window.jsonl
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
import warnings
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


gr = _module("grammar", HERE / "grammar.py")

LONGEST = 2 * max(gr.TRAIN_L) + 4  # the longest trained clause, ``end`` included: 8 words
PANELS = ("clause", "clause_held", "clause_long", "roles")


def make_brain(k: int, founder: int, arm: str = "recipe", module: int = 64) -> Any:
    import cadence as cd

    return cd.Brain.compose(gr.INPUTS, k * gr.ACTIONS, modules=(module,), seed=int(founder),
                            slots=int(k), **gr.ARMS[arm])


def heard_window(words: list[int]) -> np.ndarray:
    """The brain's previous window as it hears it: each word at 1.0, with ``SPEAK``."""
    x = gr.concept("SPEAK")
    for w in words:
        if w >= 0:
            x[w] = 1.0
    return x


def windows(target: list[int], k: int) -> list[list[int]]:
    """The target cut into windows of ``k``, the last padded with ``end``."""
    out = []
    for start in range(0, len(target), k):
        part = target[start:start + k]
        out.append(part + [gr.END] * (k - len(part)))
    return out


def _act(brain: Any, x: np.ndarray, k: int, work: dict[str, float]) -> np.ndarray:
    try:
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), k)
        work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
    except RuntimeError:
        work["exhausted"] += 1
        action = np.full((len(x), k), -1)
    work["acts"] += 1
    return action


def speak(brain: Any, rows: list[gr.Row], k: int, *, teach: bool,
          work: dict[str, float] | None = None, cap: int = 4) -> list[list[int]]:
    """One batch on fresh streams, ``k`` words per step. Teaching: a lesson on the window's
    words before the act, the next step hearing the target's window. Reading: closed loop, the
    next step hearing the brain's own window. Returns each row's words through its first
    ``end`` (or ``cap`` words past its target)."""
    work = work if work is not None else gr.new_work()
    n = len(rows)
    if len({len(r.pre) for r in rows}) != 1:
        raise ValueError("a batch's rows hear the same number of moments")
    if any(r.force for r in rows):
        raise ValueError("the window reads sentences said from their first word")
    if teach and len({len(r.target) for r in rows}) != 1:
        raise ValueError("a taught batch's sentences have one length")
    brain.reset()
    for t in range(len(rows[0].pre)):
        _act(brain, np.stack([r.pre[t] for r in rows]), k, work)
    targets = [windows(r.target, k) for r in rows]
    steps = max(len(w) for w in targets) + (0 if teach else -(-cap // k))
    said: list[list[int]] = [[] for _ in rows]
    done = [False] * n
    previous: list[list[int]] = [[] for _ in rows]
    for j in range(steps):
        if j == 0:
            x = np.tile(gr.concept("GO"), (n, 1))
        else:
            x = np.stack([heard_window(targets[i][j - 1] if teach else previous[i])
                          for i in range(n)])
        if teach:
            labels = np.array([t[j] for t in targets])
            try:
                report = brain.lesson(x, labels if k > 1 else labels[:, 0])
                work["lessons"] += 1
                work["rows"] += report.get("attempted_presentations", 0.0)
                work["agreed_rows"] += report.get("agreed_presentations", 0.0)
                work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
                work["free_budget_exhausted"] += report.get("free_budget_exhausted", 0.0)
            except RuntimeError:
                work["refused"] += 1
        action = _act(brain, x, k, work)
        for i in range(n):
            words = [int(w) for w in action[i]]
            said[i] += words
            previous[i] = words
            if gr.END in words:
                done[i] = True
        if not teach and all(done):
            break
    out = []
    for i in range(n):
        words = said[i]
        out.append(words[:words.index(gr.END) + 1] if gr.END in words else words)
    return out


def forced(brain: Any, rows: list[gr.Row], k: int) -> list[float]:
    """Each word said right with every previous window given (component evidence)."""
    brain.reset()
    for t in range(len(rows[0].pre)):
        brain.act(np.stack([r.pre[t] for r in rows]), greedy=True)
    targets = [windows(r.target, k) for r in rows]
    hits = []
    for j in range(len(targets[0])):
        x = (np.tile(gr.concept("GO"), (len(rows), 1)) if j == 0
             else np.stack([heard_window(t[j - 1]) for t in targets]))
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(rows), k)
        for i, r in enumerate(rows):
            for s in range(k):
                if j * k + s < len(r.target):
                    hits.append(float(action[i, s] == r.target[j * k + s]))
    return hits


def locate(brain: Any, batches: list[list[gr.Row]], k: int) -> dict[str, Any]:
    """``grammar.locate`` for windows: at the step whose window holds a word, whether the
    recall read's first head attends most to the word's meaning moment or the one before it
    (``delivered``), the attention it gives that moment (``mass``), and whether the slot says
    the word."""
    fork = copy.deepcopy(brain)
    groups: dict[str, list[float]] = {}
    for rows in batches:
        fork.reset()
        written = 0
        for t in range(len(rows[0].pre)):
            fork.act(np.stack([r.pre[t] for r in rows]), greedy=True)
            written += 1
        expected = [gr.sources(r) for r in rows]
        targets = [windows(r.target, k) for r in rows]
        for j in range(len(targets[0])):
            x = (np.tile(gr.concept("GO"), (len(rows), 1)) if j == 0
                 else np.stack([heard_window(t[j - 1]) for t in targets]))
            ages = np.asarray(fork.recall.ages())
            action = np.asarray(fork.act(x, greedy=True)).reshape(len(rows), k)
            p = fork.recall.attended_heads[0]
            if p.shape[1] != len(ages):
                raise RuntimeError("the read's slots and their ages differ")
            for i, r in enumerate(rows):
                landed = written - 1 - int(ages[int(np.argmax(p[i]))])
                for s in range(k):
                    w = j * k + s
                    if w >= len(r.target) or expected[i][w] is None:
                        continue
                    source = expected[i][w]
                    column = np.where(written - 1 - ages == source)[0]
                    mass = float(p[i, column[0]]) if len(column) else 0.0
                    g = groups.setdefault(gr._kind(r, w), [0.0, 0.0, 0.0, 0.0, 0.0])
                    found = landed in (source, source - 1)
                    g[0] += float(found)
                    g[1] += float(action[i, s] == r.target[w])
                    g[2] += float(found and action[i, s] == r.target[w])
                    g[3] += mass
                    g[4] += 1
            written += 1
    return {name: {"n": int(v[4]), "delivered": round(v[0] / v[4], 4),
                   "said": round(v[1] / v[4], 4),
                   "said_when_delivered": round(v[2] / v[0], 4) if v[0] else None,
                   "mass": round(v[3] / v[4], 4)}
            for name, v in sorted(groups.items())}


def read_panel(brain: Any, batches: list[list[gr.Row]], k: int, *,
               twins: bool = False) -> dict[str, Any]:
    fork = copy.deepcopy(brain)
    return gr.read_speaker(lambda rows: speak(fork, rows, k, teach=False),
                           lambda rows: forced(fork, rows, k), batches, twins=twins)


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``k``, ``founder``, ``batches`` of the first grammar (the chamber's training
    stream, every family said through the window), ``arm``, ``panel_batches``."""
    warnings.simplefilter("ignore")
    import cadence as cd

    k, founder = int(spec["k"]), int(spec["founder"])
    arm, count = spec.get("arm", "recipe"), int(spec.get("batches", 4000))
    started = time.monotonic()
    brain = make_brain(k, founder, arm, int(spec.get("module", 64)))
    split = gr.Split.of(founder)
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    work = gr.new_work()
    sentences = 0
    for _ in range(count):
        batch = gr.train_batch(rng, split, "g1")
        speak(brain, batch, k, teach=True, work=work)
        sentences += len(batch)
    work["seconds"] = round(time.monotonic() - started, 1)
    work["sentences"] = sentences
    row: dict[str, Any] = {
        "k": k, "founder": founder, "arm": arm, "batches": count,
        "settings": gr.ARMS[arm], "module": int(spec.get("module", 64)),
        "parameters": int(brain.parameters()),
        "synapses": int(brain.describe()["layout"]["synapses"]),
        "cadence": cd.__version__, "work": work,
        "panels": {name: read_panel(brain, tests[name], k, twins=name in gr.TWINS)
                   for name in PANELS},
        "located": {name: locate(brain, tests[name], k)
                    for name in ("clause", "clause_held", "clause_long")},
    }
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ks", nargs="*", type=int, default=[1, 4, LONGEST])
    parser.add_argument("--founders", nargs="*", type=int, default=list(gr.FOUNDERS))
    parser.add_argument("--batches", type=int, default=4000)
    parser.add_argument("--arm", default="recipe", choices=list(gr.ARMS))
    parser.add_argument("--panel-batches", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    specs = [{"k": k, "founder": f, "batches": args.batches, "arm": args.arm,
              "panel_batches": args.panel_batches} for k in args.ks for f in args.founders]
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
