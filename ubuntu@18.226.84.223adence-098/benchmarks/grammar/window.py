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
to the bit (the observation is the previous word with ``SPEAK``). The window advances through
the world's echo (``advance="echo"``: the caller hears the window back) or through the efference
copy (``advance="efference"``: the brain hears ``SPEAK`` alone and its copy holds the commands it
issued, in training the words it said, not the target's).

Beside the chamber's panels, two readings per ``k``: Dyck sentences that fit in one window
against those that spill over it, and context twins aligned to the window (clauses of two
distractors whose first window is given, the twins differing in the agent's number or the
meaning's verb; for ``k`` above 5 the first window holds the verb and the chamber's panels
serve).

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
PANELS = ("clause", "clause_held", "clause_long", "roles", "dyck", "dyck_deep")
TWINS = gr.TWINS + ("win_agent", "win_verb")


ADVANCES = ("echo", "efference")


def make_brain(k: int, founder: int, arm: str = "recipe", module: int = 64,
               advance: str = "echo") -> Any:
    import cadence as cd

    if advance not in ADVANCES:
        raise ValueError("advance is 'echo' or 'efference'")
    extra = {"efference_amplitude": 1.0} if advance == "efference" else {}
    return cd.Brain.compose(gr.INPUTS, k * gr.ACTIONS, backend="cpu", modules=(module,),
                            seed=int(founder), slots=int(k),
                            **{"recall": False, **gr.ARMS[arm], **extra})


def heard_window(words: list[int], advance: str = "echo") -> np.ndarray:
    """The brain's previous window as it hears it: each word at 1.0, with ``SPEAK``; under the
    efference advance ``SPEAK`` alone, the copy holding the commands."""
    x = gr.concept("SPEAK")
    if advance == "echo":
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
          work: dict[str, float] | None = None, cap: int = 4,
          advance: str = "echo") -> list[list[int]]:
    """One batch on fresh streams, ``k`` words per step. Teaching: a lesson on the window's
    words before the act, the next step hearing the target's window. Reading: closed loop, the
    next step hearing the brain's own window. A row's ``force`` (whole windows, the same for the
    batch) gives its first windows as the target's, read from the next. Returns each row's
    words from its ``force`` through its first ``end`` (or ``cap`` words past its target)."""
    work = work if work is not None else gr.new_work()
    n = len(rows)
    if len({len(r.pre) for r in rows}) != 1:
        raise ValueError("a batch's rows hear the same number of moments")
    force = rows[0].force
    if any(r.force != force for r in rows) or force % k:
        raise ValueError("the window reads sentences given whole windows, the same per batch")
    given = force // k
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
            x = np.stack([heard_window(targets[i][j - 1] if teach or j <= given
                                       else previous[i], advance) for i in range(n)])
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
            if gr.END in words and j >= given:
                done[i] = True
        if not teach and all(done):
            break
    out = []
    for i in range(n):
        words = said[i][force:]
        out.append(words[:words.index(gr.END) + 1] if gr.END in words else words)
    return out


def forced(brain: Any, rows: list[gr.Row], k: int, advance: str = "echo") -> list[float]:
    """Each word said right with every previous window given (component evidence; under the
    efference advance the copy holds the brain's own previous window)."""
    brain.reset()
    for t in range(len(rows[0].pre)):
        brain.act(np.stack([r.pre[t] for r in rows]), greedy=True)
    targets = [windows(r.target, k) for r in rows]
    hits = []
    for j in range(len(targets[0])):
        x = (np.tile(gr.concept("GO"), (len(rows), 1)) if j == 0
             else np.stack([heard_window(t[j - 1], advance) for t in targets]))
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(rows), k)
        for i, r in enumerate(rows):
            for s in range(k):
                if r.force <= j * k + s < len(r.target):
                    hits.append(float(action[i, s] == r.target[j * k + s]))
    return hits


def locate(brain: Any, batches: list[list[gr.Row]], k: int,
           advance: str = "echo") -> dict[str, Any]:
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
                 else np.stack([heard_window(t[j - 1], advance) for t in targets]))
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


def read_panel(brain: Any, batches: list[list[gr.Row]], k: int, *, twins: bool = False,
               advance: str = "echo") -> dict[str, Any]:
    """The panel on a fork; with ``fits``, exact sentences that fit in one window of ``k``
    (``end`` included) apart from those that spill over it."""
    fork = copy.deepcopy(brain)
    out = gr.read_speaker(lambda rows: speak(fork, rows, k, teach=False, advance=advance),
                          lambda rows: forced(fork, rows, k, advance), batches, twins=twins)
    rows_all = [r for rows in batches for r in rows]
    said_all = speak_all(copy.deepcopy(brain), batches, k, advance)
    for name, keep in (("fits", lambda r: len(r.target) <= k),
                       ("spills", lambda r: len(r.target) > k)):
        hits = [gr.score(r, s)["exact"] for r, s in zip(rows_all, said_all, strict=True)
                if keep(r)]
        out[name] = round(float(np.mean(hits)), 4) if hits else None
        out[name + "_n"] = len(hits)
    return out


def speak_all(brain: Any, batches: list[list[gr.Row]], k: int, advance: str) -> list[list[int]]:
    return [s for rows in batches for s in speak(brain, rows, k, teach=False, advance=advance)]


def context_panels(founder: int, k: int, batches: int = 4) -> dict[str, list[list[gr.Row]]]:
    """Context twins aligned to the window: clauses of two distractors whose first window is
    given, the twins differing in the agent's number (``win_agent``) or the meaning's verb
    (``win_verb``); the continuation holds the verb for ``k`` up to 5. For ``k = 1`` the given
    part is the agent and the distractors, as the chamber's twins."""
    if k > 5:
        return {}
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), gr.SEEDS["test"], 257]))
    out: dict[str, list[list[gr.Row]]] = {}
    force = 5 if k == 1 else k
    for name, change in (("win_agent", "agent"), ("win_verb", "verb")):
        rows_all = []
        for _ in range(batches):
            rows = []
            for _ in range(gr.BATCH // 2):
                base = gr.trained_clause(rng, split, 2, False)
                m = base.meta
                for _ in range(64):
                    other = (gr.twin(base, agent=(m["agent"][0], 1 - m["agent"][1]))
                             if change == "agent" else gr.twin(base, v=1 - m["verb"]))
                    om = other.meta
                    if not split.held(tuple(om["agent"]), tuple(om["patient"]), om["verb"]):
                        break
                    base = gr.trained_clause(rng, split, 2, False)
                    m = base.meta
                for r in (base, other):
                    r.force = force
                rows += [base, other]
            rows_all.append(rows)
        out[name] = rows_all
    return out


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``k``, ``founder``, ``batches`` of the first grammar (the chamber's training
    stream, every family said through the window), ``arm``, ``panel_batches``."""
    warnings.simplefilter("ignore")
    import cadence as cd

    k, founder = int(spec["k"]), int(spec["founder"])
    arm, count = spec.get("arm", "recipe"), int(spec.get("batches", 4000))
    advance = str(spec.get("advance", "echo"))
    started = time.monotonic()
    brain = make_brain(k, founder, arm, int(spec.get("module", 64)), advance)
    split = gr.Split.of(founder)
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    tests.update(context_panels(founder, k, int(spec.get("panel_batches", 4))))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    work = gr.new_work()
    sentences = 0
    for _ in range(count):
        batch = gr.train_batch(rng, split, "g1")
        speak(brain, batch, k, teach=True, work=work, advance=advance)
        sentences += len(batch)
    work["seconds"] = round(time.monotonic() - started, 1)
    work["sentences"] = sentences
    row: dict[str, Any] = {
        "k": k, "founder": founder, "arm": arm, "advance": advance, "batches": count,
        "settings": gr.ARMS[arm], "module": int(spec.get("module", 64)),
        "parameters": int(brain.parameters()),
        "synapses": int(brain.describe()["layout"]["synapses"]),
        "cadence": cd.__version__, "work": work,
        "panels": {name: read_panel(brain, tests[name], k, twins=name in TWINS,
                                    advance=advance)
                   for name in PANELS + tuple(n for n in ("win_agent", "win_verb")
                                              if n in tests)},
        "located": {name: locate(brain, tests[name], k, advance)
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
    parser.add_argument("--advance", default="echo", choices=ADVANCES)
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs in place of the grid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.plan is not None:
        specs = json.loads(args.plan.read_text())
    else:
        specs = [{"k": k, "founder": f, "batches": args.batches, "arm": args.arm,
                  "panel_batches": args.panel_batches, "advance": args.advance}
                 for k in args.ks for f in args.founders]
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
