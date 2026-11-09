"""Routes with a goal (cadence#257): a chamber where the first word anticipates a constraint no
earlier word settles.

The world says the facts of a small graph, one fact ``X near Y end`` at a time, heard words with
no ``SPEAK`` flag: from a start, two branches of the same length (two or three hops), each a chain
of distinct places. The meaning names the start (``FROM``) and the goal (``TO``), the last place of
one branch. At ``GO`` the brain says the route as the facts say it, ``start near a1 near ...
near goal end``, one word per moment, hearing its previous word with ``SPEAK``. The first hop is
the decision: the start has two neighbours, and only one of them leads to the goal, which the
brain names only at the route's end. A speaker word by word says the first hop before the
goal; a planning window of three settles the first hop with the words after it.

Held out (``Split.of``): eight ordered start-goal pairs per founder that no training episode
joins, so ``held`` panels are routes between places training never joined, on graphs drawn
fresh. ``trained`` panels draw from the trained pairs. Every panel is four batches of sixteen
episodes (``panel_batches`` sets the count), two and three hops in turn, from
``SeedSequence([founder, 33, 257])``; the training stream is ``SeedSequence([founder, 11,
257])``.

Speakers:

- ``word``: ``Brain.compose(INPUTS, ACTIONS)``, a lesson on each word before the greedy act.
- ``sliding``: ``Brain.compose(INPUTS, 3 * ACTIONS, slots=3)``: at each moment the lesson teaches
  the next three words (``end`` past the route's end), the act settles three slots at once, the
  brain says the first slot's word and hears it, then plans again.
- ``lstm``, ``transformer``: next-word learners at the brain's parameter count (``controls.py``'s
  architectures and rates), taught teacher-forced on the same episodes, online in the brain's
  order or offline to convergence, read greedy and closed loop by the same scorer.

Scores: ``first_hop`` (the place said after the first ``near`` is the goal branch's first place),
``exact`` (the
whole route through ``end``) and ``valid`` (every word said after the start follows the one
before it by a fact, the last is the goal, then ``end``).

    python benchmarks/grammar/routes.py run --founders 0 1 2 --out DIR/r.jsonl
    python benchmarks/grammar/routes.py controls --founders 0 1 2 --out DIR/c.jsonl
    python benchmarks/grammar/routes.py read DIR/r.jsonl DIR/c.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

PLACES = ("dog", "cat", "fox", "owl", "dogs", "cats", "foxes", "owls")
WORDS = (*PLACES, "near", "end")
NEAR, END = len(PLACES), len(PLACES) + 1
CONCEPTS = tuple(p.upper() for p in PLACES) + ("FROM", "TO", "GO", "SPEAK")
C = {name: len(WORDS) + i for i, name in enumerate(CONCEPTS)}
ACTIONS, INPUTS = len(WORDS), len(WORDS) + len(CONCEPTS)
BATCH, HOPS, WINDOW = 16, (2, 3), 3
SEEDS = {"train": (11, 257), "test": (33, 257), "split": (7, 257)}
BATCHES = 3000
FOUNDERS = tuple(range(9625700, 9625706))


def vector(*names: str) -> np.ndarray:
    x = np.zeros(INPUTS)
    for name in names:
        x[C[name]] = 1.0
    return x


def heard(word: int) -> np.ndarray:
    x = np.zeros(INPUTS)
    x[word] = 1.0
    return x


def spoken(word: int) -> np.ndarray:
    x = heard(word)
    x[C["SPEAK"]] = 1.0
    return x


@dataclass
class Row:
    pre: list[np.ndarray]  # the facts, the meaning; then GO
    target: list[int]  # the route, start to goal, then end
    edges: set[tuple[int, int]] = field(default_factory=set)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Split:
    held: frozenset[tuple[int, int]]

    @classmethod
    def of(cls, founder: int) -> Split:
        rng = np.random.default_rng(np.random.SeedSequence([int(founder), *SEEDS["split"]]))
        pairs = [(s, g) for s in range(len(PLACES)) for g in range(len(PLACES)) if s != g]
        chosen = rng.choice(len(pairs), size=8, replace=False)
        return cls(frozenset(pairs[int(i)] for i in chosen))


def episode(rng: np.random.Generator, hops: int, start: int, goal: int) -> Row:
    """A graph of two branches of ``hops`` from ``start``, one ending at ``goal``."""
    others = [p for p in range(len(PLACES)) if p not in (start, goal)]
    picked = [int(p) for p in rng.permutation(others)[: 2 * hops - 1]]
    good = [start, *picked[: hops - 1], goal]
    bad = [start, *picked[hops - 1:]]
    edges = [(a, b) for chain in (good, bad) for a, b in zip(chain, chain[1:], strict=False)]
    order = rng.permutation(len(edges))
    pre: list[np.ndarray] = []
    for i in order:
        a, b = edges[int(i)]
        pre += [heard(a), heard(NEAR), heard(b), heard(END)]
    pre += [vector(CONCEPTS[start], "FROM"), vector(CONCEPTS[goal], "TO")]
    route = [w for place in good for w in (place, NEAR)][:-1]
    return Row(pre, [*route, END], set(edges),
               {"hops": hops, "start": start, "goal": goal, "first": good[1], "other": bad[1]})


def draw(rng: np.random.Generator, split: Split, hops: int, held: bool) -> Row:
    while True:
        s, g = (int(v) for v in rng.choice(len(PLACES), size=2, replace=False))
        if ((s, g) in split.held) == held:
            return episode(rng, hops, s, g)


def batch(rng: np.random.Generator, split: Split, hops: int, held: bool = False) -> list[Row]:
    return [draw(rng, split, hops, held) for _ in range(BATCH)]


def train_batches(founder: int, count: int) -> list[list[Row]]:
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), *SEEDS["train"]]))
    split = Split.of(founder)
    return [batch(rng, split, HOPS[b % len(HOPS)]) for b in range(count)]


def panels(founder: int, batches: int = 4) -> dict[str, list[list[Row]]]:
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), *SEEDS["test"]]))
    split = Split.of(founder)
    return {name: [batch(rng, split, HOPS[b % len(HOPS)], held) for b in range(batches)]
            for name, held in (("trained", False), ("held", True))}


def score(row: Row, said: list[int]) -> dict[str, Any]:
    words = said[: said.index(END) + 1] if END in said else said
    places = words[0:-1:2]
    valid = (len(words) >= 4 and len(words) % 2 == 0 and words[-1] == END
             and all(w == NEAR for w in words[1:-1:2]) and places[0] == row.meta["start"]
             and places[-1] == row.meta["goal"]
             and all((a, b) in row.edges for a, b in zip(places, places[1:], strict=False)))
    hop = words[2] if len(words) > 2 and words[1] == NEAR else None
    return {"exact": words == row.target, "first_hop": hop == row.meta["first"],
            "other_hop": hop == row.meta["other"], "valid": bool(valid)}


def tally(rows: list[Row], said: list[list[int]]) -> dict[str, Any]:
    scores = [score(r, s) for r, s in zip(rows, said, strict=True)]
    out: dict[str, Any] = {"n": len(scores)}
    for key in scores[0]:
        out[key] = float(np.mean([s[key] for s in scores]))
    for hops in HOPS:
        kept = [s for r, s in zip(rows, scores, strict=True) if r.meta["hops"] == hops]
        if kept:
            out[f"first_hop_{hops}"] = float(np.mean([s["first_hop"] for s in kept]))
            out[f"exact_{hops}"] = float(np.mean([s["exact"] for s in kept]))
    return out


# ------------------------------------------------------------------------------- the brain


def make_brain(speaker: str, founder: int, settings: dict[str, Any] | None = None,
               module: int = 64) -> Any:
    import cadence as cd

    k = WINDOW if speaker == "sliding" else 1
    return cd.Brain.compose(INPUTS, k * ACTIONS, modules=(module,), seed=int(founder),
                            slots=k, **{"backend": "cpu", **(settings or {})})


def new_work() -> dict[str, float]:
    return {"lessons": 0, "rows": 0, "agreed_rows": 0, "lesson_row_sweeps": 0, "acts": 0,
            "act_sweeps": 0, "refused": 0, "exhausted": 0}


def ahead(target: list[int], j: int, k: int) -> list[int]:
    """The ``k`` words from position ``j``, ``end`` past the route's end."""
    part = target[j:j + k]
    return part + [END] * (k - len(part))


def _act(brain: Any, x: np.ndarray, k: int, work: dict[str, float]) -> np.ndarray:
    try:
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), k)
        work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
    except RuntimeError:
        work["exhausted"] += 1
        action = np.full((len(x), k), -1)
    work["acts"] += 1
    return action


def speak(brain: Any, rows: list[Row], k: int, *, teach: bool,
          work: dict[str, float] | None = None, cap: int = 3) -> list[list[int]]:
    """One batch on fresh streams. Taught: a lesson on the next ``k`` words before each act and
    the next moment hearing the target's word. Read: greedy, closed loop. Either way the brain
    says the first slot's word."""
    work = work if work is not None else new_work()
    n = len(rows)
    brain.reset()
    for t in range(len(rows[0].pre)):
        _act(brain, np.stack([r.pre[t] for r in rows]), k, work)
    said: list[list[int]] = [[] for _ in rows]
    steps = max(len(r.target) for r in rows) + (0 if teach else cap)
    for j in range(steps):
        if j == 0:
            x = np.tile(vector("GO"), (n, 1))
        else:
            words = [r.target[j - 1] if teach else said[i][j - 1] for i, r in enumerate(rows)]
            x = np.stack([spoken(w) if w >= 0 else vector("SPEAK") for w in words])
        if teach:
            labels = np.array([ahead(r.target, j, k) for r in rows])
            try:
                report = brain.lesson(x, labels if k > 1 else labels[:, 0])
                work["lessons"] += 1
                work["rows"] += report.get("attempted_presentations", 0.0)
                work["agreed_rows"] += report.get("agreed_presentations", 0.0)
                work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
            except RuntimeError:
                work["refused"] += 1
        action = _act(brain, x, k, work)
        for i in range(n):
            said[i].append(int(action[i, 0]))
        if not teach and all(END in s for s in said):
            break
    return said


def read(speaker: Any, batches: list[list[Row]]) -> dict[str, Any]:
    rows = [r for b in batches for r in b]
    said = [s for b in batches for s in speaker(b)]
    return tally(rows, said)


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One brain life: ``speaker`` (``word`` or ``sliding``), ``founder``, ``batches``,
    ``checkpoints``, ``module``, ``panel_batches``, ``settings``."""
    warnings.simplefilter("ignore")
    import cadence as cd

    speaker, founder = spec["speaker"], int(spec["founder"])
    k = WINDOW if speaker == "sliding" else 1
    count = int(spec.get("batches", BATCHES))
    started = time.monotonic()
    brain = make_brain(speaker, founder, spec.get("settings"), int(spec.get("module", 64)))
    tests = panels(founder, int(spec.get("panel_batches", 4)))
    checkpoints = set(int(c) for c in spec.get("checkpoints", []))
    work = new_work()
    row: dict[str, Any] = {"speaker": speaker, "founder": founder, "k": k, "batches": count,
                           "module": int(spec.get("module", 64)),
                           "parameters": int(brain.parameters()), "cadence": cd.__version__,
                           "tag": spec.get("tag"), "readings": []}

    def reading(b: int) -> dict[str, Any]:
        fork = copy.deepcopy(brain)
        return {"batches": b, "work": dict(work), "panels": {
            name: read(lambda rows: speak(fork, rows, k, teach=False), tests[name])
            for name in tests}}

    for b, rows in enumerate(train_batches(founder, count), start=1):
        speak(brain, rows, k, teach=True, work=work)
        if b in checkpoints or b == count:
            row["readings"].append(reading(b))
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


# ---------------------------------------------------------------------------- the controls


def encode(rows: list[Row]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        moments = [*r.pre, vector("GO")] + [spoken(w) for w in r.target[:-1]]
        xs.append(np.stack(moments))
        ys.append([-100] * len(r.pre) + list(r.target))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def build(model: str, width: int, seed: int) -> Any:
    """``controls.py``'s two architectures at this chamber's inputs and actions."""
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class LSTM(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTM(INPUTS, width, batch_first=True)
            self.out = nn.Linear(width, ACTIONS)

        def forward(self, x: Any) -> Any:
            return self.out(self.cell(x)[0])

    class Transformer(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(INPUTS, width)
            layer = nn.TransformerEncoderLayer(width, 2, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.out = nn.Linear(width, ACTIONS)
            position = torch.arange(64)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(64, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            return self.out(self.body(self.inp(x) + self.position[:n], mask=mask,
                                      is_causal=True))

    return LSTM() if model == "lstm" else Transformer()


def matched_width(model: str, target: int) -> int:
    best = None
    for width in range(4, 257, 2):
        n = int(sum(p.numel() for p in build(model, width, 0).parameters()))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (width, n)
    assert best is not None
    return best[0]


def control_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One control: ``model``, ``mode`` (``online``: one pass, one Adam step per batch at
    0.003; ``offline``: shuffled epochs at 0.001 until the loss stops falling, patience 3, at
    most 30), ``founder``, ``batches``, ``parameters`` (to match)."""
    import torch

    torch.set_num_threads(1)
    model, mode, founder = spec["model"], spec["mode"], int(spec["founder"])
    count = int(spec.get("batches", BATCHES))
    width = matched_width(model, int(spec["parameters"]))
    net = build(model, width, founder)
    opt = torch.optim.Adam(net.parameters(), lr=0.003 if mode == "online" else 0.001)
    data = [encode(rows) for rows in train_batches(founder, count)]

    def update(x: np.ndarray, y: np.ndarray) -> float:
        logits = net(torch.from_numpy(x))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, ACTIONS),
                                                 torch.from_numpy(y).reshape(-1),
                                                 ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        return float(loss)

    started = time.monotonic()
    epochs = 1
    if mode == "online":
        for x, y in data:
            update(x, y)
    else:
        order = np.random.default_rng(np.random.SeedSequence([founder, 99, 257]))
        best, stale, epochs = math.inf, 0, 0
        while epochs < 30 and stale < 3:
            mean = float(np.mean([update(*data[int(i)]) for i in order.permutation(count)]))
            epochs += 1
            stale = 0 if mean < best - 1e-4 else stale + 1
            best = min(best, mean)

    def speaker(rows: list[Row]) -> list[list[int]]:
        said: list[list[int]] = [[] for _ in rows]
        with torch.no_grad():
            for _ in range(max(len(r.target) for r in rows) + 3):
                moments = [[*r.pre, vector("GO")] + [spoken(w) for w in s]
                           for r, s in zip(rows, said, strict=True)]
                x = np.stack([np.stack(m) for m in moments]).astype(np.float32)
                words = net(torch.from_numpy(x))[:, -1].argmax(dim=1).numpy()
                for s, w in zip(said, words, strict=True):
                    s.append(int(w))
                if all(END in s for s in said):
                    break
        return said

    net.eval()
    tests = panels(founder, int(spec.get("panel_batches", 4)))
    return {"model": model, "mode": mode, "founder": founder, "width": width,
            "parameters": int(sum(p.numel() for p in net.parameters())), "epochs": epochs,
            "seconds": round(time.monotonic() - started, 1), "tag": spec.get("tag"),
            "panels": {name: read(speaker, tests[name]) for name in tests}}


# -------------------------------------------------------------------------------- reading


def gate(brains: list[dict[str, Any]], panel: str = "held") -> dict[str, Any]:
    """The declared gate: the sliding window's first-hop accuracy on ``panel`` minus word by
    word's, paired by founder at the last reading; planning pays where the mean difference is
    at least .05 and beyond two standard errors."""
    last = {(r["speaker"], int(r["founder"])): r["readings"][-1]["panels"][panel]["first_hop"]
            for r in brains}
    founders = sorted(f for s, f in last if s == "word" and ("sliding", f) in last)
    diffs = np.array([last[("sliding", f)] - last[("word", f)] for f in founders])
    if len(diffs) < 2:
        return {"founders": founders}
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    mean = float(diffs.mean())
    return {"founders": founders, "differences": [round(float(d), 3) for d in diffs],
            "mean": round(mean, 4), "se": round(se, 4),
            "pays": bool(mean >= 0.05 and mean > 2 * se)}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pool(fn: Any, specs: list[dict[str, Any]], out: Path, workers: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as sink:
        if workers <= 1:
            for result in map(fn, specs):
                sink.write(json.dumps(result) + "\n")
                sink.flush()
        else:
            with get_context("spawn").Pool(min(workers, len(specs))) as pool:
                for result in pool.imap_unordered(fn, specs, chunksize=1):
                    sink.write(json.dumps(result) + "\n")
                    sink.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--speakers", nargs="*", default=["word", "sliding"])
    r.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    r.add_argument("--batches", type=int, default=BATCHES)
    r.add_argument("--checkpoints", nargs="*", type=int, default=[])
    r.add_argument("--panel-batches", type=int, default=16)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=1)
    c = sub.add_parser("controls")
    c.add_argument("--models", nargs="*", default=["lstm", "transformer"])
    c.add_argument("--modes", nargs="*", default=["online", "offline"])
    c.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    c.add_argument("--batches", type=int, default=BATCHES)
    c.add_argument("--panel-batches", type=int, default=16)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--workers", type=int, default=1)
    g = sub.add_parser("read")
    g.add_argument("brains", type=Path)
    g.add_argument("controls", type=Path, nargs="?")
    args = parser.parse_args(argv)
    if args.command == "run":
        specs = [{"speaker": s, "founder": f, "batches": args.batches,
                  "checkpoints": args.checkpoints, "panel_batches": args.panel_batches,
                  "tag": f"routes/{s}/f{f}"} for s in args.speakers for f in args.founders]
        _pool(run_life, specs, args.out, args.workers)
        return 0
    if args.command == "controls":
        warnings.simplefilter("ignore")
        target = int(make_brain("word", args.founders[0]).parameters())
        specs = [{"model": m, "mode": mode, "founder": f, "batches": args.batches,
                  "parameters": target, "panel_batches": args.panel_batches}
                 for m in args.models for mode in args.modes for f in args.founders]
        _pool(control_life, specs, args.out, args.workers)
        return 0
    brains = _rows(args.brains)
    for b in sorted(brains, key=lambda r: (r["speaker"], r["founder"])):
        last = b["readings"][-1]["panels"]
        print(f"{b['speaker']:8s} f{b['founder']} " + " ".join(
            f"{p} first {v['first_hop']:.2f} exact {v['exact']:.2f} valid {v['valid']:.2f}"
            for p, v in last.items()))
    if args.controls:
        for c_ in sorted(_rows(args.controls), key=lambda r: (r["model"], r["mode"],
                                                              r["founder"])):
            print(f"{c_['model']}/{c_['mode']} f{c_['founder']} " + " ".join(
                f"{p} first {v['first_hop']:.2f} exact {v['exact']:.2f}"
                for p, v in c_["panels"].items()))
    print(json.dumps({"held": gate(brains, "held"), "trained": gate(brains, "trained")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
