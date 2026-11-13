"""Matched controls for the several-records chamber (cadence#325): a transformer whose attention
reads the records in its context, and an LSTM that reads the same context.

Each control hears one sequence per question: ``window`` told facts, each as the brain hears a
told fact (its words, the turn before its object), in a shuffled order, then the question's
words, the turn and the silent moment, where ``SLOTS`` readouts each say one of the slot's
answers. A moment is the instrument's form code through a linear input layer, so a word no
lesson used is heard as the brain hears it.

**The context is handed.** The window always holds every fact the answer needs, every other
fact of each subject the question names or its chain reaches (the records the brain's read
scores), and facts drawn at random up to ``window``. The brain finds its records among all it
holds; the control is given the window that fits. ``in_window`` is 1 by construction.

**Training.** The lessons are the brain's: the same lesson population and the same draws of
questions. ``online`` is one pass over the brain's ``lessons`` questions; ``offline`` trains on
fresh draws of lesson questions until its score on 512 held lesson questions stops rising
(patience 3 rounds of ``lessons`` questions), ``rounds`` rounds pass or the time budget ends.
Adam, cross-entropy summed over the slots, gradient norm clipped at 1.

**Width.** The width whose parameter count is nearest ``target``, the brain's parameters.
Work is counted as seconds and analytic multiply-accumulates (a backward pass at twice the
forward).

    PYTHONPATH=src python benchmarks/several/several_controls.py --plan plan.json --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import several_chamber as ch  # noqa: E402
import several_world as sw  # noqa: E402

CLIP = 1.0
LENGTH = 1024  # the longest sequence a model holds positions for


def build(kind: str, width: int, inputs: int, slot_actions: int, seed: int, layers: int = 4,
          heads: int = 8) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    actions = sw.SLOTS * slot_actions

    class LSTM(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTM(inputs, width, batch_first=True)
            self.out = nn.Linear(width, actions)

        def forward(self, x: Any, lengths: Any) -> Any:
            h, _ = self.cell(x)
            return self.out(h[torch.arange(len(x)), lengths - 1])

        def macs(self, length: int) -> int:
            return length * 4 * width * (inputs + width) + width * actions

    class Transformer(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(inputs, width)
            layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.out = nn.Linear(width, actions)
            position = torch.arange(LENGTH)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(LENGTH, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def forward(self, x: Any, lengths: Any) -> Any:
            n = x.shape[1]
            padded = torch.arange(n)[None, :] >= lengths[:, None]
            h = self.body(self.inp(x) + self.position[:n], src_key_padding_mask=padded)
            return self.out(h[torch.arange(len(x)), lengths - 1])

        def macs(self, length: int) -> int:
            per = inputs * width + layers * 8 * width * width
            return length * per + layers * 2 * width * length * length + width * actions

    return LSTM() if kind == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(kind: str, target: int, inputs: int, slot_actions: int, layers: int,
                  heads: int) -> int:
    """The width (a multiple of the heads) whose parameter count is nearest ``target``."""
    unit = heads if kind == "transformer" else 2

    def size(width: int) -> int:
        return count(build(kind, width, inputs, slot_actions, 0, layers, heads))

    lo, hi = 1, 2048 // unit
    while lo < hi:
        mid = (lo + hi) // 2
        if size(unit * mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return min({max(unit * (lo - 1), unit), unit * lo}, key=lambda w: (abs(size(w) - target), w))


def context(people: sw.Population, question: sw.Question, rng: np.random.Generator,
            window: int) -> list[sw.Fact]:
    """The facts a question's sequence holds: those its answer needs, every other fact of the
    subjects it names or its chain reaches, and facts drawn at random up to ``window``, in a
    shuffled order."""
    named = {w for w in question.said if w.startswith(people.prefix)}
    named |= {people.facts[i].answer for i in question.needs if i >= 0
              and people.facts[i].relation in sw.LINKS}
    held = {i for i in question.needs if i >= 0}
    for s in sorted(named):
        for r in (*sw.ATTRIBUTES, *sw.LINKS):
            found = people.where.get((r, s))
            if found is not None:
                held.add(found)
    if len(held) > window:
        raise RuntimeError("the window holds fewer facts than a question's subjects have")
    while len(held) < min(window, len(people.facts)):
        held.add(int(rng.integers(len(people.facts))))
    order = rng.permutation(sorted(held))
    return [people.facts[int(i)] for i in order]


def sequence(lexicon: sw.Lexicon, people: sw.Population, question: sw.Question,
             rng: np.random.Generator, window: int) -> tuple[np.ndarray, np.ndarray]:
    moments: list[np.ndarray] = []
    for fact in context(people, question, rng, window):
        moments += lexicon.told(fact)
    asked, labels = lexicon.asked(question)
    return np.stack(moments + asked).astype(np.float32), labels


def batch_of(items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[Any, Any, Any]:
    import torch

    lengths = np.array([len(x) for x, _ in items])
    x = np.zeros((len(items), int(lengths.max()), items[0][0].shape[1]), dtype=np.float32)
    for i, (moments, _) in enumerate(items):
        x[i, : len(moments)] = moments
    y = np.stack([labels for _, labels in items])
    return torch.from_numpy(x), torch.from_numpy(lengths), torch.from_numpy(y)


class Learner:
    def __init__(self, kind: str, width: int, lexicon: sw.Lexicon, seed: int, rate: float,
                 layers: int, heads: int) -> None:
        import torch

        self.torch = torch
        self.slot_actions = lexicon.slot_actions
        self.net = build(kind, width, lexicon.inputs, lexicon.slot_actions, seed, layers, heads)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "macs": 0.0, "seconds": 0.0}

    def update(self, items: list[tuple[np.ndarray, np.ndarray]]) -> float:
        torch = self.torch
        t0 = time.perf_counter()
        x, lengths, y = batch_of(items)
        self.net.train()
        logits = self.net(x, lengths).reshape(len(items) * sw.SLOTS, self.slot_actions)
        loss = torch.nn.functional.cross_entropy(logits, y.reshape(-1))
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
        self.opt.step()
        self.work["updates"] += 1
        self.work["macs"] += 3.0 * sum(self.net.macs(int(n)) for n in lengths)
        self.work["seconds"] += time.perf_counter() - t0
        return float(loss.item())

    def said(self, items: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        torch = self.torch
        x, lengths, _ = batch_of(items)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(x, lengths).reshape(len(items), sw.SLOTS, self.slot_actions)
            return np.asarray(logits.argmax(-1))


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    torch.set_num_threads(int(spec.get("threads", 1)))
    founder, kind, schedule = int(spec["founder"]), spec["kind"], spec["schedule"]
    window, lessons = int(spec.get("window", 64)), int(spec.get("lessons", 4000))
    layers, heads = int(spec.get("layers", 4)), int(spec.get("attention_heads", 8))
    lexicon = sw.Lexicon()
    taught = sw.Population([founder, 325, 1], int(spec.get("lesson_subjects", 600)), "t")
    width = int(spec.get("width") or matched_width(
        kind, int(spec["target"]), lexicon.inputs, lexicon.slot_actions, layers, heads))
    learner = Learner(kind, width, lexicon, founder, float(spec.get("rate", 1e-3)), layers, heads)
    size_batch = int(spec.get("batch", 16))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 325, 2]))  # the brain's draws
    window_rng = np.random.default_rng(np.random.SeedSequence([founder, 325, 5]))
    held_rng = np.random.default_rng(np.random.SeedSequence([founder, 325, 6]))
    held = [sequence(lexicon, taught, ch.lesson_question(taught, held_rng, spec), held_rng, window)
            for _ in range(512)]

    def held_score() -> float:
        right = []
        for start in range(0, len(held), 64):
            part = held[start : start + 64]
            right += list((learner.said(part) == np.stack([y for _, y in part])).all(axis=1))
        return float(np.mean(right))

    def one_round(draw: np.random.Generator) -> None:
        for _ in range(lessons // size_batch):
            learner.update([
                sequence(lexicon, taught, ch.lesson_question(taught, draw, spec), window_rng,
                         window) for _ in range(size_batch)])

    t0 = time.perf_counter()
    one_round(rng)  # the brain's lessons, once
    curve = [round(held_score(), 4)]
    rounds = 1
    if schedule == "offline":
        best, since = curve[0], 0
        budget = float(spec.get("budget", 3600.0))
        while rounds < int(spec.get("rounds", 60)) and time.perf_counter() - t0 < budget:
            one_round(rng)
            rounds += 1
            curve.append(round(held_score(), 4))
            if curve[-1] > best + 1e-3:
                best, since = curve[-1], 0
            else:
                since += 1
                if since >= 3:
                    break
    base = {"founder": founder, "arm": f"{kind}, {schedule}", "kind": kind,
            "schedule": schedule, "parameters": count(learner.net), "width": width,
            "layers": layers, "window": window, "lessons": lessons * rounds, "rounds": rounds,
            "nursery_curve": curve, "nursery_seconds": round(time.perf_counter() - t0),
            "nursery_macs": float(learner.work["macs"])}
    out = []
    n = int(spec.get("questions", 200))
    nothing = lexicon.action[sw.NOTHING]
    for size in [int(s) for s in spec["sizes"]]:
        world = ch.reading_population(founder, size)
        read_rng = np.random.default_rng(np.random.SeedSequence([founder, 325, 4, size]))
        scores: dict[str, list[float]] = defaultdict(list)
        costs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        def answer(question: sw.Question, world: Any = world, scores: Any = scores,
                   costs: Any = costs) -> None:
            item = sequence(lexicon, world, question, window_rng, window)
            t1 = time.perf_counter()
            said = learner.said([item])[0]
            seconds = time.perf_counter() - t1
            labels = item[1]
            group = f"{question.kind} {question.k}" + (
                " absent" if question.tags.get("absent") else "")
            asked = question.k if question.kind == "parts" else 1
            right = said == labels
            scores[f"said {group}"].append(float(right.all()))
            scores[f"part {group}"].append(float(right[:asked].mean()))
            if not (labels != nothing).any():
                scores[f"refused {group}"].append(float((said == nothing).all()))
            cost = costs[f"{question.kind} {question.k}"]
            cost["answers"] += 1
            cost["seconds"] += seconds
            cost["macs"] += learner.net.macs(len(item[0]))

        for k in spec.get("parts_read", [1, 2, 4, 8]):  # the chamber's draws, in its order
            for _ in range(n):
                answer(world.parts(read_rng, int(k)))
            for _ in range(n // 2):
                answer(world.parts(read_rng, int(k), absent=0.3))
        for hops in spec.get("chains_read", [2, 3]):
            for _ in range(n):
                answer(world.chain(read_rng, int(hops)))
            for _ in range(n // 2):
                answer(world.chain(read_rng, int(hops), broken=1.0))
        out.append({**base, "size": size, "scores": ch.summarise(scores),
                    "costs": {g: {"answer_ms": round(1e3 * c["seconds"] / c["answers"], 4),
                                  "answer_macs": int(c["macs"] / c["answers"])}
                              for g, c in sorted(costs.items())}})
    return out


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    common = {k: v for k, v in plan.items() if k not in ("founders", "models")}
    return [{**common, "founder": f, **model} for f in plan["founders"] for model in plan["models"]]


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
