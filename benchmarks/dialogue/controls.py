"""Matched controls for the conversation chamber (cadence#305): a long short-term memory network
(``lstm``) and a causal transformer (``transformer``) hear the same conversations as the brain,
one word a moment with the world's ``TURN`` channel, and are trained by cross-entropy on the same
answers at the same silent moments. Their size is the one whose parameter count is nearest to
the brain's (``target``). They are scored controls, never part of the brain's answer path.

- The LSTM hears a conversation from its start; its state continues across sessions.
- The transformer reads a causal window of ``WINDOW`` moments ending at each moment: one session
  and as many earlier sessions as fit.
- ``online``: one pass over the lesson conversations in the brain's order, one Adam step per
  batch (rate 0.003, gradient norm clipped at 1); ``offline``: the same batches replayed in
  shuffled epochs (rate 0.001) until the loss stops falling (patience 3, at most 30 epochs).

Work is counted as answers, analytic multiply-accumulates per moment (a backward pass at twice
the forward) and seconds.

    PYTHONPATH=src python benchmarks/dialogue/controls.py --plan plan.json --out rows.jsonl
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

import chamber as ch  # noqa: E402
import world as wd  # noqa: E402

RATES = {"online": 0.003, "offline": 0.001}
CLIP, PATIENCE, EPOCHS = 1.0, 3, 30
WINDOW = 256
MAX_LENGTH = 8192


def encode(lexicon: wd.Lexicon, batch: list[wd.Conversation]) -> tuple[Any, Any, list[Any]]:
    """``(batch, moments, inputs)`` moments, ``(batch, moments)`` answers (-100 where none) and
    each answer's turn, ``[(row, moment, turn)]``; shorter conversations are padded at the end."""
    xs, ys, where = [], [], []
    for row, conversation in enumerate(batch):
        moments, labels = [], []
        for session in conversation:
            for turn in session:
                for x, label in lexicon.moments(turn):
                    if label is not None:
                        where.append((row, len(moments), turn))
                    moments.append(x)
                    labels.append(-100 if label is None else label)
        xs.append(np.stack(moments))
        ys.append(labels)
    length = max(len(x) for x in xs)
    x = np.zeros((len(xs), length, lexicon.inputs), dtype=np.float32)
    y = np.full((len(xs), length), -100, dtype=np.int64)
    for i, (a, b) in enumerate(zip(xs, ys, strict=True)):
        x[i, : len(a)], y[i, : len(b)] = a, b
    return x, y, where


def build(model: str, width: int, inputs: int, actions: int, seed: int) -> Any:
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

        def macs(self, length: int) -> int:
            return length * (4 * width * (inputs + width) + width * actions)

    class Transformer(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 2

        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(inputs, width)
            layer = nn.TransformerEncoderLayer(width, self.HEADS, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(width, actions)
            position = torch.arange(MAX_LENGTH)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(MAX_LENGTH, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            i = torch.arange(n)
            blocked = (i[None, :] > i[:, None]) | (i[:, None] - i[None, :] >= WINDOW)
            mask = torch.zeros(n, n).masked_fill(blocked, float("-inf"))
            h = self.inp(x) + self.position[:n]
            return self.out(self.body(h, mask=mask))

        def macs(self, length: int) -> int:
            per = inputs * width + self.LAYERS * 8 * width * width + width * actions
            seen = min(length, WINDOW)
            return length * per + self.LAYERS * 2 * width * length * seen

    return LSTM() if model == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, target: int, inputs: int, actions: int) -> int:
    best = None
    for width in range(4, 513, 2):
        n = count(build(model, width, inputs, actions, 0))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (width, n)
    assert best is not None
    return best[0]


class Learner:
    def __init__(self, model: str, width: int, lexicon: wd.Lexicon, seed: int,
                 rate: float) -> None:
        import torch

        self.torch = torch
        self.net = build(model, width, lexicon.inputs, lexicon.actions, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "answers": 0, "macs": 0, "seconds": 0.0}

    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        torch = self.torch
        t0 = time.perf_counter()
        self.net.train()
        logits = self.net(torch.from_numpy(x))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                 torch.from_numpy(y).reshape(-1),
                                                 ignore_index=-100)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
        self.opt.step()
        self.work["updates"] += 1
        self.work["macs"] += 3 * self.net.macs(x.shape[1]) * x.shape[0]
        self.work["seconds"] += time.perf_counter() - t0
        return float(loss.item())

    def said(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            return np.asarray(self.net(torch.from_numpy(x)).argmax(-1))


Batches = list[list[wd.Conversation]]


def lessons(spec: dict[str, Any]) -> tuple[wd.Lexicon, Batches,
                                            list[tuple[dict[str, Any], Batches]]]:
    """The lesson batches and each reading's batches the brain's life hears (the same draws)."""
    founder = int(spec["founder"])
    if "task" in spec:
        folder = Path(spec["data"])
        (train_path,) = folder.glob(f"qa{int(spec['task'])}_*_train.txt")
        (test_path,) = folder.glob(f"qa{int(spec['task'])}_*_test.txt")
        train, test = wd.babi_qa(train_path), wd.babi_qa(test_path)
        lexicon = wd.babi_lexicon(train, test)
        rng = np.random.default_rng(np.random.SeedSequence([founder, 305, 4, int(spec["task"])]))
        order = [int(i) for _ in range(int(spec.get("epochs", 3)))
                 for i in rng.permutation(len(train))]
        size = ch.STREAMS
        taught = [[train[i] for i in order[k:k + size]] for k in range(0, len(order), size)]
        readings = [(r, [test[k:k + size]
                         for k in range(0, int(r.get("stories", len(test))), size)])
                    for r in spec.get("readings", [{}])]
        return lexicon, taught, readings
    world = wd.World(founder, int(spec.get("people", 12)), int(spec.get("places", 8)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 305, 2]))
    sessions, facts = int(spec.get("sessions", 4)), int(spec.get("facts", 2))
    taught = [world.conversations(rng, ch.STREAMS, sessions, facts, lesson=True)
              for _ in range(int(spec.get("rounds", 100)))]
    readings = []
    for reading in spec.get("readings", [{}]):
        read_rng = np.random.default_rng(np.random.SeedSequence([founder, 305, 3]))
        readings.append((reading, [world.conversations(
            read_rng, ch.STREAMS, int(reading.get("sessions", sessions)), facts, lesson=False,
            wording=reading.get("wording", "trained"), gap=reading.get("gap"))
            for _ in range(int(reading.get("batches", 8)))]))
    return world.lexicon, taught, readings


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    torch.set_num_threads(1)
    founder, model, mode = int(spec["founder"]), spec["model"], spec["mode"]
    lexicon, taught, readings = lessons(spec)
    if "target" not in spec:  # the parameters of the brain the plan names (``target_arm``)
        brain = ch.make_brain(lexicon, founder, spec.get("target_arm", "records"))
        spec = {**spec, "target": int(brain.parameters())}
    width = matched_width(model, int(spec["target"]), lexicon.inputs, lexicon.actions)
    learner = Learner(model, width, lexicon, founder, RATES[mode])
    encoded = [encode(lexicon, batch)[:2] for batch in taught]
    epochs, losses = 0, []
    if mode == "online":
        losses.append(float(np.mean([learner.update(x, y) for x, y in encoded])))
        epochs = 1
    else:
        rng = np.random.default_rng(founder)
        best, waited = math.inf, 0
        while epochs < EPOCHS and waited < PATIENCE:
            loss = float(np.mean([learner.update(*encoded[int(i)])
                                  for i in rng.permutation(len(encoded))]))
            losses.append(loss)
            epochs += 1
            best, waited = (loss, 0) if loss < best - 1e-4 else (best, waited + 1)
    base = {"founder": founder, "model": model, "mode": mode, "width": width,
            "parameters": count(learner.net), "target": int(spec["target"]),
            "chamber": f"babi-{spec['task']}" if "task" in spec else "synthetic",
            "epochs": epochs, "loss": round(losses[-1], 4), "train": dict(learner.work)}
    out = []
    for reading, batches in readings:
        scores: dict[str, list[float]] = defaultdict(list)
        t0, answers, macs = time.perf_counter(), 0, 0
        for batch in batches:
            x, y, where = encode(lexicon, batch)
            said = learner.said(x)
            macs += learner.net.macs(x.shape[1]) * x.shape[0]
            for row, moment, turn in where:
                answers += 1
                right = float(said[row, moment] == y[row, moment])
                for key in ch.tags(turn, lexicon, int(said[row, moment])):
                    scores[key].append(right)
        out.append({**base, "reading": reading, "scores": ch.summarise(scores),
                    "answer_ms": round(1e3 * (time.perf_counter() - t0) / max(answers, 1), 4),
                    "answer_macs": int(macs / max(answers, 1))})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data", type=Path, default=Path("."),
                        help="the folder the plan's bAbI data paths are relative to")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if "data" in plan:
        plan["data"] = str(args.data / plan["data"])
    common = {k: v for k, v in plan.items() if k not in ("founders", "models", "modes", "tasks")}
    tasks = [{"task": int(t)} for t in plan["tasks"]] if "tasks" in plan else [{}]
    todo = [{**common, **task, "founder": f, "model": m, "mode": mode} for task in tasks
            for f in plan["founders"] for m in plan["models"] for mode in plan["modes"]]
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
