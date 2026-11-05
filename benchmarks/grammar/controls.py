"""Next-word learners at matched information: the grammar chamber's controls.

A long short-term memory network (``lstm``) and a causal transformer (``transformer``) hear the
same moments as the brain (the meaning's concept moments, the world's words, ``GO`` and each
previous word with the ``SPEAK`` flag) and are trained to predict each word of the sentence,
teacher-forced, by cross-entropy. Their size is chosen so that their parameter count is the
nearest to the Cadence brain's ``parameters()``. They are independently scored controls and
never part of the brain's answer path.

- ``online``: one pass in the brain's order, one Adam step per batch the brain is taught
  (rate 0.003, gradient norm clipped at 1), the brain's exposure exactly.
- ``offline``: the same batches stored and replayed in shuffled epochs (rate 0.001) until the
  epoch's loss stops falling (patience 3, at most 30 epochs): an upper reference, reported
  apart.

Each is read on the brain's panels by the same scorer, closed loop, greedy. Work is counted as
tokens and analytic multiply-accumulates (a backward pass at twice the forward) and wall time.

    python benchmarks/grammar/controls.py --models lstm transformer --modes online offline \\
        --founders 0 1 2 --out DIR/controls.jsonl --workers 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import grammar as gr
import numpy as np

MODELS = ("lstm", "transformer")
MODES = ("online", "offline")
RATES = {"online": 0.003, "offline": 0.001}
CLIP, PATIENCE, EPOCHS = 1.0, 3, 30
MAX_LENGTH = 64  # positions the transformer's encoding covers


def encode(rows: list[gr.Row]) -> tuple[np.ndarray, np.ndarray]:
    """The moments the brain hears and the words it is taught, teacher-forced:
    ``(batch, moments, inputs)`` and ``(batch, moments)`` with -100 where nothing is taught."""
    xs, ys = [], []
    for r in rows:
        moments = [*r.pre, gr.concept("GO")] + [gr.spoken(w) for w in r.target[:-1]]
        labels = [-100] * len(r.pre) + list(r.target)
        xs.append(np.stack(moments))
        ys.append(labels)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def build(model: str, width: int, seed: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class LSTM(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTM(gr.INPUTS, width, batch_first=True)
            self.out = nn.Linear(width, gr.ACTIONS)

        def forward(self, x: Any) -> Any:
            h, _ = self.cell(x)
            return self.out(h)

        def macs(self, length: int) -> int:
            return length * (4 * width * (gr.INPUTS + width) + width * gr.ACTIONS)

    class Transformer(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 2

        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(gr.INPUTS, width)
            layer = nn.TransformerEncoderLayer(width, self.HEADS, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(width, gr.ACTIONS)
            position = torch.arange(MAX_LENGTH)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(MAX_LENGTH, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            h = self.inp(x) + self.position[:n]
            return self.out(self.body(h, mask=mask, is_causal=True))

        def macs(self, length: int) -> int:
            per_token = (gr.INPUTS * width + self.LAYERS * (4 * width * width
                                                           + 4 * width * width)
                         + width * gr.ACTIONS)
            attention = self.LAYERS * width * length * (length + 1)  # scores and mixing
            return length * per_token + attention

    return LSTM() if model == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, target: int) -> int:
    """The width whose parameter count is nearest to ``target`` (even, for the transformer)."""
    best = None
    for width in range(4, 257, 2):
        n = count(build(model, width, 0))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (width, n)
    assert best is not None
    return best[0]


class Learner:
    def __init__(self, model: str, width: int, seed: int, rate: float) -> None:
        import torch

        self.torch = torch
        self.net = build(model, width, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "tokens": 0, "macs": 0, "seconds": 0.0}

    def update(self, rows: list[gr.Row]) -> float:
        torch = self.torch
        started = time.monotonic()
        x, y = encode(rows)
        logits = self.net(torch.from_numpy(x))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, gr.ACTIONS),
                                                 torch.from_numpy(y).reshape(-1),
                                                 ignore_index=-100)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
        self.opt.step()
        self.work["updates"] += 1
        self.work["tokens"] += x.shape[0] * x.shape[1]
        self.work["macs"] += 3 * x.shape[0] * self.net.macs(x.shape[1])
        self.work["seconds"] += time.monotonic() - started
        return float(loss.item())

    def speak(self, rows: list[gr.Row], cap: int = 4) -> list[list[int]]:
        """Closed loop, greedy: each row hears its own previous word after its ``force``."""
        torch = self.torch
        n = len(rows)
        said: list[list[int]] = [[] for _ in rows]
        done = [False] * n
        moments = [[*r.pre, gr.concept("GO")] for r in rows]
        longest = max(len(r.target) for r in rows) + cap
        with torch.no_grad():
            for j in range(longest):
                x = torch.from_numpy(np.stack([np.stack(m) for m in moments]).astype(np.float32))
                logits = self.net(x)[:, -1]
                action = logits.argmax(dim=1).numpy()
                self.work["macs"] += n * self.net.macs(x.shape[1])
                for i, r in enumerate(rows):
                    said[i].append(int(action[i]))
                    if j >= r.force and action[i] == gr.END:
                        done[i] = True
                    given = j + 1 <= r.force and j < len(r.target)
                    moments[i].append(gr.spoken(r.target[j] if given else int(action[i])))
                if all(done):
                    break
        out = []
        for i, r in enumerate(rows):
            words = said[i][r.force:]
            out.append(words[:words.index(gr.END) + 1] if gr.END in words else words)
        return out

    def forced(self, rows: list[gr.Row]) -> list[float]:
        torch = self.torch
        x, y = encode(rows)
        with torch.no_grad():
            action = self.net(torch.from_numpy(x)).argmax(dim=2).numpy()
        hits = []
        for i, r in enumerate(rows):
            start = len(r.pre) + r.force
            hits += [float(action[i, t] == y[i, t]) for t in range(start, y.shape[1])]
        return hits


def reading(learner: Learner, tests: dict[str, list[list[gr.Row]]],
            names: tuple[str, ...]) -> dict[str, Any]:
    learner.net.eval()
    out = {name: gr.read_speaker(learner.speak, learner.forced, tests[name],
                                 twins=name in gr.TWINS) for name in names}
    learner.net.train()
    return out


def brain_parameters(founder: int, arm: str = "recipe") -> int:
    import warnings

    warnings.simplefilter("ignore")
    return int(gr.make_brain(arm, founder).parameters())


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One control life: ``model``, ``mode``, ``founder``, ``phases`` as the brain's."""
    import torch

    torch.set_num_threads(1)
    model, mode, founder = spec["model"], spec["mode"], int(spec["founder"])
    phases = [(str(g), int(n)) for g, n in spec.get("phases", [["g1", 2000]])]
    target = int(spec.get("parameters") or brain_parameters(founder))
    width = int(spec.get("width") or matched_width(model, target))
    started = time.monotonic()
    learner = Learner(model, width, founder, RATES[mode])
    split = gr.Split.of(founder)
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    everything = gr.G1_PANELS + gr.CONTEXT_PANELS + ("g2", "g2_long")
    row: dict[str, Any] = {
        "model": model, "mode": mode, "founder": founder, "width": width,
        "parameters": count(learner.net), "brain_parameters": target,
        "phases": [list(p) for p in phases], "rate": RATES[mode], "readings": [],
        "torch": torch.__version__,
    }
    for index, (grammar, n) in enumerate(phases):
        batches = [gr.train_batch(rng, split, grammar) for _ in range(n)]
        before = dict(learner.work)
        losses: list[float] = []
        if mode == "online":
            losses = [learner.update(b) for b in batches]
            epochs = 1
        else:
            order = np.random.default_rng(np.random.SeedSequence([founder, 99, index]))
            best, stale, epochs = math.inf, 0, 0
            while epochs < EPOCHS and stale < PATIENCE:
                epoch = [learner.update(batches[int(i)]) for i in order.permutation(n)]
                epochs += 1
                mean = float(np.mean(epoch))
                losses.append(mean)
                stale = 0 if mean < best - 1e-4 else stale + 1
                best = min(best, mean)
        work = {k: learner.work[k] - before[k] for k in learner.work}
        row["readings"].append({
            "phase": index, "grammar": grammar, "batches": n, "epochs": epochs,
            "loss_last": float(np.mean(losses[-50:])) if mode == "online" else losses[-1],
            "work": work, "panels": reading(learner, tests, everything)})
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    parser.add_argument("--modes", nargs="*", default=["online"], choices=MODES)
    parser.add_argument("--founders", nargs="*", type=int, default=list(gr.FOUNDERS))
    parser.add_argument("--phases", type=json.loads, default=[["g1", 2000]])
    parser.add_argument("--parameters", type=int, default=0,
                        help="the brain's parameter count (default: the recipe brain's)")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.plan is not None:
        specs = json.loads(args.plan.read_text())
    else:
        specs = [{"model": m, "mode": mode, "founder": f, "phases": args.phases,
                  "parameters": args.parameters}
                 for m in args.models for mode in args.modes for f in args.founders]
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
