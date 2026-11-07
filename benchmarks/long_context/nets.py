"""The matched networks on the long-context tasks (cadence#268): independently scored controls,
never part of the brain's answer path.

The chamber's controls (``benchmarks/grammar/controls.py``) at another task and window: a causal
transformer (2 layers, 2 heads, feed-forward twice the width, a sinusoidal position code over the
stream) and an LSTM, each sized so its parameter count is the nearest to the brain's
``parameters()`` at the same capacity and table. They hear the brain's moments in the brain's order
(the same episodes, from ``SeedSequence([founder, 11])``), and are taught the answers it is taught,
teacher-forced, by cross-entropy at Adam rate 0.003 with the gradient clipped at norm 1.

A window: each forward pass covers the last ``context + stride`` moments of the episode and its loss
covers the answers of the last ``stride`` moments, one Adam step per pass. So every answer attends
over at least ``context`` moments (the brain's slots); ``stride`` is how many moments share a step,
which the brain does not need and a network pays for per step. After the brain's exposure a life
goes on to ``max_answers`` or ``max_seconds`` on further episodes, so the same life reads both
comparisons: equal exposure at the brain's answers, equal compute at the brain's seconds on the
same machine and one thread.

    PYTHONPATH=src python benchmarks/long_context/nets.py --plan plan.json --out rows.jsonl \\
        --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _module("long_context_reach", Path(__file__).with_name("reach.py"))
B = R.B
RATE, CLIP = 0.003, 1.0


def build(model: str, inputs: int, actions: int, width: int, seed: int, length: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class LSTM(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTM(inputs, width, batch_first=True)
            self.out = nn.Linear(width, actions)

        def forward(self, x: Any, start: int = 0) -> Any:
            h, _ = self.cell(x)
            return self.out(h)

        def macs(self, n: int) -> int:
            return n * (4 * width * (inputs + width) + width * actions)

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

        def forward(self, x: Any, start: int = 0) -> Any:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            h = self.inp(x) + self.position[start:start + n]
            return self.out(self.body(h, mask=mask, is_causal=True))

        def macs(self, n: int) -> int:
            per_token = inputs * width + self.LAYERS * 8 * width * width + width * actions
            return n * per_token + self.LAYERS * width * n * (n + 1)

    return LSTM() if model == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, inputs: int, actions: int, target: int) -> int:
    best = None
    for width in range(4, 513, 2):
        n = count(build(model, inputs, actions, width, 0, 8))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (width, n)
        if n > 2 * target:
            break
    assert best is not None
    return best[0]


def arrays(moments: list[Any]) -> tuple[np.ndarray, np.ndarray, list[list[str] | None]]:
    x = np.stack([m.x for m in moments], axis=1).astype(np.float32)
    y = np.full(x.shape[:2], -100, dtype=np.int64)
    names: list[list[str] | None] = []
    for t, m in enumerate(moments):
        if m.label is not None:
            y[:, t] = m.label
        names.append(m.condition if m.label is not None else None)
    return x, y, names


def slices(length: int, context: int, stride: int, answered: np.ndarray) -> Any:
    """Each pass of an episode: ``(start, first, end)``: it covers ``[start, end)`` and its loss the
    answers of ``[first, end)``; a stride with no answers takes no pass."""
    first = 0
    while first < length:
        end = min(length, first + stride)
        if answered[first:end].any():
            yield max(0, end - context - stride), first, end
        first = end


class Learner:
    def __init__(self, model: str, inputs: int, actions: int, width: int, seed: int, length: int,
                 context: int, stride: int) -> None:
        import torch

        self.torch = torch
        self.net = build(model, inputs, actions, width, seed, length)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=RATE)
        self.context, self.stride, self.actions = context, stride, actions
        self.work = {"steps": 0, "answers": 0, "tokens": 0, "macs": 0, "seconds": 0.0,
                     "read_seconds": 0.0}
        self.loss = 0.0

    def teach(self, moments: list[Any]) -> Any:
        """Teach an episode, yielding after each step."""
        torch = self.torch
        x, y, _ = arrays(moments)
        answered = (y >= 0).any(axis=0)
        for start, first, end in slices(x.shape[1], self.context, self.stride, answered):
            began = time.monotonic()
            target = y[:, start:end].copy()
            target[:, : first - start] = -100
            logits = self.net(torch.from_numpy(x[:, start:end]), start)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.actions),
                                                     torch.from_numpy(target).reshape(-1),
                                                     ignore_index=-100)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
            self.opt.step()
            n = end - start
            self.work["steps"] += 1
            self.work["answers"] += int(answered[first:end].sum())
            self.work["tokens"] += x.shape[0] * n
            self.work["macs"] += 3 * x.shape[0] * self.net.macs(n)
            self.work["seconds"] += time.monotonic() - began
            self.loss = float(loss.item())
            yield

    def read(self, task: Any, rng: np.random.Generator, episodes: int) -> dict[str, Any]:
        torch = self.torch
        began = time.monotonic()
        self.net.eval()
        cells: dict[str, list[int]] = {}
        with torch.no_grad():
            for _ in range(episodes):
                x, y, names = arrays(task.episode(rng, train=False))
                answered = (y >= 0).any(axis=0)
                for start, first, end in slices(x.shape[1], self.context, self.stride, answered):
                    logits = self.net(torch.from_numpy(x[:, start:end]), start)
                    action = logits.argmax(dim=2).numpy()
                    self.work["macs"] += x.shape[0] * self.net.macs(end - start)
                    for t in np.flatnonzero(answered[first:end]) + first:
                        for r, condition in enumerate(names[t] or []):
                            cell = cells.setdefault(condition, [0, 0])
                            cell[0] += int(action[r, t - start] == y[r, t])
                            cell[1] += 1
        self.net.train()
        self.work["read_seconds"] += time.monotonic() - began
        return {k: {"hit": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(cells.items())}


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    warnings.simplefilter("ignore")
    import torch

    torch.set_num_threads(1)
    founder = int(spec["founder"])
    model = spec["model"]
    task = R.build_task(spec["task"], spec.get("task_args", {}))
    brain_spec = {"arm": "Fd", "founder": founder, "capacity": spec["capacity"],
                  "position": spec["position"], "settings": spec.get("brain_settings")}
    target = int(spec.get("parameters") or R.compose(task, brain_spec, backend="cpu").parameters())
    width = int(spec.get("width") or matched_width(model, task.inputs, task.actions, target))
    reads = [(r, R.build_task(r["task"], r.get("task_args", {}))) for r in spec["reads"]]
    context = int(spec["context"])
    stride = int(spec["stride"])
    length = int(spec.get("length", 8192))
    learner = Learner(model, task.inputs, task.actions, width, founder, length, context, stride)
    rng = np.random.default_rng(np.random.SeedSequence([founder, B.SEEDS["train"]]))
    exposure = [int(a) for a in spec.get("at_answers", [])]
    max_answers, max_seconds = int(spec["max_answers"]), float(spec["max_seconds"])
    row: dict[str, Any] = {
        "tag": spec["tag"], "model": model, "task": spec["task"], "task_settings": task.settings,
        "founder": founder, "width": width, "parameters": count(learner.net),
        "brain_parameters": target, "context": context, "stride": stride, "batch": B.BATCH,
        "readings": [], "torch": torch.__version__,
    }
    started = time.monotonic()
    partial = spec.get("partial")

    def checkpoint(reason: str) -> None:
        cells = {}
        for index, (reading, read_task) in enumerate(reads):
            test = np.random.default_rng(np.random.SeedSequence([founder, B.SEEDS["test"], index]))
            cells[reading.get("name", reading["task"])] = learner.read(
                read_task, test, int(reading.get("episodes", 1)))
        entry = {"reason": reason, "loss": learner.loss, **learner.work, "conditions": cells}
        row["readings"].append(entry)
        if partial:
            with open(partial, "a") as sink:
                sink.write(json.dumps({"tag": spec["tag"], "model": model, "width": width,
                                       "parameters": row["parameters"], **entry}) + "\n")

    next_power = 8
    stop = False
    while not stop:
        for _ in learner.teach(task.episode(rng, train=True)):
            work = learner.work
            reason = None
            while exposure and work["answers"] >= exposure[0]:
                reason = f"answers {exposure.pop(0)}"
            if work["steps"] >= next_power:
                reason = reason or f"steps {work['steps']}"
                next_power *= 2
            if work["answers"] >= max_answers or work["seconds"] >= max_seconds:
                stop = True
                break
            if reason:
                checkpoint(reason)
    checkpoint("end")
    row["seconds"] = round(time.monotonic() - started, 2)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    specs = json.loads(args.plan.read_text())
    done = set()
    if args.out.exists():
        done = {json.loads(line).get("tag") for line in args.out.read_text().splitlines() if line}
    specs = [{**s, "partial": str(args.out.with_suffix(".partial.jsonl"))} for s in specs
             if s.get("tag") not in done]
    print(f"{len(specs)} lives, {len(done)} already read", flush=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for row in map(run_life, specs):
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                print(row["tag"], row["seconds"], flush=True)
        else:
            with get_context("spawn").Pool(min(args.workers, max(len(specs), 1)),
                                           maxtasksperchild=1) as pool:
                for row in pool.imap_unordered(run_life, specs, chunksize=1):
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                    print(row["tag"], row["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
