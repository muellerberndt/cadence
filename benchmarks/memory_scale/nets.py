"""The transformer over the window (cadence#288): an independently scored reference, never part
of an answer path.

The window holds N records as N tokens, each an address code followed by a content code, and
then the cue as one more token with a silent content and a flag. A transformer encoder (2 layers,
2 heads, feed-forward twice the width, no position code: the records are a set) is taught by
backpropagation to give the right record's content at the cue's place, by the squared error, on
record sets drawn anew for every step with the instrument's own families and cues
(``world.records``, ``world.cue``). The first quarter of the steps holds 32 records, the second
128, the rest N.

It is read on fresh record sets: an answer is right when the stored content nearest to it is the
right record's. The read is by decoding, the network's own criterion; the memories' ``found`` asks
for more than half of the read's mass.

Full self-attention keeps N squared weights per head and layer, 400 MB at N = 10,000 in float32,
so the window sizes are the ones the device holds; ``read_n`` reads a taught network at a larger
window than it was taught at.

    PYTHONPATH=src python benchmarks/memory_scale/nets.py --plan plan.json --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RATE, CLIP = 0.001, 1.0


def build(width: int, value_width: int, model: int, seed: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class Reader(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 2

        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(width + value_width + 1, model)
            layer = nn.TransformerEncoderLayer(model, self.HEADS, 2 * model, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(model, value_width)

        def forward(self, tokens: Any) -> Any:
            return self.out(self.body(self.inp(tokens))[:, -1])

    return Reader()


def batch(rng: np.random.Generator, n: int, rows: int, width: int, value_width: int,
          kinds: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """``rows`` windows of ``n`` records and a cue: the tokens, each window's contents, the
    record each cue asks for and the cue's kind."""
    import world

    tokens = np.zeros((rows, n + 1, width + value_width + 1), dtype=np.float32)
    contents = np.zeros((rows, n, value_width), dtype=np.float32)
    wanted = np.zeros(rows, dtype=np.int64)
    asked = []
    for b in range(rows):
        recs = world.records(rng, n, width, value_width)
        kind = kinds[int(rng.integers(len(kinds)))]
        pool = world.targets(rng, recs, kind, 1)
        if not len(pool):  # a window too small to hold a family asks a singleton instead
            kind = "exact" if kind == "similar" else "noisy"
            pool = world.targets(rng, recs, kind, 1)
        r = int(pool[0])
        tokens[b, :n, :width] = recs.address
        tokens[b, :n, width : width + value_width] = recs.content
        tokens[b, n, :width] = world.cue(rng, recs.address[r], kind)
        tokens[b, n, -1] = 1.0
        contents[b], wanted[b] = recs.content, r
        asked.append(kind)
    return tokens, contents, wanted, asked


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    import world

    founder, n, width = int(spec["founder"]), int(spec["n"]), int(spec["width"])
    value_width = int(spec.get("value_width", width))
    model, steps, rows = int(spec.get("model", 256)), int(spec["steps"]), int(spec.get("rows", 16))
    device = torch.device(spec.get("device", "cpu"))
    torch.set_num_threads(1)
    net = build(width, value_width, model, founder).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(spec.get("rate", RATE)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, 8, n]))
    started = time.perf_counter()
    for step in range(steps):
        size = min(n, 32) if step < steps // 4 else min(n, 128) if step < steps // 2 else n
        many = max(rows, min(64, rows * n // size))
        tokens, contents, wanted, _ = batch(rng, size, many, width, value_width, world.CUES)
        answer = net(torch.from_numpy(tokens).to(device))
        target = torch.from_numpy(contents[np.arange(many), wanted]).to(device)
        loss = ((answer - target) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        optimizer.step()
    seconds = time.perf_counter() - started
    out = []
    fresh = np.random.default_rng(np.random.SeedSequence([founder, 288, 9, n]))
    net.eval()
    for size in [n, *[int(m) for m in spec.get("read_n", [])]]:
        for kind in world.CUES:
            right, total, read_seconds = 0, 0, 0.0
            few = max(1, min(rows, 64_000 // size))
            with torch.no_grad():
                while total < int(spec.get("trials", 400)):
                    tokens, contents, wanted, _ = batch(fresh, size, few, width, value_width,
                                                        (kind,))
                    t0 = time.perf_counter()
                    answer = net(torch.from_numpy(tokens).to(device)).cpu().numpy()
                    read_seconds += time.perf_counter() - t0
                    for b in range(few):  # a window without a family asks a singleton
                        right += int(int(np.argmax(contents[b] @ answer[b])) == wanted[b])
                        total += 1
            out.append({
                "founder": founder, "arm": "transformer", "n": size, "taught_n": n,
                "width": width, "cue": kind, "heads": 1, "right": round(right / total, 4),
                "trials": total, "steps": steps, "model": model,
                "parameters": int(sum(p.numel() for p in net.parameters())),
                "train_seconds": round(seconds, 1), "loss": round(float(loss.detach()), 5),
                "read_ms": round(1e3 * read_seconds / total, 4), "device": str(device),
            })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        for cell in plan["cells"]:
            for founder in plan["founders"]:
                for row in run_cell({**cell, "founder": founder,
                                     "device": plan.get("device", "cpu")}):
                    sink.write(json.dumps(row) + "\n")
                sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
