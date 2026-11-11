"""The letters chamber's matched baselines: an LSTM and a causal transformer trained by gradient.

Each hears the brain's frames and says one of the brain's symbols at every frame. They are
independently scored controls, never part of the brain's answer path.

- ``ctc``: connectionist temporal classification over a chunk's symbols (its letters and the
  space after each word, blank 0): the learner chooses its own timing.
- ``ce``: cross-entropy on the brain's per-frame lessons (``letters.labels``, the supplied
  alignment).
- ``split``: the split brain's per-frame targets, two heads: the symbol heard (``frames``) and
  whether to say it (1 at the supplied alignment); it says the first where the second says 1.

The training stream is cut into chunks of ``chunk_words`` words, each ending with its word's
silence. ``online``: one pass in the brain's order, one Adam step per chunk (rate 0.003, gradient
norm clipped at 1), the brain's exposure. ``offline``: the chunks replayed in shuffled epochs
(rate 0.001) until the epoch's loss stops falling (patience 3, at most ``epochs``). The held-out
stream is answered chunk by chunk from a zero state, greedily, and read by ``letters.score``.
The width is the one whose parameter count is nearest the brain's. Work is counted as frames,
updates, analytic multiply-accumulates (a backward pass at twice the forward) and seconds.

    python benchmarks/letters/controls.py --model lstm --loss ctc --mode online --seed 0 \\
        --protocol benchmarks/letters/protocol-step0.json --brain-parameters 21169 --out DIR
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("letters", _here / "letters.py")
assert _spec is not None and _spec.loader is not None
lt = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("letters", lt)
_spec.loader.exec_module(lt)

RATES = {"online": 0.003, "offline": 0.001}
CLIP, PATIENCE = 1.0, 3
MAX_LENGTH = 512  # frames the transformer's position table covers


def build(model: str, width: int, inputs: int, classes: int, seed: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class LSTM(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTM(inputs, width, batch_first=True)
            self.out = nn.Linear(width, classes)

        def forward(self, x: Any) -> Any:
            h, _ = self.cell(x)
            return self.out(h)

        def macs(self, length: int) -> int:
            return length * (4 * width * (inputs + width) + width * classes)

    class Transformer(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 2

        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(inputs, width)
            layer = nn.TransformerEncoderLayer(width, self.HEADS, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(width, classes)
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
            per_token = (inputs * width + self.LAYERS * 8 * width * width + width * classes)
            return length * per_token + self.LAYERS * width * length * (length + 1)

    return LSTM() if model == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, target: int, inputs: int, classes: int) -> int:
    """The width whose parameter count is nearest ``target`` (even, for the transformer)."""
    best: tuple[int, int] | None = None
    for width in range(4, 257, 2 if model == "transformer" else 1):
        n = count(build(model, width, inputs, classes, 0))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (width, n)
    assert best is not None
    return best[0]


def chunks(stream: dict[str, Any], words: int, letters: int) -> list[tuple[int, int, list[int]]]:
    """``(first frame, end frame, symbols)`` of each run of ``words`` words with their silences;
    the stream's opening silence joins the first chunk."""
    space = lt.space_symbol(letters)
    out, start, held, n = [], 0, [], 0
    for symbol, _, end in stream["sounds"]:
        held.append(symbol)
        if symbol == space:
            n += 1
            if n == words:
                out.append((start, end + 1, held))
                start, held, n = end + 1, [], 0
    if held:
        out.append((start, len(stream["track"]), held))
    return out


class Learner:
    def __init__(self, model: str, loss: str, width: int, inputs: int, classes: int,
                 seed: int, rate: float) -> None:
        import torch

        self.torch = torch
        self.loss = loss
        self.classes = classes
        self.net = build(model, width, inputs, classes, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "frames": 0, "macs": 0, "seconds": 0.0}

    def update(self, x: np.ndarray, target: list[int] | np.ndarray) -> float:
        torch = self.torch
        started = time.monotonic()
        logits = self.net(torch.from_numpy(x[None].astype(np.float32)))[0]
        if self.loss == "ctc":
            logp = torch.log_softmax(logits, -1)[:, None]
            loss = torch.nn.functional.ctc_loss(
                logp, torch.tensor([list(target)]), torch.tensor([len(x)]),
                torch.tensor([len(target)]), blank=0, zero_infinity=True)
        elif self.loss == "split":  # the symbol heard and, apart, whether to say it
            pair = torch.from_numpy(np.asarray(target, dtype=np.int64))
            width = self.classes - 2
            loss = (torch.nn.functional.cross_entropy(logits[:, :width], pair[:, 0])
                    + torch.nn.functional.cross_entropy(logits[:, width:], pair[:, 1]))
        else:
            loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(
                np.asarray(target, dtype=np.int64)))
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
        self.opt.step()
        self.work["updates"] += 1
        self.work["frames"] += len(x)
        self.work["macs"] += 3 * self.net.macs(len(x))
        self.work["seconds"] += time.monotonic() - started
        return float(loss.item())

    def answer(self, stream: dict[str, Any], words: int,
               letters: int) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        actions, preferred = [], []
        with torch.no_grad():
            for start, end, _ in chunks(stream, words, letters):
                logits = self.net(torch.from_numpy(
                    stream["frames"][start:end][None].astype(np.float32)))[0].numpy()
                if self.loss == "split":
                    width = self.classes - 2
                    heard = np.argmax(logits[:, :width], -1)
                    say = np.argmax(logits[:, width:], -1)
                    actions.append(np.where(say == 1, heard, lt.BLANK))
                    preferred.append(np.argmax(logits[:, 1:width], -1) + 1)
                else:
                    actions.append(np.argmax(logits, -1))
                    preferred.append(np.argmax(logits[:, 1:], -1) + 1)
        return np.concatenate(actions), np.concatenate(preferred)


def run(model: str, loss: str, mode: str, seed: int, protocol: dict[str, Any],
        brain_parameters: int, sizes: dict[str, int], epochs: int = 60,
        chunk_words: int = 8) -> dict[str, Any]:
    letters = int(protocol["world"]["letters"])
    inputs, classes = int(protocol["world"]["features"]), lt.symbols(letters)
    if loss == "split":
        classes += 2
    w = lt.world(seed, protocol, sizes)
    train, test = w["train"], w["test"]
    y = lt.labels(train, "onset")
    if loss == "split":  # the split brain's targets: the symbol heard, and whether to say it
        y = np.stack([lt.labels(train, "frames"), (y != lt.BLANK).astype(np.int64)], axis=1)
    width = matched_width(model, brain_parameters, inputs, classes)
    learner = Learner(model, loss, width, inputs, classes, seed, RATES[mode])
    pieces = chunks(train, chunk_words, letters)
    rng = lt.rng_for(seed, "learner")
    record: dict[str, Any] = {
        "arm": f"{model}-{loss}-{mode}", "seed": seed, "width": width,
        "parameters": count(learner.net), "probes": [],
    }

    def read(words_heard: int, epoch: int | None) -> None:
        actions, preferred = learner.answer(test, chunk_words, letters)
        result = lt.score(actions, test, letters, preferred=preferred)
        record["probes"].append({"words": words_heard, "epoch": epoch, "work": dict(learner.work),
                                 "score": result, "actions": actions.tolist(),
                                 "sample": lt.sample(actions, test, letters)})

    def one(index: int) -> float:
        start, end, symbols = pieces[index]
        target = symbols if loss == "ctc" else y[start:end]
        return learner.update(train["frames"][start:end], target)

    if mode == "online":
        for k in range(len(pieces)):
            one(k)
            heard = min((k + 1) * chunk_words, sizes["training_words"])
            if heard % sizes["probe_every"] == 0 or k == len(pieces) - 1:
                read(heard, None)
    else:
        best, waited = float("inf"), 0
        for epoch in range(epochs):
            total = sum(one(int(k)) for k in rng.permutation(len(pieces)))
            if total < best - 1e-4:
                best, waited = total, 0
            else:
                waited += 1
            if waited >= PATIENCE or epoch == epochs - 1:
                break
            if epoch % 10 == 9:
                read(sizes["training_words"] * (epoch + 1), epoch + 1)
        read(sizes["training_words"] * (epoch + 1), epoch + 1)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("lstm", "transformer"), required=True)
    parser.add_argument("--loss", choices=("ctc", "ce", "split"), required=True)
    parser.add_argument("--mode", choices=("online", "offline"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", type=Path, default=lt.PROTOCOL)
    parser.add_argument("--brain-parameters", type=int, required=True)
    parser.add_argument("--words", type=int)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(1)
    protocol = lt.load_protocol(args.protocol)
    sizes = {k: int(protocol[k]) for k in ("training_words", "probe_every", "evaluation_words")}
    if args.words:
        sizes["training_words"] = args.words
    record = run(args.model, args.loss, args.mode, args.seed, protocol, args.brain_parameters,
                 sizes, epochs=args.epochs)
    args.out.mkdir(parents=True, exist_ok=True)
    lt.write_record(args.out / f"{record['arm']}-{args.seed}.json.gz", record)
    s = record["probes"][-1]["score"]
    print(json.dumps({"arm": record["arm"], "seed": args.seed, "parameters": record["parameters"],
                      "chars": round(s["collapsed"]["characters"], 3),
                      "words": round(s["collapsed"]["words"], 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
