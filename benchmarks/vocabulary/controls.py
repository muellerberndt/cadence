"""Next-word learners at matched information, at any vocabulary: the chamber's controls.

The grammar chamber's LSTM and causal transformer (``benchmarks/grammar/controls.py``) over a
``Language``: they hear the brain's moments and are trained to predict each word, teacher-forced,
by cross-entropy, and are read on the brain's panels by the same scorer, closed loop, greedy.
Their size is the one whose parameter count is nearest the brain's ``parameters()``.

- one pass (``online``): one Adam step per batch the brain is taught, rate 0.003, the brain's
  exposure exactly; a reading at each of the brain's checkpoints.
- replay (``--seconds``): after the pass, the same stored batches in shuffled epochs at rate 0.001
  until the given seconds of training are spent, the brain's seconds for the same batches: equal
  compute. A reading at the end, and at each doubling of the epochs.

``--tied`` shares the embedding table between the input and the output for the words (the
output of a word is the dot product with its input row), the transformer's usual form for a
language model. ``--pretrained`` (with the tied form, a labelled control) fixes the embedding of
every word and concept to a table made before the life: a stem's words and concept share a
vector, each class a vector of its own, so a word no sentence used already sits with its class
and its concept. It is what a pretrained embedding would hand a network.

They are independently scored controls and never part of the brain's answer path.

    python benchmarks/vocabulary/controls.py --model transformer --words 200 --founder 0 \\
        --parameters 224311 --checkpoints 500 2000 --seconds 3000 --out DIR/controls.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from language import EVERYTHING, LEAN, SEEDS, TWINS, Language, Row, by_size
from pieces import PieceLanguage

RATES = {"online": 0.003, "offline": 0.001}
CLIP = 1.0
MAX_LENGTH = 128


def encode(lang: Language, rows: list[Row]) -> tuple[np.ndarray, np.ndarray]:
    """The moments the brain hears and the words it is taught, teacher-forced: ``(batch, moments,
    inputs)`` and ``(batch, moments)``, with -100 where nothing is taught."""
    xs, ys = [], []
    for r in rows:
        moments = [*r.pre, (lang.go,)] + [lang.spoken(w) for w in r.target[:-1]]
        xs.append(lang.dense(moments))
        ys.append([-100] * len(r.pre) + list(r.target))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def build(model: str, inputs: int, actions: int, width: int, seed: int, tied: bool = False) -> Any:
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
            self.tied = tied
            self.out = None if tied else nn.Linear(width, actions)
            self.bias = nn.Parameter(torch.zeros(actions)) if tied else None
            position = torch.arange(MAX_LENGTH)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(MAX_LENGTH, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            h = self.body(self.inp(x) + self.position[:n], mask=mask, is_causal=True)
            if self.tied:
                # an action's output row is its word's input row (the first coordinates)
                return h @ self.inp.weight[:, :actions] + self.bias
            return self.out(h)

        def macs(self, length: int) -> int:
            per_token = (inputs * width + self.LAYERS * 8 * width * width + width * actions)
            attention = self.LAYERS * width * length * (length + 1)
            return length * per_token + attention

    return LSTM() if model == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, inputs: int, actions: int, target: int, tied: bool = False) -> int:
    """The width whose parameter count is nearest ``target`` (even, for the transformer), by a
    closed-form search with the count of a built network at the end."""

    def params(width: int) -> int:
        if model == "lstm":
            return 4 * (inputs + width + 1) * width + (width + 1) * actions
        layer = 8 * width * width + 4 * width + 2 * width * 2 + 2 * width + width * 2 + 2 * width
        io = (inputs + 1) * width + (actions if tied else (width + 1) * actions)
        return 2 * layer + io

    best = min(range(4, 4098, 2), key=lambda w: abs(params(w) - target))
    return best


def pretrained_table(lang: Language, width: int, seed: int) -> tuple[list[int], np.ndarray]:
    """The fixed columns of a pretrained control over the word language: which coordinates, and
    each one's vector, the stem's own vector and its class's, at unit length."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 266, 3]))

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)

    noun, verb, number = (unit(rng.normal(size=width)) for _ in range(3))
    coordinates: list[int] = []
    vectors: list[np.ndarray] = []
    for i in range(lang.nouns):
        stem = unit(rng.normal(size=width))
        for coordinate, vector in ((lang.noun(i, 0), stem + noun),
                                   (lang.noun(i, 1), stem + noun + number),
                                   (lang.noun_concept0 + i, stem + noun)):
            coordinates.append(coordinate)
            vectors.append(unit(vector))
    for v in range(lang.verbs):
        stem = unit(rng.normal(size=width))
        for coordinate, vector in ((lang.verb(v, 0), stem + verb),
                                   (lang.verb(v, 1), stem + verb + number),
                                   (lang.verb_concept0 + v, stem + verb)):
            coordinates.append(coordinate)
            vectors.append(unit(vector))
    return coordinates, np.stack(vectors, axis=1)


class Learner:
    def __init__(self, lang: Language, model: str, width: int, seed: int, rate: float,
                 tied: bool = False, pretrained: bool = False) -> None:
        self.lang = lang
        self.net = build(model, lang.inputs, lang.actions, width, seed, tied)
        self.opt = self.torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "tokens": 0, "macs": 0, "seconds": 0.0}
        self.fixed: tuple[list[int], Any] | None = None
        if pretrained:
            coordinates, table = pretrained_table(lang, width, seed)
            self.fixed = (coordinates, self.torch.from_numpy(table.astype(np.float32)))
            self._hold()

    def _hold(self) -> None:
        if self.fixed is not None:
            with self.torch.no_grad():
                self.net.inp.weight[:, self.fixed[0]] = self.fixed[1]

    @property
    def torch(self) -> Any:
        import torch

        return torch

    def set_rate(self, rate: float) -> None:
        for group in self.opt.param_groups:
            group["lr"] = rate

    def update(self, rows: list[Row]) -> float:
        torch = self.torch
        started = time.monotonic()
        x, y = encode(self.lang, rows)
        logits = self.net(torch.from_numpy(x))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.lang.actions),
                                                 torch.from_numpy(y).reshape(-1),
                                                 ignore_index=-100)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
        self.opt.step()
        self._hold()
        self.work["updates"] += 1
        self.work["tokens"] += x.shape[0] * x.shape[1]
        self.work["macs"] += 3 * x.shape[0] * self.net.macs(x.shape[1])
        self.work["seconds"] += time.monotonic() - started
        return float(loss.item())

    def speak(self, rows: list[Row], cap: int = 4) -> list[list[int]]:
        """Closed loop, greedy: each row hears its own previous word after its ``force``."""
        torch, lang = self.torch, self.lang
        n = len(rows)
        said: list[list[int]] = [[] for _ in rows]
        done = [False] * n
        moments: list[list[tuple[int, ...]]] = [[*r.pre, (lang.go,)] for r in rows]
        longest = max(len(r.target) for r in rows) + cap
        with torch.no_grad():
            for j in range(longest):
                dense = np.stack([lang.dense(m) for m in moments]).astype(np.float32)
                action = self.net(torch.from_numpy(dense))[:, -1].argmax(dim=1).numpy()
                self.work["macs"] += n * self.net.macs(dense.shape[1])
                for i, r in enumerate(rows):
                    said[i].append(int(action[i]))
                    if j >= r.force and action[i] == lang.end:
                        done[i] = True
                    given = j + 1 <= r.force and j < len(r.target)
                    moments[i].append(lang.spoken(r.target[j] if given else int(action[i])))
                if all(done):
                    break
        out = []
        for i, r in enumerate(rows):
            words = said[i][r.force:]
            out.append(words[:words.index(lang.end) + 1] if lang.end in words else words)
        return out


def reading(learner: Learner, tests: dict[str, list[list[Row]]],
            names: tuple[str, ...]) -> dict[str, Any]:
    learner.net.eval()
    out = {name: learner.lang.read_speaker(learner.speak, None, tests[name], twins=name in TWINS)
           for name in names}
    learner.net.train()
    return out


def run_life(spec: dict[str, Any]) -> None:
    import torch

    torch.set_num_threads(1)
    if spec.get("lexicon"):
        lang: Any = PieceLanguage(spec["lexicon"], str(spec.get("route", "heard")),
                                  bool(spec.get("swap")), None, bool(spec.get("start")),
                                  float(spec.get("pseudo", 0.0)), bool(spec.get("verb_tokens")))
    else:
        nouns, verbs = (int(spec["nouns"]), int(spec["verbs"])) if "nouns" in spec else by_size(
            int(spec["words"]))
        lang = Language(nouns, verbs, int(spec.get("fresh", 0)), bool(spec.get("verb_flag")))
    model, founder = spec["model"], int(spec["founder"])
    tied = bool(spec.get("tied"))
    width = int(spec.get("width") or matched_width(model, lang.inputs, lang.actions,
                                                   int(spec["parameters"]), tied))
    learner = Learner(lang, model, width, founder, RATES["online"], tied)
    split = lang.split(founder)
    tests = lang.panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, SEEDS["train"]]))
    full = {int(c) for c in spec.get("full", [])}
    sink = Path(spec["out"])
    sink.parent.mkdir(parents=True, exist_ok=True)
    head = {"model": model, "tied": tied, "founder": founder, "words": lang.words,
            "fresh": getattr(lang, "fresh", 0), "width": width, "parameters": count(learner.net),
            "brain_parameters": int(spec.get("parameters", 0)), "torch": torch.__version__,
            "tag": spec.get("tag", "")}

    def write(mode: str, batches: int, epochs: float, names: tuple[str, ...]) -> None:
        begun = time.monotonic()
        row = {**head, "mode": mode, "batches": batches, "epochs": round(epochs, 2),
               "work": dict(learner.work), "panels": reading(learner, tests, names),
               "reading_seconds": round(time.monotonic() - begun, 1),
               "seconds_per_batch": round(learner.work["seconds"]
                                          / max(1, learner.work["updates"]), 5)}
        with sink.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def panel_names(everything: bool) -> tuple[str, ...]:
        if isinstance(lang, PieceLanguage):
            return tuple(tests)
        return EVERYTHING if everything else LEAN

    stored: list[list[Row]] = []
    batch = 0
    for checkpoint in sorted({int(c) for c in spec.get("checkpoints", [2000])}):
        while batch < checkpoint:
            rows = lang.train_batch(rng, split)
            if spec.get("seconds"):
                stored.append(rows)
            learner.update(rows)
            batch += 1
        write("online", batch, 1.0, panel_names(checkpoint in full))
    if spec.get("seconds"):
        limit = float(spec["seconds"])
        learner.set_rate(RATES["offline"])
        order = np.random.default_rng(np.random.SeedSequence([founder, 99]))
        epochs, next_read = 1.0, 2.0
        while learner.work["seconds"] < limit:
            for i in order.permutation(len(stored)):
                learner.update(stored[int(i)])
                if learner.work["seconds"] >= limit:
                    break
            epochs += 1
            if epochs >= next_read:
                write("replay", batch, epochs, panel_names(False))
                next_read *= 2
        write("replay_end", batch, epochs, panel_names(True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default="transformer", choices=("lstm", "transformer"))
    parser.add_argument("--words", type=int, default=200)
    parser.add_argument("--fresh", type=int, default=0)
    parser.add_argument("--lexicon", default="")
    parser.add_argument("--verb-flag", action="store_true")
    parser.add_argument("--route", default="heard", choices=("heard", "concept"))
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--pseudo", type=float, default=0.0)
    parser.add_argument("--verb-tokens", action="store_true")
    parser.add_argument("--founder", type=int, default=0)
    parser.add_argument("--parameters", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--tied", action="store_true")
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[2000])
    parser.add_argument("--full", nargs="*", type=int, default=[])
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--panel-batches", type=int, default=4)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_life({"model": args.model, "words": args.words, "fresh": args.fresh,
              "lexicon": args.lexicon, "route": args.route, "swap": args.swap,
              "start": args.start, "pseudo": args.pseudo, "verb_tokens": args.verb_tokens,
              "verb_flag": args.verb_flag,
              "founder": args.founder, "parameters": args.parameters, "width": args.width,
              "tied": args.tied, "checkpoints": args.checkpoints, "full": args.full,
              "seconds": args.seconds, "panel_batches": args.panel_batches, "tag": args.tag,
              "out": str(args.out)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
