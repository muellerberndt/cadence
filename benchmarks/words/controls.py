"""The words chamber's matched controls: an LSTM and a causal transformer trained by gradient on
the same script as the brain (cadence#326, W7).

**What a control is given.** The brain holds the room on its map and knows where things are
around its body; a control is given the same as tokens. Every sequence begins with the room:
for each thing, ``THING``, its kind, its colour and its place around the robot as the robot faces
at that moment (concept tokens, the same for every founder). Then:

- a naming, as two sequences: the attended thing's concepts and the heard word (``NAME kind
  colour place SAY -> word END``), and the word and the thing's concepts (``NAME word SAY ->
  kind colour place END``);
- a question of practice or of the test: ``ASK`` and its words, ``SAY``, and the answer's words
  and ``END``.

The loss is the cross-entropy of the tokens after ``SAY``. Words are tokens, one a nonce form of
the founder; a control reads no letter.

**Modes.** ``online``: one pass over the script in the brain's order, one Adam step per naming
(its two sequences) or practice question, at rate 0.003, gradient norm clipped at 1.
``offline``: shuffled epochs of 32 sequences a step from rate 0.002, halved whenever the loss on
a tenth of the training sequences held aside has not fallen for 8 epochs (no lower than
0.00002), until that loss has not fallen for 25 epochs or 300 epochs are spent; the network kept
is that of the epoch whose held-aside loss was least. In both modes the test questions are
answered greedily (until ``END``, six words at most), the new words' three namings each are one
Adam step at the mode's rate, and the new words are then asked. The width is the one whose
parameter count is nearest the brain's. Work is counted as updates, analytic
multiply-accumulates (a backward pass at twice the forward) and seconds.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import world  # noqa: E402

RATES = {"online": 0.003, "offline": 0.002}
CLIP, PATIENCE, EPOCHS, BATCH = 1.0, 25, 300, 32
HALVE, FLOOR = 8, 2e-5
MAX_LENGTH = 64
MARKERS = ("PAD", "THING", "NAME", "ASK", "SAY", "END")


class Tokens:
    """A founder's vocabulary: the markers, the concepts and the founder's nonce words."""

    def __init__(self, lex: world.Lexicon):
        self.names = list(MARKERS)
        self.names += [f"kind{k}" for k in range(world.KINDS)]
        self.names += [f"colour{c}" for c in range(world.COLOURS)]
        self.names += [f"place{p}" for p in range(len(world.PLACE_NAMES))]
        self.words = (lex.kind + lex.colour + lex.place + list(lex.question.values())
                      + [lex.none] + lex.foils)
        self.names += self.words
        self.index = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def __call__(self, name: str) -> int:
        return self.index[name]


def room_tokens(tok: Tokens, r: world.Room) -> list[int]:
    out: list[int] = []
    for t in r.things:
        out += [tok("THING"), tok(f"kind{t.kind}"), tok(f"colour{t.colour}"),
                tok(f"place{world.place_of(r.heading, t.place)}")]
    return out


def sequences(seed: int, **given: Any) -> list[tuple[str, Any, Any]]:
    """The script as the controls' items, in the brain's order: ``("train", prompt, target)``
    for a naming's two sequences (one item) or a practice question, ``("ask", gate,
    (prompt, question))`` for a test question and ``("new", prompt, target)`` for a new word's
    naming."""
    import words

    lex = world.lexicon(seed)
    tok = Tokens(lex)
    out: list[tuple[str, Any, Any]] = []
    r: world.Room | None = None
    for event in words.script(seed, **given):
        kind = event[0]
        if kind == "room":
            r = event[1]
        elif kind == "turn":
            assert r is not None
            r.heading = event[1]
        elif kind == "name":
            assert r is not None
            place, word = event[1], event[2]
            t = min(r.things, key=lambda th: float(np.linalg.norm(th.place - place)))
            concepts = [tok(f"kind{t.kind}"), tok(f"colour{t.colour}"),
                        tok(f"place{world.place_of(r.heading, t.place)}")]
            scene = room_tokens(tok, r)
            pair = [(scene + [tok("NAME")] + concepts + [tok("SAY")], [tok(word), tok("END")]),
                    (scene + [tok("NAME"), tok(word), tok("SAY")], concepts + [tok("END")])]
            out.append(("train" if event[3] == "naming" else "new", pair, None))
        elif kind == "practice":
            assert r is not None
            q = event[1]
            prompt = room_tokens(tok, r) + [tok("ASK")] + [tok(w) for w in q.text.split()] + [
                tok("SAY")]
            out.append(("train", [(prompt, [tok(w) for w in q.answer] + [tok("END")])], None))
        elif kind == "ask":
            assert r is not None
            q = event[2]
            prompt = room_tokens(tok, r) + [tok("ASK")] + [tok(w) for w in q.text.split()] + [
                tok("SAY")]
            out.append(("ask", event[1], (prompt, q)))
    return out


def build(model: str, width: int, vocabulary: int, seed: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class LSTM(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocabulary, width)
            self.cell = nn.LSTM(width, width, batch_first=True)
            self.out = nn.Linear(width, vocabulary)

        def forward(self, x: Any) -> Any:
            h, _ = self.cell(self.embed(x))
            return self.out(h)

        def macs(self, length: int) -> int:
            return length * (4 * width * (width + width) + width * vocabulary)

    class Transformer(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 2

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocabulary, width)
            layer = nn.TransformerEncoderLayer(width, self.HEADS, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(width, vocabulary)
            position = torch.arange(MAX_LENGTH)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(MAX_LENGTH, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            h = self.embed(x) + self.position[:n]
            return self.out(self.body(h, mask=mask, is_causal=True))

        def macs(self, length: int) -> int:
            per_token = self.LAYERS * 8 * width * width + width * vocabulary
            return length * per_token + self.LAYERS * width * length * (length + 1)

    return LSTM() if model == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, target: int, vocabulary: int) -> int:
    """The width whose parameter count is nearest ``target`` (even, for the transformer)."""
    step = 2 if model == "transformer" else 1
    best = min(range(4, 257, step),
               key=lambda w: (abs(count(build(model, w, vocabulary, 0)) - target), w))
    return int(best)


class Control:
    def __init__(self, model: str, width: int, vocabulary: int, seed: int, rate: float):
        import torch

        self.torch = torch
        self.net = build(model, width, vocabulary, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "tokens": 0, "macs": 0, "seconds": 0.0}

    def batch(self, pairs: list[tuple[list[int], list[int]]]) -> tuple[Any, Any]:
        torch = self.torch
        rows = [p + t for p, t in pairs]
        n = max(len(r) for r in rows)
        x = np.zeros((len(rows), n), dtype=np.int64)
        y = np.full((len(rows), n), -100, dtype=np.int64)
        for i, (p, t) in enumerate(pairs):
            row = p + t
            x[i, : len(row)] = row
            # the token at position j is predicted from the positions up to j - 1
            for j, token in enumerate(t):
                y[i, len(p) - 1 + j] = token
        return torch.from_numpy(x), torch.from_numpy(y)

    def loss(self, pairs: list[tuple[list[int], list[int]]]) -> Any:
        x, y = self.batch(pairs)
        logits = self.net(x)
        return self.torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)

    def update(self, pairs: list[tuple[list[int], list[int]]]) -> float:
        torch = self.torch
        started = time.monotonic()
        loss = self.loss(pairs)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
        self.opt.step()
        length = max(len(p) + len(t) for p, t in pairs)
        self.work["updates"] += 1
        self.work["tokens"] += len(pairs) * length
        self.work["macs"] += 3 * len(pairs) * self.net.macs(length)
        self.work["seconds"] += time.monotonic() - started
        return float(loss.item())

    def say(self, prompt: list[int], end: int, most: int = 6) -> list[int]:
        torch = self.torch
        out: list[int] = []
        with torch.no_grad():
            seq = list(prompt)
            for _ in range(most):
                logits = self.net(torch.tensor([seq]))[0, -1]
                token = int(logits.argmax())
                self.work["macs"] += self.net.macs(len(seq))
                if token == end:
                    break
                out.append(token)
                seq.append(token)
        return out


def run(seed: int, model: str, mode: str, parameters: int, **given: Any) -> dict[str, Any]:
    """One control's life on the founder's script; the readings by gate, its parameters and
    work."""
    import torch

    torch.set_num_threads(1)
    lex = world.lexicon(seed)
    tok = Tokens(lex)
    items = sequences(seed, **given)
    width = matched_width(model, parameters, len(tok))
    control = Control(model, width, len(tok), seed, RATES[mode])
    began = time.monotonic()
    train = [pairs for kind, pairs, _ in items if kind == "train"]
    if mode == "online":
        for pairs in train:
            control.update(pairs)
    else:
        flat = [p for pairs in train for p in pairs]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 9]))
        order = rng.permutation(len(flat))
        aside = [flat[i] for i in order[: len(flat) // 10]]
        fit = [flat[i] for i in order[len(flat) // 10:]]
        best, best_state, since, since_halved, rate = math.inf, None, 0, 0, RATES[mode]
        for _ in range(EPOCHS):
            shuffled = rng.permutation(len(fit))
            for start in range(0, len(fit), BATCH):
                control.update([fit[int(i)] for i in shuffled[start: start + BATCH]])
            with torch.no_grad():
                held = float(np.mean([control.loss(aside[i: i + BATCH]).item()
                                      for i in range(0, len(aside), BATCH)]))
            if held < best - 1e-6:
                best, since, since_halved = held, 0, 0
                best_state = {k: v.clone() for k, v in control.net.state_dict().items()}
            else:
                since += 1
                since_halved += 1
                if since_halved >= HALVE and rate > FLOOR:
                    rate = max(rate / 2.0, FLOOR)
                    since_halved = 0
                    for group in control.opt.param_groups:
                        group["lr"] = rate
            if since >= PATIENCE:
                break
        if best_state is not None:
            control.net.load_state_dict(best_state)
    trained_at = time.monotonic()
    readings: dict[str, list[bool]] = {}
    for kind, first, second in items:
        if kind == "new":
            control.update(first)  # a new word's naming: its two sequences, one step
        elif kind == "ask":
            gate, (prompt, q) = first, second
            said = [tok.names[t] for t in control.say(prompt, tok("END"))]
            ok = world.right(q, said)
            readings.setdefault(gate, []).append(ok)
            if gate.startswith("W1") and q.about == "out_of_view":
                readings.setdefault("W4_out_of_view", []).append(ok)
    import words

    out = words.readings_of(seed, {}, readings, [], None,
                            {"train": trained_at - began, "test": time.monotonic() - trained_at})
    out.update({"model": model, "mode": mode, "width": width, "parameters": count(control.net),
                "updates": control.work["updates"], "macs": control.work["macs"]})
    return out


if __name__ == "__main__":
    import json

    seed, model, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    told = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    print(json.dumps(run(seed, model, mode, 6656, **told), indent=1, default=float))
