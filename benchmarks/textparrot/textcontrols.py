"""The text parrot's controls, each driven by the chamber exactly as the brain is.

- ``lstm`` and ``transformer``: networks trained by gradient on the same moments (the same input
  vector a moment, the same voice and gaze labels), teacher forced. ``online``: one pass over the
  brain's curriculum (the same stages, forms and blocks as the brain's life), one Adam step a
  block of 64 exchanges at rate 0.003. ``offline``: shuffled epochs over a fixed set of exchanges
  of every form (as many as the brain's life saw, at most 20,000), 32 exchanges a step from rate
  0.002, halved when the loss on a tenth held aside has not fallen for 8 epochs, until it has
  not fallen for 25 epochs or 200 epochs are spent; the network kept is the best epoch's. The
  width is the one whose parameter count is nearest the brain's.
- ``readout``: one linear layer (two softmax heads) from what the brain's memories hold: the
  present moment, the moment before (the brain's heard history) and the question's words (the
  brain's held message). No hidden layer: it has no conjunction of a question word and a glimpse.
- ``table``: a lookup of every situation the pretraining showed, by its symbols (the question's
  words, the thing in view and the focus by kind, colour and place, the word heard, the words
  said at this thing); a situation it never saw gets silence.

Work is counted as updates, analytic multiply-accumulates (a backward pass at twice the forward)
and seconds.
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

import textchamber as tc  # noqa: E402
import textworld as tw  # noqa: E402

MAX_LENGTH = 160


def build(model: str, width: int, inputs: int, voice: int, seed: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class Heads(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.voice = nn.Linear(width, voice)
            self.gaze = nn.Linear(width, 2)

    class LSTM(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(inputs, width)
            self.cell = nn.LSTM(width, width, batch_first=True)
            self.heads = Heads()

        def forward(self, x: Any) -> tuple[Any, Any]:
            h, _ = self.cell(self.embed(x))
            return self.heads.voice(h), self.heads.gaze(h)

        def macs(self, length: int) -> int:
            return length * (inputs * width + 8 * width * width + width * (voice + 2))

    class Transformer(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 4

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(inputs, width)
            layer = nn.TransformerEncoderLayer(width, self.HEADS, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.heads = Heads()
            position = torch.arange(MAX_LENGTH)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(MAX_LENGTH, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def forward(self, x: Any) -> tuple[Any, Any]:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            h = self.body(self.embed(x) + self.position[:n], mask=mask, is_causal=True)
            return self.heads.voice(h), self.heads.gaze(h)

        def macs(self, length: int) -> int:
            per = inputs * width + self.LAYERS * 8 * width * width + width * (voice + 2)
            return length * per + self.LAYERS * width * length * (length + 1)

    return LSTM() if model == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, target: int, inputs: int, voice: int) -> int:
    step = 4 if model == "transformer" else 1
    return int(min(range(8, 400, step),
                   key=lambda w: (abs(count(build(model, w, inputs, voice, 0)) - target), w)))


def sequences(eps: list[tc.Episode], layout: tc.Layout) -> tuple[np.ndarray, ...]:
    """Teacher-forced moments of finished teacher episodes, replayed: (x, voice, gaze, mask),
    each (exchanges, moments, ...)."""
    rows = []
    for e in eps:
        replay = tc.Episode(layout, e.scene, e.q, "teacher")
        xs, vs, gs = [], [], []
        while not replay.done and len(xs) < MAX_LENGTH:
            xs.append(replay.observe())
            w, g = replay.expert()
            vs.append(layout.act_of(w))
            gs.append(g)
            replay.advance(w, g)
        rows.append((np.array(xs), np.array(vs), np.array(gs)))
    t = max(len(r[0]) for r in rows)
    x = np.zeros((len(rows), t, layout.inputs), np.float32)
    v = np.zeros((len(rows), t), np.int64)
    g = np.zeros((len(rows), t), np.int64)
    m = np.zeros((len(rows), t), np.float32)
    for i, (a, b, c) in enumerate(rows):
        x[i, :len(a)], v[i, :len(a)], g[i, :len(a)], m[i, :len(a)] = a, b, c, 1.0
    return x, v, g, m


class NetPolicy:
    """An LSTM or transformer as the chamber's learner (frozen reads only; it trains on
    sequences)."""

    def __init__(self, model: str, layout: tc.Layout, parameters: int, seed: int, rate: float):
        import torch

        self.torch = torch
        self.model, self.layout = model, layout
        self.width = matched_width(model, parameters, layout.inputs, layout.voice)
        self.net = build(model, self.width, layout.inputs, layout.voice, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.updates = 0
        self.macs = 0
        self.history: list[np.ndarray] = []

    def parameters(self) -> int:
        return count(self.net)

    def loss(self, x: np.ndarray, v: np.ndarray, g: np.ndarray, m: np.ndarray) -> Any:
        torch = self.torch
        xv, gv = self.net(torch.from_numpy(x))
        mt = torch.from_numpy(m)
        lv = torch.nn.functional.cross_entropy(xv.reshape(-1, xv.shape[-1]),
                                               torch.from_numpy(v).reshape(-1), reduction="none")
        lg = torch.nn.functional.cross_entropy(gv.reshape(-1, 2), torch.from_numpy(g).reshape(-1),
                                               reduction="none")
        return ((lv + lg) * mt.reshape(-1)).sum() / mt.sum()

    def update(self, x: np.ndarray, v: np.ndarray, g: np.ndarray, m: np.ndarray) -> float:
        self.net.train()
        self.opt.zero_grad()
        loss = self.loss(x, v, g, m)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.updates += 1
        self.macs += 3 * len(x) * self.net.macs(x.shape[1])  # a backward pass at twice the forward
        return float(loss)

    # the chamber's free-running read
    def begin(self, n: int) -> None:
        self.history = []

    def moment(self, x: np.ndarray, labels: Any, mask: np.ndarray, learn: bool) -> np.ndarray:
        torch = self.torch
        self.history.append(x.astype(np.float32))
        seq = np.stack(self.history, axis=1)[:, -MAX_LENGTH:]
        self.net.eval()
        with torch.no_grad():
            v, g = self.net(torch.from_numpy(seq))
        return np.stack([v[:, -1].argmax(-1).numpy(), g[:, -1].argmax(-1).numpy()], axis=1)


class ReadoutPolicy:
    """One linear layer from the present moment, the moment before and the question's words."""

    def __init__(self, layout: tc.Layout, seed: int, rate: float = 0.01):
        self.layout = layout
        d = 3 * layout.inputs
        rng = np.random.default_rng(seed)
        self.wv = rng.normal(0, 0.01, (d, layout.voice))
        self.wg = rng.normal(0, 0.01, (d, 2))
        self.bv, self.bg = np.zeros(layout.voice), np.zeros(2)
        self.rate = rate
        self.prev: np.ndarray | None = None
        self.held: np.ndarray | None = None
        self.closed: np.ndarray | None = None
        self.updates = 0
        self.macs = 0

    def parameters(self) -> int:
        return int(self.wv.size + self.wg.size + self.bv.size + self.bg.size)

    def begin(self, n: int) -> None:
        self.prev = np.zeros((n, self.layout.inputs))
        self.held = np.zeros((n, self.layout.inputs))
        self.closed = np.zeros(n, bool)

    def features(self, x: np.ndarray) -> np.ndarray:
        assert self.prev is not None and self.held is not None and self.closed is not None
        self.closed |= x[:, self.layout.gap] > 0
        word = np.zeros_like(x)
        word[:, :self.layout.words] = x[:, :self.layout.words]
        self.held = np.where(self.closed[:, None], self.held, np.maximum(self.held, word))
        return np.concatenate([x, self.prev, self.held], axis=1)

    def moment(self, x: np.ndarray, labels: Any, mask: np.ndarray, learn: bool) -> np.ndarray:
        f = self.features(x)
        zv, zg = f @ self.wv + self.bv, f @ self.wg + self.bg
        self.macs += len(x) * f.shape[1] * (self.layout.voice + 2)
        if learn and labels is not None:
            for z, w, b, y in ((zv, self.wv, self.bv, labels[:, 0]),
                               (zg, self.wg, self.bg, labels[:, 1])):
                p = np.exp(z - z.max(1, keepdims=True))
                p /= p.sum(1, keepdims=True)
                p[np.arange(len(y)), y] -= 1.0
                p *= mask[:, None]
                w -= self.rate * f.T @ p / max(1, mask.sum())
                b -= self.rate * p.sum(0) / max(1, mask.sum())
            self.macs += 2 * len(x) * f.shape[1] * (self.layout.voice + 2)
            self.updates += 1
        self.prev = x.copy()
        return np.stack([zv.argmax(1), zg.argmax(1)], axis=1)


def _thing(t: tw.Thing | None) -> tuple[str, ...]:
    return () if t is None else (t.kind, t.colour) + t.place


class TablePolicy:
    """Every situation the pretraining showed, by its symbols; silence where it saw none."""

    keyed = True

    def __init__(self, layout: tc.Layout):
        self.layout = layout
        self.table: dict[tuple, tuple[int, int]] = {}

    def parameters(self) -> int:
        return 2 * len(self.table)

    @staticmethod
    def key(e: tc.Episode) -> tuple:
        return (tuple(e.q.words), e.phase, e.i if e.phase == "question" else -1,
                _thing(e.thing()), _thing(e.focus), e.heard, tuple(e.at))

    def begin(self, n: int) -> None:
        pass

    def moment_keys(self, keys: list[tuple], labels: Any, mask: np.ndarray,
                    learn: bool) -> np.ndarray:
        out = np.zeros((len(keys), 2), dtype=np.int64)
        for k, key in enumerate(keys):
            if learn and labels is not None and mask[k]:
                self.table[key] = (int(labels[k, 0]), int(labels[k, 1]))
            out[k] = self.table.get(key, (self.layout.voice - 1, tc.PASS))
        return out


def online(model: str, layout: tc.Layout, parameters: int, seed: int,
           plan: list[tuple[str, tuple[str, ...], int]], streams: int = 64) -> NetPolicy:
    """One pass over ``plan`` (stage, forms, blocks): one Adam step a block."""
    import textparrot as tp

    net = NetPolicy(model, layout, parameters, seed, 0.003)
    rng = np.random.default_rng([seed, 11])
    seen: tuple[str, ...] = ()
    for stage, forms, blocks in plan:
        seen = tuple(dict.fromkeys(seen + forms))
        for _ in range(blocks):
            use = forms if rng.random() > 0.3 else seen
            items = tp.batch(rng, use, streams, stage=stage)
            eps = [tc.Episode(layout, s, q, "teacher") for s, q in items]
            net.update(*sequences(eps, layout))
    return net


def offline(model: str, layout: tc.Layout, parameters: int, seed: int, exchanges: int,
            *, epochs: int = 200, patience: int = 25, halve: int = 8) -> NetPolicy:
    """Epochs over a fixed set of ``exchanges`` exchanges of every form."""
    import textparrot as tp

    net = NetPolicy(model, layout, parameters, seed, 0.002)
    rng = np.random.default_rng([seed, 12])
    items = [tp.draw(str(rng.choice(tp.ALL_FORMS)), rng) for _ in range(exchanges)]
    eps = [tc.Episode(layout, s, q, "teacher") for s, q in items]
    cut = max(1, len(eps) // 10)
    held, train = eps[:cut], eps[cut:]
    hx = sequences(held, layout)
    best, best_state, since, since_halve = math.inf, None, 0, 0
    torch = net.torch
    for _ in range(epochs):
        order = rng.permutation(len(train))
        for i in range(0, len(order), 32):
            net.update(*sequences([train[j] for j in order[i:i + 32]], layout))
        with torch.no_grad():
            net.net.eval()
            loss = float(net.loss(*hx))
        if loss < best - 1e-4:
            best, since, since_halve = loss, 0, 0
            best_state = {k: v.clone() for k, v in net.net.state_dict().items()}
        else:
            since += 1
            since_halve += 1
            if since_halve >= halve:
                for group in net.opt.param_groups:
                    group["lr"] = max(2e-5, group["lr"] / 2)
                since_halve = 0
            if since >= patience:
                break
    if best_state is not None:
        net.net.load_state_dict(best_state)
    return net


def plain(kind: str, layout: tc.Layout, seed: int,
          plan: list[tuple[str, tuple[str, ...], int]], streams: int = 64,
          epochs: int = 1) -> Any:
    """The readout or the table, taught on the brain's curriculum by teacher forcing."""
    import textparrot as tp

    policy: Any = ReadoutPolicy(layout, seed) if kind == "readout" else TablePolicy(layout)
    rng = np.random.default_rng([seed, 13])
    seen: tuple[str, ...] = ()
    for _ in range(epochs):
        for stage, forms, blocks in plan:
            seen = tuple(dict.fromkeys(seen + forms))
            for _ in range(blocks):
                use = forms if rng.random() > 0.3 else seen
                tc.run(policy, layout, tp.batch(rng, use, streams, stage=stage), "teacher")
    return policy


def timed(fn: Any, *args: Any, **kw: Any) -> tuple[Any, float]:
    t = time.time()
    out = fn(*args, **kw)
    return out, time.time() - t
