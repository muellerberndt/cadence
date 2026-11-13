"""The hearing chamber's matched controls (cadence#329, H5): a keyword-spotting network and a
transformer in the ear's place, trained by gradient on the same sounds and the same lessons.

The parrot's map, body, lexicon and written answers stay the brain's (``cadence.words.Words``):
a control replaces the ear and its records alone. A control reads a sound's log-mel frames
(each band less its mean over the sound, over its spread) and gives a vector over the brain's
concepts, what the sound brings back; the brain asks its map with it as with the ear's
(``Hearing.question``). Every lesson the ear's records take, the control takes as a target:
the thing in focus at a naming, what a question's answer asks for and points to, the concepts
a word read aloud names (mean squared error).

- **keyword spotting (``kws``):** a one-dimensional convolution over the frames, three strided
  convolutions, the mean and the maximum over time, a linear map to the concepts;
- **transformer (``transformer``):** a strided convolution to a token every 40 ms, positions,
  two encoder layers, the mean over tokens, a linear map to the concepts.

Modes: ``online``, one Adam step a lesson in the life's order (rate 0.001); ``offline``, the
lessons kept, then shuffled epochs of 32 (rate 0.001, halved after 8 epochs without a better
loss on a tenth held aside, stopped after 20, at most 200), the network of the best epoch kept,
before the first test question. The width is the one whose parameter count is nearest the
brain's learned parameters. Work: Adam steps, analytic multiply-accumulates (a backward pass at
twice the forward) and seconds.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from cadence.speech.frontend import MELS, log_mel, mel_bank

RATE, BATCH, EPOCHS, PATIENCE, HALVE = 1e-3, 32, 200, 20, 8


def build(model: str, width: int, concepts: int, seed: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class KWS(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Conv1d(MELS, width, 5, padding=2)
            self.body = nn.ModuleList(
                [nn.Conv1d(width, width, 5, stride=2, padding=2) for _ in range(3)])
            self.out = nn.Linear(2 * width, concepts)

        def forward(self, x: Any) -> Any:  # x: (batch, frames, MELS)
            h = torch.relu(self.first(x.transpose(1, 2)))
            for layer in self.body:
                h = torch.relu(layer(h))
            return self.out(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))

        def macs(self, frames: int) -> int:
            out = frames * MELS * width * 5
            n = frames
            for _ in range(3):
                n = (n + 1) // 2
                out += n * width * width * 5
            return out + 2 * width * concepts

    class Transformer(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 4

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Conv1d(MELS, width, 4, stride=4)
            layer = nn.TransformerEncoderLayer(width, self.HEADS, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(width, concepts)

        def forward(self, x: Any) -> Any:
            h = self.embed(x.transpose(1, 2)).transpose(1, 2)
            n = h.shape[1]
            position = torch.arange(n)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(n, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            return self.out(self.body(h + table).mean(dim=1))

        def macs(self, frames: int) -> int:
            n = frames // 4
            return n * (MELS * 4 * width + self.LAYERS * 8 * width * width) + \
                self.LAYERS * width * n * n + width * concepts

    return KWS() if model == "kws" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, target: int, concepts: int) -> int:
    """The width whose parameter count is nearest ``target`` (a multiple of 4, for the
    transformer's heads)."""
    step = 4 if model == "transformer" else 1
    sizes = range(8, 1025, step)
    return int(min(sizes, key=lambda w: (abs(count(build(model, w, concepts, 0)) - target), w)))


class NetEar:
    """A control in the ear's place: ``listen`` keeps a sound's frames, ``teach`` takes a
    lesson for them, ``recall`` gives the network's concepts for them."""

    def __init__(self, model: str, mode: str, width: int, concepts: int, seed: int):
        import torch

        torch.set_num_threads(1)
        self.torch, self.mode, self.concepts = torch, mode, int(concepts)
        self.net = build(model, width, concepts, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=RATE)
        self.bank = mel_bank(MELS)
        self.sounds: list[np.ndarray] = []
        self.lessons: list[tuple[int, np.ndarray]] = []
        self.fitted = False
        self.records = 0
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 329, 11]))
        self.work = {"updates": 0, "macs": 0, "seconds": 0.0}
        self.last: dict[str, Any] = {}

    def features(self, samples: Any) -> np.ndarray:
        frames = log_mel(np.asarray(samples, dtype=np.float64), bank=self.bank)
        frames = frames - frames.mean(axis=0, keepdims=True)
        return np.asarray(frames / np.maximum(frames.std(axis=0, keepdims=True), 1e-3),
                          dtype=np.float32)

    def hear(self, samples: Any, *, learn: bool = False) -> None:
        return None

    def listen(self, samples: Any, *, learn: bool) -> tuple[np.ndarray, list[int]]:
        self.sounds.append(self.features(samples))
        return np.zeros((1, 1)), [len(self.sounds) - 1]

    def recall(self, codes: np.ndarray, *, credit: bool = True) -> np.ndarray:
        torch = self.torch
        if not self.fitted and self.mode == "offline":
            self.fit()
        x = torch.from_numpy(self.sounds[-1][None])
        with torch.no_grad():
            out = self.net(x)[0].numpy()
        self.work["macs"] += self.net.macs(x.shape[1])
        return np.asarray(out, dtype=np.float64)

    def teach(self, records: list[int], concepts: np.ndarray) -> None:
        for r in records:
            self.lessons.append((r, np.asarray(concepts, dtype=np.float32)))
            if self.mode == "online":
                self._step([(r, self.lessons[-1][1])])

    def _loss(self, batch: list[tuple[int, np.ndarray]]) -> Any:
        torch = self.torch
        frames = [self.sounds[r] for r, _ in batch]
        longest = max(len(f) for f in frames)
        x = np.zeros((len(frames), longest, MELS), dtype=np.float32)
        for i, f in enumerate(frames):  # a sound shorter than the batch's longest repeats its end
            x[i, : len(f)] = f
            x[i, len(f):] = f[-1]
        y = np.stack([t for _, t in batch])
        out = self.net(torch.from_numpy(x))
        self.work["macs"] += 3 * len(frames) * self.net.macs(longest)
        return ((out - torch.from_numpy(y)) ** 2).mean()

    def _step(self, batch: list[tuple[int, np.ndarray]]) -> float:
        began = time.monotonic()
        loss = self._loss(batch)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.work["updates"] += 1
        self.work["seconds"] += time.monotonic() - began
        return float(loss.item())

    def fit(self) -> None:
        """Offline: shuffled epochs over the lessons kept, the best epoch's network kept."""
        torch = self.torch
        self.fitted = True
        if not self.lessons:
            return
        order = self.rng.permutation(len(self.lessons))
        aside = [self.lessons[i] for i in order[: max(len(order) // 10, 1)]]
        fit = [self.lessons[i] for i in order[max(len(order) // 10, 1):]] or aside
        best, kept, since, halved, rate = math.inf, None, 0, 0, RATE
        for _ in range(EPOCHS):
            shuffled = self.rng.permutation(len(fit))
            for start in range(0, len(fit), BATCH):
                self._step([fit[int(i)] for i in shuffled[start: start + BATCH]])
            with torch.no_grad():
                held = float(np.mean([self._loss(aside[i: i + BATCH]).item()
                                      for i in range(0, len(aside), BATCH)]))
            if held < best - 1e-6:
                best, since, halved = held, 0, 0
                kept = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                since, halved = since + 1, halved + 1
                if halved >= HALVE:
                    rate, halved = rate / 2.0, 0
                    for group in self.opt.param_groups:
                        group["lr"] = rate
            if since >= PATIENCE:
                break
        if kept is not None:
            self.net.load_state_dict(kept)

    def choose(self, concepts: np.ndarray, *, top: int = 32) -> tuple[int, float]:
        return -1, 0.0

    def parameters(self) -> int:
        return count(self.net)


def make(seed: int, model: str, mode: str, parameters: int) -> Any:
    """The founder's brain with a control in the ear's place."""
    import words

    from cadence.words.hearing import Hearing

    brain = words.make_brain(seed)
    hearing = Hearing(brain)
    concepts = brain.lexicon.concepts
    width = matched_width(model, parameters, concepts)
    hearing.ear = NetEar(model, mode, width, concepts, seed)  # type: ignore[assignment]
    hearing.voice.ear = hearing.ear  # type: ignore[assignment]
    return hearing
