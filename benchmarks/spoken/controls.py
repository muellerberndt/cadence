"""The matched baselines of the spoken-phrase chamber (cadence#303): a small causal transformer
and an LSTM over the steps of a phrase, each reading every step's window through two
convolutions, trained by backpropagation on the same windows, the same phrases and the same
order as a brain's life, at a parameter count matched to the brain's.

Front end: the step's window as the brain hears it (40 bands by 98 frames), 16 filters of 5 by 5
at stride 2, then 32 of 5 by 5 at stride 2, each rectified, and a linear map to the model width.
The transformer attends causally over the phrase's steps (two layers, four heads, a
feed-forward of four times the width, learned positions); the LSTM carries one layer.

Two losses: ``step`` (the brain's per-step targets by cross-entropy: matched information) and
``ctc`` (connectionist temporal classification over the phrase's words and ``end``, with no
alignment). ``online`` is one Adam step per batch the brain is taught, in its order, once.
Corrections are one Adam step per phrase. It needs PyTorch.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

BANDS, FRAMES = 40, 98
LAYERS = 2
HEADS = 4
MAX_STEPS = 24
RATE = 1e-3


def _torch() -> Any:
    import torch

    return torch


def _out(size: int) -> int:
    return (size - 5) // 2 + 1


FLAT = 32 * _out(_out(BANDS)) * _out(_out(FRAMES))


def front_parameters() -> int:
    return 16 * (25 + 1) + 32 * (16 * 25 + 1)


def parameters(kind: str, width: int, classes: int) -> int:
    base = front_parameters() + (FLAT + 1) * width + (width + 1) * classes
    if kind == "transformer":
        layer = 4 * width * width + 4 * width + 2 * (4 * width * width) + 5 * width + 4 * width
        return base + LAYERS * layer + MAX_STEPS * width
    if kind == "lstm":
        return base + 4 * (2 * width * width + 2 * width)
    raise ValueError(kind)


def width_for(kind: str, classes: int, target: int) -> int:
    """The width whose parameter count is nearest ``target`` (a multiple of four, at least 8)."""
    best = min(range(8, 1024, 4), key=lambda w: abs(parameters(kind, w, classes) - target))
    return int(best)


def macs_per_step(kind: str, width: int, classes: int, steps: int) -> int:
    """Multiply-accumulates of one step's forward pass, attention over ``steps`` included."""
    h1, w1 = _out(BANDS), _out(FRAMES)
    h2, w2 = _out(h1), _out(w1)
    conv = h1 * w1 * 16 * 25 + h2 * w2 * 32 * 16 * 25
    body = FLAT * width + width * classes
    if kind == "transformer":
        body += LAYERS * (12 * width * width + 2 * steps * width)
    else:
        body += 8 * width * width
    return int(conv + body)


class Sequence:
    """A model over steps of windows, its loss and its optimizer."""

    def __init__(
        self, kind: str, loss: str, width: int, classes: int, seed: int, rate: float = RATE
    ) -> None:
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(int(seed))
        if loss not in ("step", "ctc"):
            raise ValueError("loss must be step or ctc")
        self.kind, self.loss, self.width, self.classes = kind, loss, width, classes
        self.outputs = classes + 1 if loss == "ctc" else classes  # the blank last
        self.front = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(FLAT, width),
        )
        if kind == "transformer":
            self.position = nn.Parameter(torch.zeros(MAX_STEPS, width))
            layer = nn.TransformerEncoderLayer(
                width, HEADS, 4 * width, dropout=0.0, batch_first=True
            )
            self.body: Any = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        elif kind == "lstm":
            self.position = None
            self.body = nn.LSTM(width, width, batch_first=True)
        else:
            raise ValueError("kind must be transformer or lstm")
        self.head = nn.Linear(width, self.outputs)
        self.optimizer = torch.optim.Adam(self.tensors(), lr=rate)
        self.work = {
            "updates": 0,
            "phrases_taught": 0,
            "train_macs": 0.0,
            "train_seconds": 0.0,
            "read_macs": 0.0,
            "read_seconds": 0.0,
        }

    def tensors(self) -> list[Any]:
        out = [*self.front.parameters(), *self.body.parameters(), *self.head.parameters()]
        if self.position is not None:
            out.append(self.position)
        return out

    def parameters(self) -> int:
        return int(sum(p.numel() for p in self.tensors()))

    def set_rate(self, rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def copy(self) -> Sequence:
        return copy.deepcopy(self)

    def logits(self, x: np.ndarray) -> Any:
        """``x``: (rows, steps, 3920), each step the window as the brain hears it."""
        torch = _torch()
        rows, steps = x.shape[:2]
        flat = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        h = self.front(flat.reshape(rows * steps, 1, BANDS, FRAMES)).reshape(rows, steps, -1)
        if self.kind == "transformer":
            h = h + self.position[:steps]
            mask = torch.triu(torch.full((steps, steps), float("-inf")), diagonal=1)
            h = self.body(h, mask=mask)
        else:
            h, _ = self.body(h)
        return self.head(h)

    def teach(
        self, x: np.ndarray, target: np.ndarray, refs: list[tuple[int, ...]], end: int
    ) -> float:
        torch = _torch()
        functional = torch.nn.functional
        started = time.process_time()
        logits = self.logits(x)
        if self.loss == "ctc":
            log_probs = functional.log_softmax(logits, dim=-1).transpose(0, 1)
            symbols = [[*r, end] for r in refs]
            flat = torch.tensor([s for row in symbols for s in row], dtype=torch.long)
            lengths = torch.tensor([len(row) for row in symbols], dtype=torch.long)
            steps = torch.full((len(x),), x.shape[1], dtype=torch.long)
            loss = functional.ctc_loss(
                log_probs, flat, steps, lengths, blank=self.classes, zero_infinity=True
            )
        else:
            loss = functional.cross_entropy(
                logits.reshape(-1, self.outputs),
                torch.from_numpy(np.asarray(target, dtype=np.int64)).reshape(-1),
            )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.work["updates"] += 1
        self.work["phrases_taught"] += len(x)
        self.work["train_macs"] += (
            3.0
            * macs_per_step(self.kind, self.width, self.outputs, x.shape[1])
            * x.shape[0]
            * x.shape[1]
        )
        self.work["train_seconds"] += time.process_time() - started
        return float(loss)

    def actions(self, x: np.ndarray, silence: int) -> np.ndarray:
        """Per-step actions (rows, steps); a CTC model's collapsed symbols, silence after them."""
        torch = _torch()
        started = time.process_time()
        with torch.no_grad():
            best = self.logits(x).argmax(dim=-1).numpy()
        self.work["read_macs"] += float(
            macs_per_step(self.kind, self.width, self.outputs, x.shape[1]) * x.shape[0] * x.shape[1]
        )
        self.work["read_seconds"] += time.process_time() - started
        if self.loss == "step":
            return np.asarray(best)
        out = np.full(best.shape, silence, np.int64)
        for r, row in enumerate(best):
            previous, at = self.classes, 0
            for a in row:
                if a != self.classes and a != previous:
                    out[r, at] = a
                    at += 1
                previous = a
        return out
