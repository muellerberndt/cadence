"""The matched controls of #332 E5: a convolutional network and a vision transformer in place of
the parrot's distance neurons.

Each control sees both eyes' whole images (at half resolution) and where the sighting is in them
(a marker channel an eye: a small bump at the figure's middle, in each eye that sees it), and
answers with a distribution over the same distances the reach prefers (log-spaced over its
span). It is taught by the same lessons as the brain's reach, as they come: the distance at
which two sightings of one thing agree, a soft target over the bins, one Adam step a lesson.
Its size is matched to the reach's learned parameters (``parameters``); ten times it is read
with no gate. The brain's other parts (figures, kinds, the map, the focus, the teacher) are the
same, so the control and the brain live the same life.

``Parrot(genes, control=ImageReach("cnn", parameters))`` puts a control in the reach's place.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

CHANNELS = 8  # left RGB, right RGB, a marker for each eye


def _torch() -> Any:
    import torch

    torch.set_num_threads(1)
    return torch


def cnn(width: int, bins: int) -> Any:
    """Three strided convolutions, a mean over the image, a linear read: ``width`` channels."""
    nn = _torch().nn
    return nn.Sequential(nn.Conv2d(CHANNELS, width, 5, stride=2, padding=2), nn.ReLU(),
                         nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, bins))


class Vit:
    """A vision transformer: patches of ``patch`` pixels embedded at ``width``, one encoder
    layer of one head, a class token read out to the bins."""

    def __init__(self, width: int, bins: int, shape: tuple[int, int], patch: tuple[int, int]):
        torch = _torch()
        nn = torch.nn
        self.patch = patch
        tokens = (shape[0] // patch[0]) * (shape[1] // patch[1])
        self.embed = nn.Linear(CHANNELS * patch[0] * patch[1], width)
        self.position = nn.Parameter(torch.zeros(1, tokens + 1, width))
        self.token = nn.Parameter(torch.zeros(1, 1, width))
        self.layer = nn.TransformerEncoderLayer(width, 1, dim_feedforward=2 * width,
                                                dropout=0.0, batch_first=True)
        self.read = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, bins))
        self.modules = nn.ModuleList([self.embed, self.layer, self.read])
        self.tokens = tokens

    def parameters(self) -> list[Any]:
        return [*self.modules.parameters(), self.position, self.token]

    def __call__(self, x: Any) -> Any:
        b, c, h, w = x.shape
        ph, pw = self.patch
        x = x[:, :, : h // ph * ph, : w // pw * pw]
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # (b, c, H, W, ph, pw)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, self.tokens, -1)
        z = self.embed(x)
        z = _torch().cat([self.token.expand(b, -1, -1), z], dim=1) + self.position
        return self.read(self.layer(z)[:, 0])


class ImageReach:
    """A control in the reach's place; ``kind`` "cnn" or "vit", sized to ``parameters``."""

    def __init__(self, kind: str, parameters: int, *, bins: int = 24,
                 span: tuple[float, float] = (0.12, 1.2), width: float = 1.5,
                 rate: float = 3e-3, window: int = 12, seed: int = 0):
        torch = _torch()
        torch.manual_seed(int(seed))
        self.kind, self.window = str(kind), int(window)
        self.preferred = np.exp(np.linspace(math.log(span[0]), math.log(span[1]), int(bins)))
        self.width = float(width)
        self.shape = (50, 88)
        best = None
        for w in range(1, 512):
            net = cnn(w, int(bins)) if kind == "cnn" else Vit(w, int(bins), self.shape, (10, 11))
            count = sum(int(p.numel()) for p in net.parameters())
            if best is None or abs(count - parameters) < abs(best[1] - parameters):
                best = (net, count, w)
            if count > parameters:
                break
        if kind not in ("cnn", "vit") or best is None:
            raise ValueError("a control is 'cnn' or 'vit'")
        self.net, self.count, self.size = best
        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=float(rate))
        self.images: dict[int, Any] = {}
        self.work = {"reads": 0, "lessons": 0}

    def parameters(self) -> int:
        return int(self.count)

    def macs(self) -> int:
        """Multiply-accumulates of one read (a forward pass)."""
        h, w = self.shape
        if self.kind == "cnn":
            c = self.size
            out1 = (h // 2) * (w // 2)
            out2 = math.ceil(h / 4) * math.ceil(w / 4)
            out3 = math.ceil(h / 8) * math.ceil(w / 8)
            return int(out1 * c * CHANNELS * 25 + out2 * c * c * 9 + out3 * c * c * 9
                       + c * len(self.preferred))
        d, tokens = self.size, (h // 10) * (w // 11) + 1
        return int(tokens * (CHANNELS * 110 * d + 4 * d * d + 4 * d * d) + tokens * tokens * d * 2
                   + d * len(self.preferred))

    def see(self, moment: int, images: np.ndarray) -> None:
        """Keep a moment's images (both eyes, half resolution) for ``window`` moments."""
        torch = _torch()
        x = np.asarray(images, dtype=np.float32)[:, ::2, ::2, :] / 255.0  # (2, 50, 88, 3)
        self.images[int(moment)] = torch.from_numpy(np.ascontiguousarray(
            np.concatenate([x[0], x[1]], axis=-1).transpose(2, 0, 1)))
        for old in [m for m in self.images if m <= int(moment) - self.window]:
            del self.images[old]

    def _input(self, queries: list[Any]) -> Any:
        torch = _torch()
        h, w = self.shape
        rows, cols = np.arange(h)[:, None], np.arange(w)[None, :]
        batch = []
        for moment, marks in queries:
            marker = np.zeros((2, h, w), dtype=np.float32)
            for eye, col, row in marks:
                marker[int(eye)] = np.maximum(marker[int(eye)], np.exp(
                    -0.5 * (((rows - row / 2.0) / 1.5) ** 2 + ((cols - col / 2.0) / 1.5) ** 2)))
            batch.append(torch.cat([self.images[int(moment)], torch.from_numpy(marker)], dim=0))
        return torch.stack(batch)

    def read(self, queries: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        """The distance of each sighting (the probability-weighted mean of the bins on a log
        scale) and the most probable bin's probability."""
        torch = _torch()
        if not queries:
            return np.zeros(0), np.zeros(0)
        with torch.no_grad():
            p = torch.softmax(self.net(self._input(queries)), dim=1).numpy()
        self.work["reads"] += len(queries)
        logs = np.log(self.preferred)
        return np.exp(p @ logs), p.max(axis=1)

    def teach(self, queries: list[Any], distances: np.ndarray) -> float:
        """One Adam step on the lessons: a soft target over the bins at each distance."""
        torch = _torch()
        keep = [k for k, (m, _) in enumerate(queries) if int(m) in self.images]
        if not keep:
            return 0.0
        logs = np.log(self.preferred)
        spacing = float(np.mean(np.diff(logs)))
        d = np.log(np.asarray(distances, dtype=np.float64)[keep])[:, None]
        target = np.exp(-0.5 * ((d - logs[None, :]) / (self.width * spacing)) ** 2)
        target /= target.sum(axis=1, keepdims=True)
        out = self.net(self._input([queries[k] for k in keep]))
        loss = -(torch.from_numpy(target.astype(np.float32))
                 * torch.log_softmax(out, dim=1)).sum(dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.work["lessons"] += len(keep)
        return float(loss.item())


__all__ = ["ImageReach", "Vit", "cnn"]
