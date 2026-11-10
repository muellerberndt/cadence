"""Learned receptive fields: an image read through fields its own local rule shapes.

``Brain.compose(inputs=(height, width[, channels]), ..., learned_fields=features)`` reads an
image through ``features`` receptive fields of ``FIELD`` by ``FIELD`` pixels, every field
repeated at every ``STRIDE``-th pixel across the image (one map of positions per field), and
gives the first module the maps pooled over a few overlapping regions. The stage holds three
parts, each local:

* **the retina:** every patch of the image is centred on its own mean and scaled by its own
  contrast, ``x = (p - mean(p)) / sqrt(var(p) + RETINA)``, as a centre-surround cell with
  contrast gain control reports it: a uniform patch, of any brightness, reads near zero;
* **the fields:** a field ``j`` (a unit-length column of the dictionary ``Phi``) responds to a
  patch by ``a_j = max(0, Phi_j . x - t_j)``, rectified above its own threshold ``t_j``;
* **the pools:** the responses of each field averaged over each of ``POOLS`` by ``POOLS``
  regions of the map, each region ``OVERLAP`` times as wide as its share of the map, so that
  neighbouring regions overlap and an object read near a border reaches both: the stage's
  output, ``features * POOLS**2`` values, scaled to unit length per row.

The fields learn from every frame the brain sees at an accepted moment (``act``, ``step``,
``live``, ``wait`` and ``lesson``; never in ``predict``, ``fit``, ``imagine``, or a reading
on a copy with ``learning`` False), each from its own response alone, on the patches that
carry structure (a mean square of at least ``STRUCTURE`` after the retina; a uniform patch
moves nothing):

* the dictionary, Hebbian on the residual of the patch the fields reconstruct,
  ``Phi += RATE * mean_patches (x - Phi a) a^T``, each column then scaled back to unit
  length: the same rule as the learned code's dictionary, with every position of a field
  reading the same synapses;
* the thresholds, homeostatic, ``t += THRESHOLD_RATE * (share_j - TARGET)``, ``share_j`` the
  share of the structured patches field ``j`` responds to, floored at zero.

A field is one set of synapses repeated across the image (a convolution): what it learns
where objects appear it reads everywhere else, including places no object of the life has
appeared. The rest of the brain reads the pools; nothing leads back into the stage, and the
lesson's contrast and the dopamine never reach it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["LearnedFields"]

# The fields' own stream of the brain's seed, apart from the streams the brain develops, acts,
# draws its fixed code and its learned code from.
_FIELDS_STREAM = 7121


@dataclass
class LearnedFields:
    """Learned receptive fields over images of ``height`` by ``width`` by ``channels``,
    ``features`` fields; see the module note. The constants are the values the readings
    selected (benchmarks/pixels); ``learning`` False holds every table still."""

    height: int
    width: int
    channels: int
    features: int
    seed: int = 0
    dictionary: np.ndarray = field(init=False, repr=False)
    thresholds: np.ndarray = field(init=False, repr=False)

    FIELD = 8
    STRIDE = 2
    POOLS = 3
    OVERLAP = 1.5
    RETINA = 0.01
    STRUCTURE = 0.5
    RATE = 0.01
    THRESHOLD = 0.3
    THRESHOLD_RATE = 0.01
    TARGET = 0.1

    def __post_init__(self) -> None:
        for name in ("height", "width", "channels", "features"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if self.features < 2:
            raise ValueError("learned fields hold at least 2 fields")
        if min(self.height, self.width) < self.FIELD:
            raise ValueError(
                f"learned fields need an image of at least {self.FIELD} by {self.FIELD} pixels"
            )
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be a nonnegative integer")
        self.seed = int(self.seed)
        self.learning = True
        self.rows = (self.height - self.FIELD) // self.STRIDE + 1
        self.cols = (self.width - self.FIELD) // self.STRIDE + 1
        self.patch = self.FIELD * self.FIELD * self.channels
        offsets = np.arange(self.FIELD)
        self._ys = (np.arange(self.rows) * self.STRIDE)[:, None] + offsets[None, :]
        self._xs = (np.arange(self.cols) * self.STRIDE)[:, None] + offsets[None, :]
        self._pools = [
            (ys, xs) for ys in self._windows(self.rows) for xs in self._windows(self.cols)
        ]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, _FIELDS_STREAM]))
        self.dictionary = _unit_columns(rng.standard_normal((self.patch, self.features)))
        self.thresholds = np.full(self.features, float(self.THRESHOLD))
        self.work = {"codes": 0, "code_rows": 0, "updates": 0, "update_rows": 0,
                     "update_patches": 0}

    def _windows(self, n: int) -> list[slice]:
        """The pools along one axis of the map: ``POOLS`` windows, each ``OVERLAP`` times its
        share of the axis wide, spread evenly from the first position to the last."""
        size = min(n, max(1, int(round(self.OVERLAP * n / self.POOLS))))
        starts = np.linspace(0, n - size, self.POOLS).round().astype(int)
        return [slice(int(start), int(start) + size) for start in starts]

    @property
    def inputs(self) -> int:
        return self.height * self.width * self.channels

    @property
    def outputs(self) -> int:
        """The width of the stage's output: one value per field and pool."""
        return self.features * self.POOLS * self.POOLS

    def _frames(self, x: np.ndarray) -> np.ndarray:
        frames = np.atleast_2d(np.asarray(x, dtype=float))
        if frames.ndim != 2 or frames.shape[1] != self.inputs or not np.isfinite(frames).all():
            raise ValueError(f"learned fields read finite rows of {self.inputs} values")
        return frames.reshape(len(frames), self.height, self.width, self.channels)

    def retina(self, x: np.ndarray) -> np.ndarray:
        """Every patch of every frame centred and contrast-normalized,
        ``(rows, positions, patch)``."""
        frames = self._frames(x)
        patches = frames[:, self._ys[:, None, :, None], self._xs[None, :, None, :], :]
        patches = patches.reshape(len(frames), self.rows * self.cols, self.patch)
        centred = patches - patches.mean(axis=2, keepdims=True)
        return np.asarray(
            centred / np.sqrt((centred * centred).mean(axis=2, keepdims=True) + self.RETINA)
        )

    def responses(self, patches: np.ndarray) -> np.ndarray:
        """The fields' rectified responses to retina patches."""
        return np.asarray(np.maximum(patches @ self.dictionary - self.thresholds, 0.0))

    def code(self, x: np.ndarray) -> np.ndarray:
        """The pooled responses of every field, each row at unit length, ``(rows, outputs)``;
        reads, never learns."""
        a = self.responses(self.retina(x)).reshape(-1, self.rows, self.cols, self.features)
        pooled = np.stack([a[:, ys, xs].mean(axis=(1, 2)) for ys, xs in self._pools], axis=1)
        out = pooled.reshape(len(a), self.outputs)
        self.work["codes"] += 1
        self.work["code_rows"] += len(a)
        return _unit_rows(out)

    def learn(self, x: np.ndarray) -> bool:
        """One step of the dictionary and the thresholds from the frames ``x``, on the patches
        that carry structure; returns whether a table moved. A nonfinite proposal refuses the
        step and leaves the tables as they were."""
        patches = self.retina(x).reshape(-1, self.patch)
        if not self.learning:
            return False
        structured = patches[(patches * patches).mean(axis=1) >= self.STRUCTURE]
        self.work["updates"] += 1
        self.work["update_rows"] += len(np.atleast_2d(x))
        if not len(structured):
            return False
        a = self.responses(structured)
        residual = structured - a @ self.dictionary.T
        step = self.dictionary + self.RATE * (residual.T @ a) / len(structured)
        thresholds = np.maximum(
            self.thresholds + self.THRESHOLD_RATE * ((a > 0.0).mean(axis=0) - self.TARGET), 0.0
        )
        if not (np.isfinite(step).all() and np.isfinite(thresholds).all()):
            raise ValueError("the learned fields must remain finite")
        cancelled = ~(step != 0.0).any(axis=0)
        step[:, cancelled] = self.dictionary[:, cancelled]
        self.dictionary = _unit_columns(step)
        self.thresholds = thresholds
        self.work["update_patches"] += len(structured)
        return True

    def take(self, dictionary: np.ndarray, thresholds: np.ndarray) -> None:
        """Install saved tables, checked against the stage's shape."""
        dictionary = np.asarray(dictionary, dtype=float)
        thresholds = np.asarray(thresholds, dtype=float)
        if dictionary.shape != (self.patch, self.features) or thresholds.shape != (self.features,):
            raise ValueError("saved learned fields do not match the stage's shape")
        if not (np.isfinite(dictionary).all() and np.isfinite(thresholds).all()):
            raise ValueError("saved learned fields must be finite")
        if (thresholds < 0).any():
            raise ValueError("saved learned field thresholds must be nonnegative")
        self.dictionary = dictionary.copy()
        self.thresholds = thresholds.copy()

    # -- reports

    def arrays(self) -> dict[str, np.ndarray]:
        return {"dictionary": self.dictionary, "thresholds": self.thresholds}

    def parameters(self) -> int:
        return int(self.dictionary.size + self.thresholds.size)

    def nbytes(self) -> int:
        return int(self.dictionary.nbytes + self.thresholds.nbytes)

    def macs(self) -> dict[str, int]:
        """Multiply-accumulates per frame: the fields' responses, and a learning step's
        reconstruction and update on a frame whose every patch carries structure."""
        responses = self.rows * self.cols * self.patch * self.features
        return {"code": responses, "learn": 3 * responses}

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height, "width": self.width, "channels": self.channels,
            "features": self.features, "seed": self.seed, "learning": bool(self.learning),
            "work": dict(self.work),
        }


def _unit_columns(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=0)
    return np.asarray(values / np.where(norms > 0.0, norms, 1.0))


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.asarray(values / np.where(norms > 0.0, norms, 1.0))
