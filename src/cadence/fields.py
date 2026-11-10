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

A three-channel image is read through opponent channels, as a retina's ganglion cells report
colour: luminance ``(R + G + B) / sqrt(3)``, red against green ``(R - G) / sqrt(2)`` and blue
against yellow ``(2 B - R - G) / sqrt(6)``. The fields divide into two groups, each with its
own patches, retina and competition for the residual: the share ``LUMINANCE`` of the fields
read the luminance channel, where an object's shape is the same whatever its hue, and the
rest read the two colour channels together, each channel of a patch centred on its own mean,
so a patch of one colour reads near zero as a patch of one brightness does. An image of any
other number of channels is read by one group over all of its channels.

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
    colour: str = "opponent"
    dictionary: np.ndarray = field(init=False, repr=False)
    thresholds: np.ndarray = field(init=False, repr=False)
    use: np.ndarray = field(init=False, repr=False)

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
    MATURITY = 0.0
    MATURE_THRESHOLDS = True
    # how a three-channel image is read: ``"opponent"`` (luminance fields and colour fields),
    # ``"channels"`` (the fields shared among the three channels, each reading one) or
    # ``"joint"`` (every field reads all three); the last two are controls
    COLOURS = ("opponent", "channels", "joint")
    LUMINANCE = 0.75
    OPPONENT = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 2.0]]) / np.sqrt(
        np.array([[3.0], [2.0], [6.0]])
    )

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
        if self.colour not in self.COLOURS:
            raise ValueError(f"colour is one of {', '.join(self.COLOURS)}")
        if self.channels != 3:
            self.colour = "joint"  # one group over every channel
        # the groups: the channels each reads and its share of the fields
        if self.colour == "opponent":
            luminance = min(self.features - 1, max(1, round(self.LUMINANCE * self.features)))
            shares = [((0,), luminance), ((1, 2), self.features - luminance)]
        elif self.colour == "channels":
            if self.features < 3:
                raise ValueError("fields per channel hold at least 3 fields")
            base, extra = divmod(self.features, 3)
            shares = [((k,), base + (k < extra)) for k in range(3)]
        else:
            shares = [(tuple(range(self.channels)), self.features)]
        self.learning = True
        self.rows = (self.height - self.FIELD) // self.STRIDE + 1
        self.cols = (self.width - self.FIELD) // self.STRIDE + 1
        self.patch = self.FIELD * self.FIELD * self.channels
        # per group: its channels, its rows of the dictionary and its fields
        self._groups: list[tuple[np.ndarray, slice, slice]] = []
        at = first = 0
        for members, count in shares:
            size = self.FIELD * self.FIELD * len(members)
            self._groups.append(
                (np.array(members), slice(at, at + size), slice(first, first + count))
            )
            at, first = at + size, first + count
        offsets = np.arange(self.FIELD)
        self._ys = (np.arange(self.rows) * self.STRIDE)[:, None] + offsets[None, :]
        self._xs = (np.arange(self.cols) * self.STRIDE)[:, None] + offsets[None, :]
        self._pools = [
            (ys, xs) for ys in self._windows(self.rows) for xs in self._windows(self.cols)
        ]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, _FIELDS_STREAM]))
        draw = rng.standard_normal((self.patch, self.features))
        if len(self._groups) > 1:  # a field reads its group's channels alone
            mask = np.zeros_like(draw)
            for _, dims, fields in self._groups:
                mask[dims, fields] = 1.0
            draw = draw * mask
        self.dictionary = _unit_columns(draw)
        self.thresholds = np.full(self.features, float(self.THRESHOLD))
        self.use = np.zeros(self.features)  # per field: the learning steps it responded in
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
        """Every patch of every frame centred and contrast-normalized, each group's part on its
        own, the groups side by side: ``(rows, positions, patch)``."""
        frames = self._frames(x)
        if self.colour == "opponent":
            frames = frames @ self.OPPONENT.T
        patches = frames[:, self._ys[:, None, :, None], self._xs[None, :, None, :], :]
        shape = (len(frames), self.rows * self.cols, self.FIELD * self.FIELD)
        parts = []
        for members, _, _ in self._groups:
            centred: np.ndarray
            if self.colour == "joint":  # centred over the whole patch
                whole = patches.reshape(*shape[:2], self.patch)
                centred = whole - whole.mean(axis=2, keepdims=True)
            else:  # each channel of the patch centred on its own mean
                own = patches[..., members].reshape(*shape, len(members))
                centred = (own - own.mean(axis=2, keepdims=True)).reshape(*shape[:2], -1)
            parts.append(
                centred / np.sqrt((centred * centred).mean(axis=2, keepdims=True) + self.RETINA)
            )
        return np.asarray(parts[0] if len(parts) == 1 else np.concatenate(parts, axis=2))

    def responses(self, patches: np.ndarray) -> np.ndarray:
        """The fields' rectified responses to retina patches, each group to its own part."""
        if len(self._groups) == 1:
            return np.asarray(np.maximum(patches @ self.dictionary - self.thresholds, 0.0))
        return np.concatenate([
            np.maximum(
                patches[..., dims] @ self.dictionary[dims, fields] - self.thresholds[fields], 0.0
            )
            for _, dims, fields in self._groups
        ], axis=-1)

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
        """One step of the dictionary and the thresholds from the frames ``x``, each group on
        its patches that carry structure; returns whether a table moved. A nonfinite proposal
        refuses the step and leaves the tables as they were."""
        patches = self.retina(x).reshape(-1, self.patch)
        if not self.learning:
            return False
        self.work["updates"] += 1
        self.work["update_rows"] += len(np.atleast_2d(x))
        dictionary, thresholds = self.dictionary.copy(), self.thresholds.copy()
        use = self.use.copy()
        taken = 0
        for _, dims, fields in self._groups:
            own = patches[:, dims]
            structured = own[(own * own).mean(axis=1) >= self.STRUCTURE]
            if not len(structured):
                continue
            held = self.dictionary[dims, fields]
            a = np.maximum(structured @ held - self.thresholds[fields], 0.0)
            residual = structured - a @ held.T
            share = (a > 0.0).mean(axis=0)
            if self.MATURITY:  # a field's steps fall with its use
                settled = 1.0 / (1.0 + self.use[fields] / self.MATURITY)
                step = held + self.RATE * settled * ((residual.T @ a) / len(structured))
                homeostasis = self.THRESHOLD_RATE * (settled if self.MATURE_THRESHOLDS else 1.0)
                moved = np.maximum(
                    self.thresholds[fields] + homeostasis * (share - self.TARGET), 0.0
                )
            else:
                step = held + self.RATE * (residual.T @ a) / len(structured)
                moved = np.maximum(
                    self.thresholds[fields] + self.THRESHOLD_RATE * (share - self.TARGET), 0.0,
                )
            if not (np.isfinite(step).all() and np.isfinite(moved).all()):
                raise ValueError("the learned fields must remain finite")
            cancelled = ~(step != 0.0).any(axis=0)
            step[:, cancelled] = held[:, cancelled]
            dictionary[dims, fields] = _unit_columns(step)
            thresholds[fields] = moved
            use[fields] += share > 0.0
            taken += len(structured)
        if not taken:
            return False
        self.dictionary, self.thresholds, self.use = dictionary, thresholds, use
        self.work["update_patches"] += taken
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
        responses = self.rows * self.cols * sum(
            (dims.stop - dims.start) * (fields.stop - fields.start)
            for _, dims, fields in self._groups
        )
        return {"code": responses, "learn": 3 * responses}

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height, "width": self.width, "channels": self.channels,
            "features": self.features, "seed": self.seed, "colour": self.colour,
            "learning": bool(self.learning),
            "work": dict(self.work),
        }


def _unit_columns(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=0)
    return np.asarray(values / np.where(norms > 0.0, norms, 1.0))


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.asarray(values / np.where(norms > 0.0, norms, 1.0))
