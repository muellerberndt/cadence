"""Owned state as a clamp: a net that carries its last equilibrium into the next one.

A settlement with a given clamp goes to one fixed point from any start, so warm-starting
the next input from the last rest state carries nothing on its own. For the last
equilibrium to shape the next, the owners must read it. Here a range of *context* owners,
one per hidden owner, is clamped to a leaky trace of the hidden owners' own activation at
the previous inputs:

    c <- decay * c + (1 - decay) * h

so the state reverberates and fades over about ``1 / (1 - decay)`` inputs rather than
vanishing at once. The context owners hear nothing (they are a source range of the block
transport, so their product is computed once per settlement) and their seams into the
hidden owners learn under the same free/nudged rule as every other seam. Credit does not
flow back through time: the trace is a clamp the rule sees, not a path it differentiates.
``stateful`` builds the wiring, ``Echo`` keeps the trace for a batch of streams.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .learning import embedded
from .settle import SettledState
from .wiring import Wiring

__all__ = ["stateful", "Echo", "FastSeams", "columns"]


def stateful(
    vocabulary: int,
    positions: int,
    dim: int,
    hidden: int,
    outputs: int,
    *,
    seed: int = 0,
    init: float = 1.0,
    context_init: float = 1.0,
) -> tuple[Wiring, np.ndarray]:
    """``embedded`` plus a context range of ``hidden`` owners wired densely into the hidden owners.

    Owners: ``positions`` blocks of ``vocabulary`` one-hot inputs, then ``hidden`` context
    owners, then the embedding, hidden and output owners of ``embedded``. Sets: ``input``,
    ``context``, ``embedding``, ``hidden``, ``output``. Returns the wiring and the tie groups
    of the embedding (pass to ``Learner(tie_groups=...)``).
    """
    base, base_tie = embedded(vocabulary, positions, dim, hidden, outputs, seed=seed, init=init)
    n_in = positions * vocabulary
    c0 = n_in  # the context range sits right after the inputs
    shift = hidden  # every non-input owner of the base moves up by the context range

    def moved(index: np.ndarray) -> np.ndarray:
        return np.where(index >= n_in, index + shift, index)

    rng = np.random.default_rng(seed + 1)
    hidden_members = np.asarray(base.sets["hidden"]) + shift
    ctx, hid = np.meshgrid(np.arange(hidden), hidden_members, indexing="ij")
    magnitude = rng.uniform(0.0, 1.0, size=hidden * hidden) * np.sqrt(6.0 / (2 * hidden))
    signs = rng.choice([-1.0, 1.0], size=hidden * hidden) * magnitude * context_init
    pre = np.concatenate([moved(base.pre), c0 + ctx.ravel()])
    post = np.concatenate([moved(base.post), hid.ravel()])
    count = np.concatenate([base.count, np.ones(hidden * hidden)])
    sign = np.concatenate([base.sign, signs])
    tie = np.concatenate([base_tie, np.full(hidden * hidden, -1)])
    sets: dict[str, Iterable[int]] = {"input": range(0, n_in), "context": range(c0, c0 + hidden)}
    sets.update({k: [int(i) + shift for i in v] for k, v in base.sets.items() if k != "input"})
    wiring = Wiring.from_edges(
        base.n + hidden,
        pre=pre,
        post=post,
        count=count,
        sign=sign,
        sets=sets,
        label=f"stateful:{positions}x{vocabulary}->{dim}->{hidden}(+{hidden} context)->{outputs}",
    )
    key = {(int(a), int(b)): int(g) for a, b, g in zip(pre, post, tie, strict=True)}
    groups = np.array(
        [key[(int(a), int(b))] for a, b in zip(wiring.pre, wiring.post, strict=True)],
        dtype=np.int64,
    )
    return wiring, groups


def columns(index: np.ndarray) -> slice | np.ndarray:
    """A slice when the owners are one contiguous range, else the index array.

    Reading or writing the columns of a wide batch through a slice is a strided pass;
    through an index array it is a gather that comes back Fortran-ordered (and a scatter
    on write), which costs tens of times more on a batch of thousands of rows.
    """
    if len(index) and np.array_equal(index, np.arange(index[0], index[0] + len(index))):
        return slice(int(index[0]), int(index[0]) + len(index))
    return np.asarray(index, dtype=np.int64)


@dataclass
class Echo:
    """The leaky trace of a batch of streams' hidden equilibria, and how it enters the clamp."""

    wiring: Wiring
    decay: float = 0.5
    amplitude: float = 1.0  # the clamp amplitude of the rule
    trace: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.decay < 1:
            raise ValueError("decay lies in [0, 1)")
        self.context = np.asarray(self.wiring.sets["context"], dtype=np.int64)
        self.hidden = np.asarray(self.wiring.sets["hidden"], dtype=np.int64)
        if len(self.context) != len(self.hidden):
            raise ValueError("one context owner per hidden owner")
        self._context_columns = columns(self.context)
        self._hidden_columns = columns(self.hidden)
        self.trace = np.zeros((0, len(self.hidden)))

    def reset(self, batch: int) -> None:
        self.trace = np.zeros((batch, len(self.hidden)))

    def keep(self, rows: np.ndarray) -> None:
        """Keep the traces of ``rows`` only (streams that ended are dropped)."""
        self.trace = self.trace[rows]

    def clamp(self, drive: np.ndarray) -> np.ndarray:
        """Write the trace into the context columns of ``drive`` (a copy is returned)."""
        out = np.array(drive, dtype=float)
        if len(self.trace) != len(out):
            self.reset(len(out))
        out[:, self._context_columns] = self.amplitude * self.trace
        return out

    def update(self, state: SettledState) -> None:
        """After a free settlement: the trace decays toward the hidden owners' activation."""
        h = np.ascontiguousarray(np.atleast_2d(state.activation)[:, self._hidden_columns])
        if len(self.trace) != len(h):
            self.reset(len(h))
        self.trace = self.decay * self.trace + (1.0 - self.decay) * h

    def to_dict(self) -> dict[str, float | int]:
        return {"decay": self.decay, "amplitude": self.amplitude, "context": int(len(self.context))}


@dataclass
class FastSeams:
    """Fast Hebbian seams between two ranges of a batch of streams, entering as a drive.

    After a settlement the strength from every ``pre`` owner to every ``post`` owner grows
    by the product of their activations (one local outer product, for the rows that
    ``write``); before the next settlement the post owners receive the pre range's clamp
    through the strengths as a drive. Strengths fade by ``decay`` per step. Nothing here is
    trained: the seams hold what happened, and the slow seams learn what to make of it.
    This is the recall rung's memory kept as owned state, per stream.
    """

    pre: np.ndarray
    post: np.ndarray
    decay: float = 1.0
    rate: float = 1.0
    amplitude: float = 1.0
    strength: np.ndarray = field(init=False)
    writes: int = 0

    def __post_init__(self) -> None:
        self.pre = np.asarray(self.pre, dtype=np.int64)
        self.post = np.asarray(self.post, dtype=np.int64)
        if not 0 <= self.decay <= 1:
            raise ValueError("decay lies in [0, 1]")
        self._pre_columns = columns(self.pre)
        self._post_columns = columns(self.post)
        self.strength = np.zeros((0, len(self.pre), len(self.post)))

    def reset(self, batch: int) -> None:
        self.strength = np.zeros((batch, len(self.pre), len(self.post)))

    def keep(self, rows: np.ndarray) -> None:
        self.strength = self.strength[rows]

    def read(self, drive: np.ndarray) -> np.ndarray:
        """The post owners' drive from the pre range's clamp in ``drive``: ``(batch, post)``."""
        if len(self.strength) != len(drive):
            self.reset(len(drive))
        cue = np.ascontiguousarray(drive[:, self._pre_columns])
        return np.asarray(self.amplitude * (cue[:, None, :] @ self.strength)[:, 0, :])

    def clamp(self, drive: np.ndarray, inplace: bool = False) -> np.ndarray:
        """Add the read to the post columns of ``drive`` (a copy, unless ``inplace``)."""
        out = drive if inplace else np.array(drive, dtype=float)
        out[:, self._post_columns] += self.read(out)
        return out

    def update(
        self,
        state: SettledState,
        write: np.ndarray | None = None,
        post: np.ndarray | None = None,
    ) -> None:
        """After a settlement: strengths fade; the rows in ``write`` add their outer product.

        ``post`` replaces the post owners' activations for the write, ``(batch, post)``: what
        actually followed (the next symbols read) rather than what the net settled on.
        """
        s = np.atleast_2d(state.activation)
        if len(self.strength) != len(s):
            self.reset(len(s))
        if self.decay < 1.0:
            self.strength *= self.decay
        if write is not None and np.any(write):
            rows = np.flatnonzero(write)
            a = np.ascontiguousarray(s[rows][:, self._pre_columns])
            b = np.ascontiguousarray(s[rows][:, self._post_columns] if post is None else post[rows])
            self.strength[rows] += self.rate * a[:, :, None] * b[:, None, :]
            self.writes += len(rows)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "pre": int(len(self.pre)),
            "post": int(len(self.post)),
            "decay": self.decay,
            "rate": self.rate,
            "amplitude": self.amplitude,
            "writes": self.writes,
        }
