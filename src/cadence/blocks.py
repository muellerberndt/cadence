"""The block transport: the overlap matrix as dense blocks between ranges of owners.

A layered wiring is mostly empty space. Its input owners receive nothing, its
hidden owners hear only the inputs and the outputs, and the inputs, once
clamped, do not move after the first step. A single dense ``n x n`` product per
step pays for all of that empty space and re-multiplies the still inputs every
step; on MNIST that made the cost of one settlement step independent of the
hidden width and about a hundred times the cost it needed to be.

The layout here cuts the owners into contiguous ranges at the boundaries of the
wiring's named sets (a set that is not one contiguous run is ignored) and keeps
one dense block per ordered pair of ranges that carries at least one overlap.
The inbox of a step is the sum of the block products, and the product of a
range whose activation is bit-for-bit what it was at the previous step is
reused rather than recomputed. Nothing changes in what an owner reads: the
inbox is the same sum of the same messages, in a different association order,
and the conformance check against the owner-by-owner reference still holds to
rounding. The same blocks give the learning rule its contrast as one small
Gram product per block instead of one over the whole net.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wiring import Wiring

__all__ = ["Layout", "layout", "BlockTransport", "block_contrast"]


@dataclass(frozen=True)
class Layout:
    """Contiguous owner ranges and the ordered pairs of ranges that carry overlaps."""

    starts: np.ndarray  # (ranges + 1,) range boundaries; range r is [starts[r], starts[r + 1])
    pair_pre: np.ndarray  # (pairs,) range id of the presynaptic side of each block
    pair_post: np.ndarray  # (pairs,) range id of the postsynaptic side
    offset: np.ndarray  # (pairs + 1,) offsets of each block in the flat block array
    edge_index: np.ndarray  # (edges,) flat index of each overlap's entry

    @property
    def ranges(self) -> int:
        return int(len(self.starts) - 1)

    @property
    def pairs(self) -> int:
        return int(len(self.pair_pre))

    @property
    def size(self) -> int:
        """Entries in the flat block array: the sum of the block sizes."""
        return int(self.offset[-1]) if len(self.offset) else 0

    def bounds(self, k: int) -> tuple[int, int, int, int]:
        """``(a0, a1, b0, b1)``: block ``k`` maps owners ``[a0, a1)`` onto owners ``[b0, b1)``."""
        a, b = int(self.pair_pre[k]), int(self.pair_post[k])
        return (
            int(self.starts[a]),
            int(self.starts[a + 1]),
            int(self.starts[b]),
            int(self.starts[b + 1]),
        )

    def flat(self, weights: np.ndarray) -> np.ndarray:
        """The flat block array holding one weight per overlap and zero elsewhere."""
        out = np.zeros(self.size)
        out[self.edge_index] = weights
        return out

    def blocks(self, flat: np.ndarray) -> list[np.ndarray]:
        """Views of the flat array as one ``(a1 - a0, b1 - b0)`` matrix per block."""
        views = []
        for k in range(self.pairs):
            a0, a1, b0, b1 = self.bounds(k)
            views.append(flat[self.offset[k] : self.offset[k + 1]].reshape(a1 - a0, b1 - b0))
        return views

    def sources(self) -> list[int]:
        """Ranges that receive no block: their activation can only follow their clamp."""
        heard = set(int(r) for r in self.pair_post)
        return [r for r in range(self.ranges) if r not in heard]

    def to_dict(self) -> dict[str, int]:
        return {"ranges": self.ranges, "blocks": self.pairs, "entries": self.size}


def layout(wiring: Wiring, *, max_pairs: int = 256) -> Layout:
    """Cut the owners at the boundaries of the wiring's contiguous sets and pair the ranges.

    A wiring with no contiguous sets, or one whose sets fragment it into more than
    ``max_pairs`` blocks, gets one block: the full matrix, as before.
    """
    n = wiring.n
    cuts = {0, n}
    for members in wiring.sets.values():
        if members and members[-1] - members[0] + 1 == len(members):  # one contiguous run
            cuts.add(int(members[0]))
            cuts.add(int(members[-1]) + 1)
    starts = np.array(sorted(cuts), dtype=np.int64)
    ranges = len(starts) - 1
    range_of = np.searchsorted(starts, np.arange(n), side="right") - 1
    rp, rq = range_of[wiring.pre], range_of[wiring.post]
    uniq, inverse = np.unique(rp * ranges + rq, return_inverse=True)
    if len(uniq) > max_pairs:
        starts = np.array([0, n], dtype=np.int64)
        ranges = 1
        uniq = np.zeros(1 if wiring.edges else 0, dtype=np.int64)
        inverse = np.zeros(wiring.edges, dtype=np.int64)
    pair_pre = (uniq // ranges).astype(np.int64)
    pair_post = (uniq % ranges).astype(np.int64)
    width = starts[1:] - starts[:-1]
    sizes = width[pair_pre] * width[pair_post]
    offset = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    if wiring.edges:
        a0 = starts[pair_pre][inverse]
        b0 = starts[pair_post][inverse]
        nb = width[pair_post][inverse]
        edge_index = offset[inverse] + (wiring.pre - a0) * nb + (wiring.post - b0)
    else:
        edge_index = np.zeros(0, dtype=np.int64)
    return Layout(starts, pair_pre, pair_post, offset, edge_index.astype(np.int64))


class BlockTransport:
    """The inbox of a batch as a sum of block products, reusing the products of still ranges.

    One instance serves one settlement: ``inbox`` is called once per step with
    the activations the step reads, and a block is recomputed only when its
    presynaptic range changed since the previous call.
    """

    def __init__(self, layout: Layout, flat: np.ndarray) -> None:
        self.layout = layout
        self.blocks = layout.blocks(flat)
        self.previous: np.ndarray | None = None
        self.cache: list[np.ndarray | None] = [None] * layout.pairs
        self.still = layout.sources()  # only a range that hears nothing is worth checking

    def inbox(self, s: np.ndarray) -> np.ndarray:
        lay = self.layout
        out = np.zeros_like(s)
        moved = [True] * lay.ranges
        if self.previous is not None:
            for r in self.still:
                a0, a1 = int(lay.starts[r]), int(lay.starts[r + 1])
                moved[r] = not np.array_equal(s[:, a0:a1], self.previous[:, a0:a1])
        for k in range(lay.pairs):
            a0, a1, b0, b1 = lay.bounds(k)
            cached = self.cache[k]
            if cached is None or moved[int(lay.pair_pre[k])]:
                cached = np.asarray(s[:, a0:a1] @ self.blocks[k])
                self.cache[k] = cached
            out[:, b0:b1] += cached
        self.previous = s.copy()
        return out


def block_contrast(layout: Layout, s_plus: np.ndarray, s_minus: np.ndarray) -> np.ndarray:
    """``sum_b s+[b, pre] s+[b, post] - s-[b, pre] s-[b, post]`` for every overlap, by blocks.

    A range whose activation is the same in both phases (the clamped inputs)
    contributes ``A.T @ (B+ - B-)``: one product instead of two.
    """
    g = np.empty(layout.size)
    for k in range(layout.pairs):
        a0, a1, b0, b1 = layout.bounds(k)
        a_plus, a_minus = s_plus[:, a0:a1], s_minus[:, a0:a1]
        b_plus, b_minus = s_plus[:, b0:b1], s_minus[:, b0:b1]
        if np.array_equal(a_plus, a_minus):
            block = a_plus.T @ (b_plus - b_minus)
        else:
            block = a_plus.T @ b_plus - a_minus.T @ b_minus
        g[layout.offset[k] : layout.offset[k + 1]] = block.ravel()
    return np.asarray(g[layout.edge_index])
