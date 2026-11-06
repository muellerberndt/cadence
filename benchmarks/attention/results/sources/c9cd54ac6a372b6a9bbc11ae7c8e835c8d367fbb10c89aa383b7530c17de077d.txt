"""The block transport: the synapse matrix as dense blocks between ranges of neurons.

A connectome can contain dense projections between populations while leaving
many other population pairs unconnected. A single dense ``n x n`` product pays
for those empty entries. Some graphs also have source populations with no
incoming synapses; once their activity stops changing, their outgoing products
can be reused. Recursive input populations receive feedback and are not such
sources. Even a source can take several steps to relax under a fixed drive.

The layout here cuts the neurons into contiguous ranges at the boundaries of the
connectome's named populations (a population that is not one contiguous run is ignored) and keeps
one dense block per ordered pair of ranges that carries at least one synapse.
The synaptic input of a step is the sum of the block products. For a source
range, an outgoing product is reused only when its activation is bit-for-bit
what it was at the previous step. Products from ranges that receive feedback
are recomputed. Nothing changes in what a neuron reads: the
synaptic input is the same sum of the same messages, in a different association order,
and the conformance check against the neuron-by-neuron reference holds to
rounding. The same blocks give the learning rule its contrast as one small
Gram product per block instead of one over the whole net.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .connectome import Connectome

__all__ = ["Layout", "layout", "BlockTransport", "block_contrast"]


def _compiled() -> bool:
    """Whether the compiled passes of ``cadence.fused`` are in use (same bits either way)."""
    try:
        from .fused import compiled
    except ImportError:  # pragma: no cover
        return False
    return compiled()


@dataclass(frozen=True)
class Layout:
    """Contiguous neuron ranges and the ordered pairs of ranges that carry synapses."""

    starts: np.ndarray  # (ranges + 1,) range boundaries; range r is [starts[r], starts[r + 1])
    pair_pre: np.ndarray  # (pairs,) range id of the presynaptic side of each block
    pair_post: np.ndarray  # (pairs,) range id of the postsynaptic side
    offset: np.ndarray  # (pairs + 1,) offsets of each block in the flat block array
    edge_index: np.ndarray  # (edges,) flat index of each synapse's entry

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
        """``(a0, a1, b0, b1)``: block ``k`` maps neurons ``[a0, a1)`` onto neurons ``[b0, b1)``."""
        a, b = int(self.pair_pre[k]), int(self.pair_post[k])
        return (
            int(self.starts[a]),
            int(self.starts[a + 1]),
            int(self.starts[b]),
            int(self.starts[b + 1]),
        )

    def flat(self, weights: np.ndarray) -> np.ndarray:
        """The flat block array, summing parallel synapses at their shared endpoints."""
        if not self.parallel_synapses and _compiled():
            from .fused import scatter

            return scatter(self.size, self.edge_index, weights)
        out = np.zeros(self.size)
        if self.parallel_synapses:
            np.add.at(out, self.edge_index, weights)
        else:
            out[self.edge_index] = weights
        return out

    @cached_property
    def block_of_edge(self) -> np.ndarray:
        """The block each synapse's entry lies in, one index per synapse; made once per layout."""
        return np.asarray(np.searchsorted(self.offset, self.edge_index, side="right") - 1)

    @cached_property
    def parallel_synapses(self) -> bool:
        """Whether distinct declared synapses share a matrix entry; checked once per layout."""
        return len(np.unique(self.edge_index)) != len(self.edge_index)

    def blocks(self, flat: np.ndarray) -> list[np.ndarray]:
        """Views of the flat array as one ``(a1 - a0, b1 - b0)`` matrix per block."""
        views = []
        for k in range(self.pairs):
            a0, a1, b0, b1 = self.bounds(k)
            views.append(flat[self.offset[k] : self.offset[k + 1]].reshape(a1 - a0, b1 - b0))
        return views

    def sources(self) -> list[int]:
        """Ranges that receive no block: their activation can only follow their stimulus."""
        heard = set(int(r) for r in self.pair_post)
        return [r for r in range(self.ranges) if r not in heard]

    def to_dict(self) -> dict[str, int]:
        return {"ranges": self.ranges, "blocks": self.pairs, "entries": self.size}


def layout(connectome: Connectome, *, max_pairs: int = 256) -> Layout:
    """Cut the neurons at the boundaries of the connectome's contiguous sets and pair the ranges.

    A connectome with no contiguous sets, or one whose sets fragment it into more than
    ``max_pairs`` blocks, gets one range and, when it has synapses, one block over the full matrix.
    """
    n = connectome.n
    cuts = {0, n}
    for members in connectome.populations.values():
        if members and members[-1] - members[0] + 1 == len(members):  # one contiguous run
            cuts.add(int(members[0]))
            cuts.add(int(members[-1]) + 1)
    starts = np.array(sorted(cuts), dtype=np.int64)
    ranges = len(starts) - 1
    range_of = np.searchsorted(starts, np.arange(n), side="right") - 1
    rp, rq = range_of[connectome.pre], range_of[connectome.post]
    uniq, inverse = np.unique(rp * ranges + rq, return_inverse=True)
    if len(uniq) > max_pairs:
        starts = np.array([0, n], dtype=np.int64)
        ranges = 1
        uniq = np.zeros(1 if connectome.synapses else 0, dtype=np.int64)
        inverse = np.zeros(connectome.synapses, dtype=np.int64)
    pair_pre = (uniq // ranges).astype(np.int64)
    pair_post = (uniq % ranges).astype(np.int64)
    width = starts[1:] - starts[:-1]
    sizes = width[pair_pre] * width[pair_post]
    offset = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    if connectome.synapses:
        a0 = starts[pair_pre][inverse]
        b0 = starts[pair_post][inverse]
        nb = width[pair_post][inverse]
        edge_index = offset[inverse] + (connectome.pre - a0) * nb + (connectome.post - b0)
    else:
        edge_index = np.zeros(0, dtype=np.int64)
    return Layout(starts, pair_pre, pair_post, offset, edge_index.astype(np.int64))


class BlockTransport:
    """The synaptic input of a batch as a sum of block products, reusing still ranges.

    One instance serves one settling run: ``synaptic_input`` is called once per step with
    the activations the step reads. A source range's products may be reused
    when its activation is unchanged; products from all other ranges are
    recomputed on every call.
    """

    def __init__(
        self, layout: Layout, flat: np.ndarray, pairs: Sequence[int] | None = None
    ) -> None:
        self.layout = layout
        self.blocks = layout.blocks(flat)
        # the blocks this transport sums, every block by default: a brain with an apical
        # compartment sums its basal and its apical blocks apart
        self.pairs = range(layout.pairs) if pairs is None else [int(k) for k in pairs]
        self.previous: np.ndarray | None = None
        self.cache: list[np.ndarray | None] = [None] * layout.pairs
        self.still = layout.sources()  # only a range that hears nothing is worth checking

    def synaptic_input(self, s: np.ndarray) -> np.ndarray:
        lay = self.layout
        out = np.zeros_like(s)
        moved = [True] * lay.ranges
        if self.previous is not None:
            for r in self.still:
                a0, a1 = int(lay.starts[r]), int(lay.starts[r + 1])
                moved[r] = not np.array_equal(s[:, a0:a1], self.previous[:, a0:a1])
        for k in self.pairs:
            a0, a1, b0, b1 = lay.bounds(k)
            cached = self.cache[k]
            if cached is None or moved[int(lay.pair_pre[k])]:
                cached = np.asarray(s[:, a0:a1] @ self.blocks[k])
                self.cache[k] = cached
            out[:, b0:b1] += cached
        self.previous = s.copy()
        return out


def block_contrast(
    layout: Layout, s_plus: np.ndarray, s_minus: np.ndarray, divisor: float = 1.0
) -> np.ndarray:
    """``sum_b s+[b, pre] s+[b, post] - s-[b, pre] s-[b, post]`` for every synapse, by blocks,
    divided by ``divisor``.

    A range whose activation is the same in both phases (the stimulated inputs)
    contributes ``A.T @ (B+ - B-)``: one product instead of two.
    """
    g = np.empty(layout.size)
    for k in range(layout.pairs):
        a0, a1, b0, b1 = layout.bounds(k)
        a_plus, a_minus = s_plus[:, a0:a1], s_minus[:, a0:a1]
        b_plus, b_minus = s_plus[:, b0:b1], s_minus[:, b0:b1]
        block = a_plus.T @ (b_plus - b_minus)
        if not np.array_equal(a_plus, a_minus):
            block = block + (a_plus - a_minus).T @ b_minus
        g[layout.offset[k] : layout.offset[k + 1]] = block.ravel()
    if _compiled():
        from .fused import gather

        return gather(g, layout.edge_index, divisor)
    return np.asarray(g[layout.edge_index] / divisor)
