"""Operating-point controls and motor-only credit measurements for teaching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blocks import Layout
from .connectome import Connectome


@dataclass(frozen=True)
class CreditPlan:
    hidden: np.ndarray
    regions: tuple[tuple[str, np.ndarray], ...]

    @classmethod
    def build(cls, graph: Connectome, outputs: np.ndarray) -> CreditPlan:
        # A hidden unit has an input and can affect the answer. Include a
        # one-way branch here so a missing return route is visible in the report.
        reaches = np.zeros(graph.n, dtype=bool)
        reaches[outputs] = True
        front = reaches.copy()
        while front.any():
            previous = np.zeros_like(front)
            previous[graph.pre[front[graph.post]]] = True
            front = previous & ~reaches
            reaches |= front
        reaches[outputs] = False
        reaches &= np.bincount(graph.post, minlength=graph.n) > 0
        covered = np.zeros_like(reaches)
        named = []
        for name, members in graph.populations.items():
            if "/" in name:
                continue
            index = np.asarray(members, dtype=np.intp)
            index = index[reaches[index] & ~covered[index]]
            if len(index):
                named.append((name, index))
                covered[index] = True
        if (reaches & ~covered).any():
            named.append(("unnamed", np.flatnonzero(reaches & ~covered)))
        return cls(reaches, tuple(named))


def projection_groups(
    layout: Layout, paired: np.ndarray, reverse: np.ndarray,
    members: np.ndarray, member_groups: np.ndarray, group_count: int,
) -> np.ndarray:
    """Join blocks which must be scaled together to keep efficacy ties."""
    block = layout.block_of_edge
    parent = np.arange(layout.pairs, dtype=np.intp)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    if len(members):
        group_block = np.full(group_count, layout.pairs, dtype=np.intp)
        np.minimum.at(group_block, member_groups, block[members])
        linked = np.unique(np.stack((block[members], group_block[member_groups]), axis=1), axis=0)
        for first, second in linked:
            parent[root(int(first))] = root(int(second))
    if len(paired):
        keys = np.unique(block[paired] * layout.pairs + block[reverse])
        for key in keys:
            first, second = divmod(int(key), layout.pairs)
            parent[root(first)] = root(second)
    return np.fromiter((root(i) for i in range(layout.pairs)), dtype=np.intp)


def limit_projections(
    layout: Layout, scale: np.ndarray, gain: np.ndarray, activity: np.ndarray,
    adjustable: np.ndarray, measured: np.ndarray, groups: np.ndarray, voltage: float,
) -> tuple[np.ndarray, int]:
    """Let each incoming projection spend an equal share of the region's budget.

    Read the candidate effective weights on the lesson's free activities. Limit
    their adjustable drive on this batch, keeping each block's direction and
    sharing the tighter factor over ties. This is not a bound on arbitrary
    observations, held current, or the spectral gain of a recurrent loop.
    """
    if not measured.any():
        return scale, 0
    counts = np.bincount(layout.pair_post[measured], minlength=layout.ranges)
    effective = layout.flat(np.where(adjustable, scale * gain, 0.0))
    factors = np.ones(layout.pairs)
    for block in np.flatnonzero(measured):
        k = int(block)
        a0, a1, b0, b1 = layout.bounds(k)
        table = effective[layout.offset[k]:layout.offset[k + 1]].reshape(a1 - a0, b1 - b0)
        peak = float(np.max(np.abs(activity[:, a0:a1] @ table)))
        share = voltage / counts[layout.pair_post[k]]
        if peak > share:
            factors[k] = share / peak
    limited = int(np.count_nonzero(factors < 1.0))
    if not limited:
        return scale, 0
    by_group = np.ones(layout.pairs)
    np.minimum.at(by_group, groups, factors)
    return np.where(adjustable, scale * by_group[groups[layout.block_of_edge]], scale), limited
