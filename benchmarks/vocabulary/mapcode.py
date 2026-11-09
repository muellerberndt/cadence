"""A development emulation: the read's maps reading a fixed code of the event.

At raw codes the address map is ``inputs`` squared and holds nearly all of a large brain's
parameters and much of its seconds: 56 M of 59.3 M at 5,000 words. Its rows mostly say what to
look for after a kind of moment, and a kind needs far fewer rows than there are words. Here the
address, position, anchored, relative and slope maps read ``z = x R`` in place of the event ``x``:
``R`` fixed, sparse and random, ``per_input`` units per input coordinate, each at ``1 /
sqrt(per_input)``, so a coordinate keeps its energy. The keys and values stay the raw events, and
the rule's form is unchanged: the same contrast on the same inputs ``[z, c, 1]``. A map holds
``units * inputs`` in place of ``inputs ** 2``. A newborn reads as a brain without it, since the
maps start at zero.

This changes no library file; it is read in development, on one process per life, and a brain with
it is not saved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from cadence.stream import Recall, _Moment, _of


class MapCoded(Recall):
    """``Recall`` whose maps read the fixed code ``self.map_code`` of the event."""

    def _moment(self, event: np.ndarray) -> _Moment:
        x = np.asarray(event, dtype=float)
        n = len(x)
        c = _of(self.previous, n)
        key = self.heard * x
        inputs = np.asarray(self.map_code.T.dot(x.T)).T
        w_x, w_c, b = self.head_maps[0]
        q = key + inputs @ w_x + c @ w_c + b
        return _Moment(x, c, q, key, inputs)


def install(brain: Any, units: int, seed: int, per_input: int = 4) -> None:
    """Make ``brain``'s recall memory read its maps through a code of ``units`` units."""
    recall = brain.recall
    if recall.window != 1 or recall.action or recall.heads != 1:
        raise ValueError("the emulation covers one head, no window and no command")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 266, 7]))
    inputs = recall.key_width
    rows = np.repeat(np.arange(inputs), per_input)
    cols = np.concatenate([rng.choice(units, size=per_input, replace=False)
                           for _ in range(inputs)])
    code = sparse.csr_matrix((np.full(len(rows), 1.0 / np.sqrt(per_input)), (rows, cols)),
                             shape=(inputs, units))
    recall.__class__ = MapCoded
    recall.map_code = code
    recall.map_width = int(units)
    count = recall.position_buckets
    recall.head_maps = recall._zero_maps(recall.heads, recall.address_width)
    recall.position_maps = recall._zero_maps(recall.heads if count else 0, count)
    recall.anchor_maps = recall._zero_maps(recall.heads if recall.anchor else 0, count + 1)
    recall.relative_maps = recall._zero_maps(recall.heads if recall.relative else 0,
                                             2 * recall.RELATIVE + 2)
    recall.slope_maps = recall._zero_maps(recall.heads if recall.slope else 0)
    recall.grid_maps = recall._zero_maps(recall.heads if recall.grid else 0, sum(recall.grid))
