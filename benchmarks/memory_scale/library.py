"""The library's own recall memory as an arm of the memory-scale instruments (cadence#288).

``Slots`` composes a newborn brain whose observation is a record's address followed by its
content, takes its recall memory (``brain.recall``, the default composition's unless ``options``
name others) at a capacity that holds ``n`` records beside the boundary slot, and reads and
writes through the memory's own methods: ``Recall.witness`` writes a moment as one slot and
``Recall.stimulate`` reads the slots with a cue. A cue is an address with a silent content, so
its logit at a slot is the cue against that slot's address. Nothing is taught: the address and
position maps are at zero, and the read is the content read at the memory's temperature.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np


class Slots:
    def __init__(
        self, founder: int, width: int, value_width: int, n: int, heads: int = 1,
        **options: Any,
    ) -> None:
        warnings.simplefilter("ignore")
        import cadence as cd

        self.width, self.value_width, self.heads = width, value_width, heads
        settings = {"recall_capacity": n + 1, **options}
        if heads > 1:
            settings["recall_heads"] = heads
        self.brain = cd.Brain.compose(
            inputs=width + value_width, actions=2, modules=(16,), seed=founder, backend="cpu",
            episodic=False, **settings,
        )
        self.recall = self.brain.recall
        self.recall.reset(1)
        neurons = self.brain.connectome.n
        self._state = SimpleNamespace(activation=np.zeros((1, neurons)))
        self._drive = np.zeros((1, neurons))
        self.records = 0

    def write(self, address: np.ndarray, content: np.ndarray) -> None:
        """One record as one written moment."""
        self.recall.witness(np.concatenate([address, content])[None, :], self._state)
        self.records += 1

    def _event(self, cue: np.ndarray) -> np.ndarray:
        return np.concatenate([cue, np.zeros(self.value_width)])[None, :]

    def read(self, cue: np.ndarray) -> list[np.ndarray]:
        """Every head's attention over the slots for ``cue``, through ``Recall.stimulate``."""
        self.recall.stimulate(self._drive, self._event(cue))
        attended = self.recall.attended_heads
        assert attended is not None
        return [np.asarray(p[0]) for p in attended if p is not None]

    def read_heads(self, cue: np.ndarray, heads: int) -> list[np.ndarray]:
        """The attention of the memory's first ``heads`` heads: heads read in turn, so the
        first k of a memory of more heads read as a memory of k heads reads."""
        recall = self.recall
        moment = recall._moment(recall._port(self._event(cue)))
        addresses = [recall._address(moment, h) for h in range(heads)]
        return [np.asarray(p[0]) for p in recall._read(addresses, moment) if p is not None]

    def slot(self, record: int) -> int:
        """The slot that holds the ``record``-th written record."""
        return int(self.recall.slots - self.records + record)

    @property
    def nbytes(self) -> int:
        recall = self.recall
        names = ("keys", "values", "nexts", "held")
        return int(sum(getattr(recall, name).nbytes for name in names))
