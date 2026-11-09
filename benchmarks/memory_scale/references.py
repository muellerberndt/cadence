"""The references of the memory-scale instruments (cadence#288): independently scored, never
part of an answer path.

``Exact`` is an exact nearest-neighbour store over the same records, the ceiling for retrieval:
its first choice is the record whose key has the largest dot product with the cue, and k heads
read the k largest. ``Cache`` holds facts by their exact identity under a budget of slots and
evicts by one of four policies: least recently used, first in first out, random, and least
frequently used with exact counts (an upper reference: it keeps the count the candidate rule may
not keep).
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

POLICIES = ("lru", "fifo", "random", "lfu")


class Exact:
    """Exact nearest neighbours by dot product."""

    def __init__(self, keys: np.ndarray) -> None:
        self.keys = np.ascontiguousarray(keys, dtype=np.float64)

    def top(self, cue: np.ndarray, k: int = 1) -> np.ndarray:
        """The ``k`` records nearest ``cue``, the nearest first."""
        scores = self.keys @ np.asarray(cue, dtype=np.float64)
        if k >= len(scores):
            return np.argsort(-scores)
        best = np.argpartition(scores, -k)[-k:]
        return np.asarray(best[np.argsort(-scores[best])])

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes)


class Cache:
    """Facts by exact identity under a budget of ``capacity`` slots. ``ask`` says whether the
    fact is held at its current version and counts as a use; ``tell`` states the fact, revising
    it in place where it is held and evicting by ``policy`` where it is not. ``holds`` is a
    probe: no use is counted. The least-frequently-used policy keeps an exact count of every
    fact's uses, held or not."""

    def __init__(self, capacity: int, policy: str, facts: int, seed: int = 0) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy is one of {POLICIES}")
        self.capacity, self.policy = int(capacity), policy
        self.version = np.full(facts, -1, dtype=np.int64)  # the version held; -1: not held
        self.uses = np.zeros(facts)
        self.queue: OrderedDict[int, None] = OrderedDict()  # lru and fifo: the next to go leads
        self.members: list[int] = []  # random: the held facts
        self.place = np.full(facts, -1, dtype=np.int64)  # random: a fact's place in ``members``
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 288, 2]))
        self.held = 0
        self.writes = 0

    def holds(self, fact: int, version: int | None = None) -> bool:
        return bool(self.version[fact] >= 0 and (version is None or self.version[fact] == version))

    def _use(self, fact: int) -> None:
        self.uses[fact] += 1
        if self.policy == "lru":
            self.queue.move_to_end(fact)

    def ask(self, fact: int, version: int) -> bool:
        if self.version[fact] < 0:
            return False
        self._use(fact)
        return bool(self.version[fact] == version)

    def tell(self, fact: int, version: int) -> bool:
        """State the fact; True where it took a slot it did not hold."""
        if self.version[fact] >= 0:
            self.version[fact] = version
            self._use(fact)
            return False
        if self.held >= self.capacity:
            self._evict()
        self.version[fact] = version
        self.uses[fact] += 1
        self.held += 1
        self.writes += 1
        if self.policy in ("lru", "fifo"):
            self.queue[fact] = None
        elif self.policy == "random":
            self.place[fact] = len(self.members)
            self.members.append(fact)
        return True

    def _evict(self) -> None:
        if self.policy in ("lru", "fifo"):  # the least recently used, or the first in, leads
            gone, _ = self.queue.popitem(last=False)
        elif self.policy == "random":
            at = int(self.rng.integers(len(self.members)))
            gone, last = self.members[at], self.members[-1]
            self.members[at] = last
            self.place[last] = at
            self.members.pop()
            self.place[gone] = -1
        else:  # the held fact used least over the whole stream
            gone = int(np.argmin(np.where(self.version >= 0, self.uses, np.inf)))
        self.version[gone] = -1
        self.held -= 1
