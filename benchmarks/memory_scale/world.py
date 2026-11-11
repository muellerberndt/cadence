"""The records, cues and fact streams of the memory-scale instruments (cadence#288).

**Records.** A record is an address code with a content code, both of unit length. The key source
is a parameter (``KEYS``): ``code`` draws every address at random on the unit sphere and stands in
for a code of meaning; ``sparse`` draws a nonnegative code with a tenth of its units active, the
shape a learned or a fixed sparse code gives. A quarter of the records, rounded down to whole
families, come in families of ``FAMILY`` whose addresses share a direction at cosine ``SHARE``;
the rest are singletons.

**Cues.** A cue never holds a content. ``exact`` is the record's address; ``noisy`` adds noise of
the address's own energy (cosine .71); ``partial`` silences half of its coordinates; ``similar``
is the exact address of a family member, whose seven siblings lie at cosine .8; ``similar_noisy``
adds the noise. For several heads, ``family`` is a family's shared direction, similar to all of
its members, and ``superposed`` is the unit sum of k unrelated singletons' addresses.

**Fact streams.** ``Stream`` draws which fact occurs at each moment by a Zipf law over the facts'
ranks, changes a fact's content before an occurrence with probability ``change``, and with
``drift`` reshuffles the ranks at the midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAMILY = 8  # records per family
SHARE = 0.8  # the cosine between two addresses of one family
FAMILIES = 0.25  # the share of the records that come in families
KEYS = ("code", "sparse")
CUES = ("exact", "noisy", "partial", "similar", "similar_noisy")
HEAD_CUES = ("family", "superposed")


def unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.asarray(x / np.where(norm > 0, norm, 1.0))


def codes(rng: np.random.Generator, n: int, width: int, keys: str = "code") -> np.ndarray:
    """``n`` unit codes of ``width`` coordinates from the key source ``keys``."""
    if keys == "code":
        return unit(rng.standard_normal((n, width)))
    if keys == "sparse":  # a tenth of the units active, nonnegative
        active = max(width // 10, 1)
        out = np.zeros((n, width))
        at = np.argsort(rng.random((n, width)), axis=1)[:, :active]
        np.put_along_axis(out, at, 0.5 + rng.random((n, active)), axis=1)
        return unit(out)
    raise ValueError(f"keys is one of {KEYS}")


@dataclass
class Records:
    address: np.ndarray  # (n, width), unit rows
    content: np.ndarray  # (n, value_width), unit rows
    family: np.ndarray  # (n,), the record's family or -1
    themes: np.ndarray  # (families, width), each family's shared direction

    @property
    def n(self) -> int:
        return int(len(self.address))

    def members(self, family: int) -> np.ndarray:
        return np.flatnonzero(self.family == family)

    @property
    def singletons(self) -> np.ndarray:
        return np.flatnonzero(self.family < 0)


def records(
    rng: np.random.Generator, n: int, width: int, value_width: int, keys: str = "code"
) -> Records:
    """``n`` records, a quarter of them in families, in a shuffled order."""
    families = int(n * FAMILIES) // FAMILY
    address = codes(rng, n, width, keys)
    family = np.full(n, -1, dtype=np.int64)
    themes = codes(rng, families, width, keys)
    order = rng.permutation(n)
    for f in range(families):
        rows = order[f * FAMILY : (f + 1) * FAMILY]
        address[rows] = unit(np.sqrt(SHARE) * themes[f] + np.sqrt(1 - SHARE) * address[rows])
        family[rows] = f
    return Records(address, codes(rng, n, value_width, "code"), family, themes)


def cue(rng: np.random.Generator, address: np.ndarray, kind: str) -> np.ndarray:
    """One cue of ``kind`` for a record's ``address``."""
    if kind in ("exact", "similar"):
        return address.copy()
    if kind in ("noisy", "similar_noisy"):
        return unit(address + rng.standard_normal(len(address)) / np.sqrt(len(address)))
    if kind == "partial":
        out = address.copy()
        out[rng.permutation(len(address))[: len(address) // 2]] = 0.0
        return out
    raise ValueError(f"a cue is one of {CUES}")


def targets(rng: np.random.Generator, recs: Records, kind: str, trials: int) -> np.ndarray:
    """The records ``trials`` cues of ``kind`` ask for: family members for the similar cues,
    singletons for the others."""
    pool = np.flatnonzero(recs.family >= 0) if kind.startswith("similar") else recs.singletons
    return pool[rng.integers(len(pool), size=trials)] if len(pool) else pool


def head_trial(
    rng: np.random.Generator, recs: Records, kind: str, heads: int
) -> tuple[np.ndarray, np.ndarray]:
    """A cue relevant to several records and those records: a family's shared direction and its
    members (``family``), or the unit sum of ``heads`` singletons' addresses and those singletons
    (``superposed``)."""
    if kind == "family":
        f = int(rng.integers(len(recs.themes)))
        return recs.themes[f].copy(), recs.members(f)
    if kind == "superposed":
        rows = rng.choice(recs.singletons, size=heads, replace=False)
        return unit(recs.address[rows].sum(axis=0)), rows
    raise ValueError(f"a cue for several heads is one of {HEAD_CUES}")


# -- fact streams


@dataclass
class Stream:
    """Which fact occurs at each of ``moments`` moments, and which occurrences follow a change of
    the fact's content. ``facts`` facts are used by a Zipf law of exponent ``zipf`` over their
    ranks; fact ``order[r]`` holds rank ``r``. With ``drift`` the ranks are reshuffled at the
    midpoint (``order_late``)."""

    facts: int
    moments: int
    seed: int
    zipf: float = 1.0
    change: float = 0.05
    drift: bool = False

    def __post_init__(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 288, 1]))
        weight = 1.0 / np.arange(1, self.facts + 1) ** self.zipf
        self.probability = weight / weight.sum()
        ranks = np.searchsorted(np.cumsum(self.probability), rng.random(self.moments))
        ranks = np.minimum(ranks, self.facts - 1)
        self.order = rng.permutation(self.facts)
        self.order_late = rng.permutation(self.facts) if self.drift else self.order
        half = self.moments // 2
        self.fact = np.concatenate([self.order[ranks[:half]], self.order_late[ranks[half:]]])
        self.changed = rng.random(self.moments) < self.change

    def top(self, share: float = 0.1, late: bool = True) -> np.ndarray:
        """The ``share`` of the facts used most, by the ranks of the stream's second half
        (``late``) or its first."""
        order = self.order_late if late else self.order
        return np.asarray(order[: max(int(self.facts * share), 1)])

    def bottom(self, share: float = 0.5, late: bool = True) -> np.ndarray:
        """The ``share`` of the facts used least."""
        order = self.order_late if late else self.order
        return np.asarray(order[self.facts - int(self.facts * share):])
