"""Owners and overlaps as arrays.

A ``Wiring`` is the declared topology of a patch net: ``n`` owners and a
list of directed overlaps, each carrying a synapse count and a sign. It is
immutable, sorted by (post, pre), and it knows nothing about dynamics.
Named sets of owners live alongside it so protocols can speak in names.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import numpy as np

__all__ = ["Wiring"]


@dataclass(frozen=True)
class Wiring:
    """``n`` owners; overlaps ``pre[i] -> post[i]`` with ``count[i]`` contacts and ``sign[i]``."""

    n: int
    pre: np.ndarray
    post: np.ndarray
    count: np.ndarray
    sign: np.ndarray
    sets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    label: str = "wiring"

    def __post_init__(self) -> None:
        for name in ("pre", "post", "count", "sign"):
            value = np.asarray(getattr(self, name))
            object.__setattr__(self, name, value)
        if not (len(self.pre) == len(self.post) == len(self.count) == len(self.sign)):
            raise ValueError("pre, post, count, and sign must have one entry per overlap")
        if len(self.pre) and (self.pre.min() < 0 or self.pre.max() >= self.n):
            raise ValueError("pre indices must lie in [0, n)")
        if len(self.post) and (self.post.min() < 0 or self.post.max() >= self.n):
            raise ValueError("post indices must lie in [0, n)")
        if np.any(self.pre == self.post):
            raise ValueError("an overlap joins two distinct owners; drop autapses first")
        order = np.lexsort((self.pre, self.post))
        for name in ("pre", "post"):
            object.__setattr__(self, name, getattr(self, name)[order].astype(np.int64))
        object.__setattr__(self, "count", self.count[order].astype(np.float64))
        object.__setattr__(self, "sign", self.sign[order].astype(np.float64))
        object.__setattr__(self, "sets", {k: tuple(sorted(set(v))) for k, v in self.sets.items()})

    # -- construction

    @classmethod
    def from_edges(
        cls,
        n: int,
        *,
        pre: Sequence[int] | np.ndarray,
        post: Sequence[int] | np.ndarray,
        count: Sequence[float] | np.ndarray | None = None,
        sign: Sequence[float] | np.ndarray | None = None,
        sets: Mapping[str, Iterable[int]] | None = None,
        label: str = "wiring",
        min_count: float = 0.0,
    ) -> Wiring:
        """Build from edge lists; ``count`` defaults to 1 and ``sign`` to +1 everywhere.

        Overlaps with fewer than ``min_count`` contacts are dropped, and
        parallel overlaps between the same pair are merged by summing counts.
        """
        pre_a = np.asarray(pre, dtype=np.int64)
        post_a = np.asarray(post, dtype=np.int64)
        count_a = np.ones(len(pre_a)) if count is None else np.asarray(count, dtype=np.float64)
        sign_a = np.ones(len(pre_a)) if sign is None else np.asarray(sign, dtype=np.float64)
        keep = (pre_a != post_a) & (count_a >= min_count)
        pre_a, post_a, count_a, sign_a = pre_a[keep], post_a[keep], count_a[keep], sign_a[keep]
        key = post_a * n + pre_a
        uniq, inverse = np.unique(key, return_inverse=True)
        merged_count = np.bincount(inverse, weights=count_a, minlength=len(uniq))
        merged_signed = np.bincount(inverse, weights=count_a * sign_a, minlength=len(uniq))
        merged_sign = np.where(
            merged_count > 0, merged_signed / np.maximum(merged_count, 1e-12), 0.0
        )
        return cls(
            n=n,
            pre=uniq % n,
            post=uniq // n,
            count=merged_count,
            sign=merged_sign,
            sets={k: tuple(v) for k, v in (sets or {}).items()},
            label=label,
        )

    def with_sets(self, **sets: Iterable[int]) -> Wiring:
        merged = {**self.sets, **{k: tuple(v) for k, v in sets.items()}}
        return Wiring(self.n, self.pre, self.post, self.count, self.sign, merged, self.label)

    # -- queries

    @property
    def edges(self) -> int:
        return int(len(self.pre))

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.post, minlength=self.n)

    def out_degree(self) -> np.ndarray:
        return np.bincount(self.pre, minlength=self.n)

    def members(self, *names: str) -> tuple[int, ...]:
        """Union of named sets."""
        out: set[int] = set()
        for name in names:
            out.update(self.sets[name])
        return tuple(sorted(out))

    def digest(self) -> str:
        """SHA-256 of the sorted overlap arrays and the named sets."""
        h = sha256()
        h.update(str(self.n).encode())
        for array in (self.pre, self.post, self.count, self.sign):
            h.update(np.ascontiguousarray(array).tobytes())
        for name in sorted(self.sets):
            h.update(name.encode())
            h.update(np.asarray(self.sets[name], dtype=np.int64).tobytes())
        return h.hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "owners": self.n,
            "overlaps": self.edges,
            "contacts": float(self.count.sum()),
            "excitatory": int((self.sign > 0).sum()),
            "inhibitory": int((self.sign < 0).sum()),
            "sets": {k: len(v) for k, v in self.sets.items()},
            "digest": self.digest(),
        }
