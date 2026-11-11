"""Neurons and synapses as arrays.

A ``Connectome`` is the declared topology of a brain: ``n`` neurons and a
list of directed synapses, each carrying a synapse count and a sign. It is
immutable, sorted by (post, pre), and it knows nothing about dynamics.
Named populations of neurons live alongside it so protocols can speak in names.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = ["Connectome"]


def _neuron_indices(values: Sequence[int] | np.ndarray, n: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a vector of integer neuron indices")
    if array.dtype.kind == "f" and (
        not np.isfinite(array).all() or np.any(array != np.floor(array))
    ):
        raise ValueError(f"{name} must contain integer neuron indices")
    if np.any(array < 0) or np.any(array >= n):
        raise ValueError(f"{name} indices must lie in [0, n)")
    return array.astype(np.int64, copy=False)


def _edge_arrays(
    n: int,
    pre: Sequence[int] | np.ndarray,
    post: Sequence[int] | np.ndarray,
    count: Sequence[float] | np.ndarray | None,
    sign: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    pre_a = _neuron_indices(pre, n, "pre")
    post_a = _neuron_indices(post, n, "post")
    count_a = np.ones(len(pre_a)) if count is None else np.asarray(count, dtype=np.float64)
    sign_a = np.ones(len(pre_a)) if sign is None else np.asarray(sign, dtype=np.float64)
    if any(a.shape != pre_a.shape for a in (post_a, count_a, sign_a)):
        raise ValueError("pre, post, count, and sign must have one entry per synapse")
    if not np.isfinite(count_a).all() or not np.isfinite(sign_a).all():
        raise ValueError("count and sign must be finite")
    if (count_a < 0).any():
        raise ValueError("contact counts must be nonnegative; use sign for inhibition")
    return pre_a, post_a, count_a, sign_a


@dataclass(frozen=True)
class Connectome:
    """``n`` neurons; synapses ``pre[i] -> post[i]`` with ``count[i]`` contacts and ``sign[i]``."""

    n: int
    pre: np.ndarray
    post: np.ndarray
    count: np.ndarray
    sign: np.ndarray
    populations: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    label: str = "connectome"

    def __post_init__(self) -> None:
        arrays = _edge_arrays(self.n, self.pre, self.post, self.count, self.sign)
        for name, value in zip(("pre", "post", "count", "sign"), arrays, strict=True):
            object.__setattr__(self, name, value)
        if np.any(self.pre == self.post):
            raise ValueError("a synapse joins two distinct neurons; drop autapses first")
        order = np.lexsort((self.pre, self.post))
        for name in ("pre", "post"):
            object.__setattr__(self, name, getattr(self, name)[order].astype(np.int64))
        object.__setattr__(self, "count", self.count[order].astype(np.float64))
        object.__setattr__(self, "sign", self.sign[order].astype(np.float64))
        object.__setattr__(
            self,
            "populations",
            MappingProxyType({
                k: tuple(int(i) for i in np.unique(_neuron_indices(tuple(v), self.n, k)))
                for k, v in self.populations.items()
            }),
        )
        for name in ("pre", "post", "count", "sign"):
            getattr(self, name).setflags(write=False)

    def __copy__(self) -> Connectome:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Connectome:
        """A connectome is immutable (frozen fields, read-only arrays, a read-only mapping of
        populations): a copy shares it."""
        return self

    # -- construction

    @classmethod
    def from_synapses(
        cls,
        n: int,
        *,
        pre: Sequence[int] | np.ndarray,
        post: Sequence[int] | np.ndarray,
        count: Sequence[float] | np.ndarray | None = None,
        sign: Sequence[float] | np.ndarray | None = None,
        populations: Mapping[str, Iterable[int]] | None = None,
        label: str = "connectome",
        min_count: float = 0.0,
    ) -> Connectome:
        """Build from synapse lists; ``count`` defaults to 1 and ``sign`` to +1 everywhere.

        Self-synapses and synapses with fewer than ``min_count`` contacts are dropped.
        Parallel synapses between the same pair are merged: their counts sum and the
        merged sign is the count-weighted mean of their signs.
        """
        if not np.isfinite(min_count) or min_count < 0:
            raise ValueError("min_count must be finite and nonnegative")
        pre_a, post_a, count_a, sign_a = _edge_arrays(n, pre, post, count, sign)
        keep = (pre_a != post_a) & (count_a >= min_count)
        pre_a, post_a, count_a, sign_a = pre_a[keep], post_a[keep], count_a[keep], sign_a[keep]
        key = post_a * n + pre_a
        uniq, inverse = np.unique(key, return_inverse=True)
        merged_count = np.bincount(inverse, weights=count_a, minlength=len(uniq))
        merged_signed = np.bincount(inverse, weights=count_a * sign_a, minlength=len(uniq))
        merged_sign = np.divide(
            merged_signed,
            merged_count,
            out=np.zeros_like(merged_count, dtype=float),
            where=merged_count > 0,
        )
        return cls(
            n=n,
            pre=uniq % n,
            post=uniq // n,
            count=merged_count,
            sign=merged_sign,
            populations={k: tuple(v) for k, v in (populations or {}).items()},
            label=label,
        )

    def with_populations(self, **populations: Iterable[int]) -> Connectome:
        merged = {**self.populations, **{k: tuple(v) for k, v in populations.items()}}
        return Connectome(self.n, self.pre, self.post, self.count, self.sign, merged, self.label)

    # -- queries

    @property
    def synapses(self) -> int:
        return int(len(self.pre))

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.post, minlength=self.n)

    def out_degree(self) -> np.ndarray:
        return np.bincount(self.pre, minlength=self.n)

    def members(self, *names: str) -> tuple[int, ...]:
        """Union of named populations."""
        out: set[int] = set()
        for name in names:
            out.update(self.populations[name])
        return tuple(sorted(out))

    def digest(self) -> str:
        """SHA-256 of the neuron count, the sorted synapse arrays and the named populations."""
        h = sha256()
        h.update(str(self.n).encode())
        for array in (self.pre, self.post, self.count, self.sign):
            h.update(np.ascontiguousarray(array).tobytes())
        for name in sorted(self.populations):
            h.update(name.encode())
            h.update(np.asarray(self.populations[name], dtype=np.int64).tobytes())
        return h.hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "neurons": self.n,
            "synapses": self.synapses,
            "contacts": float(self.count.sum()),
            "excitatory": int((self.sign > 0).sum()),
            "inhibitory": int((self.sign < 0).sum()),
            "populations": {k: len(v) for k, v in self.populations.items()},
            "digest": self.digest(),
        }
