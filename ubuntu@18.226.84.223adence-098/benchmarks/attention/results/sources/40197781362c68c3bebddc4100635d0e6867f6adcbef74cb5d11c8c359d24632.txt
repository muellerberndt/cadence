"""Held-address oracle for a two-head raw-token capacity control.

``FrozenRead(bank)`` accepts a shared two-head boundary without importing its
graph or brain classes. Each head selects the latest actually held IS+A/B
fact by name, independent of the query and answer order. This supplied address
is a control, never a candidate selector or an address teaching rule.

With fixed one-hot p and target pV, E_h = alpha_h/2 ||r_h-pV||^2 * held.any.
The query derivative is zero; the read force is alpha_h*(pV-r_h)*held.any.
An absent named fact in a nonempty bank restores toward zero. The historical
default gives an entirely empty bank zero force. If the supplied bank declares
``empty_anchor=True``, retain agreement with its zero content on empty streams
too: E_h = alpha_h/2 ||r_h||^2 and read force -alpha_h*r_h. This is finite
restoring agreement, not a clamp or a promise of zero settled read activity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

WIDTH = 19


def frozen(value: Any, dtype: Any = float) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def ports(value: Any) -> np.ndarray:
    index = np.asarray(value)
    if (index.shape != (WIDTH,) or index.dtype.kind not in "iu" or (index < 0).any()
            or (index > np.iinfo(np.int64).max).any()):
        raise ValueError("fixed reads require raw19 query/read ports")
    return frozen(index, np.int64)


def rows(value: Any, batch: int) -> np.ndarray:
    selected = np.asarray(value)
    if selected.ndim != 1:
        raise ValueError("rows require a one-dimensional mask or unique indices")
    if selected.dtype == np.bool_:
        if selected.shape != (batch,):
            raise ValueError("row mask must match the bank streams")
        return np.flatnonzero(selected)
    if not selected.size:
        return np.zeros(0, dtype=np.int64)
    if (selected.dtype.kind not in "iu" or (selected < 0).any()
            or (selected > np.iinfo(np.int64).max).any() or (selected >= batch).any()):
        raise ValueError("row indices are outside the bank streams")
    selected = selected.astype(np.int64)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("row indices must be unique")
    return selected


@dataclass(frozen=True)
class _Head:
    query: np.ndarray
    read: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    held: np.ndarray
    ages: np.ndarray
    amplitude: float
    mode: str = "held"
    empty_anchor: bool = False


@dataclass(frozen=True)
class _Bank:
    memories: tuple[_Head, ...]

    @property
    def empty_anchor(self) -> bool:
        return self.memories[0].empty_anchor


@dataclass(frozen=True, init=False)
class FrozenRead:
    memories: tuple[_Head, ...]
    p_fixed: np.ndarray
    targets: np.ndarray
    empty_anchor: bool
    fixed_selector = True
    mode = "held"
    heads = 2

    def __init__(self, bank: Any) -> None:
        members = tuple(bank.memories)
        if len(members) != 2:
            raise ValueError("fixed-name capacity control requires exactly two heads")
        first = members[0]
        empty_anchor = getattr(bank, "empty_anchor", getattr(first, "empty_anchor", False))
        if not isinstance(empty_anchor, (bool, np.bool_)):
            raise ValueError("empty_anchor must be boolean")
        empty_anchor = bool(empty_anchor)
        for member in members:
            setting = getattr(member, "empty_anchor", empty_anchor)
            if not isinstance(setting, (bool, np.bool_)) or bool(setting) != empty_anchor:
                raise ValueError("all fixed heads must share the same empty_anchor setting")
            if empty_anchor and getattr(member, "mode", "agreement") not in ("agreement", "held"):
                raise ValueError("empty_anchor requires agreement or held agreement memory")
        if empty_anchor and getattr(bank, "mode", "agreement") not in ("agreement", "held"):
            raise ValueError("empty_anchor requires agreement or held agreement memory")
        keys, values, ages = (np.asarray(getattr(first, name))
                              for name in ("keys", "values", "ages"))
        held = np.asarray(first.held)
        if (keys.ndim != 3 or keys.shape[-1] != WIDTH or values.shape != keys.shape
                or held.shape != keys.shape[:2] or held.dtype != np.bool_
                or ages.shape != (keys.shape[1],)
                or any(a.dtype.kind not in "buif" or not np.isfinite(a).all()
                       for a in (keys, values, ages)) or (ages < 0).any()):
            raise ValueError("fixed reads require a finite shared raw19 witnessed bank")
        for member in members[1:]:
            if any(not np.array_equal(getattr(member, name), getattr(first, name))
                   for name in ("keys", "values", "held", "ages")):
                raise ValueError("all fixed heads must share the same witnessed bank")
        keys, values, held, ages = (frozen(keys), frozen(values), frozen(held, bool), frozen(ages))
        owned = []
        for member in members:
            alpha = member.amplitude
            if not np.isfinite(alpha) or alpha < 0:
                raise ValueError("read amplitude must be finite and nonnegative")
            owned.append(_Head(ports(member.query), ports(member.read),
                               keys, values, held, ages, float(alpha), empty_anchor=empty_anchor))
        all_ports = np.concatenate([p for head in owned for p in (head.query, head.read)])
        if len(np.unique(all_ports)) != len(all_ports):
            raise ValueError("fixed heads require disjoint query and read ports")
        batch, slots = held.shape
        selectors = np.zeros((2, batch, slots))
        if slots:
            for name in range(2):
                actual = held & (values[:, :, 0] > 0) & (values[:, :, 5 + name] > 0)
                age = np.where(actual, ages[None, :], np.inf)
                latest = np.argmin(age, axis=1)
                present = actual.any(axis=1)
                # Actual chronology has one latest record. Conflicting ties cannot
                # be resolved by inventing a physical-slot preference.
                for stream in np.flatnonzero(present):
                    tied = actual[stream] & (ages == age[stream, latest[stream]])
                    if not np.all(values[stream, tied] == values[stream, latest[stream]]):
                        raise ValueError("latest named fact has conflicting equal-age records")
                selectors[name, np.flatnonzero(present), latest[present]] = 1.0
        target = np.einsum("hbs,bsr->hbr", selectors, values)
        object.__setattr__(self, "memories", tuple(owned))
        object.__setattr__(self, "p_fixed", frozen(selectors))
        object.__setattr__(self, "targets", frozen(target))
        object.__setattr__(self, "empty_anchor", empty_anchor)

    @property
    def streams(self) -> int:
        return len(self.keys)

    @property
    def bank(self) -> _Bank:
        """Owned immutable descriptors, suitable for ``replace(bank, memories=...)``."""
        return _Bank(self.memories)

    @property
    def keys(self) -> np.ndarray:
        return self.memories[0].keys

    @property
    def values(self) -> np.ndarray:
        return self.memories[0].values

    @property
    def held(self) -> np.ndarray:
        return self.memories[0].held

    @property
    def ages(self) -> np.ndarray:
        return self.memories[0].ages

    def _state(self, value: Any) -> np.ndarray:
        state = np.asarray(value)
        last = max(int(p.max()) for head in self.memories for p in (head.query, head.read))
        if (state.ndim != 2 or len(state) != self.streams or state.shape[1] <= last
                or state.dtype.kind not in "buif" or not np.isfinite(state).all()):
            raise ValueError("activity must match fixed-bank streams and contain all ports")
        return np.asarray(state, dtype=float)

    def weights(self, state: Any) -> np.ndarray:
        self._state(state)
        return self.p_fixed

    def take(self, selection: Any) -> FrozenRead:
        selected = rows(selection, self.streams)
        return FrozenRead(_Bank(tuple(replace(
            head, keys=self.keys[selected], values=self.values[selected], held=self.held[selected],
        ) for head in self.memories)))

    def energy(self, value: Any) -> np.ndarray:
        state = self._state(value)
        occupied = np.ones(self.streams, dtype=bool) if self.empty_anchor else self.held.any(axis=1)
        out = np.zeros(self.streams)
        for index, head in enumerate(self.memories):
            error = state[:, head.read] - self.targets[index]
            out += 0.5 * head.amplitude * np.square(error).sum(axis=1) * occupied
        return out

    def drive(self, value: Any) -> np.ndarray:
        state = self._state(value)
        occupied = (np.ones(self.streams, dtype=bool) if self.empty_anchor
                    else self.held.any(axis=1))[:, None]
        out = np.zeros_like(state)
        for index, head in enumerate(self.memories):
            out[:, head.read] = (head.amplitude
                                 * (self.targets[index] - state[:, head.read]) * occupied)
        return out
