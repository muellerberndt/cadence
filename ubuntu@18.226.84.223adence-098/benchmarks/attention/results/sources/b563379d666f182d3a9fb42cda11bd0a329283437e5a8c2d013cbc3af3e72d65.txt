"""Immutable asked-name oracle for the raw11 single-head capacity control.

The actual current question selects the latest actually held A/B fact by its
raw witnessed key. Prefix events select the zero-content null. This supplied
selector is an oracle control, never learned relevance or an address lesson.
For every stream E = alpha/2 ||r-target||^2, with Dq=0 and Dr=alpha(target-r),
including empty banks. Selection and data stay fixed across all neural phases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

WIDTH = 11


def frozen(value: Any, dtype: Any = float) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def ports(value: Any) -> np.ndarray:
    index = np.asarray(value)
    if (index.shape != (WIDTH,) or index.dtype.kind not in "iu" or (index < 0).any()
            or (index > np.iinfo(np.int64).max).any()):
        raise ValueError("asked reads require raw11 integer query/read ports")
    return frozen(index, np.int64)


def rows(value: Any, batch: int) -> np.ndarray:
    selected = np.asarray(value)
    if selected.ndim != 1:
        raise ValueError("rows require a one-dimensional mask or unique integer indices")
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
    temperature: float
    recency: float
    null: float
    mode: str = "held"
    empty_anchor: bool = True


@dataclass(frozen=True)
class _Bank:
    memories: tuple[_Head, ...]
    mode = "held"
    empty_anchor = True


@dataclass(frozen=True, init=False)
class AskedRead:
    memories: tuple[_Head, ...]
    observations: np.ndarray
    p_fixed: np.ndarray
    targets: np.ndarray
    heads = 1
    mode = "held"
    empty_anchor = True
    fixed_selector = True

    def __init__(self, bank: Any, observations: Any) -> None:
        members = tuple(bank.memories)
        if len(members) != 1:
            raise ValueError("asked-name capacity control requires exactly one head")
        source = members[0]
        for descriptor in (bank, source):
            anchor = getattr(descriptor, "empty_anchor", False)
            if not isinstance(anchor, (bool, np.bool_)) or not anchor:
                raise ValueError("asked reads require explicit empty_anchor=True")
            if getattr(descriptor, "mode", "agreement") not in ("agreement", "held"):
                raise ValueError("asked reads require agreement or held agreement memory")
        keys, values, ages = (np.asarray(getattr(source, name))
                              for name in ("keys", "values", "ages"))
        held = np.asarray(source.held)
        if (keys.ndim != 3 or keys.shape[-1] != WIDTH or values.shape != keys.shape
                or held.shape != keys.shape[:2] or held.dtype != np.bool_
                or ages.shape != (keys.shape[1],)
                or any(a.dtype.kind not in "buif" or not np.isfinite(a).all()
                       for a in (keys, values, ages)) or (ages < 0).any()):
            raise ValueError("asked reads require a finite raw11 witnessed bank")
        settings = {}
        for name, default in (("amplitude", 3.0), ("temperature", 0.25),
                              ("recency", 0.0), ("null", 0.0)):
            scalar = getattr(source, name, default)
            if (isinstance(scalar, (bool, np.bool_))
                    or not isinstance(scalar, (int, float, np.integer, np.floating))
                    or not np.isfinite(scalar)):
                raise ValueError("memory settings must be finite scalars")
            settings[name] = float(scalar)
        if (settings["amplitude"] < 0 or settings["temperature"] <= 0
                or settings["recency"] < 0):
            raise ValueError("nonnegative amplitude/recency and positive temperature required")
        query, read = ports(source.query), ports(source.read)
        if len(np.unique(np.concatenate((query, read)))) != 2 * WIDTH:
            raise ValueError("query and read ports must be unique and disjoint")
        keys, values, held, ages = (frozen(keys), frozen(values), frozen(held, bool), frozen(ages))
        owned = _Head(query, read, keys, values, held, ages, **settings)
        observation = np.asarray(observations)
        if (observation.shape != (len(keys), WIDTH) or observation.dtype.kind not in "buif"
                or not np.isfinite(observation).all()
                or not np.isin(observation[:, 5:7], (0, 1)).all()
                or (observation[:, 5:7].sum(axis=1) > 1).any()):
            raise ValueError("actual observations must be finite raw11 rows with at most one ask")
        actual_names = np.stack((np.any(keys[:, :, :2] > 0, axis=2),
                                 np.any(keys[:, :, 2:4] > 0, axis=2)), axis=2)
        if (held & actual_names.all(axis=2)).any():
            raise ValueError("an actual held record cannot name both A and B")
        batch, slots = held.shape
        selection = np.zeros((1, batch, slots))
        if slots:
            for stream in range(batch):
                asks = np.flatnonzero(observation[stream, 5:7])
                if not len(asks):
                    continue
                present = held[stream] & actual_names[stream, :, asks[0]]
                if not present.any():
                    continue
                age = np.where(present, ages, np.inf)
                latest = int(np.argmin(age))
                tied = present & (ages == age[latest])
                if (not np.all(values[stream, tied] == values[stream, latest])
                        or not np.all(keys[stream, tied] == keys[stream, latest])):
                    raise ValueError("latest named fact has conflicting equal-age records")
                selection[0, stream, latest] = 1.0
        self._install((owned,), observation, selection,
                      np.einsum("hbs,bsr->hbr", selection, values))

    def _install(self, members: tuple[_Head, ...], observation: Any,
                 selection: Any, targets: Any) -> None:
        object.__setattr__(self, "memories", members)
        object.__setattr__(self, "observations", frozen(observation))
        object.__setattr__(self, "p_fixed", frozen(selection))
        object.__setattr__(self, "targets", frozen(targets))

    @property
    def bank(self) -> _Bank:
        return _Bank(self.memories)

    @property
    def streams(self) -> int:
        return len(self.keys)

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
        head = self.memories[0]
        last = max(int(head.query.max()), int(head.read.max()))
        if (state.ndim != 2 or len(state) != self.streams or state.shape[1] <= last
                or state.dtype.kind not in "buif" or not np.isfinite(state).all()):
            raise ValueError("activity must match the bank streams and contain both ports")
        return np.asarray(state, dtype=float)

    def weights(self, state: Any) -> np.ndarray:
        self._state(state)
        return self.p_fixed

    def take(self, selection: Any) -> AskedRead:
        index = rows(selection, self.streams)
        head = replace(self.memories[0], keys=frozen(self.keys[index]),
                       values=frozen(self.values[index]), held=frozen(self.held[index], bool))
        subset = object.__new__(AskedRead)
        # Carry the captured selector directly; row teaching never asks a new question.
        subset._install((head,), self.observations[index], self.p_fixed[:, index],
                        self.targets[:, index])
        return subset

    def energy(self, value: Any) -> np.ndarray:
        state = self._state(value)
        head = self.memories[0]
        return 0.5 * head.amplitude * np.square(state[:, head.read] - self.targets[0]).sum(1)

    def drive(self, value: Any) -> np.ndarray:
        state = self._state(value)
        head = self.memories[0]
        out = np.zeros_like(state)
        out[:, head.read] = head.amplitude * (self.targets[0] - state[:, head.read])
        return out
