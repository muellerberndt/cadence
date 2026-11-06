"""Supplied asked-name read for the witnessed continuous raw9 capacity control.

Names occupy key coordinates 0/1, the fact marker 3, and questions 5/6.
The current question selects the latest held fact of that name; its entire
actual stored value is the fixed target. Prefixes and missing facts select the
zero-content null. This is an oracle control, not learned relevance.

Immutable storage, activity validation and E=alpha/2||r-target||^2 reuse the
single-read control. Its raw11 constructor and coordinate interpretation are
never called. Dq=0 and Dr=alpha(target-r), including empty banks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

WIDTH = 9
NAME_A, NAME_B, FACT, ASK_A, ASK_B = 0, 1, 3, 5, 6


def _ports(value: Any) -> np.ndarray:
    index = np.asarray(value)
    if (
        index.shape != (WIDTH,)
        or index.dtype.kind not in "iu"
        or (index < 0).any()
        or (index > np.iinfo(np.int64).max).any()
    ):
        raise ValueError("continuous asked reads require raw9 integer query/read ports")
    owned = np.array(index, dtype=np.int64, copy=True)
    owned.flags.writeable = False
    return owned


def asked_read_class(runtime: Any) -> type:
    """Bind the control's low-level law to the explicitly supplied source tree."""
    path = Path(runtime).resolve() / "benchmarks/attention/single_fixed_read.py"
    name = "_continuous_read_base_" + hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the source-bound single-read control")
    base = importlib.util.module_from_spec(spec)
    sys.modules[name] = base
    spec.loader.exec_module(base)
    frozen, rows, _Head = base.frozen, base.rows, base._Head

    @dataclass(frozen=True, init=False)
    class AskedRead(base.AskedRead):
        """Frozen one-head oracle whose captured question survives row teaching."""

        def __init__(self, bank: Any, observations: Any) -> None:
            members = tuple(bank.memories)
            if len(members) != 1:
                raise ValueError("continuous asked reads require exactly one head")
            source = members[0]
            for descriptor in (bank, source):
                anchor = getattr(descriptor, "empty_anchor", False)
                if not isinstance(anchor, (bool, np.bool_)) or not anchor:
                    raise ValueError("continuous asked reads require explicit empty_anchor=True")
                if getattr(descriptor, "mode", "agreement") not in ("agreement", "held"):
                    raise ValueError("continuous asked reads require agreement or held memory")
            keys, values, ages = (
                np.asarray(getattr(source, name)) for name in ("keys", "values", "ages")
            )
            held = np.asarray(source.held)
            if (
                keys.ndim != 3
                or keys.shape[-1] != WIDTH
                or values.shape != keys.shape
                or held.shape != keys.shape[:2]
                or held.dtype != np.bool_
                or ages.shape != (keys.shape[1],)
                or any(
                    array.dtype.kind not in "buif" or not np.isfinite(array).all()
                    for array in (keys, values, ages)
                )
                or (ages < 0).any()
            ):
                raise ValueError("continuous asked reads require a finite raw9 witnessed bank")
            settings = {}
            for name, default in (
                ("amplitude", 3.0),
                ("temperature", 0.25),
                ("recency", 0.0),
                ("null", 0.0),
            ):
                scalar = getattr(source, name, default)
                if (
                    isinstance(scalar, (bool, np.bool_))
                    or not isinstance(scalar, (int, float, np.integer, np.floating))
                    or not np.isfinite(scalar)
                ):
                    raise ValueError("memory settings must be finite scalars")
                settings[name] = float(scalar)
            if settings["amplitude"] < 0 or settings["temperature"] <= 0 or settings["recency"] < 0:
                raise ValueError("nonnegative amplitude/recency and positive temperature required")
            query, read = _ports(source.query), _ports(source.read)
            if len(np.unique(np.concatenate((query, read)))) != 2 * WIDTH:
                raise ValueError("query and read ports must be unique and disjoint")
            keys, values, held, ages = (
                frozen(keys),
                frozen(values),
                frozen(held, bool),
                frozen(ages),
            )
            owned = _Head(query, read, keys, values, held, ages, **settings)
            observation = np.asarray(observations)
            if (
                observation.shape != (len(keys), WIDTH)
                or observation.dtype.kind not in "buif"
                or not np.isfinite(observation).all()
                or not np.isin(observation[:, ASK_A : ASK_B + 1], (0, 1)).all()
                or (observation[:, ASK_A : ASK_B + 1].sum(axis=1) > 1).any()
            ):
                raise ValueError(
                    "actual observations must be finite raw9 rows with at most one ask"
                )
            names = keys[:, :, NAME_A : NAME_B + 1] > 0
            if (held & names.all(axis=2)).any():
                raise ValueError("an actual held record cannot name both A and B")
            names &= keys[:, :, FACT, None] > 0
            batch, slots = held.shape
            selection = np.zeros((1, batch, slots))
            for stream in range(batch):
                asks = np.flatnonzero(observation[stream, ASK_A : ASK_B + 1])
                if not len(asks):
                    continue
                present = held[stream] & names[stream, :, asks[0]]
                if not present.any():
                    continue
                age = np.where(present, ages, np.inf)
                latest = int(np.argmin(age))
                tied = present & (ages == age[latest])
                if not np.all(values[stream, tied] == values[stream, latest]) or not np.all(
                    keys[stream, tied] == keys[stream, latest]
                ):
                    raise ValueError("latest named fact has conflicting equal-age records")
                selection[0, stream, latest] = 1.0
            self._install(
                (owned,), observation, selection, np.einsum("hbs,bsr->hbr", selection, values)
            )

        def take(self, selection: Any) -> AskedRead:
            index = rows(selection, self.streams)
            head = replace(
                self.memories[0],
                keys=frozen(self.keys[index]),
                values=frozen(self.values[index]),
                held=frozen(self.held[index], bool),
            )
            subset = object.__new__(AskedRead)
            subset._install(
                (head,), self.observations[index], self.p_fixed[:, index], self.targets[:, index]
            )
            return subset

    return AskedRead
