"""``cadence.bound.BoundRecords`` in the core: its fields, its store (``store.``), the moment a
reward pays, each record's pay and refusal, the babbles drawn and where each said sound came
from.

The fields cross by name (``field:<name>``, their columns), the voice's field and the heard
field by their place among them. ``named`` takes the code of the row's heard sense alone, which
the caller's ``code_of`` gives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...bound import BoundRecords
from . import longterm

CLASS = BoundRecords
FROM = {None: -1, "said": 0, "heard": 1, "babble": 2}
FROM_CODE = {code: source for source, code in FROM.items()}


def _places(records: BoundRecords) -> tuple[int, int]:
    names = list(records.fields)
    voice = names.index(records._voice_field()) if records.voice is not None else -1
    heard = names.index(records.heard) if records.heard is not None else -1
    return voice, heard


def state_of(records: BoundRecords) -> dict[str, Any]:
    voice, heard = _places(records)
    out: dict[str, Any] = {
        "config.capacity": np.int64(records.capacity), "config.seed": np.int64(records.seed),
        "config.inputs": np.int64(records.inputs), "config.voice": np.int64(voice),
        "config.heard": np.int64(heard),
    }
    for name, (first, last) in records.fields.items():
        out[f"field:{name}"] = np.array([first, last], dtype=np.int64)
    out.update({f"store.{k}": v for k, v in longterm.state_of(records.store).items()})
    out["written"] = np.array(records.written, dtype=np.int64)
    out["refused"] = np.array(records.refused, dtype=bool)
    out["paid"] = np.array(records.paid, dtype=np.float64)
    out["draws"] = np.int64(records.draws)
    out["said_from"] = np.array([FROM[s] for s in records.said_from], dtype=np.int64)
    return out


def restore(records: BoundRecords, state: Mapping[str, Any]) -> None:
    """The records onto bound records of the same fields and settings (refused otherwise)."""
    for name, value in state_of(records).items():
        if (name.startswith(("config.", "field:"))) and not np.array_equal(
                np.asarray(state[name]), value):
            raise ValueError(f"{name}: bound records of other settings")
    longterm.restore(records.store, {k[len("store."):]: v for k, v in state.items()
                                     if k.startswith("store.")})
    records.written = np.array(state["written"], dtype=np.int64)
    records.refused = np.array(state["refused"], dtype=bool)
    records.paid = np.array(state["paid"], dtype=np.float64)
    records.draws = int(state["draws"])
    records.said_from = [FROM_CODE[int(code)] for code in np.asarray(state["said_from"])]


def _rows(rows: Any, codes: Any) -> dict[str, Any]:
    return {"rows": np.asarray(rows, dtype=np.float64),
            "codes": np.asarray(codes, dtype=np.float64)}


def _hear(rows: Any, codes: Any, surprise: Any = None, acted: bool = True) -> dict[str, Any]:
    out = _rows(rows, codes)
    if surprise is not None:
        out["surprise"] = np.asarray(surprise, dtype=np.float64).reshape(-1)
    out["acted"] = np.bool_(acted)
    return out


def _named(row: Any, code_of: Any) -> dict[str, Any]:
    """The row and the code of its heard sense alone (``code_of`` is the brain's
    ``_bound_keyed``; the records are the brain's)."""
    row = np.asarray(row, dtype=np.float64)
    out: dict[str, Any] = {"row": row}
    records = getattr(getattr(code_of, "__self__", None), "bound", None)
    if records is not None and records.heard is not None:
        first, last = records.fields[records.heard]
        if np.any(row[first:last]):
            alone = np.zeros_like(row)
            alone[first:last] = row[first:last]
            out["code"] = np.asarray(code_of(alone[None]), dtype=np.float64)[0]
    return out


def _salience(template: Mapping[str, Any], things: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"things": np.int64(len(things))}
    for field, part in template.items():
        out[f"template:{field}"] = np.asarray(part, dtype=np.float64)
    for k, thing in enumerate(things):
        for name, held in (thing.get("codes") or {}).items():
            out[f"thing:{k}:{name}"] = np.asarray(held, dtype=np.float64)
    return out


def _fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: np.asarray(part, dtype=np.float64) for name, part in value.items()}


METHODS = {
    "hear": (_hear, lambda value: {}),
    "pay": (lambda reward: {"reward": np.asarray(reward, dtype=np.float64).reshape(-1)},
            lambda value: {}),
    "complete": (_rows, lambda value: {"rows": np.asarray(value, dtype=np.float64)}),
    "recall": (lambda row, code: {"row": np.asarray(row, dtype=np.float64),
                                  "code": np.asarray(code, dtype=np.float64)}, _fields),
    "say": (_rows, lambda value: {"said": np.asarray(value, dtype=np.float64)}),
    "named": (_named, _fields),
    "salience": (_salience, lambda value: {"salience": np.asarray(value, dtype=np.float64)}),
}
