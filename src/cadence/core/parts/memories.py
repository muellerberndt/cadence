"""``cadence.memories`` in the core: an outcome written into the place records
(``record_action``, the function ``Brain._record`` calls with the brain's ``hippocampus``). Its
state is the place records' (``synaptic``). The order the brain reads its memories in
(``Memories.ORDER``) is the core's ``parts::memories::ORDER``; the spine passes the drive
through each member's ``stimulus`` in it."""

from __future__ import annotations

from typing import Any

import numpy as np

from ... import generic
from ...memory import SynapticMemory
from .synaptic import restore, state_of

CLASS = SynapticMemory
OWNERS = {"record_action": generic}  # a function of the module, the memory its first argument
__all__ = ["CLASS", "METHODS", "OWNERS", "restore", "state_of"]


def _record_action(keys: Any, action: Any, reward: Any, importance: Any, known: Any,
                   offsets: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"keys": np.asarray(keys, dtype=np.float64),
                           "action": np.asarray(action, dtype=np.int64),
                           "reward": np.asarray(reward, dtype=np.float64),
                           "importance": np.asarray(importance, dtype=np.float64)}
    if known is not None:
        out["known"] = np.asarray(known, dtype=bool)
    out["offsets"] = np.asarray(offsets, dtype=np.int64).reshape(-1)
    return out


METHODS: dict[str, tuple[Any, Any]] = {"record_action": (_record_action, lambda value: {})}
