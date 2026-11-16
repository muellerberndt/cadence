"""A composed brain (``cadence.generic.Brain``) in the core: each part it holds under its
attribute's name (``arousal.level``, ...), as that part's module crosses it, and ``live``.
``PARTS`` names the attributes the core holds; the others cross whole only where a part module
exists for them, so a brain with a part the core does not hold yet is refused there by name."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...generic import Brain
from . import part

CLASS = Brain
PARTS = {"arousal": "arousal"}  # attribute -> part module


def state_of(brain: Brain) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attribute, name in PARTS.items():
        held = getattr(brain, attribute, None)
        if held is not None:
            for key, value in part(name).state_of(held).items():
                out[f"{attribute}.{key}"] = value
    return out


def restore(brain: Brain, state: dict[str, Any]) -> None:
    for attribute, name in PARTS.items():
        held = getattr(brain, attribute, None)
        inner = {k[len(attribute) + 1:]: v for k, v in state.items()
                 if k.startswith(attribute + ".")}
        if held is not None and inner:
            part(name).restore(held, inner)


def _live(observations: Any, *, reward: Any = None, done: Any = None, salience: Any = None,
          decision_id: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"observations": np.asarray(observations, dtype=np.float64)}
    for name, value in (("reward", reward), ("done", done), ("salience", salience)):
        if value is not None:
            out[name] = np.asarray(value)
    if decision_id is not None:
        out["decision_id"] = np.int64(decision_id)
    return out


METHODS = {"live": (_live, lambda action: {"action": np.asarray(action, dtype=np.int64)})}
