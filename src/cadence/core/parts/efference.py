"""``cadence.stream.Efference`` in the core: a working trace written from the issued command
(``issue``), its state a trace's (``trace``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...stream import Efference
from .trace import METHODS as TRACE_METHODS
from .trace import restore, state_of

CLASS = Efference
__all__ = ["CLASS", "METHODS", "restore", "state_of"]


def _issue(command: Any) -> dict[str, Any]:
    return {"command": np.array(command, dtype=np.float64)}


METHODS = {
    **TRACE_METHODS,
    "issue": (_issue, lambda value: {}),
}
