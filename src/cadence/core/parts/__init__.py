"""The parts of a brain the native core holds, one module each: a part's state as named arrays
(``state_of``) and back onto its Python object (``restore``), and how each of its methods'
arguments and results cross (``METHODS``). ``kernels/core/check/replay.py`` records a part's
calls in real lives and replays each in the core against what the Python did.

A part module defines ``CLASS`` (the Python class it stands for), ``state_of(obj) -> dict``,
``restore(obj, state)``, and ``METHODS = {name: (arguments, result)}``: ``arguments(*args,
**kwargs) -> dict`` names a call's arguments as the core reads them, ``result(value) -> dict``
names what the call returned.
"""

from __future__ import annotations

import importlib
from types import ModuleType

NAMES = ("arousal", "bound", "brain", "code", "efference", "graph", "hearing", "history",
         "longterm", "map", "memories", "orienting", "recall", "surprise", "synaptic", "trace")


def part(name: str) -> ModuleType:
    """The module of the part ``name``."""
    if name not in NAMES:
        raise KeyError(f"no part {name!r} in the core: one of {NAMES}")
    return importlib.import_module(f"{__name__}.{name}")
