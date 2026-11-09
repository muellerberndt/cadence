"""The candidate of cadence#288 by its declared name: a long-term bank of records under a fixed
budget of slots.

The declared runs read this module as commit ``741161ba`` holds it. The library holds the same
store as ``cadence.LongTermMemory`` (``src/cadence/longterm.py``, where the laws are written
out): a sparse address as the coarse stage, the recall memory's read as the fine one,
replacement by depth and recent use, revision where a key is told again. ``Bank`` is that class
with the coarse stage at every size (``direct=0``), as the protocol declares its arms; read for
read and write for write it gives what the declared module gave. A plan that names
``"direct": null`` reads the library's own founder instead, which scores every slot up to twice
the address's units.
"""

from __future__ import annotations

from dataclasses import dataclass

from cadence.longterm import LongTermMemory
from cadence.longterm import LongTermRead as Read

ORDER = LongTermMemory.ORDER  # the order of the depth's law
FLOOR = LongTermMemory.FLOOR  # the recent use below which a slot counts as idle

__all__ = ["FLOOR", "ORDER", "Bank", "Read"]


@dataclass
class Bank(LongTermMemory):
    direct: int | None = 0
