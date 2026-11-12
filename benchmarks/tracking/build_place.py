"""Write the library's built-in place module, ``src/cadence/modules/place.npz``: the map of things
at its founder settings over the chamber's plane, empty, with a card naming the library version
that wrote it. Run it after the version changes; a test holds the card to the package's version.

    python benchmarks/tracking/build_place.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CARD = {"what": "the map of things (cadence.space.map): 32 x 26 cells of a plane of 2 m by 1.6 m, "
                "16 kinds a unit each; no taught synapse"}


def main() -> int:
    from cadence.space.map import Map

    path = Map((-1.0, 1.0, -0.2, 1.4)).save(ROOT / "src" / "cadence" / "modules" / "place.npz",
                                              card=CARD)
    print(path, path.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
