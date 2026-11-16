"""The forager chamber's gates, read from the lives' results (the declared protocol in README.md).

    python benchmarks/forager/summary.py results.json [more.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

OF = 10  # founders of twelve
KEYS = ("taught", "own", "eaten", "teacher", "turn_observation", "turn_code", "turn_state",
        "turn_newborn")


def gates(both: dict[str, Any], code: dict[str, Any]) -> dict[str, bool]:
    """One founder's gates: its ``both`` life's mean readings against its ``code`` life's."""
    return {
        "F1": both["turn_state"] > code["turn_state"],
        "F2": both["teacher"] > code["teacher"],
        "F3": both["eaten"] > code["eaten"],
    }


def main(paths: list[str]) -> int:
    lives = [life for path in paths for life in json.loads(Path(path).read_text())]
    by: dict[int, dict[str, dict[str, Any]]] = {}
    for life in lives:
        by.setdefault(life["founder"], {})[life["arm"]] = life
    print("| founder | arm | " + " | ".join(KEYS) + " | s |")
    print("|" + " --- |" * (len(KEYS) + 3))
    for founder in sorted(by):
        for arm in ("code", "both", "observation", "none"):
            if arm in by[founder]:
                life = by[founder][arm]
                print(f"| {founder} | {arm} | "
                      + " | ".join(f"{life['mean'][k]:.3f}" for k in KEYS)
                      + f" | {life['seconds']} |")
    print()
    arms = sorted({life["arm"] for life in lives})
    for arm in arms:
        means = [life["mean"] for life in lives if life["arm"] == arm]
        print(f"{arm} ({len(means)}): "
              + ", ".join(f"{k} {np.mean([m[k] for m in means]):.3f}" for k in KEYS))
    paired = [f for f in by if {"both", "code"} <= set(by[f])]
    if paired:
        held = {name: sum(gates(by[f]["both"]["mean"], by[f]["code"]["mean"])[name]
                          for f in paired) for name in ("F1", "F2", "F3")}
        print()
        for name, count in held.items():
            print(f"{name}: {count} of {len(paired)} -> "
                  f"{'holds' if count >= OF * len(paired) / 12 else 'fails'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
