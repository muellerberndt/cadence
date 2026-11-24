"""The forager chamber's gates, read from the lives' results (the declared protocol in README.md).

    python benchmarks/forager/summary.py results.json [more.json ...]

A gate holds or fails on the declared panel: every declared founder (``DECLARED``) with its
``code`` and ``both`` lives, counted against the bar of ten. Until each of them is given, the
gates read ``incomplete`` and their counts are those of the pairs given, as for a development
run. Founders outside the declared panel stay in the tables and are counted in no verdict. A
founder's arm is read once: a second life of it is refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

OF = 10  # founders of twelve
DECLARED = tuple(range(1011100, 1011112))  # the twelve founders the gates are declared on
KEYS = ("taught", "own", "eaten", "teacher", "turn_observation", "turn_code", "turn_state",
        "turn_newborn")


def gates(both: dict[str, Any], code: dict[str, Any]) -> dict[str, bool]:
    """One founder's gates: its ``both`` life's mean readings against its ``code`` life's."""
    return {
        "F1": both["turn_state"] > code["turn_state"],
        "F2": both["teacher"] > code["teacher"],
    }


# Readings without a bar: the founders on which ``both`` reads above ``code``
UNBARRED = ("own", "eaten", "taught")


def main(paths: list[str]) -> int:
    lives: list[dict[str, Any]] = []
    by: dict[int, dict[str, dict[str, Any]]] = {}
    for path in paths:
        for life in json.loads(Path(path).read_text()):
            given = by.setdefault(life["founder"], {})
            if life["arm"] in given:
                raise SystemExit(f"{path}: founder {life['founder']} already has a life of the "
                                 f"arm {life['arm']!r}; a founder's arm is read once")
            given[life["arm"]] = life
            lives.append(life)
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
    missing = [f for f in DECLARED if f not in paired]
    outside = [f for f in paired if f not in DECLARED]
    if not missing:  # the declared panel is given: its founders alone are counted
        paired = [f for f in paired if f in DECLARED]
    held = {name: sum(gates(by[f]["both"]["mean"], by[f]["code"]["mean"])[name]
                      for f in paired) for name in ("F1", "F2")}
    print()
    for name, count in held.items():
        verdict = "incomplete" if missing else "holds" if count >= OF else "fails"
        print(f"{name}: {count} of {len(paired)} -> {verdict}")
    if missing:
        print(f"incomplete: the gates are declared on founders {DECLARED[0]} to "
              f"{DECLARED[-1]}, each with its code and both lives; not given: "
              + ", ".join(str(f) for f in missing))
    elif outside:
        print("not counted, outside the declared founders: "
              + ", ".join(str(f) for f in outside))
    for key in UNBARRED:
        above = sum(by[f]["both"]["mean"][key] > by[f]["code"]["mean"][key] for f in paired)
        print(f"{key} (no bar): both above code on {above} of {len(paired)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
