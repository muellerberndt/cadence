"""The symbols chamber's gates, read from the lives' results (the declared protocol in README.md).

    python benchmarks/symbols/summary.py results.json [more.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BUDGET = 80
OF = 10


def gates(life: dict[str, Any]) -> dict[str, bool]:
    y2, y3, n = life["Y2"], life["Y3"], life["N"]
    learned = lambda r: all(x is not None and x <= BUDGET for x in r["exchanges"])  # noqa: E731
    return {
        "S1": y2["records"] >= 0.90,
        "S2": y2["state"] >= 0.90,
        "S3": y2["barred"] <= 0.25 and y2["dealt"] <= 0.25,
        "S4": learned(y3["names_paid"]) and y3["names_unpaid"]["right"] <= 0.15,
        "S5": learned(y3["colours_paid"]) and y3["colours_unpaid"]["right"] <= 0.15,
        "S6": y3["names_paid"]["held_out"] >= 0.90,
        "S7": (n["taught"]["in view"] >= 0.90 and n["taught"]["behind a screen"] >= 0.90
               and n["untaught"]["in view"] <= 0.10 and n["untaught"]["behind a screen"] <= 0.10),
    }


def main(paths: list[str]) -> int:
    lives = [life for path in paths for life in json.loads(Path(path).read_text())]
    lives.sort(key=lambda life: life["founder"])
    rows = []
    for life in lives:
        y2, y3, n = life["Y2"], life["Y3"], life["N"]
        rows.append((life["founder"], y2["records"], y2["state"], y2["barred"], y2["dealt"],
                     y3["names_paid"]["exchanges"], y3["colours_paid"]["exchanges"],
                     round(y3["names_unpaid"]["right"], 3), round(y3["colours_unpaid"]["right"], 3),
                     y3["names_paid"]["held_out"], n["taught"]["in view"],
                     n["taught"]["behind a screen"], n["untaught"]["in view"],
                     life["Y1"]["named"], life["seconds"]))
    print("| founder | S1 | S2 | barred | dealt | names (exchanges) | colours (exchanges) | "
          "unpaid names | unpaid colours | S6 | S7 in view | S7 screen | untaught | Y1h | s |")
    print("|" + " --- |" * 15)
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    held = {name: sum(gates(life)[name] for life in lives) for name in gates(lives[0])}
    print()
    for name, count in held.items():
        print(f"{name}: {count} of {len(lives)} -> {'holds' if count >= OF else 'fails'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
