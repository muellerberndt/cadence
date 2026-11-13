"""The audit's readings as tables (cadence#326): each arm or probe of a script, its shares by
gate pooled over the founders whose files are in the folder.

    python read.py <out> [script]

A cell is ``ok`` (the world's answer), with ``follows`` (the answer the change predicts),
``refused`` ("I do not know") and ``same`` (the brain's own answer to the unchanged question)
where the probe has them; each the mean over founders, and its least founder in brackets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

COLUMNS = ["W1", "W1_named_only", "W2", "W3", "W4_never_seen", "W4_never_taught", "W5_by_colour",
           "W5_by_place"]


def cell(values: list[float]) -> str:
    if not values:
        return ""
    return f"{np.mean(values):.3f} ({min(values):.2f})"


def table(lives: list[dict[str, Any]], measure: str, columns: list[str]) -> list[str]:
    arms = list(lives[0]["arms"])
    lines = ["| arm | " + " | ".join(columns) + " |", "|" + " --- |" * (len(columns) + 1)]
    for arm in arms:
        row = [arm]
        for name in columns:
            values = [life["arms"][arm]["shares"][name][measure] for life in lives
                      if name in life["arms"][arm]["shares"]
                      and life["arms"][arm]["shares"][name][measure] is not None]
            row.append(cell(values))
        if any(row[1:]):
            lines.append("| " + " | ".join(row) + " |")
    return lines


def read(out: Path, script: str | None = None) -> None:
    scripts: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out.glob("*.json")):
        name = path.stem.rsplit("_", 1)[0]
        if script is None or name == script:
            scripts.setdefault(name, []).append(json.loads(path.read_text()))
    for name, lives in scripts.items():
        if "arms" not in lives[0]:
            continue
        seeds = [life["seed"] for life in lives]
        print(f"\n## {name}: {len(lives)} founders ({min(seeds)} to {max(seeds)})")
        gates = sorted({g for life in lives for arm in life["arms"].values()
                        for g in arm["shares"]})
        extra = [g for g in gates if g not in COLUMNS and g not in ("W4", "W5", "W1_practised",
                                                                    "W2_where", "W2_colour",
                                                                    "W3_colour", "W3_by_colour",
                                                                    "W3_where", "W4_out_of_view")]
        for measure in ("ok", "follows", "refused", "same"):
            lines = table(lives, measure, COLUMNS + extra)
            if len(lines) > 2:
                print(f"\n{measure}:")
                print("\n".join(lines))
        for life in lives[:1]:
            if "after" in life:
                print("\nafter (first founder):", json.dumps(life["after"], default=float)[:3000])


if __name__ == "__main__":
    read(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
