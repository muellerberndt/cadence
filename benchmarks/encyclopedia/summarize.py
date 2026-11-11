"""The encyclopedia's tables (cadence#311): the gates read on the brain's rows and the controls'
rows, said and found apart, and each learner's costs.

    python benchmarks/encyclopedia/summarize.py --brain rows.jsonl --controls controls.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

GROUPS = ("trained", "held", "synonym", "two", "revised", "stale", "erased")
# gate: (size, group, bar, above): a founder passes where its reading is at or above the bar
# (at or below it where ``above`` is False)
GATES = {
    "E1 trained": (10000, "trained", 0.90, True),
    "E1 held": (10000, "held", 0.80, True),
    "E3 revised": (10000, "revised", 0.95, True),
    "E3 stale": (10000, "stale", 0.02, False),
    "E4 two facts": (10000, "two", 0.80, True),
}
PASSING = 10  # founders of twelve


def rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def right(row: dict[str, Any], group: str) -> float | None:
    score = row["scores"].get(group)
    return None if score is None else float(score["right"])


def span(values: list[float]) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    return f"{low:.3f}" if abs(high - low) < 5e-4 else f"{low:.3f} to {high:.3f}"


def gates(brain: list[dict[str, Any]], arm: str) -> list[str]:
    out = ["| Gate | Bar | Said, per founder | Found | Founders passing |",
           "| --- | --- | --- | --- | --- |"]
    for name, (size, group, bar, above) in GATES.items():
        lives = [r for r in brain if r["arm"] == arm and r["size"] == size]
        said = [v for v in (right(r, group) for r in lives) if v is not None]
        found = [v for v in (right(r, f"found {group}") for r in lives) if v is not None]
        passing = sum((v >= bar - 1e-9) if above else (v < bar) for v in said)
        out.append(f"| {name} | {'at least' if above else 'under'} {bar:.2f} | {span(said)} | "
                   f"{span(found)} | {passing} of {len(said)} |")
    # E2: the larger corpus within .05 of the same founder's 10,000
    by = {(r["founder"], r["size"]): r for r in brain if r["arm"] == arm}
    for group in ("trained", "held"):
        gaps = [abs((right(by[(f, s)], group) or 0) - (right(by[(f, 10000)], group) or 0))
                for (f, s) in by if s > 10000 and (f, 10000) in by]
        passing = sum(g <= 0.05 + 1e-9 for g in gaps)
        out.append(f"| E2 {group} (larger corpus) | within .05 | gap {span(gaps)} | | "
                   f"{passing} of {len(gaps)} |")
    return out


def table(brain: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[str]:
    head = ("| Learner | Size | Parameters | Memory bytes | Seconds a fact taken in | "
            "MACs a fact taken in | ms an answer | MACs an answer | "
            + " | ".join(GROUPS) + " |")
    out = [head, "| " + " | ".join(["---"] * (8 + len(GROUPS))) + " |"]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in brain:
        groups[(r["arm"], r["size"])].append(r)
    for r in controls:
        groups[(r.get("arm", r["model"]), r["size"])].append(r)
    for (name, size), lives in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        counts = sorted({int(r["parameters"]) for r in lives})
        params = (f"{counts[0]:,}" if len(counts) == 1
                  else f"{counts[0]:,} to {counts[-1]:,}")
        memory = int(np.median([r.get("record_bytes") or r.get("memory_bytes") or 0
                                for r in lives]))
        intake = np.median([r.get("intake_seconds_per_fact", r.get("intake_seconds", 0.0))
                            or 0.0 for r in lives])
        intake_macs = int(np.median([r.get("intake_macs_per_fact", r.get("intake_macs", 0))
                                     or 0 for r in lives]))
        cells = [span([v for v in (right(r, g) for r in lives) if v is not None])
                 for g in GROUPS]
        out.append(f"| {name} | {size:,} | {params} | {memory:,} | {intake:.4f} | "
                   f"{intake_macs:,} | {np.median([r['answer_ms'] for r in lives]):.2f} | "
                   f"{int(np.median([r['answer_macs'] for r in lives])):,} | "
                   + " | ".join(cells) + " |")
    return out


def parity(brain: list[dict[str, Any]], controls: list[dict[str, Any]], arm: str,
           size: int = 10000) -> list[str]:
    """E5: per founder, the brain's said reading against the best control's on the same
    questions."""
    out = ["| Reading | Brain at or above the best control | Best control per founder |",
           "| --- | --- | --- |"]
    for group in ("trained", "held", "revised", "two"):
        wins, best_names = 0, []
        founders = sorted({r["founder"] for r in brain if r["arm"] == arm and r["size"] == size})
        for f in founders:
            mine = [right(r, group) for r in brain
                    if r["arm"] == arm and r["size"] == size and r["founder"] == f]
            theirs = [(right(r, group) or 0.0, r.get("arm", r["model"])) for r in controls
                      if r["size"] == size and r["founder"] == f]
            if not mine or mine[0] is None or not theirs:
                continue
            top = max(theirs)
            best_names.append(top[1])
            wins += int(mine[0] >= top[0] - 1e-9)
        out.append(f"| {group} | {wins} of {len(best_names)} | "
                   f"{', '.join(sorted(set(best_names)))} |")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--brain", type=Path, nargs="+", required=True)
    parser.add_argument("--controls", type=Path, nargs="*", default=[])
    parser.add_argument("--arm", default="records")
    args = parser.parse_args(argv)
    brain = [r for p in args.brain for r in rows(p)]
    controls = [r for p in args.controls for r in rows(p)]
    print("\n".join(gates(brain, args.arm)))
    print()
    print("\n".join(table(brain, controls)))
    if controls:
        print()
        print("\n".join(parity(brain, controls, args.arm)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
