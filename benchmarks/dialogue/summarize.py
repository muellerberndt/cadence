"""The conversation chamber's tables from its receipts (cadence#305): each arm's answers by
reading, as the range over founders.

    python benchmarks/dialogue/summarize.py --dir benchmarks/dialogue/results/step0
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

COLUMNS = ("all", "gap 0", "gap 1", "gap 2-4", "gap 5-9", "gap 10+", "held pair", "revised")


def rows(path: Path) -> list[dict[str, Any]]:
    packed = path.with_name(path.name + ".gz")
    text = gzip.decompress(packed.read_bytes()).decode() if packed.exists() else path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def cell(v: float) -> str:
    return f"{v:.3f}".lstrip("0") if v < 1 else "1.000"


def span(values: list[float]) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    return cell(low) if low == high else f"{cell(low)} to {cell(high)}"


def label(row: dict[str, Any]) -> str:
    who = row.get("arm") or f"{row['model']} {row['mode']}"
    reading = row.get("reading", {})
    parts = [reading.get("wording", "")] if reading.get("wording") else []
    if reading.get("erase"):
        parts.append(f"{reading['erase']} emptied")
    return f"`{who}`" + (f", {', '.join(parts)}" if parts else "")


def table(found: list[dict[str, Any]], columns: tuple[str, ...] = COLUMNS) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in found:
        groups.setdefault((row["chamber"], label(row)), []).append(row)
    head = ["Chamber", "Arm, reading", "Parameters", *columns, "stale"]
    lines = ["| " + " | ".join(head) + " |", "| ---" * len(head) + " |"]
    for (chamber, name), group in groups.items():
        cells = [span([r["scores"][c]["right"] for r in group if c in r["scores"]])
                 for c in columns]
        stale = span([r["scores"]["stale"]["share"] for r in group
                      if r["scores"]["stale"]["share"] is not None])
        counts = sorted({int(r["parameters"]) for r in group})
        params = f"{counts[0]:,}" + (f" to {counts[-1]:,}" if len(counts) > 1 else "")
        lines.append(f"| {chamber} | {name} | {params} | " + " | ".join(cells) + f" | {stale} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    found = []
    for path in sorted(args.dir.glob("*.jsonl*")):
        found += rows(path.with_name(path.name.removesuffix(".gz")))
    print(table([r for r in found if "scores" in r]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
