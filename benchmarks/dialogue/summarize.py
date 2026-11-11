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

COLUMNS = ("all", "gap 0", "gap 1", "gap 2-4", "gap 5-9", "gap 10+", "anchored", "held pair",
           "revised")
DECLARED = (3050, 3051, 3052, 3053, 3054, 3055)  # the declared run's founders
NEEDED = 5


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
    parts = [reading["name"]] if reading.get("name") else []
    parts += [reading.get("wording", "")] if reading.get("wording") else []
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


def score(found: list[dict[str, Any]], arm: str, reading: str, group: str,
          field: str = "right") -> list[float | None]:
    """``group``'s score of ``arm`` in the reading named ``reading``, per declared founder."""
    out: list[float | None] = []
    for founder in DECLARED:
        rows_ = [r for r in found if r.get("arm") == arm and r["founder"] == founder
                 and r.get("reading", {}).get("name") == reading
                 and r.get("chamber") == "synthetic"]
        held = rows_[0]["scores"].get(group) if len(rows_) == 1 else None
        out.append(None if held is None else held.get(field))
    return out


def gates(found: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The declared run's gates (``PROTOCOL.md``, the declared run) on the ``turns`` arm."""
    out: dict[str, dict[str, Any]] = {}

    def gate(name: str, values: list[Any], passes: list[bool], bar: str) -> None:
        out[name] = {"values": values, "bar": bar, "founders": int(sum(passes)),
                     "passes": int(sum(passes)) >= NEEDED}

    told = score(found, "turns", "R1", "held pair")
    gate("T1", told, [v is not None and v >= 0.90 for v in told], "at least .90")
    worded = score(found, "turns", "R3", "all")
    gate("T2", worded, [v is not None and v >= 0.80 for v in worded], "at least .80")
    far = score(found, "turns", "R2", "anchored")
    revised = score(found, "turns", "R2", "revised")
    stale = score(found, "turns", "R2", "stale", "share")
    gate("T3", far, [a is not None and b is not None and c is not None
                     and a >= 0.90 and b >= 0.90 and c <= 0.02
                     for a, b, c in zip(far, revised, stale, strict=True)],
         "ten sessions back at least .90; revised at least .90 with stale at most .02")
    emptied = score(found, "turns", "R5", "anchored")
    control = score(found, "previous_word", "R2", "anchored")
    gate("T4", emptied, [a is not None and b is not None and a <= b + 0.05
                         for a, b in zip(emptied, control, strict=True)],
         "at most the previous-word control's reading plus .05")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    found = []
    for path in sorted(args.dir.glob("*.jsonl*")):
        found += rows(path.with_name(path.name.removesuffix(".gz")))
    print(table([r for r in found if "scores" in r]))
    if any(r.get("founder") in DECLARED for r in found):
        print(json.dumps(gates(found), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
