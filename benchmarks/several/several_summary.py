"""The several-records chamber's tables (cadence#325) from the rows of ``several_chamber.py``
and ``several_controls.py``: each reading as the range over the founders, the founders at a
bar, and every learner's parameters and work.

    python benchmarks/several/several_summary.py --rows rows.jsonl [--controls controls.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

GROUPS = ("parts 1", "parts 2", "parts 4", "parts 8", "chain 2", "chain 3")


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spread(values: list[float]) -> str:
    """A reading over the founders: one number where they agree, else its range."""
    if not values:
        return ""
    low, high = min(values), max(values)
    return f"{low:.3f}".lstrip("0") if high - low < 5e-4 else (
        f"{low:.3f}".lstrip("0") + " to " + f"{high:.3f}".lstrip("0"))


def reading(rows: list[dict[str, Any]], name: str) -> list[float]:
    return [r["scores"][name]["right"] for r in rows if name in r.get("scores", {})]


def at_bar(values: list[float], bar: float) -> str:
    return f"{sum(v >= bar - 1e-9 for v in values)} of {len(values)}"


def table(rows: list[dict[str, Any]], kinds: tuple[str, ...], groups: tuple[str, ...] = GROUPS,
          ) -> list[str]:
    """One line per arm and size: each kind of reading for each group of questions."""
    by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["arm"], int(r["size"]))].append(r)
    out = ["| Arm | Records | Founders | Parameters | Reading | " + " | ".join(groups) + " |",
           "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in groups) + " |"]
    for (arm, size), cell in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for kind in kinds:
            values = [spread(reading(cell, f"{kind} {g}")) for g in groups]
            if not any(values):
                continue
            out.append(f"| `{arm}` | {size:,} | {len(cell)} | {cell[0]['parameters']:,} | "
                       f"{kind} | " + " | ".join(values) + " |")
    return out


def costs(rows: list[dict[str, Any]], groups: tuple[str, ...] = GROUPS) -> list[str]:
    """Each arm's work per answer: milliseconds, the read's own, and multiply-accumulates."""
    by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("costs"):
            by[(r["arm"], int(r["size"]))].append(r)
    out = ["| Arm | Records | Reading | " + " | ".join(groups) + " |",
           "| --- | --- | --- | " + " | ".join("---" for _ in groups) + " |"]
    for (arm, size), cell in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for name, form in (("answer_ms", "{:.2f}"), ("read_ms", "{:.3f}"),
                           ("answer_macs", "{:.2e}"), ("read_macs", "{:.0f}")):
            values = []
            for g in groups:
                got = [r["costs"][g][name] for r in cell if g in r["costs"]
                       and name in r["costs"][g]]
                values.append(form.format(sum(got) / len(got)) if got else "")
            out.append(f"| `{arm}` | {size:,} | {name} | " + " | ".join(values) + " |")
    return out


def absent(rows: list[dict[str, Any]]) -> list[str]:
    """Silence and refusal: the parts no record answers and the questions no record answers."""
    by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["arm"], int(r["size"]))].append(r)
    names = [f"{kind} {g} absent" for kind in ("said", "silent", "refused") for g in GROUPS]
    held = [n for n in names if any(reading(cell, n) for cell in by.values())]
    out = ["| Arm | Records | " + " | ".join(held) + " |",
           "| --- | --- | " + " | ".join("---" for _ in held) + " |"]
    for (arm, size), cell in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        out.append(f"| `{arm}` | {size:,} | "
                   + " | ".join(spread(reading(cell, n)) for n in held) + " |")
    return out


def pooled(row: dict[str, Any], names: list[str]) -> float | None:
    """One founder's reading pooled over the groups ``names``, each by its questions."""
    held = [row["scores"][n] for n in names if n in row["scores"]]
    total = sum(h["questions"] for h in held)
    return sum(h["right"] * h["questions"] for h in held) / total if total else None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def gates(rows: list[dict[str, Any]], arm: str = "heads", size: int = 100_000,
          birth: str = "birth", need: int = 10) -> list[str]:
    """The declared gates A1 to A6 read on ``arm`` at ``size`` records (``PROTOCOL.md``)."""
    cell = sorted((r for r in rows if r["arm"] == arm and int(r.get("size", 0)) == size),
                  key=lambda r: r["founder"])
    out = ["| Gate | Reading | Bar | Founders at the bar | Range | Verdict |",
           "| --- | --- | --- | --- | --- | --- |"]

    def line(gate: str, name: str, values: list[float], bar: float) -> bool:
        at = sum(v >= bar - 1e-9 for v in values)
        passed = at >= min(need, len(values)) and bool(values)
        out.append(f"| {gate} | {name} | {bar:.2f} | {at} of {len(values)} | {spread(values)} | "
                   f"{'passes' if passed else 'fails'} |")
        return passed

    for k in (2, 4, 8):
        line("A1", f"delivered, parts {k}", reading(cell, f"delivered parts {k}"), 0.95)
    for k in (2, 4, 8):
        line("A2", f"said, parts {k}", reading(cell, f"said parts {k}"), 0.90)
    line("A3", "said, chain 2", reading(cell, "said chain 2"), 0.95)
    line("A3", "said, chain 3", reading(cell, "said chain 3"), 0.90)
    silent = [pooled(r, [f"silent parts {k} absent" for k in (2, 4, 8)]) for r in cell]
    line("A4a", "silent, parts no record answers", [v for v in silent if v is not None], 0.95)
    refused = [pooled(r, ["refused parts 1 absent", "refused chain 2 absent",
                          "refused chain 3 absent"]) for r in cell]
    line("A4b", "refused, questions no record answers",
         [v for v in refused if v is not None], 0.90)
    born = [r for r in rows if r["arm"] == birth and int(r.get("size", 0)) == size]
    share = mean(reading(cell, "said chain 3")) - mean(reading(born, "said chain 3"))
    out.append(f"| A4c | said, chain 3: `{arm}` less `{birth}`, means | 0.30 | | {share:+.3f} | "
               f"{'passes' if share >= 0.30 else 'fails'} |")
    controls = [r for r in rows if r.get("schedule") == "online"
                and int(r.get("size", 0)) == size]
    shared = {r["founder"] for r in controls}
    mine = [r for r in cell if r["founder"] in shared]
    for group in ("parts 2", "parts 4", "parts 8", "chain 2", "chain 3"):
        best = max((mean(reading([r for r in controls if r["kind"] == kind], f"said {group}"))
                    for kind in sorted({r["kind"] for r in controls})), default=float("nan"))
        own = mean(reading(mine, f"said {group}"))
        verdict = "passes" if controls and own >= best - 0.03 else "fails"
        out.append(f"| A5 | said, {group}: `{arm}` {own:.3f}, the better online control "
                   f"{best:.3f} | within .03 | {len(mine)} founders | | {verdict} |")
    one = mean([r["costs"]["parts 1"]["read_ms"] for r in cell if "parts 1" in r.get("costs", {})])
    for k in (2, 4, 8):
        got = mean([r["costs"][f"parts {k}"]["read_ms"] for r in cell
                    if f"parts {k}" in r.get("costs", {})])
        out.append(f"| A6 | the read's ms, parts {k} over parts 1: {got:.3f} over {one:.3f} | "
                   f"under {k} | | {got / one:.2f} | {'passes' if got < k * one else 'fails'} |")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--kinds", nargs="*", default=["said", "part", "delivered", "ordered"])
    parser.add_argument("--gates", nargs="?", const="heads", default=None,
                        help="the arm the declared gates are read on")
    parser.add_argument("--size", type=int, default=100_000)
    args = parser.parse_args(argv)
    rows = load(args.rows)
    if args.controls is not None and args.controls.exists():
        rows += load(args.controls)
    if args.gates:
        print("\n".join(gates(rows, args.gates, args.size)))
        print()
    print("\n".join(table(rows, tuple(args.kinds))))
    print()
    print("\n".join(absent(rows)))
    print()
    print("\n".join(costs(rows)))
    curves = defaultdict(list)
    for r in rows:
        if r.get("nursery_curve"):
            curves[r["arm"]].append((r["founder"], r["nursery_curve"]))
    print()
    for arm, held in sorted(curves.items()):
        seen = set()
        for founder, curve in sorted(held):
            if founder not in seen:
                seen.add(founder)
                print(f"nursery {arm} founder {founder}: " + " ".join(f"{v:.2f}" for v in curve))
    return 0


if __name__ == "__main__":
    sys.exit(main())
