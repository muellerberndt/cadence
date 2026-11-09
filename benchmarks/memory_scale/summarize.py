"""The declared readings of cadence#288 from their receipts: the gates of ``PROTOCOL.md`` and
the tables of the README.

    python benchmarks/memory_scale/summarize.py --dir benchmarks/memory_scale/results
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

FOUNDERS = (2880, 2881, 2882, 2883, 2884, 2885)
NEEDED = 5  # founders a gate passes on


def rows(path: Path) -> list[dict[str, Any]]:
    """The rows of one receipt, a JSON object per line; a receipt kept as ``.gz`` is read
    through it."""
    packed = path.with_name(path.name + ".gz")
    text = gzip.decompress(packed.read_bytes()).decode() if packed.exists() else path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def held(folder: Path, name: str) -> bool:
    """Whether the folder holds the receipt, plain or packed."""
    return (folder / name).exists() or (folder / (name + ".gz")).exists()


def pick(table: list[dict[str, Any]], **where: Any) -> list[dict[str, Any]]:
    return [r for r in table if all(r.get(k) == v for k, v in where.items())]


def by_founder(table: list[dict[str, Any]], field: str, **where: Any) -> list[float]:
    """``field`` of the one row per declared founder that ``where`` selects, in founder order."""
    out = []
    for founder in FOUNDERS:
        found = pick(table, founder=founder, **where)
        if len(found) == 1:
            out.append(float(found[0][field]))
    return out


def last(table: list[dict[str, Any]], **where: Any) -> list[dict[str, Any]]:
    """Each declared founder's row at the stream's end."""
    out = []
    for founder in FOUNDERS:
        found = pick(table, founder=founder, **where)
        if found:
            out.append(max(found, key=lambda r: r["moment"]))
    return out


def gates(folder: Path) -> dict[str, dict[str, Any]]:
    """Every gate of the protocol: its readings per founder, its bar and whether it passes on
    ``NEEDED`` of the declared founders. A gate whose receipt is absent is left out."""
    out: dict[str, dict[str, Any]] = {}

    def gate(name: str, values: list[float], passes: list[bool], bar: str) -> None:
        out[name] = {"values": [round(v, 4) for v in values], "bar": bar,
                     "founders": int(sum(passes)), "passes": int(sum(passes)) >= NEEDED}

    if held(folder, "retrieval.jsonl"):
        table = rows(folder / "retrieval.jsonl")
        for name, cue, heads, bar in (("R1", "exact", 1, 0.99), ("R2", "noisy", 1, 0.90),
                                      ("R3", "family", 4, 0.85)):
            values = by_founder(table, "found", arm="bank", n=10_000, width=64, cue=cue,
                                heads=heads)
            gate(name, values, [v >= bar for v in values], f"at least {bar}")
    if held(folder, "used.jsonl"):
        table = rows(folder / "used.jsonl")
        small = by_founder(table, "used", arm="bank", n=128)
        large = by_founder(table, "used", arm="bank", n=10_000)
        gate("U1", large, [b >= a - 0.05 for a, b in zip(small, large, strict=True)],
             "within .05 of used at 128")
    if held(folder, "retention.jsonl"):
        table = rows(folder / "retention.jsonl")
        fade = last(table, arm="fade", budget=1000, drift=False)
        lru = last(table, arm="lru", budget=1000, drift=False)
        gate("K1", [r["top"] for r in fade], [r["top"] >= 0.95 for r in fade], "at least 0.95")
        gate("K2", [r["revised"] for r in fade], [r["revised"] >= 0.90 for r in fade],
             "at least 0.9")
        gate("K3", [r["top"] for r in fade],
             [a["top"] > b["top"] for a, b in zip(fade, lru, strict=True)],
             "above least recently used")
        ends = [max(pick(table, founder=r["founder"], arm=r["arm"], budget=r["budget"],
                         drift=r["drift"]), key=lambda x: x["moment"])
                for r in {(r["founder"], r["arm"], r["budget"], r["drift"]): r
                          for r in table}.values()]
        out["writes"] = {
            "least": round(min(r["writes"] / r["budget"] for r in ends), 2),
            "bar": "at least 10 budgets of writes on every arm",
            "passes": all(r["writes"] >= 10 * r["budget"] for r in ends),
        }
    for name, receipt, arm in (("C1", "cost.jsonl", "bank"),
                               ("C2", "cost-device.jsonl", "device")):
        if held(folder, receipt):
            table = rows(folder / receipt)
            founders = sorted({r["founder"] for r in table})
            read = [pick(table, founder=f, arm=arm, n=10_000)[0]["read_ms"] for f in founders]
            rest = [pick(table, founder=f, arm="moment", n=128)[0]["rest_ms"] for f in founders]
            out[name] = {"values": read, "budget": rest, "bar": "at most the rest of the moment",
                         "founders": int(sum(a <= b for a, b in zip(read, rest, strict=True))),
                         "passes": all(a <= b for a, b in zip(read, rest, strict=True))}
    return out


def span(values: list[float], digits: int = 3) -> str:
    """A reading over the founders: one number where they agree, else its range."""
    if not values:
        return ""
    low, high = round(min(values), digits), round(max(values), digits)
    return f"{low:.{digits}f}" if low == high else f"{low:.{digits}f} to {high:.{digits}f}"


def retrieval_table(folder: Path, name: str = "retrieval.jsonl", width: int = 64) -> str:
    table = [r for r in rows(folder / name) if r["width"] == width]
    arms = list(dict.fromkeys(r["arm"] for r in table))
    cues = ("exact", "noisy", "partial", "similar", "similar_noisy")
    lines = ["| N | Arm | " + " | ".join(cues) + " | 2 heads | 4 heads | 8 heads | 2 superposed |",
             "| --- | --- |" + " --- |" * (len(cues) + 4)]
    for n in sorted({r["n"] for r in table}):
        for arm in arms:
            cells = [span(by_founder(table, "found", arm=arm, n=n, cue=cue, heads=1))
                     for cue in cues]
            cells += [span(by_founder(table, "found", arm=arm, n=n, cue="family", heads=k), 2)
                      for k in (2, 4, 8)]
            cells.append(span(by_founder(table, "found", arm=arm, n=n, cue="superposed",
                                         heads=2), 2))
            if any(cells):
                lines.append(f"| {n:,} | `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cost_table(folder: Path, name: str = "retrieval.jsonl", width: int = 64) -> str:
    table = [r for r in rows(folder / name) if r["width"] == width and r["heads"] == 1]
    arms = list(dict.fromkeys(r["arm"] for r in table))
    lines = ["| N | Arm | ms per read | ms per write | kB per record |",
             "| --- | --- | --- | --- | --- |"]
    for n in sorted({r["n"] for r in table}):
        for arm in arms:
            found = pick(table, arm=arm, n=n, cue="noisy")
            if not found:
                continue
            read = float(np.mean([r["read_ms"] for r in found]))
            write = [r["write_ms"] for r in found if r.get("write_ms") is not None]
            size = float(np.mean([r["bytes"] for r in found])) / n / 1e3
            lines.append(f"| {n:,} | `{arm}` | {read:.3f} | "
                         f"{float(np.mean(write)):.3f} | {size:.2f} |" if write else
                         f"| {n:,} | `{arm}` | {read:.3f} | | {size:.2f} |")
    return "\n".join(lines)


def moment_table(folder: Path, name: str = "cost.jsonl") -> str:
    """Milliseconds per read by arm against N on one box, the mean over its founders, and the
    rest of the default composition's moment."""
    table = rows(folder / name)
    arms = [a for a in dict.fromkeys(r["arm"] for r in table) if a != "moment"]
    head = ["N", "`moment`: read", "write", "rest", *[f"`{a}`: read" for a in arms]]
    lines = ["| " + " | ".join(head) + " |", "| ---" * len(head) + " |"]

    def mean(arm: str, n: int, field: str) -> str:
        found = [r[field] for r in pick(table, arm=arm, n=n) if r.get(field) is not None]
        return f"{float(np.mean(found)):.3f}" if found else ""

    for n in sorted({r["n"] for r in table}):
        cells = [mean("moment", n, f) for f in ("read_ms", "write_ms", "rest_ms")]
        cells += [mean(a, n, "read_ms") for a in arms]
        lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def used_table(folder: Path) -> str:
    table = rows(folder / "used.jsonl")
    arms = list(dict.fromkeys(r["arm"] for r in table))
    lines = ["| Arm | N | first | found | used | used where the first choice is right |",
             "| --- | --- | --- | --- | --- | --- |"]
    for arm in arms:
        for n in sorted({r["n"] for r in table}):
            cells = [span(by_founder(table, f, arm=arm, n=n), 2)
                     for f in ("first", "found", "used")]
            given = [v for v in by_founder(table, "used_first", arm=arm, n=n) if not np.isnan(v)]
            lines.append(f"| `{arm}` | {n:,} | " + " | ".join(cells) + f" | {span(given, 2)} |")
    return "\n".join(lines)


def margins_table(folder: Path, arm: str = "default") -> str:
    table = pick(rows(folder / "used.jsonl"), arm=arm)
    terms = [t for t in table[0]["margins"] if any(abs(r["margins"][t]) >= 0.05 for r in table)]
    lines = ["| N | " + " | ".join(terms) + " |", "| --- |" + " --- |" * len(terms)]
    for n in sorted({r["n"] for r in table}):
        found = pick(table, n=n)
        lines.append(f"| {n:,} | " + " | ".join(
            f"{float(np.mean([r['margins'][t] for r in found])):+.1f}" for t in terms) + " |")
    return "\n".join(lines)


def retention_table(folder: Path, budget: int = 1000) -> str:
    table = rows(folder / "retention.jsonl")
    arms = list(dict.fromkeys(r["arm"] for r in table))
    head = ("Arm", "top", "revised", "stale", "rare", "hits", "writes, budgets", "drifting: top",
            "left")
    lines = ["| " + " | ".join(head) + " |", "| ---" * len(head) + " |"]
    for arm in arms:
        plain = last(table, arm=arm, budget=budget, drift=False)
        drift = last(table, arm=arm, budget=budget, drift=True)
        if not plain:
            continue
        cells = [span([r[f] for r in plain]) for f in ("top", "revised")]
        cells += [span([r[f] for r in plain], 2) for f in ("stale", "rare", "hits")]
        cells.append(span([r["writes"] / budget for r in plain], 1))
        cells += [span([r["top"] for r in drift]), span([r["left"] for r in drift], 2)]
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def checkpoints_table(folder: Path, budget: int = 1000,
                      arms: tuple[str, ...] = ("fade", "lru", "lfu")) -> str:
    table = pick(rows(folder / "retention.jsonl"), budget=budget, drift=False)
    moments = sorted({r["moment"] for r in table})
    lines = ["| Arm | " + " | ".join(str(m // budget) for m in moments) + " |",
             "| --- |" + " --- |" * len(moments)]
    for arm in arms:
        cells = [f"{float(np.mean([r['top'] for r in pick(table, arm=arm, moment=m)])):.3f}"
                 for m in moments]
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(gates(args.dir), indent=1))
    for title, table in (
        ("found, key width 64", lambda: retrieval_table(args.dir)),
        ("cost of the retrieval arms", lambda: cost_table(args.dir)),
        ("ms on one core", lambda: moment_table(args.dir)),
        ("ms on the device's box", lambda: moment_table(args.dir, "cost-device.jsonl")),
        ("found and used", lambda: used_table(args.dir)),
        ("the default composition's margins", lambda: margins_table(args.dir)),
        ("retention, budget 1,000", lambda: retention_table(args.dir)),
        ("top by checkpoint, budget 1,000", lambda: checkpoints_table(args.dir)),
    ):
        try:
            print(f"\n{title}\n\n{table()}")
        except (FileNotFoundError, IndexError):
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
