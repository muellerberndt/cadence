"""The fourth declaration's tables and gates (cadence#310), recomputed from the receipts.

    python benchmarks/memory_scale/summarize_million.py --dir benchmarks/memory_scale/results

Reads ``fourth-capacity``, ``fourth-ten-million``, ``fourth-timing``, ``fourth-facts`` and
``fourth-shared`` (``.jsonl`` or ``.jsonl.gz``) where they exist, and prints each table in
Markdown and each gate of ``PROTOCOL_4.md`` with the founders it holds on; and ``fifth-brain``,
the gates of ``PROTOCOL_5.md``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

GATE_FORMS = (("code", 128), ("hashed", 280))
CUES = ("exact", "noisy", "partial")
C1_BARS = {"exact": 0.99, "noisy": 0.95, "partial": 0.95}
MILLION, HUNDRED = 1_000_000, 100_000


def load(folder: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob(f"{name}*.jsonl*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as source:
            rows += [json.loads(line) for line in source if line.strip()]
    return rows


def span(values: list[float], digits: int = 3) -> str:
    """A range over founders, as the README writes it."""
    if not values:
        return ""
    low, high = min(values), max(values)

    def show(v: float) -> str:
        text = f"{v:.{digits}f}"
        return text[1:] if text.startswith("0.") else text

    return show(low) if show(low) == show(high) else f"{show(low)} to {show(high)}"


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def form_name(row: dict[str, Any]) -> str:
    return f"hashed-{row['width']}" if row["form"] == "hashed" else f"code-{row['width']}"


def capacity(rows: list[dict[str, Any]], title: str) -> None:
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[form_name(row), row["n"], row["arm"], row["cue"], row["heads"]].append(row)
    print(f"\n### {title}\n")
    print("| Form | Records | Arm | exact | noisy | partial | similar, noisy | bytes a record |"
          " records scored | entries read | multiply-accumulates | fixed weights |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    keys = sorted({(k[0], k[1], k[2]) for k in cells}, key=lambda k: (k[0], k[1], k[2]))
    for form, n, arm in keys:
        def cell(cue: str, form: str = form, n: int = n, arm: str = arm) -> list[dict[str, Any]]:
            return cells.get((form, n, arm, cue, 1), [])
        first = cell("noisy")
        if not first:
            continue
        print(f"| `{form}` | {n:,} | `{arm}` | "
              + " | ".join(span([r["found"] for r in cell(cue)])
                           for cue in (*CUES, "similar_noisy"))
              + f" | {mean([r['bytes_per_record'] for r in first]):,.0f}"
              f" | {mean([r['scored'] for r in first]):,.0f}"
              f" | {mean([r['touched'] for r in first]):,.0f}"
              f" | {mean([r['macs'] for r in first]):,.0f}"
              f" | {mean([r.get('fixed_weights', 0) for r in first]):,.0f} |")
    print("\n| Form | Records | Arm | family, 2, 4, 8 heads | summed, 2, 4, 8 | each by a head,"
          " 2, 4, 8 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for form, n, arm in keys:
        parts = []
        for kind in ("family", "superposed", "each"):
            parts.append("; ".join(
                span([r["found"] for r in cells.get((form, n, arm, kind, k), [])], 2)
                for k in (2, 4, 8)))
        if any(p.strip("; ") for p in parts):
            print(f"| `{form}` | {n:,} | `{arm}` | " + " | ".join(parts) + " |")


def gates_capacity(rows: list[dict[str, Any]]) -> None:
    by: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in rows:
        if row["heads"] == 1 and row["n"] == MILLION:
            by.setdefault((row["form"], row["width"], row["founder"], row["arm"]), {})[
                row["cue"]] = row["found"]
    founders = sorted({k[2] for k in by})
    c1, c3 = [], []
    for founder in founders:
        ok1 = ok3 = True
        for form, width in GATE_FORMS:
            pairs, scan = by.get((form, width, founder, "pairs")), by.get(
                (form, width, founder, "scan"))
            if pairs is None or scan is None:
                ok1 = ok3 = False
                continue
            ok1 = ok1 and all(pairs[cue] >= bar for cue, bar in C1_BARS.items())
            ok3 = ok3 and all(pairs[cue] >= scan[cue] - 0.01 - 1e-9 for cue in CUES)
        c1.append(ok1)
        c3.append(ok3)
    print(f"\nC1 (pairs at 1,000,000: exact .99, noisy and partial .95, both gate forms): "
          f"{sum(c1)} of {len(c1)} founders: {'passes' if sum(c1) >= 10 else 'fails'}")
    print(f"C3a (pairs within .01 of the scan at 1,000,000): {sum(c3)} of {len(c3)} founders: "
          f"{'passes' if sum(c3) >= 10 else 'fails'}")
    for form, width in GATE_FORMS:
        for arm in ("scan", "pairs", "units"):
            cells = [v for k, v in by.items() if k[:2] == (form, width) and k[3] == arm]
            print(f"  {form}-{width} {arm}: " + ", ".join(
                f"{cue} {span([v[cue] for v in cells if cue in v])}" for cue in CUES))
        gaps = [by[form, width, f, "scan"][cue] - by[form, width, f, "pairs"][cue]
                for f in founders for cue in CUES
                if (form, width, f, "pairs") in by and (form, width, f, "scan") in by]
        if gaps:
            print(f"  {form}-{width}: the largest gap to the scan {max(gaps):.3f}")


def timing(rows: list[dict[str, Any]]) -> None:
    ms: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row["heads"] == 1 and row["cue"] in CUES and row.get("read_ms") is not None:
            ms[form_name(row), row["arm"], row["founder"], row["n"]].append(row["read_ms"])
    print("\n### Timing (one core, ms per read, mean of the three cue kinds)\n")
    print("| Form | Arm | 100,000 | 1,000,000 | 1,000,000 over 100,000 |")
    print("| --- | --- | --- | --- | --- |")
    verdict: dict[str, list[bool]] = defaultdict(list)
    for form in sorted({k[0] for k in ms}):
        for arm in sorted({k[1] for k in ms if k[0] == form}):
            founders = sorted({k[2] for k in ms if k[:2] == (form, arm)})
            small = [mean(ms[form, arm, f, HUNDRED]) for f in founders
                     if (form, arm, f, HUNDRED) in ms]
            large = [mean(ms[form, arm, f, MILLION]) for f in founders
                     if (form, arm, f, MILLION) in ms]
            ratio = [b / a for a, b in zip(small, large, strict=False) if a > 0]
            print(f"| `{form}` | `{arm}` | {span(small)} | {span(large)} | {span(ratio, 2)} |")
            if arm.startswith("pairs") and form in ("code-128", "hashed-280"):
                verdict[arm] += [r < 3 for r in ratio]
    for arm, held in verdict.items():
        print(f"\nC3b ({arm}: 1,000,000 over 100,000 under 3, both gate forms, 3 of 3): "
              f"{sum(held)} of {len(held)} readings: {'passes' if held and all(held) else 'fails'}")


def facts(rows: list[dict[str, Any]]) -> None:
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[row["form"], row["n"], row["arm"], row["reading"], row["k"]].append(row)
    print("\n### Facts: several records, questions of parts, chains\n")
    print("| Key | Facts | Arm | fact, exact; noisy | subject, 2; 4; 8 heads | parts each,"
          " 2; 4; 8 | parts summed, 2; 4; 8 | chain exact, 2; 3 hops | chain noisy, 2; 3 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for form, n, arm in sorted({k[:3] for k in cells}):
        def get(reading: str, k: int, form: str = form, n: int = n, arm: str = arm) -> str:
            return span([r["found"] for r in cells.get((form, n, arm, reading, k), [])])
        print(f"| {form} | {n:,} | `{arm}` | {get('fact-exact', 1)}; {get('fact-noisy', 1)} | "
              + "; ".join(get("subject", k) for k in (2, 4, 8)) + " | "
              + "; ".join(get("compound-each", k) for k in (2, 4, 8)) + " | "
              + "; ".join(get("compound-summed", k) for k in (2, 4, 8)) + " | "
              + "; ".join(get("chain-exact", k) for k in (2, 3)) + " | "
              + "; ".join(get("chain-noisy", k) for k in (2, 3)) + " |")
    by: dict[tuple[Any, ...], float] = {
        (r["n"], r["founder"], r["reading"], r["k"]): r["found"]
        for r in rows if r["form"] == "parts" and r["arm"] == "pairs"}
    founders = sorted({k[1] for k in by})

    def held(n: int, wanted: list[tuple[str, int, float]]) -> list[bool]:
        return [all(by.get((n, f, reading, k), 0.0) >= bar for reading, k, bar in wanted)
                for f in founders if any(key[:2] == (n, f) for key in by)]

    for name, n, wanted in (
        ("C4a (subject by 2, 4, 8 heads at 1,000,000: .95)", MILLION,
         [("subject", k, 0.95) for k in (2, 4, 8)]),
        ("C4b (four parts, each by a head, at 1,000,000: .90)", MILLION,
         [("compound-each", 4, 0.90)]),
        ("C5 (chains at 100,000: two hops .95, three .90, exact and noisy)", HUNDRED,
         [("chain-exact", 2, 0.95), ("chain-noisy", 2, 0.95), ("chain-exact", 3, 0.90),
          ("chain-noisy", 3, 0.90)]),
    ):
        ok = held(n, wanted)
        print(f"\n{name}: {sum(ok)} of {len(ok)} founders: "
              f"{'passes' if ok and sum(ok) >= 10 else 'fails'}")


def shared(rows: list[dict[str, Any]]) -> None:
    by: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        by.setdefault((row["arm"], row["founder"]), {})[row["reading"]] = row
    print("\n### Shared keys\n")
    print("| Arm | found, all facts | sharing facts | the others | told again: current | stale |"
          " records for the facts | ms per write |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in sorted({k[0] for k in by}):
        cells = [v for k, v in sorted(by.items()) if k[0] == arm and "shared-shared" in v]
        overall = [c["shared-shared"]["share"] * c["shared-shared"]["found"]
                   + (1 - c["shared-shared"]["share"]) * c["shared-own"]["found"] for c in cells]
        print(f"| `{arm}` | {span(overall, 4)}"
              f" | {span([c['shared-shared']['found'] for c in cells])}"
              f" | {span([c['shared-own']['found'] for c in cells])}"
              f" | {span([c['shared-again']['found'] for c in cells])}"
              f" | {span([c['shared-again']['stale'] for c in cells])}"
              f" | {span([c['shared-own']['records'] / c['shared-own']['n'] for c in cells], 4)}"
              f" | {span([c['shared-own']['write_ms'] for c in cells])} |")
        if arm == "pairs-apart":
            print(f"\nC2 (found over all facts .99 on every founder, with apart): "
                  f"{sum(v >= 0.99 for v in overall)} of {len(overall)} founders: "
                  f"{'passes' if overall and all(v >= 0.99 for v in overall) else 'fails'}\n")


def brain(rows: list[dict[str, Any]]) -> None:
    """The fifth declaration: a brain with its own records at 100,000 (C6)."""
    rows = sorted(rows, key=lambda r: r["founder"])
    print("\n### A brain with its own records\n")
    print("| Founder | Records | Facts whose key another has | found | used | found, own key |"
          " revised: current | stale | used | ms a told moment | parameters |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        u, v = r["unrevised"], r["revised"]
        print(f"| {r['founder']} | {r['records'][0]:,} | {r['shared_keys']:.2%} | {u['found']:.4f}"
              f" | {u['used']:.4f} | {u['found_own']:.4f} | {v['found']:.4f} | {v['stale']:.4f}"
              f" | {v['used']:.4f} | {r['tell_ms']:.2f} | {r['parameters']:,} |")
    a = [r["unrevised"]["used"] >= 0.95 for r in rows]
    b = [r["revised"]["found"] >= 0.95 and r["revised"]["stale"] < 0.02
         and r["revised"]["used"] >= 0.95 for r in rows]
    for name, held in (("C6a (used, unrevised, 100,000: .95)", a),
                       ("C6b (revised at 100,000: current .95, stale under .02, used .95)", b)):
        print(f"\n{name}: {sum(held)} of {len(held)} founders: "
              f"{'passes' if len(held) >= 12 and sum(held) >= 10 else 'fails'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    rows = load(args.dir, "fourth-capacity")
    if rows:
        capacity(rows, "Capacity at 100,000 and 1,000,000 records")
        gates_capacity(rows)
    rows = load(args.dir, "fourth-ten-million")
    if rows:
        capacity(rows, "Ten million records")
    rows = load(args.dir, "fourth-timing")
    if rows:
        timing(rows)
    rows = load(args.dir, "fourth-facts")
    if rows:
        facts(rows)
    rows = load(args.dir, "fourth-shared")
    if rows:
        shared(rows)
    rows = load(args.dir, "fifth-brain")
    if rows:
        brain(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
