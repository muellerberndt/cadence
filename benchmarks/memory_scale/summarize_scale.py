"""The sixth and seventh declarations' tables and gates (cadence#310), recomputed from the
receipts.

    python benchmarks/memory_scale/summarize_scale.py --dir benchmarks/memory_scale/results

Reads ``sixth-brain-million``, ``sixth-brain-sizes`` and ``sixth-window`` (the gates of
``PROTOCOL_6.md``) and ``seventh-found``, ``seventh-hundred-million`` and ``seventh-timing``
(the gates of ``PROTOCOL_7.md``), ``.jsonl`` or ``.jsonl.gz``, where they exist, and prints each
table in Markdown and each gate with the founders it holds on. The cell at 100,000,000 is read
alone on its box, so it stands among the found cells and among the timing cells.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from summarize_million import load, mean, span  # noqa: E402

KINDS = ("unrevised", "revised")
CUES = ("exact", "noisy", "partial")
MILLION = 1_000_000
SIZES = (1000, 10_000, 100_000)
# the seventh declaration's bars
FOUND = {"exact": 0.99, "noisy": 0.95, "partial": 0.95}
NEAR_SCAN = 0.02
GROWTH = 2.0  # ms per read at N over ms at N / 10, the pages in memory
BYTES_SPREAD = 0.05
CELLS_A_WRITE = 20
PAUSE_P99_MS, PAUSE_WORST_MS = 100.0, 1000.0
EXACT_MS, FULL_MS = 25.0, 500.0  # a read under the memory limit: its record's key; any cue


def verdict(name: str, held: list[Any], of: int, need: int) -> bool:
    passed = len(held) >= need
    print(f"{name}: {len(held)} of {of} founders: {'passes' if passed else 'fails'}"
          + ("" if len(held) == of else f" (holds on {sorted(held)})"))
    return passed


def ms(values: list[float], digits: int = 2) -> str:
    low, high = min(values), max(values)
    return f"{low:.{digits}f}" if f"{low:.{digits}f}" == f"{high:.{digits}f}" \
        else f"{low:.{digits}f} to {high:.{digits}f}"


def count(values: list[float]) -> str:
    low, high = min(values), max(values)
    return f"{low:,.0f}" if round(low) == round(high) else f"{low:,.0f} to {high:,.0f}"


# -- the sixth declaration


def brain_million(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """C6c and C6d: a brain with its own records at 1,000,000."""
    rows = [r for r in rows if r["n"] == MILLION]
    founders = sorted({r["founder"] for r in rows})
    print(f"\n### A brain with its own records at 1,000,000 ({len(founders)} founders)\n")
    print("| Reading | Range over the founders |\n| --- | --- |")
    for kind in KINDS:
        for name in ("found", "used", "stale"):
            print(f"| {name}, {kind} | {span([r[kind][name] for r in rows])} |")
        own = [r[kind]["found_own"] for r in rows if r[kind]["found_own"] is not None]
        print(f"| found over facts whose key is their own, {kind} | {span(own)} |")
    print(f"| facts whose key another fact has | "
          f"{span([100 * r['shared_keys'] for r in rows], 2)} % |")
    print(f"| records a stream holds | {count([r['records'][0] for r in rows])} |")
    print(f"| nursery | {span([r['nursery'] for r in rows], 4)} |")
    print(f"| parameters | {count([r['parameters'] for r in rows])} |")
    print(f"| ms per told moment; per asked moment | {ms([r['tell_ms'] for r in rows])}; "
          f"{ms([r['ask_ms'] for r in rows])} |")
    print(f"| bytes a record, in the store's arrays; on the disk | "
          f"{count([r['bytes_per_record'] for r in rows])}; "
          f"{count([r.get('disk_bytes_per_record', 0) for r in rows])} |")
    print(f"| hours a founder | {ms([r['seconds'] / 3600 for r in rows], 1)} |\n")
    used = [r["founder"] for r in rows if r["unrevised"]["used"] >= 0.90]
    revised = [r["founder"] for r in rows if r["revised"]["found"] >= 0.95
               and r["revised"]["stale"] < 0.02 and r["revised"]["used"] >= 0.90]
    need = min(10, len(founders))
    return {
        "C6c": verdict("C6c (used, unrevised facts at 1,000,000: .90)", used, len(founders),
                       need),
        "C6d": verdict("C6d (revised facts at 1,000,000: current .95, stale under .02, used "
                       ".90)", revised, len(founders), need),
    }


def window(brain: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, bool]:
    """C6e and C7a: the brain at three sizes beside the window and the index."""
    founders = sorted({r["founder"] for r in brain} & {r["founder"] for r in controls})
    by_arm: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in brain:
        by_arm["brain", row["n"]][row["founder"]] = row
    for row in controls:
        by_arm[row["arm"], row["n"]][row["founder"]] = row
    print(f"\n### The same facts in three arms ({len(founders)} founders)\n")
    print("| Facts | Arm | right, unrevised | right, revised | found | bytes a record | "
          "ms to write a fact | ms to read a question | parameters | "
          "multiply-accumulates a read |\n" + "| --- " * 10 + "|")
    for n in SIZES:
        for arm in ("brain", "context", "retrieval"):
            cells = [by_arm[arm, n][f] for f in founders if f in by_arm[arm, n]]
            if not cells:
                print(f"| {n:,} | {arm} | not read | | | | | | | |")
                continue
            if arm == "brain":
                right = [span([r[k]["used"] for r in cells]) for k in KINDS]
                found = span([r["unrevised"]["found"] for r in cells])
                write = ms([4 * r["tell_ms"] for r in cells])
                read = ms([4 * r["ask_ms"] for r in cells])
                macs = ""
            else:
                right = [span([r[k] for r in cells]) for k in KINDS]
                found = span([r["found_unrevised"] for r in cells]) \
                    if arm == "retrieval" else ""
                write = ms([r["write_ms_at_end"] for r in cells], 4)
                read = ms([r["ask_ms"] for r in cells])
                macs = count([r["macs_per_read"] for r in cells])
            print(f"| {n:,} | {arm} | {right[0]} | {right[1]} | {found} | "
                  f"{count([r['bytes_per_record'] for r in cells])} | {write} | {read} | "
                  f"{count([r['parameters'] for r in cells])} | {macs} |")
    print()
    for arm in ("brain", "context", "retrieval"):
        ratios = []
        for f in founders:
            low, high = by_arm[arm, SIZES[0]].get(f), by_arm[arm, SIZES[-1]].get(f)
            if low and high:
                ratios.append(high["ask_ms"] / low["ask_ms"])
        if ratios:
            print(f"ms to read a question at {SIZES[-1]:,} over {SIZES[0]:,}, {arm}: "
                  f"{ms(ratios)}")
    taught = [r for r in controls if r["arm"] == "context"]
    if taught:
        print(f"the window: taught at {taught[0]['taught']} facts, "
              f"{count([r['steps'] for r in taught])} steps in "
              f"{ms([r['train_seconds'] / 60 for r in taught], 0)} minutes, right on its last "
              f"taught windows {span([r['train_right'] for r in taught])}")
    apart = [f for f in founders if by_arm["brain", 100_000].get(f)
             and by_arm["brain", 100_000][f]["unrevised"]["found"] >= 0.99
             and by_arm["brain", 100_000][f]["unrevised"]["used"] >= 0.95]
    beside = []
    for f in founders:
        holds = True
        for n in SIZES:
            mine = by_arm["brain", n].get(f)
            others = [by_arm[arm, n][f] for arm in ("context", "retrieval")
                      if f in by_arm[arm, n]]
            if mine is None or not others:
                holds = False
                continue
            for kind in KINDS:
                holds &= mine[kind]["used"] >= max(o[kind] for o in others) - 0.05
        if holds:
            beside.append(f)
    need = min(10, len(founders))
    return {
        "C6e": verdict("C6e (the brain at 100,000, facts told apart: found .99, used .95)",
                       apart, len(founders), need),
        "C7a": verdict("C7a (the brain's used no more than .05 under the better control, "
                       "every size and kind)", beside, len(founders), need),
    }


# -- the seventh declaration


def form_name(row: dict[str, Any]) -> str:
    return f"hashed-{row['width']}" if row["form"] == "hashed" else f"code-{row['width']}"


def scale(rows: list[dict[str, Any]], timing: list[dict[str, Any]]) -> dict[str, bool]:
    """S1 to S5: the store on disk from 100,000 to 100,000,000 records."""
    out: dict[str, bool] = {}
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[form_name(row), row["n"], row["step"], row["arm"], row.get("cue"),
              row.get("pass"), bool(row.get("memory_limit"))].append(row)
    forms = sorted({form_name(r) for r in rows})
    sizes = sorted({r["n"] for r in rows})
    print("\n### Found, the store on disk beside the scan\n")
    print("| Form | Records | Founders | exact | noisy | partial | scan: exact | noisy | "
          "partial | entries a full cell did not take |\n" + "| --- " * 10 + "|")
    for form in forms:
        for n in sizes:
            disk = {c: cells[form, n, "read", "disk", c, "first", False] for c in CUES}
            if not disk["exact"]:
                continue
            scans = {c: cells[form, n, "scan", "scan", c, None, False] for c in CUES}
            built = cells[form, n, "build", "disk", None, None, False]
            lost = span([100 * r["dropped"] / (r["dropped"] + r["entries"]) for r in built], 2)
            print(f"| `{form}` | {n:,} | {len(disk['exact'])} | "
                  + " | ".join(span([r["found"] for r in disk[c]]) for c in CUES) + " | "
                  + " | ".join(span([r["found"] for r in scans[c]]) for c in CUES)
                  + f" | {lost} % |")
    print("\n### Bytes a record and the time to build\n")
    print("| Form | Records | bytes a record, in the arrays | on the disk | µs a record to "
          "fill and file | cells | seconds |\n" + "| --- " * 7 + "|")
    for form in forms:
        for n in sizes:
            built = cells[form, n, "build", "disk", None, None, False]
            if built:
                print(f"| `{form}` | {n:,} | {count([r['bytes_per_record'] for r in built])} | "
                      f"{count([r['disk_bytes_per_record'] for r in built])} | "
                      f"{ms([r['fill_us_per_record'] for r in built], 1)} | "
                      f"{count([r['cells'] for r in built])} | "
                      f"{count([r['seconds'] for r in built])} |")

    # S1: found on disk, and beside the scan
    for form in forms:
        for n in sizes:
            by_founder: dict[int, bool] = {}
            for cue in CUES:
                scans = {r["founder"]: r for r in cells[form, n, "scan", "scan", cue, None,
                                                        False]}
                for row in cells[form, n, "read", "disk", cue, "first", False]:
                    ok = row["found"] >= FOUND[cue] if form.startswith("code") or cue != \
                        "partial" else True
                    scanned = scans.get(row["founder"])
                    if scanned is not None:
                        ok &= row["found"] >= scanned["found"] - NEAR_SCAN
                    by_founder[row["founder"]] = by_founder.get(row["founder"], True) and ok
            if by_founder:
                held = [f for f, ok in by_founder.items() if ok]
                need = len(by_founder) if len(by_founder) < 12 else 10
                out[f"S1 {form} {n}"] = verdict(
                    f"S1 (`{form}` at {n:,}: exact .99, noisy and partial .95, within .02 of "
                    "the scan)", held, len(by_founder), need)
    return {**out, **timed(timing or rows)}


def timed(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """S2 to S5 from the timing cells: the read's time in memory and under a memory limit,
    the pause of a write, the bytes a record."""
    out: dict[str, bool] = {}
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[form_name(row), row["n"], row["step"], row["arm"], row.get("cue"),
              row.get("pass"), bool(row.get("memory_limit"))].append(row)
    forms = sorted({form_name(r) for r in rows})
    sizes = sorted({r["n"] for r in rows})
    print("\n### The read's time on one core\n")
    print("| Form | Records | Memory | exact, ms (page faults) | noisy | partial | the same "
          "again: exact | noisy | partial |\n" + "| --- " * 9 + "|")
    for form in forms:
        for n in sizes:
            for limited in (False, True):
                first = {c: cells[form, n, "read", "disk", c, "first", limited] for c in CUES}
                if not first["exact"]:
                    continue
                again = {c: cells[form, n, "read", "disk", c, "again", limited] for c in CUES}
                limit = {r["memory_limit"] for r in first["exact"]}
                label = "all of it" if not limited else \
                    " and ".join(f"{v / 2 ** 30:.0f} GB" for v in sorted(limit))
                print(f"| `{form}` | {n:,} | {label} | "
                      + " | ".join(f"{ms([r['read_ms'] for r in first[c]])} "
                                   f"({count([r['faults_per_read'] for r in first[c]])})"
                                   for c in CUES) + " | "
                      + " | ".join(ms([r["read_ms"] for r in again[c]]) for c in CUES) + " |")
    print("\n### Records written one by one while the store is read\n")
    print("| Form | Records | Memory | ms a write: mean | median | 99th of 100 | worst | cells "
          "a write | found: the new records | the earlier ones |\n" + "| --- " * 10 + "|")
    for form in forms:
        for n in sizes:
            for limited in (False, True):
                wrote = cells[form, n, "write", "disk", None, None, limited]
                if wrote:
                    label = "all of it" if not limited else "limited"
                    print(f"| `{form}` | {n:,} | {label} | "
                          + " | ".join(ms([r[k] for r in wrote], 3) for k in (
                              "write_ms", "write_ms_median", "write_ms_p99", "write_ms_worst"))
                          + f" | {ms([r['cells_per_write'] for r in wrote])} | "
                          f"{span([r['found_new'] for r in wrote])} | "
                          f"{span([r['found_earlier'] for r in wrote])} |")
    print()
    for form in forms:
        # S2: the pages in memory, each tenfold
        ratios, holds = [], True
        for low, high in zip(sizes, sizes[1:], strict=False):
            if high != 10 * low:
                continue
            a = [mean([mean([r["read_ms"] for r in cells[form, n, "read", "disk", c, "again",
                                                         False]]) for c in CUES])
                 for n in (low, high)]
            if all(cells[form, n, "read", "disk", "exact", "again", False] for n in (low, high)):
                ratios.append(f"{high:,} over {low:,}: {a[1] / a[0]:.2f}")
                holds &= a[1] / a[0] < GROWTH
        if ratios:
            print(f"S2 (`{form}`, ms per read with its pages in memory, each tenfold under "
                  f"{GROWTH:g}): {'; '.join(ratios)}: {'passes' if holds else 'fails'}")
            out[f"S2 {form}"] = holds
        # S3: under the memory limit
        limited_sizes = [n for n in sizes if cells[form, n, "read", "disk", "exact", "first",
                                                   True]]
        if limited_sizes:
            holds, said = True, []
            for n in limited_sizes:
                exact = mean([r["read_ms"] for r in cells[form, n, "read", "disk", "exact",
                                                          "first", True]])
                full = max(mean([r["read_ms"] for r in cells[form, n, "read", "disk", c,
                                                             "first", True]])
                           for c in ("noisy", "partial"))
                same = all(
                    a["found"] == b["found"] for c in CUES
                    for a in cells[form, n, "read", "disk", c, "first", True]
                    for b in cells[form, n, "read", "disk", c, "first", False]
                    if a["founder"] == b["founder"])
                holds &= exact < EXACT_MS and full < FULL_MS and same
                said.append(f"{n:,}: {exact:.2f} ms its record's key, {full:.1f} ms any cue, "
                            f"found {'the same' if same else 'differs'}")
            print(f"S3 (`{form}`, under the memory limit: under {EXACT_MS:g} ms a read of a "
                  f"record's key, under {FULL_MS:g} ms any read, found as without the limit): "
                  f"{'; '.join(said)}: {'passes' if holds else 'fails'}")
            out[f"S3 {form}"] = holds
        # S4: the pause of a write
        wrote = [r for n in sizes for limited in (False, True)
                 for r in cells[form, n, "write", "disk", None, None, limited]]
        if wrote:
            holds = all(r["write_ms_p99"] < PAUSE_P99_MS and r["write_ms_worst"] < PAUSE_WORST_MS
                        and r["cells_per_write"] <= CELLS_A_WRITE and r["found_new"] >= 0.99
                        for r in wrote)
            print(f"S4 (`{form}`, a write: 99 of 100 under {PAUSE_P99_MS:g} ms, none over "
                  f"{PAUSE_WORST_MS:g} ms, at most {CELLS_A_WRITE} cells, the new records "
                  f"found at .99): worst {max(r['write_ms_worst'] for r in wrote):.1f} ms, "
                  f"99th {max(r['write_ms_p99'] for r in wrote):.2f} ms over "
                  f"{len(wrote)} readings: {'passes' if holds else 'fails'}")
            out[f"S4 {form}"] = holds
        # S5: the bytes a record
        held = {n: mean([r["disk_bytes_per_record"] for r in cells[form, n, "build", "disk",
                                                                   None, None, False]])
                for n in sizes if n >= MILLION and cells[form, n, "build", "disk", None, None,
                                                         False]}
        if len(held) > 1:
            spread = max(held.values()) / min(held.values()) - 1.0
            holds = spread <= BYTES_SPREAD
            print(f"S5 (`{form}`, bytes a record on the disk from 1,000,000 up within "
                  f"{100 * BYTES_SPREAD:g} %): "
                  + "; ".join(f"{n:,}: {v:,.0f}" for n, v in held.items())
                  + f": {'passes' if holds else 'fails'}")
            out[f"S5 {form}"] = holds
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    million = load(args.dir, "sixth-brain-million")
    if million:
        brain_million(million)
    sizes, controls = load(args.dir, "sixth-brain-sizes"), load(args.dir, "sixth-window")
    if sizes and controls:
        window(sizes, controls)
    largest = load(args.dir, "seventh-hundred-million")
    found = load(args.dir, "seventh-found") + largest
    if found:
        scale(found, load(args.dir, "seventh-timing") + largest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
