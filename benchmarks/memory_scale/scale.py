"""Records at any size (cadence#310, instrument e): a store that grows, on disk, read at 100,000
to 100,000,000 records, with the law of its read's time and of its bytes per record.

A cell is a founder, a number of records N and a key form. It runs in steps, each a process of
its own, so that the reading can run under a memory limit the building did not have:

* ``build``: the records are drawn by parts (``PART`` at a time, never all in memory) and
  written into a ``cadence.LongTermMemory(grow=True, path=...)`` by ``fill(file=False)``. One
  ``file()`` files the first nine tenths onto an empty table; a second files the last tenth
  onto the table those left (``grown``), which takes its cells by splitting, as it does for a
  write. The store is flushed and closed. The step reports the seconds of each stage and the
  bytes the store holds, in its arrays and on the disk.
* ``read``: the store is opened again (``LongTermMemory.open``). Cues of every kind are read
  through the address and by the scan that scores every record (by parts, over the store's own
  keys): ``found`` for both. Then the timing: ms per read by cue kind, with the page faults that
  waited for the disk, and after it the same reads again (``warm``). Then ``writes`` new records
  written one by one while the store is read: each write's pause, the cells the table took, and
  whether the new records and the earlier ones are found.
* ``memory``: the same records in a store in memory (``grow=True`` and the fixed table of
  ``pairs``), filled and read with the same cues, to N the machine's memory holds.

**Keys.** ``code``: unit codes of ``width`` coordinates in float32, the first quarter of every
part in families of 8 at cosine .8. ``hashed``: 280 coordinates, 15 to 22 of them one. The
values are unit codes of 64 coordinates. Part ``p`` of founder ``f`` is drawn from the seed
``[f, 310, 6, N, width, p]``, so a part is drawn again where a step needs it.

**Cues**, as ``million.py`` draws them: ``exact``; ``noisy`` (cosine .71; a hashed key with
three in ten of its active units exchanged); ``partial`` (half of the coordinates silent).

    PYTHONPATH=src python benchmarks/memory_scale/scale.py build --plan plan.json --cell 0 \\
        --root /data/stores --out rows.jsonl
    PYTHONPATH=src python benchmarks/memory_scale/scale.py read --plan plan.json --cell 0 \\
        --root /data/stores --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import million as ml  # noqa: E402

PART = 65_536  # records drawn at once
KINDS = ("exact", "noisy", "partial")


def draw_part(founder: int, n: int, form: str, width: int, value_width: int,
              part: int) -> tuple[np.ndarray, np.ndarray]:
    """Part ``part`` of a cell's records: its keys and values, in float32."""
    many = min(PART, n - part * PART)
    rng = np.random.default_rng(np.random.SeedSequence([founder, 310, 6, n, width, part]))
    if form == "hashed":
        keys = ml.hashed_records(rng, many)
    else:
        keys = ml.unit(rng.standard_normal((many, width), dtype=np.float32))
        families = int(many * ml.FAMILIES) // ml.FAMILY
        if families:
            themes = ml.unit(rng.standard_normal((families, width), dtype=np.float32))
            rows = np.arange(families * ml.FAMILY).reshape(families, ml.FAMILY)
            mixed = np.sqrt(ml.SHARE) * themes[:, None, :] + np.sqrt(1 - ml.SHARE) * keys[rows]
            keys[rows] = ml.unit(mixed.astype(np.float32))
    values = ml.unit(rng.standard_normal((many, value_width), dtype=np.float32))
    return keys, values


def single(n: int, form: str, record: np.ndarray) -> np.ndarray:
    """Whether each record is outside every family (hashed keys have none)."""
    if form == "hashed":
        return np.ones(len(record), dtype=bool)
    within = record % PART
    part_size = np.minimum(PART, n - (record // PART) * PART)
    return np.asarray(within >= (part_size * ml.FAMILIES).astype(np.int64) // ml.FAMILY
                      * ml.FAMILY)


def settings(spec: dict[str, Any], n: int) -> dict[str, Any]:
    """The store's settings: the declaration's (a read by cosine, no fading, float32, appended
    writes) under the plan's, with the address's units by the records (``expansion_at``)."""
    out = {"cosine": True, "fade": False, "revise": 0.0, "dtype": "float32",
           **spec.get("settings", {})}
    by_size = spec.get("expansion_at")
    if by_size:
        out["expansion"] = int(by_size[str(n)])
    return out


def on_disk(folder: Path) -> int:
    """The bytes the folder's files hold on the disk (their blocks, not their lengths)."""
    return int(sum(f.stat().st_blocks * 512 for f in folder.iterdir() if f.is_file()))


def faults() -> int:
    """The page faults of this process that waited for the disk."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_majflt)


def memory_limit() -> int | None:
    """The memory limit of this process's control group in bytes; None without one."""
    try:
        group = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
        text = (Path("/sys/fs/cgroup") / group.lstrip("/") / "memory.max").read_text().strip()
        return None if text == "max" else int(text)
    except (OSError, ValueError):
        return None


def cell_of(plan: dict[str, Any], index: int) -> dict[str, Any]:
    cells = [{"founder": f, "n": n, **form} for n in plan["n"] for f in plan["founders"]
             for form in plan.get("forms", [{"form": "code", "width": 128}])]
    cell = cells[index]
    if cell.get("form") == "hashed":
        cell["width"] = ml.HASHED_WIDTH
    return {**{k: v for k, v in plan.items() if k not in ("n", "founders", "forms")}, **cell}


def name_of(cell: dict[str, Any]) -> str:
    return f"{cell['form']}{cell['width']}-n{cell['n']}-f{cell['founder']}"


def cues_of(cell: dict[str, Any], keys_of: Any, count: int) -> dict[str, tuple[np.ndarray,
                                                                               np.ndarray]]:
    """The cell's cues: for each kind the records asked for and the cues, drawn from the seed
    ``[f, 310, 7, N, width]``; ``keys_of(records)`` gives the records' keys."""
    founder, n, form, width = cell["founder"], cell["n"], cell["form"], cell["width"]
    rng = np.random.default_rng(np.random.SeedSequence([founder, 310, 7, n, width]))
    trials = int(cell.get("trials", 1000))
    out = {}
    for kind in KINDS:
        wanted = np.zeros(0, dtype=np.int64)
        while len(wanted) < trials:  # records outside every family, as million.py asks
            drawn = rng.integers(min(count, n), size=2 * trials)
            wanted = np.concatenate([wanted, drawn[single(n, form, drawn)]])[:trials]
        keys = keys_of(wanted)
        out[kind] = (wanted, np.stack([ml.make_cue(rng, key, kind, form) for key in keys]))
    return out


def scan_parts(keys: np.ndarray, count: int, cues: np.ndarray) -> np.ndarray:
    """Each cue's record of largest cosine among the first ``count`` rows of ``keys``, scored by
    parts; among equals the earlier record."""
    cues32 = np.ascontiguousarray(np.asarray(cues, dtype=np.float32).T)
    best = np.full(cues32.shape[1], -np.inf, dtype=np.float32)
    index = np.zeros(cues32.shape[1], dtype=np.int64)
    step = max(1, min(PART, (1 << 28) // max(cues32.shape[1], 1)))
    for start in range(0, count, step):
        block = np.asarray(keys[start : start + min(step, count - start)])
        scores = block @ cues32
        norm = np.sqrt(np.einsum("ij,ij->i", block, block, dtype=np.float64)).astype(np.float32)
        np.divide(scores, norm[:, None], out=scores, where=norm[:, None] > 0)
        top = np.argmax(scores, axis=0)
        value = scores[top, np.arange(scores.shape[1])]
        better = value > best
        best[better], index[better] = value[better], top[better] + start
    return index


# -- the steps


def build(cell: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    from cadence.longterm import LongTermMemory

    founder, n, form, width = cell["founder"], cell["n"], cell["form"], cell["width"]
    value_width = int(cell.get("value_width", 64))
    folder = root / name_of(cell)
    if folder.exists():
        shutil.rmtree(folder)
    kept = settings(cell, n)
    if cell.get("pass_bytes"):
        LongTermMemory.PASS = int(cell["pass_bytes"])  # type: ignore[misc]
    t0 = time.perf_counter()
    store = LongTermMemory(width, value_width, int(cell.get("capacity", 2 * n)), seed=founder,
                           grow=True, path=str(folder), **kept)
    drawn = filled = filed = grown = 0.0
    parts = -(-n // PART)
    first = max(1, round((1.0 - float(cell.get("grown", 0.1))) * parts))  # parts filed at once
    cells_first = 0
    for part in range(parts):
        t1 = time.perf_counter()
        keys, values = draw_part(founder, n, form, width, value_width, part)
        t2 = time.perf_counter()
        store.fill(keys, values, file=False, chunk=int(cell.get("chunk", 512)))
        drawn, filled = drawn + t2 - t1, filled + time.perf_counter() - t2
        if part + 1 == first:
            t1 = time.perf_counter()
            store.file()
            filed = time.perf_counter() - t1
            cells_first = int(store._level + store._split)
    t1 = time.perf_counter()
    store.file()  # the rest, onto the table as it stands: its cells split in turn
    grown = time.perf_counter() - t1
    t1 = time.perf_counter()
    store.close()
    flushed = time.perf_counter() - t1
    return [{
        **{k: cell[k] for k in ("founder", "n", "form", "width")}, "step": "build",
        "arm": "disk", "settings": kept, "capacity": store.capacity,
        "seconds": round(time.perf_counter() - t0, 1), "draw_s": round(drawn, 1),
        "fill_s": round(filled, 1), "file_s": round(filed, 1), "grow_s": round(grown, 1),
        "cells_before_growth": cells_first, "flush_s": round(flushed, 1),
        "fill_us_per_record": round(1e6 * (filled + filed + grown) / n, 2),
        "bytes": int(store.nbytes), "bytes_per_record": round(store.nbytes / n, 1),
        "disk_bytes": on_disk(folder), "disk_bytes_per_record": round(on_disk(folder) / n, 1),
        "cells": int(store._level + store._split), "entries": int(store.entries),
        "dropped": int(store.dropped), "fixed_weights": int(store.projection.size),
        "threads": {name: os.environ.get(name) for name in
                    ("OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
    }]


def timed(store: Any, cues: np.ndarray, wanted: np.ndarray) -> dict[str, Any]:
    """Each cue read once: found, the candidates, ms per read (mean, median, worst) and the
    page faults a read waited for."""
    first, among, each = [], [], []
    before = faults()
    for cue, record in zip(cues, wanted, strict=True):
        t0 = time.perf_counter()
        read = store.read(cue, credit=False)
        store.value(read)
        each.append(time.perf_counter() - t0)
        first.append(read.first() == record)
        among.append(len(read.slots))
    ms = 1e3 * np.asarray(each)
    return {"found": round(float(np.mean(first)), 4), "read_ms": round(float(ms.mean()), 4),
            "read_ms_median": round(float(np.median(ms)), 4),
            "read_ms_worst": round(float(ms.max()), 3),
            "scored": round(float(np.mean(among)), 1),
            "faults_per_read": round((faults() - before) / max(len(cues), 1), 1)}


def read(cell: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    from cadence.longterm import LongTermMemory

    founder, n, form, width = cell["founder"], cell["n"], cell["form"], cell["width"]
    value_width = int(cell.get("value_width", 64))
    folder = root / name_of(cell)
    if "scattered" in cell:
        LongTermMemory.SCATTERED = bool(cell["scattered"])  # type: ignore[misc]
    t0 = time.perf_counter()
    store = LongTermMemory.open(folder)
    opened = time.perf_counter() - t0
    base = {**{k: cell[k] for k in ("founder", "n", "form", "width")},
            "memory_limit": memory_limit(), "records": int(store.count),
            "open_ms": round(1e3 * opened, 2),
            "bytes_per_record": round(store.nbytes / store.count, 1),
            "disk_bytes_per_record": round(on_disk(folder) / store.count, 1)}
    cues = cues_of(cell, lambda rows: np.asarray(store.keys[np.sort(rows)])[
        np.argsort(np.argsort(rows))], store.count)
    rows: list[dict[str, Any]] = []
    timed_reads = int(cell.get("timed", 300))
    macs = int(width * (store.expansion + store.candidates))
    for label in ("first", "again"):  # the cues read once from the disk, then once more
        for kind in KINDS:
            wanted, asked = cues[kind]
            rows.append({**base, "step": "read", "arm": "disk", "pass": label, "cue": kind,
                         "trials": timed_reads if label == "again" else len(asked),
                         "macs": macs, "sure": store.sure,
                         **timed(store, asked[: timed_reads if label == "again" else None],
                                 wanted[: timed_reads if label == "again" else None])})
    if cell.get("scan", True):
        t0 = time.perf_counter()
        in_order = np.load(folder / "keys.npy", mmap_mode="r")  # read from end to end
        for kind in KINDS:
            wanted, asked = cues[kind]
            first = scan_parts(in_order, n, asked)
            rows.append({**base, "step": "read", "arm": "scan", "cue": kind,
                         "trials": len(asked), "found": round(float(np.mean(first == wanted)), 4),
                         "macs": n * width, "scored": n})
        rows[-1]["scan_seconds"] = round(time.perf_counter() - t0, 1)
    writes = int(cell.get("writes", 0))
    if writes:
        rng = np.random.default_rng(np.random.SeedSequence([founder, 310, 8, n, width]))
        if form == "hashed":
            new_keys = ml.hashed_records(rng, writes)
        else:
            new_keys = ml.unit(rng.standard_normal((writes, width), dtype=np.float32))
        new_values = ml.unit(rng.standard_normal((writes, value_width), dtype=np.float32))
        cells_before, held_before = store._level + store._split, store.count
        pauses, between = [], []
        before = faults()
        earlier, asked = cues["exact"]
        for i in range(writes):
            t0 = time.perf_counter()
            store.write(new_keys[i], new_values[i], recognised=-1)
            pauses.append(time.perf_counter() - t0)
            t0 = time.perf_counter()  # the store is read between the writes
            between.append(store.read(asked[i % len(asked)], credit=False).first()
                           == earlier[i % len(asked)])
        written_faults = faults() - before
        ms = 1e3 * np.asarray(pauses)
        found_new = [store.read(new_keys[i], credit=False).first() == held_before + i
                     for i in range(0, writes, max(1, writes // 1000))]
        found_old = [store.read(c, credit=False).first() == r
                     for c, r in zip(asked, earlier, strict=True)]
        t0 = time.perf_counter()
        store.flush()
        rows.append({**base, "step": "write", "arm": "disk", "writes": writes,
                     "write_ms": round(float(ms.mean()), 4),
                     "write_ms_median": round(float(np.median(ms)), 4),
                     "write_ms_p99": round(float(np.quantile(ms, 0.99)), 4),
                     "write_ms_worst": round(float(ms.max()), 3),
                     "faults_per_write": round(written_faults / writes, 1),
                     "cells_before": int(cells_before),
                     "cells_after": int(store._level + store._split),
                     "cells_per_write": round((store._level + store._split - cells_before)
                                              / writes, 2),
                     "found_new": round(float(np.mean(found_new)), 4),
                     "found_between": round(float(np.mean(between)), 4),
                     "found_earlier": round(float(np.mean(found_old)), 4),
                     "flush_ms": round(1e3 * (time.perf_counter() - t0), 1),
                     "records_after": int(store.count)})
    store.close()
    return rows


def memory(cell: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """The same records in memory: the table that grows, and the fixed table of ``pairs``."""
    from cadence.longterm import LongTermMemory

    founder, n, form, width = cell["founder"], cell["n"], cell["form"], cell["width"]
    value_width = int(cell.get("value_width", 64))
    kept = settings(cell, n)
    kept.pop("sure", None)
    rows: list[dict[str, Any]] = []
    for arm, more in (("grow", {"grow": True}), ("pairs", {})):
        if arm not in cell.get("memory_arms", ["grow", "pairs"]):
            continue
        store = LongTermMemory(width, value_width,
                               int(cell.get("capacity", 2 * n)) if arm == "grow" else n,
                               seed=founder, **kept, **more)
        t0 = time.perf_counter()
        for part in range(-(-n // PART)):
            keys, values = draw_part(founder, n, form, width, value_width, part)
            if arm == "grow":
                store.fill(keys, values, file=False, chunk=int(cell.get("chunk", 512)))
            else:
                store.fill(keys, values, chunk=int(cell.get("chunk", 512)))
        store.file()
        fill_s = time.perf_counter() - t0
        cues = cues_of(cell, lambda rows, store=store: np.asarray(store.keys[rows]), n)
        base = {**{k: cell[k] for k in ("founder", "n", "form", "width")}, "step": "memory",
                "arm": arm, "settings": kept, "fill_s": round(fill_s, 1),
                "fill_us_per_record": round(1e6 * fill_s / n, 2),
                "bytes_per_record": round(store.nbytes / n, 1), "dropped": int(store.dropped),
                "macs": int(width * (store.expansion + store.candidates))}
        timed_reads = int(cell.get("timed", 300))
        for kind in KINDS:
            wanted, asked = cues[kind]
            whole = timed(store, asked, wanted)
            again = timed(store, asked[:timed_reads], wanted[:timed_reads])
            rows.append({**base, "cue": kind, "trials": len(asked), **whole,
                         "read_ms": again["read_ms"], "read_ms_median": again["read_ms_median"]})
        del store
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("step", choices=("build", "read", "memory"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cell = cell_of(json.loads(args.plan.read_text()), args.cell)
    rows = {"build": build, "read": read, "memory": memory}[args.step](cell, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
