"""Capacity and retrieval (cadence#288, instrument a): does a read return the right record
among N stored ones, and what does it cost?

One process per cell (a founder, N, a key width, a key source). The cell draws the records
(``world.records``), stores them in every arm and reads each arm with the same cues:

* ``exact``, the exact nearest-neighbour store (``references.Exact``), the ceiling;
* ``library``, the library's recall memory (``library.Slots``) as a new brain holds it, and
  ``library@T`` at another read temperature;
* every candidate bank of the plan (``bank.Bank``), by name.

**found.** A read returns the right record when the record holds more than half of the read's
mass. ``first`` is the share of reads whose first choice is the record, and ``mass`` the mean
mass on it. For several heads a trial is found when every head holds more than half of its
mass on one relevant record and the k records are different; ``relevant`` is the mean share of
the heads whose first choice is a relevant record no earlier head chose.

Seconds are per read and per write on one thread, and bytes are what the arm holds.

    PYTHONPATH=src python benchmarks/memory_scale/retrieval.py --plan plan.json --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

HEADS = (2, 4, 8)
HALF = 0.5  # the mass above which a read is the record's


def _score(first: list[int], mass: list[float]) -> dict[str, float]:
    return {
        "trials": len(first),
        "first": round(float(np.mean(first)), 4) if first else float("nan"),
        "found": round(float(np.mean(np.asarray(mass) > HALF)), 4) if mass else float("nan"),
        "mass": round(float(np.mean(mass)), 4) if mass else float("nan"),
    }


def _heads_score(
    picks: list[list[tuple[int, float]]], relevant: list[np.ndarray]
) -> dict[str, float]:
    """``picks`` holds, per trial and head, the head's first choice (a record) and its mass."""
    found, share = [], []
    for heads, wanted in zip(picks, relevant, strict=True):
        seen: set[int] = set()
        good = 0
        whole = True
        for record, mass in heads:
            ok = record in wanted and record not in seen
            seen.add(record)
            good += int(ok)
            whole = whole and ok and mass > HALF
        found.append(int(whole))
        share.append(good / max(len(heads), 1))
    return {"trials": len(picks), "found": round(float(np.mean(found)), 4),
            "relevant": round(float(np.mean(share)), 4)}


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import world
    from bank import Bank
    from library import Slots
    from references import Exact

    founder, n, width = int(spec["founder"]), int(spec["n"]), int(spec["width"])
    value_width = int(spec.get("value_width", width))
    keys = spec.get("keys", "code")
    trials, head_trials = int(spec.get("trials", 1000)), int(spec.get("head_trials", 200))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, n, width]))
    recs = world.records(rng, n, width, value_width, keys)
    plan_cues = {kind: world.targets(rng, recs, kind, trials) for kind in world.CUES}
    cues = {kind: [world.cue(rng, recs.address[r], kind) for r in rows]
            for kind, rows in plan_cues.items()}
    head_plan: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for kind in world.HEAD_CUES:
        for k in HEADS:
            if (kind == "family" and len(recs.themes)) or (
                kind == "superposed" and len(recs.singletons) >= k
            ):
                head_plan[kind, k] = [world.head_trial(rng, recs, kind, k)
                                      for _ in range(head_trials)]
    base = {"founder": founder, "n": n, "width": width, "value_width": value_width, "keys": keys}
    rows: list[dict[str, Any]] = []

    def emit(arm: str, cue: str, heads: int, score: dict[str, float], **more: Any) -> None:
        rows.append({**base, "arm": arm, "cue": cue, "heads": heads, **score, **more})

    arms = spec.get("arms") or ["exact", "library"]
    for arm in arms:
        name = arm if isinstance(arm, str) else arm["name"]
        if name == "exact":
            store = Exact(recs.address)
            for kind, wanted in plan_cues.items():
                t0 = time.perf_counter()
                first = [int(store.top(c)[0] == r) for c, r in zip(cues[kind], wanted, strict=True)]
                ms = 1e3 * (time.perf_counter() - t0) / max(len(first), 1)
                emit(name, kind, 1, _score(first, [float(f) for f in first]),
                     read_ms=round(ms, 4), bytes=store.nbytes)
            for (kind, k), plan in head_plan.items():
                picks = [[(int(r), 1.0) for r in store.top(c, k)] for c, _ in plan]
                emit(name, kind, k, _heads_score(picks, [w for _, w in plan]))
            continue
        if name.startswith("library"):
            options = {} if isinstance(arm, str) else dict(arm.get("options", {}))
            for heads in (1, max(HEADS)):
                t0 = time.perf_counter()
                slots = Slots(founder, width, value_width, n, heads=heads, **options)
                for i in range(n):
                    slots.write(recs.address[i], recs.content[i])
                write_ms = 1e3 * (time.perf_counter() - t0) / n
                if heads == 1:
                    for kind, wanted in plan_cues.items():
                        first, mass, seconds = [], [], 0.0
                        for c, r in zip(cues[kind], wanted, strict=True):
                            t0 = time.perf_counter()
                            p = slots.read(c)[0]
                            seconds += time.perf_counter() - t0
                            at = slots.slot(int(r))
                            first.append(int(int(np.argmax(p)) == at))
                            mass.append(float(p[at]))
                        emit(name, kind, 1, _score(first, mass),
                             read_ms=round(1e3 * seconds / max(len(first), 1), 4),
                             write_ms=round(write_ms, 4), bytes=slots.nbytes,
                             temperature=slots.recall.temperature)
                else:
                    offset = slots.slot(0)
                    for (kind, k), plan in head_plan.items():
                        picks = []
                        for c, _ in plan:
                            ps = slots.read_heads(c, k)
                            picks.append([(int(np.argmax(p)) - offset, float(p.max())) for p in ps])
                        emit(name, kind, k, _heads_score(picks, [w for _, w in plan]),
                             temperature=slots.recall.temperature)
                del slots
            continue
        settings = dict(arm.get("settings", {}))
        bank = Bank(width, value_width, n, seed=founder, **settings)
        owner = np.full(n, -1, dtype=np.int64)
        t0 = time.perf_counter()
        for i in range(n):
            slot, _ = bank.write(recs.address[i], recs.content[i])
            owner[slot] = i
            bank.moment()
        write_ms = 1e3 * (time.perf_counter() - t0) / n
        where = np.full(n, -1, dtype=np.int64)
        where[owner[owner >= 0]] = np.flatnonzero(owner >= 0)
        held = float(np.mean(where >= 0))
        for kind, wanted in plan_cues.items():
            first, mass, among, seconds = [], [], [], 0.0
            for c, r in zip(cues[kind], wanted, strict=True):
                t0 = time.perf_counter()
                read = bank.read(c, credit=False)
                seconds += time.perf_counter() - t0
                at = int(where[r])
                first.append(int(read.first() == at))
                mass.append(read.mass(at))
                among.append(int(at in read.slots))
            emit(name, kind, 1, _score(first, mass),
                 read_ms=round(1e3 * seconds / max(len(first), 1), 4),
                 write_ms=round(write_ms, 4), bytes=bank.nbytes, held=round(held, 4),
                 candidate=round(float(np.mean(among)), 4) if among else float("nan"),
                 settings=settings)
        for (kind, k), plan in head_plan.items():
            picks, seconds = [], 0.0
            for c, _ in plan:
                t0 = time.perf_counter()
                read = bank.read(c, heads=k, credit=False)
                seconds += time.perf_counter() - t0
                picks.append([
                    (int(owner[read.first(h)]) if read.first(h) >= 0 else -1,
                     float(read.weights[h].max()) if len(read.weights[h]) else 0.0)
                    for h in range(k)
                ])
            emit(name, kind, k, _heads_score(picks, [w for _, w in plan]),
                 read_ms=round(1e3 * seconds / max(len(plan), 1), 4), settings=settings)
        del bank
    return rows


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """The plan's cells: every founder, N, key width and key source, with the plan's arms."""
    out = []
    for founder in plan["founders"]:
        for n in plan["n"]:
            for width in plan.get("widths", [64]):
                for keys in plan.get("keys", ["code"]):
                    arms = [a for a in plan["arms"]
                            if isinstance(a, str) or n <= int(a.get("max_n", n))]
                    out.append({"founder": founder, "n": n, "width": width, "keys": keys,
                                "arms": arms, "trials": plan.get("trials", 1000),
                                "head_trials": plan.get("head_trials", 200)})
    return sorted(out, key=lambda c: -c["n"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    todo = cells(json.loads(args.plan.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink, get_context("spawn").Pool(args.workers,
                                                               maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(run_cell, todo, chunksize=1):
            for row in rows:
                sink.write(json.dumps(row) + "\n")
            sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
