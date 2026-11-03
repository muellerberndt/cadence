"""The bandit of cadence#210: a reversal whose failures pay nothing, with and without
``memory_extinction``.

One constant observation and two arms. Arm 0 pays 1.0 for ``phase`` trials, then arm 1 for
``phase`` trials, then (with three phases) arm 0 again; the arm that does not pay pays ``fail``
(0.0 or -1.0). One continuing ``Brain.live`` stream per seed, ``Brain.compose(inputs=1,
actions=2, modules=(16,), seed=seed, arousal=True, memory_choice=True)`` with
``memory_extinction=theta`` beside it where theta is above 0.

A failure that pays 0 writes nothing to the episodic records, so without the rule the record
the first arm earned keeps voting for it after the reversal; with it, a zero outcome writes a
learned zero for a record the action earned by its own payments in this place.

Readings per life and phase: the hit rate of the phase's last 40 trials, the reversal lag (the
first trial from which 20 executed actions in a row include at least 18 of the paying arm), the
place record of each arm at the phase's end, the share of aroused trials, and in the later
phases the aroused failures of the arm that paid before and how many of them were written. A
phase is acquired when its last 40 trials hit at least 0.8.

    python benchmarks/extinction/bandit.py --thetas 0 0.5 --fails 0 -1 --seeds 0-11 \\
        --out results/bandit.jsonl --workers 4
    python benchmarks/extinction/bandit.py --summary results/bandit.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

OBS = np.array([[1.0]])
PHASE, PHASES = 120, 3
ACQUIRED = 0.8


def lag(hits: list[int]) -> int | None:
    h = np.asarray(hits)
    for k in range(0, len(h) - 20 + 1):
        if h[k:k + 20].sum() >= 18:
            return k
    return None


def life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``spec`` names ``seed``, ``theta`` and ``fail``, with ``phase`` and
    ``phases``."""
    warnings.simplefilter("ignore")
    from cadence import Brain

    seed, theta, fail = int(spec["seed"]), float(spec["theta"]), float(spec["fail"])
    phase, phases = int(spec.get("phase", PHASE)), int(spec.get("phases", PHASES))
    options: dict[str, Any] = {"memory_choice": True}
    if theta:
        options["memory_extinction"] = theta
    brain = Brain.compose(inputs=1, actions=2, modules=(16,), seed=seed, arousal=True, **options)
    stores = [m for m in (brain.hippocampus, brain.context_memory) if m is not None]
    code = brain.observation_code(OBS)
    action = int(brain.live(OBS)[0])
    out: dict[str, Any] = {"seed": seed, "theta": theta, "fail": fail, "phase": phase,
                           "phases": []}
    for p in range(phases):
        target = p % 2
        hits, aroused_fail, aroused_fail_writes, aroused = [], 0, 0, 0
        for _ in range(phase):
            reward = 1.0 if action == target else fail
            hits.append(int(action == target))
            was_aroused = brain.last_arousal["mode"] == "aroused"
            aroused += int(was_aroused)
            before = sum(m.writes for m in stores)
            following = int(brain.live(OBS, reward=[reward])[0])
            written = sum(m.writes for m in stores) - before
            if p and action == 1 - target and was_aroused:
                aroused_fail += 1
                aroused_fail_writes += written
            action = following
        record = brain.hippocampus.recall(code)[0]
        out["phases"].append({
            "target": target,
            "late": float(np.mean(hits[-40:])),
            "lag": lag(hits),
            "record": [float(record[0]), float(record[1])],
            "aroused_share": aroused / phase,
            "old_arm_aroused_failures": aroused_fail,
            "old_arm_aroused_failures_written": aroused_fail_writes,
        })
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per theta and fail: lives acquiring each phase, its mean late hit rate and median lag
    (a phase never acquired counts its length), and the first arm's record at the ends of the
    first two phases."""
    cells: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for r in rows:
        cells.setdefault((r["theta"], r["fail"]), []).append(r)
    out = []
    for (theta, fail), group in sorted(cells.items()):
        cell: dict[str, Any] = {"theta": theta, "fail": fail,
                                "seeds": sorted(r["seed"] for r in group)}
        for p in range(1, min(len(r["phases"]) for r in group)):
            lags = [r["phases"][p]["lag"] if r["phases"][p]["lag"] is not None else r["phase"]
                    for r in group]
            cell[f"phase_{p}"] = {
                "acquired": sum(r["phases"][p]["late"] >= ACQUIRED for r in group),
                "late": round(float(np.mean([r["phases"][p]["late"] for r in group])), 3),
                "lag_median": float(np.median(lags)),
            }
        cell["first_arm_record"] = [
            round(float(np.mean([r["phases"][p]["record"][0] for r in group])), 3)
            for p in range(min(2, len(group[0]["phases"])))
        ]
        out.append(cell)
    return out


def _seeds(text: list[str]) -> list[int]:
    seeds: list[int] = []
    for part in text:
        a, _, b = part.partition("-")
        seeds.extend(range(int(a), int(b or a) + 1))
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--thetas", nargs="+", type=float, default=[0.0, 0.5])
    parser.add_argument("--fails", nargs="+", type=float, default=[0.0, -1.0])
    parser.add_argument("--seeds", nargs="+", default=["0-11"], help="seeds and ranges a-b")
    parser.add_argument("--phase", type=int, default=PHASE)
    parser.add_argument("--phases", type=int, default=PHASES)
    parser.add_argument("--out", type=Path, help="JSON lines, one row per life, appended")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary", type=Path, help="print the summary of a results file")
    args = parser.parse_args(argv)
    if args.summary is not None:
        rows = [json.loads(line) for line in args.summary.read_text().splitlines() if line]
        for cell in summarize(rows):
            print(json.dumps(cell))
        return 0
    if args.out is None:
        parser.error("--out is required")
    specs = [{"seed": s, "theta": t, "fail": f, "phase": args.phase, "phases": args.phases}
             for t in args.thetas for f in args.fails for s in _seeds(args.seeds)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for row in map(life, specs):
                sink.write(json.dumps(row) + "\n")
                sink.flush()
        else:
            with get_context("spawn").Pool(min(args.workers, len(specs))) as pool:
                for row in pool.imap_unordered(life, specs, chunksize=1):
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
