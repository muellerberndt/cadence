"""Retrieval among facts keyed by held meanings (cadence#288, the second round): does a read
return the right fact among N whose keys ``cadence.Message`` made, and what binds first?

One process per cell (a founder, N, the code's units and fan-in). The cell draws N facts about
different agent-verb pairs (``meanings.facts``), tells them to every arm in one order, and reads
each arm with the same cues:

* ``exact``, an exact store over the same keys: the fact whose key shares the most active units
  with the cue, and among facts tied for the best the one told last. It is the ceiling: no read
  by overlap finds a fact it does not;
* ``memory``, the library's ``cadence.LongTermMemory`` as it stands (every slot scored up to
  16,384 records, the sparse address past that), the key as ``Message`` gives it and the fact's
  held message as its value;
* ``memory-direct``, the same with every slot scored at every size.

**found.** A read returns the right fact when the fact's own record holds more than half of the
read's mass. ``said`` is the share of reads whose value holds the fact's patient. ``single``, on
the exact store, is the share of cues whose best overlap the right fact holds alone: the rest
are ties, which a read at temperature 0 settles by slot and the exact store by the fact told
last. ``held`` is the share of the facts that keep a record of their own: a fact whose key
another fact's write recognised (``revise``) has none. For several heads the cue is an agent's
moment alone, and a trial counts when the k heads return k different facts about that agent;
agents with at least k stored facts are asked.

    PYTHONPATH=src python benchmarks/memory_scale/meaning_keys.py --plan plan.json --out rows.jsonl
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
HALF = 0.5


def latest_best(scores: np.ndarray, k: int = 1) -> np.ndarray:
    """The ``k`` records of the greatest score, the best first, and among equals the one told
    last first."""
    order = np.lexsort((-np.arange(len(scores)), -scores))
    return np.asarray(order[:k])


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import meanings as mn

    from cadence import LongTermMemory

    founder, n, units = int(spec["founder"]), int(spec["n"]), int(spec["units"])
    fan = spec.get("fan_in")
    trials, head_trials = int(spec.get("trials", 1000)), int(spec.get("head_trials", 200))
    lexicon = mn.Lexicon(int(spec.get("nouns", mn.NOUNS)), int(spec.get("verbs", mn.VERBS)))
    coder = mn.Coder(lexicon, units, founder, fan)
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, 97, n]))
    found = mn.facts(rng, coder, n)
    asked = rng.integers(n, size=trials)
    cues = {kind: [mn.cue(rng, coder, found, int(r), kind) for r in asked] for kind in mn.CUES}
    # the agents asked of several heads, and the facts about each
    by_agent: dict[int, np.ndarray] = {}
    order = np.argsort(found.agent, kind="stable")
    bounds = np.flatnonzero(np.diff(found.agent[order])) + 1
    for group in np.split(order, bounds):
        by_agent[int(found.agent[group[0]])] = group
    head_plan: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for k in HEADS:
        rich = [a for a, rows in by_agent.items() if len(rows) >= k]
        if rich:
            chosen = rng.choice(rich, size=head_trials)
            head_plan[k] = [(coder.about_agent(np.array([a]))[0], by_agent[int(a)])
                            for a in chosen]
    distinct = int(len(np.unique(found.key.astype(np.uint8), axis=0)))
    base = {"founder": founder, "n": n, "units": units, "fan_in": coder.message.separator.fan_in,
            "nouns": lexicon.nouns, "verbs": lexicon.verbs, "keys": "message",
            "distinct_keys": round(distinct / n, 5),
            "active": round(float(found.key.sum(axis=1).mean()), 2)}
    rows: list[dict[str, Any]] = []

    def score(first: list[int], mass: list[float], said: list[int]) -> dict[str, float]:
        return {"trials": len(first), "first": round(float(np.mean(first)), 4),
                "found": round(float(np.mean(np.asarray(mass) > HALF)), 4),
                "said": round(float(np.mean(said)), 4)}

    def heads_score(picks: list[list[int]], relevant: list[np.ndarray]) -> dict[str, float]:
        whole, share = [], []
        for chosen, wanted in zip(picks, relevant, strict=True):
            good = len({r for r in chosen if r in wanted})
            whole.append(int(good == len(chosen) and len(set(chosen)) == len(chosen)))
            share.append(good / len(chosen))
        return {"trials": len(picks), "found": round(float(np.mean(whole)), 4),
                "relevant": round(float(np.mean(share)), 4)}

    for arm in spec.get("arms") or ["exact", "memory"]:
        name = arm if isinstance(arm, str) else arm["name"]
        if name == "exact":
            keys = np.ascontiguousarray(found.key)
            for kind in mn.CUES:
                first, said, single, seconds = [], [], [], 0.0
                for c, r in zip(cues[kind], asked, strict=True):
                    t0 = time.perf_counter()
                    scores = keys @ c
                    best = int(latest_best(scores)[0])
                    seconds += time.perf_counter() - t0
                    first.append(int(best == r))
                    told = coder.said(found.value[best][None, :])[0]
                    said.append(int(told == found.patient[r]))
                    top = scores.max()
                    single.append(int(scores[r] == top and (scores == top).sum() == 1))
                rows.append({**base, "arm": name, "cue": kind, "heads": 1,
                             **score(first, [float(f) for f in first], said),
                             "single": round(float(np.mean(single)), 4),
                             "read_ms": round(1e3 * seconds / trials, 4)})
            for k, plan in head_plan.items():
                picks = [[int(r) for r in latest_best(keys @ c, k)] for c, _ in plan]
                rows.append({**base, "arm": name, "cue": "agent", "heads": k,
                             **heads_score(picks, [w for _, w in plan])})
            del keys
            continue
        settings = {} if isinstance(arm, str) else dict(arm.get("settings", {}))
        if settings.get("direct") == "all":
            settings["direct"] = n
        memory = LongTermMemory(units, mn.PHASES * units, n, seed=founder, **settings)
        owner = np.full(n, -1, dtype=np.int64)
        t0 = time.perf_counter()
        for i in range(n):
            slot, _ = memory.write(found.key[i], found.value[i])
            owner[slot] = i
            memory.moment()
        write_ms = 1e3 * (time.perf_counter() - t0) / n
        where = np.full(n, -1, dtype=np.int64)
        where[owner[owner >= 0]] = np.flatnonzero(owner >= 0)
        for kind in mn.CUES:
            first, mass, said, seconds = [], [], [], 0.0
            for c, r in zip(cues[kind], asked, strict=True):
                t0 = time.perf_counter()
                read = memory.read(c, credit=False)
                value = memory.value(read)
                seconds += time.perf_counter() - t0
                at = int(where[r])
                first.append(int(at >= 0 and read.first() == at))
                mass.append(read.mass(at) if at >= 0 else 0.0)
                said.append(int(coder.said(value[None, :])[0] == found.patient[r]))
            rows.append({**base, "arm": name, "cue": kind, "heads": 1,
                         **score(first, mass, said),
                         "read_ms": round(1e3 * seconds / trials, 4),
                         "write_ms": round(write_ms, 4), "bytes": memory.nbytes,
                         "held": round(float(np.mean(where >= 0)), 5),
                         "coarse": bool(memory.coarse), "settings": settings})
        for k, plan in head_plan.items():
            picks = []
            for c, _ in plan:
                read = memory.read(c, heads=k, credit=False)
                picks.append([int(owner[read.first(h)]) if read.first(h) >= 0 else -1
                              for h in range(k)])
            rows.append({**base, "arm": name, "cue": "agent", "heads": k,
                         **heads_score(picks, [w for _, w in plan]),
                         "coarse": bool(memory.coarse), "settings": settings})
        del memory
    return rows


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for founder in plan["founders"]:
        for n in plan["n"]:
            for code in plan["codes"]:
                arms = [a for a in plan["arms"] if isinstance(a, str)
                        or int(a.get("min_n", n)) <= n <= int(a.get("max_n", n))]
                out.append({"founder": founder, "n": n, "units": code["units"],
                            "fan_in": code.get("fan_in"), "arms": arms,
                            **{k: plan[k] for k in ("trials", "head_trials", "nouns", "verbs")
                               if k in plan}})
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
