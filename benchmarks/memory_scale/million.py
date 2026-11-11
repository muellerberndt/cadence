"""A million records (cadence#310, instrument d): does the store find the right record among
100,000, 1,000,000 and 10,000,000, and what does a read cost as the records grow?

One process per cell (a founder, N, a key form). The cell draws the records, then builds one arm
after another over the same records and reads each with the same cues:

* ``scan``, the read that scores every record: the control and the ceiling. Its first choice
  is the record whose key has the largest cosine with the cue, as ``cd.LongTermMemory`` with
  ``candidates=0`` chooses (a contract test holds the two together);
* every other arm of the plan, a ``cd.LongTermMemory`` with the arm's settings, written by
  ``fill``: the address of units as the store stands, and the address of pairs.

**Key forms.** ``code``: unit codes of ``width`` coordinates, a quarter of them in families of 8
at cosine .8 (``world``'s law, drawn in float32 by parts so that ten million fit). ``hashed``:
the fixed code's form, 280 coordinates of which 15 to 22 are one and the rest zero, the units
drawn at random as a hash of a sentence gives them.

**Cues.** ``exact``; ``noisy`` (a code with noise of its own energy, cosine .71; a hashed key
with three in ten of its active units exchanged); ``partial`` (half of the coordinates silent);
``similar_noisy`` (a family member's code with the noise). Several records for one question:
``family`` (a family's shared direction, read by k heads), ``superposed`` (the unit sum of k
unrelated codes, read by k heads) and ``each`` (the same k codes, each head with a cue of its
own, ``read_each``).

**found.** Every read is at temperature 0, so a read holds its first choice whole: a cue is
found when that first choice is the record. For k heads a trial is found when the heads hold k
different relevant records.

Every row carries the arm's bytes, ms per read on one thread, the records a read scores and the
table entries it reads, and the multiply-accumulates of a read.

    PYTHONPATH=src python benchmarks/memory_scale/million.py --plan plan.json --out rows.jsonl
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

FAMILY, SHARE, FAMILIES = 8, 0.8, 0.25  # world's law of families
HASHED_WIDTH, HASHED_ACTIVE = 280, (15, 22)
HEADS = (2, 4, 8)
PART = 65_536  # records drawn or scored at once


def unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.asarray(x / np.where(norm > 0, norm, 1.0), dtype=x.dtype)


def code_records(rng: np.random.Generator, n: int, width: int) -> tuple[np.ndarray, np.ndarray,
                                                                       np.ndarray]:
    """``n`` unit codes in float32, a quarter in families: the keys, each record's family (-1
    for none) and the families' shared directions."""
    keys = np.empty((n, width), dtype=np.float32)
    for start in range(0, n, PART):
        block = rng.standard_normal((min(PART, n - start), width), dtype=np.float32)
        keys[start : start + len(block)] = unit(block)
    families = int(n * FAMILIES) // FAMILY
    themes = unit(rng.standard_normal((max(families, 1), width), dtype=np.float32))[:families]
    family = np.full(n, -1, dtype=np.int32)
    rows = rng.permutation(n)[: families * FAMILY].reshape(families, FAMILY)
    for start in range(0, families, PART // FAMILY):
        part = rows[start : start + PART // FAMILY]
        mixed = np.sqrt(SHARE) * themes[start : start + len(part), None, :] \
            + np.sqrt(1 - SHARE) * keys[part]
        keys[part] = unit(mixed.astype(np.float32))
        family[part] = np.arange(start, start + len(part), dtype=np.int32)[:, None]
    return keys, family, themes


def hashed_records(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` keys in the fixed code's form: 280 coordinates, 15 to 22 of them one."""
    low, high = HASHED_ACTIVE
    keys = np.zeros((n, HASHED_WIDTH), dtype=np.float32)
    for start in range(0, n, PART):
        many = min(PART, n - start)
        order = np.argsort(rng.random((many, HASHED_WIDTH), dtype=np.float32), axis=1)[:, :high]
        active = rng.integers(low, high + 1, size=many)
        on = np.arange(high)[None, :] < active[:, None]
        block = keys[start : start + many]
        block[np.repeat(np.arange(many), high)[on.ravel()], order[on]] = 1.0
    return keys


def make_cue(rng: np.random.Generator, key: np.ndarray, kind: str, form: str) -> np.ndarray:
    """One cue of ``kind`` for a record's ``key``."""
    key = np.asarray(key, dtype=np.float64)
    if kind == "exact":
        return key.copy()
    if kind == "partial":
        out = key.copy()
        out[rng.permutation(len(key))[: len(key) // 2]] = 0.0
        return out
    if form == "hashed":  # three in ten of the active units exchanged
        out = key.copy()
        on, off = np.flatnonzero(key > 0), np.flatnonzero(key == 0)
        moved = int(round(0.3 * len(on)))
        out[rng.choice(on, size=moved, replace=False)] = 0.0
        out[rng.choice(off, size=moved, replace=False)] = 1.0
        return out
    noisy = key + rng.standard_normal(len(key)) / np.sqrt(len(key))
    return np.asarray(noisy / np.linalg.norm(noisy))


def scan(keys: np.ndarray, cues: np.ndarray, k: int = 1) -> np.ndarray:
    """The ``k`` records of largest cosine with each cue, the largest first, ``(cues, k)``:
    every record scored, by parts. Among equals the record written first leads."""
    cues32 = np.ascontiguousarray(np.asarray(cues, dtype=np.float32).T)
    many = cues32.shape[1]
    best = np.full((k, many), -np.inf, dtype=np.float32)
    index = np.zeros((k, many), dtype=np.int64)
    part = max(PART // 4, 1)
    for start in range(0, len(keys), part):
        block = keys[start : start + part]
        scores = block @ cues32
        norm = np.sqrt(np.einsum("ij,ij->i", block, block, dtype=np.float64)).astype(np.float32)
        np.divide(scores, norm[:, None], out=scores, where=norm[:, None] > 0)
        take = min(k, len(block))
        top = np.argpartition(-scores, take - 1, axis=0)[:take] if take < len(block) \
            else np.argsort(-scores, axis=0, kind="stable")[:take]
        merged = np.concatenate([best, np.take_along_axis(scores, top, axis=0)])
        where = np.concatenate([index, top + start])
        # the best k of the held and the new, the earlier record first among equals
        order = np.lexsort((where, -merged), axis=0)[:k]
        best = np.take_along_axis(merged, order, axis=0)
        index = np.take_along_axis(where, order, axis=0)
    return np.asarray(index.T)


def scan_read(keys: np.ndarray, norms: np.ndarray, cue: np.ndarray) -> int:
    """One read that scores every record, as the store's read without an address computes it:
    the keys' product with the cue, over the keys' lengths, and its first choice."""
    scores = keys @ cue.astype(keys.dtype)
    return int(np.argmax(scores / norms))


def _arm(spec: dict[str, Any], width: int, value_width: int, n: int, founder: int) -> Any:
    from cadence.longterm import LongTermMemory

    settings = {"cosine": True, "fade": False, "revise": 0.0, "dtype": "float32",
                **spec.get("settings", {})}
    return LongTermMemory(width, value_width, n, seed=founder, **settings)


def _cost(arm: Any, cue: np.ndarray) -> tuple[int, int]:
    """The records a read of ``cue`` scores and the table entries its address reads."""
    if not arm.coarse:
        return arm.count, 0
    units = arm.address(cue)
    if arm.pairs:
        return len(arm._named(units)), len(units) * (len(units) - 1) // 2 * arm.DEPTH
    return len(arm._driven(units)), int(arm.counts[units].sum())


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    founder, n = int(spec["founder"]), int(spec["n"])
    form, width = spec.get("form", "code"), int(spec.get("width", 128))
    value_width = int(spec.get("value_width", 64))
    trials, head_trials = int(spec.get("trials", 1000)), int(spec.get("head_trials", 200))
    timed = int(spec.get("timed", 200))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 310, n, width]))
    if form == "hashed":
        width = HASHED_WIDTH
        keys = hashed_records(rng, n)
        family, themes = np.full(n, -1, dtype=np.int32), np.zeros((0, width), dtype=np.float32)
    else:
        keys, family, themes = code_records(rng, n, width)
    values = np.empty((n, value_width), dtype=np.float32)
    for start in range(0, n, PART):
        block = rng.standard_normal((min(PART, n - start), value_width), dtype=np.float32)
        values[start : start + len(block)] = unit(block)
    singles, members = np.flatnonzero(family < 0), np.flatnonzero(family >= 0)
    kinds = ["exact", "noisy", "partial"] + (["similar_noisy"] if len(members) else [])
    wanted = {kind: (members if kind == "similar_noisy" else singles)[
        rng.integers(len(members if kind == "similar_noisy" else singles), size=trials)]
        for kind in kinds}
    cues = {kind: np.stack([make_cue(rng, keys[r], "noisy" if kind == "similar_noisy" else kind,
                                     form) for r in rows]) for kind, rows in wanted.items()}
    plan_heads: dict[tuple[str, int], tuple[np.ndarray, list[np.ndarray]]] = {}
    for k in HEADS:
        if len(themes):
            chosen = rng.integers(len(themes), size=head_trials)
            plan_heads["family", k] = (
                np.asarray(themes[chosen], dtype=np.float64),
                [np.flatnonzero(family == f) for f in chosen],
            )
        if form == "code":
            rows = np.stack([rng.choice(singles, size=k, replace=False)
                             for _ in range(head_trials)])
            total = keys[rows].astype(np.float64).sum(axis=1)
            plan_heads["superposed", k] = (
                total / np.linalg.norm(total, axis=1, keepdims=True), list(rows))
            plan_heads["each", k] = (np.zeros((head_trials, width)), list(rows))
    base = {"founder": founder, "n": n, "form": form, "width": width,
            "value_width": value_width}
    rows_out: list[dict[str, Any]] = []
    record_bytes = int(keys[0].nbytes + values[0].nbytes)

    def emit(arm: str, cue: str, heads: int, **more: Any) -> None:
        rows_out.append({**base, "arm": arm, "cue": cue, "heads": heads, **more})

    def relevant(picks: list[int], want: np.ndarray) -> bool:
        return len(set(picks)) == len(picks) and set(picks) <= set(want.tolist())

    for arm_spec in spec["arms"]:
        name = arm_spec["name"]
        if n > int(arm_spec.get("max_n", n)):
            continue
        if name == "scan":
            norms = np.sqrt(np.einsum("ij,ij->i", keys, keys, dtype=np.float64)).astype(
                np.float32)
            for kind in kinds:
                first = scan(keys, cues[kind])[:, 0]
                t0 = time.perf_counter()
                for c in cues[kind][: max(timed // 10, 5)]:
                    scan_read(keys, norms, c)
                ms = 1e3 * (time.perf_counter() - t0) / max(timed // 10, 5)
                emit(name, kind, 1, trials=trials,
                     found=round(float(np.mean(first == wanted[kind])), 4),
                     read_ms=round(ms, 4), bytes=n * record_bytes + norms.nbytes,
                     bytes_per_record=record_bytes + 4, scored=n, touched=0, macs=n * width)
            for (kind, k), (heads_cues, want) in plan_heads.items():
                if kind == "each":
                    picks = scan(keys, keys[np.concatenate(want)].astype(np.float64))[:, 0]
                    picks = picks.reshape(len(want), k)
                else:
                    picks = scan(keys, heads_cues, k)
                found = [relevant(list(p), w) for p, w in zip(picks.tolist(), want, strict=True)]
                emit(name, kind, k, trials=len(want), found=round(float(np.mean(found)), 4))
            continue
        arm = _arm(arm_spec, width, value_width, n, founder)
        t0 = time.perf_counter()
        arm.fill(keys, values)
        fill_s = time.perf_counter() - t0
        held = int(arm.nbytes)
        for kind in kinds:
            first, among = [], []
            t0 = time.perf_counter()
            for c in cues[kind][:timed]:
                arm.read(c, credit=False)
            ms = 1e3 * (time.perf_counter() - t0) / min(timed, trials)
            for c, r in zip(cues[kind], wanted[kind], strict=True):
                read = arm.read(c, credit=False)
                first.append(read.first() == r)
                among.append(bool(np.isin(r, read.slots)))
            costs = np.array([_cost(arm, c) for c in cues[kind][:50]])
            scored, touched = (float(v) for v in costs.mean(axis=0))
            emit(name, kind, 1, trials=trials, found=round(float(np.mean(first)), 4),
                 candidate=round(float(np.mean(among)), 4), read_ms=round(ms, 4),
                 fill_s=round(fill_s, 2), bytes=held, bytes_per_record=round(held / n, 1),
                 scored=round(scored, 1), touched=round(touched, 1),
                 macs=int(width * ((arm.expansion if arm.coarse else 0) + scored)),
                 dropped=int(arm.dropped), settings=arm_spec.get("settings", {}))
        for (kind, k), (heads_cues, want) in plan_heads.items():
            found, seconds = [], 0.0
            for i, w in enumerate(want):
                t0 = time.perf_counter()
                if kind == "each":
                    read = arm.read_each(keys[w].astype(np.float64), credit=False)
                else:
                    read = arm.read(heads_cues[i], heads=k, credit=False)
                seconds += time.perf_counter() - t0
                found.append(relevant([read.first(h) for h in range(k)], w))
            emit(name, kind, k, trials=len(want), found=round(float(np.mean(found)), 4),
                 read_ms=round(1e3 * seconds / max(len(want), 1), 4),
                 settings=arm_spec.get("settings", {}))
        del arm
    return rows_out


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """The plan's cells: every founder, N and key form, the largest first."""
    out = []
    for founder in plan["founders"]:
        for n in plan["n"]:
            for form in plan.get("forms", [{"form": "code", "width": 128}]):
                out.append({"founder": founder, "n": n, **form, "arms": plan["arms"],
                            **{name: plan[name] for name in
                               ("trials", "head_trials", "timed", "value_width")
                               if name in plan}})
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
