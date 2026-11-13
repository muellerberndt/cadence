"""Facts with a key of parts (cadence#310, instrument e): several records for one question,
questions of several parts, chains, and facts whose keys are one key.

**The world.** ``N / 8`` entities, each a unit code of ``width`` coordinates, and 64 relations,
each a unit code of ``width`` coordinates. Every entity is the subject of 8 facts with 8 different
relations, and a fact's object is another entity. A fact's value is its object's code, so what a
read holds can be the subject of the next cue. The instrument lays the codes out; nothing here is
learned.

**Key forms.** ``parts``: the subject's code beside the relation's, on coordinates of their own
(a key of two parts, ``2 * width`` coordinates). ``superposed``: the unit sum of the two codes on
the same coordinates, the control.

**Readings**, each for the read that scores every record (``scan``) and for each store of the
plan:

* ``fact``: one fact from its subject and relation, exact and with noise of the cue's own
  energy;
* ``subject``: the subject alone, read by k heads; found when the heads hold k different facts
  of that subject (k = 2, 4, 8);
* ``compound``: a question of p parts, p unrelated (subject, relation) pairs (p = 2, 4, 8),
  read by p heads each with a part as its cue (``each``) and by p heads on the unit sum of the
  p keys (``summed``); found when every part's fact is held;
* ``chain``: two and three hops, the value read laid on the next cue's subject coordinates
  (``LongTermMemory.chain``), from an exact and from a noisy first cue; found when every hop
  holds its fact.

**Shared keys (``shared``).** With ``shared=q`` a share ``q / 2`` of the entities take the code
and the relations of another entity each, so a share ``q`` of the facts have a key that another
fact has. Every fact has a source, a code of 32 coordinates drawn for the fact, which stands in
for the sentence the key was coded from. Records are written one by one, as heard: without
``apart`` a write whose key is held revises that record; with ``apart`` it compares signatures.
A share of the facts are then told again with another object. Found is by value: the read holds
the fact's current object.

    PYTHONPATH=src python benchmarks/memory_scale/facts.py --plan plan.json --out rows.jsonl
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

from million import scan, unit  # noqa: E402

PER, RELATIONS, SOURCE = 8, 64, 32
HEADS = (2, 4, 8)


class World:
    """The entities, relations and facts of one founder at ``n`` facts."""

    def __init__(self, founder: int, n: int, width: int, form: str, shared: float = 0.0) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([founder, 310, 5, n, width]))
        self.rng, self.form, self.width = rng, form, width
        self.entities = many = n // PER
        self.n = many * PER
        self.codes = unit(rng.standard_normal((many, width), dtype=np.float32))
        self.relations = unit(rng.standard_normal((RELATIONS, width), dtype=np.float32))
        # each entity's 8 relations, and each fact's object
        self.has = np.argsort(rng.random((many, RELATIONS), dtype=np.float32), axis=1)[:, :PER]
        self.twin = np.full(many, -1, dtype=np.int64)
        twins = int(many * shared / 2)
        if twins:
            pick = rng.permutation(many)[: 2 * twins]
            first, second = pick[:twins], pick[twins:]
            self.codes[second] = self.codes[first]
            self.has[second] = self.has[first]
            self.twin[second], self.twin[first] = first, second
        self.subject = np.repeat(np.arange(many), PER)
        self.relation = self.has.ravel()
        self.object = (self.subject + 1 + rng.integers(many - 1, size=self.n)) % many
        self.key_width = 2 * width if form == "parts" else width

    def key(self, subject: np.ndarray, relation: np.ndarray) -> np.ndarray:
        """The keys of the facts about ``subject`` under ``relation``, in float32."""
        return self.cue(self.codes[subject], self.relations[relation])

    def cue(self, subject: np.ndarray | None, relation: np.ndarray | None) -> np.ndarray:
        """A cue from a subject's code and a relation's code; None leaves the part silent."""
        given = subject if subject is not None else relation
        assert given is not None
        zero = np.zeros_like(given)
        s = zero if subject is None else subject
        r = zero if relation is None else relation
        if self.form == "parts":
            return np.concatenate([s, r], axis=-1)
        return unit(s + r)

    def keys(self) -> np.ndarray:
        return self.key(self.subject, self.relation)

    def values(self) -> np.ndarray:
        return np.asarray(self.codes[self.object])

    def fact(self, subject: int, relation: int) -> int:
        """The fact about ``subject`` under ``relation``."""
        return int(subject * PER + np.flatnonzero(self.has[subject] == relation)[0])


def noisy(rng: np.random.Generator, cue: np.ndarray) -> np.ndarray:
    """``cue`` with noise of its own energy (cosine .71)."""
    energy = float(np.linalg.norm(cue))
    return np.asarray(cue + energy * rng.standard_normal(len(cue)) / np.sqrt(len(cue)))


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    from million import _arm

    founder, n, width = int(spec["founder"]), int(spec["n"]), int(spec.get("width", 64))
    form, shared = spec.get("form", "parts"), float(spec.get("shared", 0.0))
    trials = int(spec.get("trials", 500))
    world = World(founder, n, width, form, shared)
    rng, n = world.rng, world.n
    keys, values = world.keys(), world.values()
    base = {"founder": founder, "n": n, "form": form, "width": world.key_width,
            "value_width": width, "shared": shared}
    rows_out: list[dict[str, Any]] = []

    def emit(arm: str, reading: str, k: int, **more: Any) -> None:
        rows_out.append({**base, "arm": arm, "reading": reading, "k": k, **more})

    if shared:
        return _shared(spec, world, keys, values, base, rows_out)

    # the questions, the same for every arm
    asked = rng.integers(n, size=trials)
    exact = keys[asked].astype(np.float64)
    noised = np.stack([noisy(rng, c) for c in exact])
    subjects = rng.integers(world.entities, size=trials)
    subject_cues = np.asarray(world.cue(world.codes[subjects], None), dtype=np.float64)
    compound = {p: np.stack([rng.choice(n, size=p, replace=False) for _ in range(trials)])
                for p in HEADS}
    chains = []  # each: the facts of three hops
    for start in rng.integers(n, size=trials):
        hops = [int(start)]
        for _ in range(2):
            subject = int(world.object[hops[-1]])
            hops.append(subject * PER + int(rng.integers(PER)))
        chains.append(hops)
    chain_facts = np.array(chains)
    chain_first = {"exact": keys[chain_facts[:, 0]].astype(np.float64)}
    chain_first["noisy"] = np.stack([noisy(rng, c) for c in chain_first["exact"]])

    def link(fact: np.ndarray) -> np.ndarray:
        """What the question holds of a later hop's cue: the relation alone."""
        return np.asarray(world.cue(None, world.relations[world.relation[fact]]),
                          dtype=np.float64)

    def follow(value: np.ndarray, fact: np.ndarray) -> np.ndarray:
        """The next cue as the chain lays it: the link plus the value on the subject's
        coordinates."""
        out = link(fact)
        out[..., :width] += value
        return np.asarray(out)

    for arm_spec in spec["arms"]:
        name = arm_spec["name"]
        if n > int(arm_spec.get("max_n", n)):
            continue
        if name == "scan":
            for label, cues in (("exact", exact), ("noisy", noised)):
                emit(name, f"fact-{label}", 1, trials=trials,
                     found=round(float(np.mean(scan(keys, cues)[:, 0] == asked)), 4))
            for k in HEADS:
                picks = scan(keys, subject_cues, k)
                good = [len(set(p)) == k and set(p) <= set(range(s * PER, s * PER + PER))
                        for p, s in zip(picks.tolist(), subjects.tolist(), strict=True)]
                emit(name, "subject", k, trials=trials, found=round(float(np.mean(good)), 4))
            for p, facts in compound.items():
                each = scan(keys, keys[facts.ravel()].astype(np.float64))[:, 0].reshape(-1, p)
                total = keys[facts].astype(np.float64).sum(axis=1)
                summed = scan(keys, total, p)
                emit(name, "compound-each", p, trials=trials,
                     found=round(float(np.mean((each == facts).all(axis=1))), 4))
                emit(name, "compound-summed", p, trials=trials, found=round(float(np.mean(
                    [set(a) == set(b) for a, b in zip(summed.tolist(), facts.tolist(),
                                                      strict=True)])), 4))
            for label, first in chain_first.items():
                held = scan(keys, first)[:, 0]
                right = held == chain_facts[:, 0]
                for hop in (1, 2):
                    held = scan(keys, follow(values[held].astype(np.float64),
                                             chain_facts[:, hop]))[:, 0]
                    right = right & (held == chain_facts[:, hop])
                    emit(name, f"chain-{label}", hop + 1, trials=trials,
                         found=round(float(np.mean(right)), 4))
            continue
        arm = _arm(arm_spec, world.key_width, width, n, founder)
        t0 = time.perf_counter()
        arm.fill(keys, values)
        fill_s = time.perf_counter() - t0
        more = {"bytes_per_record": round(arm.nbytes / n, 1), "fill_s": round(fill_s, 2),
                "dropped": int(arm.dropped), "settings": arm_spec.get("settings", {})}
        for label, cues in (("exact", exact), ("noisy", noised)):
            t0 = time.perf_counter()
            first = [arm.read(c, credit=False).first() for c in cues]
            ms = 1e3 * (time.perf_counter() - t0) / trials
            emit(name, f"fact-{label}", 1, trials=trials, read_ms=round(ms, 4),
                 found=round(float(np.mean(np.array(first) == asked)), 4), **more)
        for k in HEADS:
            good, t0 = [], time.perf_counter()
            for c, s in zip(subject_cues, subjects.tolist(), strict=True):
                read = arm.read(c, heads=k, credit=False)
                picks = [read.first(h) for h in range(k)]
                good.append(len(set(picks)) == k and all(p // PER == s for p in picks))
            emit(name, "subject", k, trials=trials, found=round(float(np.mean(good)), 4),
                 read_ms=round(1e3 * (time.perf_counter() - t0) / trials, 4))
        for p, facts in compound.items():
            good_each, good_sum, seconds = [], [], [0.0, 0.0]
            for row in facts:
                parts = keys[row].astype(np.float64)
                t0 = time.perf_counter()
                read = arm.read_each(parts, credit=False)
                seconds[0] += time.perf_counter() - t0
                good_each.append([read.first(h) for h in range(p)] == row.tolist())
                t0 = time.perf_counter()
                read = arm.read(parts.sum(axis=0), heads=p, credit=False)
                seconds[1] += time.perf_counter() - t0
                good_sum.append({read.first(h) for h in range(p)} == set(row.tolist()))
            emit(name, "compound-each", p, trials=trials,
                 found=round(float(np.mean(good_each)), 4),
                 read_ms=round(1e3 * seconds[0] / trials, 4))
            emit(name, "compound-summed", p, trials=trials,
                 found=round(float(np.mean(good_sum)), 4),
                 read_ms=round(1e3 * seconds[1] / trials, 4))
        for label in ("exact", "noisy"):
            right = {2: [], 3: []}
            seconds = 0.0
            for hops, first in zip(chain_facts, chain_first[label], strict=True):
                t0 = time.perf_counter()
                reads = arm.chain(first, link(hops[1:]), at=0, credit=False)
                seconds += time.perf_counter() - t0
                held = [r.first() for r in reads]
                right[2].append(held[:2] == hops[:2].tolist())
                right[3].append(held == hops.tolist())
            for hop in (2, 3):
                emit(name, f"chain-{label}", hop, trials=trials,
                     found=round(float(np.mean(right[hop])), 4),
                     read_ms=round(1e3 * seconds / trials, 4))
        del arm
    return rows_out


def _shared(spec: dict[str, Any], world: World, keys: np.ndarray, values: np.ndarray,
            base: dict[str, Any], rows_out: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Facts whose keys are one key, written one by one in a shuffled order, a share told
    again with another object; found by value, over the facts that share a key and over the
    others."""
    from million import _arm

    rng, n, width = world.rng, world.n, world.width
    trials, again = int(spec.get("trials", 500)), float(spec.get("again", 0.01))
    sources = rng.standard_normal((n, SOURCE), dtype=np.float32)
    order = rng.permutation(n)
    told = rng.permutation(n)[: int(n * again)]  # told again, after every fact, another object
    current = values.copy()
    current[told] = world.codes[(world.object[told] + 1 + rng.integers(
        world.entities - 2, size=len(told))) % world.entities]
    twin = world.twin[world.subject] >= 0
    asked = {"shared": rng.choice(np.flatnonzero(twin), size=trials),
             "own": rng.choice(np.flatnonzero(~twin), size=trials),
             "again": rng.choice(told, size=min(trials, len(told)), replace=False)}
    for arm_spec in spec["arms"]:
        name = arm_spec["name"]
        arm = _arm({**arm_spec, "settings": {"revise": 0.999, **arm_spec.get("settings", {})}},
                   world.key_width, width, n, int(base["founder"]))
        apart = bool(arm.apart)
        t0 = time.perf_counter()
        for i in order.tolist():
            arm.write(keys[i], values[i], source=sources[i] if apart else None)
        write_ms = 1e3 * (time.perf_counter() - t0) / n
        held = arm.count
        for i in told.tolist():
            arm.write(keys[i], current[i], source=sources[i] if apart else None)
        more = {"records": held, "records_after": arm.count, "write_ms": round(write_ms, 4),
                "bytes_per_record": round(arm.nbytes / n, 1), "dropped": int(arm.dropped),
                "share": round(float(twin.mean()), 4), "settings": arm_spec.get("settings", {})}
        for label, facts in asked.items():
            found, stale, t0 = [], [], time.perf_counter()
            for i in facts.tolist():
                read = arm.read(keys[i].astype(np.float64), credit=False,
                                source=sources[i] if apart else None)
                value = arm.value(read).astype(np.float32)
                found.append(bool(np.array_equal(value, current[i])))
                stale.append(bool(label == "again" and np.array_equal(value, values[i])))
            ms = 1e3 * (time.perf_counter() - t0) / max(len(facts), 1)
            rows_out.append({**base, "arm": name, "reading": f"shared-{label}", "k": 1,
                             "trials": len(facts), "found": round(float(np.mean(found)), 4),
                             "stale": round(float(np.mean(stale)), 4),
                             "read_ms": round(ms, 4), **more})
        del arm
    return rows_out


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for founder in plan["founders"]:
        for n in plan["n"]:
            for form in plan.get("forms", ["parts"]):
                out.append({"founder": founder, "n": n, "form": form, "arms": plan["arms"],
                            **{name: plan[name] for name in ("trials", "width", "shared", "again")
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
