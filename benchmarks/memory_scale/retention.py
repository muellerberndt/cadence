"""Retention under finite storage (cadence#288, instrument b): what a memory of B slots keeps of
a stream of facts whose use follows a Zipf law.

**The stream** (``world.Stream``). ``facts`` facts, each an address code with a content code, a
quarter of them in families of similar addresses. At every moment one fact occurs, drawn by a
Zipf law over the facts' ranks. An occurrence asks the fact (a read with its address) and then
states it (a write of its address and its content): the world answers every question. Before an
occurrence the fact's content changes with probability ``change``; the question that follows
meets the old content in any memory that holds the fact, and the statement tells the new one.
With ``drift`` the ranks are reshuffled at the midpoint.

**Arms.** A candidate bank (``bank.Bank``) by name and settings, and the references
(``references.Cache``): least recently used, first in first out, random and least frequently used
over exact keys.

**Keys.** The facts are ``world.records`` unless the plan names ``"keys": "message"``: then they
are ``meanings.facts``, a key that says what the fact is about and a value that says it, both
made by ``cadence.Message``; a fact that changes keeps its key and takes another patient.

**Readings**, at every checkpoint, by probes that count as no use:

* ``top``: the share of the 10% most-used facts that are retrievable: a read with the fact's
  address holds more than half of its mass on a record of that fact at its current content;
* ``revised``: the same share over the facts of that set whose content changed and was stated
  since;
* ``stale`` and ``mixed``: the shares of those revised facts whose read is held by a record of
  an old content, and by no single record;
* ``rare``: the share of the half of the facts used least that is held;
* ``left``, with drift: the share held of the facts that were among the most used before the
  midpoint and are not after it.

``writes`` counts statements that took a slot which did not hold the fact, ``hits`` the
questions answered with the current content.

    PYTHONPATH=src python benchmarks/memory_scale/retention.py --plan plan.json --out rows.jsonl
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

HALF = 0.5
CHECKPOINTS = 10  # readings per stream, evenly spaced, the last at its end


class BankArm:
    """A candidate bank over the stream's facts: it is asked with a fact's address and told its
    address and content, and the instrument keeps which fact and version every slot holds."""

    def __init__(self, capacity: int, address: np.ndarray, settings: dict[str, Any],
                 seed: int, value_width: int | None = None) -> None:
        from bank import Bank

        self.address = address
        self.bank = Bank(address.shape[1], value_width or address.shape[1], capacity, seed=seed,
                         **settings)
        self.fact = np.full(capacity, -1, dtype=np.int64)
        self.version = np.full(capacity, -1, dtype=np.int64)
        self.writes = 0

    def _state(self, fact: int, version: int, credit: bool) -> str:
        read = self.bank.read(self.address[fact], credit=credit)
        if not len(read.slots):
            return "absent"
        w = read.weights[0]
        best = int(np.argmax(w))
        slot = int(read.slots[best])
        if w[best] <= HALF:
            mine = self.fact[read.slots] == fact
            return "mixed" if w[mine].sum() > HALF else "absent"
        if self.fact[slot] != fact:
            return "absent"
        return "current" if self.version[slot] == version else "stale"

    def ask(self, fact: int, version: int) -> bool:
        return self._state(fact, version, True) == "current"

    def probe(self, fact: int, version: int) -> str:
        return self._state(fact, version, False)

    def tell(self, fact: int, version: int, content: np.ndarray) -> bool:
        slot, _ = self.bank.write(self.address[fact], content)
        fresh = self.fact[slot] != fact
        self.fact[slot], self.version[slot] = fact, version
        self.writes += int(fresh)
        return bool(fresh)

    def moment(self) -> None:
        self.bank.moment()

    def held(self, facts: np.ndarray) -> float:
        return float(np.isin(facts, self.fact[self.fact >= 0]).mean()) if len(facts) else 0.0


class CacheArm:
    def __init__(self, capacity: int, policy: str, facts: int, seed: int) -> None:
        from references import Cache

        self.cache = Cache(capacity, policy, facts, seed)

    def ask(self, fact: int, version: int) -> bool:
        return self.cache.ask(fact, version)

    def probe(self, fact: int, version: int) -> str:
        return "current" if self.cache.holds(fact, version) else (
            "stale" if self.cache.holds(fact) else "absent")

    def tell(self, fact: int, version: int, content: np.ndarray) -> bool:
        return self.cache.tell(fact, version)

    def moment(self) -> None:
        return None

    @property
    def writes(self) -> int:
        return self.cache.writes

    def held(self, facts: np.ndarray) -> float:
        return float((self.cache.version[facts] >= 0).mean()) if len(facts) else 0.0


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import world
    from references import POLICIES

    founder, budget = int(spec["founder"]), int(spec["budget"])
    width = int(spec.get("width", 64))
    facts = int(spec.get("facts_per_slot", 4)) * budget
    moments = int(spec.get("moments_per_slot", 100)) * budget
    drift = bool(spec.get("drift", False))
    arm = spec["arm"]
    name = arm if isinstance(arm, str) else arm["name"]
    stream = world.Stream(facts, moments, founder, zipf=float(spec.get("zipf", 1.0)),
                          change=float(spec.get("change", 0.05)), drift=drift)
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, 3]))
    fresh = np.random.default_rng(np.random.SeedSequence([founder, 288, 4]))
    if spec.get("keys") == "message":
        # facts keyed by held meanings: the key says what a fact is about, the value what it
        # says, and a fact that changes keeps its key and takes another patient
        import meanings as mn

        coder = mn.Coder(mn.Lexicon(), int(spec.get("units", 280)), founder, spec.get("fan_in"))
        found = mn.facts(rng, coder, facts)
        address, content, width = found.key, found.value.copy(), coder.units

        def changed(fact: int) -> np.ndarray:
            at = np.array([fact])
            patient = fresh.integers(coder.lexicon.forms, size=1)
            return np.asarray(coder.state(found.agent[at], found.verb[at], patient)[0])
    else:
        recs = world.records(rng, facts, width, width, spec.get("keys", "code"))
        address, content = recs.address, recs.content.copy()

        def changed(fact: int) -> np.ndarray:
            return np.asarray(world.codes(fresh, 1, width)[0])
    if name in POLICIES:
        memory: Any = CacheArm(budget, name, facts, founder)
    else:
        memory = BankArm(budget, address, dict(arm.get("settings", {})), founder,
                         value_width=content.shape[1])
    version = np.zeros(facts, dtype=np.int64)
    told = np.zeros(facts, dtype=bool)  # the fact changed and its new content was stated
    marks = {int(round(moments * (i + 1) / CHECKPOINTS)) for i in range(CHECKPOINTS)}
    top_early, rare_early = stream.top(late=False), stream.bottom(late=False)
    top_late, rare_late = stream.top(late=True), stream.bottom(late=True)
    left = np.setdiff1d(top_early, top_late)
    base = {"founder": founder, "budget": budget, "width": width, "facts": facts,
            "arm": name, "drift": drift, "zipf": stream.zipf, "change": stream.change,
            "keys": spec.get("keys", "code")}
    if spec.get("keys") == "message":
        base["distinct_keys"] = round(len(np.unique(address.astype(np.uint8), axis=0)) / facts, 5)
    if not isinstance(arm, str):
        base["settings"] = arm.get("settings", {})
    rows: list[dict[str, Any]] = []
    hits = 0
    started = time.perf_counter()
    for t in range(moments):
        fact = int(stream.fact[t])
        if stream.changed[t]:
            version[fact] += 1
            content[fact] = changed(fact)
            told[fact] = True
        hits += int(memory.ask(fact, int(version[fact])))
        memory.tell(fact, int(version[fact]), content[fact])
        memory.moment()
        if t + 1 in marks:
            late = t + 1 > moments // 2
            top = top_late if late else top_early
            rare = rare_late if late else rare_early
            states = [memory.probe(int(f), int(version[f])) for f in top]
            revised = [s for s, f in zip(states, top, strict=True) if told[f]]
            rows.append({
                **base, "moment": t + 1, "writes": int(memory.writes),
                "hits": round(hits / (t + 1), 4),
                "top": round(float(np.mean([s == "current" for s in states])), 4),
                "revised": round(float(np.mean([s == "current" for s in revised])), 4)
                if revised else float("nan"),
                "revised_facts": len(revised),
                "stale": round(float(np.mean([s == "stale" for s in revised])), 4)
                if revised else float("nan"),
                "mixed": round(float(np.mean([s == "mixed" for s in revised])), 4)
                if revised else float("nan"),
                "rare": round(memory.held(rare), 4),
                "left": round(memory.held(left), 4) if drift and late else float("nan"),
                "seconds": round(time.perf_counter() - started, 2),
            })
    return rows


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for founder in plan["founders"]:
        for budget in plan["budgets"]:
            for drift in plan.get("drift", [False]):
                for arm in plan["arms"]:
                    if not isinstance(arm, str) and budget > int(arm.get("max_budget", budget)):
                        continue
                    out.append({
                        "founder": founder, "budget": budget, "drift": drift, "arm": arm,
                        **{k: plan[k] for k in ("width", "facts_per_slot", "moments_per_slot",
                                                "zipf", "change", "keys", "units", "fan_in")
                           if k in plan},
                    })
    return sorted(out, key=lambda c: -c["budget"])


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
