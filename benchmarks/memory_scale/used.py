"""Found and used (cadence#288): does a brain answer with the content of the record its read
found, as the records it holds grow?

**The task.** A fact is one moment: an address code followed by one of ``TOKENS`` content tokens.
A question is the address alone, and the answer is the token. In the nursery every round is a
fresh episode of 16 streams: a few facts with addresses and tokens drawn anew, then questions
about them, each taught by a lesson before the act. No fact of a reading occurs in the nursery.

**The reading.** After the nursery the brain is read at each N: its memory is emptied, N fresh
facts are written as N moments (``Recall.witness``), and questions about records drawn at random
are acted on. A question is read through the brain's own act and writes no slot. Two scores are
kept apart:

* **found**: the read's first choice is the record (``first``), and the record holds more than
  half of the read's mass (``found``);
* **used**: the brain's act is the record's token (``used``), and the same among the reads whose
  first choice is right (``used_first``).

``margins`` splits the read's logit margin, the right slot less the best other slot, by source:
the key, the address map's event rows, cortex columns and bias, and the same three of the
position maps. The split follows the vocabulary chamber's (cadence#266).

**Arms.** An arm names ``options`` of ``Brain.compose`` and, with ``bank``, a candidate bank
beside the brain (``bank.Bank`` with those settings). The bank is told every fact (its address as
the key, its token as the value) and is read with the address of every moment before the brain
acts; what it reads is held on ``TOKENS`` further input coordinates of that moment, as a recall
port is held through a settle. With a bank ``found`` is the bank's read, and a brain composed
with ``recall=False`` has that read as its only memory of the facts.

    PYTHONPATH=src python benchmarks/memory_scale/used.py --plan plan.json --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TOKENS = 8
STREAMS = 16
HALF = 0.5
TERMS = ("key", "event", "cortex", "bias", "position_event", "position_cortex", "position_bias")


def fact(address: np.ndarray, token: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
    """Facts as moments, ``(rows, width + TOKENS)``: each address followed by its token, and
    with a bank by what the bank's read holds (``held``, ``(rows, TOKENS)``)."""
    out = np.zeros((len(address), address.shape[1] + TOKENS))
    out[:, : address.shape[1]] = address
    out[np.arange(len(address)), address.shape[1] + np.asarray(token)] = 1.0
    return out if held is None else np.concatenate([out, held], axis=1)


def question(address: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros((len(address), address.shape[1] + TOKENS))
    out[:, : address.shape[1]] = address
    return out if held is None else np.concatenate([out, held], axis=1)


class Aid:
    """A candidate bank beside the brain: told facts by address and token, read by address."""

    def __init__(self, width: int, capacity: int, settings: dict[str, Any], seed: int) -> None:
        from bank import Bank

        self.bank = Bank(width, TOKENS, capacity, seed=seed, **settings)
        self.owner = np.full(capacity, -1, dtype=np.int64)

    def tell(
        self, address: np.ndarray, token: np.ndarray, record: np.ndarray | None = None
    ) -> None:
        for i in range(len(address)):
            slot, _ = self.bank.write(address[i], np.eye(TOKENS)[int(token[i])])
            if record is not None:
                self.owner[slot] = record[i]
            self.bank.moment()

    def read(self, address: np.ndarray) -> tuple[np.ndarray, list[tuple[int, float]]]:
        """What the bank's read holds for each address, ``(rows, TOKENS)``, and each read's
        first choice (the record its slot holds) with its mass."""
        held, picks = np.zeros((len(address), TOKENS)), []
        for i in range(len(address)):
            read = self.bank.read(address[i])
            held[i] = self.bank.value(read)
            first = read.first()
            picks.append((int(self.owner[first]) if first >= 0 else -1,
                          read.mass(first) if first >= 0 else 0.0))
        return held, picks


def logit_terms(brain: Any, x: np.ndarray) -> dict[str, np.ndarray]:
    """Head 0's logits over the slots at the moment ``x`` by what they come from, read before
    the act; they sum to the logits of the read where the memory charges no recency."""
    recall = brain.recall
    moment = recall._moment(brain.observation_code(x))
    inputs, c = moment.inputs, moment.cortex
    w_x, w_c, b = recall.head_map(0)
    keys, temperature = recall.keys, recall.temperature
    out = {
        "key": np.einsum("bk,bmk->bm", moment.key, keys) / temperature,
        "event": np.einsum("bk,bmk->bm", inputs @ w_x, keys) / temperature,
        "cortex": np.einsum("bk,bmk->bm", c @ w_c, keys) / temperature,
        "bias": np.einsum("k,bmk->bm", b, keys) / temperature,
    }
    zero_x, zero_c = np.zeros_like(inputs), np.zeros_like(c)
    whole = recall.position_term(0, inputs, c)
    if whole is None:
        whole = np.zeros_like(out["key"])
        event_only = bias_only = whole
    else:
        event_only = recall.position_term(0, inputs, zero_c)
        bias_only = recall.position_term(0, zero_x, zero_c)
    out["position_cortex"] = whole - event_only
    out["position_event"] = event_only - bias_only
    out["position_bias"] = bias_only
    return out


def nursery(brain: Any, rng: np.random.Generator, width: int, rounds: int,
            bank: dict[str, Any] | None = None) -> dict[str, float]:
    """``rounds`` episodes of facts and taught questions; the share answered right before each
    lesson over the last tenth of the rounds. With ``bank`` every episode has a bank of its own
    beside the brain, every slot scored, under those settings."""
    import world

    right: list[float] = []
    for r in range(rounds):
        brain.reset()
        facts = int(rng.integers(4, 17))
        address = world.codes(rng, STREAMS * facts, width).reshape(STREAMS, facts, width)
        token = rng.integers(TOKENS, size=(STREAMS, facts))
        aid = None if bank is None else Aid(width, STREAMS * facts, {**bank, "candidates": 0}, r)
        for f in range(facts):
            held = None if aid is None else aid.read(address[:, f])[0]
            brain.act(fact(address[:, f], token[:, f], held), greedy=True)
            if aid is not None:
                aid.tell(address[:, f], token[:, f])
        for _ in range(4):
            asked = rng.integers(facts, size=STREAMS)
            rows = np.arange(STREAMS)
            held = None if aid is None else aid.read(address[rows, asked])[0]
            x, label = question(address[rows, asked], held), token[rows, asked]
            brain.lesson(x, label)
            said = np.asarray(brain.act(x, greedy=True)).reshape(STREAMS, -1)[:, 0]
            if r >= rounds - max(rounds // 10, 1):
                right.append(float(np.mean(said == label)))
    return {"nursery": round(float(np.mean(right)), 4) if right else float("nan")}


def read_at(brain: Any, rng: np.random.Generator, width: int, n: int, questions: int,
            terms: bool = True, bank: dict[str, Any] | None = None,
            seed: int = 0) -> dict[str, Any]:
    """The frozen brain over ``n`` fresh facts: found and used on ``questions`` questions. With
    ``bank`` the facts are told to a bank of ``n`` slots under those settings as well, ``found``
    is the bank's read, and a memory the brain holds keeps the facts its slots have room for."""
    from types import SimpleNamespace

    import world

    recall = brain.recall
    recs = world.records(rng, n, width, width)
    token = rng.integers(TOKENS, size=n)
    aid = None if bank is None else Aid(width, n, bank, seed)
    silent = None if aid is None else np.zeros((1, TOKENS))
    brain.reset()
    brain.act(fact(recs.address[:1], token[:1], silent), greedy=True)
    if aid is not None:
        aid.tell(recs.address, token, np.arange(n))
    offset = 0
    if recall is not None:
        state = SimpleNamespace(activation=np.zeros((1, brain.connectome.n)))
        for i in range(1, n):
            recall.witness(
                brain.observation_code(fact(recs.address[i : i + 1], token[i : i + 1], silent)),
                state,
            )
        offset = recall.slots - n
        recall._private = True  # a question is read and writes no slot
    asked = rng.integers(n, size=questions)
    first, mass, used, margins = [], [], [], {name: [] for name in TERMS}
    terms = terms and aid is None and recall is not None
    seconds = 0.0
    try:
        for r in asked:
            held, picks = (None, []) if aid is None else aid.read(recs.address[r : r + 1])
            x = question(recs.address[r : r + 1], held)
            at = offset + int(r)
            if terms:
                parts = logit_terms(brain, x)
                total = sum(parts[name][0] for name in TERMS)
                other = total.copy()
                other[at] = -np.inf
                rival = int(np.argmax(other))
                for name in TERMS:
                    margins[name].append(float(parts[name][0, at] - parts[name][0, rival]))
            t0 = time.perf_counter()
            said = int(np.asarray(brain.act(x, greedy=True)).reshape(1, -1)[0, 0])
            seconds += time.perf_counter() - t0
            if aid is not None:
                first.append(int(picks[0][0] == int(r)))
                mass.append(picks[0][1] if picks[0][0] == int(r) else 0.0)
            else:
                attended = recall.attended_heads
                assert attended is not None and attended[0] is not None
                p = np.asarray(attended[0][0])
                first.append(int(int(np.argmax(p)) == at))
                mass.append(float(p[at]))
            used.append(int(said == int(token[r])))
    finally:
        if recall is not None:
            recall._private = False
    first_a, used_a = np.asarray(first, dtype=bool), np.asarray(used, dtype=bool)
    return {
        "n": n, "questions": questions,
        "first": round(float(first_a.mean()), 4),
        "found": round(float(np.mean(np.asarray(mass) > HALF)), 4),
        "mass": round(float(np.mean(mass)), 4),
        "used": round(float(used_a.mean()), 4),
        "used_first": round(float(used_a[first_a].mean()), 4) if first_a.any() else float("nan"),
        "act_ms": round(1e3 * seconds / questions, 4),
        "margins": {name: round(float(np.mean(v)), 3) for name, v in margins.items()} if terms
        else {},
    }


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    warnings.simplefilter("ignore")
    import cadence as cd

    founder, width = int(spec["founder"]), int(spec.get("width", 64))
    sizes = [int(n) for n in spec["n"]]
    arm = spec["arm"]
    name = arm if isinstance(arm, str) else arm["name"]
    options = {} if isinstance(arm, str) else dict(arm.get("options", {}))
    bank = None if isinstance(arm, str) else arm.get("bank")
    if options.get("recall") is not False and "recall_capacity" not in options:
        # without a bank the brain's own memory holds every fact of the largest reading
        options["recall_capacity"] = max(sizes) + 1 if bank is None else 128
    brain = cd.Brain.compose(
        inputs=width + TOKENS * (1 if bank is None else 2), actions=TOKENS, modules=(64,),
        seed=founder, backend="cpu", episodic=False, **options,
    )
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, 6]))
    t0 = time.perf_counter()
    report = nursery(brain, rng, width, int(spec.get("rounds", 1500)), bank)
    report["nursery_seconds"] = round(time.perf_counter() - t0, 1)
    rows = []
    for n in sizes:
        fresh = np.random.default_rng(np.random.SeedSequence([founder, 288, 7, n]))
        rows.append({"founder": founder, "width": width, "arm": name, "options": options,
                     "bank": bank, "rounds": int(spec.get("rounds", 1500)), **report,
                     **read_at(brain, fresh, width, n, int(spec.get("questions", 400)),
                               bank=bank, seed=founder)})
    return rows


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"founder": founder, "arm": arm, "n": plan["n"],
         **{k: plan[k] for k in ("width", "rounds", "questions") if k in plan}}
        for founder in plan["founders"] for arm in plan["arms"]
    ]


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
