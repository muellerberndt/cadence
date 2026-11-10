"""A brain that writes and reads its own long-term records (cadence#288, the third round).

The fact chamber of the second round with a brain in place of the instrument: the brain hears
every fact, its held message keys the record at a turn, and it writes, reads and answers by
itself (``Brain.compose(longterm=records)``).

**The world.** ``meanings.Lexicon(64, 96)``: 64 noun stems in two numbers and 96 verbs, 12,288
agent-verb pairs, one more input channel for the turn. A fact is an episode of four moments: the
act (the verb's concept), the agent (the stem's concept with ``AGENT``, and ``PLURAL``), the turn,
the patient (one of ``PATIENTS`` stems with ``PATIENT``). A question is the act, the agent, the
turn and a silent moment, at which the brain answers with the patient: ``PATIENTS`` actions.
Every episode begins with ``Brain.reset()``, which empties the recall memory's slots, so the
records are the only carrier from a fact to its question.

**The brain.** ``Brain.compose(388, 8, modules=(64,), backend="cpu", message=280,
message_fan_in=177, message_boundary=387, message_boundary_holds=True, longterm=..., seed=...)``:
the default composition with the held message at #286's declared code (280 units a block, each
unit reading the chamber's share of the channels), held from the turn.

**The nursery.** ``ROUNDS`` rounds over ``STREAMS`` streams: a fact about a pair of the nursery's
pool is told, then a fact told before is asked and its answer taught by a lesson. The nursery's
pool is ``NURSERY`` pairs; the reading's facts are about the other pairs, which no lesson asked.

**The reading**, at N records: the nursery's brain, its records emptied (``longterm.clear()``),
hears N facts per stream about fresh pairs, then a tenth of them again with another patient
(revised), then ``QUESTIONS`` questions per stream about unrevised facts and as many about
revised ones, without lessons. Scores per question:

* **found**: the record the read held at the answer (``longterm.read_slots``) holds the
  question's key and the fact's current patient;
* **stale**: it holds the patient the fact had before;
* **used**: the brain's answer is the current patient.

The control ``plain`` is the same brain without ``longterm`` through the same nursery and reading.

    PYTHONPATH=src python benchmarks/memory_scale/brain_records.py --plan plan.json --out rows.jsonl
"""

from __future__ import annotations

import argparse
import copy
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

NOUNS, VERBS = 64, 96
PATIENTS = 8
STREAMS = 16
NURSERY = 2048  # agent-verb pairs the nursery's facts are about
ROUNDS = 1500
QUESTIONS = 25  # per stream and kind
REVISED = 0.1  # the share of the read facts told again with another patient
UNITS = 280


class World:
    """The lexicon's moments with the turn channel, and the pairs split by founder."""

    def __init__(self, founder: int) -> None:
        import meanings as mn

        self.lexicon = mn.Lexicon(NOUNS, VERBS)
        self.inputs = self.lexicon.inputs + 1
        self.turn = self.lexicon.inputs
        rng = np.random.default_rng(np.random.SeedSequence([founder, 288, 3, 1]))
        order = rng.permutation(self.lexicon.abouts)
        self.nursery, self.pool = order[:NURSERY], order[NURSERY:]

    def pad(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([x, np.zeros((len(x), 1))], axis=1)

    def episode(self, about: np.ndarray, patient: np.ndarray | None) -> list[np.ndarray]:
        """The moments of facts (``patient`` given) or questions (``patient`` None)."""
        lex = self.lexicon
        agent, verb = about // lex.verbs, about % lex.verbs
        turn = np.zeros((len(about), self.inputs))
        turn[:, self.turn] = 1.0
        last = (np.zeros((len(about), self.inputs)) if patient is None
                else self.pad(lex.noun(patient, "PATIENT")))
        return [self.pad(lex.act(verb)), self.pad(lex.noun(agent, "AGENT")), turn, last]

    def fan_in(self) -> int:
        return round(16 / 70 * 2 * self.inputs)


def make_brain(world: World, founder: int, records: int, heads: int = 1) -> Any:
    warnings.simplefilter("ignore")
    import cadence as cd

    settings: dict[str, Any] = {} if not records else {"longterm": records,
                                                       "longterm_heads": heads}
    return cd.Brain.compose(
        world.inputs, PATIENTS, modules=(64,), seed=founder, backend="cpu", message=UNITS,
        message_fan_in=world.fan_in(), message_boundary=world.turn,
        message_boundary_holds=True, **settings,
    )


def tell(brain: Any, moments: list[np.ndarray]) -> None:
    brain.reset()
    for x in moments:
        brain.act(x, greedy=True)


def ask(brain: Any, moments: list[np.ndarray], label: np.ndarray | None = None) -> np.ndarray:
    brain.reset()
    for x in moments[:-1]:
        brain.act(x, greedy=True)
    if label is not None:
        brain.lesson(moments[-1], label)
    return np.asarray(brain.act(moments[-1], greedy=True)).reshape(len(label if label is not None
                                                                       else moments[0]), -1)[:, 0]


def nursery(brain: Any, world: World, founder: int, rounds: int) -> dict[str, float]:
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, 3, 2]))
    told: list[dict[int, int]] = [{} for _ in range(STREAMS)]
    right: list[float] = []
    for r in range(rounds):
        about = world.nursery[rng.integers(len(world.nursery), size=STREAMS)]
        patient = rng.integers(PATIENTS, size=STREAMS)
        tell(brain, world.episode(about, patient))
        for s in range(STREAMS):
            told[s][int(about[s])] = int(patient[s])
        asked = np.array([list(t)[int(rng.integers(len(t)))] for t in told])
        label = np.array([told[s][int(asked[s])] for s in range(STREAMS)])
        said = ask(brain, world.episode(asked, None), label)
        if r >= rounds - max(rounds // 10, 1):
            right.append(float(np.mean(said == label)))
    return {"nursery": round(float(np.mean(right)), 4)}


def read_at(brain: Any, world: World, founder: int, n: int,
            questions: int = QUESTIONS) -> dict[str, Any]:
    """The reading at ``n`` records per stream; the brain is changed by it."""
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, 3, 3, n]))
    kept = getattr(brain, "longterm", None)
    if kept is not None:
        kept.clear()
    abouts = np.stack([rng.choice(world.pool, size=n, replace=False) for _ in range(STREAMS)])
    first = rng.integers(PATIENTS, size=(STREAMS, n))
    current = first.copy()
    t0 = time.perf_counter()
    for i in range(n):
        tell(brain, world.episode(abouts[:, i], current[:, i]))
    tell_seconds = time.perf_counter() - t0
    revised = rng.choice(n, size=max(int(n * REVISED), 1), replace=False)
    for i in revised:
        shift = rng.integers(1, PATIENTS, size=STREAMS)
        current[:, i] = (first[:, i] + shift) % PATIENTS
        tell(brain, world.episode(abouts[:, i], current[:, i]))
    plain = np.setdiff1d(np.arange(n), revised)
    out: dict[str, Any] = {"n": n, "tell_ms": round(1e3 * tell_seconds / (4 * n), 4)}
    if kept is not None:
        out.update(records=[s.count for s in kept.stores[:STREAMS]], counts=dict(kept.counts))
    for kind, pool in (("unrevised", plain), ("revised", revised)):
        found, stale, used = [], [], []
        for _ in range(questions):
            at = pool[rng.integers(len(pool), size=STREAMS)]
            rows = np.arange(STREAMS)
            about = abouts[rows, at]
            said = ask(brain, world.episode(about, None))
            used += list(said == current[rows, at])
            for s in range(STREAMS):
                hit = was = False
                if kept is not None and kept.read_slots[s, 0] >= 0:
                    store, slot = kept.stores[s], int(kept.read_slots[s, 0])
                    key = brain.message.key()[s]
                    value = np.asarray(store.values[slot])
                    same = np.array_equal(np.asarray(store.keys[slot]), key)
                    now = world.pad(world.lexicon.noun(current[s:s + 1, at[s]], "PATIENT"))[0]
                    old = world.pad(world.lexicon.noun(first[s:s + 1, at[s]], "PATIENT"))[0]
                    hit = same and np.array_equal(value, now)
                    was = same and kind == "revised" and np.array_equal(value, old)
                found.append(hit)
                stale.append(was)
        out[kind] = {"found": round(float(np.mean(found)), 4),
                     "used": round(float(np.mean(used)), 4),
                     "stale": round(float(np.mean(stale)), 4), "questions": len(used)}
    return out


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    founder, arm = int(spec["founder"]), spec["arm"]
    world = World(founder)
    records = int(spec.get("capacity", 10_000)) if arm != "plain" else 0
    brain = make_brain(world, founder, records, int(spec.get("heads", 1)))
    t0 = time.perf_counter()
    report = nursery(brain, world, founder, int(spec.get("rounds", ROUNDS)))
    report["nursery_seconds"] = round(time.perf_counter() - t0, 1)
    base = {"founder": founder, "arm": arm, "parameters": int(brain.parameters()),
            "neurons": int(brain.connectome.n), **report}
    rows = []
    for n in spec["n"]:
        reader = copy.deepcopy(brain)
        t0 = time.perf_counter()
        row = read_at(reader, world, founder, int(n), int(spec.get("questions", QUESTIONS)))
        rows.append({**base, **row, "seconds": round(time.perf_counter() - t0, 1)})
        del reader
    return rows


def cost_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """ms per moment of question and fact episodes over one or many streams, with the records
    filled to N per stream by the store's own writes of facts' keys and values (the keys the
    brain's message gives them), and without the option. ``"fill": "record"`` fills them by the
    records' own write (``LongTermRecords.record``), which keeps a bank of the streams' records
    in step with the stores."""
    founder, streams = int(spec["founder"]), int(spec.get("streams", 1))
    world = World(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, 3, 4]))
    rows = []
    backend = spec.get("backend", "cpu")
    for n in spec["n"]:
        brain = make_brain(world, founder, int(n) if n else 0)
        if backend != "cpu":
            from cadence import Brain

            brain = Brain.compose(
                world.inputs, PATIENTS, modules=(64,), seed=founder, backend=backend,
                device=spec.get("device"), message=UNITS, message_fan_in=world.fan_in(),
                message_boundary=world.turn, message_boundary_holds=True,
                **({"longterm": int(n)} if n else {}))
        kept = brain.longterm
        abouts = rng.choice(world.pool, size=max(int(n), 1000), replace=False)
        if kept is not None:
            import cadence as cd

            fields = ("inputs", "phases", "units", "winners", "fan_in", "seed")
            coder = cd.Message(**{k: brain.message.to_dict()[k] for k in fields})
            kept._streams(streams)
            filled = abouts[: int(n)]
            moments = world.episode(filled, rng.integers(PATIENTS, size=len(filled)))
            coder.reset(len(filled))
            coder.hear(moments[0])
            coder.hear(moments[1])
            keys = coder.key()
            for s in range(streams):
                for i in range(int(n)):
                    if spec.get("fill") == "record":
                        kept.record(s, keys[i], moments[3][i])  # the brain's own write
                    else:
                        kept.stores[s].write(keys[i], moments[3][i])
        told = abouts[: max(int(n), 1)] if n else abouts
        asked = told[rng.integers(len(told), size=(int(spec.get("episodes", 100)), streams))]
        patient = rng.integers(PATIENTS, size=asked.shape)
        for warm in range(5):
            ask(brain, world.episode(asked[warm % len(asked)], None))
        t0 = time.perf_counter()
        for e in range(len(asked)):
            ask(brain, world.episode(asked[e], None))
        question = time.perf_counter() - t0
        t0 = time.perf_counter()
        for e in range(len(asked)):
            tell(brain, world.episode(asked[e], patient[e]))
        fact = time.perf_counter() - t0
        rows.append({"founder": founder, "n": int(n), "streams": streams, "backend": backend,
                     "device": spec.get("device"),
                     "question_ms": round(1e3 * question / (4 * len(asked)), 4),
                     "fact_ms": round(1e3 * fact / (4 * len(asked)), 4),
                     "records": None if kept is None else [s.count for s in kept.stores[:1]],
                     "parameters": int(brain.parameters())})
    return rows


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**{k: v for k, v in plan.items() if k not in ("founders", "arms")},
             "founder": f, **arm} for f in plan["founders"] for arm in plan["arms"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cost", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    todo = cells(plan) if not args.cost else [
        {**{k: v for k, v in plan.items() if k not in ("founders", "settings")}, "founder": f, **s}
        for f in plan["founders"] for s in plan["settings"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    job = cost_cell if args.cost else run_cell
    with args.out.open("a") as sink, get_context("spawn").Pool(args.workers,
                                                               maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(job, todo, chunksize=1):
            for row in rows:
                sink.write(json.dumps(row) + "\n")
            sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
