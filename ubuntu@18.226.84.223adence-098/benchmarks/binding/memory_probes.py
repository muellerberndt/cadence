"""Component probes of the index memory: what it attends to, on untrained founders.

The readings of ``INDEX_PROTOCOL.md``'s component probes, for named arms of the index memory
(``index_read``: the address computed before the settle, the recalled drive held through it).
Each probe writes a small history with greedy acts and reads the attention the next moment
uses, before that moment is written:

* ``age``: one target event, then ``k`` distractor events that share nothing with it; the
  weight the target's own event finds on the target, against the newest distractor.
* ``repeat``: a fact (a name and a value), then the name alone asked ``r`` times; the weight
  the ``r``-th question puts on the fact. An arm that writes aroused moments only
  (``write="aroused"``) has the fact marked aroused and the questions calm, so no question is
  written; every other arm is also read that way (``repeat_aroused``).
* ``nomatch``: eight events, then an event that shares nothing with any of them and an
  all-zero event; the null slot's share (one minus the slots' weights) and the largest slot.
* ``successor``: three events; the next region's activity when the first is heard again (its
  successor known) and when the last is (its successor not yet written).
* ``dense``: every input on at widths 256 to 2,000, with lessons on random targets; refused
  settles and lessons, and whether every parameter stays finite.

Attention, not accuracy; nothing here changes the library.

    PYTHONPATH=src CADENCE_FUSED=0 python benchmarks/binding/memory_probes.py --out OUT.json \
        --seeds 0 1 2 3 4 5 --arms '{"index": {}, "improved": {"address_step": "toward"}}'
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def _sibling(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ir = _sibling("index_read")
tb = ir.tb

AGES = (0, 8, 16, 32, 48, 63)
REPEATS = (1, 2, 4, 8)
DENSE = (256, 512, 2000)
CAPACITY = 64


class Reader:
    """One arm of the index memory on a probe's ports: build, act, and the attention of the
    next moment."""

    def __init__(self, genes: dict[str, Any]) -> None:
        self.genes = dict(genes)

    def build(self, seed: int, inputs: int, actions: int = 2, **options: Any) -> Any:
        genes = dict(ir.GENES, **self.genes)
        return ir.IndexBrain.build(seed, read=genes, inputs=inputs, actions=actions, **options)

    def attend(self, brain: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The slot weights and the null share the moment ``x`` reads, before its write."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        brain.stimulus(x)
        index = brain.index
        assert index.last is not None
        p = index.weights(index.last[2])
        if p is None:
            return np.zeros((len(x), 0)), np.ones(len(x))
        return p, 1.0 - p.sum(axis=1)

    def gated(self) -> bool:
        """Whether the arm writes aroused moments only: a probe's history is then marked
        aroused, and its questions calm."""
        return self.genes.get("write") == "aroused"


def _act(brain: Any, x: np.ndarray, aroused: bool | None = None) -> None:
    if aroused is not None:
        brain.aroused_rows = np.array([aroused])
    brain.act(np.atleast_2d(x), greedy=True)
    if aroused is not None:
        brain.aroused_rows = None


def _history(reader: Reader) -> bool | None:
    return True if reader.gated() else None


def _order(brain: Any) -> np.ndarray:
    """Slot index of each written moment of stream 0, oldest first."""
    return np.arange(brain.index.slots.slots)


def age(reader: Reader, seed: int) -> list[dict[str, float]]:
    events = np.eye(CAPACITY)
    rows = []
    for k in AGES:
        brain = reader.build(seed, CAPACITY)
        brain.reset()
        for x in events[: k + 1]:
            _act(brain, x, _history(reader))
        p, null = reader.attend(brain, events[0])
        order = _order(brain)
        rows.append({"age": k, "target": float(p[0, order[0]]),
                     "newest": float(p[0, order[-1]]), "null": float(null[0])})
    return rows


def repeat(reader: Reader, seed: int) -> list[dict[str, float]]:
    inputs = 16
    fact = np.zeros(inputs)
    fact[[0, 8]] = 1.0  # a name and a value
    question = np.zeros(inputs)
    question[0] = 1.0
    gated = reader.gated()
    rows = []
    for r in REPEATS:
        brain = reader.build(seed, inputs)
        brain.reset()
        _act(brain, fact, True if gated else None)
        for _ in range(r - 1):
            _act(brain, question, False if gated else None)
        p, null = reader.attend(brain, question)
        rows.append({"asks": r, "fact": float(p[0, _order(brain)[0]]), "null": float(null[0]),
                     "slots": int(brain.index.slots.slots)})
    return rows


def nomatch(reader: Reader, seed: int) -> dict[str, float]:
    events = np.eye(CAPACITY)
    brain = reader.build(seed, CAPACITY)
    brain.reset()
    for x in events[:8]:
        _act(brain, x, _history(reader))
    out = {}
    for name, x in (("unrelated", events[20]), ("zero", np.zeros(CAPACITY))):
        twin = tb.fork(brain)
        p, null = reader.attend(twin, x)
        out[name + "_null"] = float(null[0])
        out[name + "_largest"] = float(p[0].max())
    return out


def successor(reader: Reader, seed: int) -> dict[str, float]:
    inputs = 16
    events = np.eye(inputs)[[1, 5, 9]]
    brain = reader.build(seed, inputs)
    if not bool(reader.genes.get("next", ir.GENES["next"])):
        return {}
    brain.reset()
    for x in events:
        _act(brain, x, _history(reader))
    out = {}
    for name, x, follower in (("known", events[0], events[1]), ("unknown", events[2], None)):
        twin = tb.fork(brain)
        drive = twin.stimulus(np.atleast_2d(x))
        nxt = np.atleast_2d(drive)[0, twin.index_next[:inputs]]
        out[name + "_norm"] = float(np.linalg.norm(nxt))
        if follower is not None:
            out[name + "_cosine"] = float(nxt @ follower / max(np.linalg.norm(nxt), 1e-12))
    return out


def dense(reader: Reader, seed: int, lessons: int) -> list[dict[str, Any]]:
    rows = []
    for width in DENSE:
        brain = reader.build(seed, width, learning_surprise=True)
        rng = np.random.default_rng(seed)
        work = tb.new_work()
        brain.reset()
        x = np.ones((1, width))
        refused = 0
        for _ in range(lessons):
            if tb.act(brain, x, work) is None:
                refused += 1
            try:
                ir.lesson(brain, x, rng.integers(0, 2, size=1), work)
            except RuntimeError:
                refused += 1
        graph, index = brain.brain, brain.index
        finite = bool(np.isfinite(graph.efficacy).all() and np.isfinite(graph.bias).all()
                      and np.isfinite(index.w_x).all() and np.isfinite(index.b).all())
        rows.append({"width": width, "lessons": lessons, "refused": refused,
                     "lesson_refusals": int(work["lesson_refusals"]), "finite": finite})
    return rows


def run(reader: Reader, seed: int, lessons: int) -> dict[str, Any]:
    started = time.monotonic()
    out = {"age": age(reader, seed), "repeat": repeat(reader, seed),
           "nomatch": nomatch(reader, seed), "successor": successor(reader, seed),
           "dense": dense(reader, seed, lessons)}
    if not reader.gated():
        out["repeat_aroused"] = repeat(Reader(dict(reader.genes, write="aroused")), seed)
    out["seconds"] = time.monotonic() - started
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--arms", type=json.loads, required=True,
                        help="arm name to genes over the runner's index arm")
    parser.add_argument("--lessons", type=int, default=200)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    runs = _sibling("index_runs")
    rows = []
    for seed in args.seeds:
        for name, extra in args.arms.items():
            genes = dict(runs.ARMS["index"], capacity=CAPACITY, **extra)
            rows.append({"seed": seed, "arm": name, "genes": genes,
                         **run(Reader(genes), seed, args.lessons)})
            print(json.dumps({k: rows[-1][k] for k in ("seed", "arm", "nomatch")}), flush=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
