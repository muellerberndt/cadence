"""Chained recall: one link or two, from links heard once in this trial.

A trial shows links ``link + from x + to y`` (a chain x -> y -> z and a distractor link), each
link one event in random order, then neutral events, then a question ``ask1 + from x`` (the
answer is y) or ``ask2 + from x`` (the answer is z), followed by ``think`` events; the answer is
the act at the last event. Tokens are drawn fresh every trial; each founder holds out one
target per source, so held-out links never occur in training while every token does, as a
source and as a target. The question shares a token with the first link only. A read keyed by
the present event can recall y and never the second link, whose source y the question does
not contain: a two-link answer needs the first recall to become the next key.

Learners, from the same founder seeds: the composed brain; plus a content memory read by the
present event (``token_binding``'s fixed read); plus the learned read (``learned_read``); a
supervised online GRU. Lessons at the last event only, judged by the brain's own act or on
every row. Instrument level; nothing here changes the library. The protocol and its gates are
declared in ``CHAIN_PROTOCOL.md`` before the first run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import get_context
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


tb = _sibling("token_binding")
lr = _sibling("learned_read")

TOKENS = 8
MARKERS = ("link", "ask1", "ask2", "think", "neutral")
INPUTS = len(MARKERS) + 2 * TOKENS  # markers, then sources, then targets
NEUTRAL = 1
THINK = 1  # think events after the question in training
BATCH = 16
BUDGETS = (256, 1024, 4096)
FOUNDERS = tuple(range(6))
SCHEDULE_SEED, HOLDOUT_SEED, PANEL_SEED, GRU_SEED = 61017, 61018, 61019, 61020
LEARNERS = ("graph", "content", "learned", "gru")
RULES = ("act", "every")
READ = {"alpha": 3.0, "rule": "hebb", "heard": 0.0}
GATE = {"two_held": 0.6, "one_held": 0.75, "founders": 5}


def coordinate(kind: str, index: int = 0) -> int:
    if kind in MARKERS:
        return MARKERS.index(kind)
    if kind == "from":
        return len(MARKERS) + index
    if kind == "to":
        return len(MARKERS) + TOKENS + index
    raise ValueError(kind)


def link_event(sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
    out = np.zeros((len(sources), INPUTS))
    rows = np.arange(len(sources))
    out[:, coordinate("link")] = 1.0
    out[rows, coordinate("from") + np.asarray(sources)] = 1.0
    out[rows, coordinate("to") + np.asarray(targets)] = 1.0
    return out


def marker_event(n: int, marker: str, sources: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros((n, INPUTS))
    out[:, coordinate(marker)] = 1.0
    if sources is not None:
        out[np.arange(n), coordinate("from") + np.asarray(sources)] = 1.0
    return out


def holdout(seed: int) -> np.ndarray:
    """``(TOKENS, TOKENS)`` True where the link ``a -> b`` is held out: one target per source,
    a permutation without fixed points."""
    rng = np.random.default_rng(np.random.SeedSequence([HOLDOUT_SEED, seed]))
    while True:
        order = rng.permutation(TOKENS)
        if not (order == np.arange(TOKENS)).any():
            break
    held = np.zeros((TOKENS, TOKENS), bool)
    held[np.arange(TOKENS), order] = True
    return held


def draw(
    rng: np.random.Generator, n: int, held: np.ndarray, *, hops: int | None = None,
    use_held: bool = False,
) -> dict[str, np.ndarray]:
    """``n`` trials: the chain x -> y -> z, a distractor link w -> u (five distinct tokens),
    the order of the three link events, and the hops asked (1 or 2, drawn when ``hops`` is
    None). Training trials contain no held-out link; with ``use_held`` the asked path's links
    (x -> y, and y -> z for two hops) include at least one held-out link."""
    tokens = np.zeros((n, 5), np.int64)
    asked = rng.integers(1, 3, size=n) if hops is None else np.full(n, hops)
    order = np.stack([rng.permutation(3) for _ in range(n)])
    for i in range(n):
        while True:
            x, y, z, w, u = rng.permutation(TOKENS)[:5]
            links = held[x, y], held[y, z], held[w, u]
            path = links[: asked[i]]
            if use_held:
                if any(path) and not links[2]:
                    break
            elif not any(links):
                break
        tokens[i] = (x, y, z, w, u)
    return {"tokens": tokens, "hops": asked, "order": order}


def events(trials: dict[str, np.ndarray], neutral: int = NEUTRAL, think: int = THINK
           ) -> list[np.ndarray]:
    t, order, hops = trials["tokens"], trials["order"], trials["hops"]
    n = len(hops)
    links = [link_event(t[:, 0], t[:, 1]), link_event(t[:, 1], t[:, 2]),
             link_event(t[:, 3], t[:, 4])]
    out = []
    for position in range(3):
        event = np.zeros((n, INPUTS))
        for k in range(3):
            chosen = order[:, position] == k
            event[chosen] = links[k][chosen]
        out.append(event)
    out += [marker_event(n, "neutral") for _ in range(neutral)]
    question = np.where((hops == 1)[:, None], marker_event(n, "ask1", t[:, 0]),
                        marker_event(n, "ask2", t[:, 0]))
    out.append(question)
    out += [marker_event(n, "think") for _ in range(think)]
    return out


def answers(trials: dict[str, np.ndarray]) -> np.ndarray:
    t, hops = trials["tokens"], trials["hops"]
    return np.where(hops == 1, t[:, 1], t[:, 2]).astype(np.int64)


# -- learners on this task's ports


def build(learner: str, seed: int, rule: str) -> Any:
    surprise = rule == "act"
    if learner == "graph":
        return lr.ReadBrain.build(seed, inputs=INPUTS, actions=TOKENS, learning_surprise=surprise)
    if learner == "learned":
        return lr.ReadBrain.build(seed, read=dict(READ), inputs=INPUTS, actions=TOKENS,
                                  learning_surprise=surprise)
    if learner == "content":
        return ContentChainBrain.build(seed, learning_surprise=surprise)
    raise ValueError(learner)


class ContentChainBrain(tb.ContentBrain):
    """``token_binding.ContentBrain``'s fixed read on this task's ports."""

    @classmethod
    def build(cls, seed: int, **options: Any) -> ContentChainBrain:  # type: ignore[override]
        from cadence.genome import Genome, Projection, develop
        from cadence.regions import Region, motor_cortex, prefrontal_cortex
        from cadence.stream import FastSynapses

        regions = [
            Region("sensory", INPUTS), Region("association", tb.WIDTH),
            prefrontal_cortex(tb.WIDTH), motor_cortex(TOKENS, lateral=tb.LATERAL, slots=1),
            Region("recall", INPUTS),
        ]
        projections = [
            Projection("sensory", "association", scale=1.0, reciprocal=False),
            Projection("association", "motor"),
            Projection("prefrontal", "association", scale=1.2, reciprocal=False),
            Projection("recall", "association", scale=tb.CONTENT["scale"], reciprocal=False),
        ]
        genome = Genome(tuple(regions), tuple(projections), label="composed-brain")
        brain = cls(develop(genome, seed=seed), seed=seed, slots=1, **options)
        brain.recall_index = np.asarray(brain.connectome.populations["recall"], dtype=np.int64)
        brain.content = FastSynapses(brain.sensory_index, brain.recall_index, decay=1.0,
                                     rate=1.0, amplitude=1.0, rule="hebb")
        brain.content_amplitude = float(tb.CONTENT["amplitude"])
        return brain


def train_round(brain: Any, trials: dict[str, np.ndarray], work: dict[str, Any]) -> bool:
    brain.reset()
    seq = events(trials)
    for x in seq[:-1]:
        if tb.act(brain, x, work) is None:
            return False
    lr.lesson(brain, seq[-1], answers(trials), work)
    return tb.act(brain, seq[-1], work) is not None


def memory_of(brain: Any) -> Any:
    coupling = getattr(brain, "coupling", None)
    if coupling is not None:
        return coupling.memory
    return getattr(brain, "content", None)


def read(brain: Any, trials: dict[str, np.ndarray], condition: str, think: int) -> np.ndarray:
    twin = tb.fork(brain)
    twin.reset()
    seq = events(trials, think=think)
    work = tb.new_work()
    for x in seq[:-1]:
        tb.act(twin, x, work)
    memory = memory_of(twin)
    if condition == "memory_erased" and memory is not None:
        memory.reset(0)
    elif condition == "trace_erased" and twin.working_memory is not None:
        twin.working_memory.reset(0)
    out = tb.act(twin, seq[-1], work)
    return np.full(len(trials["hops"]), -1) if out is None else out


class Supervised(tb.Supervised):
    """``token_binding.Supervised`` on this task's ports."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([GRU_SEED, seed]))
        self.hidden, self.rate, self.clip = tb.GRU["hidden"], tb.GRU["rate"], tb.GRU["clip"]
        self.cell = tb._competitors().GRUCell(INPUTS, self.hidden, rng)
        self.readout = {"W": np.zeros((TOKENS, self.hidden)), "b": np.zeros(TOKENS)}
        self.work = {"updates": 0, "rows": 0, "clipped": 0}


def panels(seed: int, held: np.ndarray) -> dict[str, tuple[dict[str, np.ndarray], int]]:
    rng = np.random.default_rng(np.random.SeedSequence([PANEL_SEED, seed]))
    return {
        "one": (draw(rng, 128, held, hops=1), THINK),
        "two": (draw(rng, 128, held, hops=2), THINK),
        "one_held": (draw(rng, 128, held, hops=1, use_held=True), THINK),
        "two_held": (draw(rng, 128, held, hops=2, use_held=True), THINK),
        "two_more_think": (draw(rng, 128, held, hops=2), THINK + 2),
    }


def read_life(model: Any, seed: int, held: np.ndarray, gru: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (trials, think) in panels(seed, held).items():
        truth = answers(trials)
        if gru:
            out[name] = {"intact": tb.accuracy(model.answer(events(trials, think=think)), truth)}
            continue
        row = {c: tb.accuracy(read(model, trials, c, think), truth)
               for c in ("intact", "trace_erased")}
        if memory_of(model) is not None:
            row["memory_erased"] = tb.accuracy(read(model, trials, "memory_erased", think), truth)
        out[name] = row
    return out


def life(spec: dict[str, Any]) -> dict[str, Any]:
    seed, learner, rule = spec["seed"], spec["learner"], spec["rule"]
    held = holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([SCHEDULE_SEED, seed]))
    started = time.monotonic()
    readings, work = [], tb.new_work()
    gru = learner == "gru"
    model = Supervised(seed) if gru else build(learner, seed, rule)
    done = 0
    for budget in spec.get("budgets", BUDGETS):
        while done < budget:
            trials = draw(rng, BATCH, held)
            if gru:
                model.lesson(events(trials), answers(trials))
            else:
                train_round(model, trials, work)
            done += 1
        readings.append({"rounds": budget, "panels": read_life(model, seed, held, gru),
                         "seconds": time.monotonic() - started})
    return {**spec, "held": held.tolist(), "readings": readings,
            "work": model.work if gru else work}


def gate(reading: dict[str, Any]) -> bool:
    p = reading["panels"]
    return bool(p["two_held"]["intact"] >= GATE["two_held"]
                and p["one_held"]["intact"] >= GATE["one_held"])


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| learner | rule | rounds | founder | one | two | one held | two held | "
             "two held, memory erased | two, more think | gate |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        for reading in row["readings"]:
            p = reading["panels"]
            erased = p["two_held"].get("memory_erased")
            lines.append(
                f"| {row['learner']} | {row['rule']} | {reading['rounds']} | {row['seed']} | "
                f"{p['one']['intact']:.2f} | {p['two']['intact']:.2f} | "
                f"{p['one_held']['intact']:.2f} | {p['two_held']['intact']:.2f} | "
                f"{'' if erased is None else f'{erased:.2f}'} | "
                f"{p['two_more_think']['intact']:.2f} | {'pass' if gate(reading) else ''} |"
            )
    return "\n".join(lines)


def digests() -> dict[str, str]:
    files = [Path(__file__), HERE / "token_binding.py", HERE / "learned_read.py",
             tb.COMPETITORS_PATH, HERE / "CHAIN_PROTOCOL.md"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learners", nargs="*", default=list(LEARNERS), choices=LEARNERS)
    parser.add_argument("--rules", nargs="*", default=["act"], choices=RULES)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("out already exists; results are never overwritten")
    specs = [
        {"learner": learner, "rule": rule, "seed": seed, "budgets": sorted(args.budgets)}
        for learner in args.learners
        for rule in (["gru"] if learner == "gru" else args.rules)
        for seed in args.seeds
    ]
    began = time.monotonic()
    with get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(life, specs, chunksize=1)
    body = {"schema": "chain-binding/1", "digests": digests(), "rows": rows,
            "elapsed_seconds": time.monotonic() - began,
            "constants": {"TOKENS": TOKENS, "NEUTRAL": NEUTRAL, "THINK": THINK, "BATCH": BATCH,
                          "READ": READ, "GATE": GATE}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
