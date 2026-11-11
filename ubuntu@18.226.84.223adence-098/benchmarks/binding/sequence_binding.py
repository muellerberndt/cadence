"""In-context succession: what followed this token, in a sequence heard once.

A trial shows a sequence of distinct tokens, one event ``seq + token`` each, then a cue
``ask + token`` naming one token of the sequence other than the last; the answer is the token
that followed it in this sequence, one output slot of ``TOKENS`` choices. Every sequence is
drawn fresh. Each founder holds out one successor per token: those transitions never occur in
training, while every token occurs, before and after other tokens. A table over transitions
learned in the weights answers a held-out transition with a successor trained for the cue;
a learner that remembers this sequence's transitions and reads them by the cue answers it.
Reach panels: longer sequences and a token never shown in training.

Learners, from the same founder seeds (the composition of ``token_binding.ContentBrain``):

* ``graph``: the composed brain, its working trace the only carrier of the sequence.
* ``event``: plus a recall region read by a content memory that writes each event keyed by
  itself (the store of ``token_binding``), a control: it recalls the cue's own event.
* ``successor``: plus a recall region read by a successor memory that writes each event keyed
  by the event before it, so the present event recalls what followed it.
* ``gru``: a supervised online GRU on the same trials.

Lessons at the cue only, judged by the brain's own act or on every row. Instrument level;
nothing here changes the library. The protocol is declared in ``SEQUENCE_PROTOCOL.md``.

    PYTHONPATH=src python benchmarks/binding/sequence_binding.py --out OUT.json --workers 2
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

from cadence import Brain
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, motor_cortex, prefrontal_cortex
from cadence.stream import FastSynapses

HERE = Path(__file__).resolve().parent


def _sibling(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


tb = _sibling("token_binding")

TOKENS = 9  # the last token is never shown in training: the new-token reach panel
TRAIN_TOKENS = TOKENS - 1
MARKERS = ("seq", "ask", "neutral")
INPUTS = len(MARKERS) + TOKENS
LENGTH = 5  # tokens per training sequence
BATCH = 16
BUDGETS = (64, 256, 1024)
FOUNDERS = tuple(range(6))
WIDTH = 64
LATERAL = 0.0
MEMORY = {"amplitude": 3.0, "scale": 1.2, "decay": 1.0, "rule": "hebb"}
GRU = {"hidden": 64, "rate": 1.0, "clip": 5.0}
SCHEDULE_SEED, HOLDOUT_SEED, PANEL_SEED, GRU_SEED = 51017, 51018, 51019, 51020
LEARNERS = ("graph", "event", "successor", "gru")
RULES = ("act", "every")
PANELS = ("train", "held_out", "long", "new_token")
GATE = {"held_out": 0.75, "erased_margin": 0.15, "founders": 5}


# -- the task


def input_width(token_count: int = TOKENS) -> int:
    """Input coordinates: the markers and ``token_count`` tokens."""
    return len(MARKERS) + token_count


def token_event(tokens: np.ndarray, marker: str, token_count: int = TOKENS) -> np.ndarray:
    out = np.zeros((len(tokens), input_width(token_count)))
    out[:, MARKERS.index(marker)] = 1.0
    out[np.arange(len(tokens)), len(MARKERS) + np.asarray(tokens)] = 1.0
    return out


def holdout(seed: int, token_count: int = TOKENS) -> np.ndarray:
    """``(token_count - 1, token_count - 1)`` True where the transition ``a -> b`` of two
    training tokens is held out: one successor per token, a permutation without fixed points,
    so every token keeps other successors and other predecessors in training. The last token
    is never trained."""
    train = token_count - 1
    if train < 3:
        raise ValueError("the held-out successors need three training tokens at least")
    rng = np.random.default_rng(np.random.SeedSequence([HOLDOUT_SEED, seed]))
    while True:
        order = rng.permutation(train)
        if not (order == np.arange(train)).any():
            break
    held = np.zeros((train, train), bool)
    held[np.arange(train), order] = True
    return held


def draw(
    rng: np.random.Generator,
    n: int,
    held: np.ndarray,
    *,
    length: int = LENGTH,
    cue_held: bool = False,
    new_token: bool = False,
) -> dict[str, np.ndarray]:
    """``n`` trials: sequences ``(n, length)`` of distinct tokens and the cued position.

    The tokens are the training tokens of ``held`` (one per row). Training sequences contain
    no held-out transition. With ``cue_held`` the cued transition is a held-out one and the
    others are not; with ``new_token`` the token never shown in training sits at the cued
    position."""
    if not 2 <= length <= len(held):
        raise ValueError("a sequence holds two to all of the training tokens")
    sequences = np.zeros((n, length), np.int64)
    cue = rng.integers(length - 1, size=n)
    for i in range(n):
        while True:
            seq = rng.permutation(len(held))[:length]
            steps = held[seq[:-1], seq[1:]]
            if cue_held:
                if steps[cue[i]] and steps.sum() == 1:
                    break
            elif not steps.any():
                break
        if new_token:
            seq[cue[i]] = len(held)  # the token after the training tokens
        sequences[i] = seq
    return {"sequences": sequences, "cue": cue}


def events(trials: dict[str, np.ndarray], token_count: int = TOKENS) -> list[np.ndarray]:
    """The events of a batch of trials on the inputs of ``token_count`` tokens: each token of
    the sequence, then the cue."""
    seqs, cue = trials["sequences"], trials["cue"]
    out = [token_event(seqs[:, j], "seq", token_count) for j in range(seqs.shape[1])]
    out.append(token_event(seqs[np.arange(len(cue)), cue], "ask", token_count))
    return out


def answers(trials: dict[str, np.ndarray]) -> np.ndarray:
    seqs, cue = trials["sequences"], trials["cue"]
    return np.asarray(seqs[np.arange(len(cue)), cue + 1])


# -- the composed brain with an optional memory read into a recall region


class MemoryBrain(Brain):
    """The composed brain plus a recall region read by one fast associative memory: ``event``
    writes each event keyed by itself, ``successor`` keyed by the event before it; both are
    read by the present observation before each settle."""

    memory: FastSynapses | None = None
    mode: str | None = None
    recall_index: np.ndarray | None = None
    memory_amplitude: float = 0.0
    previous: np.ndarray | None = None

    @classmethod
    def build(
        cls, seed: int, *, mode: str | None = None, width: int = WIDTH,
        memory: dict[str, Any] | None = None, **options: Any,
    ) -> MemoryBrain:
        memory = dict(MEMORY, **(memory or {}))
        regions = [
            Region("sensory", INPUTS),
            Region("association", width),
            prefrontal_cortex(width),
            motor_cortex(TOKENS, lateral=LATERAL, slots=1),
        ]
        projections = [
            Projection("sensory", "association", scale=1.0, reciprocal=False),
            Projection("association", "motor"),
            Projection("prefrontal", "association", scale=1.2, reciprocal=False),
        ]
        if mode is not None:
            regions.append(Region("recall", INPUTS))
            projections.append(
                Projection("recall", "association", scale=memory["scale"], reciprocal=False)
            )
        genome = Genome(tuple(regions), tuple(projections), label="composed-brain")
        brain = cls(develop(genome, seed=seed), seed=seed, slots=1, **options)
        if mode is not None:
            if mode not in ("event", "successor"):
                raise ValueError("mode is 'event' or 'successor'")
            brain.mode = mode
            brain.recall_index = np.asarray(
                brain.connectome.populations["recall"], dtype=np.int64
            )
            brain.memory = FastSynapses(
                brain.sensory_index, brain.recall_index, decay=memory["decay"],
                rate=1.0, amplitude=1.0, rule=memory["rule"],
            )
            brain.memory_amplitude = float(memory["amplitude"])
        return brain

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        drive = super().stimulus(observations, memory=memory)
        if memory and self.memory is not None and self.recall_index is not None:
            x = np.asarray(self._observations(observations), dtype=float)
            if len(self.memory.strength) == len(x):
                drive[:, self.recall_index] = self.memory_amplitude * self.memory.recall(x)
        return drive

    def _choose(self, x: np.ndarray, drive: np.ndarray, free: Any, **kwargs: Any) -> np.ndarray:
        action = super()._choose(x, drive, free, **kwargs)
        if self.memory is not None:
            x = np.asarray(x, dtype=float)
            if len(self.memory.strength) != len(x):
                self.memory.reset(len(x))
                self.previous = None
            if self.mode == "event":
                self.memory.observe(x, x)
            elif self.previous is not None and len(self.previous) == len(x):
                self.memory.observe(self.previous, x)  # what followed the previous event
            self.previous = x.copy()
        return action

    def reset(self) -> None:
        super().reset()
        if self.memory is not None:
            self.memory.reset(0)
        self.previous = None


def train_round(brain: MemoryBrain, trials: dict[str, np.ndarray], work: dict[str, Any]) -> bool:
    brain.reset()
    seq = events(trials)
    for x in seq[:-1]:
        if tb.act(brain, x, work) is None:
            return False
    tb.lesson(brain, seq[-1], answers(trials), work)
    return tb.act(brain, seq[-1], work) is not None


def read(brain: MemoryBrain, trials: dict[str, np.ndarray], condition: str) -> np.ndarray:
    twin = tb.fork(brain)
    twin.reset()
    seq = events(trials)
    work = tb.new_work()
    for x in seq[:-1]:
        tb.act(twin, x, work)
    if condition == "memory_erased" and twin.memory is not None:
        twin.memory.reset(0)
    elif condition == "trace_erased" and twin.working_memory is not None:
        twin.working_memory.reset(0)
    out = tb.act(twin, seq[-1], work)
    return np.full(len(trials["cue"]), -1) if out is None else out


# -- the supervised GRU control


class Supervised:
    """``token_binding.Supervised`` for this task's inputs and outputs."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([GRU_SEED, seed]))
        self.hidden, self.rate, self.clip = GRU["hidden"], GRU["rate"], GRU["clip"]
        self.cell = tb._competitors().GRUCell(INPUTS, self.hidden, rng)
        self.readout = {"W": np.zeros((TOKENS, self.hidden)), "b": np.zeros(TOKENS)}
        self.work = {"updates": 0, "rows": 0, "clipped": 0}

    parameters = tb.Supervised.parameters
    run = tb.Supervised.run
    probabilities = tb.Supervised.probabilities
    answer = tb.Supervised.answer
    lesson = tb.Supervised.lesson


# -- readings


def panels(seed: int, held: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(np.random.SeedSequence([PANEL_SEED, seed]))
    return {
        "train": draw(rng, 128, held),
        "held_out": draw(rng, 128, held, cue_held=True),
        "long": draw(rng, 128, held, length=8),
        "new_token": draw(rng, 128, held, new_token=True),
    }


def read_life(learner: Any, seed: int, held: np.ndarray, gru: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, trials in panels(seed, held).items():
        truth = answers(trials)
        if gru:
            out[name] = {"intact": tb.accuracy(learner.answer(events(trials)), truth)}
            continue
        row = {"intact": tb.accuracy(read(learner, trials, "intact"), truth),
               "trace_erased": tb.accuracy(read(learner, trials, "trace_erased"), truth)}
        if learner.memory is not None:
            row["memory_erased"] = tb.accuracy(read(learner, trials, "memory_erased"), truth)
        out[name] = row
    return out


def life(spec: dict[str, Any]) -> dict[str, Any]:
    seed, learner, rule = spec["seed"], spec["learner"], spec["rule"]
    held = holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([SCHEDULE_SEED, seed]))
    started = time.monotonic()
    readings, work = [], tb.new_work()
    gru = learner == "gru"
    if gru:
        model: Any = Supervised(seed)
    else:
        model = MemoryBrain.build(
            seed, mode=None if learner == "graph" else learner,
            learning_surprise=rule == "act",
        )
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
    held = reading["panels"]["held_out"]
    erased = held.get("memory_erased", held.get("trace_erased", 0.0))
    chance = 1.0 / TRAIN_TOKENS
    return bool(held["intact"] >= GATE["held_out"] and erased <= chance + GATE["erased_margin"])


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| learner | rule | rounds | founder | train | held-out | memory erased | "
        "trace erased | long | new token | gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        for reading in row["readings"]:
            p = reading["panels"]
            held = p["held_out"]
            erased = held.get("memory_erased")
            trace = held.get("trace_erased")
            lines.append(
                f"| {row['learner']} | {row['rule']} | {reading['rounds']} | {row['seed']} | "
                f"{p['train']['intact']:.2f} | {held['intact']:.2f} | "
                f"{'' if erased is None else f'{erased:.2f}'} | "
                f"{'' if trace is None else f'{trace:.2f}'} | "
                f"{p['long']['intact']:.2f} | {p['new_token']['intact']:.2f} | "
                f"{'pass' if row['learner'] != 'gru' and gate(reading) else ''} |"
            )
    return "\n".join(lines)


def digests() -> dict[str, str]:
    files = [Path(__file__), HERE / "token_binding.py", tb.COMPETITORS_PATH,
             HERE / "SEQUENCE_PROTOCOL.md"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learners", nargs="*", default=list(LEARNERS), choices=LEARNERS)
    parser.add_argument("--rules", nargs="*", default=list(RULES), choices=RULES)
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
    body = {"schema": "sequence-binding/1", "digests": digests(), "rows": rows,
            "elapsed_seconds": time.monotonic() - began,
            "constants": {"TOKENS": TOKENS, "LENGTH": LENGTH, "BATCH": BATCH, "WIDTH": WIDTH,
                          "LATERAL": LATERAL, "MEMORY": MEMORY, "GRU": GRU, "GATE": GATE}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
