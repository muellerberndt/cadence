"""Token binding: named facts as shared tokens, read on name-value pairs never shown in training.

A trial shows facts, each one event ``is + name + value`` in shared token coordinates, then
neutral events, then a query ``ask + name`` that repeats the name's token; the answer is the
value bound to that name, one output slot of ``VALUES`` choices. Every value token occurs in
training, and every name token, but each founder's held-out name-value pairs never occur. A
table over trained patterns of names and values cannot answer a held-out pair; a learner that
binds a value to its name and reads it back by the name can. Reach panels go beyond training:
three facts where training shows two, longer delays, and a name never shown in training.

Learners on the same trials, from the same founder seeds:

* ``graph``: the composed brain, its working trace the only carrier of the facts.
* ``content``: the same composition plus a ``recall`` region (one neuron per input) read by
  the association region through a plastic projection. A content memory, the library's
  ``FastSynapses`` from the sensory neurons to the recall region, writes every event as it is
  heard (key and value: the event itself, an unnormalized Hebbian outer product) and is read
  by the present observation before each settle, so the query recalls the event that shares
  its name token, at the strength of that one shared coordinate. This is an instrument-
  level candidate for a witnessed-content store; nothing here changes the library.
* ``gru``: a supervised online GRU (``benchmarks/competitors.py``'s cell, a softmax over the
  values, cross-entropy through the trial), the competent conventional control.

Lessons come at the query only: judged by the brain's own act (``learning_surprise`` with the
act's answer) or on every row. Each life is read at every budget on copies: intact, with the
content memory erased before the query, with the working trace erased, and with another
trial's content memory transplanted. The protocol and gates are in ``BINDING_PROTOCOL.md``.

    PYTHONPATH=src python benchmarks/binding/token_binding.py --out DIR --workers 2
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from cadence import Brain
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, motor_cortex, prefrontal_cortex
from cadence.stream import FastSynapses

HERE = Path(__file__).resolve().parent
COMPETITORS_PATH = HERE.parent / "competitors.py"

NAMES = 5  # the last name is never shown in training: the new-name reach panel
TRAIN_NAMES = NAMES - 1
VALUES = 8
MARKERS = ("is", "ask", "neutral")
INPUTS = len(MARKERS) + NAMES + VALUES
FACTS = 2  # facts per training trial
NEUTRAL = 1  # neutral events between the facts and the query in training
BATCH = 16  # trials per round, one stream each, one lesson per round
BUDGETS = (256, 1024, 4096)
FOUNDERS = tuple(range(6))
WIDTH = 64
LATERAL = 0.0
CONTENT = {"amplitude": 3.0, "scale": 1.2, "decay": 1.0, "rule": "hebb"}
GRU = {"hidden": 64, "rate": 1.0, "clip": 5.0}
SCHEDULE_SEED, HOLDOUT_SEED, PANEL_SEED, GRU_SEED = 41017, 41018, 41019, 41020
LEARNERS = ("graph", "content", "content_delta", "gru")
RULES = ("act", "every")
CONDITIONS = ("intact", "content_erased", "trace_erased", "content_transplant")
PANELS = ("train", "held_out", "three_facts", "long_delay", "new_name", "restated")
GATE = {"held_out": 0.75, "erased_margin": 0.15, "follows_donor": 0.75, "founders": 5}


# -- the task


def coordinate(kind: str, index: int = 0) -> int:
    """The input coordinate of a marker, a name token or a value token."""
    if kind in MARKERS:
        return MARKERS.index(kind)
    if kind == "name":
        return len(MARKERS) + index
    if kind == "value":
        return len(MARKERS) + NAMES + index
    raise ValueError(kind)


def fact_event(names: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros((len(names), INPUTS))
    out[:, coordinate("is")] = 1.0
    out[np.arange(len(names)), coordinate("name") + np.asarray(names)] = 1.0
    out[np.arange(len(names)), coordinate("value") + np.asarray(values)] = 1.0
    return out


def query_event(names: np.ndarray) -> np.ndarray:
    out = np.zeros((len(names), INPUTS))
    out[:, coordinate("ask")] = 1.0
    out[np.arange(len(names)), coordinate("name") + np.asarray(names)] = 1.0
    return out


def neutral_event(n: int) -> np.ndarray:
    out = np.zeros((n, INPUTS))
    out[:, coordinate("neutral")] = 1.0
    return out


def holdout(seed: int) -> np.ndarray:
    """``(TRAIN_NAMES, VALUES)`` True where a name-value pair is held out: two values per
    training name, each value held out for exactly one name, so every value and every
    training name occurs in training."""
    rng = np.random.default_rng(np.random.SeedSequence([HOLDOUT_SEED, seed]))
    order = rng.permutation(VALUES)
    held = np.zeros((TRAIN_NAMES, VALUES), bool)
    per = VALUES // TRAIN_NAMES
    for name in range(TRAIN_NAMES):
        held[name, order[per * name : per * (name + 1)]] = True
    return held


def draw(
    rng: np.random.Generator,
    n: int,
    held: np.ndarray,
    *,
    facts: int = FACTS,
    queried_held: bool = False,
    new_name: bool = False,
    restated: bool = False,
) -> dict[str, np.ndarray]:
    """``n`` trials: names ``(n, facts)``, values ``(n, facts)``, the queried position ``(n,)``.

    Names are distinct within a trial. Training trials avoid every held-out pair. With
    ``queried_held`` the queried fact is a held-out pair; with ``new_name`` the queried fact
    carries the name never shown in training (its value any). With ``restated`` (three facts)
    the first fact's name is stated again in the last with another value, and the query asks
    that name: the later value is the answer."""
    names = np.stack([rng.permutation(TRAIN_NAMES)[:facts] for _ in range(n)])
    values = np.zeros((n, facts), np.int64)
    query = rng.integers(facts, size=n)
    for i in range(n):
        for j in range(facts):
            allowed = np.flatnonzero(~held[names[i, j]])
            if queried_held and j == query[i]:
                allowed = np.flatnonzero(held[names[i, j]])
            values[i, j] = rng.choice(allowed)
        if new_name:
            names[i, query[i]] = NAMES - 1
            values[i, query[i]] = rng.integers(VALUES)
        if restated:
            names[i, facts - 1] = names[i, 0]
            allowed = np.flatnonzero(~held[names[i, 0]])
            allowed = allowed[allowed != values[i, 0]]
            values[i, facts - 1] = rng.choice(allowed)
            query[i] = facts - 1
    return {"names": names, "values": values, "query": query}


def events(trials: dict[str, np.ndarray], neutral: int = NEUTRAL) -> list[np.ndarray]:
    """The events of a batch of trials: each fact, the neutral events, then the query."""
    names, values, query = trials["names"], trials["values"], trials["query"]
    n = len(query)
    out = [fact_event(names[:, j], values[:, j]) for j in range(names.shape[1])]
    out += [neutral_event(n) for _ in range(neutral)]
    out.append(query_event(names[np.arange(n), query]))
    return out


def answers(trials: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(trials["values"][np.arange(len(trials["query"])), trials["query"]])


def revalue(trials: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The same names, order and query with every value drawn anew, the queried one
    different from the original: a donor whose memory differs only in what is bound."""
    values = rng.integers(VALUES, size=trials["values"].shape)
    rows = np.arange(len(trials["query"]))
    asked = trials["values"][rows, trials["query"]]
    shift = rng.integers(1, VALUES, size=len(rows))
    values[rows, trials["query"]] = (asked + shift) % VALUES
    return {"names": trials["names"].copy(), "values": values, "query": trials["query"].copy()}


# -- the composed brain, with and without the content memory


class ContentBrain(Brain):
    """The composed brain, optionally with a recall region read by a content memory.

    ``build`` declares the composition of ``Brain.compose(INPUTS, VALUES, modules=(width,),
    lateral=LATERAL, seed=seed)`` region by region, so that without the recall region the
    connectome is the composed one (a contract test checks equality); the recall region and
    its one projection are appended last, so the brain with it is the brain without it plus
    those neurons and synapses."""

    content: FastSynapses | None = None
    recall_index: np.ndarray | None = None
    content_amplitude: float = 0.0

    @classmethod
    def build(
        cls,
        seed: int,
        *,
        width: int = WIDTH,
        content: dict[str, Any] | None = None,
        **options: Any,
    ) -> ContentBrain:
        regions = [
            Region("sensory", INPUTS),
            Region("association", width),
            prefrontal_cortex(width),
            motor_cortex(VALUES, lateral=LATERAL, slots=1),
        ]
        projections = [
            Projection("sensory", "association", scale=1.0, reciprocal=False),
            Projection("association", "motor"),
            Projection("prefrontal", "association", scale=1.2, reciprocal=False),
        ]
        if content is not None:
            regions.append(Region("recall", INPUTS))
            projections.append(
                Projection("recall", "association", scale=content["scale"], reciprocal=False)
            )
        genome = Genome(tuple(regions), tuple(projections), label="composed-brain")
        brain = cls(develop(genome, seed=seed), seed=seed, slots=1, **options)
        if content is not None:
            brain.recall_index = np.asarray(
                brain.connectome.populations["recall"], dtype=np.int64
            )
            brain.content = FastSynapses(
                brain.sensory_index, brain.recall_index, decay=content["decay"],
                rate=1.0, amplitude=1.0, rule=content["rule"],
            )
            brain.content_amplitude = float(content["amplitude"])
        return brain

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        drive = super().stimulus(observations, memory=memory)
        if memory and self.content is not None and self.recall_index is not None:
            x = np.asarray(self._observations(observations), dtype=float)
            if len(self.content.strength) == len(x):
                drive[:, self.recall_index] = self.content_amplitude * self.content.recall(x)
        return drive

    def _choose(self, x: np.ndarray, drive: np.ndarray, free: Any, **kwargs: Any) -> np.ndarray:
        action = super()._choose(x, drive, free, **kwargs)
        if self.content is not None:
            x = np.asarray(x, dtype=float)
            if len(self.content.strength) != len(x):
                self.content.reset(len(x))
            self.content.observe(x, x)  # the event as heard, keyed by itself
        return action

    def reset(self) -> None:
        super().reset()
        if self.content is not None:
            self.content.reset(0)


def fork(brain: ContentBrain) -> ContentBrain:
    """An independent copy of a life: every array, store and random state copied, the
    immutable connectome shared. Read-only reports (mapping proxies) are copied as plain
    mappings and restored as proxies."""
    proxies = {k: v for k, v in vars(brain).items() if isinstance(v, MappingProxyType)}
    for key, value in proxies.items():
        setattr(brain, key, dict(value))
    try:
        twin = copy.deepcopy(brain, {id(brain.connectome): brain.connectome})
    finally:
        for key, value in proxies.items():
            setattr(brain, key, value)
    for key in proxies:
        setattr(twin, key, MappingProxyType(getattr(twin, key)))
    return twin


# -- work accounting and the lesson at the query


def new_work() -> dict[str, Any]:
    return {
        "acts": 0, "act_rows": 0, "act_sweeps": 0, "act_refusals": 0,
        "lessons": 0, "lesson_rows": 0, "rows_taught": 0, "lesson_sweeps": 0,
        "lesson_refusals": 0,
    }


def act(brain: Brain, x: np.ndarray, work: dict[str, Any]) -> np.ndarray | None:
    work["acts"] += 1
    work["act_rows"] += len(x)
    try:
        answer = brain.act(x, greedy=True)
    except RuntimeError:
        work["act_refusals"] += 1
        return None
    work["act_sweeps"] += int(brain.last_settlement["steps"])
    return np.asarray(answer)


def lesson(brain: Brain, x: np.ndarray, target: np.ndarray, work: dict[str, Any]) -> None:
    """One lesson at the query from the complete pre-query stimulus; under the surprise rule
    the rows are judged by the brain's own answer, the settle its act reads."""
    from cadence.learning import LearningPhaseError

    drive = brain.stimulus(x)
    answer = None
    if brain.learner.config.surprise:
        answer = brain._choices(brain._qualified(drive, brain._activity(), operation="act"))
    work["lessons"] += 1
    work["lesson_rows"] += len(target)
    try:
        _, report = brain.learner.step(drive, target, answer=answer)
    except LearningPhaseError:
        work["lesson_refusals"] += 1
        return
    work["rows_taught"] += int(report["accepted_presentations"])
    work["lesson_sweeps"] += int(report["total_steps"])


def graph_round(brain: ContentBrain, trials: dict[str, np.ndarray], work: dict[str, Any]) -> bool:
    brain.reset()
    seq = events(trials)
    for x in seq[:-1]:
        if act(brain, x, work) is None:
            return False
    lesson(brain, seq[-1], answers(trials), work)
    return act(brain, seq[-1], work) is not None


def graph_read(
    brain: ContentBrain, trials: dict[str, np.ndarray], condition: str, neutral: int,
    donor: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Answers of a copy on a panel: the prefix, the lesion, then the greedy query act."""
    copy_ = fork(brain)
    copy_.reset()
    seq = events(trials, neutral)
    work = new_work()
    for x in seq[:-1]:
        act(copy_, x, work)
    if condition == "content_erased" and copy_.content is not None:
        copy_.content.reset(0)
    elif condition == "trace_erased" and copy_.working_memory is not None:
        copy_.working_memory.reset(0)
    elif condition == "content_transplant" and copy_.content is not None and donor is not None:
        other = fork(brain)
        other.reset()
        for x in events(donor, neutral)[:-1]:
            act(other, x, work)
        assert other.content is not None
        copy_.content.strength = other.content.strength.copy()
        copy_.content.mass = other.content.mass.copy()
    out = act(copy_, seq[-1], work)
    return np.full(len(trials["query"]), -1) if out is None else out


# -- the supervised GRU control


def _competitors() -> Any:
    spec = importlib.util.spec_from_file_location("competitors", COMPETITORS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Supervised:
    """A GRU read by one softmax over the values, one online step per round on the mean
    cross-entropy at the query, through every event of the trial, bounded in norm."""

    def __init__(self, seed: int, *, hidden: int = GRU["hidden"], rate: float = GRU["rate"],
                 clip: float = GRU["clip"]) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([GRU_SEED, seed]))
        self.hidden, self.rate, self.clip = hidden, rate, clip
        self.cell = _competitors().GRUCell(INPUTS, hidden, rng)
        self.readout = {"W": np.zeros((VALUES, hidden)), "b": np.zeros(VALUES)}
        self.work = {"updates": 0, "rows": 0, "clipped": 0}

    def parameters(self) -> dict[str, np.ndarray]:
        return {**self.readout, **self.cell.parameters()}

    def run(self, seq: list[np.ndarray]) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        n = len(seq[0])
        h = np.zeros((n, self.hidden))
        caches: list[list[dict[str, Any]]] = [[] for _ in range(n)]
        for x in seq:
            for i in range(n):
                h[i], cache = self.cell.forward(x[i], h[i].copy())
                caches[i].append(cache)
        return h, caches

    def probabilities(self, h: np.ndarray) -> np.ndarray:
        logits = h @ self.readout["W"].T + self.readout["b"]
        logits = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        return np.asarray(p / p.sum(axis=-1, keepdims=True))

    def answer(self, seq: list[np.ndarray]) -> np.ndarray:
        h, _ = self.run(seq)
        return np.argmax(self.probabilities(h), axis=-1).astype(np.int64)

    def lesson(self, seq: list[np.ndarray], target: np.ndarray) -> None:
        h, caches = self.run(seq)
        p = self.probabilities(h)
        n = len(target)
        onehot = np.zeros_like(p)
        onehot[np.arange(n), target] = 1.0
        dlogits = (p - onehot) / n
        grads = {k: np.zeros_like(v) for k, v in self.parameters().items()}
        grads["W"] = dlogits.T @ h
        grads["b"] = dlogits.sum(axis=0)
        w = self.cell.weights
        for i in range(n):
            g = self.readout["W"].T @ dlogits[i]
            for cache in reversed(caches[i]):
                part = self.cell.backward(g, cache)
                for key, value in part.items():
                    grads[key] += value
                g = (
                    g * cache["z"] + w["Uz"].T @ part["bz"] + w["Ur"].T @ part["br"]
                    + (w["Un"].T @ part["bn"]) * cache["r"]
                )
        norm = float(np.sqrt(sum(float(np.sum(v * v)) for v in grads.values())))
        scale = min(1.0, self.clip / norm) if norm > 0 else 1.0
        self.work["clipped"] += int(scale < 1.0)
        for key, value in self.parameters().items():
            value -= self.rate * scale * grads[key]
        self.work["updates"] += 1
        self.work["rows"] += n


# -- panels and readings


def panels(seed: int, held: np.ndarray) -> dict[str, tuple[dict[str, np.ndarray], int]]:
    """The reading panels of a founder, each with its neutral-event count."""
    rng = np.random.default_rng(np.random.SeedSequence([PANEL_SEED, seed]))
    return {
        "train": (draw(rng, 128, held), NEUTRAL),
        "held_out": (draw(rng, 128, held, queried_held=True), NEUTRAL),
        "three_facts": (draw(rng, 128, held, facts=3), NEUTRAL),
        "long_delay": (draw(rng, 128, held), 4),
        "new_name": (draw(rng, 128, held, new_name=True), NEUTRAL),
        "restated": (draw(rng, 128, held, facts=3, restated=True), NEUTRAL),
    }


def accuracy(given: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.asarray(given) == np.asarray(truth)))


def read_graph(brain: ContentBrain, seed: int, held: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(np.random.SeedSequence([PANEL_SEED, seed, 1]))
    for name, (trials, neutral) in panels(seed, held).items():
        truth = answers(trials)
        donor = revalue(trials, rng)
        row = {}
        for condition in CONDITIONS:
            if condition.startswith("content") and brain.content is None:
                continue
            given = graph_read(brain, trials, condition, neutral, donor)
            row[condition] = accuracy(given, truth)
            if condition == "content_transplant":
                row["follows_donor"] = accuracy(given, answers(donor))
        out[name] = row
    return out


def read_gru(model: Supervised, seed: int, held: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (trials, neutral) in panels(seed, held).items():
        out[name] = {"intact": accuracy(model.answer(events(trials, neutral)), answers(trials))}
    return out


def life(spec: dict[str, Any]) -> dict[str, Any]:
    """One founder, learner and rule, read at every budget."""
    seed, learner, rule = spec["seed"], spec["learner"], spec["rule"]
    held = holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([SCHEDULE_SEED, seed]))
    started = time.monotonic()
    readings, work = [], new_work()
    if learner == "gru":
        model = Supervised(seed)
    else:
        brain = ContentBrain.build(
            seed, width=spec.get("width", WIDTH),
            content=(
                dict(CONTENT, **spec.get("content", {}))
                if learner == "content"
                else dict(CONTENT, rule="delta", **spec.get("content", {}))
                if learner == "content_delta"
                else None
            ),
            learning_surprise=rule == "act",
        )
    done = 0
    for budget in spec.get("budgets", BUDGETS):
        while done < budget:
            trials = draw(rng, BATCH, held)
            if learner == "gru":
                model.lesson(events(trials), answers(trials))
            else:
                graph_round(brain, trials, work)
            done += 1
        reading = read_gru(model, seed, held) if learner == "gru" else read_graph(brain, seed, held)
        readings.append({"rounds": budget, "panels": reading,
                         "seconds": time.monotonic() - started})
    return {**spec, "held": held.tolist(), "readings": readings,
            "work": model.work if learner == "gru" else work}


def gate(row: dict[str, Any]) -> bool:
    """The declared per-founder gate on one reading: held-out accuracy at least the floor;
    erasing the carrier of the facts brings it within the margin of chance; and, for a content
    memory, the answers follow a transplanted donor memory on at least the floor's share."""
    held = row["panels"]["held_out"]
    chance = 1.0 / VALUES
    erased = held.get("content_erased", held.get("trace_erased", 0.0))
    follows = held.get("follows_donor")
    transplant = follows is None or follows >= GATE["follows_donor"]
    return bool(
        held["intact"] >= GATE["held_out"]
        and erased <= chance + GATE["erased_margin"]
        and transplant
    )


def digests() -> dict[str, str]:
    files = [Path(__file__), COMPETITORS_PATH, HERE / "BINDING_PROTOCOL.md"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| learner | rule | rounds | founder | train | held-out | erased | three facts | "
        "long delay | new name | restated | transplant follows donor | gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        for reading in row["readings"]:
            p = reading["panels"]
            held = p["held_out"]
            erased = held.get("content_erased", held.get("trace_erased"))
            follows = held.get("follows_donor")
            lines.append(
                f"| {row['learner']} | {row['rule']} | {reading['rounds']} | {row['seed']} | "
                f"{p['train']['intact']:.2f} | {held['intact']:.2f} | "
                f"{'' if erased is None else f'{erased:.2f}'} | "
                f"{p['three_facts']['intact']:.2f} | {p['long_delay']['intact']:.2f} | "
                f"{p['new_name']['intact']:.2f} | {p['restated']['intact']:.2f} | "
                f"{'' if follows is None else f'{follows:.2f}'} | "
                f"{'pass' if row['learner'] != 'gru' and gate(reading) else ''} |"
            )
    return "\n".join(lines)


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
    specs = []
    for learner in args.learners:
        for rule in (["gru"] if learner == "gru" else args.rules):
            for seed in args.seeds:
                specs.append({"learner": learner, "rule": rule, "seed": seed,
                              "budgets": sorted(args.budgets)})
    began = time.monotonic()
    with get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(life, specs, chunksize=1)
    body = {"schema": "token-binding/1", "digests": digests(), "rows": rows,
            "elapsed_seconds": time.monotonic() - began,
            "constants": {"NAMES": NAMES, "VALUES": VALUES, "FACTS": FACTS, "NEUTRAL": NEUTRAL,
                          "BATCH": BATCH, "WIDTH": WIDTH, "LATERAL": LATERAL,
                          "CONTENT": CONTENT, "GRU": GRU, "GATE": GATE}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
