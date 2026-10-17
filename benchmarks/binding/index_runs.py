"""Declared lives of the index memory on the binding chambers; protocol in ``INDEX_PROTOCOL.md``.

One life per task, arm and founder, read at every budget on forks that never change the life.
The tasks reuse their instruments' trials, panels and gates function by function:

- ``token``: ``token_binding``, named facts and a query that repeats the name; the slots are
  erased, or replaced by a donor trial's;
- ``fresh``: ``relevance/bindings``, facts in blocks and a query that names the block by a
  marker; the slots are erased (with the trace: ``erased``);
- ``succession``: ``sequence_binding``, what followed the cued token; the slots are erased;
- ``chain``: ``chain_binding``, one link or two; the slots are erased;
- ``continuing``: ``token_binding`` with two earlier trials of the same life in the slots; the
  slots are erased;
- ``wide``: ``token_binding`` with every event rendered at camera width (each token a fixed
  patch of 400 of 12,824 inputs, noise pixels each frame); the memory addresses its sparse
  code; the slots are erased;
- ``camera``: ``token_binding`` with every event a frame of 12,824 inputs (each token a
  contiguous block of a 128 x 100 image, noise pixels each frame) seen through a retina of
  local pooling units, the brain's observation and the memory's key alike; the slots are
  erased.

At camera width the training frames draw their noise from the training stream's generator
alone, and each reading panel from one of its own, drawn afresh at every reading: every
reading and every condition sees the same frames, and a reading changes no training frame.

Arms: ``index`` (the declared genes) and ``heard`` (the same with the address lesson off, the
address staying the event heard). Nothing here changes the library.

    PYTHONPATH=src python benchmarks/binding/index_runs.py --task token --out DIR --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "relevance"))


def _sibling(name: str, path: Path | None = None) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path or HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ir = _sibling("index_read")
tb = ir.tb

TASKS = ("token", "fresh", "named", "succession", "chain", "continuing", "wide", "camera")
ARMS = {
    "index": {"address": 0.01, "recency": 0.25},
    "heard": {"address": 0.0, "recency": 0.25},
    # the options of amendment 5, each beside the index arm
    "toward": {"address": 0.01, "recency": 0.25, "address_step": "toward"},
    "horizon": {"address": 0.01, "recency": None, "capacity": 64},
    "null": {"address": 0.01, "recency": 0.25},
    "improved": {"address": 0.01, "recency": None, "capacity": 64, "address_step": "toward"},
}
HALF_NULL = ("null", "improved")  # a null slot at half the logit of a full unit-key match
CONTINUING = {"capacity": 12}  # the continuing task's slots: three trials
BUDGETS = {
    "token": (256, 1024),
    "fresh": (512, 2048),
    "named": (512, 2048, 4096),
    "succession": (256, 1024),
    "chain": (1024, 4096),
    "continuing": (256, 1024),
    "wide": (256, 1024),
    "camera": (256, 1024),
}
# the camera task: each token a contiguous block of a 128 x 100 image (plus 24 body inputs,
# 12,824 inputs in all), noise pixels on every frame; a retina of local pooling units codes the
# frame for the cortex and the memory alike
CAMERA = {"height": 100, "width": 128, "body": 24, "block": 20, "noise": 0.01, "window": 8,
          "stride": 4, "threshold": 0.5, "winners": 128, "heard": 1.5, "render_seed": 61222}
# the wide task: each token a fixed patch of a camera-sized input, noise pixels on every frame
WIDE = {"inputs": 12_824, "patch": 400, "noise": 0.01, "code": 1024, "code_fan_in": 32,
        "code_winners": 24, "render_seed": 61221}
FOUNDERS = tuple(range(6))
WINDOW = 64  # rounds per entry of the training log
# the panels the camera-width tasks read; each is rendered with noise of its own, the same at
# every reading and in every condition, and never drawn from the training stream's
FRAME_PANELS = ("train", "held_out", "new_name", "three_facts")


def genes_for(task: str, arm: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    genes = dict(ir.GENES, **ARMS[arm])
    if task == "continuing":
        genes.update(CONTINUING)
    if task == "wide":
        genes.update({k: WIDE[k] for k in ("code", "code_fan_in", "code_winners")})
    if task == "camera":
        genes.update({"heard": CAMERA["heard"]})  # unit-length codes: the key's scale
    if arm in HALF_NULL:
        genes["null"] = genes["heard"] ** 2 / 2
    genes.update(extra or {})
    return genes


def erase(brain: Any, condition: str, donor: Any = None) -> None:
    """The memory's lesion before the read act: the slots erased or replaced, or the trace
    erased."""
    slots = brain.index.slots
    if condition == "memory_erased":
        slots.reset(slots.streams)
    elif condition == "trace_erased" and brain.working_memory is not None:
        brain.working_memory.reset(0)
    elif condition == "memory_transplant" and donor is not None:
        slots.take(donor)


def episode(brain: Any, seq: list[np.ndarray], answer: np.ndarray, work: dict[str, Any]) -> None:
    """One training trial: the events, the lesson at the last, then its act. The training log
    keeps, per window of rounds, the share of judged answers that were right and the size of
    the address map."""
    for x in seq[:-1]:
        tb.act(brain, x, work)
    judged = ir.lesson(brain, seq[-1], answer, work)
    tb.act(brain, seq[-1], work)
    log = work.setdefault("log", {"right": [], "address_norm": []})
    window = work.setdefault("_window", [0.0, 0])
    window[0] += float(np.mean(np.asarray(judged).reshape(len(answer), -1)
                               == np.asarray(answer).reshape(len(answer), -1)))
    window[1] += 1
    if window[1] == WINDOW:
        index = brain.index
        log["right"].append(round(window[0] / WINDOW, 4))
        log["address_norm"].append(round(float(np.sqrt(
            (index.w_x ** 2).sum() + (index.w_c ** 2).sum() + (index.b ** 2).sum())), 4))
        window[:] = [0.0, 0]


# -- token binding


def token_read(brain: Any, trials: dict[str, np.ndarray], condition: str, neutral: int,
               donor: dict[str, np.ndarray] | None = None) -> np.ndarray:
    twin = tb.fork(brain)
    twin.reset()
    seq = tb.events(trials, neutral)
    work = tb.new_work()
    for x in seq[:-1]:
        tb.act(twin, x, work)
    slots = None
    if condition == "memory_transplant" and donor is not None:
        other = tb.fork(brain)
        other.reset()
        for x in tb.events(donor, neutral)[:-1]:
            tb.act(other, x, work)
        slots = other.index.slots
    erase(twin, condition, slots)
    out = tb.act(twin, seq[-1], work)
    return np.full(len(trials["query"]), -1) if out is None else out


def token_panels(brain: Any, seed: int, held: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(np.random.SeedSequence([tb.PANEL_SEED, seed, 1]))
    for name, (trials, neutral) in tb.panels(seed, held).items():
        truth = tb.answers(trials)
        donor = tb.revalue(trials, rng)
        row: dict[str, float] = {}
        for condition in ("intact", "memory_erased", "trace_erased", "memory_transplant"):
            given = token_read(brain, trials, condition, neutral, donor)
            row[condition] = tb.accuracy(given, truth)
            if condition == "memory_transplant":
                row["follows_donor"] = tb.accuracy(given, tb.answers(donor))
        out[name] = row
    return out


def token_gate(panels: dict[str, Any]) -> bool:
    held = panels["held_out"]
    return bool(held["intact"] >= tb.GATE["held_out"]
                and held["memory_erased"] <= 1.0 / tb.VALUES + tb.GATE["erased_margin"]
                and held["follows_donor"] >= tb.GATE["follows_donor"])


def token_life(spec: dict[str, Any]) -> dict[str, Any]:
    seed = spec["seed"]
    held = tb.holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([tb.SCHEDULE_SEED, seed]))
    brain = ir.IndexBrain.build(seed, read=spec["genes"], learning_surprise=True)
    work: dict[str, Any] = tb.new_work()
    readings, done, started = [], 0, time.monotonic()
    for budget in spec["budgets"]:
        while done < budget:
            trials = tb.draw(rng, tb.BATCH, held)
            brain.reset()
            episode(brain, tb.events(trials), tb.answers(trials), work)
            done += 1
        panels = token_panels(brain, seed, held)
        readings.append({"rounds": budget, "seconds": time.monotonic() - started,
                         "panels": panels, "gate": token_gate(panels)})
    return {"readings": readings, "work": work}


# -- token binding in a continuing life


def continuing_read(brain: Any, trials: dict[str, np.ndarray], condition: str,
                    earlier: list[dict[str, np.ndarray]]) -> np.ndarray:
    twin = tb.fork(brain)
    twin.reset()  # the panel's streams: two earlier trials, then the one read
    work = tb.new_work()
    for prior in earlier:
        for x in tb.events(prior):
            tb.act(twin, x, work)
    seq = tb.events(trials)
    for x in seq[:-1]:
        tb.act(twin, x, work)
    erase(twin, condition)
    out = tb.act(twin, seq[-1], work)
    return np.full(len(trials["query"]), -1) if out is None else out


def continuing_life(spec: dict[str, Any]) -> dict[str, Any]:
    """Token binding without a reset between trials: the slots hold earlier trials whose facts
    name the same tokens with other values, the capacity bounding them."""
    seed = spec["seed"]
    held = tb.holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([tb.SCHEDULE_SEED, seed]))
    brain = ir.IndexBrain.build(seed, read=spec["genes"], learning_surprise=True)
    brain.reset()
    work: dict[str, Any] = tb.new_work()
    readings, done, started = [], 0, time.monotonic()
    for budget in spec["budgets"]:
        while done < budget:
            trials = tb.draw(rng, tb.BATCH, held)
            episode(brain, tb.events(trials), tb.answers(trials), work)
            done += 1
        panel_rng = np.random.default_rng(np.random.SeedSequence([tb.PANEL_SEED, seed, 9]))
        panels: dict[str, Any] = {}
        for name, (trials, _) in tb.panels(seed, held).items():
            if name not in ("train", "held_out", "new_name"):
                continue
            n = len(trials["query"])
            earlier = [tb.draw(panel_rng, n, held), tb.draw(panel_rng, n, held)]
            truth = tb.answers(trials)
            panels[name] = {c: tb.accuracy(continuing_read(brain, trials, c, earlier), truth)
                            for c in ("intact", "memory_erased", "trace_erased")}
        held_row = panels["held_out"]
        gate = (held_row["intact"] >= tb.GATE["held_out"]
                and held_row["memory_erased"] <= 1.0 / tb.VALUES + tb.GATE["erased_margin"])
        readings.append({"rounds": budget, "seconds": time.monotonic() - started,
                         "panels": panels, "gate": bool(gate)})
    return {"readings": readings, "work": work}


# -- succession


def succession_life(spec: dict[str, Any]) -> dict[str, Any]:
    sb = _sibling("sequence_binding")
    seed = spec["seed"]
    held = sb.holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([sb.SCHEDULE_SEED, seed]))
    brain = ir.IndexBrain.build(seed, read=spec["genes"], inputs=sb.INPUTS, actions=sb.TOKENS,
                                lateral=sb.LATERAL, learning_surprise=True)
    work: dict[str, Any] = tb.new_work()
    readings, done, started = [], 0, time.monotonic()

    def read(trials: dict[str, np.ndarray], condition: str) -> np.ndarray:
        twin = tb.fork(brain)
        twin.reset()
        seq = sb.events(trials)
        w = tb.new_work()
        for x in seq[:-1]:
            tb.act(twin, x, w)
        erase(twin, condition)
        out = tb.act(twin, seq[-1], w)
        return np.full(len(trials["cue"]), -1) if out is None else out

    for budget in spec["budgets"]:
        while done < budget:
            trials = sb.draw(rng, sb.BATCH, held)
            brain.reset()
            episode(brain, sb.events(trials), sb.answers(trials), work)
            done += 1
        panels = {name: {c: tb.accuracy(read(trials, c), sb.answers(trials))
                         for c in ("intact", "memory_erased", "trace_erased")}
                  for name, trials in sb.panels(seed, held).items()}
        readings.append({"rounds": budget, "seconds": time.monotonic() - started,
                         "panels": panels, "gate": bool(sb.gate({"panels": panels}))})
    return {"readings": readings, "work": work}


# -- chained recall


def chain_life(spec: dict[str, Any]) -> dict[str, Any]:
    cb = _sibling("chain_binding")
    seed = spec["seed"]
    held = cb.holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([cb.SCHEDULE_SEED, seed]))
    brain = ir.IndexBrain.build(seed, read=spec["genes"], inputs=cb.INPUTS, actions=cb.TOKENS,
                                learning_surprise=True)
    work: dict[str, Any] = tb.new_work()
    readings, done, started = [], 0, time.monotonic()

    def read(trials: dict[str, np.ndarray], condition: str, think: int) -> np.ndarray:
        twin = tb.fork(brain)
        twin.reset()
        seq = cb.events(trials, think=think)
        w = tb.new_work()
        for x in seq[:-1]:
            tb.act(twin, x, w)
        erase(twin, condition)
        out = tb.act(twin, seq[-1], w)
        return np.full(len(trials["hops"]), -1) if out is None else out

    for budget in spec["budgets"]:
        while done < budget:
            trials = cb.draw(rng, cb.BATCH, held)
            brain.reset()
            episode(brain, cb.events(trials), cb.answers(trials), work)
            done += 1
        panels = {name: {c: tb.accuracy(read(trials, c, think), cb.answers(trials))
                         for c in ("intact", "memory_erased", "trace_erased")}
                  for name, (trials, think) in cb.panels(seed, held).items()}
        gate = (panels["two_held"]["intact"] >= cb.GATE["two_held"]
                and panels["one_held"]["intact"] >= cb.GATE["one_held"])
        readings.append({"rounds": budget, "seconds": time.monotonic() - started,
                         "panels": panels, "gate": bool(gate)})
    return {"readings": readings, "work": work}


# -- fresh bindings (the task and scores of relevance/bindings, the brain and lesson of this memory)

FRESH = {"bits": 3, "width": 64, "lateral": -0.5}  # the runner's m64 brain on two-neuron slots
FRESH_CONDITIONS = ("intact", "erased", "trace_erased", "memory_erased", "transplant")


_ENCODING = {"task": "fresh"}  # which encoding the fresh-bindings life reads in this process


def _bindings() -> Any:
    if _ENCODING["task"] == "named":
        return _sibling("named_bindings").module()
    return _sibling("bindings", HERE.parent / "relevance" / "bindings.py")


def fresh_brain(seed: int, genes: dict[str, Any], bits: int) -> Any:
    """The founder on the task's ports with the index memory, as the runner composes its
    graph: ``episodic=False`` and one two-neuron slot per bit."""
    bd = _bindings()
    return ir.IndexBrain.build(
        seed, width=FRESH["width"], read=genes, inputs=bd.width(bits), actions=2 * bits,
        lateral=FRESH["lateral"], slots=(2,) * bits, episodic=False, learning_surprise=True,
    )


def fresh_round(brain: Any, work: dict[str, Any], rows: np.ndarray, bits: int) -> tuple[Any, Any]:
    """One round: every stream from rest through its prefix, the lesson at the query (judged by
    the settle the act reads), then the act. Returns the judged answer and the refusal that
    ended the round, if any."""
    bd = _bindings()
    prefix, query, target = bd.inputs(rows, bits)
    brain.reset()
    for observation in prefix:
        if tb.act(brain, observation, work) is None:
            return None, "refused_prefix"
    try:
        answer = ir.lesson(brain, query, target, work)
    except RuntimeError:
        work["act_refusals"] += 1
        return None, "refused_query"
    if tb.act(brain, query, work) is None:
        return answer, "refused_query"
    return answer, None


def fresh_lesion(brain: Any, condition: str, donor: np.ndarray) -> None:
    """The carriers of the prefix before the query: the working trace (the runner's lesion)
    and the slots, each kept, erased or replaced by the donor history's."""
    bd = _bindings()
    trace, slots = brain.working_memory, brain.index.slots
    if condition in ("erased", "trace_erased"):
        bd.lesion(trace, "erased", donor)
    if condition in ("erased", "memory_erased"):
        slots.reset(len(donor))
    if condition == "transplant":
        bd.lesion(trace, "transplant", donor)
        slots.take(slots, donor)


def fresh_competition(brain: Any, panel: np.ndarray, q: int) -> np.ndarray:
    """The weight the query moment's read put on the asked fact's slot, per history: the
    attention the read used, recency included, as the memory recorded it before the query
    moment was written."""
    p = brain.index.attended
    assert p is not None, "the query moment read the slots"
    asked = np.where(q == 0, panel[:, 2], 1 - panel[:, 2])  # A is first under order 0
    return np.asarray(p[np.arange(len(panel)), asked])


def fresh_reading(brain: Any, bits: int, held: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every history through its prefix on a fork of the trained brain, then each condition
    and query on its own fork of that state; no lesson. The life must read the same
    afterwards. ``competition`` reports, on the intact condition, the mean weight on the
    asked fact's slot and the share of cases where it exceeds one half."""
    bd = _bindings()
    before = (brain.brain.efficacy.copy(), brain.brain.bias.copy(), brain.learner.updates)
    panel = bd.histories(bits)
    donor = bd.donors(panel, bits)
    answers = {c: np.full((len(panel), 2, bits), -1, dtype=np.int64) for c in FRESH_CONDITIONS}
    weights = np.full((len(panel), 2), np.nan)
    work = tb.new_work()
    walker = tb.fork(brain)
    walker.reset()
    prefix, _, _ = bd.inputs(bd.asked(panel, 0), bits)
    complete = True
    for observation in prefix:
        if tb.act(walker, observation, work) is None:
            complete = False
            break
    if complete:
        for condition in FRESH_CONDITIONS:
            for q in (0, 1):
                branch = tb.fork(walker)
                fresh_lesion(branch, condition, donor)
                _, query, _ = bd.inputs(bd.asked(panel, q), bits)
                answer = tb.act(branch, query, work)
                if answer is None:
                    complete = False
                else:
                    answers[condition][:, q] = np.asarray(answer).reshape(len(panel), bits)
                    if condition == "intact":
                        weights[:, q] = fresh_competition(branch, panel, q)
    held_rows = held[panel[:, 0], panel[:, 1]]
    contest = {}
    for name, rows in (("training", ~held_rows), ("held_out", held_rows)):
        w = weights[rows]
        contest[name] = {
            "p_asked": float(np.nanmean(w)) if np.isfinite(w).any() else None,
            "decided": float(np.mean(w[np.isfinite(w)] > 0.5)) if np.isfinite(w).any() else None,
        }
    if (
        not np.array_equal(brain.brain.efficacy, before[0])
        or not np.array_equal(brain.brain.bias, before[1])
        or brain.learner.updates != before[2]
    ):
        raise AssertionError("the reading changed the trained brain")
    return {
        "completed": complete,
        "answers": {c: a.tolist() for c, a in answers.items()},
        "scores": bd.score(answers, held, bits),
        "competition": contest,
    }, work


def named_life(spec: dict[str, Any]) -> dict[str, Any]:
    """The fresh-bindings life on ``named_bindings``' encoding: the facts' bits on shared
    coordinates, so an even mixture of the facts cannot answer."""
    _ENCODING["task"] = "named"
    try:
        return fresh_life(spec)
    finally:
        _ENCODING["task"] = "fresh"


def fresh_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life to the longest budget, read at every budget."""
    bd = _bindings()
    seed, bits = spec["seed"], FRESH["bits"]
    budgets = sorted(spec["budgets"])
    held = bd.holdout(seed, bits)
    brain = fresh_brain(seed, spec["genes"], bits)
    work: dict[str, Any] = tb.new_work()
    readings: list[dict[str, Any]] = []
    log: dict[str, Any] = {"window": bd.WINDOW, "taught_rows": [], "act_exact": [],
                           "act_per_bit": [], "lessons_exhausted": []}
    window = np.zeros(4)
    rounds, stop = 0, None
    for rows in bd.trials(seed, bits, budgets[-1], held):
        taught, exhausted = work["rows_taught"], work.get("lessons_exhausted", 0)
        answer, stop = fresh_round(brain, work, rows, bits)
        if stop is not None:
            break
        assert answer is not None
        target = bd.inputs(rows, bits)[2]
        window += (
            work["rows_taught"] - taught,
            float((answer == target).all(axis=1).sum()),
            float((answer == target).mean() * len(rows)),
            work.get("lessons_exhausted", 0) - exhausted,
        )
        rounds += 1
        if rounds % bd.WINDOW == 0 or rounds == budgets[-1]:
            size = (rounds - 1) % bd.WINDOW + 1
            log["taught_rows"].append(int(window[0]))
            log["act_exact"].append(round(window[1] / (size * bd.BATCH), 4))
            log["act_per_bit"].append(round(window[2] / (size * bd.BATCH), 4))
            log["lessons_exhausted"].append(int(window[3]))
            window[:] = 0
        if rounds in budgets:
            reading, _ = fresh_reading(brain, bits, held)
            readings.append({"rounds": rounds, "gate": bool(bd.gate(reading, bits)),
                             "scores": reading["scores"], "competition": reading["competition"],
                             "completed": reading["completed"]})
    training = {"rounds": rounds, "completed": rounds == budgets[-1], "stop": stop, "log": log,
                "held_out_pairs": np.argwhere(held).tolist(), "bits": bits,
                "brain": dict(FRESH)}
    return {"readings": readings, "training": training, "work": work}


# -- token binding at camera width


class Renderer:
    """Each token coordinate a fixed patch of ``patch`` pixels of ``inputs``; an event lights
    its tokens' patches and, each frame anew, ``noise`` of the other pixels. The noise is
    drawn from ``rng``, the training stream's own generator when none is given."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([WIDE["render_seed"], seed]))
        self.patches = np.zeros((tb.INPUTS, WIDE["inputs"]))
        for token in range(tb.INPUTS):
            self.patches[token, rng.choice(WIDE["inputs"], WIDE["patch"], replace=False)] = 1.0
        self.rng = np.random.default_rng(np.random.SeedSequence([WIDE["render_seed"], seed, 1]))

    def __call__(
        self, events: list[np.ndarray], rng: np.random.Generator | None = None
    ) -> list[np.ndarray]:
        rng = self.rng if rng is None else rng
        out = []
        for x in events:
            frame = np.minimum(np.asarray(x) @ self.patches, 1.0)
            noise = rng.random(frame.shape) < WIDE["noise"]
            out.append(np.maximum(frame, noise.astype(float)))
        return out


def panel_noise(render_seed: int, seed: int, name: str) -> np.random.Generator:
    """The noise of one reading panel: a generator of its own for the founder and the panel,
    drawn afresh at every reading, so every reading and every condition sees the same
    frames and no reading draws from the training stream's generator."""
    return np.random.default_rng(
        np.random.SeedSequence([render_seed, seed, 2, FRAME_PANELS.index(name)])
    )


def frame_read(brain: Any, seq: list[np.ndarray], condition: str, n: int) -> np.ndarray:
    """One condition of a rendered panel on a fork of the life: the events, the lesion, the
    query's act."""
    twin = tb.fork(brain)
    twin.reset()
    w = tb.new_work()
    for x in seq[:-1]:
        tb.act(twin, x, w)
    erase(twin, condition)
    out = tb.act(twin, seq[-1], w)
    return np.full(n, -1) if out is None else out


def frame_panels(brain: Any, seed: int, held: np.ndarray, render: Any,
                 render_seed: int) -> dict[str, Any]:
    """Every camera-width panel rendered once with its own noise, then each condition read
    on the same frames."""
    panels: dict[str, Any] = {}
    for name, (trials, neutral) in tb.panels(seed, held).items():
        if name not in FRAME_PANELS:
            continue
        truth = tb.answers(trials)
        seq = render(tb.events(trials, neutral), panel_noise(render_seed, seed, name))
        panels[name] = {c: tb.accuracy(frame_read(brain, seq, c, len(truth)), truth)
                        for c in ("intact", "memory_erased", "trace_erased")}
    return panels


def wide_life(spec: dict[str, Any]) -> dict[str, Any]:
    """Token binding with every event rendered at camera width; the memory addresses its sparse
    code."""
    seed = spec["seed"]
    held = tb.holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([tb.SCHEDULE_SEED, seed]))
    render = Renderer(seed)
    brain = ir.IndexBrain.build(seed, read=spec["genes"], inputs=WIDE["inputs"],
                                learning_surprise=True)
    work: dict[str, Any] = tb.new_work()
    readings, done, started = [], 0, time.monotonic()
    for budget in spec["budgets"]:
        while done < budget:
            trials = tb.draw(rng, tb.BATCH, held)
            brain.reset()
            episode(brain, render(tb.events(trials)), tb.answers(trials), work)
            done += 1
        panels = frame_panels(brain, seed, held, render, int(WIDE["render_seed"]))
        held_row = panels["held_out"]
        gate = (held_row["intact"] >= tb.GATE["held_out"]
                and held_row["memory_erased"] <= 1.0 / tb.VALUES + tb.GATE["erased_margin"])
        readings.append({"rounds": budget, "seconds": time.monotonic() - started,
                         "panels": panels, "gate": bool(gate)})
    return {"readings": readings, "work": work}


# -- token binding through a camera and a retina


class Camera:
    """Each token a fixed ``block`` x ``block`` square of a ``height`` x ``width`` image, on
    distinct cells of a grid of such squares, plus ``body`` inputs; an event lights its tokens'
    blocks and, each frame anew, ``noise`` of all the inputs."""

    def __init__(self, seed: int) -> None:
        c = CAMERA
        rng = np.random.default_rng(np.random.SeedSequence([c["render_seed"], seed]))
        self.pixels = c["height"] * c["width"]
        self.inputs = self.pixels + c["body"]
        cells = [(r, col) for r in range(c["height"] // c["block"])
                 for col in range(c["width"] // c["block"])]
        if len(cells) < tb.INPUTS:
            raise ValueError("the image holds fewer cells than there are tokens")
        chosen = rng.choice(len(cells), tb.INPUTS, replace=False)
        self.patches = np.zeros((tb.INPUTS, self.inputs))
        for token, cell in enumerate(chosen):
            r, col = cells[cell]
            rows = np.arange(r * c["block"], (r + 1) * c["block"])
            cols = np.arange(col * c["block"], (col + 1) * c["block"])
            self.patches[token, (rows[:, None] * c["width"] + cols[None, :]).ravel()] = 1.0
        self.rng = np.random.default_rng(np.random.SeedSequence([c["render_seed"], seed, 1]))

    def __call__(
        self, events: list[np.ndarray], rng: np.random.Generator | None = None
    ) -> list[np.ndarray]:
        """The frames of ``events``, the noise drawn from ``rng`` or, without one, from the
        training stream's own generator."""
        rng = self.rng if rng is None else rng
        out = []
        for x in events:
            frame = np.minimum(np.asarray(x) @ self.patches, 1.0)
            noise = rng.random(frame.shape) < CAMERA["noise"]
            out.append(np.maximum(frame, noise.astype(float)))
        return out


class Retina:
    """Local pooling units over the image: one unit per ``window`` x ``window`` square at
    ``stride``, its drive the mean of its pixels; the units at or above ``threshold`` stay,
    at most ``winners`` of them, the code scaled to unit length; the body inputs pass through.
    A fixed first stage between the pixels and the brain: the cortex and the memory read the
    same code."""

    def __init__(self) -> None:
        c = CAMERA
        xs = range(0, c["width"] - c["window"] + 1, c["stride"])
        ys = range(0, c["height"] - c["window"] + 1, c["stride"])
        self.index = np.array([
            [(y + dy) * c["width"] + (x + dx) for dy in range(c["window"])
             for dx in range(c["window"])]
            for y in ys for x in xs
        ], dtype=np.int64)
        self.pixels = c["height"] * c["width"]
        self.units = len(self.index) + c["body"]

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        frames = np.atleast_2d(np.asarray(frames, dtype=float))
        drive = frames[:, self.index].mean(axis=2)
        drive = np.where(drive >= CAMERA["threshold"], drive, 0.0)
        if CAMERA["winners"] < drive.shape[1]:
            cut = np.partition(drive, -CAMERA["winners"], axis=1)[:, -CAMERA["winners"]]
            drive = np.where(drive >= cut[:, None], drive, 0.0)
        norm = np.linalg.norm(drive, axis=1, keepdims=True)
        code = drive / np.maximum(norm, 1e-12)
        return np.concatenate([code, frames[:, self.pixels:]], axis=1)


class GenericCode:
    """A fixed sparse code of the whole frame with no spatial assumption (``index_read``'s
    ``SparseCode``, the library's covering code: each pixel read through an ON and an OFF
    channel, every channel by two of the ``units`` code neurons at least, each reading
    ``fan_in`` channels or more, ``winners`` kept, unit length), the body inputs passed
    through: the generic stage in place of the retina, for the comparison."""

    def __init__(self, inputs: int, pixels: int, units: int, fan_in: int, winners: int,
                 seed: int) -> None:
        self.code = ir.SparseCode(pixels, units, fan_in, winners, seed)
        self.pixels, self.units = pixels, units + (inputs - pixels)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        frames = np.atleast_2d(np.asarray(frames, dtype=float))
        return np.concatenate([self.code(frames[:, : self.pixels]), frames[:, self.pixels:]],
                              axis=1)


def camera_life(spec: dict[str, Any]) -> dict[str, Any]:
    """Token binding seen through the camera and the retina: the brain's observation is the
    retinal code, and the memory keys on it (no further code). With ``spec["generic"]`` (a
    winners count) the generic sparse code replaces the retina."""
    seed = spec["seed"]
    held = tb.holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([tb.SCHEDULE_SEED, seed]))
    camera = Camera(seed)
    retina: Any = Retina()
    if spec.get("generic"):
        retina = GenericCode(camera.inputs, camera.pixels, 1024, 32, int(spec["generic"]), seed)

    def see(
        events: list[np.ndarray], noise: np.random.Generator | None = None
    ) -> list[np.ndarray]:
        return [retina(frame) for frame in camera(events, noise)]

    brain = ir.IndexBrain.build(seed, read=spec["genes"], inputs=retina.units,
                                learning_surprise=True)
    work: dict[str, Any] = tb.new_work()
    readings, done, started = [], 0, time.monotonic()
    for budget in spec["budgets"]:
        while done < budget:
            trials = tb.draw(rng, tb.BATCH, held)
            brain.reset()
            episode(brain, see(tb.events(trials)), tb.answers(trials), work)
            done += 1
        panels = frame_panels(brain, seed, held, see, int(CAMERA["render_seed"]))
        held_row = panels["held_out"]
        gate = (held_row["intact"] >= tb.GATE["held_out"]
                and held_row["memory_erased"] <= 1.0 / tb.VALUES + tb.GATE["erased_margin"])
        readings.append({"rounds": budget, "seconds": time.monotonic() - started,
                         "panels": panels, "gate": bool(gate), "units": retina.units,
                         "inputs": camera.inputs})
    return {"readings": readings, "work": work}


LIVES = {"token": token_life, "fresh": fresh_life, "named": named_life,
         "succession": succession_life,
         "chain": chain_life, "continuing": continuing_life, "wide": wide_life,
         "camera": camera_life}


def life(spec: dict[str, Any]) -> dict[str, Any]:
    warnings.filterwarnings("ignore", "finite teaching free phase", RuntimeWarning)
    began = time.monotonic()
    out = LIVES[spec["task"]](spec)
    return {**spec, **out, "elapsed": time.monotonic() - began}


def summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per task, arm and budget: the founders at the gate and the held-out readings."""
    cells: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        for reading in row["readings"]:
            cells.setdefault((row["task"], row["arm"], reading["rounds"]), []).append(
                {"seed": row["seed"], "gate": reading["gate"], "held": held_of(row["task"],
                                                                               reading)})
    return [{"task": t, "arm": a, "rounds": r, "founders": len(v),
             "at_gate": sum(c["gate"] for c in v), "held": [c["held"] for c in v],
             "seeds": [c["seed"] for c in v]}
            for (t, a, r), v in sorted(cells.items())]


def held_of(task: str, reading: dict[str, Any]) -> float:
    if task in ("fresh", "named"):
        return float(reading["scores"]["intact"]["held_out"]["exact"])
    if task == "chain":
        return float(reading["panels"]["two_held"]["intact"])
    return float(reading["panels"]["held_out"]["intact"])


def digests() -> dict[str, str]:
    files = [Path(__file__), HERE / "index_read.py", HERE / "token_binding.py",
             HERE / "sequence_binding.py", HERE / "chain_binding.py",
             HERE / "named_bindings.py", HERE.parent / "relevance" / "bindings.py",
             HERE.parent / "relevance" / "selective_recall.py", HERE / "INDEX_PROTOCOL.md"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--budgets", nargs="*", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--generic", type=int, default=None,
                        help="camera task: the generic sparse code with this many winners")
    parser.add_argument("--genes", type=json.loads, default={},
                        help="genes over the arm's, for sizing; declared runs leave it empty")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("out exists; results are never overwritten")
    budgets = sorted(args.budgets or BUDGETS[args.task])
    specs = [{"task": args.task, "arm": arm, "seed": seed, "budgets": budgets,
              "genes": genes_for(args.task, arm, args.genes),
              **({"generic": args.generic} if args.generic is not None else {})}
             for arm in args.arms for seed in args.seeds]
    began = time.monotonic()
    with get_context("spawn").Pool(max(1, min(args.workers, len(specs)))) as pool:
        rows = pool.map(life, specs, chunksize=1)
    body = {"schema": "index-runs/1", "task": args.task, "genes_over": args.genes,
            "digests": digests(), "rows": rows,
            "summary": summary(rows), "elapsed_seconds": time.monotonic() - began}
    args.out.mkdir(parents=True)
    (args.out / "receipt.json").write_text(json.dumps(body, indent=1, default=float) + "\n")
    for cell in body["summary"]:
        print(f"{cell['task']} {cell['arm']} {cell['rounds']}: {cell['at_gate']} of "
              f"{cell['founders']} at the gate; held " + " ".join(f"{h:.2f}" for h in cell["held"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
