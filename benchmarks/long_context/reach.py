"""Long context (cadence#268): how far the read reaches, and where a failure sits.

One life per spec, each in a process of its own. The brain is ``Brain.compose(inputs, actions,
modules=(64,), seed=founder, **ARMS[arm], recall_capacity=C, recall_position=P)``. Every moment is a
greedy act; a taught answer moment has its lesson before the act (the battery's order). The answer
moments are counted, so a life's work stays comparable across distances: a spec names ``answers``
(answer batches taught), and the life runs whole episodes until it has them.

Every frozen read also reads, at each answer moment and for head 0, the attention and its terms
(read-only, before the act):

- the right slots: the slot of the answer's own moment (age ``a``, delivered on the recall port) and
  the slot before it (age ``a + 1``, delivered on the next port); on a periodic stream every period
  back as well;
- mass and first choice on each, how far the first choice lies from them, the effective slots (the
  exponential of the attention's entropy), the mass on the right token through each port and each
  port's named token;
- each term's share of the margin between the better right slot and the best other slot, and
  between it and the slots beside it: content, table, anchor, relative, slope, grid.

    PYTHONPATH=src python benchmarks/long_context/reach.py --plan plan.json --out rows.jsonl \\
        --workers 6
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = _module("sequence_battery", Path(__file__).resolve().parent.parent / "sequence" / "battery.py")

FD: dict[str, Any] = {
    "episodic": False, "teaching": "deep", "recall": True, "recall_recency": 0.0,
    "learning_surprise": True, "recall_boundary": True, "recall_anchor": True,
    "recall_slope": 0.1, "recall_position_rate": 0.01, "recall_relative": True,
}
GRID = (7, 8, 9, 11)  # 35 terms, ages apart by less than 5,544 differ in a residue
ARMS: dict[str, dict[str, Any]] = {
    "Fd": FD,  # the sequence composition's rule at the deep teaching point, no apical
    "FdT": {**FD, "working_memory_amplitude": 0.0},  # with the working trace silent
    "FdL": {**FD, "recall_grid": GRID},  # with the position code past the table
    "FdLT": {**FD, "recall_grid": GRID, "working_memory_amplitude": 0.0},
}
TERMS = ("content", "table", "anchor", "relative", "slope", "grid")


class DenseNBack(B.Task):
    """A continuing stream of tokens from ``vocab``; after ``warm`` moments every ``every``-th
    moment is a cue ``k``, whose answer is the token ``k`` moments back. Every ``k`` is odd and
    ``every`` is 4, so an answer is never a cue's moment. ``lessons`` cue moments per episode."""

    def __init__(self, vocab: int = 8, ks=(1, 3, 5, 7), every: int = 4, lessons: int = 100,
                 test_lessons: int | None = None):
        if every % 2 or any(k % 2 == 0 for k in ks):
            raise ValueError("cues fall on one residue of an even ``every``, and every k is odd")
        super().__init__("nback_dense", vocab + len(ks), vocab,
                         dict(vocab=vocab, ks=list(ks), every=every, lessons=lessons))
        self.vocab, self.ks, self.every = vocab, tuple(ks), every
        self.warm = max(ks) + 1
        self.lessons, self.test_lessons = lessons, test_lessons or lessons

    def episode(self, rng, train):
        length = self.warm + self.every * (self.lessons if train else self.test_lessons)
        tokens = rng.integers(self.vocab, size=(B.BATCH, length))
        rows = np.arange(B.BATCH)
        moments = []
        for t in range(length):
            if t >= self.warm and t % self.every == self.every - 1:
                index = rng.integers(len(self.ks), size=B.BATCH)
                ks = np.array(self.ks)[index]
                moments.append(B.Moment(B.onehot(self.vocab + index, self.inputs),
                                        tokens[rows, t - ks], ks - 1,
                                        [f"k{k}" for k in ks]))
            else:
                moments.append(B.Moment(B.onehot(tokens[:, t], self.inputs)))
        return moments


class ReachBack(B.NBack):
    """The battery's n-back, each answer's condition named by its ``k``."""


class ReachLookBack(B.LookBack):
    """The battery's look back; at a frozen read each condition carries its length, and with
    ``same`` every story holds one token (no distractor: the answer needs no distant moment)."""

    def __init__(self, vocab=4, lengths=(5, 12), test_lengths=None, cues=B.LookBack.CUES,
                 same: bool = False):
        super().__init__(vocab, lengths, test_lengths, cues)
        self.same = same

    def episode(self, rng, train):
        moments = super().episode(rng, train)
        answer = moments[-1]
        length = len(moments) - 1
        if self.same:
            token = rng.integers(self.vocab, size=B.BATCH)
            for m in moments[:-1]:
                m.x = B.onehot(token, self.inputs)
            answer.label = token
        answer.condition = [f"{'same-' if self.same else ''}{c}@{length}"
                            for c in answer.condition]
        return moments


class DenseLookBack(B.Task):
    """One story of ``length`` tokens from ``vocab``; every ``every``-th moment a cue, ``first``,
    ``second`` or ``last``, answered by the token at that place: lessons at every length up to
    ``length``, from one stream. A condition carries the length's bin."""

    def __init__(self, vocab=4, length=100, every=4, cues=("first", "last", "second")):
        super().__init__("lookback_dense", vocab + 3, vocab,
                         dict(vocab=vocab, length=length, every=every, cues=list(cues)))
        self.vocab, self.length, self.every, self.cues = vocab, length, every, tuple(cues)

    def episode(self, rng, train):
        tokens = rng.integers(self.vocab, size=(B.BATCH, self.length))
        rows = np.arange(B.BATCH)
        moments = []
        for t in range(self.length):
            if t % self.every == self.every - 1:
                c = rng.integers(len(self.cues), size=B.BATCH)
                names = [self.cues[i] for i in c]
                where = {"first": 0, "second": 1, "last": t - 1}
                pos = np.array([where[n] for n in names])
                ages = t - 1 - pos
                bin_ = int(2 ** np.ceil(np.log2(max(t, 2))))
                moments.append(B.Moment(B.onehot(self.vocab + c, self.inputs), tokens[rows, pos],
                                        ages, [f"{n}@{bin_}" for n in names]))
            else:
                moments.append(B.Moment(B.onehot(tokens[:, t], self.inputs)))
        return moments


class ReachPeriodic(B.Periodic):
    """The battery's periodic continuation, each answer's condition carrying its period index:
    the second period reads from an unvaried period (ceiling 1.0), the later ones from a varied
    one (a read of the slot one period back reaches .917; three periods mean .944)."""

    def episode(self, rng, train):
        moments = super().episode(rng, train)
        for t, m in enumerate(moments):
            if m.label is not None:
                p = int(m.condition[0][1:])
                m.condition = [f"P{p}:r{t // p}"] * B.BATCH
                m.period = p
        return moments


TASKS: dict[str, Any] = {
    "nback_dense": DenseNBack, "nback": ReachBack, "lookback": ReachLookBack,
    "lookback_dense": DenseLookBack, "periodic": ReachPeriodic,
}


def compose(task: Any, spec: dict[str, Any]) -> Any:
    import cadence as cd

    options = {**ARMS[spec["arm"]], **(spec.get("settings") or {})}
    if spec.get("capacity"):
        options["recall_capacity"] = int(spec["capacity"])
    if spec.get("position") is not None:
        options["recall_position"] = int(spec["position"])
    return cd.Brain.compose(task.inputs, task.actions, modules=(int(spec.get("module", 64)),),
                            seed=int(spec["founder"]), **options)


def replicate(brain: Any, replica: int) -> None:
    """A replica: the newborn's efficacies changed at the last bits, tied pairs alike."""
    learner = brain.learner
    rng = np.random.default_rng(np.random.SeedSequence([replica, 260]))
    noise = rng.uniform(-1.0, 1.0, learner.brain.connectome.synapses)
    index = np.flatnonzero(learner.reverse >= 0)
    noise[index] = noise[np.minimum(index, learner.reverse[index])]
    learner.brain = learner.brain.with_parameters(
        efficacy=learner.brain.efficacy * (1.0 + 1e-13 * noise))


# ----------------------------------------------------------------------------------- reading


def terms(recall: Any, x: np.ndarray) -> dict[str, np.ndarray] | None:
    """Head 0's logit terms at every slot for the moment about to be read, ``(rows, slots)``,
    read-only; None before there is a slot."""
    if not recall.slots or recall.streams != len(x):
        return None
    from cadence.stream import _term

    moment = recall._moment(x)
    inputs, cortex = moment.inputs, moment.cortex
    ages = recall.ages()
    out = {
        "content": np.einsum("bk,bmk->bm", moment.address, recall.keys) / recall.temperature,
        "table": _term(recall.position_maps[0], inputs, cortex)[:, recall.buckets()],
    }
    zero = np.zeros_like(out["table"])
    out["anchor"] = (np.take_along_axis(_term(recall.anchor_maps[0], inputs, cortex),
                                        recall.anchored_buckets(), axis=1)
                     if recall.anchor_maps else zero)
    out["relative"] = (np.take_along_axis(_term(recall.relative_maps[0], inputs, cortex),
                                          recall.relative_buckets(), axis=1)
                       if recall.relative_maps else zero)
    out["slope"] = (_term(recall.slope_maps[0], inputs, cortex)[:, None]
                    * (ages / recall.SLOPE)[None, :] if recall.slope_maps else zero)
    out["grid"] = zero
    if recall.grid_maps:
        grid = _term(recall.grid_maps[0], inputs, cortex)[:, recall.grid_buckets()].sum(axis=1)
        out["grid"] = grid * (ages >= recall.position)
    return out


NUMBERS = ("n", "hit", "read_value", "read_next", "held", "holds", "mass_value", "mass_next",
           "first_value", "first_next", "within_1", "within_2", "within_8", "within_32",
           "first_oldest", "first_newest", "effective", "top", "token_value", "token_next",
           "margin", "near", *(f"margin_{t}" for t in TERMS), *(f"near_{t}" for t in TERMS))


def new_cell() -> dict[str, float]:
    return dict.fromkeys(NUMBERS, 0.0)


def score_moment(cells: dict[str, dict[str, float]], m: Any, action: np.ndarray,
                 split: dict[str, np.ndarray] | None, p: np.ndarray | None, values: np.ndarray,
                 nexts: np.ndarray, vocab: int) -> None:
    """Add one answer moment to the cells of its conditions. ``values`` and ``nexts`` are the
    slots as the read used them, before the moment was written."""
    rows = np.arange(len(action))
    conditions = m.condition or ["all"] * len(rows)
    hit = action == m.label
    for r, condition in enumerate(conditions):
        cell = cells.setdefault(condition, new_cell())
        cell["n"] += 1
        cell["hit"] += float(hit[r])
    if p is None:
        return
    slots = p.shape[1]
    read_value = np.einsum("bm,bmc->bc", p, values)[:, :vocab]
    read_next = np.einsum("bm,bmc->bc", p, nexts)[:, :vocab]
    entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0), axis=1)
    first = np.argmax(p, axis=1)
    for r, condition in enumerate(conditions):
        cell = cells[condition]
        label = int(m.label[r])
        cell["read_value"] += float(np.argmax(read_value[r]) == label)
        cell["read_next"] += float(np.argmax(read_next[r]) == label)
        cell["token_value"] += float(read_value[r, label])
        cell["token_next"] += float(read_next[r, label])
        cell["effective"] += float(np.exp(entropy[r]))
        cell["top"] += float(p[r, first[r]])
        cell["first_oldest"] += float(first[r] == 0)
        cell["first_newest"] += float(first[r] == slots - 1)
        if m.age is None:
            continue
        value_slot, next_slot = slots - 1 - int(m.age[r]), slots - 2 - int(m.age[r])
        if value_slot >= 0:
            cell["held"] += 1.0
            cell["holds"] += float(values[r, value_slot, label] > 0)
            cell["mass_value"] += float(p[r, value_slot])
            cell["first_value"] += float(first[r] == value_slot)
        if next_slot >= 0:
            cell["mass_next"] += float(p[r, next_slot])
            cell["first_next"] += float(first[r] == next_slot)
        if value_slot < 0:  # the answer's slot is gone: nothing below can name it
            continue
        distance = min(abs(int(first[r]) - value_slot),
                       abs(int(first[r]) - next_slot) if next_slot >= 0 else slots)
        for width in (1, 2, 8, 32):
            cell[f"within_{width}"] += float(distance <= width)
        if split is None:
            continue
        total = sum(split.values())[r]
        right = [value_slot] + ([next_slot] if next_slot >= 0 else [])
        period = getattr(m, "period", 0)
        if period:  # every period back holds the motif as well, varied or not
            right = [i for i in range(slots)
                     if (slots - 1 - i - int(m.age[r])) % period in (0, 1)
                     and slots - 1 - i >= int(m.age[r])]
        best = right[int(np.argmax(total[right]))]
        others = total.copy()
        others[right] = -np.inf
        if slots > len(right) and np.isfinite(others).any():
            wrong = int(np.argmax(others))
            cell["margin"] += float(total[best] - total[wrong])
            for name, term in split.items():
                cell[f"margin_{name}"] += float(term[r, best] - term[r, wrong])
        # the slots beside the right ones: does the code separate neighbouring ages?
        neighbours = [i for i in (next_slot - 1, value_slot + 1) if 0 <= i < slots]
        if neighbours:
            near = neighbours[int(np.argmax(total[neighbours]))]
            cell["near"] += float(total[best] - total[near])
            for name, term in split.items():
                cell[f"near_{name}"] += float(term[r, best] - term[r, near])


def summarize_cells(cells: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for condition, cell in sorted(cells.items()):
        n = max(cell["n"], 1.0)
        out[condition] = {k: round(v / (n if k != "n" else 1.0), 4) for k, v in cell.items()}
    return out


def frozen_read(brain: Any, task: Any, rng: np.random.Generator, episodes: int,
                check: dict[str, float]) -> tuple[dict[str, Any], dict[str, float]]:
    """A frozen read of ``episodes`` on a copy of the brain, with its work."""
    brain = copy.deepcopy(brain)
    recall = brain.recall
    vocab = int(getattr(task, "vocab", task.actions))
    cells: dict[str, dict[str, float]] = {}
    work = {"moments": 0, "answers": 0, "act_sweeps": 0.0, "exhausted": 0, "seconds": 0.0}
    started = time.monotonic()
    for _ in range(episodes):
        brain.reset()
        for m in task.episode(rng, train=False):
            split = None
            if m.label is not None:
                split = terms(recall, m.x)
                values, nexts = recall.values, recall.nexts
            try:
                action = np.asarray(brain.act(m.x, greedy=True)).reshape(len(m.x), -1)[:, 0]
                work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
            except RuntimeError:
                work["exhausted"] += 1
                action = np.full(len(m.x), -1)
            work["moments"] += 1
            if m.label is None:
                continue
            work["answers"] += 1
            heads = recall.attended_heads
            p = heads[0] if heads else None
            if p is not None and split is not None and check["checked"] < 32:
                total = sum(split.values())
                q = np.exp(total - total.max(axis=1, keepdims=True))
                q /= q.sum(axis=1, keepdims=True)
                check["worst"] = max(check["worst"], float(np.abs(q - p).max()))
                check["checked"] += 1
            score_moment(cells, m, action, split, p, values, nexts, vocab)
    work["seconds"] = round(time.monotonic() - started, 3)
    return summarize_cells(cells), work


def map_norms(brain: Any) -> dict[str, float]:
    recall = brain.recall
    out = {}
    for name in ("head_maps", "position_maps", "anchor_maps", "relative_maps", "slope_maps"):
        maps = getattr(recall, name)
        if maps:
            out[name] = round(float(np.sqrt(sum(float(np.sum(a * a)) for a in maps[0]))), 4)
    return out


# -------------------------------------------------------------------------------------- life


def build_task(name: str, args: dict[str, Any]) -> Any:
    return TASKS[name](**args)


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    warnings.simplefilter("ignore")
    import cadence as cd

    founder = int(spec["founder"])
    task = build_task(spec["task"], spec.get("task_args", {}))
    brain = compose(task, spec)
    if spec.get("replica"):
        replicate(brain, int(spec["replica"]))
    recall = brain.recall
    rng = np.random.default_rng(np.random.SeedSequence([founder, B.SEEDS["train"]]))
    answers = int(spec.get("answers", 1500))
    points = sorted({max(1, int(answers * f)) for f in spec.get("checkpoints", [])} | {answers})
    bins = int(spec.get("bins", 8))
    size = max(1, answers // bins)
    log = [{"answers": 0, "attempted": 0.0, "agreed": 0.0, "position_rows": 0.0,
            "address_error": 0.0, "lean": 0.0, "lean_max": 0.0, "row_sweeps": 0.0}
           for _ in range(bins + 1)]
    work = {"moments": 0, "answers": 0, "episodes": 0, "act_sweeps": 0.0, "lesson_row_sweeps": 0.0,
            "refused": 0, "exhausted": 0, "acts": 0}
    row: dict[str, Any] = {
        "tag": spec.get("tag"), "task": spec["task"], "task_settings": task.settings,
        "arm": spec["arm"], "founder": founder, "replica": int(spec.get("replica", 0)),
        "capacity": recall.capacity, "position": recall.position,
        "buckets": recall.position_buckets, "module": int(spec.get("module", 64)),
        "settings": spec.get("settings") or {}, "answers": answers,
        "parameters": int(brain.parameters()), "cadence": cd.__version__, "batch": B.BATCH,
        "reads": [],
    }
    check = {"checked": 0, "worst": 0.0}

    def read_all(at: int) -> None:
        for index, reading in enumerate(spec.get("reads", [])):
            if at != answers and not reading.get("always"):
                continue
            read_task = build_task(reading["task"], reading.get("task_args", {}))
            test = np.random.default_rng(np.random.SeedSequence([founder, B.SEEDS["test"], index]))
            cells, read_work = frozen_read(brain, read_task, test, int(reading.get("episodes", 1)),
                                           check)
            entry = {"at": at, "name": reading.get("name", reading["task"]),
                     "task_settings": read_task.settings, "conditions": cells, "work": read_work}
            row["reads"].append(entry)
            if spec.get("partial"):  # a long life's checkpoints, as they come
                with open(spec["partial"], "a") as sink:
                    sink.write(json.dumps({"tag": spec.get("tag"), "seconds":
                                           round(time.monotonic() - started, 1), **entry}) + "\n")

    started = time.monotonic()
    next_point = 0
    done = False
    while not done:
        brain.reset()
        work["episodes"] += 1
        for m in task.episode(rng, train=True):
            if m.label is not None and work["answers"] < answers:
                cell = log[min(work["answers"] // size, bins)]
                if recall.slope_maps and recall.slots:
                    from cadence.stream import _term

                    moment = recall._moment(m.x)
                    lean = _term(recall.slope_maps[0], moment.inputs, moment.cortex)
                    cell["lean"] += float(np.abs(lean).mean())
                    cell["lean_max"] = max(cell["lean_max"], float(np.abs(lean).max()))
                try:
                    report = brain.lesson(m.x, m.label)
                    cell["attempted"] += float(report.get("attempted_presentations", 0.0))
                    cell["agreed"] += float(report.get("agreed_presentations", 0.0))
                    cell["position_rows"] += float(report.get("position_rows", 0.0))
                    cell["address_error"] += float(report.get("address_error", 0.0))
                    cell["row_sweeps"] += float(report.get("total_row_sweeps", 0.0))
                    work["lesson_row_sweeps"] += float(report.get("total_row_sweeps", 0.0))
                except RuntimeError:
                    work["refused"] += 1
                cell["answers"] += 1
                work["answers"] += 1
            try:
                brain.act(m.x, greedy=True)
                work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
            except RuntimeError:
                work["exhausted"] += 1
            work["acts"] += 1
            work["moments"] += 1
            if next_point < len(points) and work["answers"] >= points[next_point]:
                # the episode's remaining moments are not needed once the last answer is taught
                at = points[next_point]
                next_point += 1
                if at == answers:
                    done = True
                    break
                train_seconds = time.monotonic() - started
                read_all(at)
                started = time.monotonic() - train_seconds
    row["train_seconds"] = round(time.monotonic() - started, 2)
    row["work"] = work
    row["log"] = [c for c in log if c["answers"]]
    row["maps"] = map_norms(brain)
    read_all(answers)
    row["check"] = check
    row["seconds"] = round(time.monotonic() - started, 2)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    specs = json.loads(args.plan.read_text())
    done = set()
    if args.out.exists():
        done = {json.loads(line).get("tag") for line in args.out.read_text().splitlines() if line}
    specs = [{**s, "partial": str(args.out.with_suffix(".partial.jsonl"))} for s in specs
             if s.get("tag") not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(specs)} lives, {len(done)} already read", flush=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for row in map(run_life, specs):
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                print(row["tag"], row["seconds"], flush=True)
        else:
            with get_context("spawn").Pool(min(args.workers, max(len(specs), 1)),
                                           maxtasksperchild=1) as pool:
                for row in pool.imap_unordered(run_life, specs, chunksize=1):
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                    print(row["tag"], row["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
