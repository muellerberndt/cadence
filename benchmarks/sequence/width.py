"""The width sweep (cadence#239): does a wider module learn as a narrow one does?

One life: a task, an arm, a module width and a founder. The brain is taught for ``lessons``
batches drawn in turn from a fixed pool of training episodes (``SeedSequence([founder,
239])``, ``POOL`` batches), so the error on the pool and on fresh episodes (``SeedSequence(
[founder, 932])``, ``READS`` batches) can be told apart, and is read at checkpoints:

- accuracy on the pool's first ``READS`` batches (train) and on the fresh ones (held-out);
- the share of motor and association units at a bound of the activation (within 2% of -0.1 or
  1.0, where the slope vanishes and no learning signal passes) at the fresh answer moments;
- the share of synapses at the efficacy cap, by projection;
- work: settle sweeps per act, acts refused (counted as misses), answers rescued.

Tasks (the battery's, ``battery.py``): ``induction`` over a vocabulary of 128, ``associative``
recall of 64 keys and 16 values, and ``lookup``, one noisy view of one of 256 fixed prototypes
labelled with its class. The brain is the battery's composition at the module width, with the
recall memory and surprise lessons where ``memory`` is set. Arms: ``founder``, and ``relative``,
every synapse's step scaled to its neuron's fan-in in the same brain composed at width 64
(``fan_in_rates`` below, ``Learner.synapse_rate``). ``settings`` add options, such as
``{"learning_surprise_share": true}``.

    python benchmarks/sequence/width.py --task induction --memory --arms relative \\
        --widths 1024 --settings '{"learning_surprise_share": true}' --out results/width.jsonl
    python benchmarks/sequence/width.py --summary results/width.jsonl
"""

from __future__ import annotations

import argparse
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
POOL, READS, LESSONS = 128, 24, 1500
CHECKPOINTS = (250, 500, 1000)
ARMS = ("founder", "relative")
NOISE = 0.3


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = _module("sequence_battery", HERE / "battery.py")


class Lookup(B.Task):
    """A capacity task without memory: one moment, a noisy view of one of ``items`` fixed
    prototypes, labelled with the prototype's class."""

    def __init__(self, items: int = 256, dim: int = 64, classes: int = 16, seed: int = 99):
        super().__init__("lookup", dim, classes, dict(items=items, dim=dim, classes=classes))
        rng = np.random.default_rng(seed)
        self.prototypes = rng.normal(size=(items, dim))
        self.labels = rng.integers(classes, size=items)
        self.items = items

    def episode(self, rng, train):
        kind = rng.integers(self.items, size=B.BATCH)
        x = self.prototypes[kind] + NOISE * rng.normal(size=(B.BATCH, self.inputs))
        return [B.Moment(x, self.labels[kind], None, ["all"] * B.BATCH)]


TASKS = {
    "lookup": lambda: Lookup(),
    "induction": lambda: B.Induction(vocab=128, lengths=(8, 16)),
    "associative": lambda: B.Associative(keys=64, values=16, pairs=(4,)),
}


def compose(task: Any, width: int, founder: int, memory: bool,
            settings: dict[str, Any] | None = None) -> Any:
    import cadence as cd

    options = {"recall": True, "recall_founder": True, "learning_surprise": True} if memory else {}
    return cd.Brain.compose(task.inputs, task.actions, modules=(width,), seed=int(founder),
                            episodic=False, teaching="normalized",
                            **{**options, **(settings or {})})


def fan_in_rates(wide: Any, narrow: Any) -> np.ndarray:
    """Each synapse's step scaled to the largest fan-in of its neuron's population in
    ``narrow``, the same composition at a smaller width; capped at 1."""
    c, n = wide.connectome, narrow.connectome
    fan = np.bincount(c.post, minlength=c.n).astype(float)
    fan_narrow = np.bincount(n.post, minlength=n.n).astype(float)
    reference = fan.copy()
    for name, neurons in c.populations.items():
        index = np.asarray(neurons, dtype=np.int64)
        if name in n.populations and len(index):
            top = fan_narrow[np.asarray(n.populations[name], dtype=np.int64)].max(initial=0.0)
            if top > 0:
                reference[index] = np.minimum(reference[index], top)
    return np.minimum(1.0, reference[c.post] / np.maximum(fan[c.post], 1.0))


def run(brain: Any, episodes: list, teach: bool, work: dict[str, Any],
        record: list | None = None) -> float:
    """Lessons (``teach``) or a frozen reading of ``episodes``; a reading returns its accuracy
    and, with ``record``, keeps every answer moment's activation."""
    hits = total = 0
    for moments in episodes:
        brain.reset()
        for m in moments:
            if teach and m.label is not None:
                try:
                    brain.lesson(m.x, m.label)
                except RuntimeError:
                    work["refused_lessons"] += 1
            try:
                action = np.asarray(brain.act(m.x, greedy=True)).reshape(len(m.x), -1)[:, 0]
            except RuntimeError as exc:
                if "did not settle" not in str(exc):
                    raise
                work["refused_acts"] += 1
                if not teach and m.label is not None:
                    total += len(m.x)
                continue
            settled = brain.last_settlement
            if settled:
                work["sweeps"].append(int(settled["steps"]))
                work["rescued"] += int("rescue_steps" in settled)
            if not teach and m.label is not None:
                hits += int(np.sum(action == m.label))
                total += len(action)
                if record is not None:
                    record.append(np.asarray(brain.basal_ganglia.state.activation).copy())
    return hits / max(total, 1)


def at_cap(brain: Any) -> dict[str, float]:
    c, cap = brain.connectome, brain.learner.config.scale_cap
    efficacy = np.abs(np.asarray(brain.brain.efficacy))
    pops = brain.connectome.populations
    out = {}
    for pre, post in (("sensory", "association"), ("association", "motor"),
                      ("motor", "association"), ("prefrontal", "association"),
                      ("recall", "association"), ("next", "association")):
        if pre not in pops or post not in pops:
            continue
        chosen = np.isin(c.pre, pops[pre]) & np.isin(c.post, pops[post])
        if chosen.any():
            out[f"{pre}->{post}"] = float(np.mean(efficacy[chosen] >= 0.99 * cap))
    return out


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``spec`` names ``task``, ``arm``, ``width`` and ``founder``, with ``memory``,
    ``settings``, ``lessons`` and ``checkpoints``."""
    warnings.simplefilter("ignore")
    import cadence as cd

    task = TASKS[spec["task"]]()
    arm, width, founder = spec["arm"], int(spec["width"]), int(spec["founder"])
    memory, settings = bool(spec.get("memory")), spec.get("settings") or {}
    lessons = int(spec.get("lessons", LESSONS))
    checkpoints = set(spec.get("checkpoints", CHECKPOINTS)) | {lessons}
    brain = compose(task, width, founder, memory, settings)
    if arm == "relative":
        brain.learner.synapse_rate = fan_in_rates(brain, compose(task, 64, founder, memory,
                                                                 settings))
    elif arm != "founder":
        raise ValueError(f"unknown arm {arm!r}")
    names = {name: np.asarray(index) for name, index in brain.connectome.populations.items()}
    rng = np.random.default_rng(np.random.SeedSequence([founder, 239]))
    pool = [task.episode(rng, train=True) for _ in range(POOL)]
    fresh_rng = np.random.default_rng(np.random.SeedSequence([founder, 932]))
    fresh = [task.episode(fresh_rng, train=False) for _ in range(READS)]
    work: dict[str, Any] = {"refused_acts": 0, "refused_lessons": 0, "rescued": 0, "sweeps": []}
    curve, started = [], time.monotonic()
    for lesson in range(1, lessons + 1):
        run(brain, [pool[(lesson - 1) % len(pool)]], teach=True, work=work)
        if lesson in checkpoints:
            states: list = []
            point = {"lessons": lesson,
                     "train": run(brain, pool[:READS], teach=False, work=work),
                     "test": run(brain, fresh, teach=False, work=work, record=states)}
            if states:
                a = np.concatenate(states)
                for name in ("association", "motor"):
                    x = a[:, names[name]]
                    point[f"{name}_at_bound"] = float(np.mean((x > 0.98) | (x < -0.098)))
            point.update(at_cap=at_cap(brain), refused_acts=work["refused_acts"],
                         refused_lessons=work["refused_lessons"], rescued=work["rescued"],
                         sweeps_per_act=float(np.mean(work["sweeps"][-200:])),
                         seconds=round(time.monotonic() - started, 1))
            curve.append(point)
    return {"task": spec["task"], "task_settings": task.settings, "arm": arm, "width": width,
            "founder": founder, "memory": memory, "settings": settings, "lessons": lessons,
            "pool_batches": POOL, "batch": B.BATCH, "neurons": int(brain.connectome.n),
            "synapses": int(brain.connectome.synapses), "chance": task.chance(),
            "cadence": cd.__version__, "curve": curve,
            "seconds": round(time.monotonic() - started, 1)}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per task, arm, width, memory and settings: the founders' held-out accuracy at every
    checkpoint (least and greatest), the last train accuracy, refusals and rescues."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = json.dumps([r["task"], r["arm"], r["width"], r["memory"], r["settings"]],
                         sort_keys=True)
        cells.setdefault(key, []).append(r)
    out = []
    for key, group in sorted(cells.items()):
        task, arm, width, memory, settings = json.loads(key)
        points = min(len(r["curve"]) for r in group)
        out.append({
            "task": task, "arm": arm, "width": width, "memory": memory, "settings": settings,
            "founders": sorted(r["founder"] for r in group),
            "lessons": [group[0]["curve"][i]["lessons"] for i in range(points)],
            "held_out": [[round(min(r["curve"][i]["test"] for r in group), 3),
                          round(max(r["curve"][i]["test"] for r in group), 3)]
                         for i in range(points)],
            "train": [round(min(r["curve"][-1]["train"] for r in group), 3),
                      round(max(r["curve"][-1]["train"] for r in group), 3)],
            "refused_acts": sum(r["curve"][-1]["refused_acts"] for r in group),
            "rescued": sum(r["curve"][-1]["rescued"] for r in group),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--memory", action="store_true", help="the recall memory on")
    parser.add_argument("--arms", nargs="*", default=["founder"], choices=ARMS)
    parser.add_argument("--widths", nargs="*", type=int, default=[64])
    parser.add_argument("--founders", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--lessons", type=int, default=LESSONS)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=list(CHECKPOINTS))
    parser.add_argument("--settings", type=json.loads, default={})
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs in place of the grid")
    parser.add_argument("--out", type=Path, help="JSON lines, one row per life, appended")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary", type=Path, help="print the summary of a results file")
    args = parser.parse_args(argv)
    if args.summary is not None:
        rows = [json.loads(line) for line in args.summary.read_text().splitlines() if line]
        for cell in summarize(rows):
            print(json.dumps(cell))
        return 0
    if args.out is None:
        parser.error("--out is required")
    if args.plan is not None:
        specs = json.loads(args.plan.read_text())
    else:
        if args.task is None:
            parser.error("--task or --plan is required")
        specs = [
            {"task": args.task, "arm": arm, "width": width, "founder": founder,
             "memory": args.memory, "settings": args.settings, "lessons": args.lessons,
             "checkpoints": args.checkpoints}
            for arm in args.arms for width in args.widths for founder in args.founders
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for row in map(run_life, specs):
                sink.write(json.dumps(row) + "\n")
                sink.flush()
        else:
            with get_context("spawn").Pool(min(args.workers, len(specs))) as pool:
                for row in pool.imap_unordered(run_life, specs, chunksize=1):
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
