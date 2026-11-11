"""The scaling evaluation of the recall memory; declared in ``SCALING_PROTOCOL.md``.

Does the index memory's recall help the composed brain reach mastery, and does mastery come
with enough training, a wider brain and a larger memory? Five task families, each with a
difficulty knob, run on the lives of ``index_runs`` through their parameters:

- ``token``: token binding with ``facts`` per trial and a ``neutral`` delay before the query;
- ``succession``: what followed the cued token, in sequences of ``length`` tokens;
- ``named``: fresh bindings of ``bits`` bits whose query shares nothing with its fact;
- ``continuing``: token binding without a reset, read after ``earlier`` trials of the life;
- ``camera``: token binding seen at 12,824 inputs through the generic code of ``units``.

The factors: the memory (``none``, the composed brain whose working trace is the only carrier
of the facts; ``recall``, the index memory of the ``horizon`` arm; ``recall_linear``, the
``index`` arm's linear charge, amendment 1's supplement), the association width and the
memory's capacity. One life per founder, level and setting, read at 256, 1024, 4096 and 16384
rounds on forks; a life ends once two consecutive readings are at mastery. ``--grid
supplement`` lists amendment 1's lives, run after the declared grid into the same output.

    PYTHONPATH=src python benchmarks/binding/scaling.py plan --grid trimmed --out DIR/jobs.json
    PYTHONPATH=src python benchmarks/binding/scaling.py run --jobs DIR/jobs.json --out DIR \\
        --workers 2
    PYTHONPATH=src python benchmarks/binding/scaling.py summary --out DIR

``probe`` measures a setting's processor seconds (the brain's build, a training round, a
reading) at a founder's untrained brain, and ``plan --costs`` extrapolates them to a grid.
Nothing here changes the library.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
from functools import partial
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


runs = _sibling("index_runs")
ir, tb = runs.ir, runs.tb

SCHEMA = "index-scaling/1"
BUDGETS = (256, 1024, 4096, 16384)
SEEDS = (0, 1, 2)
WIDTHS = (64, 256, 1024)
CAPACITIES = (16, 64, 256)
CAPACITY = 64  # where the task's span fits in it
MASTERY = 0.95  # held-out score
HOLD = 2  # consecutive readings at mastery end a life
# each memory setting's arm of the index protocol: the recall's power-law charge (recency None
# at the capacity) and, amendment 1, the linear charge (recency 0.25 per moment of age)
MEMORIES: dict[str, str | None] = {"none": None, "recall": "horizon", "recall_linear": "index"}
COMPARISONS = (("recall", "none"), ("recall_linear", "none"), ("recall_linear", "recall"))
TOKEN_NAMES = 9  # eight training names and the new one: eight facts need eight names
SUCCESSION_TOKENS = 17  # sixteen training tokens and the new one
FACTS, DELAYS = (2, 4, 8), (1, 8, 32)
LENGTHS, BITS, EARLIER, UNITS = (4, 8, 16), (3, 4, 5), (2, 8, 32), (256, 1024, 4096)
PANEL = 128  # trials of a reading panel
BOX = 2.0  # processor seconds on the box per processor second measured here

FAMILIES: dict[str, dict[str, Any]] = {
    "token": {"task": "token", "fixed": {"names": TOKEN_NAMES},
              "levels": [{"facts": f, "neutral": d} for f in FACTS for d in DELAYS]},
    "succession": {"task": "succession", "fixed": {"tokens": SUCCESSION_TOKENS},
                   "levels": [{"length": n} for n in LENGTHS]},
    "named": {"task": "named", "fixed": {}, "levels": [{"bits": b} for b in BITS]},
    "continuing": {"task": "continuing", "fixed": {},
                   "levels": [{"earlier": k} for k in EARLIER]},
    "camera": {"task": "camera", "fixed": {},
               "levels": [{"units": u, "generic": round(u / 10)} for u in UNITS]},
}


# -- the grid


def level_key(family: str, level: dict[str, Any]) -> str:
    if family == "token":
        return f"f{level['facts']}d{level['neutral']}"
    if family == "succession":
        return f"l{level['length']}"
    if family == "named":
        return f"b{level['bits']}"
    if family == "continuing":
        return f"k{level['earlier']}"
    return f"u{level['units']}"


def span(family: str, level: dict[str, Any]) -> int:
    """The moments a query reads back over: its trial's events before it, and in the
    continuing life the earlier trials' (four events each) too."""
    if family == "token":
        return int(level["facts"] + level["neutral"])
    if family == "succession":
        return int(level["length"])
    if family == "continuing":
        return 4 * int(level["earlier"]) + 3
    return 3


def events(family: str, level: dict[str, Any]) -> int:
    """Events of one training trial, the query included."""
    if family == "token":
        return int(level["facts"] + level["neutral"] + 1)
    if family == "succession":
        return int(level["length"] + 1)
    return 4


def capacities(family: str, level: dict[str, Any]) -> tuple[int, ...]:
    """The capacity factor where the task's span needs it (the longest delay, the continuing
    life); elsewhere the capacity holds the span."""
    if family == "continuing" or (family == "token" and level["neutral"] == max(DELAYS)):
        return CAPACITIES
    return (CAPACITY,)


def job(family: str, level: dict[str, Any], memory: str, width: int, capacity: int | None,
        seed: int, budgets: tuple[int, ...] = BUDGETS) -> dict[str, Any]:
    """One life: the spec ``index_runs.life`` reads, plus the grid's own coordinates."""
    task = FAMILIES[family]["task"]
    if memory not in MEMORIES:
        raise ValueError(f"memory is one of {tuple(MEMORIES)}")
    if (memory == "none") != (capacity is None):
        raise ValueError("a capacity belongs to a memory alone")
    key, arm = level_key(family, level), MEMORIES[memory]
    genes = None if arm is None else runs.genes_for(task, arm, {"capacity": int(capacity)})
    name = (f"{family}-{key}-{memory}-w{width}" + ("" if capacity is None else f"-c{capacity}")
            + f"-s{seed}")
    return {"name": name, "family": family, "key": key, "task": task, "memory": memory,
            "arm": arm or "none", "width": int(width),
            "capacity": capacity, "span": span(family, level), "seed": int(seed),
            "budgets": sorted(budgets), "level": True, "master": MASTERY, "hold": HOLD,
            "genes": genes, **FAMILIES[family]["fixed"], **level}


def levels() -> list[tuple[str, dict[str, Any]]]:
    return [(family, level) for family, spec in FAMILIES.items() for level in spec["levels"]]


def settings(family: str, level: dict[str, Any]) -> list[tuple[str, int | None]]:
    return [("none", None)] + [("recall", c) for c in capacities(family, level)]


def full_grid() -> list[dict[str, Any]]:
    """Every family, level, width and setting at every budget, founders 0 to 2."""
    return [job(f, lv, m, w, c, s) for f, lv in levels() for w in WIDTHS
            for m, c in settings(f, lv) for s in SEEDS]


# the trimmed grid: (family, level key) -> {width: ((budgets, capacities), ...)}: the recall
# reads those budgets at each listed capacity (``None``: the level's own capacities), and the
# composed brain without the memory reads the longest budgets listed at that width
B4, B3, B2 = BUDGETS, BUDGETS[:3], BUDGETS[:2]
Plan = dict[int, tuple[tuple[tuple[int, ...], tuple[int, ...] | None], ...]]
LONG = ((B4, (64, 256)), (B3, (16,)))  # the longest delay: capacity 16 cannot hold its facts
TRIMMED: dict[tuple[str, str], Plan] = {
    ("token", "f2d1"): {64: ((B4, None),), 256: ((B4, None),), 1024: ((B4, None),)},
    ("token", "f4d1"): {64: ((B4, None),), 256: ((B4, None),)},
    ("token", "f8d1"): {64: ((B4, None),), 256: ((B4, None),)},
    ("token", "f2d8"): {64: ((B4, None),), 256: ((B4, None),)},
    ("token", "f4d8"): {64: ((B4, None),), 256: ((B4, None),)},
    ("token", "f8d8"): {64: ((B4, None),), 256: ((B4, None),), 1024: ((B3, None),)},
    ("token", "f2d32"): {64: LONG, 256: ((B4, (64,)),)},
    ("token", "f4d32"): {64: LONG, 256: ((B4, (64,)),)},
    ("token", "f8d32"): {64: LONG, 256: ((B4, (64,)),), 1024: ((B2, (64,)),)},
    ("succession", "l4"): {64: ((B4, None),), 256: ((B4, None),)},
    ("succession", "l8"): {64: ((B4, None),), 256: ((B4, None),)},
    ("succession", "l16"): {64: ((B4, None),), 256: ((B4, None),), 1024: ((B3, None),)},
    ("named", "b3"): {64: ((B4, None),), 256: ((B4, None),)},
    ("named", "b4"): {64: ((B4, None),), 256: ((B4, None),)},
    ("named", "b5"): {64: ((B4, None),), 256: ((B4, None),), 1024: ((B4, None),)},
    ("continuing", "k2"): {64: ((B4, None),), 256: ((B4, (16,)),)},
    ("continuing", "k8"): {64: ((B4, None),), 256: ((B4, (64,)),)},
    ("continuing", "k32"): {64: ((B4, None),), 256: ((B4, (256,)),), 1024: ((B3, (256,)),)},
    ("camera", "u256"): {64: ((B4, None),), 256: ((B4, None),)},
    ("camera", "u1024"): {64: ((B4, None),), 256: ((B4, None),)},
    ("camera", "u4096"): {64: ((B3, None),), 256: ((B2, None),)},
}


def trimmed_grid() -> list[dict[str, Any]]:
    """The grid proposed for one box: ``TRIMMED``'s widths, budgets and capacities."""
    out = []
    for family, level in levels():
        for width, plan in TRIMMED.get((family, level_key(family, level)), {}).items():
            longest = max((budgets for budgets, _ in plan), key=len)
            for seed in SEEDS:
                out.append(job(family, level, "none", width, None, seed, longest))
                for budgets, caps in plan:
                    chosen = capacities(family, level) if caps is None else caps
                    out += [job(family, level, "recall", width, c, seed, budgets)
                            for c in chosen]
    return out


def declared_budgets(family: str, level: dict[str, Any], width: int,
                     capacity: int) -> tuple[int, ...]:
    """The budgets the declared grid reads for the recall at this level, width and capacity
    (its longest at that width where it does not read the capacity)."""
    plan = TRIMMED[(family, level_key(family, level))][width]
    for budgets, caps in plan:
        if capacity in (capacities(family, level) if caps is None else caps):
            return budgets
    return max((budgets for budgets, _ in plan), key=len)


def linear_capacity(family: str, level: dict[str, Any]) -> int:
    """Amendment 1's capacity: 64 where every trial starts from empty slots (its span fits, no
    slot is dropped, and the linear charge does not read the capacity); in the continuing life
    the smallest capacity of the factor that holds the panel's history."""
    if family == "continuing":
        return min(c for c in CAPACITIES if c >= span(family, level))
    return CAPACITY


def supplement_grid() -> list[dict[str, Any]]:
    """Amendment 1's lives: the linear charge (``recall_linear``) on the camera family at every
    width and budget the declared grid reads, and at width 64 on the token levels of D 8 and 32
    and the continuing levels of K 8 and 32, at the declared grid's budgets."""
    out = []
    for family, level in levels():
        key = level_key(family, level)
        if family == "camera":
            widths = list(TRIMMED[(family, key)])
        elif ((family == "token" and level["neutral"] in (8, 32))
              or (family == "continuing" and level["earlier"] in (8, 32))):
            widths = [64]
        else:
            continue
        capacity = linear_capacity(family, level)
        for width in widths:
            budgets = declared_budgets(family, level, width, capacity)
            out += [job(family, level, "recall_linear", width, capacity, seed, budgets)
                    for seed in SEEDS]
    return out


def smoke_grid() -> list[dict[str, Any]]:
    """Tiny budgets, end to end: every family's first level with and without the memory, its
    last level with the memory; one life with mastery at zero, so its stop rule and the
    summary's carried readings run."""
    out = []
    for family, spec in FAMILIES.items():
        first, last = spec["levels"][0], spec["levels"][-1]
        out.append(job(family, first, "none", 64, None, 0, (2, 4)))
        out.append(job(family, first, "recall", 64, CAPACITY, 0, (2, 4)))
        out.append(job(family, last, "recall", 64, max(capacities(family, last)), 0, (2, 4)))
    out.append(job("camera", FAMILIES["camera"]["levels"][0], "recall_linear", 64, CAPACITY, 0,
                   (2, 4)))
    stop = job("token", FAMILIES["token"]["levels"][0], "recall", 64, CAPACITY, 0, (2, 4, 8))
    out.append(dict(stop, name=stop["name"] + "-master0", key=stop["key"] + "-master0",
                    master=0.0))
    return out


GRIDS = {"full": full_grid, "trimmed": trimmed_grid, "supplement": supplement_grid,
         "smoke": smoke_grid}


# -- costs


def _cpu() -> float:
    return time.process_time()


def probe(spec: dict[str, Any], rounds: int = 2) -> dict[str, Any]:
    """Processor seconds of a life's parts at its setting, on a founder's untrained brain:
    the build, one training round (the life's own draws and episode, the mean of ``rounds``
    after one warm-up round) and one reading (a condition's read of the held-out panel on a
    fork, times the conditions the life reads; the fresh tasks' whole reading). An act
    settles for a bounded number of sweeps, so a trained brain's acts cost the same."""
    family, seed = spec["family"], spec["seed"]
    rng = np.random.default_rng(np.random.SeedSequence([7, seed]))
    work = tb.new_work()
    began = _cpu()
    if family == "named":
        runs._ENCODING["task"] = "named"
        try:
            bd, bits = runs._bindings(), int(spec["bits"])
            held = bd.holdout(seed, bits)
            brain = runs.fresh_brain(seed, spec["genes"], bits, int(spec["width"]))
            build = _cpu() - began
            batches = bd.trials(seed, bits, rounds + 1, held)
            runs.fresh_round(brain, work, next(batches), bits)
            began = _cpu()
            for rows in batches:
                runs.fresh_round(brain, work, rows, bits)
            per_round = (_cpu() - began) / rounds
            began = _cpu()
            runs.fresh_reading(brain, bits, held)
            reading = _cpu() - began
        finally:
            runs._ENCODING["task"] = "fresh"
        return {"build": build, "per_round": per_round, "reading": reading}
    if family == "succession":
        sb = runs._sibling("sequence_binding")
        tokens, length = int(spec["tokens"]), int(spec["length"])
        held = sb.holdout(seed, tokens)
        brain = runs.built(spec, inputs=sb.input_width(tokens), actions=tokens,
                           lateral=sb.LATERAL)
        build = _cpu() - began

        def train() -> None:
            trials = sb.draw(rng, sb.BATCH, held, length=length)
            brain.reset()
            runs.episode(brain, sb.events(trials, tokens), sb.answers(trials), work)

        panel = sb.events(sb.draw(rng, PANEL, held, length=length, cue_held=True), tokens)
    elif family == "camera":
        camera = runs.Camera(seed)
        code = runs.GenericCode(camera.inputs, camera.pixels, int(spec["units"]), 32,
                                int(spec["generic"]), seed)

        def see(evs: list[np.ndarray]) -> list[np.ndarray]:
            return [code(frame) for frame in camera(evs)]

        held = tb.holdout(seed)
        brain = runs.built(spec, inputs=code.units)
        build = _cpu() - began

        def train() -> None:
            trials = tb.draw(rng, tb.BATCH, held)
            brain.reset()
            runs.episode(brain, see(tb.events(trials)), tb.answers(trials), work)

        panel = see(tb.events(tb.draw(rng, PANEL, held, queried_held=True)))
    else:
        names = int(spec.get("names", tb.NAMES))
        facts, neutral = int(spec.get("facts", tb.FACTS)), int(spec.get("neutral", tb.NEUTRAL))
        held = tb.holdout(seed, names)
        brain = runs.built(spec, inputs=tb.input_width(names))
        build = _cpu() - began
        continuing = family == "continuing"
        if continuing:
            brain.reset()

        def train() -> None:
            trials = tb.draw(rng, tb.BATCH, held, facts=facts)
            if not continuing:
                brain.reset()
            runs.episode(brain, tb.events(trials, neutral, names), tb.answers(trials), work)

        panel = []
        for _ in range(int(spec.get("earlier", 0)) if continuing else 0):
            panel += tb.events(tb.draw(rng, PANEL, held))
        panel += tb.events(tb.draw(rng, PANEL, held, facts=facts, queried_held=True), neutral,
                           names)
    train()
    began = _cpu()
    for _ in range(rounds):
        train()
    per_round = (_cpu() - began) / rounds
    began = _cpu()
    runs.frame_read(brain, panel, "intact", PANEL)
    condition = _cpu() - began
    reads = 1 + len(runs.level_conditions(brain, "held_out"))  # the training panel intact
    return {"build": build, "per_round": per_round, "reading": reads * condition,
            "condition": condition, "reads": reads}


def probe_key(spec: dict[str, Any]) -> str:
    """A setting's cost class: the capacity bounds a handful of slots and leaves it, and the
    two charges cost the same."""
    memory = "none" if spec["memory"] == "none" else "recall"
    return f"{spec['family']}-{spec['key']}-{memory}-w{spec['width']}"


def probe_job(spec: dict[str, Any], rounds: int) -> dict[str, Any]:
    import warnings

    warnings.filterwarnings("ignore", "finite teaching free phase", RuntimeWarning)
    began = time.monotonic()
    out = probe(spec, rounds)
    return {"key": probe_key(spec), "family": spec["family"], "level": spec["key"],
            "memory": spec["memory"], "width": spec["width"], "rounds": rounds, **out,
            "wall": time.monotonic() - began}


def estimate(spec: dict[str, Any], costs: dict[str, dict[str, Any]]) -> float | None:
    """Processor seconds here for a life read at every budget (no stop at mastery): the build,
    the last budget's rounds and one reading per budget."""
    cost = costs.get(probe_key(spec))
    if cost is None:
        return None
    return float(cost["build"] + max(spec["budgets"]) * cost["per_round"]
                 + len(spec["budgets"]) * cost["reading"])


def weight(spec: dict[str, Any]) -> float:
    """A life's estimated seconds, or without a probe a proxy that orders lives by cost alone:
    rounds, events per trial, the width and the memory, the code's inputs at camera width."""
    if spec.get("estimate"):
        return float(spec["estimate"])
    inputs = int(spec.get("units", 0)) / 256
    per_event = (spec["width"] / 64) ** 0.8 * (1.0 if spec["memory"] == "none" else 1.6)
    return max(spec["budgets"]) * (events(spec["family"], spec) + 6) * per_event * (1 + inputs)


def tally(jobs: list[dict[str, Any]], costs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Lives, processor hours here and core-hours on the box, per family and in all; the
    longest life's hours on the box bound the run's wall time from below."""
    rows: dict[str, dict[str, Any]] = {}
    for spec in jobs:
        seconds = estimate(spec, costs)
        row = rows.setdefault(spec["family"], {"lives": 0, "estimated": 0, "hours_here": 0.0,
                                               "longest_box_hours": 0.0})
        row["lives"] += 1
        if seconds is not None:
            row["estimated"] += 1
            row["hours_here"] += seconds / 3600
            row["longest_box_hours"] = max(row["longest_box_hours"], BOX * seconds / 3600)
    total = {"lives": sum(r["lives"] for r in rows.values()),
             "estimated": sum(r["estimated"] for r in rows.values()),
             "hours_here": sum(r["hours_here"] for r in rows.values()),
             "longest_box_hours": max((r["longest_box_hours"] for r in rows.values()),
                                      default=0.0)}
    for row in [*rows.values(), total]:
        row["box_core_hours"] = BOX * row["hours_here"]
    return {"families": rows, "total": total}


# -- running


def versions() -> dict[str, str]:
    import cadence

    return {"cadence": cadence.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "machine": platform.machine()}


def digests() -> dict[str, str]:
    files = [Path(__file__), HERE / "SCALING_PROTOCOL.md", HERE / "index_runs.py",
             HERE / "index_read.py", HERE / "token_binding.py", HERE / "sequence_binding.py",
             HERE / "named_bindings.py", HERE.parent / "relevance" / "bindings.py"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def run_job(spec: dict[str, Any], out: str) -> str:
    """One life; its receipt is written to a temporary name and renamed, so a stopped run
    leaves finished receipts only. A life that raises leaves its trace under ``errors``."""
    lives, began = Path(out) / "lives", time.monotonic()
    try:
        row = runs.life(dict(spec))
    except Exception:  # noqa: BLE001 - recorded, and the run goes on
        errors = Path(out) / "errors"
        errors.mkdir(exist_ok=True)
        (errors / f"{spec['name']}.txt").write_text(traceback.format_exc())
        return f"  {spec['name']}: failed after {time.monotonic() - began:.0f}s"
    receipt = {"schema": SCHEMA, "job": spec, "readings": row["readings"],
               "work": row["work"], "training": row.get("training"),
               "elapsed": row["elapsed"], "cpu": row["cpu"], "versions": versions(),
               "digests": digests()}
    path = lives / f"{spec['name']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=1, default=float) + "\n")
    temporary.replace(path)
    (Path(out) / "errors" / f"{spec['name']}.txt").unlink(missing_ok=True)  # an earlier failure
    held = " ".join(f"{r['rounds']}: {held_out(spec, r):.2f}" for r in row["readings"])
    return f"  {spec['name']}: {held} ({row['cpu']:.0f} cpu s)"


def run(jobs: list[dict[str, Any]], out: Path, workers: int) -> int:
    lives = out / "lives"
    lives.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in lives.glob("*.json")}
    todo = [j for j in jobs if j["name"] not in done]
    todo.sort(key=lambda j: -weight(j))  # the longest lives first
    print(f"{len(todo)} lives to go, {len(jobs) - len(todo)} done", flush=True)
    if todo:
        context = get_context("spawn")
        with context.Pool(max(1, min(workers, len(todo))), maxtasksperchild=1) as pool:
            for line in pool.imap_unordered(partial(run_job, out=str(out)), todo):
                print(line, flush=True)
    return len(list((out / "errors").glob("*.txt"))) if (out / "errors").exists() else 0


# -- readings and the summary


def held_out(spec: dict[str, Any], reading: dict[str, Any]) -> float:
    """The task's held-out measure: the held-out panel's intact accuracy, or for the named
    bindings the exact match on every held-out pair, order and query."""
    return float(runs.held_of(spec["task"], reading))


def conditions(spec: dict[str, Any], reading: dict[str, Any]) -> dict[str, float | None]:
    """The held-out measure intact, with the slots erased, with the trace erased (and for the
    named bindings with both erased), and the training panel's intact score."""
    if spec["task"] == "named":
        s = reading["scores"]
        return {"intact": s["intact"]["held_out"]["exact"],
                "memory_erased": (None if spec["memory"] == "none"
                                  else s["memory_erased"]["held_out"]["exact"]),
                "trace_erased": s["trace_erased"]["held_out"]["exact"],
                "both_erased": s["erased"]["held_out"]["exact"],
                "train": s["intact"]["training"]["exact"]}
    held = reading["panels"]["held_out"]
    return {"intact": held["intact"], "memory_erased": held.get("memory_erased"),
            "trace_erased": held["trace_erased"], "both_erased": None,
            "train": reading["panels"]["train"]["intact"]}


def at_budget(receipt: dict[str, Any], budget: int) -> tuple[dict[str, Any] | None, bool]:
    """The reading at ``budget``, or the last one carried when the life ended at mastery
    before it; None when the life did not reach it otherwise."""
    readings = receipt["readings"]
    for reading in readings:
        if reading["rounds"] == budget:
            return reading, False
    spec = receipt["job"]
    if readings and readings[-1]["rounds"] < budget and runs.ended(spec, readings):
        return readings[-1], True
    return None, False


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def cell_summary(receipts: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """One setting over its founders: per budget the mean held-out score (carried readings
    counted and marked), the founders at mastery, the erased conditions and the training
    score; the first budget every founder reads at mastery; each founder's first."""
    spec = next(iter(receipts.values()))["job"]
    budgets = sorted({b for r in receipts.values() for b in r["job"]["budgets"]})
    master = float(spec["master"])
    rows = []
    for budget in budgets:
        per: dict[int, dict[str, Any]] = {}
        for seed, receipt in sorted(receipts.items()):
            reading, carried = at_budget(receipt, budget)
            if reading is not None:
                per[seed] = {**conditions(receipt["job"], reading), "carried": carried}
        rows.append({
            "rounds": budget, "founders": len(per),
            "carried": sum(v["carried"] for v in per.values()),
            "held": _mean([v["intact"] for v in per.values()]),
            "held_by_founder": {s: v["intact"] for s, v in per.items()},
            "at_mastery": sum(v["intact"] >= master for v in per.values()),
            **{c: _mean([v[c] for v in per.values()])
               for c in ("memory_erased", "trace_erased", "both_erased", "train")},
        })
    first = {}
    for seed, receipt in sorted(receipts.items()):
        first[seed] = next((r["rounds"] for r in receipt["readings"]
                            if held_out(receipt["job"], r) >= master), None)
    everyone = next((row["rounds"] for row in rows if row["founders"] == len(receipts)
                     and row["at_mastery"] == len(receipts)), None)
    return {"budgets": rows, "first_mastery": everyone, "first_by_founder": first,
            "founders": sorted(receipts)}


def summarize(out: Path) -> dict[str, Any]:
    receipts = [json.loads(p.read_text()) for p in sorted((out / "lives").glob("*.json"))]
    groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    for receipt in receipts:
        j = receipt["job"]
        groups.setdefault((j["family"], j["key"], j["memory"], j["width"], j["capacity"]),
                          {})[j["seed"]] = receipt
    order = {(f, level_key(f, lv)): i for i, (f, lv) in enumerate(levels())}

    def rank(k: tuple[Any, ...]) -> tuple[Any, ...]:  # a level's memories side by side
        return (order.get((k[0], k[1]), len(order)), k[3], list(MEMORIES).index(k[2]), k[4] or 0)

    cells = [{"family": k[0], "level": k[1], "memory": k[2], "width": k[3], "capacity": k[4],
              **cell_summary(v)} for k, v in sorted(groups.items(), key=lambda kv: rank(kv[0]))]
    paired = []
    for k in sorted(groups, key=rank):
        for first, second in COMPARISONS:
            other = (k[0], k[1], second, k[3], None if second == "none" else k[4])
            if k[2] != first or other not in groups:
                continue
            ours, theirs = groups[k], groups[other]
            budgets = sorted({b for r in ours.values() for b in r["job"]["budgets"]}
                             & {b for r in theirs.values() for b in r["job"]["budgets"]})
            rows = []
            for budget in budgets:
                diffs = {}
                for seed in sorted(set(ours) & set(theirs)):
                    a, _ = at_budget(ours[seed], budget)
                    b, _ = at_budget(theirs[seed], budget)
                    if a is not None and b is not None:
                        diffs[seed] = (held_out(ours[seed]["job"], a)
                                       - held_out(theirs[seed]["job"], b))
                rows.append({"rounds": budget, "difference": _mean(list(diffs.values())),
                             "by_founder": diffs})
            paired.append({"family": k[0], "level": k[1], "width": k[3], "capacity": k[4],
                           "comparison": f"{first} less {second}", "budgets": rows})
    side = []
    for k in sorted(groups, key=rank):  # the linear setting beside the others at its capacity
        if k[2] != "recall_linear":
            continue
        trio = [(k[0], k[1], "none", k[3], None), (k[0], k[1], "recall", k[3], k[4]), k]
        found = {m[2]: next((c for c in cells if (c["family"], c["level"], c["memory"],
                                                  c["width"], c["capacity"]) == m), None)
                 for m in trio}
        side.append({"family": k[0], "level": k[1], "width": k[3], "capacity": k[4],
                     "cells": {m: None if c is None else {
                         "held": {r["rounds"]: r["held"] for r in c["budgets"]},
                         "carried": {r["rounds"]: r["carried"] for r in c["budgets"]},
                         "first_mastery": c["first_mastery"]} for m, c in found.items()}})
    mastery = []
    for family, level in levels():
        key = level_key(family, level)
        entry: dict[str, Any] = {"family": family, "level": key}
        for memory in MEMORIES:
            found = [c for c in cells if c["family"] == family and c["level"] == key
                     and c["memory"] == memory]
            mastered = [c for c in found if c["first_mastery"] is not None]
            best = min(mastered, key=lambda c: (c["width"], c["capacity"] or 0,
                                                c["first_mastery"]), default=None)
            entry[memory] = {"settings": len(found), "mastered": len(mastered),
                             "first": None if best is None else {
                                 "width": best["width"], "capacity": best["capacity"],
                                 "rounds": best["first_mastery"]}}
        if any(entry[memory]["settings"] for memory in MEMORIES):
            mastery.append(entry)
    missing = sorted(p.stem for p in (out / "errors").glob("*.txt")) \
        if (out / "errors").exists() else []
    return {"schema": SCHEMA + "/summary", "lives": len(receipts), "mastery_score": MASTERY,
            "cells": cells, "paired": paired, "side_by_side": side, "mastery": mastery,
            "failed": missing,
            "versions": versions(), "digests": digests()}


def _f(value: float | None, carried: bool = False) -> str:
    return "" if value is None else f"{value:.2f}" + ("*" if carried else "")


def markdown(body: dict[str, Any]) -> str:
    """The tables: per family each setting's mean held-out score at every budget (founders at
    mastery in brackets, * where a reading is carried), its first budget at mastery, the
    conditions at its last budget; the paired differences; mastery by level."""
    out = [f"# Scaling readings: {body['lives']} lives", "",
           f"Mastery: held-out at least {body['mastery_score']}. A starred mean counts readings "
           "carried from a life that ended at mastery. Erased and training scores are at the "
           "last budget read.", ""]
    for family in FAMILIES:
        cells = [c for c in body["cells"] if c["family"] == family]
        if not cells:
            continue
        budgets = sorted({r["rounds"] for c in cells for r in c["budgets"]})
        out += [f"## {family}", "",
                "| level | memory | width | capacity | " + " | ".join(map(str, budgets))
                + " | first at mastery | slots erased | trace erased | both erased | training |",
                "| --- | --- | --- | --- | " + " | ".join("---" for _ in budgets)
                + " | --- | --- | --- | --- | --- |"]
        for c in cells:
            by = {r["rounds"]: r for r in c["budgets"]}
            row = [c["level"], c["memory"], str(c["width"]), str(c["capacity"] or "")]
            for b in budgets:
                r = by.get(b)
                row.append("" if r is None or r["held"] is None else
                           f"{_f(r['held'], bool(r['carried']))} ({r['at_mastery']}/"
                           f"{r['founders']})")
            last = next((r for r in reversed(c["budgets"]) if r["held"] is not None), None)
            row.append(str(c["first_mastery"] or "none"))
            row += ["" if last is None else _f(last[k])
                    for k in ("memory_erased", "trace_erased", "both_erased", "train")]
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    if body.get("side_by_side"):
        budgets = sorted({int(b) for s in body["side_by_side"] for c in s["cells"].values()
                          if c is not None for b in c["held"]})
        out += ["## The linear charge beside the power-law charge and no memory", "",
                "Each cell: none / recall / recall_linear, the recall at the linear setting's "
                "capacity; a dash where a setting was not read.", "",
                "| family | level | width | capacity | " + " | ".join(map(str, budgets))
                + " | first at mastery |",
                "| --- | --- | --- | --- | " + " | ".join("---" for _ in budgets) + " | --- |"]
        for s in body["side_by_side"]:
            row = [s["family"], s["level"], str(s["width"]), str(s["capacity"])]
            for b in budgets:
                parts = []
                for memory in MEMORIES:
                    c = s["cells"][memory]
                    held = None if c is None else c["held"].get(b, c["held"].get(str(b)))
                    carried = bool(c and c["carried"].get(b, c["carried"].get(str(b))))
                    parts.append(_f(held, carried) or "-")
                row.append(" / ".join(parts))
            row.append(" / ".join("-" if s["cells"][m] is None else
                                  str(s["cells"][m]["first_mastery"] or "none")
                                  for m in MEMORIES))
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    if body["paired"]:
        budgets = sorted({r["rounds"] for p in body["paired"] for r in p["budgets"]})
        out += ["## Paired differences by founder", "",
                "| comparison | family | level | width | capacity | "
                + " | ".join(map(str, budgets)) + " |",
                "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in budgets) + " |"]
        for p in body["paired"]:
            by = {r["rounds"]: r["difference"] for r in p["budgets"]}
            out.append(f"| {p.get('comparison', 'recall less none')} | {p['family']} | "
                       f"{p['level']} | {p['width']} | {p['capacity']} | "
                       + " | ".join(_f(by.get(b)) for b in budgets) + " |")
        out.append("")
    out += ["## Mastery by level", "",
            "| family | level | " + " | ".join(
                f"{m}: settings at mastery | {m}: first (width, capacity, rounds)"
                for m in ("recall", "recall_linear", "none")) + " |",
            "| --- | --- | " + " | ".join("--- | ---" for _ in MEMORIES) + " |"]
    for m in body["mastery"]:
        cells_ = []
        for memory in ("recall", "recall_linear", "none"):
            e = m[memory]
            first = e["first"]
            cells_ += [f"{e['mastered']} of {e['settings']}",
                       "none" if first is None else
                       f"{first['width']}, {first['capacity'] or ''}, {first['rounds']}"]
        out.append(f"| {m['family']} | {m['level']} | " + " | ".join(cells_) + " |")
    if body["failed"]:
        out += ["", "Failed lives (see `errors/`): " + ", ".join(body["failed"])]
    return "\n".join(out) + "\n"


# -- the command line


def _load_costs(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {row["key"]: row for row in json.loads(path.read_text())["probes"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="write a grid's job list and its estimated cost")
    p.add_argument("--grid", choices=list(GRIDS), default="trimmed")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--costs", type=Path, help="a probe's costs.json")
    p = sub.add_parser("probe", help="measure the cost classes of a grid at small settings")
    p.add_argument("--grid", choices=list(GRIDS), default="full")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("run", help="run a job list's lives not yet done, one receipt each")
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("summary", help="the tables over a run's receipts")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "plan":
        jobs = GRIDS[args.grid]()
        costs = _load_costs(args.costs)
        for spec in jobs:
            spec["estimate"] = estimate(spec, costs)
        totals = tally(jobs, costs)
        if args.out.exists():
            parser.error("out exists; job lists are never overwritten")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"schema": SCHEMA + "/jobs", "grid": args.grid,
                                        "costs": str(args.costs) if args.costs else None,
                                        "box_factor": BOX, "totals": totals, "jobs": jobs},
                                       indent=1) + "\n")
        for family, row in [*totals["families"].items(), ("all", totals["total"])]:
            print(f"{family}: {row['lives']} lives; estimated {row['estimated']}: "
                  f"{row['hours_here']:.1f} processor hours here, {row['box_core_hours']:.1f} "
                  f"core-hours on the box, longest life {row['longest_box_hours']:.2f} h there")
        return 0
    if args.command == "probe":
        if args.out.exists():
            parser.error("out exists; probes are never overwritten")
        classes: dict[str, dict[str, Any]] = {}
        for spec in GRIDS[args.grid]():  # a class's own capacity where the grid holds it
            if spec["capacity"] in (None, CAPACITY) or probe_key(spec) not in classes:
                classes[probe_key(spec)] = dict(spec, seed=0)
        todo = list(classes.values())
        todo.sort(key=lambda s: -s["width"] * events(s["family"], s))
        print(f"{len(todo)} cost classes", flush=True)
        rows = []
        with get_context("spawn").Pool(max(1, args.workers), maxtasksperchild=4) as pool:
            for row in pool.imap_unordered(partial(probe_job, rounds=args.rounds), todo):
                rows.append(row)
                print(f"  {row['key']}: build {row['build']:.2f}s, round {row['per_round']:.3f}s,"
                      f" reading {row['reading']:.2f}s ({row['wall']:.0f}s wall)", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"schema": SCHEMA + "/costs", "grid": args.grid,
                                        "rounds": args.rounds, "versions": versions(),
                                        "probes": sorted(rows, key=lambda r: r["key"])},
                                       indent=1) + "\n")
        return 0
    if args.command == "run":
        body = json.loads(args.jobs.read_text())
        failed = run(body["jobs"], args.out, args.workers)
        return 1 if failed else 0
    body = summarize(args.out)
    (args.out / "summary.json").write_text(json.dumps(body, indent=1) + "\n")
    text = markdown(body)
    (args.out / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
