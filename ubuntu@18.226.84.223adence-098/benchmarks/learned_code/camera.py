"""The size reading: camera binding through a learned code of 64, 128 or 256 units against the
generic random code of 1,024 units.

The task is ``benchmarks/binding/index_runs.py``'s camera task: token binding (two facts
``is + name + value``, a neutral event, a query ``ask + name``; held-out name-value pairs)
with every event a frame of 12,824 inputs, each token a 20 x 20 block of a 100 x 128 image,
1% noise pixels on every frame. A code stage between the frame and the brain gives both the
cortex and the recall memory their observation; the brain is that instrument's index brain
with the ``index`` recall genes.

Stages:

* ``generic``: the instrument's generic sparse code, 1,024 code neurons, 96 winners, unit
  length; the life is ``index_runs.life`` itself;
* ``learned``: a ``CodeRegion`` of ``units`` units over the frame's pixels (the pixels at unit
  length), without its top-down boundary: in this task the moment before carries no
  structure about the next event. The code at unit length and the 24 body inputs, as the
  generic stage passes them, are the brain's observation. A nursery of ``NURSERY`` rounds of
  the task's frames (its own draws and noise, no lessons) runs the code alone; its first
  ``units`` frames, at unit length, are the dictionary's first columns (a dictionary drawn at
  random over 12,800 pixels drives each unit about a twelfth of its threshold, so it would not
  start); then the life trains on the generic arm's own rounds and frames, the code learning
  from every frame, and is read on copies with the code frozen.

Memory is the recall memory's address map (``w_x``, ``w_c``, ``b``) and the code's own tables
(the learned dictionary, predictor and thresholds, or the generic code's index and weight
tables).

    PYTHONPATH=src python benchmarks/learned_code/camera.py --out DIR --workers 2
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
BINDING = HERE.parent / "binding"


def _load(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runs = _load("index_runs", BINDING / "index_runs.py")
cr = _load("code_region", HERE / "code_region.py")
tb, ir = runs.tb, runs.ir

GENERIC = {"units": 1024, "winners": 96}  # the index protocol's generic code (fan-in 32)
UNITS = (64, 128, 256)
NURSERY = 256
BUDGETS = (256, 1024)
FOUNDERS = tuple(range(6))
NURSERY_SEED = 61301
CODE = cr.CodeSettings(top_down=0.0, threshold=0.1, step=0.1, budget=400, tolerance=1e-4,
                       dictionary_rate=2.0, predictor_rate=0.5, threshold_rate=0.01,
                       target=0.1)


class LearnedStage:
    """The code region over a frame's pixels at unit length, the body inputs passed through;
    it learns from every frame it codes while ``learning`` is on."""

    def __init__(self, camera: Any, units: int, seed: int) -> None:
        settings = cr.CodeSettings(**{**CODE.to_dict(), "units": int(units)})
        self.pixels = camera.pixels
        self.code = cr.CodeRegion(camera.pixels, tb.WIDTH, settings, seed=seed)
        self.units = int(units) + (camera.inputs - camera.pixels)

    def imprint(self, frames: np.ndarray) -> None:
        """The dictionary's columns set to frames, one per unit, at unit length."""
        x = cr.unit_rows(np.atleast_2d(np.asarray(frames, dtype=float))[:, : self.pixels])
        if len(x) != self.code.settings.units:
            raise ValueError("imprinting takes one frame per unit")
        self.code.phi = np.ascontiguousarray(x.T)

    def settled(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames = np.atleast_2d(np.asarray(frames, dtype=float))
        x = cr.unit_rows(frames[:, : self.pixels])
        zeros = np.zeros((len(x), self.code.settings.units))
        a, _ = self.code.settle(x, zeros)
        self.code.learn(x, a, zeros, None)
        return x, a

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        frames = np.atleast_2d(np.asarray(frames, dtype=float))
        _, a = self.settled(frames)
        return np.concatenate([cr.unit_rows(a), frames[:, self.pixels:]], axis=1)


def memory_bytes(brain: Any, stage: Any) -> dict[str, int]:
    """The recall memory's address map and the code's own tables, in bytes."""
    index = brain.index
    address = int(index.w_x.nbytes + index.w_c.nbytes + index.b.nbytes)
    if isinstance(stage, LearnedStage):
        code = stage.code.nbytes()
    else:
        separator = stage.code.separator
        code = int(separator.index.nbytes + separator.weight.nbytes)
    return {"address_map": address, "code_tables": code, "total": address + code}


def code_report(stage: LearnedStage, camera: Any, seed: int) -> dict[str, Any]:
    """The code on frames, frozen: the bottom-up residual at rest over a panel's frames, units
    in use, active units per code, and the mean cosine between the codes of the tokens' own
    noiseless single-token frames (identity: low is distinct)."""
    learning = stage.code.learning
    stage.code.learning = False
    try:
        rng = np.random.default_rng(np.random.SeedSequence([NURSERY_SEED, seed, 9]))
        held = tb.holdout(seed)
        trials = tb.draw(rng, 64, held)
        frames = np.concatenate(camera(tb.events(trials), rng))
        x, a = stage.settled(frames)
        residual = stage.code.residuals(x, a, np.zeros_like(a))["bottom_up"]
        tokens = np.minimum(camera.patches, 1.0)[:, : camera.pixels]
        _, single = stage.settled(tokens)
        unit = cr.unit_rows(single)
        sim = unit @ unit.T
        off = ~np.eye(len(sim), dtype=bool)
        active = a > 0
        return {
            "bottom_up_residual": float(residual.mean()),
            "units_in_use": int(np.sum(active.any(axis=0))),
            "active_per_code": float(active.sum(axis=1).mean()),
            "silent_codes": int(np.sum(~active.any(axis=1))),
            "token_cosine_mean": float(sim[off].mean()),
            "token_cosine_max": float(sim[off].max()),
            "token_silent": int(np.sum(~(single > 0).any(axis=1))),
        }
    finally:
        stage.code.learning = learning


def learned_life(spec: dict[str, Any]) -> dict[str, Any]:
    seed, units = int(spec["seed"]), int(spec["units"])
    held = tb.holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([tb.SCHEDULE_SEED, seed]))
    camera = runs.Camera(seed)
    stage = LearnedStage(camera, units, seed)
    nursery_rng = np.random.default_rng(np.random.SeedSequence([NURSERY_SEED, seed]))
    clock = runs.Clock()
    rounds = int(spec.get("nursery", NURSERY))
    first: list[np.ndarray] = []
    while sum(len(f) for f in first) < units:  # the first frames heard, imprinted
        trials = tb.draw(nursery_rng, tb.BATCH, held)
        first.extend(camera(tb.events(trials), nursery_rng))
    stage.imprint(np.concatenate(first)[:units])
    for _ in range(rounds):
        trials = tb.draw(nursery_rng, tb.BATCH, held)
        for frame in camera(tb.events(trials), nursery_rng):
            stage(frame)
    nursery = {"code": code_report(stage, camera, seed), **clock.stamp()}
    brain = runs.built(spec, inputs=stage.units)

    def see(events: list[np.ndarray], noise: np.random.Generator | None = None,
            ) -> list[np.ndarray]:
        return [stage(frame) for frame in camera(events, noise)]

    work: dict[str, Any] = tb.new_work()
    readings: list[dict[str, Any]] = []
    done = 0
    for budget in spec["budgets"]:
        while done < budget:
            trials = tb.draw(rng, tb.BATCH, held)
            brain.reset()
            runs.episode(brain, see(tb.events(trials)), tb.answers(trials), work)
            done += 1
        clock.read()
        stage.code.learning = False
        try:
            panels = runs.frame_panels(brain, seed, held, see,
                                       int(runs.CAMERA["render_seed"]))
        finally:
            stage.code.learning = True
        readings.append({"rounds": budget, **clock.stamp(), "panels": panels,
                         "gate": runs.level_gate(panels["held_out"], 1.0 / tb.VALUES),
                         "units": stage.units, "inputs": camera.inputs,
                         "code": code_report(stage, camera, seed),
                         "memory": memory_bytes(brain, stage)})
    return {"readings": readings, "work": work, "nursery": nursery,
            "code_work": dict(stage.code.work)}


def generic_life(spec: dict[str, Any]) -> dict[str, Any]:
    """``index_runs``' camera life with the generic code, as the index chamber's runs took it,
    plus its memory in bytes."""
    out = runs.camera_life(spec)
    camera = runs.Camera(int(spec["seed"]))
    stage = runs.GenericCode(camera.inputs, camera.pixels, GENERIC["units"], 32,
                             int(spec["generic"]), int(spec["seed"]))
    brain = runs.built(spec, inputs=stage.units)
    out["memory"] = memory_bytes(brain, stage)
    return out


def life(spec: dict[str, Any]) -> dict[str, Any]:
    warnings.filterwarnings("ignore", "finite teaching free phase", RuntimeWarning)
    began, cpu = time.monotonic(), time.process_time()
    out = learned_life(spec) if spec["stage"] == "learned" else generic_life(spec)
    return {**spec, **out, "elapsed": time.monotonic() - began,
            "cpu": time.process_time() - cpu}


def specs(stages: list[str], seeds: list[int], budgets: list[int],
          nursery: int = NURSERY) -> list[dict[str, Any]]:
    genes = runs.genes_for("camera", "index")
    out = []
    for stage in stages:
        for seed in seeds:
            base = {"task": "camera", "arm": "index", "seed": seed, "budgets": budgets,
                    "genes": genes}
            if stage == "generic":
                out.append({**base, "stage": "generic", "generic": GENERIC["winners"]})
            else:
                out.append({**base, "stage": "learned", "units": int(stage),
                            "nursery": nursery})
    return out


def digests() -> dict[str, str]:
    files = [Path(__file__), HERE / "code_region.py", BINDING / "index_runs.py",
             BINDING / "index_read.py", BINDING / "token_binding.py"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def life_name(spec: dict[str, Any]) -> str:
    return f"{spec['stage']}-{spec.get('units', GENERIC['units'])}-s{spec['seed']}.json"


def kept_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life, written to ``spec["lives"]`` when it ends (through a temporary file)."""
    row = life({k: v for k, v in spec.items() if k != "lives"})
    row["digests"] = digests()
    target = Path(spec["lives"]) / life_name(spec)
    partial = target.with_suffix(".partial")
    partial.write_text(json.dumps(row, indent=1, default=float) + "\n")
    partial.replace(target)
    return {"name": life_name(spec), "cpu": row["cpu"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True, help="a directory")
    parser.add_argument("--stages", nargs="*", default=[str(u) for u in UNITS] + ["generic"])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    parser.add_argument("--nursery", type=int, default=NURSERY)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    lives = args.out / "lives"
    lives.mkdir(parents=True, exist_ok=True)
    jobs = [{**job, "lives": str(lives)}
            for job in specs(args.stages, args.seeds, sorted(args.budgets), args.nursery)]
    jobs = [job for job in jobs if not (lives / life_name(job)).exists()]
    began = time.monotonic()
    if jobs:
        with get_context("spawn").Pool(max(1, min(args.workers, len(jobs)))) as pool:
            for done in pool.imap_unordered(kept_life, jobs, chunksize=1):
                print(f"{done['name']} {done['cpu']:.0f} s", flush=True)
    rows = [json.loads(path.read_text()) for path in sorted(lives.glob("*.json"))]
    body = {"schema": "learned-code-camera/1", "digests": digests(), "rows": rows,
            "constants": {"GENERIC": GENERIC, "UNITS": UNITS, "NURSERY": args.nursery,
                          "CODE": CODE.to_dict()},
            "elapsed_seconds": time.monotonic() - began}
    (args.out / "receipt.json").write_text(json.dumps(body, indent=1, default=float) + "\n")
    for row in rows:
        r = row["readings"][-1]
        print(row["stage"], row.get("units", GENERIC["units"]), row["seed"], r["rounds"],
              f"{r['panels']['held_out']['intact']:.3f}", r.get("memory"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
