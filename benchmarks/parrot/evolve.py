"""Selection of the parrot brain's wiring (#332 E4): the library's ``evolve`` over named genes.

The genome is a dict of named genes (the founder's ``cadence.parrot.brain.GENES``); ``SPACE``
declares the genes this lineage searches, as ``cadence.genes`` takes them, with the names of the
parrot's wiring guide (docs/parrot-brain.md). A genome lives ``worlds`` short lives, each on a
world shared by its whole generation, and is scored from the chamber's readings:

    fitness = -(m1 / h1 + m2 / h2 + 0.5 m3 / h3 + m4 / h4) - 0.2 cost / cost_h

- m1, seen: median distance from each thing to the nearest place the map holds, after the
  parrot's own look-around (cm, at most 30);
- m2, held: the same for things held while they are put in the blind spot;
- m3, one thing: the share of things not held as exactly one thing;
- m4, distance: median |log(read / true)| of the distances read for things looked at, by both
  eyes and by one, the mean of the two;
- cost: multiply-accumulates a moment.

Each term is divided by the hand-set genome's value on the reference worlds (``--reference``),
so the hand-set scores about -3.7. The covered-eye deficit (E3) never enters the fitness.

``python -m benchmarks.parrot.evolve --reference runs/ga/reference.json --out runs/ga``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.parrot.chamber import live, test
from cadence import evolve, genes
from cadence.parrot.brain import GENES, Parrot

SPACE: dict[str, tuple[Any, ...]] = {
    "disparity_channels": ("int", 12, 48),
    "disparity_max": ("log", 0.2, 8.0, 24.0),
    "disparity_tuning": ("log", 0.3, 0.2, 2.0),
    "disparity_form": ("choice", "edges", "energy"),
    "parallax_delay": ("int", 1, 6),
    "parallax_channels": ("int", 4, 24),
    "reach_bins": ("int", 6, 36),
    "reach_rate": ("log", 0.3, 0.05, 1.0),
    "reach_width": ("log", 0.3, 0.5, 4.0),
    "p_glimpse_dl": ("choice", 0, 1),
    "p_both_cues": ("choice", 0, 1),
    "agreement_views": ("choice", "before-after", "binocular", "both"),
    "agreement_scale": ("log", 0.3, 0.004, 0.04),
    "agreement_parallax": ("log", 0.3, 0.3, 4.0),
    "agreement_window": ("int", 2, 20),
    "place_keep_angle": ("log", 0.3, 2.0, 15.0),
    "match_step": ("log", 0.3, 0.05, 0.4),
    "match_stride": ("log", 0.3, 0.1, 1.0),
    "kinds_vigilance": ("linear", 0.05, 0.1, 0.5),
    "novelty": ("log", 0.3, 0.1, 3.0),
    "isthmi_local": ("log", 0.3, 0.2, 5.0),
    "focus_habituate": ("log", 0.3, 1.0, 30.0),
    "focus_explore": ("linear", 0.05, 0.0, 0.5),
    "peer_amplitude": ("linear", 0.1, 0.0, 1.0),
    "peer_period": ("int", 2, 12),
}
LIFE = {"scenes": 16, "moments": 50}
TESTS = {"scenes": 3, "moments": 60, "behind": 20, "margin": 0.08, "alone": 0.15,
         "front_panel": False}
WEIGHTS = {"m1": 1.0, "m2": 1.0, "m3": 0.5, "m4": 1.0}
REFERENCE: dict[str, float] = {}


def terms(genome: dict[str, Any], world: int, seed: int) -> dict[str, float]:
    """One life on world ``world`` with the brain's own seed ``seed``: the fitness terms."""
    rng = np.random.default_rng(np.random.SeedSequence([int(world), 332, 1]))
    parrot = Parrot(genome, seed=seed)
    live(parrot, rng, LIFE["scenes"], LIFE["moments"])
    reading = test(parrot, np.random.default_rng(np.random.SeedSequence([int(world), 332, 2])),
                   **TESTS)
    cost = parrot.cost()
    looked = [reading["E3_looked"], reading["E3_looked_one"]]
    return {"m1": 100.0 * reading["gap_seen"],
            "m2": 100.0 * (reading["gap_behind"] if np.isfinite(reading["gap_behind"])
                           else reading["gap_seen"]),
            "m3": 1.0 - (reading["E2_one"] if np.isfinite(reading["E2_one"]) else 0.0),
            "m4": float(np.nanmean(looked)) if np.isfinite(looked).any() else 1.0,
            "cost": float(sum(cost["macs"].values()))}


def score(values: dict[str, float], reference: dict[str, float]) -> float:
    total = sum(w * values[k] / max(reference[k], 1e-9) for k, w in WEIGHTS.items())
    return float(-total - 0.2 * values["cost"] / max(reference["cost"], 1e-9))


def fitness(genome: dict[str, Any], seed: int) -> float:
    """The mean score over the generation's ``worlds`` worlds (shared by the generation:
    life ``seed`` lives on worlds ``seed // 1000 * 1000 + w``)."""
    worlds = int(os.environ.get("PARROT_WORLDS", "2"))
    base = seed // 1000 * 1000
    values = [terms(genome, base + w, seed) for w in range(worlds)]
    scores = [score(v, REFERENCE) for v in values]
    record = Path(os.environ.get("PARROT_LOG", "")) if os.environ.get("PARROT_LOG") else None
    if record is not None:
        with record.open("a") as f:
            f.write(json.dumps({"seed": seed, "genome": {k: genome[k] for k in SPACE},
                                "terms": values, "scores": scores}) + "\n")
    out = float(np.mean(scores))
    return out if math.isfinite(out) else -100.0


def _set_reference(values: dict[str, float]) -> None:
    REFERENCE.update(values)


def draw(space: dict[str, tuple[Any, ...]], rng: np.random.Generator) -> dict[str, Any]:
    """A genome drawn uniformly over ``space`` (log genes on a log scale), the founder's
    genes elsewhere: random search's member."""
    child = dict(GENES)
    for name, spec in space.items():
        if spec[0] == "log":
            child[name] = float(math.exp(rng.uniform(math.log(spec[2]), math.log(spec[3]))))
        elif spec[0] == "linear":
            child[name] = float(rng.uniform(spec[2], spec[3]))
        elif spec[0] == "int":
            child[name] = int(rng.integers(spec[1], spec[2] + 1))
        else:
            child[name] = spec[1 + int(rng.integers(len(spec) - 1))]
    return child


def _reference_world(world: int) -> dict[str, float]:
    return terms(dict(GENES), world, world)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=33290000)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--rate", type=float, default=0.15)
    parser.add_argument("--processes", type=int, default=60)
    parser.add_argument("--references", type=int, default=6)
    parser.add_argument("--random", action="store_true",
                        help="random search at equal evaluations in place of selection")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["PARROT_WORLDS"] = str(args.worlds)
    os.environ["PARROT_LOG"] = str(args.out / "lives.jsonl")
    reference_path = args.out / "reference.json"
    with Pool(args.processes) as pool:
        if reference_path.exists():
            reference = json.loads(reference_path.read_text())
        else:
            worlds = [args.seed - 1000 + w for w in range(args.references)]
            rows = pool.map(_reference_world, worlds)
            reference = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            reference_path.write_text(json.dumps({**reference, "worlds": worlds,
                                                  "rows": rows}))
        REFERENCE.update({k: float(reference[k]) for k in ("m1", "m2", "m3", "m4", "cost")})
        pool.close()
    start = time.perf_counter()

    def report(lineage: Any) -> None:
        (args.out / "lineage.json").write_text(json.dumps({
            "generations": lineage.generations, "best": lineage.best,
            "best_fitness": lineage.best_fitness, "seconds": time.perf_counter() - start,
            "space": {k: list(v) for k, v in SPACE.items()}, "reference": REFERENCE,
            "random": args.random}, default=str))
        g = lineage.generations[-1]
        print(f"generation {g['generation']}: best {g['best_fitness']:.3f} "
              f"mean {g['mean_fitness']:.3f} ({time.perf_counter() - start:.0f}s)", flush=True)

    mutate = genes(SPACE, rate=args.rate)

    def fresh(genome: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        return draw(SPACE, rng)

    step = fresh if args.random else mutate
    with Pool(args.processes, initializer=_set_reference, initargs=(dict(REFERENCE),)) as pool:
        lineage = evolve(fitness, dict(GENES), generations=args.generations,
                         population=args.population, keep=args.keep, seed=args.seed,
                         mapper=pool.map, report=report, mutate=step)
    print("best", json.dumps({k: lineage.best[k] for k in SPACE}, default=str))


if __name__ == "__main__":
    main()
