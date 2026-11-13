"""Step S2 of the meaning plan (cadence#334): a life of exchanges, its readings, its QC, and the
genetic algorithm over the brain's composition (lineage L1).

    python s2.py life <founder> '<json>' <out.json>        one life, read after every stage
    python s2.py qc <founder> '<json>' <out.json>          one life, then the fixed QC set
    python s2.py ga '<json>' <out_dir>                     lineage L1 and random search

A life lives the stages of ``STAGES`` in order, each a mix of exchanges (``exchange.py``), and is
read after each stage on a frozen copy: replies after questions and after statements (in the
gap), the words said back, the names of things held up, and the looks at named things.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exchange as xc  # noqa: E402

from cadence import evolve, genes  # noqa: E402
from cadence.meaning import HAND_SET, SPACE, random_genome  # noqa: E402

STAGES: list[tuple[str, dict[str, float]]] = [
    ("babble", {"babble": 1.0}),
    ("turns", {"T0": 0.5, "T1": 0.5}),
    ("questions", {"T1": 1.0}),
    ("imitate", {"imitate": 0.8, "T1": 0.2}),
    ("names", {"name_model": 0.5, "name": 0.4, "imitate": 0.1}),
    ("look", {"look": 0.6, "name": 0.3, "name_model": 0.1}),
]
BLOCKS = {"babble": 30, "turns": 150, "questions": 150, "imitate": 600, "names": 600,
          "look": 300}
QC_SET = {"T1": 40, "imitate": 40, "name": 40, "look": 20}


def read(p: xc.Parrot, rng: np.random.Generator, counts: dict[str, int]) -> dict[str, float]:
    """The frozen brain's readings on fresh exchanges, ``counts`` of each kind."""
    out: dict[str, float] = {}
    for kind, n in counts.items():
        exs: list[xc.Exchange] = []
        got: dict[str, Any] = {"replied": [], "looked": [], "over": 0, "late": 0}
        while len(exs) < n:
            block = xc.block(p, rng, kind)
            lived = xc.live(p, block, rng, learn=False)
            exs += block
            got["replied"] += lived["replied"]
            got["looked"] += lived["looked"]
            got["over"] += lived["over"]
            got["late"] += lived["late"]
        exs, replied, looked = exs[:n], got["replied"][:n], got["looked"][:n]
        if kind == "T1":
            q = [r is not None for r, ex in zip(replied, exs, strict=True) if ex.question]
            s = [r is not None for r, ex in zip(replied, exs, strict=True) if not ex.question]
            out["after_questions"] = float(np.mean(q)) if q else float("nan")
            out["after_statements"] = float(np.mean(s)) if s else float("nan")
            out["spoke_over"] = got["over"] / n
            out["spoke_late"] = got["late"] / n
        elif kind == "look":
            out["look"] = float(np.mean([a == ex.look for a, ex in zip(looked, exs,
                                                                         strict=True)]))
        else:
            name = {"imitate": "said_back", "name": "names"}[kind]
            out[name] = float(np.mean([r == ex.answer for r, ex in zip(replied, exs,
                                                                         strict=True)]))
            out[f"{name}_silent"] = float(np.mean([r is None for r in replied]))
    return out


def life(seed: int, genome: dict[str, Any] | None = None, *, blocks: dict[str, int] | None = None,
         streams: int = 8, reads: int = 16, backend: str = "cpu", world: int | None = None,
         log: Any = None) -> dict[str, Any]:
    """One life through the stages. ``seed`` draws the brain; ``world`` (default the seed) draws
    the founder's words and rooms."""
    budget = {**BLOCKS, **(blocks or {})}
    founder = int(seed if world is None else world)
    rng = np.random.default_rng(np.random.SeedSequence([founder, 334, 21]))
    p = xc.Parrot(founder, genome, streams=streams, backend=backend)
    if world is not None:  # the genome's own draw of the brain on the generation's world
        from cadence.meaning import compose

        p.brain = compose(p.genome, p.inputs, len(p.voice), ear=p.ear, gaze=p.gaze,
                          seed=int(seed), backend=backend)
        p.synapses = int(p.brain.connectome.synapses)
    out: dict[str, Any] = {"seed": int(seed), "founder": founder, "streams": streams,
                           "parameters": int(p.brain.parameters()), "synapses": p.synapses,
                           "stages": {}}
    earned: list[float] = []
    began = time.time()
    for stage, mix in STAGES:
        kinds, weights = list(mix), np.array(list(mix.values()))
        paid = []
        for b in range(int(budget[stage])):
            kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
            lived = xc.live(p, xc.block(p, rng, kind), rng)
            got = float((lived["pistachios"].sum(axis=0) > 0).mean())
            if kind != "babble":
                paid.append(got)
            if log is not None and b % 25 == 0:
                log(f"{stage} {b} {kind} paid={got:.2f} t={time.time() - began:.0f}s")
        earned += paid
        n = max(1, reads // p.streams) * p.streams
        reading = read(p, rng, {"T1": n, "imitate": n, "name": n, "look": n})
        reading["paid"] = float(np.mean(paid)) if paid else float("nan")
        out["stages"][stage] = reading
        if log is not None:
            log(f"READ {stage} {json.dumps(reading)}")
    out["paid_mean"] = float(np.mean(earned)) if earned else 0.0
    out["work"] = dict(p.work)
    out["macs"] = float((p.work["act_sweeps"] + p.work["lesson_sweeps"]
                         + p.work["reward_row_sweeps"]) * p.synapses)
    out["seconds"] = time.time() - began
    out["final"] = out["stages"][STAGES[-1][0]]
    p_final = out["final"]
    out["score"] = float(np.nanmean([
        max(0.0, p_final["after_questions"] - p_final["after_statements"]),
        p_final["said_back"], p_final["names"], p_final["look"]]))
    out["parrot"] = p
    return out


HAND_MACS: dict[str, float] = {}


def fitness(genome: dict[str, Any], seed: int, *, blocks: dict[str, int] | None = None,
            streams: int = 8, hand_macs: float | None = None) -> float:
    """How well a genome learns in a short life on the generation's world, less its compute:
    half the final readings' mean, half the share of pistachios earned, less 0.2 times its
    multiply-accumulates over the hand-set genome's."""
    try:
        got = life(seed, genome, blocks=blocks, streams=streams, world=seed // 1000)
    except Exception as error:  # a genome whose brain cannot be built or settle scores lowest
        print(f"genome failed: {error!r}", flush=True)
        return -1.0
    cost = got["macs"] / hand_macs if hand_macs else 1.0
    return float(0.5 * got["score"] + 0.5 * got["paid_mean"] - 0.2 * cost)


def _job(args: tuple[dict[str, Any], int, dict[str, Any]]) -> float:
    genome, seed, settings = args
    return fitness(genome, seed, **settings)


class Pool:
    """``evolve``'s mapper over a process pool, with the settings every life shares."""

    def __init__(self, workers: int, settings: dict[str, Any]):
        self.executor = ProcessPoolExecutor(max_workers=workers)
        self.settings = settings

    def map(self, fn: Any, jobs: Any) -> list[float]:
        work = [(genome, seed, self.settings) for genome, seed in jobs]
        return list(self.executor.map(_job, work))


def ga(settings: dict[str, Any], out_dir: str) -> dict[str, Any]:
    """Lineage L1 from the hand-set genome, then random search at equal evaluations."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    population, generations = int(settings.get("population", 32)), int(
        settings.get("generations", 20))
    keep, seed = int(settings.get("keep", 8)), int(settings.get("seed", 3340000))
    life_settings = {"blocks": settings.get("blocks"), "streams": int(settings.get("streams", 8))}
    hand = {**HAND_SET, **xc.S2_HAND_SET}
    base = life(seed, hand, blocks=life_settings["blocks"], streams=life_settings["streams"],
                world=seed // 1000)
    life_settings["hand_macs"] = base["macs"]
    (out / "hand_set_life.json").write_text(json.dumps(
        {k: v for k, v in base.items() if k != "parrot"}, default=float, indent=1))
    pool = Pool(int(settings.get("workers", 60)), life_settings)
    space = {k: v for k, v in SPACE.items() if k not in settings.get("fixed", [])}

    def report(lineage: Any) -> None:
        (out / "lineage.json").write_text(json.dumps(
            {"generations": lineage.generations, "best": lineage.best,
             "best_fitness": lineage.best_fitness}, default=float, indent=1))

    lineage = evolve(lambda g, s: fitness(g, s, **life_settings), hand, generations=generations,
                     population=population, keep=keep, seed=seed, mapper=pool.map,
                     mutate=genes(space, rate=float(settings.get("rate", 0.2))), report=report)
    report(lineage)
    rng = np.random.default_rng(seed + 7)
    draws = [{**hand, **random_genome(rng)} for _ in range(population * generations)]
    jobs = [(g, seed + 500_000 + 1000 * (i // population) + i % population)
            for i, g in enumerate(draws)]
    scores = pool.map(None, jobs)
    best = int(np.argmax(scores))
    result = {"lineage_best": lineage.best, "lineage_best_fitness": lineage.best_fitness,
              "random_best": draws[best], "random_best_fitness": float(scores[best]),
              "random_scores": [float(s) for s in scores], "hand_set": hand,
              "hand_set_life_score": base["score"], "hand_macs": base["macs"]}
    (out / "ga.json").write_text(json.dumps(result, default=float, indent=1))
    pool.executor.shutdown()
    return result


def main(argv: list[str]) -> None:
    mode = argv[1]
    if mode in ("life", "qc"):
        seed, given, out = int(argv[2]), json.loads(argv[3]), argv[4]
        got = life(seed, given.get("genome"), blocks=given.get("blocks"),
                   streams=int(given.get("streams", 8)), backend=str(given.get("backend", "cpu")),
                   log=lambda s: print(s, flush=True))
        p = got.pop("parrot")
        if mode == "qc":
            rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 99]))
            got["qc"] = read(p, rng, QC_SET)
            print("QC " + json.dumps(got["qc"]), flush=True)
        got["settings"] = given
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(got, default=float, indent=1))
    elif mode == "ga":
        given, out_dir = json.loads(argv[2]), argv[3]
        print(json.dumps(ga(given, out_dir), default=float)[:2000], flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)


__all__ = ["BLOCKS", "QC_SET", "STAGES", "fitness", "ga", "life", "read"]
