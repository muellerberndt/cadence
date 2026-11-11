"""Source-bound native robot-arena acquisition and private greedy behavior comparison.

Run each candidate source in a separate process; retain every arm's receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", type=Path, required=True, help="Cadence source checkout")
    parser.add_argument("--arena", type=Path, required=True, help="cadence-robot-arena checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--founders", nargs="+", type=int, choices=(11, 12, 13), default=[11, 12, 13],
    )
    parser.add_argument("--moments", type=int, default=4000)
    parser.add_argument("--evaluate", type=int, default=1000)
    parser.add_argument("--probe-every", type=int, default=0)
    args = parser.parse_args()
    if args.moments < 1000 or args.moments % 1000 or args.evaluate < 500 or args.evaluate % 500:
        parser.error("training must use full 1000-moment blocks; evaluation full 500-moment blocks")
    core, arena = args.cadence.resolve(), args.arena.resolve()
    sys.path[:0] = [str(core / "src"), str(arena)]
    import arena.nursery as nursery
    from arena.parts import stock_designs
    from arena.probe import fingerprint

    import cadence

    if Path(cadence.__file__).resolve().parents[2] != core:
        raise RuntimeError("imported Cadence does not match the declared source checkout")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def sources() -> dict[str, dict[str, str]]:
        return {
            "cadence": {
                str(p.relative_to(core)): digest(p) for p in (core / "src/cadence").rglob("*.py")
            },
            "arena": {
                str(p.relative_to(arena)): digest(p) for p in (arena / "arena").rglob("*.py")
            },
        }

    def revision(path: Path) -> str:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()

    original = sources()
    producer = {"file": Path(__file__).name, "sha256": digest(Path(__file__))}
    observe = nursery.observe

    def without_target_bearings(robot, world):
        observations = observe(robot, world)
        observations[:, :4] = 0.0
        return observations

    args.output.mkdir(parents=True, exist_ok=True)
    # These world seeds were fixed before the paired additional-founder comparison.
    schedules = {11: (7, 19), 12: (8, 19), 13: (9, 20)}
    for founder in args.founders:
        world, evaluation = schedules[founder]
        blueprint = replace(stock_designs()[0], seed=founder)
        prefix = args.output / f"{args.arm}-tumbler-seed{founder}-{args.moments}"
        if prefix.with_suffix(".json").exists() or prefix.with_suffix(".npz").exists():
            raise FileExistsError(
                f"preserve the previous attempt at {prefix}; use a new output folder"
            )
        training = nursery.run_nursery(
            blueprint, moments=args.moments, seed=world, block=1000,
            save_to=prefix.with_suffix(".npz"), verbose=True, probe_every=args.probe_every,
        )
        probe = fingerprint(blueprint, prefix.with_suffix(".npz"))
        greedy = nursery.run_nursery(
            blueprint, moments=args.evaluate, seed=evaluation, policy="frozen",
            brain_path=prefix.with_suffix(".npz"), block=500,
        )
        random = nursery.run_nursery(
            blueprint, moments=args.evaluate, seed=evaluation, policy="random", block=500,
        )
        try:
            nursery.observe = without_target_bearings
            lesion = nursery.run_nursery(
                blueprint, moments=args.evaluate, seed=evaluation, policy="frozen",
                brain_path=prefix.with_suffix(".npz"), block=500,
            )
        finally:
            nursery.observe = observe
        if sources() != original:
            raise RuntimeError("source changed during the comparison")
        receipt = {
            "protocol": "arena-policy-development/1", "producer": producer,
            "arm": args.arm, "brainseed": founder, "worldseed": world, "evalseed": evaluation,
            "cadence_commit": revision(core), "arena_commit": revision(arena),
            "source_sha256": original, "python": sys.version,
            "checkpoint_sha256": digest(prefix.with_suffix(".npz")),
            "training": training, "fingerprint": probe, "greedy": greedy, "random": random,
            "without_target_bearings": lesion,
            "limitations": [
                "Development comparison; not independent confirmation or an arena-wide guarantee",
                "run_nursery leaves its final world reward undelivered; no resumed-training claim",
                "Greedy and ablated evaluations use private loaded copies and do not learn",
                "The ablation zeros target bearing/proximity coordinates 0:4; other senses remain",
                "Actions can change dummy relocation times despite matched initial world seeds",
                "Nursery timing includes intermediate probes; its sweep counters omit them",
                "Final standalone fingerprint is outside nursery timing and has no sweep count",
            ],
        }
        prefix.with_suffix(".json").write_text(json.dumps(receipt, indent=2, default=float) + "\n")
        print(json.dumps({
            "arm": args.arm, "seed": founder, "train": training["total"],
            "distinct": probe["distinct_answers"], "sensitivity": probe["sensitivity"],
            "greedy": greedy["total"], "random": random["total"],
            "without_target_bearings": lesion["total"],
        }, default=float), flush=True)


if __name__ == "__main__":
    main()
