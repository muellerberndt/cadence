"""The memory benchmarks on deep brains (``cadence-depth-memory/1``).

The look-back, contextual-choice and held-heads benchmarks of ``benchmarks/attention`` run
unedited, with every brain they compose deepened: each module width repeated ``depth`` times
(look-back's ``(24,)`` becomes ``(24, 24)`` at depth 2), the apical compartment
(``apical=0.25``), the projection-relative floor (``learning_normalize_relative=0.1``) where the
learner normalizes, and normalized teaching where a benchmark sets no teaching of its own. Each
benchmark's own gates decide each founder, against its own controls; the receipt keeps every
founder's readings.

    python benchmarks/depth/memory.py --out /tmp/depth-memory --depths 2 3 --jobs 32
    python benchmarks/depth/memory.py --verify /tmp/depth-memory/summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import cadence as cd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KIND = "cadence-depth-memory/1"
BENCHMARKS = ("look_back", "contextual_choice", "held_heads")
SEEDS = (0, 1, 2, 3)
RULES = ("report", "flee")  # held heads asks for two facts reported, or one action from both
APICAL, RELATIVE = 0.25, 0.1


def benchmark(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"depth_{name}", ROOT / "benchmarks" / "attention" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources() -> list[tuple[str, Path]]:
    """This script's bytes and those of every benchmark it runs, with their library sources."""
    # the compartment and the learner's changes live in brain.py and learning.py
    names = {"benchmarks/depth/memory.py", "src/cadence/brain.py", "src/cadence/learning.py"}
    for name in BENCHMARKS:
        names.update(path for path, _ in benchmark(name).sources())
    return [(name, ROOT / name) for name in sorted(names)]


def deepen(depth: int) -> None:
    """Every ``Brain.compose`` in this process composes ``depth`` times the modules, with the
    compartment, the relative floor and normalized teaching where the call sets none."""
    original = cd.Brain.compose.__func__

    def compose(cls: type, *args: Any, **options: Any) -> cd.Brain:
        options["modules"] = tuple(options.get("modules", (64,))) * depth
        options.setdefault("apical", APICAL)
        if "learning" not in options and options.get("teaching") is None:
            options["teaching"] = "normalized"
        options.setdefault("learning_normalize_relative", RELATIVE)
        return original(cls, *args, **options)

    cd.Brain.compose = classmethod(compose)  # type: ignore[method-assign]


def run_cell(name: str, depth: int, seed: int, rule: str | None) -> dict[str, Any]:
    deepen(depth)
    module = benchmark(name)
    started = time.time()
    if name == "held_heads":
        result = module.run_founder(seed, rule)
    else:
        result = module.run_founder(seed)
    return {"benchmark": name, "depth": depth, "seed": seed, "rule": rule,
            "passed": bool(result["passed"]), "result": result,
            "seconds": round(time.time() - started, 1)}


def cells(depths: list[int], seeds: list[int]) -> list[tuple[str, int, int, str | None]]:
    out: list[tuple[str, int, int, str | None]] = []
    for depth in depths:
        for name in BENCHMARKS:
            rules: tuple[str | None, ...] = RULES if name == "held_heads" else (None,)
            out += [(name, depth, seed, rule) for rule in rules for seed in seeds]
    return out


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Founders passing each benchmark's own gates, per benchmark and depth."""
    table: dict[str, dict[str, list[int]]] = {}
    for run in runs:
        row = table.setdefault(run["benchmark"], {}).setdefault(str(run["depth"]), [0, 0])
        row[0] += int(run["passed"])
        row[1] += 1
    return {name: {depth: {"passed": n, "founders": m} for depth, (n, m) in sorted(rows.items())}
            for name, rows in sorted(table.items())}


def check(body: dict[str, Any]) -> str | None:
    """Every founder's decision follows from its benchmark's gates, and the summary from the
    founders."""
    try:
        if body["schema"] != KIND:
            return "wrong schema"
        for run in body["runs"]:
            gates = run["result"].get("gates")
            if gates is not None and run["passed"] != all(gates.values()):
                return f"{run['benchmark']} founder {run['seed']}: the decision does not follow"
            if run["passed"] != bool(run["result"]["passed"]):
                return f"{run['benchmark']} founder {run['seed']}: the decision was altered"
        if summarize(body["runs"]) != body["summary"]:
            return "summary does not match the stored founders"
    except (KeyError, TypeError, ValueError):
        return "incomplete or invalid readings"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    goal = parser.add_mutually_exclusive_group(required=True)
    goal.add_argument("--out", type=Path)
    goal.add_argument("--verify", type=Path)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--jobs", type=int, default=1, help="founders run in parallel")
    args = parser.parse_args()
    if args.verify is not None:
        valid, message = cd.Receipt.verify(args.verify, sources=sources(), check=check)
        print(message)
        raise SystemExit(0 if valid else 1)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    todo = cells(args.depths, args.seeds)
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(run_cell, *cell) for cell in todo]
        runs = []
        for future in futures:  # in the declared order, whatever finishes first
            run = future.result()
            runs.append(run)
            with open(out / "rows.jsonl", "a") as handle:
                handle.write(json.dumps(run) + "\n")
            print(json.dumps({k: run[k] for k in ("benchmark", "depth", "seed", "rule", "passed",
                                                  "seconds")}), flush=True)
    body = {
        "schema": KIND,
        "compose": {"apical": APICAL, "learning_normalize_relative": RELATIVE,
                    "teaching": "normalized where the benchmark sets none",
                    "modules": "each benchmark's widths repeated depth times"},
        "depths": args.depths, "seeds": args.seeds, "rules": list(RULES),
        "version": cd.__version__, "runs": runs, "summary": summarize(runs),
    }
    cd.Receipt.build(KIND, body, sources()).write(out / "summary.json")
    print(json.dumps(body["summary"], indent=1))


if __name__ == "__main__":
    main()
