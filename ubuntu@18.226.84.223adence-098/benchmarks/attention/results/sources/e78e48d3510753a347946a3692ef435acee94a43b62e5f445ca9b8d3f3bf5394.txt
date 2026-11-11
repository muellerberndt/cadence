"""Looking back by position: the recall memory's position term (``recall_position``).

Each episode is a stream of five symbols (four kinds, drawn at random), then a cue, ``first``
or ``last``. The answer is the symbol at the cued place, so it depends only on when a moment
came. Only the cue moment is taught (``Brain.lesson`` on the cue, then the act that writes
it), at the teaching point for many lessons (``teaching="normalized"``), 1200 episodes per
founder; each episode begins with empty streams. The cue matches no stored key, and the slots
hold random symbols, so content cannot name the cued slot; recency is off
(``recall_recency=0``), so no fixed preference for the newest slot serves either cue. The
founder memory with its position term (``recall_position=32``) is compared with the memory
without it (``recall_position=0``) at the same lessons and test episodes, and with uniform
random (1/4). Both memories hold the episode's boundary (``recall_boundary=True``): a silent
slot before the first symbol whose next the first symbol teaches, so the first symbol, like
every later one, can be read through its own slot's value or its predecessor's next.

This is bounded component evidence that lessons can teach a cue to choose how far back the
read looks. It is not evidence for order questions in language or music.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYMBOLS, LENGTH, INPUTS = 4, 5, 8
FIRST, LAST = 4, 5  # the cues' inputs
LESSONS, TESTS = 1200, 160
UNIFORM = 1 / SYMBOLS


def sources() -> list[tuple[str, Path]]:
    """The public implementation bytes this result depends on."""
    names = (
        "benchmarks/attention/look_back.py",
        "src/cadence/generic.py",
        "src/cadence/stream.py",
    )
    return [(name, ROOT / name) for name in names]


def compose(seed: int, position: int) -> cd.Brain:
    return cd.Brain.compose(
        INPUTS, SYMBOLS, modules=(24,), seed=seed, episodic=False, recall=True,
        recall_position=position, recall_recency=0.0, recall_temperature=0.25,
        teaching="normalized", recall_boundary=True,
    )


def episode(rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    symbols = rng.integers(SYMBOLS, size=LENGTH)
    cue = FIRST if rng.random() < 0.5 else LAST
    return symbols, cue, int(symbols[0] if cue == FIRST else symbols[-1])


def hear(brain: cd.Brain, symbols: np.ndarray) -> None:
    brain.reset()
    for s in symbols:
        brain.act(np.eye(INPUTS)[[int(s)]], greedy=True)


def run_arm(seed: int, position: int) -> dict[str, Any]:
    brain = compose(seed, position)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a lesson after an act on the same moment is refused
        for _ in range(LESSONS):
            symbols, cue, label = episode(rng)
            hear(brain, symbols)
            cue_x = np.eye(INPUTS)[[cue]]
            brain.lesson(cue_x, np.array([label]))
            brain.act(cue_x, greedy=True)
    test = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    hits = {FIRST: 0, LAST: 0}
    count = {FIRST: 0, LAST: 0}
    for _ in range(TESTS):
        symbols, cue, label = episode(test)
        hear(brain, symbols)
        answer = int(np.asarray(brain.act(np.eye(INPUTS)[[cue]], greedy=True)).reshape(-1)[0])
        hits[cue] += int(answer == label)
        count[cue] += 1
    return {
        "position": position,
        "first": hits[FIRST] / count[FIRST],
        "last": hits[LAST] / count[LAST],
        "accuracy": (hits[FIRST] + hits[LAST]) / TESTS,
        "tests": dict(first=count[FIRST], last=count[LAST]),
        "maps_moved": bool(
            any(np.abs(m).sum() > 0 for maps in brain.recall.position_maps for m in maps)
        ),
    }


def run_founder(seed: int) -> dict[str, Any]:
    with_term, without = run_arm(seed, 32), run_arm(seed, 0)
    gates = {
        "both_cues_with_the_term": min(with_term["first"], with_term["last"]) >= 0.9,
        "beats_the_memory_without_it": with_term["accuracy"] >= without["accuracy"] + 0.2,
        "beats_uniform_random": with_term["accuracy"] >= UNIFORM + 0.5,
        "the_term_was_taught": with_term["maps_moved"],
    }
    return {"seed": seed, "with_position": with_term, "without_position": without,
            "uniform_random": UNIFORM, "gates": gates, "passed": all(gates.values())}


def check(body: dict[str, Any]) -> str | None:
    """The decision the readings support, or the reason they do not."""
    founders = body.get("founders", [])
    if len(founders) < 4:
        return "fewer than four founders"
    for founder in founders:
        a, b = founder["with_position"], founder["without_position"]
        gates = {
            "both_cues_with_the_term": min(a["first"], a["last"]) >= 0.9,
            "beats_the_memory_without_it": a["accuracy"] >= b["accuracy"] + 0.2,
            "beats_uniform_random": a["accuracy"] >= UNIFORM + 0.5,
            "the_term_was_taught": a["maps_moved"],
        }
        if gates != founder["gates"] or founder["passed"] != all(gates.values()):
            return f"founder {founder['seed']}: the gates do not follow from the readings"
    if body["passed"] != all(f["passed"] for f in founders):
        return "the decision does not follow from the founders"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--founder-out", type=Path, help="write one founder's readings (JSON)")
    parser.add_argument("--merge", nargs="+", type=Path, help="founder readings to combine")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.merge:
        founders = [json.loads(path.read_text()) for path in args.merge]
    else:
        if args.founder_out is not None and len(args.seeds) != 1:
            raise SystemExit("--founder-out writes one founder: pass one seed")
        founders = [run_founder(seed) for seed in args.seeds]
        if args.founder_out is not None:
            args.founder_out.write_text(json.dumps(founders[0], sort_keys=True, indent=2) + "\n")
            return
    result = {
        "schema": "cadence-look-back/2",
        "scope": "a taught cue chooses how far back the read looks; bounded component evidence",
        "configuration": {
            "inputs": INPUTS, "symbols": SYMBOLS, "length": LENGTH, "modules": [24],
            "lessons": LESSONS, "tests": TESTS, "teaching": "normalized", "recall_recency": 0.0,
            "recall_temperature": 0.25, "episodic": False, "positions": [32, 0],
            "recall_boundary": True,
        },
        "founders": founders,
        "passed": all(founder["passed"] for founder in founders),
    }
    problem = check(result)
    if problem is not None:
        raise AssertionError(problem)
    if args.out is not None:
        cd.Receipt.build("cadence-look-back/2", result, sources()).write(args.out)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
