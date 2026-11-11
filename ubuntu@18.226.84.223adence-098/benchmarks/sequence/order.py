"""Ordered recall from reward alone (cadence#236): which write does the read find?

Each episode writes two values and then asks for one: ``START``, ``WRITE`` (the first value),
``WRITE`` (the second), ``NEUTRAL``, then ``QUERY`` for the latest value written or
``QUERY_FIRST`` for the first. In the ``-gap`` conditions a ``NEUTRAL`` event falls between the
writes. Ten inputs: the markers ``START`` (0), ``WRITE`` (1), ``NEUTRAL`` (2), ``QUERY`` (4) and
``QUERY_FIRST`` (9), and four payload inputs (5 to 8) that carry a write's value, 0 as
``[1, 1, 0, 0]`` and 1 as ``[0, 0, 1, 1]`` (input 3, a distractor marker, is never shown). Two
actions. The query's action is paid +1 when it names the asked value and -1 otherwise; every
other action is ignored and paid nothing. Nothing after a write carries its value, so the
answer needs the read to find the right write by when it happened. This is the ordered recall
world of the Stone Age sequence instrument (``benchmarks/sequence_v1/worlds.py`` in
muellerberndt/cadence-stone-age).

``--world recall`` reads the same instrument's recall world instead, whose ``QUERY`` always
asks the latest value written: nine inputs (no ``QUERY_FIRST``) and seven conditions. In
``clean-k`` one ``WRITE`` is followed by ``k`` ``NEUTRAL`` events; in ``distractor-k`` by ``k``
``DISTRACTOR`` events (input 3), each with a payload drawn uniformly from ``[-0.25, 0.25]``
from the episode's own seed; ``replacement`` writes the other value, then a ``NEUTRAL``, then
the asked value and a ``NEUTRAL``; ``order-latest`` writes the other value, then the asked one,
then a ``NEUTRAL``. Every episode begins with ``START`` and ends with ``QUERY``.

Two ways to teach (``--teach``). ``reward``: one continuing stream through ``Brain.live``, the
outcome of each act arriving with the next event, the query's with the next episode's first
event together with ``done``; nothing teaches but the outcomes. ``lesson``: every episode on a
fresh stream (``reset``), every event a greedy ``act``, and at the query a ``lesson`` naming the
asked value before the act, as the sequence battery teaches. Training episodes come in
balanced blocks of the world's conditions with both values, shuffled (``SeedSequence([founder,
236])``); in the recall world each episode of a block then draws its seed from the same
generator, as the Stone Age schedule does.

At every budget the life is read on forks (``copy.deepcopy``) that never change it:
``EVALUATION`` episodes in balanced blocks from ``SeedSequence([founder, 632])``, each on a
fresh stream (``reset``), every event an ``act``, greedy on one fork and sampled on another.
An episode succeeds when the query's act names the asked value.

    python benchmarks/sequence/order.py --arms founder position_address \\
        --founders 1001 1002 1003 1004 1005 --out results/order.jsonl --workers 5
    python benchmarks/sequence/order.py --world recall --arms founder position \\
        --out results/recall.jsonl --workers 5
    python benchmarks/sequence/order.py --summary results/order.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

START, WRITE, NEUTRAL, DISTRACTOR, QUERY, QUERY_FIRST = 0, 1, 2, 3, 4, 9
PAYLOAD = slice(5, 9)
INPUTS, ACTIONS = 10, 2
VALUES = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
CONDITIONS = ("ask-latest", "ask-first", "ask-latest-gap", "ask-first-gap")
# the recall world: the latest value written is asked, nine inputs (no QUERY_FIRST)
RECALL_CONDITIONS = ("clean-0", "clean-1", "clean-2", "distractor-1", "distractor-2",
                     "replacement", "order-latest")
WORLDS = {"order": (INPUTS, CONDITIONS), "recall": (9, RECALL_CONDITIONS)}
RIGHT, WRONG = 1.0, -1.0
BUDGETS = (1536, 3072)
EVALUATION = 224
FOUNDERS = (1001, 1002, 1003, 1004, 1005)
WINDOW = 256  # episodes per entry of the training log
BRAIN = {
    "modules": (32,), "lateral": 0.0, "episodic": False, "arousal": True,
    "learning_qualified": True, "learning_free_steps": 1024, "learning_nudged_steps": 1024,
    "learning_tolerance": 0.003, "working_memory_decay": 0.0,
    "working_memory_amplitude": 0.3, "recall": True, "recall_founder": True,
}
ARMS: dict[str, dict[str, Any]] = {
    "founder": {},
    "position": {"recall_position_reward": 4.0},
    # reward on both maps, the first write found by its place in the episode, one port
    "reward_recipe": {"recall_position_reward": 4.0, "recall_address_reward": 1.0,
                      "recall_anchor": True, "recall_next": False},
    "address": {"recall_address_reward": 1.0},
    "position_address": {"recall_position_reward": 4.0, "recall_address_reward": 1.0},
    "position_address_anchor": {"recall_position_reward": 4.0, "recall_address_reward": 1.0,
                                "recall_anchor": True},
    "slow": {"recall_position_rate": 0.01},
    "recipe": {"recall_anchor": True, "recall_slope": 0.1, "recall_position_rate": 0.01,
               "recall_recency": 0.0, "recall_boundary": True},
}
TEACH = ("reward", "lesson")


def events(condition: str, value: int, seed: int = 0) -> list[np.ndarray]:
    """The observations of one episode whose query asks ``value``, in the world the condition
    belongs to; ``seed`` draws the payloads of the recall world's distractors."""
    recall = condition in RECALL_CONDITIONS

    def marker(index: int, payload: np.ndarray | None = None) -> np.ndarray:
        x = np.zeros((1, WORLDS["recall"][0] if recall else INPUTS))
        x[0, index] = 1.0
        if payload is not None:
            x[0, PAYLOAD] = payload
        return x

    if recall:
        rng = np.random.default_rng(seed)
        kind, _, count = condition.partition("-")
        if kind in ("clean", "distractor"):
            told = [marker(WRITE, VALUES[value])]
            for _ in range(int(count)):
                told.append(marker(DISTRACTOR, rng.uniform(-0.25, 0.25, 4))
                            if kind == "distractor" else marker(NEUTRAL))
        elif condition == "replacement":
            told = [marker(WRITE, VALUES[1 - value]), marker(NEUTRAL),
                    marker(WRITE, VALUES[value]), marker(NEUTRAL)]
        else:  # order-latest
            told = [marker(WRITE, VALUES[1 - value]), marker(WRITE, VALUES[value]),
                    marker(NEUTRAL)]
        return [marker(START), *told, marker(QUERY)]
    first = condition.startswith("ask-first")
    writes = (VALUES[value], VALUES[1 - value]) if first else (VALUES[1 - value], VALUES[value])
    gap = [marker(NEUTRAL)] if condition.endswith("-gap") else []
    return [marker(START), marker(WRITE, writes[0]), *gap, marker(WRITE, writes[1]),
            marker(NEUTRAL), marker(QUERY_FIRST if first else QUERY)]


def schedule(
    episodes: int, rng: np.random.Generator, world: str = "order"
) -> list[tuple[str, int, int]]:
    """Balanced blocks: every condition of the world with both values once per block,
    shuffled, each episode with its seed: in the recall world drawn after the shuffle, in
    block order, as the Stone Age schedule draws it; 0 in the ordered world, which draws
    none."""
    out: list[tuple[str, int, int]] = []
    while len(out) < episodes:
        block = [(c, v) for c in WORLDS[world][1] for v in (0, 1)]
        rng.shuffle(block)
        out.extend((c, v, int(rng.integers(2**31)) if world == "recall" else 0)
                   for c, v in block)
    return out[:episodes]


def make_brain(
    arm: str, founder: int, settings: dict[str, Any] | None = None, world: str = "order"
) -> Any:
    import cadence as cd

    return cd.Brain.compose(WORLDS[world][0], ACTIONS, seed=int(founder),
                            **{"backend": "cpu", "recall": False,
                               **{**BRAIN, **ARMS[arm], **(settings or {})}})


def read(brain: Any, founder: int, greedy: bool, world: str = "order") -> dict[str, Any]:
    """The success share of a fork's acts on the evaluation episodes, per condition."""
    fork = copy.deepcopy(brain)
    cells: dict[str, list[int]] = {c: [0, 0] for c in WORLDS[world][1]}
    rng = np.random.default_rng(np.random.SeedSequence([founder, 632]))
    for condition, value, seed in schedule(EVALUATION, rng, world):
        fork.reset()
        try:
            for x in events(condition, value, seed):
                action = int(np.asarray(fork.act(x, greedy=greedy)).reshape(-1)[0])
        except RuntimeError:
            action = -1
        cells[condition][0] += int(action == value)
        cells[condition][1] += 1
    return {"success": sum(c[0] for c in cells.values()) / EVALUATION,
            "conditions": {c: round(h / n, 4) for c, (h, n) in cells.items()}}


def live(brain: Any, x: np.ndarray, outcome: tuple[float, bool] | None) -> tuple[int | None, int]:
    """One event through ``live`` with the outcome of the act before it, and the refusals it
    met. A refused answer is retried once without the outcome, which a refusal after the
    outcome was taken keeps learned; the action is None when the retry is refused too."""
    try:
        if outcome is None:
            return int(np.asarray(brain.live(x)).reshape(-1)[0]), 0
        out = brain.live(x, reward=np.array([outcome[0]]), done=np.array([outcome[1]]))
        return int(np.asarray(out).reshape(-1)[0]), 0
    except RuntimeError:
        pass
    try:
        return int(np.asarray(brain.live(x)).reshape(-1)[0]), 1
    except RuntimeError:
        return None, 2


def taught(brain: Any, condition: str, value: int, seed: int = 0) -> int | None:
    """One episode taught by a lesson at the query, on a fresh stream; the act after it, None
    when an act or the lesson is refused."""
    brain.reset()
    try:
        steps = events(condition, value, seed)
        for x in steps[:-1]:
            brain.act(x, greedy=True)
        brain.lesson(steps[-1], np.array([value]))
        return int(np.asarray(brain.act(steps[-1], greedy=True)).reshape(-1)[0])
    except RuntimeError:
        return None


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``spec`` names ``arm`` and ``founder``, with ``world``, ``teach``,
    ``settings`` over the arm's and ``budgets``. Under reward a life whose act is refused twice
    in a row ends there (``ended``); under lessons a refused episode counts as a miss
    (``refused``)."""
    warnings.simplefilter("ignore")
    import cadence as cd

    arm, founder = spec["arm"], int(spec["founder"])
    world = spec.get("world", "order")
    teach = spec.get("teach", "reward")
    budgets = sorted(int(b) for b in spec.get("budgets", BUDGETS))
    started = time.monotonic()
    brain = make_brain(arm, founder, spec.get("settings"), world)
    plan = schedule(
        budgets[-1], np.random.default_rng(np.random.SeedSequence([founder, 236])), world
    )
    readings, paid, refused, ended = [], [], 0, None
    outcome: tuple[float, bool] | None = None
    for episode, (condition, value, seed) in enumerate(plan, start=1):
        if teach == "lesson":
            action = taught(brain, condition, value, seed)
            refused += int(action is None)
            paid.append(int(action == value))
        else:
            for x in events(condition, value, seed):
                action, refusals = live(brain, x, outcome)
                refused += refusals
                if action is None:
                    break
                outcome = (0.0, False)
            if action is None:
                ended = episode
                break
            outcome = (RIGHT if action == value else WRONG, True)
            paid.append(int(action == value))
        if episode in budgets:
            readings.append({"episodes": episode, "greedy": read(brain, founder, True, world),
                             "sampled": read(brain, founder, False, world),
                             "seconds": round(time.monotonic() - started, 1)})
    return {"world": world, "teach": teach, "arm": arm, "founder": founder,
            "settings": {**ARMS[arm], **spec.get("settings", {})},
            "budgets": budgets, "cadence": cd.__version__, "refused": refused, "ended": ended,
            "training": [round(float(np.mean(paid[i:i + WINDOW])), 4)
                         for i in range(0, len(paid), WINDOW)],
            "readings": readings, "seconds": round(time.monotonic() - started, 1)}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per world, way of teaching, arm and budget: the founders' greedy and sampled success,
    least, mean, greatest."""
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r.get("world", "order"), r.get("teach", "reward"), r["arm"])
        cells.setdefault(key, []).append(r)
    out = []
    for (world, teach, arm), group in sorted(cells.items()):
        for i, budget in enumerate(group[0]["budgets"]):
            cell = {"world": world, "teach": teach, "arm": arm, "episodes": budget,
                    "founders": sorted(r["founder"] for r in group)}
            for mode in ("greedy", "sampled"):  # a life that ended before the budget reads 0
                values = [r["readings"][i][mode]["success"] if len(r["readings"]) > i else 0.0
                          for r in group]
                cell[mode] = [round(min(values), 3), round(float(np.mean(values)), 3),
                              round(max(values), 3)]
            cell["refused"] = sum(r["refused"] for r in group)
            out.append(cell)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arms", nargs="*", default=["founder"], choices=list(ARMS))
    parser.add_argument("--world", choices=list(WORLDS), default="order")
    parser.add_argument("--teach", choices=TEACH, default="reward")
    parser.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    parser.add_argument("--settings", type=json.loads, default={}, help="over the arms'")
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
    specs = [{"arm": arm, "founder": founder, "world": args.world, "teach": args.teach,
              "settings": args.settings, "budgets": args.budgets}
             for arm in args.arms for founder in args.founders]
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
