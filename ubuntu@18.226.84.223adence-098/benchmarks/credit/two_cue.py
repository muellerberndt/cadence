"""The two-cue chamber: a cue kept to an identical answer (cadence#208, cadence#209).

A cue is shown once, then ``delay`` distractor events, then the suffix ``S1``, ``S2``. The
suffix is the same for both cues of a pair; the right action at ``S2`` depends on the cue
alone. Twelve one-hot inputs: four cue tokens ``C0`` to ``C3`` (inputs 0 to 3; the pair
``C0``, ``C1`` is the one trained here, ``C2``, ``C3`` a second pair the conditions never show),
six distractors ``D0`` to ``D5`` (inputs 4 to 9), ``S1`` (10) and ``S2`` (11). Two actions:
within a pair the two cues take the two actions, the assignment counterbalanced across
founders (bit 0 of the founder number flips the first pair, bit 1 the second).

An episode at delay ``k`` is ``k + 3`` events. The training schedule takes both cues of the
pair in a random order in every block of two episodes, and draws each episode's distractors
independently and uniformly from the tuples permitted in training. At delays 2 and more, one
distractor tuple in four (a founder-salted digest of the tuple) is never trained; the
held-out panel reads those tuples. At delays 0 and 1 no tuple can be held out without holding
out a token, so the held-out panel is the training distribution itself (``nothing_held_out``).

Two information conditions, one life per founder, arm, condition and delay:

- ``reward``: one continuing stream through ``step``, every act the library's sampled act,
  zero reward within the episode; the answer's outcome, +1 right and -1 wrong, arrives with
  the next episode's first event together with ``done``. Every outcome is learned from. A
  reading at budget ``n`` has learned from ``n - 1`` outcomes.
- ``lesson``: a fresh stream per episode (``reset``); every event before the answer is a
  greedy act; at ``S2`` one ``step(x, teacher=[label])``, whose own act is the library's
  sampled act. Nothing else teaches.

The brain is ``Brain.compose(12, 2, seed=founder, arousal=True)`` at the composed defaults,
with the arm's settings beside it (``ARMS``). At 256, 1,024 and 2,048 episodes the train and
held-out panels are read on a fork of the life that never changes it: each panel episode on
a fresh stream of the fork (``reset``, the records kept), every event a greedy act, the answer
the greedy act at ``S2``. A setting learns a delay when held-out accuracy at 2,048 episodes is
at least .90 on at least 5 of the 6 development founders (0 to 5). At the last budget the
held-out panel is also read with carriers (``carriers``) emptied just before the answer
(``erase``), or with a donor's in place of the stream's own (``transplant``).

    python benchmarks/credit/two_cue.py --condition reward --arms founder slow_episodic \\
        --delays 0 1 2 --out results/reward.jsonl --workers 3
    python benchmarks/credit/two_cue.py --summary results/reward.jsonl
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

PAIRS = ((0, 1), (2, 3))
DISTRACTOR = 4  # D0 is input 4
DISTRACTORS = 6
S1, S2 = 10, 11
INPUTS, ACTIONS = 12, 2
DELAYS = (0, 1, 2, 4, 8)
HOLDOUT_MODULUS = 4  # one tuple in four is held out at delays 2 and more
PANEL = 64  # tuples drawn for a panel whose kind has more than EXHAUSTIVE tuples
EXHAUSTIVE = 256
BUDGETS = (256, 1024, 2048)
WINDOW = 64  # episodes per entry of the training log
RIGHT, WRONG = 1.0, -1.0
FOUNDERS = tuple(range(6))
CONFIRMATION = tuple(range(500, 512))
GATE = {"held_out": 0.9, "founders": 5}
SEEDS = {"schedule": 51018, "holdout": 51019, "panel": 51020}
CONDITIONS = ("reward", "lesson")
SLOW = {"actor_eta": 0.1, "actor_eta_bias": 0.01}  # the actor step of cadence#198
ARMS: dict[str, dict[str, Any]] = {
    "founder": {},
    "slow": dict(SLOW),
    "slow_episodic": {**SLOW, "memory_choice": True},
    "slow_episodic_extinction": {**SLOW, "memory_choice": True, "memory_extinction": 0.5},
    "slow_episodic_hold": {**SLOW, "memory_choice": True, "memory_context_hold": "unchanged"},
    "recall": {"recall": True, "recall_founder": True},
    "recall_recency0": {"recall": True, "recall_founder": True, "recall_recency": 0.0},
    "recall_recency0_boundary": {"recall": True, "recall_founder": True, "recall_recency": 0.0,
                                 "recall_boundary": True},
}


# ------------------------------------------------------------------------------------ task


def label_of(cue: int, founder: int) -> int:
    """The right action for a cue: its place in its pair, flipped by the founder's bit of
    that pair."""
    pair, place = divmod(int(cue), 2)
    return place ^ ((int(founder) >> pair) & 1)


def events_of(cue: int, distractors: tuple[int, ...]) -> np.ndarray:
    """The episode's one-hot events, one row each: the cue, the distractors, ``S1``, ``S2``."""
    tokens = [int(cue), *(DISTRACTOR + int(d) for d in distractors), S1, S2]
    return np.eye(INPUTS)[tokens]


def held_out(founder: int, distractors: tuple[int, ...]) -> bool:
    """Whether a distractor tuple is never trained: one in ``HOLDOUT_MODULUS`` at delays 2
    and more, by a digest of the tuple salted with the founder."""
    if len(distractors) < 2:
        return False
    text = f"{SEEDS['holdout']}:{int(founder)}:{','.join(str(int(d)) for d in distractors)}"
    digest = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return digest % HOLDOUT_MODULUS == 0


def _draw(rng: np.random.Generator, founder: int, delay: int, kind: str) -> tuple[int, ...]:
    """One distractor tuple of the kind (``train`` or ``held_out``), uniform over its kind."""
    while True:
        tuple_ = tuple(int(d) for d in rng.integers(DISTRACTORS, size=delay))
        if delay < 2 or held_out(founder, tuple_) == (kind == "held_out"):
            return tuple_


def schedule(founder: int, delay: int, episodes: int, pair: int = 0) -> list[tuple[int, tuple]]:
    """The life's episodes: both cues of the pair in a random order in every block of two,
    each with its own training tuple."""
    rng = np.random.default_rng(np.random.SeedSequence([SEEDS["schedule"], founder, delay, pair]))
    out: list[tuple[int, tuple]] = []
    while len(out) < episodes:
        for cue in rng.permutation(PAIRS[pair]):
            out.append((int(cue), _draw(rng, founder, delay, "train")))
    return out[:episodes]


def panel(founder: int, delay: int, kind: str, pair: int = 0) -> tuple[list[tuple], bool]:
    """A reading panel and whether nothing is held out at this delay. Every tuple of the kind
    where it has at most ``EXHAUSTIVE`` of them, else ``PANEL`` distinct tuples drawn from the
    panel seed; each tuple followed by both cues of the pair."""
    if delay < 2:
        tuples = [tuple(t) for t in itertools.product(range(DISTRACTORS), repeat=delay)]
        return [(cue, t) for t in tuples for cue in PAIRS[pair]], True
    wanted = kind == "held_out"
    if DISTRACTORS**delay <= 4096:
        every = [
            tuple(t)
            for t in itertools.product(range(DISTRACTORS), repeat=delay)
            if held_out(founder, tuple(t)) == wanted
        ]
        if len(every) <= EXHAUSTIVE:
            tuples = every
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence([SEEDS["panel"], founder, delay, int(wanted)])
            )
            tuples = [every[i] for i in sorted(rng.choice(len(every), PANEL, replace=False))]
    else:
        rng = np.random.default_rng(
            np.random.SeedSequence([SEEDS["panel"], founder, delay, int(wanted)])
        )
        chosen: set[tuple[int, ...]] = set()
        while len(chosen) < PANEL:
            chosen.add(_draw(rng, founder, delay, kind))
        tuples = sorted(chosen)
    return [(cue, t) for t in tuples for cue in PAIRS[pair]], False


# ----------------------------------------------------------------------------------- brain


def make_brain(arm: str, founder: int, settings: dict[str, Any] | None = None) -> Any:
    """``Brain.compose(12, 2, seed=founder, arousal=True)`` with the arm's settings, and any
    ``settings`` over them."""
    import cadence as cd

    return cd.Brain.compose(
        INPUTS, ACTIONS, seed=int(founder), arousal=True,
        **{"backend": "cpu", "recall": False, **{**ARMS[arm], **(settings or {})}},
    )


def _vote_one(brain: Any, x: np.ndarray, *, before: bool) -> float | None:
    """The episodic records' preference for action 1 over action 0 at the observation ``x``,
    under the key of a decision about to be taken on it (``before``) or of the stream's last
    decision; None without the episodic vote."""
    records = brain.memories.get("episodic")
    if records is None:
        return None
    code = brain.observation_code(x[None, :])
    key = records.key(code) if before else records.decided(1)
    vote = np.asarray(records.vote(code, key), dtype=float)[0]
    return float(vote[1] - vote[0])


def _policy(brain: Any, x: np.ndarray) -> tuple[float, float]:
    """After an act on ``x``: the readout's probability of action 1 and the complete choice's
    (the readout with the episodic vote of that decision, where the brain has one)."""
    state = brain.basal_ganglia.state
    readout = float(np.asarray(brain.basal_ganglia.probabilities(state))[0, 1])
    records = brain.memories.get("episodic")
    if records is None:
        return readout, readout
    vote = records.vote(brain.observation_code(x[None, :]), records.decided(1))
    full = float(np.asarray(brain.basal_ganglia.probabilities(state, bias=vote))[0, 1])
    return readout, full


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(1e-12, 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))))


# ------------------------------------------------------------------------------- carriers


ACTIVITY = "activity"  # the stream's settled activity, from which the next settle starts


def carriers(name: str) -> tuple[str, ...]:
    """The carriers a reading names, joined by ``+``: members of ``brain.memories``, and
    ``activity``, the stream's settled state from which the answer's settle starts."""
    return tuple(part for part in name.split("+") if part)


def carries(brain: Any, name: str) -> bool:
    """Whether the brain has every carrier ``name`` joins."""
    return all(part == ACTIVITY or part in brain.memories for part in carriers(name))


def empty(brain: Any, name: str) -> None:
    """Empty the stream of every carrier ``name`` joins: a memory's stream (``reset``: the
    trace, or the slots and the cortex of the moment before), and for ``activity`` the
    settled state, so the next settle starts from rest."""
    for part in carriers(name):
        if part == ACTIVITY:
            brain.basal_ganglia.reset()
        else:
            brain.memories[part].reset(1)


def carried(brain: Any, name: str) -> dict[str, Any]:
    """A copy of the stream state of every carrier ``name`` joins: a memory's ``snapshot``
    where it keeps one (the recall memory's slots and cortex of the moment before), else its
    ``arrays`` (the working trace), and for ``activity`` the settled state."""
    out: dict[str, Any] = {}
    for part in carriers(name):
        if part == ACTIVITY:
            out[part] = copy.deepcopy(brain.basal_ganglia.state)
        else:
            memory = brain.memories[part]
            kept = memory.snapshot() if hasattr(memory, "snapshot") else memory.arrays()
            out[part] = copy.deepcopy(kept)
    return out


def carry(brain: Any, state: dict[str, Any]) -> None:
    """Put a ``carried`` state into the brain's stream, in place of its own."""
    for part, value in state.items():
        if part == ACTIVITY:
            # the warm start of the next settle (``Brain.act`` settles from the actor's state)
            brain.basal_ganglia._free = copy.deepcopy(value)
            continue
        memory = brain.memories[part]
        if hasattr(memory, "restore"):
            memory.restore(copy.deepcopy(value))
        else:
            for field, array in value.items():
                setattr(memory, field, np.array(array, copy=True))


def read_panel(
    brain: Any,
    founder: int,
    delay: int,
    kind: str,
    *,
    erase: str | None = None,
    pair: int = 0,
) -> dict[str, Any]:
    """Greedy answers on one fork of the life, each panel episode on a fresh stream of it.

    ``erase`` names carriers (``carriers``) whose streams are emptied just before the
    answer's act (``working_memory``: the trace; ``recall``: the slots; ``episodic``: the
    temporal context; ``working_memory+recall``: both). Reported: accuracy, the share
    answering action 1 and the readout's probability of action 1 per cue, the episodic vote's
    preference for action 1 per cue, and the relative separation of the two cues' association
    activity after every event, matched by tuple."""
    fork = copy.deepcopy(brain)
    episodes, nothing = panel(founder, delay, kind, pair)
    association = np.asarray(fork.association_index, dtype=np.int64)
    right, answers = [], {cue: [] for cue in PAIRS[pair]}
    readout: dict[int, list[float]] = {cue: [] for cue in PAIRS[pair]}
    votes: dict[int, list[float]] = {cue: [] for cue in PAIRS[pair]}
    activity: dict[tuple, dict[int, list[np.ndarray]]] = {}
    for cue, tuple_ in episodes:
        events = events_of(cue, tuple_)
        fork.reset()
        trail = []
        for x in events[:-1]:
            fork.act(x[None, :], greedy=True)
            trail.append(np.asarray(fork.basal_ganglia.state.activation)[0, association].copy())
        if erase is not None:
            empty(fork, erase)
        vote = _vote_one(fork, events[-1], before=True)
        answer = int(fork.act(events[-1][None, :], greedy=True)[0])
        trail.append(np.asarray(fork.basal_ganglia.state.activation)[0, association].copy())
        readout[cue].append(_policy(fork, events[-1])[0])
        if vote is not None:
            votes[cue].append(vote)
        answers[cue].append(answer)
        right.append(answer == label_of(cue, founder))
        activity.setdefault(tuple_, {})[cue] = trail
    first, second = PAIRS[pair]
    pairs = [v for v in activity.values() if first in v and second in v]
    separation = [
        float(np.mean([_relative(v[first][e], v[second][e]) for v in pairs]))
        for e in range(delay + 3)
    ]
    return {
        "accuracy": float(np.mean(right)),
        "episodes": len(right),
        "nothing_held_out": nothing,
        "choice_one": {str(c): float(np.mean(a)) for c, a in answers.items()},
        "p_one": {str(c): float(np.mean(p)) for c, p in readout.items()},
        "vote_one": {str(c): float(np.mean(v)) for c, v in votes.items() if v} or None,
        "separation": separation,
    }


def read_transplant(
    brain: Any, founder: int, delay: int, kind: str, name: str, *, pair: int = 0
) -> dict[str, Any]:
    """Greedy answers on one fork of the life with a donor's carriers: for every panel
    episode (the recipient), a fresh stream of the fork lives the episode of the other cue
    with the same distractors (the donor) up to the answer, and the stream state of the
    carriers ``name`` joins is kept (``carried``); a fresh stream then lives the recipient's
    episode up to the answer, the donor's state replaces its own (``carry``), and the answer
    is the greedy act at ``S2``. Reported: the share of answers that follow the donor's cue
    (its label) and the accuracy against the recipient's own."""
    fork = copy.deepcopy(brain)
    episodes, _ = panel(founder, delay, kind, pair)
    first, second = PAIRS[pair]
    follows, right = [], []
    for cue, tuple_ in episodes:
        donor = second if cue == first else first
        state = None
        for shown in (donor, cue):
            events = events_of(shown, tuple_)
            fork.reset()
            for x in events[:-1]:
                fork.act(x[None, :], greedy=True)
            if shown == donor:
                state = carried(fork, name)
        assert state is not None
        carry(fork, state)
        answer = int(fork.act(events[-1][None, :], greedy=True)[0])
        follows.append(answer == label_of(donor, founder))
        right.append(answer == label_of(cue, founder))
    return {
        "follows_donor": float(np.mean(follows)),
        "accuracy": float(np.mean(right)),
        "episodes": len(follows),
    }


# ------------------------------------------------------------------------------------- life


def _windows(values: list[Any], window: int = WINDOW) -> list[float | None]:
    out: list[float | None] = []
    for start in range(0, len(values), window):
        chunk = [v for v in values[start : start + window] if v is not None]
        out.append(float(np.mean(chunk)) if chunk else None)
    return out


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``spec`` names ``condition``, ``arm``, ``founder`` and ``delay``, with
    ``budgets`` (default ``BUDGETS``), ``settings`` over the arm's, ``erase`` (carriers
    erased, one reading each, on the held-out panel at the last budget) and ``transplant``
    (carriers taken from a donor, one reading each, on the same panel); a name the brain has
    no carrier for is skipped."""
    warnings.simplefilter("ignore")
    import cadence as cd

    condition, arm = spec["condition"], spec["arm"]
    founder, delay = int(spec["founder"]), int(spec["delay"])
    budgets = sorted(int(b) for b in spec.get("budgets", BUDGETS))
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    started, cpu = time.monotonic(), time.process_time()
    row: dict[str, Any] = {
        "condition": condition, "arm": arm, "founder": founder, "delay": delay,
        "settings": {**ARMS[arm], **spec.get("settings", {})}, "budgets": budgets,
        "labels": {str(c): label_of(c, founder) for c in PAIRS[0]},
        "cadence": cd.__version__,
    }
    brain = make_brain(arm, founder, spec.get("settings"))
    row["genes"] = {
        k: v for k, v in brain.describe()["genes"].items()
        if k.startswith(("actor_", "learning_surprise", "memory_", "recall_", "working_memory"))
    }
    work = {"episodes": 0, "steps": 0, "lessons": 0}
    right: list[int] = []
    greedy_right: list[int | None] = []
    explored_right: list[int | None] = []
    explored: list[int] = []
    readings: list[dict[str, Any]] = []
    owed: float | None = None
    try:
        for index, (cue, tuple_) in enumerate(schedule(founder, delay, budgets[-1])):
            events, label = events_of(cue, tuple_), label_of(cue, founder)
            if condition == "reward":
                for j, x in enumerate(events):
                    if j == 0 and owed is not None:
                        action = brain.step(x[None, :], reward=[owed], done=[True])
                    elif j == 0:
                        action = brain.step(x[None, :])
                    else:
                        action = brain.step(x[None, :], reward=[0.0])
                    work["steps"] += 1
                answer = int(action[0])
                full = _policy(brain, events[-1])[1]
                greedy = int(full > 0.5)
                owed = RIGHT if answer == label else WRONG
                right.append(int(answer == label))
                explored.append(int(answer != greedy))
                greedy_right.append(int(answer == label) if answer == greedy else None)
                explored_right.append(int(answer == label) if answer != greedy else None)
            else:
                brain.reset()
                for x in events[:-1]:
                    brain.act(x[None, :], greedy=True)
                before = copy.deepcopy(brain)
                right.append(int(int(before.act(events[-1][None, :], greedy=True)[0]) == label))
                brain.step(events[-1][None, :], teacher=[label])
                work["steps"] += 1
                work["lessons"] += 1
            work["episodes"] += 1
            if index + 1 in budgets:
                reading: dict[str, Any] = {"episodes": index + 1}
                for kind in ("train", "held_out"):
                    reading[kind] = read_panel(brain, founder, delay, kind)
                if index + 1 == budgets[-1]:
                    for name in spec.get("erase", ()):
                        if carries(brain, name):
                            reading[f"{name}_erased"] = read_panel(
                                brain, founder, delay, "held_out", erase=name
                            )["accuracy"]
                    for name in spec.get("transplant", ()):
                        if carries(brain, name):
                            reading[f"{name}_transplant"] = read_transplant(
                                brain, founder, delay, "held_out", name
                            )
                reading["seconds"] = round(time.monotonic() - started, 2)
                readings.append(reading)
    except Exception as error:  # a refused act or lesson is a recorded failure of the life
        row["error"] = f"{type(error).__name__}: {error}"[:300]
    row["readings"] = readings
    row["log"] = {"window": WINDOW, "right": _windows(right)}
    if condition == "reward":
        row["log"].update(
            explored=_windows(explored),
            greedy_right=_windows(greedy_right),
            explored_right=_windows(explored_right),
        )
    final = readings[-1] if readings and readings[-1]["episodes"] == budgets[-1] else None
    held = None if final is None or "error" in row else final["held_out"]["accuracy"]
    row["verdict"] = {"held_out": held, "mastered": held is not None and held >= GATE["held_out"]}
    row["work"] = work
    row["seconds"] = round(time.monotonic() - started, 2)
    row["cpu"] = round(time.process_time() - cpu, 2)
    return row


# ---------------------------------------------------------------------------------- summary


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per condition, arm and delay: founders at the gate at every budget, and the mean
    held-out accuracy at every budget."""
    cells: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        cells.setdefault((r["condition"], r["arm"], r["delay"]), []).append(r)
    out = []
    for (condition, arm, delay), group in sorted(cells.items()):
        budgets = group[0]["budgets"]
        passing, means = [], []
        for i in range(len(budgets)):
            held = [
                r["readings"][i]["held_out"]["accuracy"] if len(r["readings"]) > i else 0.0
                for r in group
            ]
            passing.append(sum(h >= GATE["held_out"] for h in held))
            means.append(round(float(np.mean(held)), 3))
        out.append({
            "condition": condition, "arm": arm, "delay": delay, "founders": len(group),
            "budgets": budgets, "at_gate": passing, "held_out_mean": means,
            "per_founder": {str(r["founder"]): r["verdict"]["held_out"] for r in group},
            "passes": passing[-1] >= GATE["founders"] and len(group) == len(FOUNDERS),
            "errors": sum("error" in r for r in group),
        })
    return out


def _wait_for_load(gate: float | None) -> None:
    """A batch starts only while the host's one-minute load is below ``gate``, re-checked
    every five minutes while it is not."""
    while gate is not None and os.getloadavg()[0] >= gate:
        print(f"load {os.getloadavg()[0]:.1f} at or above {gate}; waiting", flush=True)
        time.sleep(300)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--arms", nargs="*", default=["founder"], choices=list(ARMS))
    parser.add_argument("--delays", nargs="*", type=int, default=[0])
    parser.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    parser.add_argument(
        "--erase", nargs="*", default=[], help="carriers erased at the end, joined by +"
    )
    parser.add_argument(
        "--transplant", nargs="*", default=[], help="carriers taken from a donor at the end"
    )
    parser.add_argument("--settings", type=json.loads, default={}, help="over the arms'")
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs in place of the grid")
    parser.add_argument("--out", type=Path, help="JSON lines, one row per life, appended")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--load-gate", type=float, default=None)
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
        if args.condition is None:
            parser.error("--condition or --plan is required")
        specs = [
            {"condition": args.condition, "arm": arm, "founder": founder, "delay": delay,
             "budgets": args.budgets, "settings": args.settings, "erase": args.erase,
             "transplant": args.transplant}
            for arm in args.arms for delay in args.delays for founder in args.founders
        ]
    _wait_for_load(args.load_gate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            results = map(run_life, specs)
            for row in results:
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
