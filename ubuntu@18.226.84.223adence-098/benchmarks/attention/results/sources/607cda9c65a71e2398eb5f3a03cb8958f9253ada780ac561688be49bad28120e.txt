"""Several facts in one settle: held memory-read heads (``recall_heads``).

Each trial's recall bank holds four witnessed episodes ``neutral, role_j, value_vj``
(j = 0..3), the values drawn fresh per trial. The question ``neutral, ASK`` is answered in one
settle, in one of two tasks:

``report``: the values of roles 0 and 1 in two motor slots (six values each);
``flee``: one action of ten positions: the food's position (role 0) when the predator (role 1)
stands at an even position, else the position five steps from the predator.

Roles 2 and 3 are distractors. Both orders of the value sets {0, 1}, {2, 3}, ... for roles
(0, 1) are never taught (held out). Two held heads (``Brain.compose(..., learned_code=16,
attention=True, recall_heads=2)``) are compared with one head that learns its address at the
same settings, at the same lessons and readings, and with uniform random. Lessons teach every
row while the running share of the brain's own wrong answers is at least 0.02 and only the
wrong rows below it (``learning_surprise``), the calm teaching of a brain that already answers
right.

This is bounded component evidence for reading several facts in one settle. A single action
combines the heads in the shared association, and on value pairs it was never taught a
reporting slot can answer the other asked value; the held-out reading is recorded, and gated
on the one-action task only.
"""

from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS, ROLES, ROLE, VALUE = 16, 4, 0, 4
CODE, MODULE = 16, 64
VALUES = {"report": 6, "flee": 10}
CALM = 0.02  # the running share of wrong answers below which only wrong rows are taught


def sources() -> list[tuple[str, Path]]:
    """The public implementation bytes this result depends on."""
    names = (
        "benchmarks/attention/held_heads.py",
        "src/cadence/generic.py",
        "src/cadence/stream.py",
        "src/cadence/code.py",
    )
    return [(name, ROOT / name) for name in names]


class Task:
    """The witnessed banks, the question and the answers of one task."""

    def __init__(self, rule: str):
        self.rule, self.values = rule, VALUES[rule]
        self.neutral, self.ask = VALUE + self.values, VALUE + self.values + 1
        self.slots = 2 if rule == "report" else 1
        self.actions = self.slots * self.values
        self.held_out = {
            (a, b) for i in range(0, self.values - 1, 2) for a, b in ((i, i + 1), (i + 1, i))
        }

    def event(self, index: int, n: int = 1) -> np.ndarray:
        x = np.zeros((n, INPUTS))
        x[:, index] = 1.0
        return x

    def labels(self, patterns: list[tuple[int, ...]]) -> np.ndarray:
        if self.rule == "report":
            return np.array([[p[0], p[1]] for p in patterns], dtype=np.int64)
        return np.array(
            [p[0] if p[1] % 2 == 0 else (p[1] + 5) % self.values for p in patterns],
            dtype=np.int64,
        )

    def sample(self, rng: np.random.Generator, n: int, held_out: bool) -> list[tuple[int, ...]]:
        out: list[tuple[int, ...]] = []
        while len(out) < n:
            p = tuple(int(v) for v in rng.integers(self.values, size=ROLES))
            if (p[:2] in self.held_out) == held_out:
                out.append(p)
        return out

    def compose(self, seed: int, heads: int) -> cd.Brain:
        """The opt-in recipe; one head learns its address at the heads' settings. The protocol
        predates the recall position term (0.89.0), which it leaves out."""
        return cd.Brain.compose(
            INPUTS, self.actions, learned_code=CODE, attention=True, recall_heads=heads,
            recall_address=0.01, recall_temperature=0.25, recall_position=0, modules=(MODULE,),
            slots=self.slots, seed=seed, episodic=False, teaching="normalized",
            learning_qualified=True, learning_damping=3, learning_free_steps=1024,
            learning_nudged_steps=1024, learning_tolerance=0.003,
        )

    def warm(self, brain: cd.Brain, rounds: int = 16) -> None:
        """The learned code meets every event, then stops learning."""
        events = [*range(ROLES), *(VALUE + v for v in range(self.values)), self.neutral, self.ask]
        for _ in range(rounds):
            for index in events:
                brain.reset(keep=())
                brain.act(self.event(index), greedy=True)
        assert brain.learned_code is not None
        brain.learned_code.learning = False
        brain.reset(keep=())

    def banks(self, brain: cd.Brain) -> dict[tuple[int, ...], dict[str, np.ndarray]]:
        """Every pattern's witnessed slots, written by the brain itself in one batched pass."""
        patterns = list(itertools.product(range(self.values), repeat=ROLES))
        n, recall = len(patterns), brain.recall
        assert recall is not None
        brain.reset(keep=())
        for j in range(ROLES):
            if j:
                brain.reset(keep=("slots",))
            brain.act(self.event(self.neutral, n), greedy=True)
            brain.act(self.event(ROLE + j, n), greedy=True)
            values = np.zeros((n, INPUTS))
            values[np.arange(n), VALUE + np.array([p[j] for p in patterns])] = 1.0
            brain.act(values, greedy=True)
        names = ("keys", "values", "nexts", "held")
        out = {
            p: {name: getattr(recall, name)[i:i + 1].copy() for name in names}
            for i, p in enumerate(patterns)
        }
        brain.reset(keep=())
        return out

    def install(self, brain: cd.Brain, banks: dict[Any, Any], patterns: list[Any]) -> None:
        recall = brain.recall
        assert recall is not None
        brain.reset(keep=())
        for name in ("keys", "values", "nexts", "held"):
            setattr(recall, name, np.concatenate([banks[p][name] for p in patterns]).copy())
        n = len(patterns)
        recall.previous = np.zeros((n, len(recall.association)))
        recall.past = np.zeros((n, recall.window - 1, recall.key_width))
        recall.ended = np.zeros(n, dtype=bool)

    def answer(self, brain: cd.Brain, banks: dict[Any, Any], patterns: list[Any]) -> np.ndarray:
        n = len(patterns)
        self.install(brain, banks, patterns)
        brain.act(self.event(self.neutral, n), greedy=True)
        return np.asarray(brain.act(self.event(self.ask, n), greedy=True)).reshape(n, self.slots)

    def teach(self, brain: cd.Brain, banks: dict[Any, Any], patterns: list[Any]) -> None:
        n = len(patterns)
        self.install(brain, banks, patterns)
        brain.act(self.event(self.neutral, n), greedy=True)
        labels = self.labels(patterns)
        brain.lesson(self.event(self.ask, n), labels if self.slots > 1 else labels[:, None][:, 0])


def train(
    task: Task, brain: cd.Brain, banks: dict[Any, Any], lessons: int, seed: int
) -> dict[str, Any]:
    """Lessons of 16 fresh trials of taught pairs, calm once the brain answers right."""
    rng = np.random.default_rng(1000 + seed)
    level, accepted, refused, calm_from = 1.0, 0, 0, None
    for lesson in range(lessons):
        patterns = task.sample(rng, 16, held_out=False)
        try:
            wrong = task.answer(brain, banks, patterns) != task.labels(patterns).reshape(16, -1)
            level = 0.95 * level + 0.05 * float(wrong.any(axis=1).mean())
            calm = level < CALM
            if calm != brain.learner.config.surprise:
                brain.retune(learning_surprise=calm)
                if calm and calm_from is None:
                    calm_from = lesson
            task.teach(brain, banks, patterns)
            accepted += 1
        except RuntimeError:
            refused += 1
    return {"lessons": lessons, "accepted": accepted, "refused": refused, "calm_from": calm_from}


def evaluate(
    task: Task, brain: cd.Brain, banks: dict[Any, Any], *, held_out: bool, n: int = 96
) -> dict[str, Any]:
    """All-slot accuracy on fresh trials, the wrong slot answers that give another asked
    role's value, and which role record each head attends most."""
    trial = deepcopy(brain)
    patterns = task.sample(np.random.default_rng(99), n, held_out)
    try:
        choice = task.answer(trial, banks, patterns)
    except RuntimeError as error:
        return {"refused": str(error)[:80]}
    labels = task.labels(patterns).reshape(n, -1)
    correct = (choice == labels).all(axis=1)
    wrong = choice != labels
    other = np.zeros_like(wrong)
    if task.slots == 2:  # a wrong slot that answers the other asked role's value
        other = wrong & (choice == labels[:, ::-1])
    recall = trial.recall
    assert recall is not None
    heads = recall.attended_heads or ()
    role_records = [3 * j + 1 for j in range(ROLES)]
    tops = [np.asarray(p)[:, role_records].argmax(axis=1) for p in heads if p is not None]
    distinct = (
        float(np.mean([len({int(t[i]) for t in tops}) == len(tops) for i in range(n)]))
        if len(tops) > 1 else None
    )
    return {
        "accuracy": float(correct.mean()),
        "wrong_slots": int(wrong.sum()),
        "other_asked_value": int(other.sum()),
        "refusals": 0,
        "head_top_role": [int(np.bincount(t, minlength=ROLES).argmax()) for t in tops],
        "distinct_top": distinct,
    }


def run_founder(seed: int, rule: str, *, lessons: int = 1024) -> dict[str, Any]:
    task = Task(rule)
    arms: dict[str, Any] = {}
    for heads in (2, 1):
        brain = task.compose(seed, heads)
        task.warm(brain)
        banks = task.banks(brain)
        training = train(task, brain, banks, lessons, seed)
        arms[f"heads_{heads}"] = {
            "training": training,
            "trained": evaluate(task, brain, banks, held_out=False),
            "held_out": evaluate(task, brain, banks, held_out=True),
        }
    random = 1.0 / task.values ** task.slots
    two, one = arms["heads_2"], arms["heads_1"]
    gates = {
        "training_completed": two["training"]["refused"] == 0,
        "trained_accuracy": two["trained"].get("accuracy", 0.0) >= 0.95,
        "beats_one_head": two["trained"].get("accuracy", 0.0)
        - one["trained"].get("accuracy", 0.0) >= 0.2,
        "beats_uniform_random": two["trained"].get("accuracy", 0.0) - random >= 0.2,
        "distinct_heads": (two["trained"].get("distinct_top") or 0.0) >= 0.95,
    }
    if rule == "flee":
        gates["held_out_accuracy"] = two["held_out"].get("accuracy", 0.0) >= 0.9
    return {
        "seed": seed, "rule": rule, "values": task.values, "arms": arms,
        "uniform_random_baseline_accuracy": random, "gates": gates,
        "passed": all(gates.values()),
    }


def check(body: dict[str, Any]) -> str | None:
    """Recompute the retained pass decisions from their recorded readings."""
    try:
        if body["schema"] != "cadence-held-heads/1" or len(body["founders"]) < 2:
            return "wrong schema or too few founders"
        for founder in body["founders"]:
            task = Task(founder["rule"])
            if founder["uniform_random_baseline_accuracy"] != 1.0 / task.values ** task.slots:
                return f"founder {founder['seed']} ({founder['rule']}) has a wrong baseline"
            two, one = founder["arms"]["heads_2"], founder["arms"]["heads_1"]
            accuracy = two["trained"].get("accuracy", 0.0)
            expected = {
                "training_completed": two["training"]["refused"] == 0,
                "trained_accuracy": accuracy >= 0.95,
                "beats_one_head": accuracy - one["trained"].get("accuracy", 0.0) >= 0.2,
                "beats_uniform_random": accuracy
                - founder["uniform_random_baseline_accuracy"] >= 0.2,
                "distinct_heads": (two["trained"].get("distinct_top") or 0.0) >= 0.95,
            }
            if founder["rule"] == "flee":
                expected["held_out_accuracy"] = two["held_out"].get("accuracy", 0.0) >= 0.9
            if founder["gates"] != expected or founder["passed"] != all(expected.values()):
                return f"founder {founder['seed']} ({founder['rule']}) decisions do not match"
        if body["passed"] != all(founder["passed"] for founder in body["founders"]):
            return "aggregate decision does not match founder decisions"
    except (KeyError, TypeError, ValueError):
        return "incomplete or invalid held-heads readings"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--rules", nargs="+", default=["report", "flee"])
    parser.add_argument("--lessons", type=int, default=1024)
    parser.add_argument("--founder-out", type=Path, help="write one founder's readings (JSON)")
    parser.add_argument("--merge", nargs="+", type=Path, help="founder readings to combine")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.merge:
        founders = [json.loads(path.read_text()) for path in args.merge]
    else:
        if args.founder_out is not None and len(args.seeds) * len(args.rules) != 1:
            raise SystemExit("--founder-out writes one founder: pass one seed and one rule")
        founders = [run_founder(s, r, lessons=args.lessons) for r in args.rules for s in args.seeds]
        if args.founder_out is not None:
            args.founder_out.write_text(json.dumps(founders[0], sort_keys=True, indent=2) + "\n")
            return
    result = {
        "schema": "cadence-held-heads/1",
        "scope": "several facts in one settle with held heads; bounded component evidence",
        "configuration": {
            "inputs": INPUTS, "learned_code": CODE, "modules": [MODULE], "lessons": args.lessons,
            "recall_heads": 2, "recall_window": 2, "recall_temperature": 0.25,
            "recall_address": 0.01, "recall_recency": 0.0, "recall_position": 0, "calm_below": CALM,
            "values": VALUES,
        },
        "founders": founders,
        "passed": len(founders) >= 2 and all(founder["passed"] for founder in founders),
    }
    problem = check(result)
    if problem is not None:
        raise AssertionError(problem)
    if args.out is not None:
        cd.Receipt.build("cadence-held-heads/1", result, sources()).write(args.out)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
