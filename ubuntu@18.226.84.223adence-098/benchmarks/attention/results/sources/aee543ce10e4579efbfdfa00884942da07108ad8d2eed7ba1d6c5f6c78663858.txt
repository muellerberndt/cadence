"""The brain's own executed actions in its recall memory (``recall_action``).

Every observation is the same one, so only what the brain itself did tells its moments apart.
Two chambers, two actions, one stream, each bout of twelve moments beginning with ``reset``:

- **Alternate.** The answer is the opposite of the brain's own last executed action (the
  first moment's answer is action 0): A, B, A, B on identical inputs.
- **Timing.** Action 1 presses, action 0 waits. The answer is to press when the brain's own
  last two actions were both waits, else to wait: a press every third moment, at a fixed
  cadence the brain keeps by itself.

Each moment is taught (``Brain.lesson`` with the answer its own history implies, then the
greedy act that writes it), 150 bouts, at ``teaching="normalized"`` with lessons where the
answer differs (``learning_surprise=True``), ``episodic=False``, one module of 24 neurons. A
test bout runs the greedy acts alone and scores each against the answer the brain's own history
implies, so a constant policy scores near zero and uniform random 0.5. The arms: the founder
brain without a recall memory, the founder recall memory (``recall=True``), the recall memory
with ``recall_action=True``, and the efference copy (``efference_amplitude=3.0``,
``efference_decay=0.5``, beside the founder recall memory) as the existing reference for
reading one's own commands; uniform random beside them. The ``recall_action`` arm is also read
with its commands erased before every test moment (the command coordinates of every slot and the
command of the moment before set to zero), so the reading shows whether the commands do the
work.

Component evidence that the recall memory can carry the brain's own actions to a later
moment. It is not evidence for the Stone Age chain views, the composer's free play or a
continuing world; those keep their own measurements.
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS, ACTIONS = 4, 2
X = np.eye(INPUTS)[[0]]  # every observation
BOUTS, LENGTH, TESTS, CADENCE = 150, 12, 4, 3
UNIFORM = 0.5
TASKS = ("alternate", "timing")
ARMS: dict[str, dict[str, Any]] = {
    "plain": {},
    "recall": {"recall": True},
    "action": {"recall": True, "recall_action": True},
    "efference": {"recall": True, "efference_amplitude": 3.0, "efference_decay": 0.5},
}
SEEDS = (0, 1, 2, 3, 4, 5)
# every moment is a new moment; the observation repeats by design, so the library's warning
# against teaching a moment just acted on does not apply
REPEATED = "lesson on the observations the brain just acted on"


def sources() -> list[tuple[str, Path]]:
    """The public implementation bytes this result depends on."""
    names = (
        "benchmarks/attention/own_action.py",
        "src/cadence/generic.py",
        "src/cadence/stream.py",
    )
    return [(name, ROOT / name) for name in names]


def compose(seed: int, arm: str) -> cd.Brain:
    return cd.Brain.compose(
        INPUTS, ACTIONS, modules=(24,), seed=seed, episodic=False, teaching="normalized",
        learning_surprise=True, **ARMS[arm],
    )


def label(task: str, history: list[int]) -> int:
    """The answer the brain's own history implies."""
    if task == "alternate":
        return 0 if not history else 1 - history[-1]
    recent = history[-(CADENCE - 1):]
    return int(len(recent) == CADENCE - 1 and not any(recent))


def erase(brain: cd.Brain) -> None:
    """The commands erased: every slot's command coordinates and the command of the moment
    before set to zero (new arrays, as every write makes)."""
    recall = brain.recall
    width = recall.command_width
    for name in ("keys", "values", "nexts"):
        array = getattr(recall, name).copy()
        array[..., -width:] = 0.0
        setattr(recall, name, array)
    recall.acted = np.zeros_like(recall.acted)


def score(brain: cd.Brain, task: str, lesion: bool = False) -> float:
    hits = total = 0
    for _ in range(TESTS):
        brain.reset()
        history: list[int] = []
        for _ in range(LENGTH):
            if lesion:
                erase(brain)
            action = int(np.asarray(brain.act(X, greedy=True)).reshape(-1)[0])
            hits += action == label(task, history)
            total += 1
            history.append(action)
    return hits / total


def run_one(job: tuple[str, str, int]) -> dict[str, Any]:
    task, arm, seed = job
    brain = compose(seed, arm)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message=REPEATED)
        for _ in range(BOUTS):
            brain.reset()
            history: list[int] = []
            for _ in range(LENGTH):
                brain.lesson(X, np.array([label(task, history)]))
                history.append(int(np.asarray(brain.act(X, greedy=True)).reshape(-1)[0]))
    out = {"task": task, "arm": arm, "seed": seed, "score": score(brain, task)}
    if arm == "action":
        out["erased"] = score(brain, task, lesion=True)
    return out


def uniform(task: str, seed: int) -> float:
    rng = np.random.default_rng([seed, 9])
    hits = total = 0
    for _ in range(TESTS):
        history: list[int] = []
        for _ in range(LENGTH):
            action = int(rng.integers(ACTIONS))
            hits += action == label(task, history)
            total += 1
            history.append(action)
    return hits / total


def jobs() -> list[tuple[str, str, int]]:
    return [(task, arm, seed) for task in TASKS for arm in ARMS for seed in SEEDS]


def summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in TASKS:
        for arm in ARMS:
            rows = [r for r in runs if r["task"] == task and r["arm"] == arm]
            out[f"{task}/{arm}"] = float(np.mean([r["score"] for r in rows]))
        rows = [r for r in runs if r["task"] == task and r["arm"] == "action"]
        out[f"{task}/action_erased"] = float(np.mean([r["erased"] for r in rows]))
        out[f"{task}/uniform"] = float(np.mean([uniform(task, seed) for seed in SEEDS]))
    return out


def gates_of(runs: list[dict[str, Any]]) -> dict[str, bool]:
    means = summary(runs)
    out = {}
    for task in TASKS:
        out[f"{task}_with_the_commands"] = means[f"{task}/action"] >= 0.85
        out[f"{task}_beats_the_founder_recall"] = (
            means[f"{task}/action"] >= means[f"{task}/recall"] + 0.2
        )
        out[f"{task}_beats_uniform_random"] = means[f"{task}/action"] >= UNIFORM + 0.3
        out[f"{task}_the_commands_do_the_work"] = (
            means[f"{task}/action_erased"] <= means[f"{task}/action"] - 0.2
        )
    return out


def check(body: dict[str, Any]) -> str | None:
    runs = body.get("runs", [])
    if sorted((r["task"], r["arm"], r["seed"]) for r in runs) != sorted(jobs()):
        return "the runs are not the declared arms and founders"
    if body.get("summary") != summary(runs):
        return "the summary does not follow from the runs"
    gates = gates_of(runs)
    if body.get("gates") != gates or body.get("passed") != all(gates.values()):
        return "the gates do not follow from the readings"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(run_one, jobs()))
    gates = gates_of(runs)
    result = {
        "schema": "cadence-own-action/1",
        "scope": "the recall memory carries the brain's own actions; component evidence",
        "configuration": {
            "inputs": INPUTS, "actions": ACTIONS, "bouts": BOUTS, "length": LENGTH,
            "tests": TESTS, "cadence": CADENCE, "modules": [24], "teaching": "normalized",
            "learning_surprise": True, "episodic": False, "arms": ARMS, "seeds": list(SEEDS),
            "uniform_random": UNIFORM,
        },
        "runs": runs,
        "summary": summary(runs),
        "gates": gates,
        "passed": all(gates.values()),
    }
    problem = check(result)
    if problem is not None:
        raise AssertionError(problem)
    if args.out is not None:
        cd.Receipt.build("cadence-own-action/1", result, sources()).write(args.out)
    print(json.dumps({"summary": result["summary"], "gates": gates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
