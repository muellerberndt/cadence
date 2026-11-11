"""A scan-and-fire chamber where one answer dominates the lessons.

    PYTHONPATH=src python benchmarks/coaching/imbalance.py \\
        --out benchmarks/coaching/results/imbalance.json

A one-dimensional field of 32 columns shows a target as a bump around its column, or nothing.
The demonstrator scans LEFT while nothing is in view, turns LEFT or RIGHT toward a target off
centre, and FIREs at a target within two columns of the centre. A turn shifts the view by three
columns; a shot clears the target, and a new one appears after a few moments of scanning. LEFT
covers both the scan and the turn toward a target on the left, so it is the commonest answer by
far, as a scanning turn is in a shooting game.

Each arm composes a default brain per founder, watches 40 demonstrated episodes of 40 moments
(taught before each act), and is read frozen on 20 held-out demonstrated episodes: per answer,
the share of its moments the brain answers as the demonstrator does. Arms:

- ``plain``: every lesson at weight 1;
- ``inverse``: each lesson weighted by the inverse of its label's share among the lessons so far;
- ``sqrt``: by the inverse square root of that share;
- ``every``: every lesson at weight 1 on a brain that teaches every row
  (``learning_surprise=False``), where the other arms skip the rows the brain answers right.

Both weightings are normalised to a mean weight of 1 over the lessons (``LessonBalance`` below).
The readings did not favour a weighting overall, so it stays in this chamber. Teaching every row
agreed best overall.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from cadence import Brain

COLUMNS, CENTRE, SHIFT, WINDOW = 32, 15.5, 3, 2.0
LEFT, FIRE, RIGHT = 0, 1, 2
NAMES = ("LEFT", "FIRE", "RIGHT")


class LessonBalance:
    """Weights of ``share ** -power`` over the normaliser that keeps the mean lesson weight at 1,
    a share being the label's count plus one over the lessons counted plus the actions."""

    def __init__(self, actions: int, power: float):
        self.power, self.counts = power, np.zeros(actions)

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        np.add.at(self.counts, labels, 1)
        share = (self.counts + 1.0) / (self.counts.sum() + len(self.counts))
        return share[labels] ** -self.power / float(np.sum(share ** (1.0 - self.power)))


def view(column: float | None) -> np.ndarray:
    """The field: a bump one column wide around the target's column, zero elsewhere."""
    x = np.zeros((1, COLUMNS))
    if column is not None:
        bump = np.exp(-0.5 * (np.arange(COLUMNS) - column) ** 2)
        x[0] = np.where(bump < 0.05, 0.0, bump)
    return x


def demonstrator(column: float | None) -> int:
    if column is None or column < CENTRE - WINDOW:
        return LEFT
    return FIRE if column <= CENTRE + WINDOW else RIGHT


def episode(seed: int, length: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """One demonstrated episode: the field at each moment and the demonstrator's answer."""
    rng = np.random.default_rng(seed)
    column: float | None = None
    wait = int(rng.integers(2, 8))
    views, labels = [], []
    for _ in range(length):
        if column is None and wait == 0:
            column = float(rng.uniform(0, COLUMNS - 1))
        label = demonstrator(column)
        views.append(view(column))
        labels.append(label)
        if column is None:
            wait -= 1
        elif label == FIRE:
            column, wait = None, int(rng.integers(2, 8))
        else:  # turning left moves the target right in the view, and back
            column += SHIFT if label == LEFT else -SHIFT
            if not 0 <= column < COLUMNS:
                column = None
                wait = int(rng.integers(2, 8))
    return np.concatenate(views), np.array(labels)


def life(arm: str, founder: int) -> dict:
    options = {"learning_surprise": False} if arm == "every" else {}
    brain = Brain.compose(COLUMNS, 3, seed=founder, **options)
    balance = None
    if arm in ("inverse", "sqrt"):
        balance = LessonBalance(3, power=1.0 if arm == "inverse" else 0.5)
    started = time.time()
    for e in range(40):
        x, labels = episode(1000 * founder + e)
        brain.reset()
        for row in range(len(x)):
            weights = None if balance is None else balance(labels[row:row + 1])
            brain.lesson(x[row:row + 1], labels[row:row + 1], weights=weights)
            brain.act(x[row:row + 1], greedy=True)
    frozen = copy.deepcopy(brain)
    answers, truth = [], []
    for e in range(20):
        x, labels = episode(90_000 + e)
        frozen.reset()
        answers.extend(int(frozen.act(x[row:row + 1], greedy=True)[0]) for row in range(len(x)))
        truth.extend(labels.tolist())
    answers, truth = np.array(answers), np.array(truth)
    per = {NAMES[c]: float(np.mean(answers[truth == c] == c)) for c in range(3)}
    return {"arm": arm, "founder": founder, "overall": float(np.mean(answers == truth)),
            "per_answer": per, "shares": {NAMES[c]: float(np.mean(truth == c)) for c in range(3)},
            "seconds": round(time.time() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arms", nargs="+", default=["plain", "inverse", "sqrt", "every"])
    parser.add_argument("--founders", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    rows = [life(arm, f) for arm in args.arms for f in args.founders]
    for row in rows:
        per = " ".join(f"{k} {v:.2f}" for k, v in row["per_answer"].items())
        print(f"{row['arm']:8} founder {row['founder']}: overall {row['overall']:.2f}  {per}",
              flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
