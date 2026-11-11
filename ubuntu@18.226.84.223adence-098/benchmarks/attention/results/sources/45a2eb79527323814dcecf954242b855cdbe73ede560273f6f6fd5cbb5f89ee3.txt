"""Depth on a task that needs it: the AND-count (``cadence-depth-and-count/1``).

Every observation is 24 pairs of bits. A pair is on when both its bits are; the answer is the
number of pairs on, less three, capped at six: seven classes, of which a uniform-random choice
names the right one a seventh of the time. A pair's AND is a threshold of two inputs and the
answer bins a count of such thresholds, so no linear readout of the bits names it: one hidden
layer approximates the task and two represent it.

The arms are Cadence brains of one to four processing regions (``modules`` (128,), (128, 64),
(128, 64, 64) and (128, 64, 64, 64)), composed with the apical compartment, normalized teaching
with the projection-relative floor and nudged phases of a whole 32 sweeps; tanh networks of one
and two hidden layers of the same widths trained by backpropagation with Adam; and the
uniform-random choice. Every lesson teaches 64 rows drawn from the training rows, every row (no
surprise rule); every ``--every`` lessons the 2,000 held-out rows are answered by the brain's own
free settle.

    python benchmarks/depth/and_count.py --out /tmp/depth --lessons 4000 --seeds 0 1 --jobs 4
    python benchmarks/depth/and_count.py --verify /tmp/depth/summary.json

A run writes one JSON line per arm and seed to ``rows.jsonl`` as it finishes and, at the end,
the receipt ``summary.json``: every reading, the settle work and the depth gains, bound to the
benchmark's and the library's source.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KIND = "cadence-depth-and-count/1"
PAIRS, SHIFT, CLASSES = 24, 3, 7
DATA_SEED, ROWS, HELD, BATCH = 12345, 200_000, 2_000, 64
CADENCE = {"one": (128,), "two": (128, 64), "three": (128, 64, 64), "four": (128, 64, 64, 64)}
MLP = {"mlp_one": (128,), "mlp_two": (128, 64)}
ARMS = (*CADENCE, *MLP, "random")
# the composed brain of every Cadence arm: the apical compartment, normalized steps with the
# projection-relative floor, nudged phases of a whole 32 sweeps from a free phase at the
# founder's tolerance, no memories
COMPOSE: dict[str, Any] = {
    "apical": 0.25,
    "teaching": "normalized",
    "learning_normalize_relative": 0.1,
    "learning_nudged_steps": 32,
    "learning_nudged_fixed": True,
    "learning_surprise": False,
    "episodic": False,
    "working_memory_amplitude": 0.0,
}
MLP_RATE = 1e-3


def sources() -> list[tuple[str, Path]]:
    names = (
        "benchmarks/depth/and_count.py",
        "src/cadence/brain.py",
        "src/cadence/generic.py",
        "src/cadence/learning.py",
    )
    return [(name, ROOT / name) for name in names]


def task(rows: int = ROWS, seed: int = DATA_SEED) -> tuple[np.ndarray, np.ndarray]:
    """The observations ``(rows, 48)`` and their answers, a fixed pool of the data seed."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(rows, PAIRS, 2), dtype=np.int8)
    on = bits[..., 0] & bits[..., 1]
    answers = np.clip(on.sum(axis=1) - SHIFT, 0, CLASSES - 1)
    return bits.reshape(rows, -1).astype(float), answers.astype(np.int64)


def split(rows: int, seed: int, held: int = HELD) -> tuple[np.ndarray, np.ndarray]:
    """The run seed's held-out and training rows."""
    order = np.random.default_rng(seed).permutation(rows)
    return order[:held], order[held:]


class Network:
    """A tanh network trained by backpropagation with Adam on the softmax cross-entropy."""

    def __init__(self, hidden: tuple[int, ...], seed: int, rate: float = MLP_RATE) -> None:
        rng = np.random.default_rng(seed)
        sizes = (2 * PAIRS, *hidden, CLASSES)
        self.weights = [rng.normal(0.0, np.sqrt(1.0 / a), size=(a, b))
                        for a, b in zip(sizes, sizes[1:], strict=False)]
        self.biases = [np.zeros(b) for b in sizes[1:]]
        self.first = [np.zeros_like(p) for p in self.weights + self.biases]
        self.second = [np.zeros_like(p) for p in self.weights + self.biases]
        self.rate, self.steps = rate, 0

    def layers(self, x: np.ndarray) -> list[np.ndarray]:
        out = [np.asarray(x, float)]
        for k, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = out[-1] @ w + b
            out.append(np.tanh(z) if k < len(self.weights) - 1 else z)
        return out

    def answer(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.layers(x)[-1].argmax(axis=1))

    def teach(self, x: np.ndarray, labels: np.ndarray) -> None:
        layers = self.layers(x)
        logits = layers[-1] - layers[-1].max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        delta = p
        delta[np.arange(len(labels)), labels] -= 1.0
        delta /= len(labels)
        grads_w: list[np.ndarray] = []
        grads_b: list[np.ndarray] = []
        for k in range(len(self.weights) - 1, -1, -1):
            grads_w.insert(0, layers[k].T @ delta)
            grads_b.insert(0, delta.sum(axis=0))
            if k:
                delta = (delta @ self.weights[k].T) * (1.0 - layers[k] ** 2)
        self.steps += 1
        params = self.weights + self.biases
        for p_, g, m, v in zip(params, grads_w + grads_b, self.first, self.second, strict=True):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            corrected = (m / (1 - 0.9**self.steps)) / (np.sqrt(v / (1 - 0.999**self.steps)) + 1e-8)
            p_ -= self.rate * corrected


def cadence_answers(brain: cd.Brain, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """The brain's answers by its own free settle, and the settle's work."""
    state = brain.learner.free(brain.stimulus(x, memory=False))
    motor = np.asarray(state.activation)[:, brain.learner.output_index]
    return motor.argmax(axis=1), {"read_sweeps": int(state.steps)}


def run_arm(
    arm: str, seed: int, *, lessons: int, every: int, rows: int = ROWS, held: int = HELD
) -> dict[str, Any]:
    """Teach one arm for ``lessons`` lessons of 64 rows; read the held-out rows every ``every``."""
    x, y = task(rows)
    held_rows, train = split(rows, seed, held)
    rng = np.random.default_rng(seed + 1)
    curve: list[dict[str, Any]] = []
    started = time.time()
    if arm == "random":
        draw = np.random.default_rng(seed + 2).integers(0, CLASSES, size=len(held_rows))
        accuracy = float(np.mean(draw == y[held_rows]))
        curve = [{"lesson": lesson, "accuracy": accuracy}
                 for lesson in range(every, lessons + 1, every)]
        return {"arm": arm, "seed": seed, "curve": curve, "seconds": 0.0}
    if arm in MLP:
        network = Network(MLP[arm], seed)
        for lesson in range(1, lessons + 1):
            rows_now = train[rng.integers(0, len(train), BATCH)]
            network.teach(x[rows_now], y[rows_now])
            if lesson % every == 0:
                accuracy = float(np.mean(network.answer(x[held_rows]) == y[held_rows]))
                curve.append({"lesson": lesson, "accuracy": accuracy,
                              "seconds": round(time.time() - started, 1)})
        return {"arm": arm, "seed": seed, "curve": curve, "rate": MLP_RATE,
                "parameters": int(sum(p.size for p in network.weights + network.biases)),
                "seconds": round(time.time() - started, 1)}
    brain = cd.Brain.compose(2 * PAIRS, CLASSES, modules=CADENCE[arm], seed=seed, **COMPOSE)
    work = {"free_steps": 0.0, "exhausted": 0.0, "lessons": 0}
    for lesson in range(1, lessons + 1):
        rows_now = train[rng.integers(0, len(train), BATCH)]
        report = brain.lesson(x[rows_now], y[rows_now])
        work["free_steps"] += float(report.get("free_steps", 0.0))
        work["exhausted"] += float(report.get("free_budget_exhausted", 0.0))
        work["lessons"] += 1
        if lesson % every == 0:
            answers, read = cadence_answers(brain, x[held_rows])
            curve.append({
                "lesson": lesson,
                "accuracy": float(np.mean(answers == y[held_rows])),
                "free_steps": work["free_steps"] / max(work["lessons"], 1),
                "free_budget_exhausted": work["exhausted"] / max(work["lessons"], 1),
                "read_sweeps": read["read_sweeps"],
                "seconds": round(time.time() - started, 1),
            })
            work = {"free_steps": 0.0, "exhausted": 0.0, "lessons": 0}
    return {"arm": arm, "seed": seed, "curve": curve, "modules": list(CADENCE[arm]),
            "synapses": int(brain.connectome.synapses), "apical": brain.brain.apical.to_dict(),
            "learning": brain.learner.config.to_dict(), "seconds": round(time.time() - started, 1)}


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Each arm's mean final accuracy over its seeds, and the depth gains over one region."""
    final: dict[str, list[float]] = {}
    for run in runs:
        final.setdefault(run["arm"], []).append(run["curve"][-1]["accuracy"])
    means = {arm: float(np.mean(values)) for arm, values in sorted(final.items())}
    gains = {}
    for deep, shallow in (("two", "one"), ("three", "one"), ("three", "two"), ("four", "one"),
                          ("four", "three"), ("mlp_two", "mlp_one")):
        if deep in means and shallow in means:
            gains[f"{deep}-{shallow}"] = means[deep] - means[shallow]
    return {"final_accuracy": means, "depth_gain": gains}


def check(body: dict[str, Any]) -> str | None:
    """Recompute the summary from the stored readings."""
    try:
        if body["schema"] != KIND:
            return "wrong schema"
        if summarize(body["runs"]) != body["summary"]:
            return "summary does not match the stored readings"
    except (KeyError, TypeError, ValueError, IndexError):
        return "incomplete or invalid readings"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    goal = parser.add_mutually_exclusive_group(required=True)
    goal.add_argument("--out", type=Path)
    goal.add_argument("--verify", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--lessons", type=int, default=4000)
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=1, help="arms and seeds run in parallel")
    args = parser.parse_args()
    if args.verify is not None:
        valid, message = cd.Receipt.verify(args.verify, sources=sources(), check=check)
        print(message)
        raise SystemExit(0 if valid else 1)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    cells = [(arm, seed) for arm in args.arms for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(run_arm, arm, seed, lessons=args.lessons, every=args.every)
                   for arm, seed in cells]
        runs = []
        for future in futures:  # in the declared order, whatever finishes first
            run = future.result()
            runs.append(run)
            with open(out / "rows.jsonl", "a") as handle:
                handle.write(json.dumps(run) + "\n")
            print(json.dumps({"arm": run["arm"], "seed": run["seed"], "final": run["curve"][-1]}),
                  flush=True)
    body = {
        "schema": KIND,
        "task": {"pairs": PAIRS, "shift": SHIFT, "classes": CLASSES, "data_seed": DATA_SEED,
                 "rows": ROWS, "held": HELD, "batch": BATCH},
        "compose": COMPOSE, "cadence_modules": {k: list(v) for k, v in CADENCE.items()},
        "mlp_hidden": {k: list(v) for k, v in MLP.items()}, "mlp_rate": MLP_RATE,
        "lessons": args.lessons, "every": args.every, "seeds": args.seeds,
        "version": cd.__version__, "runs": runs, "summary": summarize(runs),
    }
    cd.Receipt.build(KIND, body, sources()).write(out / "summary.json")
    print(json.dumps(body["summary"], indent=1))


if __name__ == "__main__":
    main()
