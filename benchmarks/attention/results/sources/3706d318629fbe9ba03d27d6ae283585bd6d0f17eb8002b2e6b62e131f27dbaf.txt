"""Frozen diagnostic readouts of free association state, never a brain answer path.

``fit(features, targets)`` fits a centered two-slot, four-value ridge readout
with the fixed penalty .01. ``predict(probe, features)`` returns its labels.
``analyze(train_features, held_features, *, frozen_train, frozen_held,
swapped_features, erased_features)`` owns the benchmark split: exactly the
48 training rows and 16 held rows in ``dual_binding.panel`` order. Swapped
features use ``dual_binding.swapped(held)`` order. Erasures use original held
order and may be one panel or a name-to-panel mapping. Each panel may instead
be a runner reading dictionary containing ``association``; recorded refusals,
unrun status or inconsistent row identity are rejected. Arrays assert complete
panels and cannot omit or filter rows. Raw-query controls are derived here.

Only training targets fit the three probes. Held, swapped and erased states
are read with frozen coefficients. This tests contextual accessibility to a
held-out readout; it does not show general meaning or a within-settle change
of the held, context-free LearnedCode output. No campaign runs on import.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "attention_representation_task", Path(__file__).with_name("dual_binding.py")
)
assert SPEC is not None and SPEC.loader is not None
task = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = task
SPEC.loader.exec_module(task)

RIDGE = 0.01


def readonly(value: Any) -> np.ndarray:
    out = np.array(value, dtype=float, copy=True)
    out.flags.writeable = False
    return out


def features(value: Any, *, rows: int | None = None, width: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 2 or not array.shape[0] or not array.shape[1]
            or array.dtype.kind not in "buif" or not np.isfinite(array).all()
            or (rows is not None and len(array) != rows)
            or (width is not None and array.shape[1] != width)):
        raise ValueError("features require a complete finite numeric row-by-coordinate matrix")
    return np.array(array, dtype=float, copy=True)


def labels(value: Any, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape != (rows, task.SLOTS) or array.dtype.kind not in "iu"
            or (array < 0).any() or (array >= task.VALUES).any()):
        raise ValueError("targets require two integer values 0..3 for every feature row")
    return array.astype(np.int64, copy=True)


@dataclass(frozen=True)
class Probe:
    feature_mean: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    training_rows: int
    centered_rank: int

    def __post_init__(self) -> None:
        for name in ("feature_mean", "target_mean", "weights"):
            object.__setattr__(self, name, readonly(getattr(self, name)))

    def record(self) -> dict[str, Any]:
        return {"ridge": RIDGE, "training_rows": self.training_rows,
                "centered_rank": self.centered_rank,
                "feature_mean": self.feature_mean.tolist(),
                "target_mean": self.target_mean.tolist(), "weights": self.weights.tolist()}


def fit(value: Any, targets: Any) -> Probe:
    """Diagnostic least squares, with a free intercept and no parameter selection.

    Minimize ``||X_centered W - Y_centered||² + .01 ||W||²``; Y contains
    four one-hot entries per slot. The caller of this low-level helper chooses
    the rows; ``analyze`` enforces the declared training panel without filtering.
    """
    x = features(value)
    target = labels(targets, len(x))
    y = np.eye(task.VALUES)[target].reshape(len(x), -1)
    mean, target_mean = x.mean(axis=0), y.mean(axis=0)
    centered = x - mean
    with np.errstate(over="ignore", invalid="ignore"):
        gram = centered.T @ centered + RIDGE * np.eye(x.shape[1])
        rhs = centered.T @ (y - target_mean)
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise ValueError("feature scale makes the fixed ridge system nonfinite")
    weights = np.linalg.solve(gram, rhs)
    if not np.isfinite(weights).all():
        raise ValueError("fixed ridge coefficients must be finite")
    return Probe(mean, target_mean.reshape(task.SLOTS, task.VALUES),
                 weights.reshape(x.shape[1], task.SLOTS, task.VALUES),
                 len(x), int(np.linalg.matrix_rank(centered)))


def decision(probe: Probe, value: Any) -> np.ndarray:
    x = features(value, width=len(probe.feature_mean))
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.einsum("nd,dsv->nsv", x - probe.feature_mean, probe.weights) + probe.target_mean
    if not np.isfinite(out).all():
        raise ValueError("readout decisions must be finite")
    return out


def predict(probe: Probe, value: Any) -> np.ndarray:
    """Lowest-index argmax breaks ties; no threshold or held-out calibration."""
    return np.argmax(decision(probe, value), axis=-1).astype(np.int64)


def panel(value: Any, rows: np.ndarray) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("status", "completed") != "completed":
            raise ValueError("refused or unrun representation panels cannot be filtered")
        if value.get("work", {}).get("action_refusals", 0):
            raise ValueError("refused representation panels cannot be filtered")
        if "rows" in value and not np.array_equal(np.asarray(value["rows"]), rows):
            raise ValueError("representation panel rows differ from the declared split/order")
        if "answers" in value:
            labels(value["answers"], len(rows))  # -1 means refused/unrun, never a target.
        value = value.get("association")
    return features(value, rows=len(rows))


def reading(probe: Probe, x: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    scores = decision(probe, x)
    answers = np.argmax(scores, axis=-1).astype(np.int64)
    return {"rows": rows.tolist(), "features": x.tolist(), "targets": task.targets(rows).tolist(),
            "answers": answers.tolist(), "decision_scores": scores.tolist(),
            "scores": task.scores(answers, rows)}


def analyze(train_features: Any, held_features: Any, *, frozen_train: Any,
            frozen_held: Any, swapped_features: Any, erased_features: Any) -> dict[str, Any]:
    """Fit only the fixed train panel; retain every decision and enforce all gates.

    A geometric state shift alone earns no pass. Both normal and swapped held
    pairs must transfer at .90, beat raw-query and newborn readouts by .20,
    and lose .20 after each supplied relevant-record erasure, without refitting.
    Caller must establish the preceding acquisition/retrieval/causal gates.
    """
    train, held = task.panel("train"), task.panel("held_out")
    train_x, held_x = panel(train_features, train), panel(held_features, held)
    swapped_x = panel(swapped_features, task.swapped(held))
    erased = (erased_features if isinstance(erased_features, dict)
              and "association" not in erased_features else {"erased": erased_features})
    if not erased:
        raise ValueError("at least one complete relevant-record erasure panel is required")
    erased_x = {name: panel(value, held) for name, value in erased.items()}
    newborn_train, newborn_held = panel(frozen_train, train), panel(frozen_held, held)
    raw_train, raw_held = task.events(train)[1], task.events(held)[1]
    targets = task.targets(train)
    # All probes are fixed before any held-out decisions are made.
    probe, raw_probe, newborn_probe = (fit(x, targets) for x in (
        train_x, raw_train, newborn_train))
    learned = {"probe": probe.record(), "train": reading(probe, train_x, train),
               "held_out": reading(probe, held_x, held),
               "swapped": reading(probe, swapped_x, task.swapped(held)),
               "erased": {name: reading(probe, x, held) for name, x in erased_x.items()}}
    controls = {
        "raw_query": {"probe": raw_probe.record(),
                      "train": reading(raw_probe, raw_train, train),
                      "held_out": reading(raw_probe, raw_held, held)},
        "frozen_newborn": {"probe": newborn_probe.record(),
                           "train": reading(newborn_probe, newborn_train, train),
                           "held_out": reading(newborn_probe, newborn_held, held)},
    }
    accuracy = learned["held_out"]["scores"]["exact_pair_accuracy"]
    swapped_accuracy = learned["swapped"]["scores"]["exact_pair_accuracy"]
    gains = {name: accuracy - result["held_out"]["scores"]["exact_pair_accuracy"]
             for name, result in controls.items()}
    losses = {name: accuracy - result["scores"]["exact_pair_accuracy"]
              for name, result in learned["erased"].items()}
    gates = {"held_out_transfer": accuracy >= 0.90, "swapped_transfer": swapped_accuracy >= 0.90,
             "gain_over_raw_query": gains["raw_query"] >= 0.20,
             "gain_over_frozen_newborn": gains["frozen_newborn"] >= 0.20,
             "relevant_erasure": all(loss >= 0.20 for loss in losses.values())}
    return {"schema": "cadence-association-representation/1", "ridge": RIDGE,
            "scope": "Diagnostic contextual accessibility; "
                     "no installed answer path or semantic-code claim",
            "learned": learned, "controls": controls, "held_out_gains": gains,
            "erasure_losses": losses, "gates": gates, "passed": all(gates.values())}
