"""``cadence.arousal.Arousal`` in the core: the law's genes, its readings and its counts."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...arousal import MODES, Arousal, ArousalConfig

CLASS = Arousal
GENES = ("threshold", "decay", "tolerance", "floor", "fast", "slow", "heat", "value_surprise",
         "need", "need_scale")


def state_of(held: Arousal) -> dict[str, Any]:
    c = held.config
    out: dict[str, Any] = {f"config.{g}": np.float64(getattr(c, g)) for g in GENES}
    out["config.youth"] = np.int64(c.youth)
    out.update({
        "level": np.float64(held.level), "age": np.int64(held.age),
        "outcomes": np.int64(held.outcomes), "rewards": np.int64(held.rewards),
        "spreads": np.int64(held.spreads), "square": np.float64(held._square),
        "recent": np.float64(held._recent), "longrun": np.float64(held._longrun),
        "usual": np.float64(held._usual), "learning_sweeps": np.int64(held.learning_sweeps),
        "moments": np.array([held.moments[m] for m in MODES], dtype=np.int64),
        "sweeps": np.array([held.sweeps[m] for m in MODES], dtype=np.int64),
    })
    return out


def restore(held: Arousal, state: dict[str, Any]) -> None:
    genes = {g: float(state[f"config.{g}"]) for g in GENES}
    held.config = ArousalConfig(**genes, youth=int(state["config.youth"]))
    held.level, held.age = float(state["level"]), int(state["age"])
    held.outcomes, held.rewards = int(state["outcomes"]), int(state["rewards"])
    held.spreads = int(state["spreads"])
    held._square, held._recent = float(state["square"]), float(state["recent"])
    held._longrun, held._usual = float(state["longrun"]), float(state["usual"])
    held.learning_sweeps = int(state["learning_sweeps"])
    held.moments = {m: int(v) for m, v in zip(MODES, state["moments"], strict=True)}
    held.sweeps = {m: int(v) for m, v in zip(MODES, state["sweeps"], strict=True)}


def _outcome(error: float, reward: float, *, own: bool = True,
             learned: bool = True) -> dict[str, Any]:
    return {"error": np.float64(error), "reward": np.float64(reward), "own": np.bool_(own),
            "learned": np.bool_(learned)}


def _pair(value: tuple[float, float]) -> dict[str, Any]:
    return {"surprise": np.float64(value[0]), "want": np.float64(value[1])}


def _lived(sweeps: int, learning_sweeps: int = 0) -> dict[str, Any]:
    return {"sweeps": np.int64(sweeps), "learning_sweeps": np.int64(learning_sweeps)}


METHODS = {
    "outcome": (_outcome, _pair),
    "lived": (_lived, lambda value: {}),
}
