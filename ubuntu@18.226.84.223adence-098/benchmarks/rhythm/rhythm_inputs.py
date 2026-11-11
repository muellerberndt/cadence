"""Frozen inputs and event schedules for the steady-rhythm chamber; no brain solves.

See RHYTHM_PROTOCOL.md. Every observation, cue assignment, uniform-random action,
trace-transplant permutation and physical slot schedule is written before any model
runs. One event is one accepted observation and its one free act. Slot schedules
are declared physical times for the real-time runs; they never enter the brain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DRIVE, DISTRACTOR, CUE_A, CUE_B = range(4)
INPUTS = 4
ROWS = 4
ACTIONS = 2
KIND_DRIVE, KIND_PAUSE, KIND_DISTRACTOR, KIND_CUE = range(4)
KIND_NAMES = ("drive", "pause", "distractor", "cue")
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
REQUIRED_KEYS = (
    "schema",
    "event",
    "rows",
    "inputs",
    "actions",
    "teaching",
    "window",
    "disturbances",
    "cadence",
    "recipes",
    "seeds",
    "controls",
    "policy",
    "caps",
)


def load_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict, str]:
    """The frozen protocol and the SHA-256 of its exact bytes."""
    raw = Path(path).read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in protocol]
    if missing:
        raise ValueError(f"protocol lacks {missing}")
    if protocol["rows"] != ROWS or protocol["inputs"] != INPUTS or protocol["actions"] != ACTIONS:
        raise ValueError("protocol dimensions differ from the frozen input module")
    if len(set(protocol["seeds"]["development"]) & set(protocol["seeds"]["confirmation"])):
        raise ValueError("confirmation seeds must be fresh")
    return protocol, hashlib.sha256(raw).hexdigest()


def observation(kind: int, cues: np.ndarray | None = None) -> np.ndarray:
    """One event's observation for every row; a cue event carries the drive and one cue."""
    out = np.zeros((ROWS, INPUTS))
    if kind == KIND_PAUSE:
        return out
    if kind == KIND_DISTRACTOR:
        out[:, DISTRACTOR] = 1.0
        return out
    out[:, DRIVE] = 1.0
    if kind == KIND_CUE:
        if cues is None or cues.shape != (ROWS,) or set(np.unique(cues)) - {0, 1}:
            raise ValueError("a cue event needs one binary cue per row")
        out[np.arange(ROWS), CUE_A + cues] = 1.0
    return out


def bout(rng: np.random.Generator, events: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A cue event with two rows per phase, then identical constant drive for every row."""
    if events < 2:
        raise ValueError("a bout needs the cue event and at least one drive event")
    cues = rng.permutation(np.array([0, 0, 1, 1]))
    kinds = np.array([KIND_CUE] + [KIND_DRIVE] * (events - 1))
    observations = np.stack([observation(kind, cues) for kind in kinds])
    return observations, cues, kinds


def disturbance_events(name: str, pre: int, post: int) -> np.ndarray:
    """Event kinds of one disturbance run: drive, the disturbance, drive."""
    if name.startswith("pause"):
        middle = [KIND_PAUSE] * int(name[len("pause") :])
    elif name == "distractor":
        middle = [KIND_DISTRACTOR]
    else:
        raise ValueError(f"unknown disturbance {name!r}")
    return np.array([KIND_DRIVE] * pre + middle + [KIND_DRIVE] * post)


def slot_schedule(name: str, events: int, cadence_ms: int, disturbed_slot: int) -> dict:
    """Declared physical due times (ms) and external slot numbers for one real-time run."""
    slots = list(range(events))
    if name == "extra":
        slots.insert(disturbed_slot + 1, disturbed_slot)
    elif name == "skipped":
        slots.pop(disturbed_slot)
    elif name != "regular":
        raise ValueError(f"unknown slot schedule {name!r}")
    return {
        "slot": np.array(slots, dtype=np.int64),
        "due_ms": np.array(slots, dtype=np.int64) * cadence_ms,
    }


def shuffle_permutation(cues: np.ndarray) -> np.ndarray:
    """Each row receives the trace of a row that saw the other window cue."""
    first, second = np.flatnonzero(cues == 0), np.flatnonzero(cues == 1)
    if len(first) != 2 or len(second) != 2:
        raise ValueError("the window cue must have two rows per phase")
    permutation = np.empty(ROWS, dtype=np.int64)
    permutation[first] = second
    permutation[second] = first
    return permutation


def interval_schedules(seed: int, events: int, intervals: list[float]) -> dict:
    """Two schedules with identical interval multisets; timing never enters an observation."""
    values = np.resize(np.asarray(intervals, dtype=float), events - 1)
    if events < 3 or not len(intervals) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("positive intervals and at least three events are required")
    ordered = np.sort(values)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    shuffled = rng.permutation(ordered)
    if np.array_equal(ordered, shuffled) and len(np.unique(values)) > 1:
        shuffled = np.roll(shuffled, 1)
    return {
        name: {"slot": np.arange(events), "due_ms": np.r_[0.0, np.cumsum(gaps)]}
        for name, gaps in (("ordered", ordered), ("shuffled_time", shuffled))
    }


def freeze_inputs(path: Path, *, seed: int, protocol: dict) -> dict[str, np.ndarray]:
    """Write every observation, cue, random action and slot schedule before any model runs."""
    teaching, window, disturbances = (
        protocol["teaching"],
        protocol["window"],
        protocol["disturbances"],
    )
    cadence = protocol["cadence"]
    arrays: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    bouts = [bout(rng, teaching["events_per_bout"]) for _ in range(teaching["bouts"])]
    arrays["teach/observations"] = np.concatenate([item[0] for item in bouts])
    arrays["teach/cues"] = np.stack([item[1] for item in bouts])
    arrays["teach/kinds"] = np.concatenate([item[2] for item in bouts])
    arrays["teach/bout"] = np.repeat(np.arange(teaching["bouts"]), teaching["events_per_bout"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    observations, cues, kinds = bout(rng, 1 + window["lead"] + window["events"])
    arrays["window/observations"] = observations
    arrays["window/cues"] = cues
    arrays["window/kinds"] = kinds
    arrays["shuffle/permutation"] = shuffle_permutation(cues)
    names = [f"pause{k}" for k in disturbances["pauses"]] + ["distractor"]
    for name in names:
        kinds = disturbance_events(name, disturbances["pre"], disturbances["post"])
        arrays[f"disturbance/{name}/kinds"] = kinds
        arrays[f"disturbance/{name}/observations"] = np.stack([observation(kind) for kind in kinds])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    arrays["random/window"] = rng.integers(ACTIONS, size=(window["events"], ROWS))
    for name in names:
        arrays[f"random/disturbance/{name}"] = rng.integers(
            ACTIONS, size=(len(arrays[f"disturbance/{name}/kinds"]), ROWS)
        )
    for name in ("regular", "extra", "skipped"):
        schedule = slot_schedule(
            name, cadence["events"], protocol["event"]["cadence_ms"], cadence["disturbed_slot"]
        )
        for key, value in schedule.items():
            arrays[f"cadence/{name}/{key}"] = value
    for variant in protocol["event"]["cadence_variants_ms"]:
        schedule = slot_schedule("regular", cadence["events"], variant, cadence["disturbed_slot"])
        arrays[f"cadence/regular{variant}/slot"] = schedule["slot"]
        arrays[f"cadence/regular{variant}/due_ms"] = schedule["due_ms"]
    if "intervals_ms" in cadence:
        for name, schedule in interval_schedules(
            seed, cadence["events"], cadence["intervals_ms"]
        ).items():
            for key, value in schedule.items():
                arrays[f"cadence/{name}/{key}"] = value
    np.savez_compressed(path, **arrays)
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


def cue_labels(kinds: np.ndarray, cues: np.ndarray, bouts: np.ndarray) -> np.ndarray:
    """The cued action at each cue event and -1 elsewhere, per row."""
    labels = -np.ones((len(kinds), ROWS), dtype=np.int64)
    for index in np.flatnonzero(kinds == KIND_CUE):
        labels[index] = cues[bouts[index]]
    return labels


def ideal_alternation(anchor: np.ndarray, events: int) -> np.ndarray:
    """The alternation that continues ``anchor`` (the last executed action of each row)."""
    anchor = np.asarray(anchor, dtype=np.int64)
    if anchor.shape != (ROWS,) or set(np.unique(anchor)) - {0, 1}:
        raise ValueError("anchor must hold one executed action per row")
    return (anchor[None, :] + 1 + np.arange(events)[:, None]) % ACTIONS
