"""Frozen input construction for the proposed finite recall protocol; no brain solves.

See FINITE_HORIZON_PROTOCOL.md. The evaluator must admit a reviewed, source-bound
protocol before using these arrays for scientific work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NEUTRAL, DISTRACTOR, START, WRITE = range(4)
QUERY = slice(4, 8)
PAYLOAD = slice(8, 12)
HISTORY = slice(12, 28)
INPUTS, STREAMS = 28, 8
TOKENS = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])


@dataclass(frozen=True)
class Condition:
    name: str
    delay: int
    load: int = 1
    query: int = 0
    distractors: bool = False
    partial: bool = False
    noise: float = 0.0
    opposite: bool = False
    interposed_neutral: bool = False


TRAIN_CONDITIONS = (
    Condition("clean-0", 0), Condition("clean-1", 1), Condition("clean-2", 2),
    Condition("distractor-1", 1, distractors=True),
    Condition("distractor-2", 2, distractors=True),
    Condition("partial-1", 1, partial=True), Condition("noise-1", 1, noise=0.05),
    Condition("replacement-1", 1, load=2, query=1, opposite=True, interposed_neutral=True),
    Condition("order-latest-1", 1, load=2, query=1, opposite=True),
    Condition("capacity-2-first-1", 1, load=2),
    Condition("capacity-4-first-1", 1, load=4), Condition("clean-4", 4),
)
TEST_CONDITIONS = (*TRAIN_CONDITIONS, Condition("clean-8", 8))


@dataclass(frozen=True)
class Episode:
    observations: np.ndarray
    labels: np.ndarray
    token_values: np.ndarray
    timestamps: np.ndarray
    irregular_timestamps: np.ndarray
    permutation: np.ndarray
    uniform_actions: np.ndarray


def make_episode(rng: np.random.Generator, condition: Condition) -> Episode:
    """Opposite queried values; all unrelated events/current inputs match within pairs."""
    if condition.delay < 0 or condition.load not in (1, 2, 4):
        raise ValueError("invalid delay or bounded load")
    if not 0 <= condition.query < condition.load or not 0 <= condition.noise <= 0.05:
        raise ValueError("invalid query or bounded noise")
    if condition.opposite and (condition.load != 2 or condition.query != 1):
        raise ValueError("replacement/order must query the second of two opposite tokens")
    labels = np.concatenate([rng.permutation(2) for _ in range(STREAMS // 2)])
    values = np.repeat(rng.integers(2, size=(condition.load, STREAMS // 2)), 2, axis=1)
    values[condition.query] = labels
    if condition.opposite:
        values[0] = 1 - labels

    def marker(coordinate: int) -> np.ndarray:
        obs = np.zeros((STREAMS, INPUTS))
        obs[:, coordinate] = 1
        return obs

    steps = [marker(START)]
    for position, value in enumerate(values):
        if position and condition.interposed_neutral:
            steps.append(marker(NEUTRAL))
        obs = marker(WRITE)
        payload = TOKENS[value].copy()
        if condition.partial:
            retained = np.repeat(rng.integers(2, size=STREAMS // 2), 2)
            payload[:] = 0
            payload[np.arange(STREAMS), 2 * value + retained] = 1
        if condition.noise:
            # Noise is independent of queried values and identical within each pair.
            payload += np.repeat(
                rng.uniform(-condition.noise, condition.noise, (STREAMS // 2, 4)), 2, axis=0,
            )
        obs[:, PAYLOAD] = payload
        steps.append(obs)
    for _ in range(condition.delay):
        obs = marker(DISTRACTOR if condition.distractors else NEUTRAL)
        if condition.distractors:
            obs[:, PAYLOAD] = np.repeat(rng.uniform(-0.25, 0.25, (STREAMS // 2, 4)), 2, axis=0)
        steps.append(obs)
    steps.append(marker(QUERY.start + condition.query))
    observations = np.stack(steps)
    increments = rng.uniform(0.1, 2, len(steps) - 1)
    irregular = np.r_[0, np.cumsum(increments)]
    irregular *= (len(steps) - 1) / irregular[-1]
    episode = Episode(
        observations, labels, values, np.arange(len(steps), dtype=float), irregular,
        np.arange(STREAMS) ^ 1, rng.integers(2, size=STREAMS),
    )
    for name in episode.__dataclass_fields__:
        getattr(episode, name).flags.writeable = False
    return episode


def append_observed_history(observations: np.ndarray) -> np.ndarray:
    """Control input is raw observed WRITE payloads; labels/answers are never an argument."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 3 or observations.shape[1:] != (STREAMS, INPUTS):
        raise ValueError("expected frozen (events, eight streams, 28 inputs)")
    if not np.isfinite(observations).all() or observations[:, :, HISTORY].any():
        raise ValueError("raw vanished-cue input must be finite with no appended history")
    result = observations.copy()
    stored = np.zeros((STREAMS, 4, 4))
    count = np.zeros(STREAMS, dtype=int)
    for index, obs in enumerate(observations):
        starts = obs[:, START] == 1
        stored[starts], count[starts] = 0, 0
        for row in np.flatnonzero(obs[:, WRITE] == 1):
            if count[row] >= 4:
                raise ValueError("history comparator capacity exceeded")
            stored[row, count[row]] = obs[row, PAYLOAD]
            count[row] += 1
        queries = obs[:, QUERY].any(axis=1)
        result[index, queries, HISTORY] = stored[queries].reshape((-1, 16))
    return result


def freeze_episodes(path: Path, *, seed: int) -> dict[str, np.ndarray]:
    """Write all future stimuli/baselines before any training; no source or runtime admission."""
    arrays = {}
    for phase, repeats, conditions, salt in (
        ("train", 16, TRAIN_CONDITIONS, 0), ("test", 24, TEST_CONDITIONS, 1),
    ):
        rng = np.random.default_rng(np.random.SeedSequence([seed, salt]))
        order = np.tile(np.arange(len(conditions)), repeats)
        if phase == "train":
            rng.shuffle(order)
        arrays[phase + "/condition"] = order
        for index, selected in enumerate(order):
            episode = make_episode(rng, conditions[selected])
            for name in episode.__dataclass_fields__:
                arrays[f"{phase}/{index:04d}/{name}"] = getattr(episode, name)
    np.savez_compressed(path, **arrays)
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


def check_trace_transition(before: dict, after: dict, source_activation: np.ndarray) -> float:
    """Independent literal source recurrence for this focus=0/decay=.8 protocol."""
    h = np.asarray(source_activation)
    if h.shape != np.asarray(before["trace"]).shape:
        raise ValueError("trace audit requires the live rows/coordinates before the event")
    expected = 0.8 * np.asarray(before["trace"]) + 0.2 * h
    error = float(np.max(np.abs(np.asarray(after["trace"]) - expected)))
    if not np.array_equal(after["last"], h) or np.asarray(after["cold"]).any():
        raise ValueError("last/cold do not record exactly one admitted real event")
    if not np.isfinite(error) or error > 1e-12:
        raise ValueError("trace does not match one declared accepted-event update")
    return error
