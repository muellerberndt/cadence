"""Pure world schedules, reserved cues and measured random controls for #85.

This module constructs no brain and runs no learning or behavioral solve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

STREAMS, INPUTS, ACTIONS = 8, 8, 2
EXPOSURES = (0, 128, 512)
CONDITIONS = ("orthogonal", "correlated")
PHASES = ("acquisition", "routine", "revision", "restored")
WORLD = {"acquisition": 0, "routine": 0, "revision": 1, "restored": 0}
COUNTS = {"acquisition": 256, "revision": 256, "restored": 256}
MAPPING = np.array([[0, 1, 0, -1], [1, 0, 0, -1]], dtype=np.int64)
CURVE = (0, 32, 64, 128, 256)


def prototypes(condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown key condition")
    cues = np.zeros((4, INPUTS), dtype=np.float64)
    for cue in range(4):
        cues[cue, 2 * cue:2 * cue + 2] = 1 / np.sqrt(2)
    if condition == "correlated":
        cues[1] = 0.9 * cues[2] + np.sqrt(0.19) * cues[1]
    return cues


def actual_reward(identities, actions, world):
    identities, actions = np.asarray(identities), np.asarray(actions)
    if (world not in (0, 1) or identities.shape != actions.shape
            or identities.dtype.kind not in "iu" or actions.dtype.kind not in "iu"
            or np.any((identities < 0) | (identities > 3))
            or np.any((actions < 0) | (actions > 1))):
        raise ValueError("invalid executed world action")
    reward = (actions == MAPPING[world, identities]).astype(np.float64)
    reward[(identities == 2) & (actions != 0)] = -1
    reward[identities == 3] = 0
    return reward


def schedule(rng, count, *, balanced=False):
    if count % 16:
        raise ValueError("schedule requires complete sixteen-event blocks")
    if balanced:
        return np.array([rng.permutation(np.repeat(np.arange(4), 2))
                         for _ in range(count)], dtype=np.int64)
    block = np.array([0] * 7 + [1] * 7 + [2, 3])
    rows = [np.concatenate([rng.permutation(block) for _ in range(count // 16)])
            for _ in range(STREAMS)]
    return np.stack(rows, axis=1)


def reserved_probes(rng, condition):
    proto = prototypes(condition)
    variants = []
    for cue in range(3):
        full = [scale * proto[cue] for scale in (0.75, 0.9, 1.1)]
        partial = []
        for removed in range(2):
            for scale in (0.75, 1.25):
                value = proto[cue].copy()
                for pair in ([1, 2] if condition == "correlated" and cue == 1 else [cue]):
                    value[2 * pair + removed] = 0
                partial.append(scale * value)
        noise = [np.maximum(0, proto[cue] + rng.uniform(-0.025, 0.025, INPUTS))
                 for _ in range(3)]
        variants.append(np.stack(full + partial + noise))
    return np.stack(variants)


def frozen_arrays(seed):
    """Exposure arms share identical acquisition and post-switch schedules."""
    rng = np.random.default_rng(np.random.SeedSequence([85, seed, 20261004]))
    arrays = {"schedule/acquisition": schedule(rng, 256, balanced=True),
              "schedule/routine": schedule(rng, 512),
              "schedule/revision": schedule(rng, 256),
              "schedule/restored": schedule(rng, 256)}
    for phase in PHASES:
        shape = arrays["schedule/" + phase].shape
        arrays["uniform/" + phase] = rng.integers(2, size=shape, dtype=np.int64)
    for condition in CONDITIONS:
        arrays[condition + "/prototypes"] = prototypes(condition)
        arrays[condition + "/probes"] = reserved_probes(rng, condition)
        # One real measured policy, reused on identical query opportunities.
        arrays[condition + "/uniform_probes"] = rng.integers(2, size=(3, 10), dtype=np.int64)
    # After acquisition: at most 1280 factual batches, one eight-record replay per four.
    arrays["replay/indices"] = rng.integers(64, size=(320, STREAMS), dtype=np.int64)
    return arrays


def freeze(path: Path, seed):
    arrays = frozen_arrays(seed)
    np.savez_compressed(path, **arrays)
    return arrays


def load(path, seed):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected = frozen_arrays(seed)
    if arrays.keys() != expected.keys() or any(
        not np.array_equal(arrays[name], value) for name, value in expected.items()
    ):
        raise ValueError("frozen world/probe/random/replay schedule differs")
    for value in arrays.values():
        value.flags.writeable = False
    return arrays
