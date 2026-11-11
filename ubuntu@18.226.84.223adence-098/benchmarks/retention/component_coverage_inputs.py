"""Pure, fixed actual-experience schedules; no brain or learning construction."""

from __future__ import annotations

import chamber_inputs as original
import numpy as np

ARMS = ("full-only", "component-coverage")
PHASES = ("continued", "revision", "restored")
WORLD = {"continued": 0, "revision": 1, "restored": 0}
BATCHES, STREAMS, SEED = 256, 8, 850064
SPECIMENS = (("orthogonal", 0), ("orthogonal", 128), ("orthogonal", 512), ("correlated", 0))


def frozen_arrays():
    rng = np.random.default_rng(SEED)
    arrays = {}
    for phase in PHASES:
        identities = original.schedule(rng, BATCHES)
        arrays[phase + "/identities"] = identities
        arrays[phase + "/uniform_actions"] = rng.integers(2, size=identities.shape, dtype=np.int64)
        for condition in original.CONDITIONS:
            proto = original.prototypes(condition)
            full = proto[identities].copy()
            partial = full.copy()
            codes = np.full(identities.shape, -1, dtype=np.int64)
            # Each cue's own occurrence stream receives a prewritten four-cycle.
            # Neutral events stay full; their outcome is zero for either action.
            count = np.zeros((STREAMS, 3), dtype=np.int64)
            for batch in range(BATCHES):
                for stream in range(STREAMS):
                    cue = int(identities[batch, stream])
                    if cue == 3:
                        continue
                    code = int(count[stream, cue] % 4)
                    count[stream, cue] += 1
                    if code in (1, 3):
                        removed = 0 if code == 1 else 1
                        codes[batch, stream] = removed
                        pairs = (1, 2) if condition == "correlated" and cue == 1 else (cue,)
                        for pair in pairs:
                            partial[batch, stream, 2 * pair + removed] = 0.0
            arrays[f"{condition}/{phase}/full-only"] = full
            arrays[f"{condition}/{phase}/component-coverage"] = partial
            arrays[f"{condition}/{phase}/removed_component"] = codes
    return arrays


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected = frozen_arrays()
    if arrays.keys() != expected.keys() or any(
        not np.array_equal(arrays[name], value) for name, value in expected.items()
    ):
        raise ValueError("frozen component-coverage schedule differs")
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


def coverage_audit(arrays, reserved):
    result = {
        "seed": SEED,
        "new_solves": 0,
        "new_outcomes": 0,
        "new_learning": 0,
        "phases": {},
        "new_experience_scope": (
            "explicit partial TRAIN physical experience, not unseen-deletion generalization"
        ),
    }
    for phase in PHASES:
        ids = arrays[phase + "/identities"]
        assert ids.shape == (BATCHES, STREAMS)
        for start in range(0, BATCHES, 16):
            for stream in range(STREAMS):
                assert np.array_equal(
                    np.bincount(ids[start : start + 16, stream], minlength=4), [7, 7, 1, 1]
                )
        info = {}
        for condition in original.CONDITIONS:
            full = arrays[f"{condition}/{phase}/full-only"]
            keys = arrays[f"{condition}/{phase}/component-coverage"]
            codes = arrays[f"{condition}/{phase}/removed_component"]
            assert np.array_equal(full, original.prototypes(condition)[ids])
            assert np.array_equal(keys[ids == 3], full[ids == 3])
            probes = reserved[condition + "/probes"]
            for cue in range(3):
                # TRAIN halves at amplitude1 are distinct from the retained .75/1.25 halves.
                for key in keys[(ids == cue) & (codes >= 0)]:
                    assert not np.any(np.all(probes[cue] == key, axis=1))
            info[condition] = {
                "per_cue_component_counts": [
                    {
                        "cue": cue,
                        "full": int(((ids == cue) & (codes == -1)).sum()),
                        "delete0": int(((ids == cue) & (codes == 0)).sum()),
                        "delete1": int(((ids == cue) & (codes == 1)).sum()),
                    }
                    for cue in range(4)
                ],
                "logical_key_bytes_per_arm": int(keys.nbytes),
            }
        random = arrays[phase + "/uniform_actions"]
        reward = original.actual_reward(ids, random, WORLD[phase])
        result["phases"][phase] = {
            "coverage": info,
            "uniform_random": {
                "events": int(ids.size),
                "non_neutral": int((ids != 3).sum()),
                "correct_non_neutral": int(
                    ((random == original.MAPPING[WORLD[phase], ids]) & (ids != 3)).sum()
                ),
                "reward_sum": float(reward.sum()),
            },
        }
    return result
