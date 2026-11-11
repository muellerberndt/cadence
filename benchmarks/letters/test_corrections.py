"""Corrections of read speech (corrections.py) on a tiny brain and made-up utterances."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from cadence import Brain

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("letters_corrections", _here / "corrections.py")
assert _spec is not None and _spec.loader is not None
co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(co)


def utterances(rng: np.random.Generator, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for _ in range(count):
        text = rng.integers(1, 29, size=int(rng.integers(2, 5)))
        out.append((rng.standard_normal((12, 16)).astype(np.float32), text))
    return out


def test_an_episode_reads_a_speaker_before_and_after_its_corrections(tmp_path):
    rng = np.random.default_rng(0)
    brain = Brain.compose(16, 29, seed=0, backend="cpu", learning_surprise=False)
    path = str(brain.save(tmp_path / "life.npz"))
    own, others, bank = utterances(rng, 4), utterances(rng, 3), utterances(rng, 5)
    for mode in co.MODES:
        out = co.episode(path, own, others, bank, mode, 0.1, 4, 0)
        assert out["corrections"] == 2 and out["read"] == 2 and out["others"] == 3
        for side in ("before", "after"):
            assert set(out[side]) == {"speaker", "others"}
            assert all(v <= 1.0 for v in out[side].values())
        assert out["seconds_per_correction"] > 0
    assert co.speakers(["1-a", "2-a", "1-b", "1-c"], 2) == {"1": [0, 2, 3]}
    assert co.edits("go on", "go no") == 2
