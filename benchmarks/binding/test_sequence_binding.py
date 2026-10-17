"""Contracts of the in-context succession instrument."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _instrument():
    spec = importlib.util.spec_from_file_location("sequence_binding", HERE / "sequence_binding.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sb = _instrument()


def test_holdout_is_one_successor_per_token_and_never_a_self_transition() -> None:
    for seed in range(6):
        held = sb.holdout(seed)
        assert (held.sum(axis=1) == 1).all() and (held.sum(axis=0) == 1).all()
        assert not np.diag(held).any()


def test_training_sequences_avoid_held_transitions_and_panels_cue_them() -> None:
    held = sb.holdout(1)
    rng = np.random.default_rng(0)
    trials = sb.draw(rng, 300, held)
    seqs = trials["sequences"]
    assert not held[seqs[:, :-1], seqs[:, 1:]].any()
    assert all(len(set(row)) == sb.LENGTH for row in seqs)
    cued = sb.draw(rng, 200, held, cue_held=True)
    s, c = cued["sequences"], cued["cue"]
    rows = np.arange(200)
    assert held[s[rows, c], s[rows, c + 1]].all()
    assert (held[s[:, :-1], s[:, 1:]].sum(axis=1) == 1).all()
    assert (sb.answers(cued) == s[rows, c + 1]).all()
    new = sb.draw(rng, 40, held, new_token=True)
    assert (new["sequences"][np.arange(40), new["cue"]] == sb.TOKENS - 1).all()


def test_the_successor_memory_recalls_what_followed_the_cue() -> None:
    held = sb.holdout(2)
    trials = sb.draw(np.random.default_rng(3), 8, held)
    seq = sb.events(trials)
    rows = np.arange(8)
    for mode in ("successor", "event"):
        brain = sb.MemoryBrain.build(0, mode=mode)
        brain.reset()
        for x in seq[:-1]:
            brain.act(x, greedy=True)
        recalled = brain.memory.recall(seq[-1])
        cued = len(sb.MARKERS) + trials["sequences"][rows, trials["cue"]]
        after = len(sb.MARKERS) + sb.answers(trials)
        if mode == "successor":
            np.testing.assert_allclose(recalled[rows, after], 1.0)
            np.testing.assert_allclose(recalled[rows, cued], 0.0)
        else:
            np.testing.assert_allclose(recalled[rows, cued], 1.0)
            np.testing.assert_allclose(recalled[rows, after], 0.0)
        brain.reset()
        assert brain.previous is None and not np.any(brain.memory.recall(seq[-1]))


def test_without_memory_the_brain_is_the_token_binding_composition() -> None:
    brain = sb.MemoryBrain.build(3)
    assert brain.memory is None and "recall" not in brain.connectome.populations
