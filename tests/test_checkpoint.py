"""A learner survives a round trip through a checkpoint, backend changes included."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cadence as cd


def trained_learner(seed: int = 0) -> tuple[cd.Learner, np.ndarray]:
    wiring = cd.layered(6, 5, 3, density=1.0, seed=seed).with_sets(extra=[0, 1])
    config = cd.LearnerConfig(eta=0.4, eta_bias=0.1, momentum=0.5, normalize=0.5, decay=1e-3)
    tie = np.full(wiring.edges, -1, dtype=np.int64)
    tie[:4] = 0
    learner = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule(dt=1.0)),
        wiring.sets["output"],
        config,
        tie_groups=tie,
    )
    rng = np.random.default_rng(seed)
    drive = learner.engine.clamp_levels(rng.random((5, wiring.n)) * 0.6)
    for _ in range(3):
        learner.step(drive, np.array([0, 1, 2, 1, 0]))
    return learner, drive


def test_round_trip_reproduces_settlements_and_state(tmp_path: Path) -> None:
    learner, drive = trained_learner()
    path = learner.save(tmp_path / "net")
    assert path.suffix == ".npz"
    back = cd.Learner.load(path)
    assert np.allclose(back.engine.edge_scale, learner.engine.edge_scale)
    assert np.allclose(back.engine.bias, learner.engine.bias)
    assert np.allclose(back.free(drive).activation, learner.free(drive).activation)
    assert back.updates == learner.updates
    assert back.config == learner.config
    assert back.engine.wiring.sets == learner.engine.wiring.sets
    assert back.engine.wiring.digest() == learner.engine.wiring.digest()
    assert back.tie_groups is not None and np.array_equal(back.tie_groups, learner.tie_groups)
    assert np.allclose(back.velocity, learner.velocity)
    assert np.allclose(back.second_moment, learner.second_moment)
    # learning continues identically from the checkpoint
    labels = np.array([2, 2, 1, 0, 0])
    a = learner.step(drive, labels)[0].nudged.activation
    b = back.step(drive, labels)[0].nudged.activation
    assert np.allclose(a, b)
    assert np.allclose(back.engine.edge_scale, learner.engine.edge_scale)


def test_load_can_change_backend_and_freeze(tmp_path: Path) -> None:
    learner, drive = trained_learner()
    path = cd.save(learner, tmp_path / "net.npz")
    frozen = cd.load(path, config=cd.LearnerConfig(eta=0.0, eta_bias=0.0))
    before = frozen.engine.edge_scale.copy()
    frozen.step(drive, np.array([0, 1, 2, 1, 0]))
    assert np.array_equal(frozen.engine.edge_scale, before)  # a deployment that only settles
    if "torch" in cd.available_backends():
        on_torch = cd.load(path, backend="torch")
        assert on_torch.engine.backend == "torch"
        assert np.allclose(
            on_torch.free(drive).activation, learner.free(drive).activation, atol=1e-4
        )


def test_rule_with_adaptation_and_masks_survive(tmp_path: Path) -> None:
    wiring = cd.layered(4, 3, 2, density=1.0, seed=1)
    rule = cd.GradedRule(dt=0.5, adaptation=cd.Adaptation(tau_steps=20, strength=0.3), leak=0.1)
    overlaps = np.zeros(wiring.edges, dtype=bool)
    overlaps[::2] = True
    owners = np.zeros(wiring.n, dtype=bool)
    owners[-2:] = True
    learner = cd.Learner(
        cd.Settlement(wiring, rule),
        wiring.sets["output"],
        trainable_overlaps=overlaps,
        trainable_owners=owners,
    )
    back = cd.load(learner.save(tmp_path / "adapt"))
    assert back.engine.rule == rule
    assert back.trainable_overlaps is not None and np.array_equal(back.trainable_overlaps, overlaps)
    assert back.trainable_owners is not None and np.array_equal(back.trainable_owners, owners)


def test_refuses_foreign_files(tmp_path: Path) -> None:
    path = tmp_path / "other.npz"
    np.savez(path, meta=np.array('{"format": "something-else"}'), x=np.zeros(3))
    with pytest.raises(ValueError):
        cd.load(path)
