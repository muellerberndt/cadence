"""The repair of a settlement: the total movement of the published activations, per row."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def test_repair_is_the_path_length_of_the_activations_and_agrees_across_paths() -> None:
    w = cd.layered(6, 5, 3, density=1.0, seed=0)
    rule = cd.learning_rule(dt=1.0)
    engine = cd.Settlement(w, rule)
    drive = engine.clamp_levels(np.random.default_rng(0).random((3, w.n)) * 0.5)
    fused = engine.settle_batch(drive, steps=40)
    loop = engine.settle_batch(drive, steps=40, trajectory=True)
    assert fused.repair is not None and loop.repair is not None and loop.trajectory is not None
    assert fused.repair.shape == (3,) and (fused.repair > 0).all()
    assert np.allclose(fused.repair, loop.repair, atol=1e-12)
    path = np.abs(np.diff(loop.trajectory, axis=0, prepend=0.0)).sum(axis=2).sum(axis=0)
    assert np.allclose(loop.repair, path, atol=1e-12)
    still = engine.settle_batch(drive, steps=5, state=fused)  # already at rest: nothing moves
    assert still.repair is not None and still.repair.max() < 40 * 1e-4
    single = engine.settle(drive[0], steps=40)
    assert single.repair is not None and np.isclose(float(single.repair), fused.repair[0])


@pytest.mark.skipif("torch" not in cd.available_backends(), reason="torch not installed")
def test_torch_reports_the_same_repair() -> None:
    w = cd.layered(6, 5, 3, density=1.0, seed=1)
    rule = cd.learning_rule(dt=1.0)
    drive = cd.Settlement(w, rule).clamp_levels(np.random.default_rng(1).random((2, w.n)) * 0.5)
    cpu = cd.Settlement(w, rule).settle_batch(drive, steps=30)
    acc = cd.Settlement(w, rule, backend="torch", device="cpu").settle_batch(drive, steps=30)
    assert cpu.repair is not None and acc.repair is not None
    assert np.allclose(cpu.repair, acc.repair, atol=1e-9)
