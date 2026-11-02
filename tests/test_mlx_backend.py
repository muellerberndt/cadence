"""A composed brain on the MLX backend against the CPU reference (#105), where MLX is installed:
it settles and learns as the CPU does within float32 rounding, lives, imagines, saves and
continues a pending outcome, reads several recall heads and grows; an apical compartment is
refused."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd

pytestmark = pytest.mark.skipif("mlx" not in cd.available_backends(), reason="mlx not installed")


def pair(tmp_path, **settings):
    cpu = cd.Brain.compose(8, 4, seed=3, modules=(32,), **settings)
    cpu.save(tmp_path / "brain.npz")
    return cpu, cd.Brain.load(tmp_path / "brain.npz", backend="mlx")


X = np.random.default_rng(0).random((4, 8))


def test_a_composed_brain_settles_and_learns_as_on_the_cpu(tmp_path):
    cpu, mlx = pair(tmp_path)
    assert mlx.brain.backend == "mlx"
    np.testing.assert_array_equal(cpu.act(X, greedy=True), mlx.act(X, greedy=True))
    np.testing.assert_allclose(
        np.asarray(mlx.basal_ganglia.state.activation),
        np.asarray(cpu.basal_ganglia.state.activation),
        rtol=0,
        atol=1e-5,
    )
    labels = np.array([0, 1, 2, 3])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu.lesson(X, labels)
        mlx.lesson(X, labels)
    np.testing.assert_allclose(mlx.brain.efficacy, cpu.brain.efficacy, rtol=0, atol=1e-5)


def test_a_living_brain_saves_and_continues_on_mlx(tmp_path):
    _, mlx = pair(tmp_path, arousal=True)
    rng = np.random.default_rng(1)
    reward = None
    for _ in range(12):
        action = mlx.live(rng.random((1, 8)), reward=reward)
        reward = np.array([float(action[0] == 0)])
    mlx.save(tmp_path / "living.npz")
    again = cd.Brain.load(tmp_path / "living.npz", backend="mlx")
    x = rng.random((1, 8))
    np.testing.assert_array_equal(mlx.live(x, reward=reward), again.live(x, reward=reward))


def test_imagination_observers_recall_heads_and_growth_run_on_mlx(tmp_path):
    _, mlx = pair(tmp_path, observers=(8,), recall=True, recall_heads=2)
    phases = mlx.imagine([X[:1], X[1:2]])
    assert all(np.all(phase.qualified) for phase in phases)
    for _ in range(3):
        mlx.act(X, greedy=True)
    grown = mlx.grow(recall_heads=3)
    assert grown.brain.backend == "mlx"
    grown.act(X, greedy=True)


def test_mlx_refuses_an_apical_compartment(tmp_path):
    cpu = cd.Brain.compose(8, 4, seed=3, modules=(32, 32), apical=0.25)
    cpu.save(tmp_path / "deep.npz")
    with pytest.raises(ValueError, match="apical"):
        cd.Brain.load(tmp_path / "deep.npz", backend="mlx")
