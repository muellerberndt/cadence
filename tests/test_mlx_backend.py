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
    cpu = cd.Brain.compose(8, 4, seed=3, modules=(32,), **{"recall": False, **settings})
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
    _, mlx = pair(tmp_path, observers=(8,), recall=True, recall_founder=True, recall_heads=2)
    phases = mlx.imagine([X[:1], X[1:2]])
    assert all(np.all(phase.qualified) for phase in phases)
    for _ in range(3):
        mlx.act(X, greedy=True)
    grown = mlx.grow(recall_heads=3)
    assert grown.brain.backend == "mlx"
    grown.act(X, greedy=True)


def test_mlx_refuses_an_apical_compartment(tmp_path):
    cpu = cd.Brain.compose(8, 4, seed=3, modules=(32, 32), apical=0.25, recall=False)
    cpu.save(tmp_path / "deep.npz")
    with pytest.raises(ValueError, match="apical"):
        cd.Brain.load(tmp_path / "deep.npz", backend="mlx")


def test_an_update_gives_the_new_brain_a_kernel_made_from_its_weights(tmp_path):
    """A compiled update on MLX gives the new brain a kernel of its own whose bias and blocks are
    the ones construction makes from the new weights; the brain before it keeps its kernel and
    its blocks; a weight float32 cannot hold is refused before anything changes."""
    _, mlx = pair(tmp_path)
    graph = mlx.brain
    learner = mlx.learner
    before = [np.array(block) for block in graph._mlx.blocks]
    rng = np.random.default_rng(4)
    learner.apply(rng.normal(size=graph.connectome.synapses) * 0.01,
                  rng.normal(size=graph.connectome.n) * 0.01)
    updated = mlx.brain
    assert updated is not graph and updated._mlx is not graph._mlx
    built = cd.NeuralGraph(graph.connectome, graph.neuron_model, backend="mlx",
                           efficacy=updated.efficacy, bias=updated.bias, layout=graph.layout)
    np.testing.assert_array_equal(np.array(updated._mlx.bias), np.array(built._mlx.bias))
    for mine, made in zip(updated._mlx.blocks, built._mlx.blocks, strict=True):
        np.testing.assert_array_equal(np.array(mine), np.array(made))
    for kept, block in zip(before, graph._mlx.blocks, strict=True):
        np.testing.assert_array_equal(kept, np.array(block))
    assert updated._mlx.edge_index is graph._mlx.edge_index  # the layout's tables are shared
    drive = mlx.stimulus(X)
    for name in ("v", "activation", "adaptation"):  # it settles as the brain built on them
        np.testing.assert_array_equal(getattr(updated.settle_batch(drive, steps=20), name),
                                      getattr(built.settle_batch(drive, steps=20), name))
    huge = updated.weights * 1e300
    with pytest.raises(ValueError, match="runtime precision"):
        updated._with_update(updated.efficacy * 1e300, huge, updated.bias, (True, True))


def test_a_settle_is_the_same_bits_as_one_graph_and_step_by_step(tmp_path):
    """MLX makes every block's products at every step and evaluates a settle's graph at its end
    (at least every 64 steps): the same state, to the bit, as the same sweeps evaluated one by
    one; a movement tolerance still stops at the first quiet step."""
    _, mlx = pair(tmp_path)
    graph = mlx.brain
    drive = mlx.stimulus(X)
    whole = graph.settle_batch(drive, steps=130)
    state = None
    for _ in range(130):
        state = graph.settle_batch(drive, steps=1, state=state)
    for name in ("v", "activation", "adaptation"):
        np.testing.assert_array_equal(getattr(whole, name), getattr(state, name))
    halves = graph.settle_batch(drive, steps=65, state=graph.settle_batch(drive, steps=65))
    np.testing.assert_array_equal(whole.v, halves.v)
    stopped = graph.settle_batch(drive, steps=500, tolerance=1e-4)
    again = graph.settle_batch(drive, steps=stopped.steps)
    assert 0 < stopped.steps < 500
    np.testing.assert_array_equal(stopped.v, again.v)
