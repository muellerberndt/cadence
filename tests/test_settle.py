"""The settlement engine: nudges, adaptation, transports, batches, validation, and the torch kernel."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.settle import Nudge


def ring(n: int = 6) -> cd.Wiring:
    return cd.Wiring.from_edges(
        n, pre=list(range(n)), post=[(i + 1) % n for i in range(n)], count=[100] * n
    )


def test_available_backends_names_cpu() -> None:
    backends = cd.available_backends()
    assert backends["cpu"].startswith("numpy")


def test_settle_reports_steps_and_stops_at_tolerance() -> None:
    engine = cd.Settlement(ring(), cd.GradedRule(gain=0.03, dt=0.5))
    loose = engine.settle(clamp={0: 3.0}, steps=200, tolerance=1e-2)
    tight = engine.settle(clamp={0: 3.0}, steps=200, tolerance=1e-6)
    assert 0 < loose.steps <= tight.steps <= 200
    assert tight.activation.shape == (6,)
    with_trajectory = engine.settle(clamp={0: 3.0}, steps=5, trajectory=True)
    assert (
        with_trajectory.trajectory is not None
        and with_trajectory.trajectory.shape[0] == with_trajectory.steps
    )


def test_settled_state_helpers() -> None:
    engine = cd.Settlement(ring(), cd.GradedRule(gain=0.03))
    batch = engine.settle_batch(
        np.stack([engine.clamp_vector({0: 3.0}), engine.clamp_vector({3: 3.0})]), steps=40
    )
    assert batch.batched and batch.activation.shape == (2, 6)
    assert batch.row(1).shape == (6,)
    assert 0 <= batch.mean([0, 1, 2]) <= 1 and batch.mean([]) == 0.0
    assert 0 <= batch.fraction_active() <= 1 and 0 <= batch.fraction_active([0, 1]) <= 1
    assert batch.active(0.5) <= 6
    single = engine.settle(clamp={0: 3.0}, steps=40)
    assert not single.batched


def test_clamp_vector_and_levels() -> None:
    engine = cd.Settlement(ring(), cd.GradedRule(clamp_amplitude=2.0))
    v = engine.clamp_vector({2: 1.5})
    assert v.shape == (6,) and v[2] == 3.0 and v.sum() == 3.0  # level times the clamp amplitude
    levels = engine.clamp_levels(np.array([[0.5, 0, 0, 0, 0, 1.0]]))
    assert np.allclose(levels[0], [1.0, 0, 0, 0, 0, 2.0])


def test_dense_and_segmented_transports_agree_with_nudges_and_adaptation() -> None:
    wiring = cd.layered(5, 4, 3, density=1.0, seed=3)
    rule = cd.learning_rule(dt=0.5).replace(adaptation=cd.Adaptation(tau_steps=10, strength=0.2))
    dense = cd.Settlement(wiring, rule, dense_limit=10_000)
    segmented = cd.Settlement(wiring, rule, dense_limit=1)
    assert (
        dense.to_dict()["transport"] == "dense" and segmented.to_dict()["transport"] == "segmented"
    )
    drive = dense.clamp_levels(np.random.default_rng(0).random((3, wiring.n)) * 0.5)
    mask = np.zeros(wiring.n)
    mask[list(wiring.sets["output"])] = 1.0
    target = np.zeros((3, wiring.n))
    target[:, list(wiring.sets["output"])] = 0.8
    for nudge in (
        None,
        Nudge(target, mask, 0.2),
        Nudge(target, mask, 0.2, softmax_temperature=0.3, weight=np.array([1.0, -0.5, 0.0])),
    ):
        a = dense.settle_batch(drive, steps=30, nudge=nudge)
        b = segmented.settle_batch(drive, steps=30, nudge=nudge)
        assert np.allclose(a.activation, b.activation, atol=1e-9)
        assert np.allclose(a.adaptation, b.adaptation, atol=1e-9)
    assert dense.dense().shape == (wiring.n, wiring.n) and dense.weights.shape == (wiring.edges,)


def test_nudge_drive_shapes() -> None:
    mask = np.array([0.0, 1.0, 1.0])
    quadratic = Nudge(np.array([0.5, 0.5, 0.5]), mask, 0.1)
    push = quadratic.drive(np.array([0.2, 0.2, 0.2]))
    assert np.allclose(push, [0.0, 0.03, 0.03])
    weighted = Nudge(np.array([[0.5] * 3, [0.5] * 3]), mask, 0.1, weight=np.array([1.0, 2.0]))
    push2 = weighted.drive(np.array([[0.2] * 3, [0.2] * 3]))
    assert np.allclose(push2[1], 2 * push2[0])


def test_settlement_validates_inputs() -> None:
    wiring = ring()
    rule = cd.GradedRule()
    with pytest.raises(ValueError):
        cd.Settlement(wiring, rule, edge_scale=np.ones(3))
    with pytest.raises(ValueError):
        cd.Settlement(wiring, rule, bias=np.ones(2))
    with pytest.raises(ValueError):
        cd.Settlement(wiring, rule, backend="abacus")  # type: ignore[arg-type]
    engine = cd.Settlement(wiring, rule)
    with pytest.raises(ValueError):
        engine.settle_batch(np.zeros((2, 5)))
    state = engine.settle_batch(np.zeros((2, 6)), steps=2)
    with pytest.raises(ValueError):
        engine.settle_batch(np.zeros((3, 6)), state=state)
    one_row = engine.settle_batch(np.zeros(6), steps=2)
    assert one_row.activation.shape == (1, 6)


def test_with_parameters_and_readings() -> None:
    wiring = ring().with_sets(head=[0, 1], tail=[4, 5])
    engine = cd.Settlement(wiring, cd.GradedRule(gain=0.03))
    changed = engine.with_parameters(edge_scale=engine.edge_scale * 2.0, bias=np.full(6, 0.1))
    assert np.allclose(changed.edge_scale, engine.edge_scale * 2.0) and changed.bias[0] == 0.1
    state = engine.settle(clamp={0: 3.0}, steps=40)
    readings = engine.readings(state, ["head", "tail"])
    assert set(readings) == {"head", "tail"} and set(readings["head"]) >= {"mean", "fraction"}


@pytest.mark.skipif("torch" not in cd.available_backends(), reason="torch not installed")
def test_torch_kernel_matches_cpu_with_every_feature() -> None:
    wiring = cd.layered(8, 6, 4, density=1.0, seed=5)
    rule = cd.learning_rule(dt=0.5, leak=0.2).replace(
        adaptation=cd.Adaptation(tau_steps=15, strength=0.1)
    )
    cpu = cd.Settlement(wiring, rule)
    torch_dense = cd.Settlement(wiring, rule, backend="torch", dense_limit=10_000)
    torch_segmented = cd.Settlement(wiring, rule, backend="torch", dense_limit=1)
    drive = cpu.clamp_levels(np.random.default_rng(1).random((4, wiring.n)) * 0.5)
    out = list(wiring.sets["output"])
    mask = np.zeros(wiring.n)
    mask[out] = 1.0
    target = np.zeros((4, wiring.n))
    target[np.arange(4), [out[i % 4] for i in range(4)]] = 1.0
    for nudge in (
        None,
        Nudge(target, mask, 0.1),
        Nudge(target, mask, 0.1, softmax_temperature=0.2, weight=np.array([1.0, 0.5, -0.5, 0.0])),
    ):
        a = cpu.settle_batch(drive, steps=25, nudge=nudge, tolerance=None)
        for engine in (torch_dense, torch_segmented):
            b = engine.settle_batch(drive, steps=25, nudge=nudge, tolerance=None)
            assert np.allclose(a.activation, b.activation, atol=1e-4), engine.to_dict()["transport"]
            assert np.allclose(a.adaptation, b.adaptation, atol=1e-4)
    traced = torch_dense.settle_batch(drive, steps=6, trajectory=True, tolerance=1e-9)
    assert traced.trajectory is not None and traced.trajectory.shape == (traced.steps, 4, wiring.n)
    warm = torch_dense.settle_batch(drive, steps=3, state=traced)
    assert warm.activation.shape == (4, wiring.n)
