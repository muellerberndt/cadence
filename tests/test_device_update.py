"""The learner's update on the torch device gives the same parameters as the host update."""

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")


def _net(seed: int = 0) -> tuple[cd.Wiring, np.ndarray]:
    return cd.embedded(12, 3, 6, 24, 5, seed=seed)  # tied embedding, symmetric feedback seams


def _drive(
    wiring: cd.Wiring, rng: np.random.Generator, batch: int
) -> tuple[np.ndarray, np.ndarray]:
    d = np.zeros((batch, wiring.n))
    for p in range(3):
        d[np.arange(batch), p * 12 + rng.integers(0, 12, batch)] = 1.0
    return d, rng.integers(0, 5, batch)


@pytest.mark.parametrize("freeze", [False, True])
def test_device_update_matches_host_update(freeze: bool) -> None:
    wiring, tie = _net()
    rng = np.random.default_rng(1)
    config = cd.LearnerConfig(
        eta=0.5, eta_bias=0.05, tolerance=1e-6, free_steps=200, nudged_steps=200, decay=0.01
    )
    host = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule(dt=1.0)),
        wiring.sets["output"],
        config,
        tie_groups=tie,
    )
    dev = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule(dt=1.0), backend="torch", device="cpu"),
        wiring.sets["output"],
        config,
        tie_groups=tie,
    )
    if freeze:
        mask = np.ones(wiring.edges, dtype=bool)
        mask[::7] = False
        host.trainable_overlaps = mask.copy()
        dev.trainable_overlaps = mask.copy()
    warm_h = warm_d = None
    for _ in range(4):
        d, labels = _drive(wiring, rng, 8)
        learned_h, rep_h = host.step(d, labels, warm=warm_h)
        learned_d, rep_d = dev.step(d, labels, warm=warm_d)
        warm_h, warm_d = learned_h.free, learned_d.free
        assert (
            learned_d.free.device is not None
            and learned_d.free.device["owner"] is dev.engine._torch
        )
    assert np.allclose(dev.engine.edge_scale, host.engine.edge_scale, atol=1e-9)
    assert np.allclose(dev.engine.bias, host.engine.bias, atol=1e-9)
    assert abs(rep_d["scale_step"] - rep_h["scale_step"]) < 1e-9
    if freeze:
        assert np.array_equal(dev.engine.edge_scale[~mask], wiring.sign[~mask])
    # the parameters stayed on the device: the host copy is fetched, and a save reads it
    assert dev.engine._edge_scale is not None  # fetched by the comparison above
    assert dev.updates == 4


def test_device_engine_continues_and_reads_back_after_updates() -> None:
    wiring, tie = _net(2)
    rng = np.random.default_rng(3)
    learner = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule(dt=1.0), backend="torch", device="cpu"),
        wiring.sets["output"],
        cd.LearnerConfig(eta=0.3),
        tie_groups=tie,
    )
    d, labels = _drive(wiring, rng, 4)
    before = learner.engine.edge_scale.copy()
    learner.step(d, labels)
    after = learner.engine.edge_scale
    assert not np.allclose(before, after)
    # the kernel's weights follow the parameters: a free settlement on the device equals the cpu engine's
    cpu = cd.Settlement(
        wiring, cd.learning_rule(dt=1.0), edge_scale=after, bias=learner.engine.bias
    )
    a = learner.engine.settle_batch(d, steps=200, tolerance=1e-8).activation
    b = cpu.settle_batch(d, steps=200, tolerance=1e-8).activation
    assert np.allclose(a, b, atol=1e-6)
