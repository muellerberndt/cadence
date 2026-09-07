from __future__ import annotations

import numpy as np

import cadence as cd


def two_blobs(n_per: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two classes of 8-pixel patterns: the left half lit, or the right half, with noise."""
    rng = np.random.default_rng(seed)
    x = np.zeros((2 * n_per, 8))
    y = np.repeat([0, 1], n_per)
    x[:n_per, :4] = 1.0
    x[n_per:, 4:] = 1.0
    x = np.clip(x + 0.3 * rng.standard_normal(x.shape), 0.0, 1.0)
    return x, y


def test_learner_separates_two_classes_and_the_rule_is_local() -> None:
    wiring = cd.layered(8, 16, 2, density=0.6, seed=1)
    learner = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"], cd.LearnerConfig(eta=2.0)
    )
    x, y = two_blobs()
    drive = learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, wiring.n - 8))))
    before = learner.accuracy(drive, y)
    rng = np.random.default_rng(0)
    for _ in range(3):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 20):
            idx = order[start : start + 20]
            learner.step(drive[idx], y[idx])
    after = learner.accuracy(drive, y)
    assert after >= 0.95 and after > before

    # Locality: an overlap's update is a function of its own two endpoints only (and, when
    # tied, of its reverse partner's two endpoints, which are the same two owners).
    learner.symmetric = False
    learner.reverse[:] = -1
    free = learner.free(drive[:8])
    target = learner.targets(y[:8])
    plus = learner.nudged(drive[:8], free, target)
    minus = learner.nudged(drive[:8], free, target, sign=-1.0)
    scale_before = learner.engine.edge_scale.copy()
    learner.update(free, plus, minus)
    delta = learner.engine.edge_scale - scale_before
    w = wiring
    expected = (
        learner.config.eta
        / (2.0 * learner.config.beta)
        * (
            (plus.activation[:, w.pre] * plus.activation[:, w.post]).mean(axis=0)
            - (minus.activation[:, w.pre] * minus.activation[:, w.post]).mean(axis=0)
        )
    )
    clipped = np.abs(scale_before + expected) > learner.config.scale_cap
    assert np.allclose(delta[~clipped], expected[~clipped])


def test_free_phase_never_sees_the_target() -> None:
    wiring = cd.layered(4, 6, 2, seed=2)
    learner = cd.Learner(cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"])
    drive = learner.engine.clamp_levels(np.pad(np.eye(4)[:2], ((0, 0), (0, wiring.n - 4))))
    a = learner.free(drive).activation
    b = learner.free(drive).activation
    assert np.array_equal(a, b)  # deterministic and label-free by construction
    free = learner.free(drive)
    target = learner.targets(np.array([0, 1]))
    plus = learner.nudged(drive, free, target)
    minus = learner.nudged(drive, free, target, sign=-1.0)
    out = wiring.sets["output"]
    for row, label in ((0, 0), (1, 1)):
        assert plus.activation[row, out[label]] > free.activation[row, out[label]]
        assert minus.activation[row, out[label]] < free.activation[row, out[label]]


def test_contrast_tracks_the_loss_gradient() -> None:
    """The centered contrast points along the finite-difference gradient of the nudge's loss."""
    wiring = cd.layered(8, 12, 3, density=0.7, seed=3)
    config = cd.LearnerConfig(beta=0.05, tolerance=1e-9, free_steps=400, nudged_steps=400)
    learner = cd.Learner(cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"], config)
    rng = np.random.default_rng(0)
    x = rng.random((16, 8))
    labels = rng.integers(0, 3, 16)
    drive = learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, wiring.n - 8))))
    out = np.asarray(wiring.sets["output"])

    def loss(engine: cd.Settlement) -> float:
        s = engine.settle_batch(drive, steps=400, tolerance=1e-9).activation[:, out]
        z = s / config.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return float(-np.log(p[np.arange(16), labels]).mean())

    free = learner.free(drive)
    target = learner.targets(labels)
    overlap_term, _ = learner.contrast(
        free,
        learner.nudged(drive, free, target),
        learner.nudged(drive, free, target, sign=-1.0),
    )
    tied = overlap_term + np.where(learner.reverse >= 0, overlap_term[learner.reverse], 0.0)
    base = loss(learner.engine)
    sample = rng.choice(wiring.edges, 40, replace=False)
    finite = []
    for e in sample:
        scale = learner.engine.edge_scale.copy()
        scale[e] += 1e-4
        if learner.reverse[e] >= 0:
            scale[learner.reverse[e]] += 1e-4
        finite.append(-(loss(learner.engine.with_parameters(edge_scale=scale)) - base) / 1e-4)
    correlation = np.corrcoef(np.asarray(finite), tied[sample])[0, 1]
    assert correlation > 0.9


def test_leak_keeps_rest_exact_and_responds_below_rest() -> None:
    rule = cd.learning_rule(leak=0.1)
    assert rule.activation(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]
    below = rule.activation(np.array([-1.0, -3.0, -50.0]))
    assert (below < 0).all() and (below >= -0.1).all()
    assert rule.activation(np.array([1.0]))[0] > 0
    assert rule.slope_at(np.array([-1.0]))[0] > 0


def test_settlement_stops_at_tolerance_and_reports_steps() -> None:
    wiring = cd.layered(4, 6, 2, seed=4)
    engine = cd.Settlement(wiring, cd.learning_rule())
    drive = engine.clamp_levels(np.pad(np.eye(4)[:1], ((0, 0), (0, wiring.n - 4))))
    fixed = engine.settle_batch(drive, steps=500)
    early = engine.settle_batch(drive, steps=500, tolerance=1e-6)
    assert early.steps < 500
    assert np.abs(early.activation - fixed.activation).max() < 1e-4
    assert early.trajectory is None
    with_trace = engine.settle_batch(drive, steps=500, tolerance=1e-6, trajectory=True)
    assert with_trace.trajectory is not None and len(with_trace.trajectory) == with_trace.steps


def test_dense_and_segmented_transport_agree() -> None:
    wiring = cd.layered(6, 10, 3, seed=5)
    rule = cd.learning_rule()
    drive = cd.Settlement(wiring, rule).clamp_levels(
        np.pad(np.random.default_rng(0).random((5, 6)), ((0, 0), (0, wiring.n - 6)))
    )
    dense = cd.Settlement(wiring, rule).settle_batch(drive, steps=80)
    segmented = cd.Settlement(wiring, rule, dense_limit=0).settle_batch(drive, steps=80)
    assert np.abs(dense.activation - segmented.activation).max() < 1e-12
    assert cd.Settlement(wiring, rule).to_dict()["transport"] == "dense"
    assert cd.Settlement(wiring, rule, dense_limit=0).to_dict()["transport"] == "segmented"


def test_weighted_nudge_pushes_each_row_its_own_way() -> None:
    wiring = cd.layered(4, 6, 2, seed=6)
    config = cd.LearnerConfig(tolerance=1e-12, free_steps=1000, nudged_steps=200)
    learner = cd.Learner(cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"], config)
    drive = learner.engine.clamp_levels(np.pad(np.eye(4)[:2], ((0, 0), (0, wiring.n - 4))))
    free = learner.free(drive)
    target = learner.targets(np.array([0, 0]))
    out = wiring.sets["output"]
    pulled = learner.nudged(drive, free, target, weight=np.array([1.0, -1.0]))
    assert pulled.activation[0, out[0]] > free.activation[0, out[0]]  # advantage: toward
    assert pulled.activation[1, out[0]] < free.activation[1, out[0]]  # penalty: away
    silent = learner.nudged(drive, free, target, weight=np.array([0.0, 0.0]))
    assert np.allclose(silent.activation, free.activation, atol=1e-6)


def test_normalized_steps_stay_local_and_bounded() -> None:
    wiring = cd.layered(4, 6, 2, seed=8)
    config = cd.LearnerConfig(eta=0.05, normalize=0.9)
    learner = cd.Learner(cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"], config)
    drive = learner.engine.clamp_levels(np.pad(np.eye(4)[:2], ((0, 0), (0, wiring.n - 4))))
    before = learner.engine.edge_scale.copy()
    learner.step(drive, np.array([0, 1]))
    moved = np.abs(learner.engine.edge_scale - before)
    assert moved.max() > 0
    # with the RMS floor of 1e-3 and one update, no overlap moves more than eta / (1 - rho) ** 0.5
    assert moved.max() <= config.eta / np.sqrt(1 - config.normalize) + 1e-9
    assert learner.second_moment.shape == (wiring.edges,)
