from __future__ import annotations

import dataclasses

import numpy as np
import pytest

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


def test_adaptive_local_step_is_bias_corrected() -> None:
    """With momentum and normalization the first step of every moving overlap is eta in size
    (the running average and the RMS are corrected for their short history, as Adam's are),
    and a step is the same whether the contrast is large or small."""
    wiring = cd.layered(4, 6, 2, seed=8)
    config = cd.LearnerConfig(
        eta=0.01, eta_bias=0.0, momentum=0.9, normalize=0.999, normalize_floor=1e-12
    )
    learner = cd.Learner(cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"], config)
    drive = learner.engine.clamp_levels(np.pad(np.eye(4)[:2], ((0, 0), (0, wiring.n - 4))))
    before = learner.engine.edge_scale.copy()
    learner.step(drive, np.array([0, 1]))
    moved = np.abs(learner.engine.edge_scale - before)
    moving = moved > 0
    assert moving.any()
    assert np.allclose(moved[moving], config.eta, rtol=1e-6)
    # the same net, contrasts scaled down a hundredfold by a smaller nudge: the same first step
    small = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule()),
        wiring.sets["output"],
        dataclasses.replace(config, beta=0.001),
    )
    small.step(drive, np.array([0, 1]))
    moved_small = np.abs(small.engine.edge_scale - before)
    assert np.allclose(moved_small[moving], config.eta, rtol=1e-6)


def test_tie_groups_share_one_scale_across_positions() -> None:
    wiring, groups = cd.embedded(vocabulary=5, positions=3, dim=2, hidden=4, outputs=2, seed=1)
    learner = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"], tie_groups=groups
    )
    # every (token, unit) seam starts equal across the positions and stays equal after an update
    tied = groups >= 0
    rng = np.random.default_rng(0)
    windows = rng.integers(0, 5, size=(6, 3))
    drive = np.zeros((6, wiring.n))
    for r in range(6):
        for p in range(3):
            drive[r, p * 5 + windows[r, p]] = 1.0
    learner.step(drive, rng.integers(0, 2, 6))
    scale = learner.engine.edge_scale
    for g in np.unique(groups[tied]):
        members = scale[groups == g]
        assert members.size == 3 and np.allclose(members, members[0])
    # one embedding table, the tied dense seams, and the biases
    assert learner.parameters() == 5 * 2 + (3 * 2) * 4 + 4 * 2 + wiring.n


def test_decay_fades_seams_that_are_not_relearned() -> None:
    wiring = cd.layered(4, 3, 2, density=1.0, seed=0)
    engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0))
    config = cd.LearnerConfig(eta=0.0, eta_bias=0.0, decay=0.1)
    learner = cd.Learner(engine, wiring.sets["output"], config)
    before = learner.engine.edge_scale.copy()
    drive = engine.clamp_levels(np.zeros((2, wiring.n)))
    free = learner.free(drive)
    target = learner.targets(np.array([0, 1]))
    nudged = learner.nudged(drive, free, target)
    learner.update(free, nudged, learner.nudged(drive, free, target, sign=-1.0))
    # no contrast step at eta 0, only the leak
    assert np.allclose(learner.engine.edge_scale, before * 0.9)
    try:
        cd.LearnerConfig(decay=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("decay of 1 would erase the net every update and must be refused")


def test_trainable_masks_leave_the_rest_of_the_net_alone() -> None:
    wiring = cd.layered(4, 3, 2, density=1.0, seed=0)
    engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0))
    overlaps = np.zeros(wiring.edges, dtype=bool)
    overlaps[: wiring.edges // 2] = True
    owners = np.zeros(wiring.n, dtype=bool)
    owners[list(wiring.sets["output"])] = True
    config = cd.LearnerConfig(eta=0.5, eta_bias=0.5, decay=0.1)
    learner = cd.Learner(
        engine, wiring.sets["output"], config, trainable_overlaps=overlaps, trainable_owners=owners
    )
    scale0, bias0 = learner.engine.edge_scale.copy(), learner.engine.bias.copy()
    drive = engine.clamp_levels(np.ones((2, wiring.n)) * 0.5)
    learner.step(drive, np.array([0, 1]))
    assert np.array_equal(learner.engine.edge_scale[~overlaps], scale0[~overlaps])
    assert np.array_equal(learner.engine.bias[~owners], bias0[~owners])
    assert not np.array_equal(learner.engine.bias[owners], bias0[owners])


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_contrast_on_the_device_matches_the_host(backend: str) -> None:
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    w = cd.layered(12, 8, 4, density=1.0, seed=3)
    rule = cd.learning_rule(dt=1.0)
    kw = {"device": "cpu"} if backend == "torch" else {}
    device = cd.Settlement(w, rule, backend=backend, **kw)  # type: ignore[arg-type]
    host = cd.Settlement(w, rule)
    config = cd.LearnerConfig(eta=1.0, beta=0.1, temperature=0.1, tolerance=1e-4)
    drive = host.clamp_levels(np.random.default_rng(4).random((6, w.n)) * 0.5)
    labels = np.array([0, 1, 2, 3, 0, 1])
    for engine in (device, host):
        learner = cd.Learner(engine, w.sets["output"], config)
        state, _ = learner.step(drive, labels)
        if engine is device:
            assert state.nudged.device is not None
            assert engine.contrast_on_device(state.nudged, state.opposite) is not None  # type: ignore[arg-type]
            on_device = learner.engine.edge_scale.copy()
        else:
            on_host = learner.engine.edge_scale.copy()
    tolerance = 1e-12 if backend == "torch" else 1e-5
    assert np.abs(on_device - on_host).max() < tolerance
    assert host.contrast_on_device(state.nudged, state.opposite) is None  # type: ignore[arg-type]


def test_consolidation_pulls_the_seams_toward_a_slow_copy_that_follows() -> None:
    """With ``restore`` on and no learning signal, the seams return toward the slow copy;
    with ``consolidate`` on, the slow copy follows what is learned and kept."""
    import cadence as cd

    wiring = cd.layered(4, 6, 2, density=1.0, seed=0)
    engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0))
    learner = cd.Learner(engine, wiring.sets["output"], cd.LearnerConfig(eta=1.0, restore=0.5, consolidate=0.0))
    start = engine.edge_scale.copy()
    step = np.full(wiring.edges, 0.2)
    learner.apply(step, np.zeros(wiring.n))  # moved by 0.2, then pulled halfway back
    assert np.allclose(learner.engine.edge_scale - start, 0.1)
    learner.apply(np.zeros(wiring.edges), np.zeros(wiring.n))  # no step: halfway back again
    assert np.allclose(learner.engine.edge_scale - start, 0.05)
    following = cd.Learner(cd.Settlement(wiring, cd.learning_rule(dt=1.0)), wiring.sets["output"], cd.LearnerConfig(eta=1.0, restore=0.0, consolidate=0.5))
    following.apply(step, np.zeros(wiring.n))
    assert following._slow is not None and np.allclose(following._slow[0] - start, 0.1)  # the slow copy went halfway to the seams
    following.apply(np.zeros(wiring.edges), np.zeros(wiring.n))
    assert np.allclose(following._slow[0] - start, 0.15)


def test_consolidation_is_the_same_on_the_device() -> None:
    import pytest

    import cadence as cd

    if "torch" not in cd.available_backends():
        pytest.skip("no torch")
    wiring = cd.layered(4, 6, 2, density=1.0, seed=0)
    config = cd.LearnerConfig(eta=1.0, restore=0.3, consolidate=0.2)
    results = []
    for backend in ("cpu", "torch"):
        engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0), backend=backend, device="cpu" if backend == "torch" else None, precision="float64" if backend == "torch" else None)  # type: ignore[arg-type]
        learner = cd.Learner(engine, wiring.sets["output"], config)
        if backend == "torch":
            kernel = learner.engine._torch
            torch = kernel.torch
            for k in range(3):
                learner._apply_device(kernel, torch.full((wiring.edges,), 0.1 * (k + 1), dtype=kernel.param_dtype), torch.zeros(wiring.n, dtype=kernel.param_dtype))
        else:
            for k in range(3):
                learner.apply(np.full(wiring.edges, 0.1 * (k + 1)), np.zeros(wiring.n))
        results.append(np.asarray(learner.engine.edge_scale))
    assert np.allclose(results[0], results[1], atol=1e-9)
