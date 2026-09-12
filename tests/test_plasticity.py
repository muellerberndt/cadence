from __future__ import annotations

import numpy as np

import cadence as cd


def _contextual_bandit(rng: np.random.Generator, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Two contexts, two actions; action 0 pays in context 0 and action 1 in context 1."""
    context = rng.integers(0, 2, size=batch)
    x = np.zeros((batch, 4))
    x[np.arange(batch), context * 2] = 1.0
    x[np.arange(batch), context * 2 + 1] = 1.0
    return x, context


def test_actor_critic_learns_a_contextual_bandit_from_dopamine() -> None:
    wiring = cd.layered(4, 8, 2, density=1.0, seed=0)
    learner = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule(dt=1.0)),
        wiring.sets["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    ac = cd.ActorCritic(
        learner,
        wiring.sets["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3),
        seed=0,
    )
    rng = np.random.default_rng(0)
    batch = 32

    def drive_of(x: np.ndarray) -> np.ndarray:
        return learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, wiring.n - 4))))

    def hit_rate() -> float:
        x, context = _contextual_bandit(rng, 200)
        ac.reset()
        action = ac.act(drive_of(x), greedy=True)
        return float((action == context).mean())

    before = hit_rate()
    ac.reset()
    x, context = _contextual_bandit(rng, batch)
    drive = drive_of(x)
    for _ in range(150):
        action = ac.act(drive)
        reward = (action == context).astype(float)
        x, context = _contextual_bandit(rng, batch)
        drive = drive_of(x)
        ac.learn(reward, np.ones(batch, dtype=bool), drive)
    after = hit_rate()
    assert after >= 0.9 and after > before


def test_traces_reset_on_done_and_updates_are_local() -> None:
    wiring = cd.layered(4, 6, 2, density=1.0, seed=1)
    learner = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule(dt=1.0)),
        wiring.sets["output"],
        cd.LearnerConfig(eta=1.0),
    )
    ac = cd.ActorCritic(
        learner, wiring.sets["hidden"], cd.ActorCriticConfig(gamma=0.9, lam=0.5, eta=0.1), seed=1
    )
    rng = np.random.default_rng(1)
    x, _ = _contextual_bandit(rng, 4)
    drive = learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, wiring.n - 4))))
    ac.act(drive)
    ac.learn(np.zeros(4), np.array([True, False, False, True]), drive)
    assert ac.trace is not None
    assert np.all(ac.trace[[0, 3]] == 0.0)
    assert np.any(ac.trace[[1, 2]] != 0.0)
    # every seam's step is delta times its own trace: recompute one step by hand
    learner.symmetric = False
    learner.reverse[:] = -1
    ac.act(drive)
    kind, plus, minus, value = ac._pending
    plus, minus = plus.activation, minus.activation  # the pending phases are states
    w = wiring
    contrast = (plus[:, w.pre] * plus[:, w.post] - minus[:, w.pre] * minus[:, w.post]) / (
        2.0 * learner.config.beta
    )
    before = learner.engine.edge_scale.copy()
    trace_before = ac.trace.copy()
    w_critic, b_critic = ac.w_critic.copy(), ac.b_critic
    reward = np.array([1.0, 0.0, 0.5, 0.0])
    ac.learn(reward, np.zeros(4, dtype=bool), drive)
    next_value = (
        ac._free.activation[:, ac.critic_index] @ w_critic + b_critic
    )  # the critic as it was
    delta = reward + 0.9 * next_value - value
    expected_trace = 0.9 * 0.5 * trace_before + contrast
    expected = 0.1 * (delta[:, None] * expected_trace).mean(axis=0)
    got = learner.engine.edge_scale - before
    clipped = np.abs(before + expected) > learner.config.scale_cap
    assert np.allclose(got[~clipped], expected[~clipped])


def test_population_actor_critic_learns_a_continuous_bandit() -> None:
    pop = cd.Population(dims=1, size=9, width=0.25, sigma=0.3)
    wiring = cd.layered(4, 8, pop.size, density=1.0, seed=2)
    learner = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule(dt=1.0)),
        wiring.sets["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, nudge="quadratic", tolerance=3e-3, nudged_steps=12),
    )
    ac = cd.ActorCritic(
        learner,
        wiring.sets["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3),
        seed=2,
        population=pop,
    )
    rng = np.random.default_rng(2)
    wanted = np.array([0.6, -0.6])  # context 0 wants +0.6, context 1 wants -0.6

    def drive_of(x: np.ndarray) -> np.ndarray:
        return learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, wiring.n - 4))))

    def error() -> float:
        x, context = _contextual_bandit(rng, 200)
        ac.reset()
        a = ac.act(drive_of(x), greedy=True)[:, 0]
        return float(np.abs(a - wanted[context]).mean())

    before = error()
    ac.reset()
    x, context = _contextual_bandit(rng, 32)
    drive = drive_of(x)
    for _ in range(200):
        a = ac.act(drive)[:, 0]
        reward = -((a - wanted[context]) ** 2)
        x, context = _contextual_bandit(rng, 32)
        drive = drive_of(x)
        ac.learn(reward, np.ones(32, dtype=bool), drive)
    after = error()
    assert after < 0.25 and after < before


def test_grouped_softmax_nudge_agrees_between_kernels_and_bins_learn_a_continuous_bandit() -> None:

    from cadence import settle as S

    bins = cd.Bins(dims=2, size=5)
    wiring = cd.layered(4, 8, bins.dims * bins.size, density=1.0, seed=3)
    engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0))
    rng = np.random.default_rng(3)
    drive = engine.clamp_levels(np.pad(rng.random((6, 4)), ((0, 0), (0, wiring.n - 4))))
    out = np.asarray(wiring.sets["output"])
    target = np.zeros((6, wiring.n))
    target[:, out[[0, 7]]] = 1.0
    mask = np.zeros(wiring.n)
    mask[out] = 1.0
    nudge = cd.Nudge(target, mask, 0.1, softmax_temperature=0.2, groups=bins.groups(out, wiring.n))
    free = engine.settle_batch(drive, steps=60, tolerance=3e-3)
    fused = engine.settle_batch(drive, steps=12, state=free, nudge=nudge, tolerance=3e-3)
    was = S._FUSED
    S._FUSED = False
    try:
        plain = engine.settle_batch(drive, steps=12, state=free, nudge=nudge, tolerance=3e-3)
    finally:
        S._FUSED = was
    assert np.abs(fused.activation - plain.activation).max() < 1e-12

    learner = cd.Learner(
        engine,
        wiring.sets["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    ac = cd.ActorCritic(
        learner,
        wiring.sets["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3),
        seed=3,
        population=bins,
    )
    wanted = np.array([[0.5, -1.0], [-0.5, 1.0]])

    def batch(k: int) -> tuple[np.ndarray, np.ndarray]:
        c = rng.integers(0, 2, k)
        x = np.zeros((k, 4))
        x[np.arange(k), 2 * c] = 1.0
        x[np.arange(k), 2 * c + 1] = 1.0
        return engine.clamp_levels(np.pad(x, ((0, 0), (0, wiring.n - 4)))), c

    def error() -> float:
        d, c = batch(200)
        ac.reset()
        return float(np.abs(ac.act(d, greedy=True) - wanted[c]).mean())

    before = error()
    ac.reset()
    d, c = batch(32)
    for _ in range(200):
        a = ac.act(d)
        reward = -((a - wanted[c]) ** 2).sum(axis=1)
        d, c = batch(32)
        ac.learn(reward, np.ones(32, dtype=bool), d)
    after = error()
    assert after < 0.2 and after < before


def test_value_net_fits_a_value_and_serves_the_actor_critic() -> None:
    rng = np.random.default_rng(4)
    critic = cd.ValueNet(
        4, 8, cd.learning_rule(dt=1.0), cd.ValueConfig(scale=2.0, offset=0.2, eta=1.0), seed=4
    )
    x = rng.random((32, 4))
    drive = critic.learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, 0))))
    target = x[:, 0] - x[:, 1]  # a value in [-1, 1]
    before = float(np.abs(critic.value(drive) - target).mean())
    for _ in range(150):
        critic.learn(drive, target)
    after = float(np.abs(critic.value(drive) - target).mean())
    assert after < 0.25 and after < before
    wiring = cd.layered(4, 8, 2, density=1.0, seed=4)
    learner = cd.Learner(
        cd.Settlement(wiring, cd.learning_rule(dt=1.0)),
        wiring.sets["output"],
        cd.LearnerConfig(eta=1.0),
    )
    ac = cd.ActorCritic(
        learner,
        cd.ValueNet(4, 8, cd.learning_rule(dt=1.0), seed=5),
        cd.ActorCriticConfig(gamma=0.9),
        seed=4,
    )
    d = learner.engine.clamp_levels(np.pad(x[:4], ((0, 0), (0, wiring.n - 4))))
    ac.act(d)
    report = ac.learn(np.ones(4), np.zeros(4, dtype=bool), d)
    assert np.isfinite(report["delta"])


def test_one_stream_learns_the_same_on_the_device_as_on_the_host() -> None:
    """With one stream settled on the torch kernel, the trace and the step stay on the device
    and the seams end where the host path puts them."""
    import pytest

    if "torch" not in cd.available_backends():
        pytest.skip("no torch")
    wiring = cd.layered(6, 10, 4, density=1.0, seed=1)
    config = cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.3, tolerance=1e-6, nudged_steps=30, free_steps=200)
    ac_config = cd.ActorCriticConfig(gamma=0.9, lam=0.8, lam_critic=0.8, eta=0.3, eta_bias=0.03, eta_critic=0.1, dopamine_cap=1.0)
    rng = np.random.default_rng(3)
    drives = [np.concatenate([rng.random(6), np.zeros(14)])[None] for _ in range(6)]
    rewards = [0.5, -0.2, 1.0, 0.0, 0.3, -1.0]
    results = []
    for backend in ("cpu", "torch"):
        engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0), backend=backend, device="cpu" if backend == "torch" else None, precision="float64" if backend == "torch" else None)  # type: ignore[arg-type]
        learner = cd.Learner(engine, wiring.sets["output"], config, slots=2)
        ac = cd.ActorCritic(learner, wiring.sets["hidden"], ac_config, seed=0, population=cd.Bins(dims=2, size=2))
        actions = []
        for k in range(5):
            actions.append(ac.act(drives[k]).copy())
            ac.learn(np.array([rewards[k]]), np.array([k == 3]), drives[k + 1])
        results.append((np.stack(actions), np.asarray(ac.learner.engine.edge_scale), np.asarray(ac.learner.engine.bias), ac.w_critic.copy()))
    (a_host, s_host, b_host, c_host), (a_dev, s_dev, b_dev, c_dev) = results
    assert np.array_equal(a_host, a_dev)
    assert np.allclose(s_host, s_dev, atol=1e-6) and np.abs(s_host - wiring.sign).max() > 1e-4
    assert np.allclose(b_host, b_dev, atol=1e-6)
    assert np.allclose(c_host, c_dev, atol=1e-6)
