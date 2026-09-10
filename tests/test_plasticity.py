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
        learner, wiring.sets["hidden"], cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3), seed=0
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
    learner = cd.Learner(cd.Settlement(wiring, cd.learning_rule(dt=1.0)), wiring.sets["output"], cd.LearnerConfig(eta=1.0))
    ac = cd.ActorCritic(learner, wiring.sets["hidden"], cd.ActorCriticConfig(gamma=0.9, lam=0.5, eta=0.1), seed=1)
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
    w = wiring
    contrast = (plus[:, w.pre] * plus[:, w.post] - minus[:, w.pre] * minus[:, w.post]) / (2.0 * learner.config.beta)
    before = learner.engine.edge_scale.copy()
    trace_before = ac.trace.copy()
    w_critic, b_critic = ac.w_critic.copy(), ac.b_critic
    reward = np.array([1.0, 0.0, 0.5, 0.0])
    ac.learn(reward, np.zeros(4, dtype=bool), drive)
    next_value = ac._free.activation[:, ac.critic_index] @ w_critic + b_critic  # the critic as it was
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
    ac = cd.ActorCritic(learner, wiring.sets["hidden"], cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3), seed=2, population=pop)
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
