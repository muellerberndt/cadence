from __future__ import annotations

import numpy as np

import cadence as cd


def test_imagine_and_feel_learns_a_continuous_contextual_bandit() -> None:
    pop = cd.Population(dims=1, size=9, sigma=0.5)
    w, actor_mask, critic_mask = cd.actor_critic_wiring(4, 16, 9, 16, seed=0, per_dim=9)
    assert not (actor_mask & critic_mask).any()
    ac = cd.DreamActorCritic(
        w, actor_mask, critic_mask, pop, cd.learning_rule(dt=1.0),
        cd.DreamConfig(gamma=0.0, q_scale=1.0, batch=64, warmup=64, eta_critic=5.0, eta_actor=1.0, sigma=0.5, candidates=8, candidate_temperature=0.05),
        seed=0,
    )
    rng = np.random.default_rng(0)
    wanted = np.array([0.6, -0.6])

    def batch(k: int) -> tuple[np.ndarray, np.ndarray]:
        c = rng.integers(0, 2, k)
        x = np.zeros((k, 4))
        x[np.arange(k), 2 * c] = 1.0
        x[np.arange(k), 2 * c + 1] = 1.0
        return ac.engine.clamp_levels(np.pad(x, ((0, 0), (0, w.n - 4)))), c

    def error() -> float:
        d, c = batch(200)
        a, _ = ac.act(d, greedy=True)
        return float(np.abs(a[:, 0] - wanted[c]).mean())

    before = error()
    for _ in range(300):
        d, c = batch(32)
        a, _ = ac.act(d)
        ac.remember(d, a, -0.4 * (a[:, 0] - wanted[c]) ** 2, d, np.ones(32, dtype=bool))
        ac.update()
    after = error()
    assert after < 0.15 and after < before


def test_the_critic_dream_moves_only_critic_seams_and_the_actor_only_actor_seams() -> None:
    pop = cd.Population(dims=1, size=5)
    w, actor_mask, critic_mask = cd.actor_critic_wiring(3, 4, 5, 4, seed=1, per_dim=5)
    ac = cd.DreamActorCritic(w, actor_mask, critic_mask, pop, cd.learning_rule(dt=1.0), cd.DreamConfig(q_scale=1.0, batch=8, warmup=8, eta_critic=1.0, eta_actor=1.0), seed=1)
    drive = ac.engine.clamp_levels(np.pad(np.random.default_rng(1).random((8, 3)), ((0, 0), (0, w.n - 3))))
    before = ac.engine.edge_scale.copy()
    ac._dream(ac._clamp_action(drive, np.zeros((8, 1))), np.full(8, 0.7), ac.critic_mask, 1.0, owners=ac.no_actor)
    moved = ac.engine.edge_scale != before
    assert moved.any() and not (moved & ~critic_mask).any()
    before = ac.engine.edge_scale.copy()
    ac._imitate(drive, np.zeros((8, 1)))
    moved = ac.engine.edge_scale != before
    assert moved.any() and not (moved & ~actor_mask).any()
