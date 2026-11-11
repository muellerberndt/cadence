"""The child's capabilities, each out of simpler components already in the core.

A child learning games keeps a permanent world model (the synapses), sees one frame and keeps
each moment reverberating for seconds before it vanishes (the settled state and its
afterglow), and changes the world model when a moment carried a positive or negative signal,
in proportion (the eligibility trace times the dopamine).
These tests are the small experiments for each capability."""

from __future__ import annotations

import numpy as np

import cadence as cd


def _bandit_drive(
    rng: np.random.Generator, batch: int, connectome: cd.Connectome, learner: cd.Learner
) -> tuple[np.ndarray, np.ndarray]:
    """Two contexts, two actions; action 0 pays in context 0 and action 1 in context 1."""
    context = rng.integers(0, 2, size=batch)
    x = np.zeros((batch, 4))
    x[np.arange(batch), context * 2] = 1.0
    x[np.arange(batch), context * 2 + 1] = 1.0
    return learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4)))), context


def _actor(connectome: cd.Connectome, **config: float) -> tuple[cd.ActorCritic, cd.Learner]:
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    return cd.ActorCritic(
        learner, connectome.populations["hidden"], cd.ActorCriticConfig(**config), seed=0
    ), learner


def test_the_trace_credits_an_action_paid_three_moments_later() -> None:
    """A press whose consequence comes later: the eligibility trace carries the moment to the
    signal. With no trace only the last moment is credited and nothing is learned."""
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)

    def hit_rate(lam: float) -> float:
        rng = np.random.default_rng(0)
        ac, learner = _actor(
            connectome,
            gamma=1.0,
            lam=lam,
            eta=1.0,
            eta_critic=0.0,
            dopamine_cap=0.0,
        )
        batch = 32
        blank = learner.brain.stimulus_levels(np.zeros((batch, connectome.n)))
        drive, context = _bandit_drive(rng, batch, connectome, learner)
        for _ in range(200):
            action = ac.act(drive)  # the moment that matters
            ac.learn(np.zeros(batch), np.zeros(batch, dtype=bool), blank)
            ac.act(blank)  # two moments of nothing, with presses of their own
            ac.learn(np.zeros(batch), np.zeros(batch, dtype=bool), blank)
            ac.act(blank)
            reward = (action == context).astype(float)  # paid now, three moments on
            drive, context = _bandit_drive(rng, batch, connectome, learner)
            ac.learn(reward, np.ones(batch, dtype=bool), drive)
        ac.reset()
        x, ctx = _bandit_drive(rng, 400, connectome, learner)
        return float((ac.act(x, greedy=True) == ctx).mean())

    with_trace, without = hit_rate(0.9), hit_rate(0.0)
    print(f"delayed credit: hit rate {with_trace:.2f} with the trace, {without:.2f} without")
    assert with_trace >= 0.8
    assert without <= 0.65


def test_a_capped_activity_change_carries_the_moment_before_and_a_full_one_forgets_it() -> None:
    """Settling as the memory: settled for a few steps from the previous equilibrium, the
    state still holds the moment before; settled to convergence, it does not."""
    w, _ = cd.embedded(4, 1, 4, 24, 4, seed=0)
    brain = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0))
    hidden = list(w.populations["hidden"])
    a = np.zeros((1, w.n))
    a[:, 0] = 1.0
    b = np.zeros((1, w.n))
    b[:, 1] = 1.0
    eq_a = brain.settle_batch(a, steps=300, tolerance=1e-5)
    cold_b = brain.settle_batch(b, steps=300, tolerance=1e-5)
    partial = brain.settle_batch(b, steps=3, state=eq_a)
    full = brain.settle_batch(b, steps=300, state=eq_a, tolerance=1e-5)
    d = lambda s: float(np.abs(s.activation[:, hidden] - eq_a.activation[:, hidden]).sum())  # noqa: E731
    print(
        f"distance from the moment before: partial settling {d(partial):.4f}, full settling {d(full):.4f}, cold {d(cold_b):.4f}"
    )
    assert d(partial) < d(cold_b)
    assert abs(d(full) - d(cold_b)) < 0.05 * d(cold_b) + 1e-6


def test_the_eligibility_weighted_by_the_afterimage_credits_the_cue_not_the_background() -> None:
    """A context seen for one moment with the press it calls for, paid three moments later, and
    twelve neurons of background always on: the synapses out of the background carry eligibility
    at every moment, the synapses out of the context only at the one that mattered. With the
    eligibility weighted by the afterimage (what is still ringing is what gets written) the
    signal lands on the context's synapses. The plain trace learns it here: the background's
    eligibility is noise that averages out over rounds. The weighting is reported and not
    asserted. With the trace's focus removed in 0.85 the afterimage is a plain decaying copy of
    the input, and the weighted run learns the cue on some seeds and platforms and stays at
    chance on others (seed 0 reads 1.00 with Accelerate and .48 with OpenBLAS)."""
    v, background, hidden, batch, rounds = 4, 12, 8, 32, 100
    n_in = v + background
    genome = cd.Genome(
        regions=(
            cd.Region("input", n_in),
            cd.Region("afterglow", n_in),
            cd.Region("hidden", hidden),
            cd.Region("output", 2),
        ),
        projections=(
            cd.Projection("input", "hidden", density=1.0, sign=0.0, scale=1.0, reciprocal=False),
            cd.Projection(
                "afterglow", "hidden", density=1.0, sign=0.0, scale=1.0, reciprocal=False
            ),
            cd.Projection("hidden", "output", density=1.0, sign=0.0, scale=1.0),
        ),
        label="cue",
    )

    def hit_rate(weighted: bool) -> float:
        rng = np.random.default_rng(0)
        connectome = cd.develop(genome, seed=0)
        learner = cd.Learner(
            cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)),
            connectome.populations["output"],
            cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
        )
        ac = cd.ActorCritic(
            learner,
            connectome.populations["hidden"],
            cd.ActorCriticConfig(
                gamma=1.0,
                lam=0.9,
                eta=1.0,
                eta_bias=0.05,  # the historical recipe of this measurement (issue 143)
                eta_critic=0.0,
                dopamine_cap=0.0,
            ),
            seed=0,
        )
        glow = cd.Trace(connectome, decay=0.7, source="input", target="afterglow")
        glow.reset(batch)

        def raw(context: np.ndarray | None) -> np.ndarray:
            d = np.zeros((batch, connectome.n))
            d[:, v:n_in] = 1.0  # the background, every moment
            if context is not None:
                d[np.arange(batch), context * 2] = 1.0
                d[np.arange(batch), context * 2 + 1] = 1.0
            return d

        def moment(drive: np.ndarray) -> np.ndarray:
            action = ac.act(drive)
            glow.update(ac._free)
            ac.salience = glow.ringing() if weighted else None
            return action

        context = rng.integers(0, 2, size=batch)
        drive = glow.stimulate(raw(context))
        for _ in range(rounds):
            action = moment(drive)  # the moment that matters, with the context in view
            drive = glow.stimulate(raw(None))
            ac.learn(np.zeros(batch), np.zeros(batch, dtype=bool), drive)
            moment(drive)  # two moments of background only, with presses of their own
            drive = glow.stimulate(raw(None))
            ac.learn(np.zeros(batch), np.zeros(batch, dtype=bool), drive)
            moment(drive)
            reward = (action == context).astype(float)
            context = rng.integers(0, 2, size=batch)
            drive = glow.stimulate(
                raw(context)
            )  # the afterimage runs on: the background has long stopped ringing, the new context rings
            ac.learn(reward, np.ones(batch, dtype=bool), drive)
        ac.reset()
        ac.salience = None
        hits = []
        for _ in range(8):
            ctx = rng.integers(0, 2, size=batch)
            hits.append((ac.act(glow.stimulate(raw(ctx)), greedy=True) == ctx).mean())
            glow.update(ac._free)
            ac.reset()
        return float(np.mean(hits))

    plain, weighted = hit_rate(False), hit_rate(True)
    print(
        f"a cue paid three moments later against twelve neurons of background: hit rate {plain:.2f} with the plain trace, {weighted:.2f} weighted by the afterimage"
    )
    assert plain > 0.9
