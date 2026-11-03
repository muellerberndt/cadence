from __future__ import annotations

import numpy as np

import cadence as cd
from cadence.learning import SCALE_CAP


def _contextual_bandit(rng: np.random.Generator, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Two contexts, two actions; action 0 pays in context 0 and action 1 in context 1."""
    context = rng.integers(0, 2, size=batch)
    x = np.zeros((batch, 4))
    x[np.arange(batch), context * 2] = 1.0
    x[np.arange(batch), context * 2 + 1] = 1.0
    return x, context


def test_actor_critic_learns_a_contextual_bandit_from_dopamine() -> None:
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3),
        seed=0,
    )
    rng = np.random.default_rng(0)
    batch = 32

    def drive_of(x: np.ndarray) -> np.ndarray:
        return learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4))))

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
    connectome = cd.layered(4, 6, 2, density=1.0, seed=1)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(eta=1.0),
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(gamma=0.9, lam=0.5, eta=0.1),
        seed=1,
    )
    rng = np.random.default_rng(1)
    x, _ = _contextual_bandit(rng, 4)
    drive = learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4))))
    ac.act(drive)
    ac.learn(np.zeros(4), np.array([True, False, False, True]), drive)
    assert ac.trace is not None
    assert np.all(ac.trace[[0, 3]] == 0.0)
    assert np.any(ac.trace[[1, 2]] != 0.0)
    # every synapse's step is delta times its own trace: recompute one step by hand
    learner.reciprocal = False
    learner.reverse[:] = -1
    ac.act(drive)
    kind, plus, minus, value = ac._pending
    plus, minus = plus.activation, minus.activation  # the pending phases are states
    w = connectome
    contrast = (plus[:, w.pre] * plus[:, w.post] - minus[:, w.pre] * minus[:, w.post]) / (
        2.0 * learner.config.beta
    )
    before = learner.brain.efficacy.copy()
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
    got = learner.brain.efficacy - before
    clipped = np.abs(before + expected) > SCALE_CAP
    assert np.allclose(got[~clipped], expected[~clipped])


def test_grouped_softmax_nudge_agrees_between_kernels() -> None:

    from cadence import brain as S

    dims, size = 2, 5
    connectome = cd.layered(4, 8, dims * size, density=1.0, seed=3)
    brain = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0))
    rng = np.random.default_rng(3)
    drive = brain.stimulus_levels(np.pad(rng.random((6, 4)), ((0, 0), (0, connectome.n - 4))))
    out = np.asarray(connectome.populations["output"])
    target = np.zeros((6, connectome.n))
    target[:, out[[0, 7]]] = 1.0
    mask = np.zeros(connectome.n)
    mask[out] = 1.0
    groups = np.full(connectome.n, -1, dtype=np.int64)
    groups[out] = np.repeat(np.arange(dims), size)  # one softmax group per output dimension
    nudge = cd.Nudge(target, mask, 0.1, softmax_temperature=0.2, groups=groups)
    free = brain.settle_batch(drive, steps=60, tolerance=3e-3)
    fused = brain.settle_batch(drive, steps=12, state=free, nudge=nudge, tolerance=3e-3)
    was = S._FUSED
    S._FUSED = False
    try:
        plain = brain.settle_batch(drive, steps=12, state=free, nudge=nudge, tolerance=3e-3)
    finally:
        S._FUSED = was
    assert np.abs(fused.activation - plain.activation).max() < 1e-12


def test_one_stream_learns_the_same_on_the_device_as_on_the_host() -> None:
    """With one stream settled on the torch kernel, the trace and the step stay on the device
    and the synapses end where the host path puts them."""
    import pytest

    if "torch" not in cd.available_backends():
        pytest.skip("no torch")
    connectome = cd.layered(6, 10, 4, density=1.0, seed=1)
    config = cd.LearnerConfig(
        beta=0.1, eta=1.0, temperature=0.3, tolerance=1e-6, nudged_steps=30, free_steps=200
    )
    ac_config = cd.ActorCriticConfig(
        gamma=0.9, lam=0.8, eta=0.3, eta_bias=0.03, eta_critic=0.1, dopamine_cap=1.0
    )
    rng = np.random.default_rng(3)
    drives = [np.concatenate([rng.random(6), np.zeros(14)])[None] for _ in range(6)]
    rewards = [0.5, -0.2, 1.0, 0.0, 0.3, -1.0]
    results = []
    for backend in ("cpu", "torch"):
        brain = cd.NeuralGraph(
            connectome,
            cd.learning_neuron_model(dt=1.0),
            backend=backend,
            device="cpu" if backend == "torch" else None,
            precision="float64" if backend == "torch" else None,
        )  # type: ignore[arg-type]
        learner = cd.Learner(brain, connectome.populations["output"], config, slots=2)
        ac = cd.ActorCritic(learner, connectome.populations["hidden"], ac_config, seed=0)
        actions = []
        for k in range(5):
            actions.append(ac.act(drives[k]).copy())
            ac.learn(np.array([rewards[k]]), np.array([k == 3]), drives[k + 1])
        results.append(
            (
                np.stack(actions),
                np.asarray(ac.learner.brain.efficacy),
                np.asarray(ac.learner.brain.bias),
                ac.w_critic.copy(),
            )
        )
    (a_host, s_host, b_host, c_host), (a_dev, s_dev, b_dev, c_dev) = results
    assert np.array_equal(a_host, a_dev)
    assert np.allclose(s_host, s_dev, atol=1e-6) and np.abs(s_host - connectome.sign).max() > 1e-4
    assert np.allclose(b_host, b_dev, atol=1e-6)
    assert np.allclose(c_host, c_dev, atol=1e-6)


def test_a_batch_of_streams_learns_the_same_on_the_device_as_on_the_host() -> None:
    import pytest

    if "torch" not in cd.available_backends():
        pytest.skip("no torch")
    connectome = cd.layered(6, 10, 4, density=1.0, seed=2)
    config = cd.LearnerConfig(
        beta=0.1, eta=1.0, temperature=0.3, tolerance=1e-6, nudged_steps=30, free_steps=200
    )
    ac_config = cd.ActorCriticConfig(
        gamma=0.9, lam=0.8, eta=0.3, eta_bias=0.03, eta_critic=0.1, dopamine_cap=1.0
    )
    rng = np.random.default_rng(5)
    batch = 3
    drives = [
        np.concatenate([rng.random((batch, 6)), np.zeros((batch, 14))], axis=1) for _ in range(6)
    ]
    rewards = [rng.normal(size=batch) for _ in range(6)]
    dones = [np.zeros(batch, dtype=bool) for _ in range(6)]
    dones[2][1] = True
    results = []
    for backend in ("cpu", "torch"):
        brain = cd.NeuralGraph(
            connectome,
            cd.learning_neuron_model(dt=1.0),
            backend=backend,
            device="cpu" if backend == "torch" else None,
            precision="float64" if backend == "torch" else None,
        )  # type: ignore[arg-type]
        learner = cd.Learner(brain, connectome.populations["output"], config, slots=2)
        ac = cd.ActorCritic(learner, connectome.populations["hidden"], ac_config, seed=0)
        actions = []
        for k in range(5):
            actions.append(ac.act(drives[k]).copy())
            ac.learn(rewards[k], dones[k], drives[k + 1])
        results.append(
            (
                np.stack(actions),
                np.asarray(ac.learner.brain.efficacy),
                np.asarray(ac.learner.brain.bias),
                ac.w_critic.copy(),
            )
        )
    (a_host, s_host, b_host, c_host), (a_dev, s_dev, b_dev, c_dev) = results
    assert np.array_equal(a_host, a_dev)
    assert np.allclose(s_host, s_dev, atol=1e-6) and np.abs(s_host - connectome.sign).max() > 1e-4
    assert np.allclose(b_host, b_dev, atol=1e-6)
    assert np.allclose(c_host, c_dev, atol=1e-6)


def test_padding_cannot_teach_the_actor_critic_or_change_the_active_update() -> None:
    import pytest

    connectome = cd.layered(2, 4, 2, seed=4)
    for backend in ("cpu", "torch"):
        if backend == "torch" and "torch" not in cd.available_backends():
            continue
        results = []
        for batch in (1, 2):
            dynamics = cd.NeuralGraph(
                connectome,
                cd.learning_neuron_model(dt=1.0),
                backend=backend,
                device="cpu" if backend == "torch" else None,
                precision="float64" if backend == "torch" else None,
            )
            learner = cd.Learner(
                dynamics,
                connectome.populations["output"],
                cd.LearnerConfig(tolerance=1e-8, free_steps=200, nudged_steps=40),
            )
            actor = cd.ActorCritic(
                learner,
                connectome.populations["hidden"],
                cd.ActorCriticConfig(eta=0.1),
                seed=3,
            )
            drive = np.zeros((batch, connectome.n))
            drive[:, :2] = [0.3, 0.7]
            actor.act(drive)
            pending = actor._pending
            with pytest.raises(ValueError, match="observed"):
                actor.learn(
                    np.ones(batch), np.zeros(batch, bool), drive, observed=np.zeros(batch, bool)
                )
            assert actor._pending is pending
            observed = np.arange(batch) == 0
            actor.learn(
                np.array([1.0, 999.0])[:batch], np.ones(batch, bool), drive, observed=observed
            )
            assert actor._pending is None
            if actor.trace is not None:
                assert not actor.trace.any()
            if actor._trace_device is not None:
                assert not actor._trace_device[0].any().item()
            results.append(
                (
                    np.asarray(learner.brain.efficacy),
                    np.asarray(learner.brain.bias),
                    actor.w_critic,
                    actor.b_critic,
                )
            )
        for alone, padded in zip(results[0], results[1], strict=True):
            assert np.allclose(alone, padded, atol=1e-8), backend


def test_scale_cap_bounds_the_plastic_efficacy_and_is_validated() -> None:
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    brain = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0))
    learner = cd.Learner(brain, connectome.populations["output"], cd.LearnerConfig(scale_cap=0.5))
    learner.apply(np.full(connectome.synapses, 10.0), np.zeros(connectome.n))
    assert float(np.abs(learner.brain.efficacy).max()) <= 0.5
    wide = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)), connectome.populations["output"]
    )
    wide.apply(np.full(connectome.synapses, 10.0), np.zeros(connectome.n))
    assert float(np.abs(wide.brain.efficacy).max()) == SCALE_CAP
    assert cd.LearnerConfig().to_dict()["scale_cap"] == SCALE_CAP
    for bad in (0.0, -1.0, float("nan")):
        try:
            cd.LearnerConfig(scale_cap=bad)
        except ValueError:
            continue
        raise AssertionError(f"scale_cap={bad} was accepted")


def test_the_critic_learns_the_actors_capped_dopamine() -> None:
    """The critic steps on its trace, divided by the trace's energy, by the actor's dopamine:
    the prediction error capped at ``dopamine_cap``, so a reward far above the cap moves it as
    the cap does. The centred dopamine and the critic's other signals are not settings."""
    import pytest

    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=0.0, eta_critic=0.5),
        seed=0,
    )
    drive = np.pad(np.random.default_rng(3).random((1, 4)), ((0, 0), (0, connectome.n - 4)))
    ac.act(drive)
    assert ac.state is not None
    trace = np.append(ac.state.activation[0, ac.critic_index], 1.0)
    ac.learn(np.array([50.0]), np.zeros(1, dtype=bool), drive)  # the error 50, capped at 1
    step = 0.5 * trace / (1.0 + (trace**2).sum())
    np.testing.assert_allclose(ac.w_critic, step[:-1], rtol=1e-12)
    assert ac.b_critic == pytest.approx(step[-1], rel=1e-12)
    for name, value in (("dopamine_center", 0.5), ("dopamine_floor", 1.0),
                        ("center_scale", False), ("critic_normalize", False),
                        ("critic_signal", "td")):
        with pytest.raises(TypeError):
            cd.ActorCriticConfig(**{name: value})
        with pytest.raises(ValueError, match=f"actor_{name} was retired.*0.84.3"):
            cd.Brain.compose(4, 2, modules=(8,), **{f"actor_{name}": value})
    assert not hasattr(cd, "Valence")


def test_the_report_reads_the_outputs_saturation_and_the_traces_size() -> None:
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    plastic = np.zeros(connectome.synapses, dtype=bool)
    plastic[: connectome.synapses // 2] = True
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
        plastic_synapses=plastic,
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0),
        seed=0,
    )
    rng = np.random.default_rng(1)
    x, context = _contextual_bandit(rng, 8)
    drive = learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4))))
    action = ac.act(drive)
    report = ac.learn((action == context).astype(float), np.ones(8, dtype=bool), drive)
    assert 0.0 <= report["saturation"] <= 1.0
    assert report["trace"] >= 0.0
    assert report["trace"] == float(np.abs(ac.trace[:, plastic]).mean())


def test_the_report_counts_the_outcomes_the_dopamine_cap_clipped() -> None:
    """``capped`` is the share of observed rows whose signal exceeded ``dopamine_cap``."""
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    rng = np.random.default_rng(2)
    x, context = _contextual_bandit(rng, 4)

    def learner() -> cd.Learner:
        return cd.Learner(
            cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)),
            connectome.populations["output"],
            cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
        )

    def report(config: cd.ActorCriticConfig, reward: np.ndarray) -> dict[str, float]:
        ac = cd.ActorCritic(learner(), connectome.populations["hidden"], config, seed=0)
        drive = ac.learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4))))
        ac.act(drive)
        return ac.learn(reward, np.ones(4, dtype=bool), drive)

    small = cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0)
    assert report(small, np.array([0.5, -0.5, 0.0, 0.25]))["capped"] == 0.0
    # Rewards in units far beyond the cap of 1.0: every row is clipped to its sign.
    assert report(small, np.array([5.0, -5.0, 3.0, -7.0]))["capped"] == 1.0
    assert report(small, np.array([5.0, 0.5, 0.0, -7.0]))["capped"] == 0.5
    uncapped = cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, dopamine_cap=0.0)
    assert report(uncapped, np.array([5.0, -5.0, 3.0, -7.0]))["capped"] == 0.0
