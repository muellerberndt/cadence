"""State ownership, isolated streams and backend-independent learning contracts."""

import numpy as np
import pytest

import cadence as cd


def controller(backend="cpu"):
    graph = cd.layered(2, 3, 2, density=1, seed=7)
    brain = cd.NeuralGraph(
        graph,
        cd.learning_neuron_model(dt=0.3),
        backend=backend,
        device="cpu" if backend == "torch" else None,
        precision="float64" if backend == "torch" else None,
    )
    learner = cd.Learner(
        brain,
        graph.populations["output"],
        cd.LearnerConfig(free_steps=3, nudged_steps=2, tolerance=None),
    )
    return cd.ActorCritic(
        learner,
        graph.populations["hidden"],
        cd.ActorCriticConfig(eta=0, eta_bias=0, eta_critic=0),
        seed=1,
    )


def test_new_observation_is_not_replaced_by_cached_state():
    agent = controller()
    drive = np.zeros((1, agent.n))
    agent.act(drive, greedy=True)
    old = agent.state
    drive[0, 0] = 2  # environments commonly reuse the same observation buffer
    expected = agent.learner.free(drive, warm=old)
    agent.act(drive, greedy=True)
    np.testing.assert_allclose(agent.state.activation, expected.activation)
    assert not np.array_equal(agent.state.activation, old.activation)


def test_greedy_action_cannot_reuse_previous_actions_eligibility():
    agent = controller()
    drive = np.ones((1, agent.n))
    agent.act(drive)
    agent.act(drive, greedy=True)
    with pytest.raises(RuntimeError, match="act"):
        agent.learn(np.ones(1), np.zeros(1, bool), drive)


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_ending_one_stream_does_not_resettle_other_streams(backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    agent = controller(backend)
    drive = np.ones((2, agent.n))
    agent.act(drive)
    free = agent.state
    expected = agent.learner.free(drive, warm=free).activation[1]
    cold = agent.learner.free(drive).activation[0]
    agent.learn(np.zeros(2), np.array([True, False]), drive)
    np.testing.assert_allclose(agent.state.activation[1], expected, atol=1e-12)
    np.testing.assert_allclose(agent.state.activation[0], cold, atol=1e-12)


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_frozen_large_synapse_survives_clipping(backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    agent = controller(backend)
    learner = agent.learner
    scale = learner.brain.efficacy.copy()
    scale[0] = 12.0
    learner.brain = learner.brain.with_parameters(efficacy=scale)
    learner.plastic_synapses[0] = False
    learner.config = cd.LearnerConfig()
    if backend == "cpu":
        learner.apply(np.zeros_like(scale), np.zeros(agent.n))
    else:
        kernel = learner.brain._torch
        learner._apply_device(
            kernel,
            kernel.torch.zeros_like(kernel.scale),
            kernel.torch.zeros_like(kernel.bias_param),
        )
    assert learner.brain.efficacy[0] == 12


def test_device_mask_cache_reuses_tensors_and_detects_inplace_edits():
    pytest.importorskip("torch")
    agent = controller("torch")
    learner, kernel = agent.learner, agent.learner.brain._torch
    first = learner._device_indices(kernel)["paired"]
    assert learner._device_indices(kernel)["paired"] is first
    learner.plastic_synapses[0] = False
    assert learner._device_indices(kernel)["synapses"][0].item() == 0


@pytest.mark.parametrize("backend", ["cpu", "torch", "mlx"])
def test_duplicate_synapses_sum_in_every_transport(backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    graph = cd.Connectome(2, np.array([0, 0]), np.array([1, 1]), np.ones(2), np.array([1.0, 2.0]))
    brain = cd.NeuralGraph(
        graph,
        cd.learning_neuron_model(dt=1),
        backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    reference = cd.NeuralGraph(
        cd.Connectome.from_synapses(2, pre=[0], post=[1], sign=[3.0]),
        cd.learning_neuron_model(dt=1),
    )
    expected = reference.settle_batch(np.array([[1.0, 0.0]]), steps=5).activation
    for _ in range(2):
        got = brain.settle_batch(np.array([[1.0, 0.0]]), steps=5).activation
        np.testing.assert_allclose(got, expected, atol=1e-6)
        brain = brain.with_parameters(efficacy=brain.efficacy.copy())


def test_overlapping_weight_ties_form_one_constraint():
    graph = cd.Connectome.from_synapses(3, pre=[0, 1, 1, 2], post=[1, 0, 2, 1])
    # Explicitly tie one direction of each reciprocal pair: all four must move together.
    groups = np.where(graph.pre < graph.post, 1000000000, -1)
    learner = cd.Learner(
        cd.NeuralGraph(graph, cd.learning_neuron_model()),
        [2],
        tie_groups=groups,
        plastic_neurons=np.zeros(3, bool),
    )
    assert learner.parameters() == 1
    before = learner.brain.efficacy.copy()
    learner.apply(np.arange(4, dtype=float) * 0.01, np.zeros(3))
    np.testing.assert_allclose(learner.brain.efficacy - before, np.full(4, 0.015))


def test_generic_brain_exposes_current_learned_parameters():
    brain = cd.Brain.compose(2, 2, modules=(3,), recall=False)
    brain.fit(np.eye(2), np.array([0, 1]), epochs=1)
    assert brain.brain is brain.learner.brain


def test_generic_brain_owns_action_and_observation_buffers():
    brain = cd.Brain.compose(2, 2, modules=(3,), episodic=True, recall=False)
    cue = np.array([[1.0, 0.0]])
    action = brain.act(cue)
    chosen = int(action[0])
    cue[:] = [[0.0, 1.0]]
    action[:] = 1 - chosen
    brain.learn(np.ones(1), np.ones(1, bool), cue)
    remembered = brain.hippocampus.recall(np.array([[1.0, 0.0]]))
    assert remembered[0, chosen] == pytest.approx(1)
    np.testing.assert_allclose(brain.hippocampus.recall(cue), 0)


@pytest.mark.parametrize("labels", [[0.9, 1.0], [[0], [1]], [0]])
def test_generic_fit_rejects_invalid_labels_before_learning(labels):
    brain = cd.Brain.compose(2, 2, modules=(3,), recall=False)
    with pytest.raises(ValueError):
        brain.fit(np.eye(2), labels, epochs=1)
    assert brain.learner.updates == 0


def test_bad_transition_does_not_change_memory_or_eligibility():
    brain = cd.Brain.compose(2, 2, modules=(3,), episodic=True, recall=False)
    brain.act(np.eye(2))
    trace = brain.working_memory.trace.copy()
    with pytest.raises(ValueError):
        brain.learn(np.ones(2), np.ones(2, bool), np.ones((2, 4)))
    np.testing.assert_array_equal(brain.working_memory.trace, trace)
    assert brain.hippocampus.writes == 0
    assert brain.basal_ganglia.trace is None


@pytest.mark.parametrize(
    "field,value",
    [("gamma", 1.1), ("lam", -1), ("eta", float("nan")), ("eta_critic", -1), ("dopamine_cap", -1)],
)
def test_reward_configuration_is_validated(field, value):
    with pytest.raises(ValueError):
        cd.ActorCriticConfig(**{field: value})


@pytest.mark.parametrize("softmax", [False, True])
def test_partial_nudge_masks_agree_with_numpy_reference(monkeypatch, softmax):
    import cadence.brain as module

    if not module._FUSED:
        pytest.skip("Numba unavailable")
    graph = cd.layered(2, 3, 2, density=1)
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model())
    nudge = cd.Nudge(
        np.ones(graph.n),
        np.linspace(0, 1, graph.n),
        0.3,
        softmax_temperature=0.2 if softmax else None,
        groups=np.array([-1, -1, 2, 2, 5, 5, -1]),
    )
    drive = np.ones((2, graph.n))
    fast = brain.settle_batch(drive, steps=10, nudge=nudge)
    monkeypatch.setattr(module, "_FUSED", False)
    slow = brain.settle_batch(drive, steps=10, nudge=nudge)
    np.testing.assert_allclose(fast.activation, slow.activation, atol=1e-12)


@pytest.mark.parametrize("backend", ["cpu", "torch", "mlx"])
def test_replacing_parameters_updates_the_running_backend(backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    graph = cd.layered(2, 3, 2, density=1)
    brain = cd.NeuralGraph(
        graph,
        cd.learning_neuron_model(),
        backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    drive = np.zeros((1, graph.n))
    brain.settle_batch(drive)
    brain.efficacy = np.zeros(graph.synapses)
    brain.bias = np.ones(graph.n)
    result = brain.settle_batch(drive, steps=60)
    np.testing.assert_allclose(
        result.activation, brain.neuron_model.activation(np.ones((1, graph.n))), atol=1e-6
    )


@pytest.mark.parametrize("field", ["drive", "mask", "target", "warm"])
def test_nonfinite_state_is_rejected_before_settling(field):
    agent = controller()
    brain = agent.learner.brain
    drive = np.ones((1, agent.n))
    with pytest.raises(ValueError, match="finite"):
        if field == "drive":
            drive[0, 0] = np.nan
            brain.settle_batch(drive)
        elif field == "mask":
            brain.settle_batch(drive, mask=np.full(agent.n, np.nan))
        elif field == "target":
            cd.Nudge(np.full(agent.n, np.nan), np.ones(agent.n), 0.1)
        else:
            warm = cd.BrainState(np.full((1, agent.n), np.nan), drive, drive, 0)
            brain.settle_batch(drive, state=warm)


def test_equilibrate_checks_equations_and_never_overspends():
    graph = cd.Connectome.from_synapses(1, pre=[], post=[])
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(dt=0.01))
    drive = np.array([[100.0]])  # activity saturates long before the potential reaches 100
    quiet = brain.settle_batch(drive, steps=1000, tolerance=1e-5)
    assert brain.residual(drive, quiet)[0] > 1
    capped = brain.equilibrate(drive, budget=37, chunk=16, tolerance=1e-4)
    assert capped.steps == 37 and not capped.converged[0]
    solved = brain.equilibrate(drive, state=capped.state, budget=2000, tolerance=1e-4)
    assert solved.converged.all()
    np.testing.assert_array_equal(solved.residual, brain.residual(drive, solved.state))
    zero = brain.equilibrate(np.zeros_like(drive), budget=0)
    assert zero.steps == 0 and zero.converged.all()


def test_equilibrate_checks_the_same_nudge_and_ablation():
    agent = controller()
    brain = agent.learner.brain
    drive = np.ones((2, agent.n))
    mask = np.ones_like(drive)
    mask[1, -1] = 0
    nudge = agent.learner.nudge_for(agent.learner.targets(np.array([0, 1])), 0.1)
    result = brain.equilibrate(drive, mask=mask, nudge=nudge, tolerance=1e-7)
    assert result.converged.all() and result.state.activation[1, -1] == 0
    np.testing.assert_array_equal(
        result.residual, brain.residual(drive, result.state, mask=mask, nudge=nudge)
    )


def test_genome_rejects_colliding_region_names_and_changed_ports():
    from cadence.regions import Region

    with pytest.raises(ValueError, match="unique"):
        cd.Genome((Region("same", 2), Region("same", 3)), ())
    region = cd.regions.visual_cortex(4, 4)
    genome = cd.Genome((region,), ())
    from dataclasses import replace

    with pytest.raises(ValueError, match="differs"):
        cd.Genome.from_dict(
            genome.to_dict(), designed={region.name: replace(region, outputs="input")}
        )


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_complete_brain_checkpoint_resumes_learning_and_memory(tmp_path, backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    original = cd.Brain.compose(
        2,
        2,
        modules=(3,),
        episodic=True,
        backend=backend,
        device="cpu" if backend == "torch" else None,
        reward=cd.ActorCriticConfig(), recall=False,
    )
    drive = np.eye(2)
    for _ in range(3):
        original.act(drive)
        original.learn(np.array([1.0, -0.1]), np.zeros(2, bool), drive)
    path = original.save(tmp_path / "complete.npz")
    restored = cd.Brain.load(
        path, backend=backend, device="cpu" if backend == "torch" else None
    )
    assert restored.brain is restored.learner.brain
    np.testing.assert_array_equal(original.hippocampus.strength, restored.hippocampus.strength)
    np.testing.assert_array_equal(original.working_memory.trace, restored.working_memory.trace)
    for _ in range(3):
        np.testing.assert_array_equal(original.act(drive), restored.act(drive))
        for brain in (original, restored):
            brain.learn(np.array([0.5, -0.5]), np.zeros(2, bool), drive)
        np.testing.assert_allclose(original.brain.efficacy, restored.brain.efficacy, atol=1e-10)
        np.testing.assert_allclose(
            original.basal_ganglia.w_critic, restored.basal_ganglia.w_critic, atol=1e-10
        )


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    brain = cd.Brain.compose(2, 2, modules=(3,), recall=False)
    path = brain.save(tmp_path / "safe.npz")
    before = path.read_bytes()

    def fail(handle, **kwargs):
        handle.write(b"partial archive")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk full"):
        brain.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_pending_action_cannot_be_silently_lost_on_save(tmp_path):
    brain = cd.Brain.compose(2, 2, modules=(3,), recall=False)
    brain.act(np.eye(2))
    restored = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert restored.basal_ganglia._pending is not None
    for model in (brain, restored):
        model.learn(np.ones(2), np.ones(2, bool), np.eye(2))
    np.testing.assert_array_equal(brain.brain.efficacy, restored.brain.efficacy)


def test_signed_working_memory_has_bounded_nonnegative_salience():
    graph = cd.Connectome.from_synapses(
        4, pre=[], post=[], populations={"hidden": [0, 1], "context": [2, 3]}
    )
    trace = cd.Trace(graph, decay=0)
    s = np.array([[0.5, -0.5, 0, 0]])
    trace.update(cd.BrainState(s, s, np.zeros_like(s), 1))
    np.testing.assert_allclose(trace.ringing()[0, :2], [1.1, 1.1])
    assert np.isfinite(trace.ringing()).all()


def test_zero_density_builds_a_disconnected_sensory_projection():
    graph = cd.layered(2, 3, 2, density=0)
    assert not np.isin(graph.pre, graph.populations["input"]).any()
    assert graph.synapses > 0


def test_corrupt_optimizer_state_is_rejected_on_load(tmp_path):
    learner = controller().learner
    path = learner.save(tmp_path / "invalid.npz")
    with np.load(path, allow_pickle=False) as data:
        payload = dict(data)
    payload["second_moment"][0] = -1
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="second moment"):
        cd.Learner.load(path)


def test_sparse_learning_never_uses_dense_contrast_and_matches_torch(monkeypatch):
    import cadence.learning as module

    def forbidden(*args):
        raise AssertionError("dense Gram allocation for a sparse graph")

    monkeypatch.setattr(module, "block_contrast", forbidden)
    graph = cd.layered(4, 5, 3, density=1)
    results = []
    for backend in ("cpu", "torch"):
        if backend not in cd.available_backends():
            continue
        brain = cd.NeuralGraph(
            graph,
            cd.learning_neuron_model(),
            dense_limit=0,
            backend=backend,
            device="cpu" if backend == "torch" else None,
        )
        learner = cd.Learner(brain, graph.populations["output"])
        learner.step(np.ones((2, graph.n)), np.array([0, 1]))
        results.append(learner.brain.efficacy)
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], atol=1e-10)


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_bias_only_learning_has_finite_reports(backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    graph = cd.Connectome.from_synapses(2, pre=[], post=[])
    brain = cd.NeuralGraph(
        graph,
        cd.learning_neuron_model(),
        backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    learner = cd.Learner(brain, [0, 1])
    _, report = learner.step(np.zeros((1, 2)), np.array([0]))
    assert np.isfinite(list(report.values())).all() and report["scale_step"] == 0
