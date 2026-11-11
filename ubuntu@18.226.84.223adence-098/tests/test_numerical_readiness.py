"""Adversarial finite-state and equilibrium admission boundaries."""

import numpy as np
import pytest

import cadence as cd


@pytest.mark.parametrize("parameter", ["efficacy", "bias"])
@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_device_precision_overflow_is_rejected_at_construction(parameter, backend):
    pytest.importorskip(backend)
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1])
    value = np.full(1 if parameter == "efficacy" else 2, 1e100)
    with pytest.raises(ValueError, match="runtime precision"):
        cd.NeuralGraph(graph, cd.learning_neuron_model(), backend=backend, device="cpu",
                 precision="float32", **{parameter: value})


@pytest.mark.parametrize("invalid", ["scale", "bias", "product"])
def test_device_replacement_failure_leaves_shared_kernel_intact(invalid):
    torch = pytest.importorskip("torch")
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1], count=[1e100])
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend="torch", device="cpu")
    kernel = brain._torch
    before = {key: getattr(kernel, key).clone() for key in ("scale", "bias_param", "flat", "bias")}
    scale, bias = kernel.scale.clone(), kernel.bias_param.clone()
    if invalid == "scale":
        scale.fill_(float("nan"))
    elif invalid == "bias":
        bias.fill_(float("inf"))
    else:
        scale.fill_(1e300)  # finite parameters, unrepresentable effective product
    with pytest.raises(ValueError, match="finite"):
        brain._with_device_parameters(scale, bias)
    for key, value in before.items():
        assert torch.equal(getattr(kernel, key), value)


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_rejected_update_keeps_optimizer_history_and_retry(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1])
    learner = cd.Learner(
        cd.NeuralGraph(graph, cd.learning_neuron_model(), backend=backend,
                 **({"device": "cpu"} if backend == "torch" else {})),
        [1], cd.LearnerConfig(momentum=.5, normalize=.5, eta_bias=1e308),
    )
    free = learner.brain.settle_batch(np.zeros((1, 2)), steps=0)
    plus = learner.brain.settle_batch(np.full((1, 2), 3.), steps=10)
    names = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
    before = {name: getattr(learner, name).copy() for name in names}
    scale, bias = learner.brain.efficacy.copy(), learner.brain.bias.copy()
    # A finite update overflows when added to the existing finite bias.
    learner.brain.bias = np.full(2, 1.7e308)
    bias = learner.brain.bias.copy()
    # Recreate resident phases on the replacement kernel for the device path.
    if backend == "torch":
        free = learner.brain.settle_batch(np.zeros((1, 2)), steps=0)
        plus = learner.brain.settle_batch(np.zeros((1, 2)), steps=1)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="finite"):
        learner.update(free, plus)
    for name, saved in before.items():
        np.testing.assert_array_equal(getattr(learner, name), saved)
    np.testing.assert_array_equal(learner.brain.efficacy, scale)
    np.testing.assert_array_equal(learner.brain.bias, bias)
    assert learner.updates == learner.contrast_updates == 0


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_critic_overflow_rejects_the_actor_trace_atomically(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    graph = cd.layered(1, 2, 2, seed=3)
    learner = cd.Learner(
        cd.NeuralGraph(graph, cd.learning_neuron_model(), backend=backend,
                 **({"device": "cpu"} if backend == "torch" else {})),
        graph.populations["output"],
    )
    # uncapped, the critic's step on a reward of ten overflows
    actor = cd.ActorCritic(learner, graph.populations["hidden"],
                         cd.ActorCriticConfig(eta_critic=1e308, dopamine_cap=0.0))
    drive = np.full((1, graph.n), .1)
    actor.act(drive)
    pending = actor._pending
    weights, bias = learner.brain.efficacy.copy(), learner.brain.bias.copy()
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="critic"):
        actor.learn(np.array([10.]), np.array([True]), drive)
    assert actor._pending is pending
    assert actor.trace is actor.trace_bias is actor.trace_critic is actor._trace_device is None
    assert actor.updates == learner.updates == 0
    assert actor.b_critic == 0 and not actor.w_critic.any()
    np.testing.assert_array_equal(learner.brain.efficacy, weights)
    np.testing.assert_array_equal(learner.brain.bias, bias)
    actor.learn(np.array([0.]), np.array([True]), drive)
    assert actor.updates == learner.updates == 1


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -1])
def test_invalid_contact_filter_cannot_silently_drop_the_graph(threshold):
    with pytest.raises(ValueError, match="min_count"):
        cd.Connectome.from_synapses(2, pre=[0], post=[1], min_count=threshold)


def test_negative_contact_counts_are_not_signed_weights():
    with pytest.raises(ValueError, match="nonnegative"):
        cd.Connectome.from_synapses(2, pre=[0], post=[1], count=[-1])


def test_topology_is_read_only_and_population_changes_construct_a_new_graph():
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1], populations={"input": [0]})
    before = graph.digest()
    for name in ("pre", "post", "count", "sign"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(graph, name)[0] = 0
    with pytest.raises(TypeError):
        graph.populations["other"] = (1,)
    changed = graph.with_populations(other=[1])
    assert graph.digest() == before and changed.populations["other"] == (1,)


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_parameter_views_cannot_bypass_transport_and_setters_remain_effective(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1])
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend=backend,
                     **({"device": "cpu"} if backend == "torch" else {}))
    drive = np.array([[1., 0.]])
    initial = brain.settle_batch(drive, steps=10).activation.copy()
    for name in ("efficacy", "bias", "log_gain", "weights"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(brain, name)[0] = 99
    np.testing.assert_array_equal(brain.settle_batch(drive, steps=10).activation, initial)
    brain.efficacy = np.array([2.])
    brain.bias = np.array([.1, .2])
    brain.log_gain = np.array([.3, 0.])
    reference = cd.NeuralGraph(graph, brain.neuron_model, efficacy=brain.efficacy,
                         bias=brain.bias, log_gain=brain.log_gain)
    np.testing.assert_allclose(brain.settle_batch(drive, steps=10).activation,
                               reference.settle_batch(drive, steps=10).activation, atol=1e-12)
    assert not np.array_equal(initial, reference.settle_batch(drive, steps=10).activation)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize("model", [
    cd.NeuronModel(threshold=-5),  # finite float64 rest rounds to1 in float32: divide by0
    cd.NeuronModel(slope=1e100, threshold=1e-100),  # runtime slope overflows
    cd.NeuronModel(dt=1e-100),  # Euler integration would become an unreported no-op
])
def test_runtime_precision_cannot_destroy_a_valid_float64_neuron_rule(backend, model):
    pytest.importorskip(backend)
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1])
    with pytest.raises(ValueError, match="runtime precision"):
        cd.NeuralGraph(graph, model, backend=backend, device="cpu", precision="float32")
