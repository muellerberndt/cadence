"""Stalled numerical orbits skip work, never the original equation qualification."""

import numpy as np
import pytest

import cadence as cd


def motor_graph(backend="cpu", precision=None):
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "mlx":
        pytest.importorskip("mlx.core")
    pre, post = np.where(~np.eye(36, dtype=bool))
    graph = cd.Connectome.from_synapses(
        36, pre=pre, post=post, sign=np.full(len(pre), -0.5),
    )
    return cd.NeuralGraph(
        graph, cd.learning_neuron_model(dt=1), backend=backend,
        device="cpu" if backend == "torch" else None, precision=precision,
    )


def equal_portion_reference(graph, drive, budget):
    """The previous bounded strategy, retaining every equation check and sweep."""
    used, state, answer = 0, None, None
    for halving in range(4):
        candidate = cd.NeuralGraph(
            graph.connectome, graph.neuron_model.replace(dt=2.0 ** -halving),
            efficacy=graph.efficacy, log_gain=graph.log_gain, bias=graph.bias,
        )
        answer = candidate.equilibrate(
            drive, state=state, budget=(budget - used + 3 - halving) // (4 - halving),
            tolerance=3e-3,
        )
        used += answer.steps
        state = answer.state
        if np.all(graph.residual(drive, state) <= 3e-3):
            break
    assert answer is not None
    return used, state


@pytest.mark.parametrize("backend", ["cpu", "torch", "mlx"])
def test_stalled_orbit_qualifies_original_equations_with_fewer_sweeps(backend):
    graph = motor_graph(backend)
    drive = np.full((2, 36), 0.2)
    answer = graph.equilibrate(drive, budget=4096, tolerance=3e-3, damping=3)
    assert answer.qualified.all() and answer.stagnation_checks > 0
    assert answer.damping_halvings == 3 and graph.neuron_model.dt == 1

    # Evaluate a literal sigmoid and explicit dense equation independently of
    # production activation, residual and transport, including its activity cache.
    model = graph.neuron_model
    emission = 1 / (1 + np.exp(-model.slope * (answer.state.v - model.threshold)))
    relative = emission - model.rest_emission
    activity = np.maximum(relative, 0) / (1 - model.rest_emission)
    activity += model.leak * np.minimum(relative, 0) / model.rest_emission
    matrix = np.zeros((36, 36))
    np.add.at(matrix, (graph.connectome.pre, graph.connectome.post), graph.weights)
    assert np.abs(activity @ matrix + drive + graph.bias - answer.state.v).max() <= 3e-3
    np.testing.assert_allclose(answer.state.activation, activity, atol=2e-6, rtol=0)
    if backend == "cpu":
        previous_steps, previous_state = equal_portion_reference(graph, drive, 4096)
        assert answer.steps < previous_steps // 2
        np.testing.assert_allclose(answer.state.v, previous_state.v, atol=3e-3, rtol=0)


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 31, 64, 256, 4096])
def test_every_executed_sweep_transport_and_state_comparison_is_counted(monkeypatch, budget):
    graph = motor_graph()
    executed, checks, comparisons = [], [], []
    settle, residual, repeated = (
        cd.NeuralGraph.settle_batch, cd.NeuralGraph.residual, cd.NeuralGraph._repeated_state,
    )

    def counted_settle(owner, *args, **kwargs):
        state = settle(owner, *args, **kwargs)
        executed.append(state.steps)
        return state

    def counted_residual(owner, *args, **kwargs):
        checks.append(owner.neuron_model.dt)
        return residual(owner, *args, **kwargs)

    def counted_comparison(owner, *args, **kwargs):
        comparisons.append(owner.neuron_model.dt)
        return repeated(owner, *args, **kwargs)

    monkeypatch.setattr(cd.NeuralGraph, "settle_batch", counted_settle)
    monkeypatch.setattr(cd.NeuralGraph, "residual", counted_residual)
    monkeypatch.setattr(cd.NeuralGraph, "_repeated_state", counted_comparison)
    answer = graph.equilibrate(np.full((2, 36), 0.2), budget=budget, damping=3, tolerance=3e-3)
    assert answer.steps == sum(executed) <= budget
    assert answer.residual_checks == len(checks)
    assert answer.stagnation_checks == len(comparisons)
    if not answer.qualified.all():
        assert answer.steps == budget
    if budget == 0:
        assert answer.residual_checks == 1 and not answer.qualified.any()


@pytest.mark.parametrize("backend", ["cpu", "torch", "mlx"])
def test_slow_adaptation_is_not_a_repeated_complete_state(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "mlx":
        pytest.importorskip("mlx.core")
    graph = cd.NeuralGraph(
        cd.Connectome.from_synapses(2, pre=[], post=[]),
        cd.learning_neuron_model(dt=1).replace(
            adaptation=cd.Adaptation(tau_steps=1e6, strength=0),
        ),
        backend=backend, device="cpu" if backend == "torch" else None,
    )
    drive = np.ones((1, 2))
    warm = graph.settle_batch(drive, steps=32)
    later = graph.settle_batch(drive, state=warm, steps=32)
    np.testing.assert_array_equal(warm.v, later.v)
    np.testing.assert_array_equal(warm.activation, later.activation)
    assert not graph._repeated_state(warm, later)
    answer = graph.equilibrate(drive, state=warm, budget=1024, damping=3, tolerance=3e-3)
    assert answer.steps == 1024 and not answer.qualified.any()
    assert answer.residual.min() > 0.4


def test_undamped_solver_never_uses_stagnation_to_end_its_budget(monkeypatch):
    graph = motor_graph()

    def forbidden(*args, **kwargs):
        raise AssertionError("the undamped algorithm must keep its declared sweep behavior")

    monkeypatch.setattr(cd.NeuralGraph, "_repeated_state", forbidden)
    answer = graph.equilibrate(np.full((1, 36), 0.2), budget=256, damping=0, tolerance=3e-3)
    assert answer.steps == 256 and answer.stagnation_checks == 0
    assert not answer.qualified.any()


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_torch_comparison_keeps_complete_arrays_on_device(precision):
    graph = motor_graph("torch", precision)
    drive = np.full((1, 36), 0.2)
    first = graph.settle_batch(drive, steps=128)
    second = graph.settle_batch(drive, state=first, steps=32)
    assert graph._repeated_state(first, second)
    for state in (first, second):
        assert all(state.__dict__[name] is None for name in ("v", "activation", "adaptation"))


def test_stagnation_does_not_qualify_a_stale_cached_activity():
    graph = motor_graph()
    drive = np.full((1, 36), 0.2)
    warm = graph.settle_batch(drive, steps=128)
    stale = cd.BrainState(warm.v, np.zeros_like(warm.activation), warm.adaptation, warm.steps)
    answer = graph.equilibrate(drive, state=stale, budget=0, damping=3, tolerance=3e-3)
    assert not answer.qualified.any() and answer.stagnation_checks == 0
    np.testing.assert_array_equal(answer.state.activation, graph.neuron_model.activation(warm.v))


@pytest.mark.parametrize("backend", ["cpu", "torch", "mlx"])
def test_a_qualified_row_does_not_hide_another_rows_orbit(backend):
    graph = motor_graph(backend)
    drive = np.vstack((np.zeros(36), np.full(36, 0.2)))
    answer = graph.equilibrate(drive, budget=4096, damping=3, tolerance=3e-3)
    assert answer.qualified.all() and answer.damping_halvings == 3
    assert answer.stagnation_checks > 0
    np.testing.assert_array_equal(answer.state.v[0], 0)
    np.testing.assert_array_equal(answer.state.activation[0], 0)
    assert graph.residual(drive, answer.state).max() <= 3e-3


def test_the_ladders_rule_check_is_made_once_per_rule_and_refuses_at_every_call():
    """The smallest step's rule is checked by value: a rule that passed is not checked again,
    while one the backend cannot represent, or a step that rounds to zero, is refused at every
    call, as it was when every call checked it."""
    from cadence import brain as brain_module

    graph = motor_graph()
    drive = np.full((1, 36), 0.4)
    brain_module._check_ladder.cache_clear()
    for _ in range(3):
        graph.equilibrate(drive, budget=40, tolerance=1e-12, damping=2)
    info = brain_module._check_ladder.cache_info()
    assert info.misses == 1 and info.hits == 2
    saturated = graph.neuron_model.replace(slope=4.0, threshold=-5.0)  # rests at one in float32
    for _ in range(2):
        with pytest.raises(ValueError, match="not representable"):
            brain_module._check_ladder(saturated, 1, "float32")
    brain_module._check_ladder(saturated, 1, "float64")
    for _ in range(2):
        with pytest.raises(ValueError, match="dt must lie"):
            graph.equilibrate(drive, budget=4000, tolerance=1e-12, damping=1200)
    assert brain_module._check_ladder.cache_info().currsize == 2


def test_a_ladder_takes_a_rule_whose_fields_are_arrays():
    """A neuron rule built from 0-d arrays (as ``np.load`` hands scalars back) cannot be a
    cache key; its damping ladder is checked at every call and solves as the rule of floats
    does."""
    brain = cd.Brain.compose(8, 4, modules=(16,), seed=2)
    graph = brain.brain
    fields = {k: v for k, v in graph.neuron_model.to_dict().items() if k != "adaptation"}
    arrays = cd.NeuronModel(**{k: np.asarray(v) for k, v in fields.items()})
    floats = cd.NeuronModel(**fields)
    drive = brain.stimulus(np.random.default_rng(0).random((2, 8)), memory=False)
    solved = []
    for model in (arrays, floats):
        g = cd.NeuralGraph(graph.connectome, model, efficacy=graph.efficacy, bias=graph.bias,
                           layout=graph.layout)
        solved.append(g.equilibrate(drive, budget=96, chunk=16, tolerance=1e-6, damping=2))
    assert solved[0].steps == solved[1].steps
    assert np.array_equal(solved[0].state.activation, solved[1].state.activation)
