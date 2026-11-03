"""Independent energy, derivative, qualification and counterexample checks."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

spec = importlib.util.spec_from_file_location(
    "coupled_attention", Path(__file__).with_name("coupled.py"))
assert spec is not None and spec.loader is not None
coupled = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = coupled
spec.loader.exec_module(coupled)
AttentionBrain, Graph, Memory = coupled.AttentionBrain, coupled.Graph, coupled.Memory


def boundary(mode: str = "agreement") -> Memory:
    rng = np.random.default_rng(72)
    return Memory(np.array([0, 1]), np.array([2, 3, 4]),
                  rng.normal(size=(2, 4, 2)), rng.normal(size=(2, 4, 3)),
                  np.array([[True, False, True, True], [False, False, False, False]]),
                  np.arange(4.0), amplitude=0.3, temperature=0.8, recency=0.1,
                  null=-0.2, mode=mode)


@pytest.mark.parametrize("mode", ["agreement", "linear", "competitive"])
def test_drive_is_negative_energy_derivative_and_jacobian_is_symmetric(mode: str) -> None:
    memory = boundary(mode)
    state = np.random.default_rng(50).normal(size=(2, 6))
    epsilon = 1e-5
    jacobian = []
    for column in range(state.shape[1]):
        delta = np.zeros_like(state)
        delta[:, column] = epsilon
        derivative = (memory.energy(state + delta) - memory.energy(state - delta)) / (2 * epsilon)
        np.testing.assert_allclose(memory.drive(state)[:, column], -derivative, atol=2e-9)
        jacobian.append((memory.drive(state + delta) - memory.drive(state - delta)) / (2 * epsilon))
    jacobian = np.stack(jacobian, axis=-1)
    np.testing.assert_allclose(jacobian, jacobian.swapaxes(1, 2), atol=2e-9)
    assert np.array_equal(memory.drive(state)[1], np.zeros(6))  # absent stream


def test_snapshot_is_independent_and_attention_respects_absent_slots() -> None:
    memory = boundary()
    old = memory.keys.copy()
    with pytest.raises(ValueError):
        memory.keys[:] = 0
    assert np.array_equal(memory.keys, old)
    p = memory.weights(np.ones((2, 6)))
    assert p[0, 1] == 0 and (p[1] == 0).all()
    assert 0 < p[0].sum() < 1  # null retained
    empty = replace(memory, keys=np.zeros((2, 0, 2)), values=np.zeros((2, 0, 3)),
                    held=np.zeros((2, 0), dtype=bool), ages=np.zeros(0))
    assert empty.weights(np.zeros((2, 6))).shape == (2, 0)
    assert not empty.drive(np.ones((2, 6))).any()


def tiny_graph() -> Graph:
    connectome = cd.Connectome.from_synapses(
        4, pre=[0, 2, 1, 2, 2, 3], post=[2, 0, 2, 1, 3, 2], sign=[0.12] * 6)
    graph = Graph(connectome, cd.learning_neuron_model(dt=0.1), bias=np.full(4, 0.1))
    graph.memory = Memory(np.array([0]), np.array([1]),
                          np.array([[[-1.0], [1.0]]]), np.array([[[-0.7], [0.8]]]),
                          np.ones((1, 2), dtype=bool), np.array([1.0, 0.0]),
                          amplitude=0.4, temperature=0.9)
    return graph


def test_local_query_synapse_contrast_matches_independent_answer_gradient() -> None:
    graph = tiny_graph()
    config = cd.LearnerConfig(beta=1e-4, nudge="quadratic", free_steps=1500, nudged_steps=1500,
                             tolerance=1e-12, qualified=True, damping=0)
    learner = cd.Learner(graph, [2, 3], config)
    drive = np.array([[0.4, 0.0, 0.1, 0.2]])
    target = np.array([[0.0, 0.0, 0.2, 0.7]])
    free = learner.free(drive)
    plus = learner.nudged(drive, free, target)
    minus = learner.nudged(drive, free, target, sign=-1.0)
    contrast, _ = learner.contrast(free, plus, minus)
    for i, j in ((0, 2), (1, 2), (2, 3)):
        pair = ((graph.connectome.pre == i) & (graph.connectome.post == j)) | (
            (graph.connectome.pre == j) & (graph.connectome.post == i))

        def loss(delta: float, pair: np.ndarray = pair) -> float:
            weight = graph.efficacy.copy()
            weight[pair] += delta
            changed = graph.with_parameters(efficacy=weight)
            assert isinstance(changed, Graph) and changed.memory is graph.memory
            state = changed.equilibrate(drive, budget=1500, tolerance=1e-12)
            assert state.qualified.all()
            return float(0.5 * ((state.state.activation[:, [2, 3]] - target[:, [2, 3]]) ** 2).sum())

        numerical = -(loss(1e-5) - loss(-1e-5)) / 2e-5
        np.testing.assert_allclose(contrast[pair], numerical, rtol=3e-4, atol=1e-8)
    assert graph.residual(drive, free).max() < 1e-11
    # The same state is not a solution of the graph with its memory term removed.
    graph.memory = None
    assert graph.residual(drive, free).max() > 0.01


def test_read_disagreement_changes_query_and_selection_during_the_same_solve() -> None:
    graph = tiny_graph()
    assert graph.memory is not None
    drive = np.array([[0.4, 0.0, 0.1, 0.2]])
    phase = graph.settle_batch(drive, steps=500, trajectory=True)
    trajectory = phase.trajectory
    assert trajectory is not None
    initial = graph.memory.weights(trajectory[0])
    final = graph.memory.weights(trajectory[-1])
    assert np.max(np.abs(final - initial)) > 0.01
    state = np.atleast_2d(phase.activation)
    changed = state.copy()
    changed[:, graph.memory.read] += 0.3
    before, after = graph.memory.drive(state), graph.memory.drive(changed)
    assert abs(float((after - before)[0, 0])) > 0.01


def test_competitive_memory_has_unsupported_nonzero_equilibria_even_with_a_null() -> None:
    """A qualified content completion is not evidence that it was warranted."""
    # One scalar read, two contradictory records, no query evidence. This is
    # the old positive-feedback failure; the null makes no general cure.
    memory = Memory(np.array([0]), np.array([1]), np.zeros((1, 2, 1)),
                    np.array([[[-1.0], [1.0]]]), np.ones((1, 2), dtype=bool), np.zeros(2),
                    amplitude=3.0, temperature=0.25, mode="competitive")
    graph = Graph(cd.Connectome.from_synapses(2, pre=[], post=[]),
                  cd.learning_neuron_model(dt=0.1, leak=1.0))
    graph.memory = memory
    answers = []
    for value in (-0.01, 0.0, 0.01):
        v = np.array([[0.0, value]])
        warm = cd.BrainState(v, graph.neuron_model.activation(v), np.zeros_like(v), 0)
        phase = graph.equilibrate(np.zeros_like(v), state=warm, budget=1500, tolerance=1e-10)
        assert phase.qualified.all()
        answers.append(float(phase.state.activation[0, 1]))
    assert answers[0] < -0.5 and answers[1] == 0 and answers[2] > 0.5
    # With identical data, restoring agreement goes back to the uninformative mixture.
    graph.memory = replace(memory, mode="agreement")
    for value in (-0.01, 0.0, 0.01):
        v = np.array([[0.0, value]])
        warm = cd.BrainState(v, graph.neuron_model.activation(v), np.zeros_like(v), 0)
        phase = graph.equilibrate(np.zeros_like(v), state=warm, budget=1500, tolerance=1e-10)
        assert phase.qualified.all()
        assert np.abs(phase.state.activation).max() < 1e-9


@pytest.mark.parametrize("successor", [False, True])
def test_experimental_brain_reads_witnessed_events_and_qualifies_joint_state(
    successor: bool,
) -> None:
    brain = AttentionBrain.compose(3, 2, modules=(4,), episodic=False,
                                   recall_next=successor,
                                   learning_free_steps=256, learning_nudged_steps=256)
    brain.act([[1, 0, 0]], greedy=True)
    brain.act([[0, 1, 0]], greedy=True)
    assert brain.recall is not None and brain.recall.slots == 2
    assert isinstance(brain.brain, Graph)
    assert brain.brain.memory is not None and brain.brain.memory.keys.shape[1] == 1
    before = brain.recall.keys.copy()
    pre, post = brain.connectome.pre, brain.connectome.post
    query = np.asarray(brain.connectome.populations["query"])
    association = np.asarray(brain.connectome.populations["association"])
    joins = ((np.isin(pre, query) & np.isin(post, association))
             | (np.isin(post, query) & np.isin(pre, association)))
    assert not brain.brain.efficacy[joins].any()
    report = brain.lesson([[0, 0, 1]], [1])
    assert report["accepted_presentations"] == 1
    assert np.max(np.abs(brain.brain.efficacy[joins])) > 1e-10
    assert np.array_equal(brain.recall.keys, before)  # teacher is not witnessed content
    brain.act([[0, 0, 1]], greedy=True)
    assert brain.last_settlement is not None and brain.last_settlement["qualified"]
    assert brain.recall.slots == 3


def test_linear_control_does_not_report_an_unused_softmax_as_its_selection() -> None:
    brain = AttentionBrain.compose(3, 2, mode="linear", modules=(4,), episodic=False)
    brain.act([[1, 0, 0]], greedy=True)
    brain.act([[0, 1, 0]], greedy=True)
    assert brain.recall is not None and brain.recall.attended_heads is None


def test_all_parameter_copies_preserve_the_same_witnessed_boundary() -> None:
    graph = tiny_graph()
    for changed in (graph.with_parameters(bias=graph.bias + 0.1),
                    graph.with_parameters(log_gain=graph.log_gain + 0.01)):
        assert isinstance(changed, Graph) and changed.memory is graph.memory


def test_compiled_path_cannot_silently_omit_memory(monkeypatch) -> None:
    import cadence.brain as brain_module

    graph = tiny_graph()
    drive = np.array([[0.4, 0.0, 0.1, 0.2]])
    monkeypatch.setattr(brain_module, "_FUSED", False)
    reference = graph.equilibrate(drive, budget=1500, tolerance=1e-10)
    monkeypatch.setattr(brain_module, "_FUSED", True)
    actual = graph.equilibrate(drive, budget=1500, tolerance=1e-10)
    assert actual.qualified.all()
    np.testing.assert_array_equal(actual.state.v, reference.state.v)
    np.testing.assert_array_equal(actual.residual, reference.residual)


def test_abstention_and_duplicate_invariance_are_not_established_by_agreement() -> None:
    graph = tiny_graph()
    memory = Memory(np.array([0]), np.array([1]), np.zeros((1, 2, 1)),
                    np.array([[[0.2], [0.8]]]), np.ones((1, 2), dtype=bool), np.zeros(2),
                    amplitude=0.4, temperature=0.9)
    graph.memory = memory
    # No query evidence; the mean of biased contents still drives the read.
    # This candidate must not be advertised as a general cure for unwarranted recall.
    silent = Graph(cd.Connectome.from_synapses(2, pre=[], post=[]),
                   cd.learning_neuron_model(dt=0.1))
    silent.memory = memory
    result = silent.equilibrate(np.zeros((1, 2)), budget=500, tolerance=1e-10)
    assert result.qualified.all() and result.state.activation[0, 1] > 0.01
    state = np.zeros((1, 2))
    duplicated = replace(memory, keys=np.repeat(memory.keys, 2, axis=1),
                         values=np.repeat(memory.values, 2, axis=1),
                         held=np.ones((1, 4), dtype=bool), ages=np.zeros(4))
    assert duplicated.weights(state).sum() > memory.weights(state).sum()


@pytest.mark.parametrize("operation", ["imagine", "save", "learn", "live", "step", "wait"])
def test_lifecycle_calls_require_complete_arguments_before_advancing_memory(operation: str) -> None:
    brain = AttentionBrain.compose(3, 2, modules=(4,), episodic=False)
    assert brain.recall is not None and brain.recall.slots == 0
    with pytest.raises(TypeError):
        getattr(brain, operation)()
    assert brain.recall.slots == 0 and brain.basal_ganglia.state is None


def test_refused_action_restores_the_previous_memory_boundary_and_read() -> None:
    brain = AttentionBrain.compose(3, 2, modules=(4,), episodic=False)
    brain.act([[1, 0, 0]], greedy=True)
    recall, graph = brain.recall, brain.brain
    assert recall is not None and isinstance(graph, Graph)
    reading, previous = recall.reading(), graph.memory
    keys = recall.keys.copy()
    brain.learner.config = replace(brain.learner.config, free_steps=1, tolerance=1e-12)
    with pytest.raises(RuntimeError, match="settle"):
        brain.act([[0, 1, 0]], greedy=True)
    assert graph.memory is previous
    assert recall.reading()[0] is reading[0] and recall.reading()[1] is reading[1]
    np.testing.assert_array_equal(recall.keys, keys)


def test_partial_rows_and_changed_equations_refuse_instead_of_broadcasting_memories() -> None:
    graph = tiny_graph()
    assert graph.memory is not None
    with pytest.raises(ValueError, match="streams"):
        graph.memory.drive(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="damping"):
        graph.equilibrate(np.zeros((1, 4)), damping=-1)
    with pytest.raises(ValueError, match="CPU"):
        Graph(graph.connectome, graph.neuron_model, backend="torch")
