"""Independent energy, symmetry, shared-boundary and single-head checks."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "attention_heads_tests", Path(__file__).with_name("heads.py")
)
assert SPEC is not None and SPEC.loader is not None
heads = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = heads
SPEC.loader.exec_module(heads)
Memory, MultiMemory, MultiHeadBrain = heads.Memory, heads.MultiMemory, heads.MultiHeadBrain


def boundary(mode: str = "agreement") -> MultiMemory:
    rng = np.random.default_rng(81)
    first = Memory(
        np.array([0, 1]), np.array([2, 3, 4]),
        rng.normal(size=(2, 4, 2)), rng.normal(size=(2, 4, 3)),
        np.array([[True, False, True, True], [False, False, False, False]]),
        np.arange(4.0), amplitude=0.3, temperature=0.8, recency=0.1, mode=mode,
    )
    return MultiMemory((first, replace(first, query=np.array([5, 6]), read=np.array([7, 8, 9]))))


@pytest.mark.parametrize("mode", ["agreement", "linear", "competitive"])
def test_summed_drives_are_independent_negative_energy_derivatives(mode: str) -> None:
    memory = boundary(mode)
    state = np.random.default_rng(43).normal(size=(2, 11))
    epsilon = 1e-5
    columns = []
    for column in range(state.shape[1]):
        delta = np.zeros_like(state)
        delta[:, column] = epsilon
        numerical = (memory.energy(state + delta) - memory.energy(state - delta)) / (2 * epsilon)
        np.testing.assert_allclose(memory.drive(state)[:, column], -numerical, atol=3e-9)
        columns.append((memory.drive(state + delta) - memory.drive(state - delta)) / (2 * epsilon))
    jacobian = np.stack(columns, axis=-1)
    np.testing.assert_allclose(jacobian, jacobian.swapaxes(1, 2), atol=3e-9)
    np.testing.assert_array_equal(memory.drive(state)[1], np.zeros(11))


def test_one_head_is_exactly_the_original_equation() -> None:
    original = boundary().memories[0]
    memory = MultiMemory((original,))
    state = np.random.default_rng(2).normal(size=(2, 11))
    np.testing.assert_array_equal(memory.energy(state), original.energy(state))
    np.testing.assert_array_equal(memory.drive(state), original.drive(state))
    np.testing.assert_array_equal(memory.weights(state)[0], original.weights(state))
    assert memory.weights(state).shape == (1, 2, 4)


def test_all_heads_share_one_owned_readonly_witnessed_bank() -> None:
    memory = boundary()
    assert memory.streams == memory.heads == 2
    for name in ("keys", "values", "held", "ages"):
        array = getattr(memory, name)
        assert all(getattr(member, name) is array for member in memory.memories)
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        memory.memories = ()
    assert not memory.memories[0].query.flags.writeable
    assert not memory.memories[1].read.flags.writeable


@pytest.mark.parametrize("selection, expected", [
    (np.array([False, True]), [1]), (np.array([1, 0]), [1, 0]),
    (np.array([], dtype=np.int64), []),
])
def test_row_selection_restricts_the_shared_bank_without_mixing_head_streams(
    selection: np.ndarray, expected: list[int],
) -> None:
    memory = boundary()
    state = np.random.default_rng(9).normal(size=(2, 11))
    taken = memory.take(selection)
    assert taken.streams == len(expected)
    np.testing.assert_array_equal(taken.keys, memory.keys[expected])
    np.testing.assert_array_equal(taken.values, memory.values[expected])
    np.testing.assert_array_equal(taken.held, memory.held[expected])
    np.testing.assert_array_equal(
        taken.weights(state[expected]), memory.weights(state)[:, expected],
    )
    np.testing.assert_array_equal(taken.drive(state[expected]), memory.drive(state)[expected])
    assert taken.memories[0].keys is taken.memories[1].keys
    assert taken.memories[0].held is taken.memories[1].held
    assert taken.memories[0].amplitude == taken.memories[1].amplitude == 0.3
    assert memory.streams == 2


@pytest.mark.parametrize("selection", [
    [True], [0, 0], [-1], [2], [0.5], np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
])
def test_invalid_row_selection_cannot_change_the_boundary(selection) -> None:
    memory = boundary()
    before = memory.keys.copy()
    with pytest.raises(ValueError):
        memory.take(selection)
    np.testing.assert_array_equal(memory.keys, before)


def test_heads_reject_overlapping_ports_or_disagreeing_banks() -> None:
    memory = boundary()
    first, second = memory.memories
    with pytest.raises(ValueError, match="disjoint"):
        MultiMemory((first, first))
    with pytest.raises(ValueError, match="same witnessed"):
        MultiMemory((first, replace(second, values=second.values + 0.1)))
    with pytest.raises(ValueError, match="same witnessed"):
        MultiMemory((first, replace(second, temperature=0.7)))
    with pytest.raises(ValueError, match="nonempty"):
        MultiMemory(())


def test_head_exchange_preserves_energy_and_exchanges_only_head_diagnostics() -> None:
    memory = boundary()
    state = np.random.default_rng(19).normal(size=(2, 11))
    exchanged = MultiMemory(tuple(reversed(memory.memories)))
    np.testing.assert_array_equal(memory.energy(state), exchanged.energy(state))
    np.testing.assert_array_equal(memory.drive(state), exchanged.drive(state))
    np.testing.assert_array_equal(memory.weights(state)[::-1], exchanged.weights(state))
    permutation = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10])
    np.testing.assert_allclose(memory.energy(state[:, permutation]), memory.energy(state))
    np.testing.assert_array_equal(
        memory.drive(state[:, permutation]), memory.drive(state)[:, permutation],
    )


def test_identical_heads_have_a_collapsed_invariant_subspace_without_diversity_pressure() -> None:
    memory = boundary()
    state = np.random.default_rng(55).normal(size=(2, 11))
    state[:, 5:10] = state[:, :5]
    # Identical local equations and starting activities cannot create head specialization.
    for _ in range(8):
        drive = memory.drive(state)
        np.testing.assert_array_equal(drive[:, :5], drive[:, 5:10])
        np.testing.assert_array_equal(memory.weights(state)[0], memory.weights(state)[1])
        state += 0.01 * drive
    np.testing.assert_array_equal(state[:, :5], state[:, 5:10])


def test_erasing_one_head_does_not_erase_the_other_heads_memory_force() -> None:
    memory = boundary()
    state = np.random.default_rng(12).normal(size=(2, 11))
    lesioned = MultiMemory((replace(memory.memories[0], amplitude=0), memory.memories[1]))
    np.testing.assert_array_equal(lesioned.drive(state)[:, :5], np.zeros((2, 5)))
    np.testing.assert_array_equal(lesioned.drive(state)[:, 5:], memory.drive(state)[:, 5:])


def brain_options() -> dict:
    return dict(modules=(4,), episodic=False, recall_next=False,
                learning_free_steps=512, learning_nudged_steps=512)


def test_default_one_head_preserves_anatomy_actions_saved_teaching_and_parameters() -> None:
    original = heads.AttentionBrain.compose(3, 2, seed=2, **brain_options())
    candidate = MultiHeadBrain.compose(3, 2, seed=2, **brain_options())
    for name in ("pre", "post", "count", "sign"):
        np.testing.assert_array_equal(getattr(original.connectome, name),
                                      getattr(candidate.connectome, name))
    for observation in (np.array([[1, 0, 0], [0, 1, 0]]),
                        np.array([[0, 1, 0], [1, 0, 0]])):
        np.testing.assert_array_equal(original.act(observation, greedy=True),
                                      candidate.act(observation, greedy=True))
        np.testing.assert_array_equal(original.basal_ganglia.state.v,
                                      candidate.basal_ganglia.state.v)
    query = np.array([[0, 0, 1], [0, 0, 1]])
    drives = [brain.stimulus(query).copy() for brain in (original, candidate)]
    memories = [brain.brain.memory for brain in (original, candidate)]
    np.testing.assert_array_equal(drives[0], drives[1])
    actions = [brain.act(query, greedy=True) for brain in (original, candidate)]
    np.testing.assert_array_equal(*actions)
    targets = 1 - actions[0]
    lessons = [
        brain.teach_saved(drive, targets, [1], memory)
        for brain, drive, memory in zip((original, candidate), drives, memories, strict=True)
    ]
    assert lessons[0][1] == lessons[1][1]
    for name in ("efficacy", "bias"):
        np.testing.assert_array_equal(getattr(original.brain, name), getattr(candidate.brain, name))
    assert candidate.brain.memory is memories[1]
    assert candidate.head_attended.shape == (1, 2, 2)
    np.testing.assert_array_equal(candidate.head_attended[0], original.recall.attended_heads[0])


@pytest.mark.parametrize("heads_count, copies", [(2, 1), (1, 2), (2, 2)])
def test_head_anatomy_qualifies_joint_state_and_keeps_total_coupling(heads_count: int, copies: int):
    brain = MultiHeadBrain.compose(
        3, 2, heads=heads_count, port_copies=copies, seed=3, **brain_options(),
    )
    brain.act([[1, 0, 0]], greedy=True)
    brain.act([[0, 1, 0]], greedy=True)
    memory = brain.brain.memory
    assert memory.heads == heads_count and memory.streams == 1
    assert sum(member.amplitude for member in memory.memories) == pytest.approx(3.0)
    assert brain.last_settlement["qualified"]
    residual = brain.brain.residual(brain.basal_ganglia._drive, brain.basal_ganglia.state)
    assert residual.max() <= 0.003
    assert memory.memories[0].keys is memory.memories[-1].keys
    assert memory.weights(brain.basal_ganglia.state.activation).shape == (heads_count, 1, 1)
    assert brain.recall.slots == 2
    assert brain.recall.attended_heads[0].shape == (1, 1)


def test_more_heads_and_wider_single_head_match_live_coordinate_budget() -> None:
    two = MultiHeadBrain.compose(3, 2, heads=2, **brain_options())
    wide = MultiHeadBrain.compose(3, 2, port_copies=2, **brain_options())
    assert two.connectome.n == wide.connectome.n
    assert two.connectome.synapses == wide.connectome.synapses
    assert sum(map(len, two.head_queries)) == sum(map(len, wide.head_queries)) == 6
    assert sum(map(len, two.head_reads)) == sum(map(len, wide.head_reads)) == 6
    for brain in (two, wide):
        brain.act([[1, 0, 0]], greedy=True)
        brain.stimulus([[0, 1, 0]])
    np.testing.assert_allclose(np.linalg.norm(two.brain.memory.keys, axis=2),
                               np.linalg.norm(wide.brain.memory.keys, axis=2))
    np.testing.assert_allclose(np.linalg.norm(two.brain.memory.values, axis=2),
                               np.linalg.norm(wide.brain.memory.values, axis=2))
    wide_bank = wide.brain.memory.memories[0]
    np.testing.assert_array_equal(wide_bank.keys[:, :, :3], wide_bank.keys[:, :, 3:])
    np.testing.assert_array_equal(wide_bank.values[:, :, :3], wide_bank.values[:, :, 3:])


@pytest.mark.parametrize("heads_count, copies", [(2, 1), (1, 2)])
def test_saved_wrong_row_lesson_qualifies_all_heads_and_restores_full_boundary(
    heads_count: int, copies: int,
) -> None:
    brain = MultiHeadBrain.compose(
        3, 2, heads=heads_count, port_copies=copies, seed=3, **brain_options(),
    )
    brain.act([[1, 0, 0], [0, 1, 0]], greedy=True)
    drive = brain.stimulus([[0, 0, 1], [0, 0, 1]]).copy()
    original_graph = brain.brain
    snapshot = original_graph.memory
    witnessed = brain.recall.keys.copy()
    lesson, report = brain.teach_saved(drive, np.array([0, 1]), [1], snapshot)
    assert report["qualified"] == 1 and report["accepted_presentations"] == 1
    assert lesson.free.activation.shape == (1, brain.connectome.n)
    assert brain.brain.memory is original_graph.memory is snapshot
    assert snapshot.streams == 2 and snapshot.memories[0].keys is snapshot.memories[-1].keys
    np.testing.assert_array_equal(brain.recall.keys, witnessed)


def test_heads_have_ordinary_independent_reciprocal_seams_with_zero_query_start() -> None:
    brain = MultiHeadBrain.compose(3, 2, heads=2, **brain_options())
    graph = brain.brain
    pre, post = graph.connectome.pre, graph.connectome.post
    association = brain.association_index
    weights = []
    for query, read in zip(brain.head_queries, brain.head_reads, strict=True):
        outgoing = np.isin(pre, read) & np.isin(post, association)
        edges = np.flatnonzero(outgoing)
        assert len(edges) == len(read) * len(association)
        np.testing.assert_array_equal(graph.efficacy[edges],
                                      graph.efficacy[brain.learner.reverse[edges]])
        weights.append(graph.efficacy[edges])
        query_edges = np.isin(pre, query) | np.isin(post, query)
        assert not graph.efficacy[query_edges].any()
    assert not np.array_equal(*weights)


@pytest.mark.parametrize("name", ["heads", "port_copies"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_head_counts_are_explicit_positive_integers(name: str, value) -> None:
    with pytest.raises(ValueError, match=name):
        MultiHeadBrain.compose(3, 2, **{name: value}, **brain_options())
