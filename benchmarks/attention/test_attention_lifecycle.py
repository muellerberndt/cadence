"""Mechanical lifecycle custody; these small tests establish no task capability."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cadence import Brain
from cadence.receipts import canonical_sha256

SPEC = importlib.util.spec_from_file_location(
    "attention_lifecycle_tests_heads", Path(__file__).with_name("heads.py"))
assert SPEC is not None and SPEC.loader is not None
heads = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = heads
SPEC.loader.exec_module(heads)
coupled = heads.coupled


@pytest.mark.parametrize("count", [1, 2])
def test_learned_code_preserves_founder_numerical_defaults_and_explicit_overrides(count):
    settings = dict(modules=(4,), episodic=False, recall=True, recall_next=False)
    raw = heads.MultiHeadBrain.compose(3, 2, heads=count, **settings)
    native = Brain.compose(3, 2, learned_code=6, **settings)
    learned = heads.MultiHeadBrain.compose(3, 2, heads=count, learned_code=6, **settings)
    assert raw.brain.neuron_model.dt == .1 and raw.learner.config.damping == 0
    assert learned.brain.neuron_model.dt == native.brain.neuron_model.dt == 1.0
    assert learned.learner.config.damping == native.learner.config.damping == 3
    explicit = heads.MultiHeadBrain.compose(
        3, 2, heads=count, learned_code=6, learning_damping=1, **settings)
    assert explicit.learner.config.damping == 1


def brain(**options):
    settings = dict(modules=(4,), episodic=True, recall_next=False, recall_amplitude=1.0,
                    recall_address=0.0, learning_free_steps=1024,
                    learning_nudged_steps=1024, learning_damping=3)
    settings.update(options)
    dt = 1.0 if settings.get("learned_code") else 0.1
    return heads.MultiHeadBrain.compose(3, 2, heads=2, seed=7, dt=dt,
                                        empty_anchor=True, **settings)


def bank(memory):
    if memory is None:
        return None
    return [{name: getattr(m, name) for name in (
        "query", "read", "keys", "values", "held", "ages", "amplitude", "temperature",
        "recency", "mode", "empty_anchor")} for m in getattr(memory, "memories", (memory,))]


def state(phase):
    return None if phase is None else {name: getattr(phase, name)
                                      for name in ("v", "activation", "adaptation", "steps")}


def fingerprint(b):
    agent = b.basal_ganglia
    pending = agent._pending
    return canonical_sha256({
        "parameters": [b.brain.efficacy, b.brain.bias, b.brain.log_gain],
        "optimizer": [b.learner.velocity, b.learner.velocity_bias,
                      b.learner.second_moment, b.learner.second_moment_bias],
        "updates": [b.learner.updates, b.learner.contrast_updates, agent.updates],
        "actor": {name: getattr(agent, name) for name in (
            "w_critic", "b_critic", "trace", "trace_bias", "trace_critic", "trace_memory")},
        "state": state(agent.state), "drive": agent._drive,
        "pending": None if pending is None else [pending[0], state(pending[1]),
                                                 state(pending[2]), pending[3]],
        "banks": [bank(b.brain.memory), bank(b._pending_boundary), bank(b._action_boundary),
                  bank(b._forecast_boundary)],
        "rng": [b.rng.bit_generator.state, agent.rng.bit_generator.state],
        "memories": {name: [m.to_dict(), m.arrays()] for name, m in b.memories.items()},
        "arousal": None if b.arousal is None else b.arousal.to_dict(),
        "head_attended": b.head_attended,
    })


def test_default_action_samples_and_eligibility_keeps_the_actual_prewitness_bank(monkeypatch):
    b = brain()
    seen = []
    original = coupled.Graph.settle_batch

    def settle(graph, drive, **kwargs):
        if kwargs.get("nudge") is not None:
            seen.append((graph.memory, kwargs["steps"]))
        return original(graph, drive, **kwargs)

    monkeypatch.setattr(coupled.Graph, "settle_batch", settle)
    action = b.act([[1, 0, 0]])
    assert b.basal_ganglia._pending is not None
    assert b._pending_boundary is b._action_boundary is b.brain.memory
    assert seen == [(b._pending_boundary, 12), (b._pending_boundary, 12)]
    assert not b._pending_boundary.held.any()  # Before issuing/witnessing this moment.
    assert b.recall.held.any() and b.recall.writes == 1
    np.testing.assert_array_equal(b._moment[1], action)
    assert 0 <= b.basal_ganglia._pending[1].steps <= b.basal_ganglia.config.eligibility_steps == 12


def test_explicit_greedy_action_preserves_single_head_conformance():
    settings = dict(modules=(4,), episodic=False, recall_next=False,
                    learning_free_steps=1024, learning_nudged_steps=1024, learning_damping=3)
    one = coupled.AttentionBrain.compose(3, 2, seed=9, dt=0.1, **settings)
    multi = heads.MultiHeadBrain.compose(3, 2, seed=9, dt=0.1, heads=1, **settings)
    for observation in ([[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]):
        np.testing.assert_array_equal(one.act(observation, greedy=True),
                                      multi.act(observation, greedy=True))
        np.testing.assert_array_equal(one.basal_ganglia.state.v, multi.basal_ganglia.state.v)
    assert one.basal_ganglia._pending is multi.basal_ganglia._pending is None
    assert one._pending_boundary is multi._pending_boundary is None


def test_step_credits_one_actual_action_and_issues_the_next_sampled_action():
    b = brain()
    first = b.step([[1, 0, 0]])
    np.testing.assert_array_equal(b._moment[1], first)
    before = b.basal_ganglia.updates
    b.step([[0, 1, 0]], reward=[1.0])
    assert b.basal_ganglia.updates == before + 1
    assert b.basal_ganglia._pending is not None and b._pending_boundary is b.brain.memory
    b.learn([0.0], [False], [[0, 0, 1]])
    assert b.basal_ganglia._pending is b._pending_boundary is None
    with pytest.raises(RuntimeError):
        b.learn([0.0], [False], [[0, 0, 1]])


@pytest.mark.parametrize("invalid", ["nonfinite", "bootstrap", "done"])
def test_invalid_feedback_restores_the_actual_pending_boundary_before_any_update(invalid):
    b = brain()
    b.act([[1, 0, 0]])
    saved, pending, boundary = fingerprint(b), b.basal_ganglia._pending, b._pending_boundary
    reward, done, extras = [1.0], [False], {}
    if invalid == "nonfinite":
        reward = [float("nan")]
    elif invalid == "bootstrap":
        extras["bootstrap"] = [float("nan")]
    else:
        done = [0]
    with pytest.raises(ValueError):
        b.learn(reward, done, [[0, 1, 0]], **extras)
    assert fingerprint(b) == saved
    assert b.basal_ganglia._pending is pending and b._pending_boundary is boundary


def test_refused_bootstrap_rolls_back_partial_done_and_all_memories(monkeypatch):
    b = brain()
    b.act([[1, 0, 0], [0, 1, 0]])
    saved, boundary = fingerprint(b), b._pending_boundary
    seen = []

    def fail(drive, done, warm):
        seen.append(b.brain.memory)
        np.testing.assert_array_equal(done, [True, False])
        assert not b.brain.memory.held[0].any() and b.brain.memory.held[1].any()
        raise RuntimeError("refused next state")

    monkeypatch.setattr(b.basal_ganglia, "_next_state", fail)
    with pytest.raises(RuntimeError, match="refused"):
        b.learn([1.0, 0.0], [True, False], [[0, 0, 1], [1, 0, 0]])
    assert seen and fingerprint(b) == saved
    assert b._pending_boundary is b.brain.memory is boundary


def test_accepted_partial_done_uses_new_boundary_and_clears_only_ended_credit():
    b = brain()
    b.act([[1, 0, 0], [0, 1, 0]])
    previous = b._pending_boundary
    b.learn([1.0, 0.0], [True, False], [[0, 0, 1], [1, 0, 0]])
    assert b._pending_boundary is None and b.brain.memory is not previous
    assert not b.brain.memory.held[0].any() and b.brain.memory.held[1].any()
    assert not b.basal_ganglia.trace[0].any()
    assert not b.basal_ganglia.trace_bias[0].any()
    assert not b.basal_ganglia.trace_critic[0].any()
    assert b.basal_ganglia.trace_critic[1, -1] != 0


def test_terminal_routine_forecast_keeps_the_new_empty_boundary_for_its_answer():
    b = brain(arousal=True, arousal_youth=0, arousal_need=0.0, arousal_need_scale=0.0)
    b.live([[1, 0, 0]])
    previous = b._action_boundary
    b.live([[0, 1, 0]], reward=[0.0], done=[True])
    assert b._forecast_boundary is b._action_boundary is b.brain.memory
    assert b.brain.memory is not previous and not b.brain.memory.held.any()
    assert b.recall.held.sum() == 1  # Only the accepted next moment is witnessed.


def test_accepted_feedback_is_not_rolled_back_when_the_next_action_refuses(monkeypatch):
    b = brain()
    b.step([[1, 0, 0]])
    updates = b.basal_ganglia.updates

    def fail(*args, **kwargs):
        raise RuntimeError("next action refused")

    monkeypatch.setattr(b, "act", fail)
    with pytest.raises(RuntimeError, match="next action"):
        b.step([[0, 1, 0]], reward=[1.0])
    assert b.basal_ganglia.updates == updates + 1
    assert b.basal_ganglia._pending is b._pending_boundary is None
    with pytest.raises(RuntimeError):
        b.learn([1.0], [False], [[0, 1, 0]])


def test_pending_action_prevents_an_unowned_teacher_update():
    b = brain()
    b.act([[1, 0, 0]])
    saved = fingerprint(b)
    with pytest.raises(RuntimeError, match="outcome"):
        b.teach_saved(b.basal_ganglia._drive, [0], [True], b._pending_boundary)
    with pytest.raises(RuntimeError, match="outcome"):
        b.lesson([[0, 1, 0]], [0])
    assert fingerprint(b) == saved


def test_calm_live_keeps_weights_and_outcome_stores_still_but_witnesses_each_moment():
    b = brain(arousal=True, arousal_youth=0, arousal_need=0.0, arousal_need_scale=0.0)
    parameters = b.brain.efficacy.copy()
    b.live([[1, 0, 0]])
    assert b.last_arousal["mode"] == "routine"
    writes = b.recall.writes
    store = {name: value.copy() for name, value in b.hippocampus.arrays().items()}
    b.live([[1, 0, 0]], reward=[0.0])
    assert b.last_arousal["mode"] == "routine"
    assert b.recall.writes == writes + 1
    assert b.basal_ganglia._pending is None and b.learner.updates == 0
    np.testing.assert_array_equal(b.brain.efficacy, parameters)
    for name, value in store.items():
        np.testing.assert_array_equal(b.hippocampus.arrays()[name], value)


def test_wait_keeps_original_action_eligibility_and_decision_ownership():
    b = brain(arousal=True)
    b.live([[1, 0, 0]])
    token, pending, boundary = b.decision_id, b.basal_ganglia._pending, b._pending_boundary
    b.wait([[0, 1, 0]])
    assert b.basal_ganglia._pending is pending and b._pending_boundary is boundary
    assert b.brain.memory is not boundary and b.decision_id == token
    saved = fingerprint(b)
    with pytest.raises(ValueError, match="decision_id"):
        b.live([[0, 0, 1]], reward=[1.0], decision_id=token + 1)
    assert fingerprint(b) == saved
    b.live([[0, 0, 1]], reward=[1.0], decision_id=token)
    assert b.basal_ganglia.updates == 1 and b.decision_id != token


def test_accepted_routine_outcome_cannot_remain_pending_after_answer_refusal(monkeypatch):
    b = brain(arousal=True, arousal_youth=0, arousal_need=0.0, arousal_need_scale=0.0)
    b.live([[1, 0, 0]])
    original = b._choose

    def fail(*args, **kwargs):
        raise RuntimeError("answer refused after outcome")

    monkeypatch.setattr(b, "_choose", fail)
    with pytest.raises(RuntimeError, match="after outcome"):
        b.live([[1, 0, 0]], reward=[0.0])
    assert b._lived is b._pending_boundary is b._action_boundary is None
    with pytest.raises(RuntimeError, match="preceding action"):
        b.live([[1, 0, 0]], reward=[0.0])
    monkeypatch.setattr(b, "_choose", original)
    b.live([[1, 0, 0]])


@pytest.mark.parametrize("operation", ["predict", "accuracy", "imagine", "refused_imagine"])
def test_private_operations_preserve_live_pending_state_and_coupled_bank(operation, monkeypatch):
    b = brain()
    b.act([[1, 0, 0]])
    saved, graph, pending = fingerprint(b), b.brain, b.basal_ganglia._pending
    boundary, seen = b._pending_boundary, []
    original = coupled.Graph.equilibrate

    def solve(current, drive, **kwargs):
        if operation.endswith("imagine"):
            assert current is not graph and current.memory is not None
            assert len(current.memory.memories) == 2
            for member in current.memory.memories:
                assert not drive[:, member.read].any()  # Coupled read, no held source.
                np.testing.assert_array_equal(member.held, b.recall.held)
            seen.append(current.memory)
        return original(current, drive, **kwargs)

    monkeypatch.setattr(coupled.Graph, "equilibrate", solve)
    if operation == "predict":
        b.predict([[0, 1, 0]])
    elif operation == "accuracy":
        b.accuracy([[0, 1, 0]], [0])
    else:
        phases = b.imagine([[[0, 1, 0]], [[0, 0, 1]]],
                           budget=0 if operation == "refused_imagine" else 1024,
                           tolerance=0.003)
        assert seen
        assert len(phases) == (1 if operation == "refused_imagine" else 2)
    assert fingerprint(b) == saved and b.brain is graph
    assert b.basal_ganglia._pending is pending and b._pending_boundary is boundary


def test_refused_action_and_wait_restore_the_current_boundary_and_rng():
    b = brain(arousal=True)
    b.live([[1, 0, 0]])
    b.learner.config = replace(b.learner.config, free_steps=0, tolerance=0.0)
    saved = fingerprint(b)
    for operation in (b.act, b.wait):
        with pytest.raises(RuntimeError, match="settle"):
            operation([[0, 1, 0]])
        assert fingerprint(b) == saved


def test_builder_uses_identical_immutable_records_for_live_and_private_recall():
    b = brain()
    b.act([[1, 0, 0]], greedy=True)
    drive = b.stimulus([[0, 1, 0]])
    live = b.brain.memory
    branch = b.recall.branch()
    branch.stimulus(np.zeros_like(drive), b._read_moment(
        np.array([[0.0, 1.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]])))
    private = b._memory_from_recall(branch)
    assert canonical_sha256(bank(live)) == canonical_sha256(bank(private))
    assert all(not m.keys.flags.writeable and not m.values.flags.writeable
               for m in private.memories)


@pytest.mark.parametrize("heads_count", [1, 2])
def test_no_separate_address_reward_rule_is_silently_enabled(heads_count):
    with pytest.raises(ValueError, match="address"):
        heads.MultiHeadBrain.compose(3, 2, modules=(4,), heads=heads_count,
                                    recall_address_reward=1.0)


@pytest.mark.parametrize("heads_count", [1, 2])
def test_held_learned_code_keeps_its_native_outer_step_contract(heads_count):
    b = heads.MultiHeadBrain.compose(3, 2, modules=(4,), heads=heads_count, learned_code=6)
    assert b.brain.neuron_model.dt == 1.0
    raw = heads.MultiHeadBrain.compose(3, 2, modules=(4,), heads=heads_count)
    assert raw.brain.neuron_model.dt == 0.1
    with pytest.raises(ValueError, match="outer dt"):
        heads.MultiHeadBrain.compose(3, 2, modules=(4,), heads=heads_count,
                                    learned_code=6, dt=0.1)


def test_imagination_restores_held_learned_code_state_without_learning():
    b = brain(learned_code=6)
    b.act([[1, 0, 0]])
    region = b.learned_code
    arrays = deepcopy(region.arrays())
    before = b._private_code_state()
    b.imagine([[[0, 1, 0]]], budget=1024)
    assert region.halvings == before[2][0] and region.graph is before[2][1]
    assert region.last is before[2][2] and region.work == before[2][3]
    for name, array in arrays.items():
        np.testing.assert_array_equal(region.arrays()[name], array)


@pytest.mark.parametrize("operation", ["predict", "accuracy"])
def test_independent_scoring_restores_learned_code_phase_and_pending_caches(operation):
    b = brain(learned_code=6)
    b.act([[1, 0, 0]])
    region = b.learned_code
    arrays = deepcopy(region.arrays())
    before, pending = b._private_code_state(), b._pending_boundary
    if operation == "predict":
        b.predict([[0, 1, 0]])
    else:
        b.accuracy([[0, 1, 0]], [0])
    after = b._private_code_state()
    assert after[0] is before[0] and after[1] is before[1]
    assert after[2][0] == before[2][0] and after[2][1] is before[2][1]
    assert after[2][2] is before[2][2] and after[2][3] == before[2][3]
    assert b._pending_boundary is pending
    for name, array in arrays.items():
        np.testing.assert_array_equal(region.arrays()[name], array)


def test_calm_live_advances_held_code_phases_without_learning_its_tables():
    b = brain(learned_code=6, arousal=True, arousal_youth=0,
              arousal_need=0.0, arousal_need_scale=0.0)
    region = b.learned_code
    arrays = deepcopy(region.arrays())
    b.live([[1, 0, 0]])
    b.live([[0, 1, 0]], reward=[0.0])
    assert b.last_arousal["mode"] == "routine" and b.recall.writes == 2
    assert b.learner.updates == 0 and b.basal_ganglia.updates == 0
    for name, array in arrays.items():
        np.testing.assert_array_equal(region.arrays()[name], array)
