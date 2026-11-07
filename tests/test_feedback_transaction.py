"""Reward feedback is atomic and has its own finite eligibility budget."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


def assert_checkpoints_equal(first, second):
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        assert set(left.files) == set(right.files)
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)


@pytest.mark.parametrize("steps", [True, np.bool_(False), -1, 1.5, np.nan, "12"])
def test_invalid_eligibility_budget_is_rejected(steps):
    with pytest.raises(ValueError, match="eligibility_steps"):
        cd.ActorCriticConfig(eligibility_steps=steps)


@pytest.mark.parametrize("steps", [None, 0, 3, np.int64(5)])
def test_standalone_reward_eligibility_has_a_finite_independent_budget(steps, monkeypatch):
    graph = cd.layered(2, 4, 2, density=1, seed=3)
    learner = cd.Learner(
        cd.NeuralGraph(graph, cd.learning_neuron_model(dt=1)),
        graph.populations["output"],
        cd.LearnerConfig(free_steps=64, nudged_steps=7, tolerance=None),
    )
    config = cd.ActorCriticConfig(eligibility_steps=steps)
    agent = cd.ActorCritic(learner, graph.populations["hidden"], config)
    requested = []
    original = cd.NeuralGraph.settle_batch

    def measured(self, *args, **kwargs):
        if kwargs.get("nudge") is not None:
            requested.append(kwargs["steps"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cd.NeuralGraph, "settle_batch", measured)
    agent.act(np.zeros((1, graph.n)))
    expected = 7 if steps is None else steps
    assert requested == [expected, expected]
    assert agent._pending is not None
    assert agent._pending[1].steps == agent._pending[2].steps == expected
    if steps is not None:
        assert type(config.to_dict()["eligibility_steps"]) is int


def test_qualified_supervision_does_not_expand_default_brain_reward_eligibility(tmp_path, monkeypatch):
    brain = cd.Brain.compose(
        2, 2, modules=(4,),
        learning=cd.LearnerConfig(
            qualified=True, damping=3, free_steps=1024, nudged_steps=4096, tolerance=3e-3,
        ), recall=False,
    )
    requested = []
    original = cd.NeuralGraph.settle_batch

    def measured(self, *args, **kwargs):
        if kwargs.get("nudge") is not None:
            requested.append(kwargs["steps"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cd.NeuralGraph, "settle_batch", measured)
    x = np.array([[1.0, 0.0]])
    action = brain.act(x)
    assert requested == [12, 12]
    agent = brain.basal_ganglia
    assert agent.state is not None and agent._pending is not None
    target = brain.learner.targets(action)
    drive = agent._drive
    assert drive is not None
    for state, beta in zip(agent._pending[1:3], (0.1, -0.1), strict=True):
        # Independent literal finite phases, with no qualification or damping.
        reference = original(
            brain.brain, drive, steps=12, state=agent.state,
            nudge=cd.Nudge(
                target, brain.learner.output_mask, beta,
                softmax_temperature=brain.learner.config.temperature,
                groups=agent.group_id,
            ),
            tolerance=3e-3,
        )
        np.testing.assert_array_equal(state.v, reference.v)
        np.testing.assert_array_equal(state.activation, reference.activation)
    monkeypatch.undo()
    restored = cd.Brain.load(brain.save(tmp_path / "reward-budget"))
    assert restored.basal_ganglia.config.eligibility_steps == 12
    assert restored.learner.config.nudged_steps == 4096
    feedback = (np.array([0.4]), np.array([False]), [[0.0, 1.0]])
    brain.learn(*feedback)
    restored.learn(*feedback)
    np.testing.assert_array_equal(brain.act(feedback[2]), restored.act(feedback[2]))
    assert_checkpoints_equal(
        brain.save(tmp_path / "original"), restored.save(tmp_path / "restored")
    )


@pytest.mark.parametrize("kind", ["consolidating", "fast"])
@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_refused_reward_preserves_both_memories_and_retries_exactly_once(
    tmp_path, kind, backend,
):
    if backend == "torch":
        pytest.importorskip("torch")
    brain = cd.Brain.compose(
        2, 2, modules=(4,), seed=3, backend=backend,
        device="cpu" if backend == "torch" else None,
        learning=cd.LearnerConfig(
            qualified=True, damping=3, free_steps=1024, nudged_steps=128, tolerance=3e-3,
        ),
        reward=cd.ActorCriticConfig(eta=0.1, eta_bias=0.01, eligibility_steps=12), recall=False,
    )
    separator = cd.PatternSeparator(2, 8, 3, seed=7, center=0.3 if kind == "fast" else 0)
    memory_class = cd.SynapticMemory if kind == "consolidating" else cd.FastSynapses
    brain.hippocampus = memory_class(
        brain.sensory_index, brain.motor_index, separator=separator, rule="delta", decay=0.85,
    )
    x = np.eye(2)
    brain.hippocampus.observe(x, np.array([[0.3, 0.7], [-0.2, 0.4]]))
    brain.act(x)
    brain.learn(np.array([0.1, 0.2]), np.array([False, False]), x)
    brain.act(x)
    agent = brain.basal_ganglia
    if backend == "torch":
        assert agent._trace_device is not None and agent._trace_device[0].abs().max() > 0
    else:
        assert agent.trace is not None and np.any(agent.trace)
    assert brain.working_memory is not None and np.any(brain.working_memory.trace)
    control = cd.Brain.load(
        brain.save(tmp_path / "control-start"), backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=0)
    before = brain.save(tmp_path / "before-refusal")
    memory, trace = brain.hippocampus, brain.working_memory
    pending = agent._pending
    moment, prepared = brain._moment, brain._prepared
    reward, done, following = np.array([0.8, -0.4]), np.array([True, False]), x[:, ::-1]
    with pytest.raises(cd.LearningPhaseError, match="free learning phase"):
        brain.learn(reward, done, following)
    assert brain.hippocampus is memory and brain.working_memory is trace
    assert agent._pending is pending and brain._moment is moment and brain._prepared is prepared
    assert_checkpoints_equal(before, brain.save(tmp_path / "after-refusal"))
    # Retry the same real outcome after fixing the budget, then continue against
    # a saved control that received this outcome only once.
    brain.learner.config = config
    writes, updates = memory.writes, agent.updates
    brain.learn(reward, done, following)
    control.learn(reward, done, following)
    assert memory.writes == writes + 2 and agent.updates == updates + 1
    assert brain._moment is None and agent._pending is None
    with pytest.raises(RuntimeError, match="non-greedy act"):
        brain.learn(reward, done, following)
    assert memory.writes == writes + 2 and agent.updates == updates + 1
    np.testing.assert_array_equal(brain.act(following), control.act(following))
    assert_checkpoints_equal(
        brain.save(tmp_path / "retried"), control.save(tmp_path / "once")
    )
