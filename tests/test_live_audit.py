"""Adversarial continuation and feedback boundaries of a continuing live brain."""

import json

import checkpoints
import numpy as np
import pytest
from checkpoints import flat, write

import cadence as cd


def _brain(*, youth=0):
    return cd.Brain.compose(
        4, 2, modules=(8,), working_memory_amplitude=0.3,
        arousal=cd.ArousalConfig(youth=youth), recall=False,
    )


def _rewrite(path, change):
    with np.load(path, allow_pickle=False) as saved:
        arrays = flat({name: saved[name].copy() for name in saved})
    metadata = checkpoints.metadata(arrays)
    change(metadata, arrays)
    arrays["generic"] = np.array(json.dumps(metadata))
    write(path, arrays)


@pytest.mark.parametrize("acted", [False, True])
def test_checkpoint_rejects_explicit_null_lived_metadata(tmp_path, acted):
    brain = _brain()
    if acted:
        brain.act(np.eye(4)[[0]], greedy=True)
    path = brain.save(tmp_path / "null-lived.npz")
    _rewrite(path, lambda metadata, _: metadata.update(lived=None))
    with pytest.raises(ValueError, match="lived"):
        cd.Brain.load(path)


@pytest.mark.parametrize(
    "defect",
    ["observations", "action", "forecast", "missing_observations", "missing_action", "missing_moment"],
)
def test_sampled_lived_checkpoint_preserves_one_pending_action(tmp_path, defect):
    brain = _brain(youth=5)
    brain.live(np.eye(4)[[0]])
    path = brain.save(tmp_path / "pending-lived.npz")

    def corrupt(metadata, arrays):
        if defect == "observations":
            arrays["lived/observations"] = np.eye(4)[[1]]
        elif defect == "action":
            arrays["lived/action"] = 1 - arrays["moment/action"]
        elif defect == "forecast":
            metadata["lived"]["forecast"] += 1.0
        elif defect == "missing_moment":
            del arrays["moment/observations"], arrays["moment/action"]
        else:
            del arrays["moment/" + defect.removeprefix("missing_")]

    _rewrite(path, corrupt)
    # a lived action that differs from the pending action it was saved with is refused by name
    mismatch = defect in ("observations", "action", "forecast")
    with pytest.raises(ValueError, match="pending feedback" if mismatch else None):
        cd.Brain.load(path)


def test_rejected_terminal_mood_update_preserves_routine_feedback(monkeypatch):
    brain = _brain()
    brain.live(np.eye(4)[[0]])
    state, lived = brain.basal_ganglia.state, brain._lived
    traces = {
        name: getattr(brain.working_memory, name).copy() for name in ("trace", "last", "cold")
    }
    mood = brain.arousal.to_dict()
    outcome = brain.arousal.outcome

    def reject(*args, **kwargs):
        raise ValueError("arousal update cannot remain finite")

    monkeypatch.setattr(brain.arousal, "outcome", reject)
    with pytest.raises(ValueError, match="cannot remain finite"):
        brain.live(np.eye(4)[[1]], reward=[0.0], done=[True])
    assert brain.basal_ganglia.state is state
    assert brain._lived is lived
    assert brain.arousal.to_dict() == mood
    for name, value in traces.items():
        np.testing.assert_array_equal(getattr(brain.working_memory, name), value)
    monkeypatch.setattr(brain.arousal, "outcome", outcome)
    brain.live(np.eye(4)[[1]], reward=[0.0], done=[True])
    assert brain.arousal.outcomes == mood["outcomes"] + 1


def test_rejected_mood_after_sampled_feedback_keeps_accepted_learning(tmp_path, monkeypatch):
    brain = _brain(youth=5)
    brain.live(np.eye(4)[[0]])
    updates, writes = brain.basal_ganglia.updates, brain.hippocampus.writes
    mood = brain.arousal.to_dict()

    def reject(*args, **kwargs):
        raise ValueError("arousal update cannot remain finite")

    monkeypatch.setattr(brain.arousal, "outcome", reject)
    with pytest.raises(ValueError, match="feedback was accepted.*without reward or done"):
        brain.live(np.eye(4)[[1]], reward=[1.0], done=[True])
    assert brain._lived is None and brain.basal_ganglia._pending is None
    assert brain.arousal.to_dict() == mood
    assert brain.basal_ganglia.updates == updates + 1
    assert brain.hippocampus.writes == writes + 1
    with pytest.raises(RuntimeError, match="preceding action"):
        brain.live(np.eye(4)[[1]], reward=[1.0], done=[True])
    twin = cd.Brain.load(brain.save(tmp_path / "accepted-feedback.npz"))
    np.testing.assert_array_equal(brain.live(np.eye(4)[[1]]), twin.live(np.eye(4)[[1]]))
    assert brain.basal_ganglia.updates == updates + 1
    assert brain.hippocampus.writes == writes + 1


def test_terminal_routine_forecast_commits_a_fresh_trace(tmp_path):
    brain = _brain()
    brain.live(np.eye(4)[[0]])
    twin = cd.Brain.load(brain.save(tmp_path / "before-terminal.npz"))
    twin.reset()
    np.testing.assert_array_equal(
        brain.live(np.eye(4)[[1]], reward=[0.0], done=[True]),
        twin.live(np.eye(4)[[1]]),
    )
    np.testing.assert_array_equal(brain.basal_ganglia.state.v, twin.basal_ganglia.state.v)
    for name in ("trace", "last", "cold"):
        np.testing.assert_array_equal(
            getattr(brain.working_memory, name), getattr(twin.working_memory, name)
        )


@pytest.mark.parametrize("sampled", [False, True])
def test_unrepresentable_arousal_spread_obeys_feedback_acceptance_boundary(sampled):
    brain = _brain(youth=5 if sampled else 0)
    brain.live(np.eye(4)[[0]])
    brain.live(np.eye(4)[[0]], reward=[0.0])
    agent, memory = brain.basal_ganglia, brain.hippocampus
    state, lived, mood = agent.state, brain._lived, brain.arousal.to_dict()
    updates, writes = agent.updates, memory.writes
    trace = brain.working_memory.trace.copy()
    expected = "feedback was accepted" if sampled else "arousal moments must remain finite"
    with pytest.raises(ValueError, match=expected):
        brain.live(np.eye(4)[[1]], reward=[1e200], done=[True])
    assert brain.arousal.to_dict() == mood
    assert agent.updates == updates + int(sampled)
    assert memory.writes == writes + int(sampled)
    if sampled:
        assert brain._lived is None and agent._pending is None
        brain.live(np.eye(4)[[1]])
    else:
        assert brain._lived is lived and agent.state is state
        np.testing.assert_array_equal(brain.working_memory.trace, trace)
        brain.live(np.eye(4)[[1]], reward=[0.0], done=[True])
