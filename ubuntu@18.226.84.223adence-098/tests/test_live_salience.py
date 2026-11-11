"""Explicit outcome importance in live preserves arousal and actual-action custody."""

import json
from dataclasses import replace

import numpy as np
import pytest

import cadence as cd

OBSERVATIONS = np.eye(4)


def brain(*, sampled=True, episodic=True, learned=False):
    return cd.Brain.compose(
        4, 2, modules=(8,), seed=3, working_memory_amplitude=0.3,
        arousal={"youth": 100 if sampled else 0},
        memory_choice=episodic, learned_code=8 if learned else None,
        recall=True, recall_founder=True, history_decays=(0.5,), actor_eta=0.03, actor_eta_bias=0.003,
    )


def archive(brain, path):
    with np.load(brain.save(path), allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def same_archives(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def test_a_marked_zero_is_written_without_closing_the_reward_replay_path():
    body = brain()
    body.live(OBSERVATIONS[[0]])
    body.live(OBSERVATIONS[[1]], reward=[0.0], salience=[0.005])
    assert body.hippocampus.writes == body.context_memory.writes == 1
    path = body.memories["episodic"].paths[0]
    assert len(path) == 2 and path[0][3] == 0.0 and path[0][4]
    # The final reward still credits the marked intermediate decision: one direct write
    # and one replay write in each store. The mark supplied no invented reward.
    before = body.hippocampus.writes
    body.live(OBSERVATIONS[[2]], reward=[1.0], done=[True])
    assert body.hippocampus.writes == body.context_memory.writes == before + 2


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("bad", [[-1.0], [np.nan], [np.inf], [], [1.0, 2.0], 1.0])
def test_invalid_importance_refuses_before_any_live_state_changes(tmp_path, sampled, bad):
    body = brain(sampled=sampled)
    body.live(OBSERVATIONS[[0]])
    decision = body.decision_id
    before = archive(body, tmp_path / "before")
    with pytest.raises(ValueError, match="salience"):
        body.live(OBSERVATIONS[[1]], reward=[0.0], salience=bad, decision_id=decision)
    assert body.decision_id == decision and body.pending_feedback
    same_archives(before, archive(body, tmp_path / "after"))


def test_salience_alone_requires_a_preceding_real_action(tmp_path):
    body = brain()
    before = archive(body, tmp_path / "before")
    with pytest.raises(RuntimeError, match="preceding action"):
        body.live(OBSERVATIONS[[0]], salience=[0.5])
    same_archives(before, archive(body, tmp_path / "after"))


@pytest.mark.parametrize("sampled", [False, True])
def test_a_mark_for_a_stale_decision_is_rejected_without_changes(tmp_path, sampled):
    body = brain(sampled=sampled)
    body.live(OBSERVATIONS[[0]])
    before = archive(body, tmp_path / "before")
    with pytest.raises(ValueError, match="does not own"):
        body.live(OBSERVATIONS[[1]], reward=[0.0], salience=[0.5],
                  decision_id=body.decision_id + 1)
    same_archives(before, archive(body, tmp_path / "after"))


@pytest.mark.parametrize("learned", [False, True])
def test_delayed_mark_uses_the_issued_actions_saved_key_after_wait_and_load(
    tmp_path, monkeypatch, learned,
):
    body = brain(learned=learned)
    body.live(OBSERVATIONS[[0]])
    key = body._moment_code.copy() if learned else OBSERVATIONS[[0]]
    decision = body.decision_id
    body.wait(OBSERVATIONS[[1]])
    twin = cd.Brain.load(body.save(tmp_path / "waiting"))
    written = []
    observe = body.hippocampus.observe

    def witnessed(keys, *args, **kwargs):
        written.append(keys.copy())
        return observe(keys, *args, **kwargs)

    monkeypatch.setattr(body.hippocampus, "observe", witnessed)
    feedback = {"reward": [0.0], "salience": [0.005], "decision_id": decision}
    np.testing.assert_array_equal(
        body.live(OBSERVATIONS[[2]], **feedback), twin.live(OBSERVATIONS[[2]], **feedback)
    )
    assert len(written) == 1
    np.testing.assert_array_equal(written[0], key)
    same_archives(archive(body, tmp_path / "original"), archive(twin, tmp_path / "resumed"))


def test_refused_marked_feedback_rolls_back_and_is_retried_once(tmp_path):
    body = brain()
    body.retune(learning_qualified=True, learning_nudged_steps=1024)
    body.live(OBSERVATIONS[[0]])
    control = cd.Brain.load(body.save(tmp_path / "control"))
    config = body.learner.config
    body.learner.config = replace(config, free_steps=0)
    before = archive(body, tmp_path / "before")
    feedback = {"reward": [0.0], "salience": [0.5], "decision_id": body.decision_id}
    with pytest.raises(cd.LearningPhaseError, match="free learning phase"):
        body.live(OBSERVATIONS[[1]], **feedback)
    assert body.pending_feedback
    same_archives(before, archive(body, tmp_path / "after"))
    body.learner.config = config
    np.testing.assert_array_equal(
        body.live(OBSERVATIONS[[1]], **feedback), control.live(OBSERVATIONS[[1]], **feedback)
    )
    assert body.hippocampus.writes == body.context_memory.writes == 1
    same_archives(archive(body, tmp_path / "retried"), archive(control, tmp_path / "once"))


def test_a_refused_routine_forecast_keeps_the_same_marked_outcome_pending(tmp_path):
    body = brain(sampled=False)
    body.live(OBSERVATIONS[[0]])
    body.retune(learning_free_steps=0)
    before = archive(body, tmp_path / "before")
    with pytest.raises(RuntimeError, match="did not settle"):
        body.live(OBSERVATIONS[[1]], reward=[0.0], salience=[0.5])
    assert body.pending_feedback
    same_archives(before, archive(body, tmp_path / "after"))


def test_an_accepted_mark_is_not_resubmitted_when_the_next_answer_refuses(tmp_path, monkeypatch):
    body = brain()
    body.live(OBSERVATIONS[[0]])
    control = cd.Brain.load(body.save(tmp_path / "control"))
    settled = body._settled

    def refuse(*args, **kwargs):
        raise RuntimeError("next answer refused")

    monkeypatch.setattr(body, "_settled", refuse)
    with pytest.raises(RuntimeError, match="next answer refused"):
        body.live(OBSERVATIONS[[1]], reward=[0.0], salience=[0.5])
    assert not body.pending_feedback and body.hippocampus.writes == 1
    with pytest.raises(RuntimeError, match="preceding action"):
        body.live(OBSERVATIONS[[1]], salience=[0.5])
    monkeypatch.setattr(body, "_settled", settled)
    np.testing.assert_array_equal(
        body.live(OBSERVATIONS[[1]]),
        control.live(OBSERVATIONS[[1]], reward=[0.0], salience=[0.5]),
    )
    assert body.hippocampus.writes == 1
    left, right = archive(body, tmp_path / "retried"), archive(control, tmp_path / "once")
    # The existing work ledger charges a feedback phase only with the call's completed
    # live moment. Compare actual-action state independently of that split-call counter.
    left_meta, right_meta = json.loads(str(left.pop("brain"))), json.loads(str(right.pop("brain")))
    left_meta["arousal"].pop("learning_sweeps")
    right_meta["arousal"].pop("learning_sweeps")
    assert left_meta == right_meta
    same_archives(left, right)


@pytest.mark.parametrize("episodic", [False, True])
def test_a_mark_does_not_wake_a_calm_brain_or_write_its_records(episodic):
    body = brain(sampled=False, episodic=episodic, learned=True)
    body.live(OBSERVATIONS[[0]])
    tables = {
        "efficacy": body.brain.efficacy.copy(), "bias": body.brain.bias.copy(),
        "critic": np.append(body.basal_ganglia.w_critic, body.basal_ganglia.b_critic),
        **{"code_" + name: value.copy() for name, value in body.learned_code.arrays().items()},
    }
    stores = [store for store in (body.hippocampus, body.context_memory) if store is not None]
    before = [{name: value.copy() for name, value in store.arrays().items()} for store in stores]
    body.live(OBSERVATIONS[[1]], reward=[0.0], salience=[10.0])
    assert body.last_arousal["mode"] == "routine" and not body.last_arousal["recorded"]
    assert body.learner.updates == body.basal_ganglia.updates == 0
    np.testing.assert_array_equal(body.brain.efficacy, tables["efficacy"])
    np.testing.assert_array_equal(body.brain.bias, tables["bias"])
    np.testing.assert_array_equal(
        np.append(body.basal_ganglia.w_critic, body.basal_ganglia.b_critic), tables["critic"]
    )
    for name, value in body.learned_code.arrays().items():
        np.testing.assert_array_equal(value, tables["code_" + name])
    for store, saved in zip(stores, before, strict=True):
        assert store.writes == 0
        for name, value in store.arrays().items():
            np.testing.assert_array_equal(value, saved[name], err_msg=name)


@pytest.mark.parametrize("episodic", [False, True])
def test_the_outcome_that_wakes_a_routine_brain_uses_its_supplied_importance(
    tmp_path, episodic,
):
    body = brain(sampled=False, episodic=episodic)
    body.live(OBSERVATIONS[[0]])
    body.live(OBSERVATIONS[[0]], reward=[0.0])  # establish the usual forecast error
    assert body.last_arousal["mode"] == "routine"
    control = cd.Brain.load(body.save(tmp_path / "routine"))
    body.live(OBSERVATIONS[[1]], reward=[1.0], salience=[3.0])
    control.live(OBSERVATIONS[[1]], reward=[1.0])
    assert body.last_arousal["mode"] == "aroused" and body.last_arousal["recorded"]
    assert body.arousal.to_dict() == control.arousal.to_dict()
    assert body.hippocampus.writes == control.hippocampus.writes == 1
    assert np.linalg.norm(body.hippocampus.consolidated) > np.linalg.norm(
        control.hippocampus.consolidated
    )
    if episodic:
        assert body.context_memory.writes == control.context_memory.writes == 1
        assert np.linalg.norm(body.context_memory.consolidated) > np.linalg.norm(
            control.context_memory.consolidated
        )


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("episodic", [False, True])
def test_default_and_explicit_none_have_identical_continuation(tmp_path, sampled, episodic):
    body = brain(sampled=sampled, episodic=episodic)
    body.live(OBSERVATIONS[[0]])
    twin = cd.Brain.load(body.save(tmp_path / "pending"))
    for index in range(1, 4):
        feedback = {"reward": [float(index % 2)], "done": [index == 2]}
        np.testing.assert_array_equal(
            body.live(OBSERVATIONS[[index]], **feedback),
            twin.live(OBSERVATIONS[[index]], salience=None, **feedback),
        )
    same_archives(archive(body, tmp_path / "implicit"), archive(twin, tmp_path / "explicit"))
