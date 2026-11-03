"""Free-answer diagnostics expose attempted work without changing a continuing life."""

from dataclasses import replace

import numpy as np
import pytest

from cadence import Brain, NeuralGraph, record_settlements


def checkpoint(brain, path):
    with np.load(brain.save(path), allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def assert_same_checkpoint(before, after):
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)


@pytest.mark.parametrize("qualified", [False, True])
def test_action_report_counts_actual_free_work_and_excludes_reward_eligibility(
    monkeypatch, qualified,
):
    brain = Brain.compose(2, 2, modules=(4,), observers=(2,), seed=3)
    brain.learner.config = replace(brain.learner.config, qualified=qualified, damping=3)
    assert brain.last_settlement is None
    records, checks = [], []
    original = NeuralGraph.residual

    def counted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        checks.append(result.copy())
        return result

    monkeypatch.setattr(NeuralGraph, "residual", counted)
    with record_settlements(records.append):
        action = brain.act([[0.2, -0.1], [-0.1, 0.3]])
    report = brain.last_settlement
    assert report is not None
    assert action.shape == (2,)
    assert report["operation"] == "act" and "scope" not in report
    assert report["qualified"] and report["row_qualified"] == (True, True)
    np.testing.assert_array_equal(report["residual"], checks[-1])
    assert report["max_residual"] == max(report["residual"]) <= report["tolerance"]
    assert report["residual_checks"] == len(checks)
    free = [record for record in records if record.nudge is None]
    eligibility = [record for record in records if record.nudge is not None]
    assert 0 < report["steps"] == sum(record.steps for record in free) <= report["budget"]
    assert len(eligibility) == 2 and sum(record.steps for record in eligibility) > 0
    assert report["steps"] < sum(record.steps for record in records)
    assert brain.last_learning == {}
    with pytest.raises(TypeError):
        report["steps"] = 999
    with pytest.raises(TypeError):
        report["residual"][0] = 999
    with pytest.raises(AttributeError):
        brain.last_settlement = {}


def test_refused_row_replaces_report_without_changing_pending_life(tmp_path):
    brain = Brain.compose(
        2, 2, modules=(4,), observers=(2,), seed=3,
        episodic=False, working_memory_amplitude=0,
    )
    observation = np.array([[0.2, -0.1], [-0.1, 0.3]])
    brain.act(observation)
    prior = brain.last_settlement
    prior_values = dict(prior)
    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=0)
    brain.basal_ganglia.state.v[1, brain.motor_index[0]] += 0.5
    expected = brain.brain.residual(brain.stimulus(observation), brain.basal_ganglia.state)
    assert expected[0] <= config.tolerance < expected[1]
    before = checkpoint(brain, tmp_path / "before")
    with pytest.raises(RuntimeError, match="no action issued"):
        brain.act(observation)
    report = brain.last_settlement
    assert report is not prior and dict(prior) == prior_values
    assert not report["qualified"] and report["row_qualified"] == (True, False)
    assert report["steps"] == report["budget"] == report["damping_halvings"] == 0
    assert report["residual_checks"] == 1
    np.testing.assert_array_equal(report["residual"], expected)
    assert_same_checkpoint(before, checkpoint(brain, tmp_path / "after"))
    brain.learner.config = config
    brain.act(observation)
    assert brain.last_settlement["qualified"]
    assert brain.last_settlement is not report and not report["qualified"]


def test_reports_replace_on_solves_only_and_do_not_enter_checkpoint_continuation(tmp_path):
    brain = Brain.compose(2, 2, modules=(4,), seed=3)
    observation = [[0.2, -0.1]]
    brain.step(observation)
    assert brain.last_settlement["operation"] == "act"
    before = checkpoint(brain, tmp_path / "pending")
    restored = Brain.load(tmp_path / "pending.npz")
    assert restored.last_settlement is None
    brain.predict([[-0.1, 0.2]])
    prediction_report = brain.last_settlement
    assert prediction_report["operation"] == "predict"
    assert_same_checkpoint(before, checkpoint(brain, tmp_path / "after-predict"))
    with pytest.raises(ValueError):
        brain.act([[np.nan, 0.2]])
    assert brain.last_settlement is prediction_report
    phases = brain.imagine([observation])
    assert phases and brain.last_settlement is prediction_report
    reward, done, following = np.array([0.4]), np.array([False]), [[-0.1, 0.3]]
    np.testing.assert_array_equal(
        brain.step(following, reward=reward, done=done),
        restored.step(following, reward=reward, done=done),
    )
    assert brain.last_settlement["operation"] == "act"
    assert brain.last_learning and not any("settlement" in key for key in brain.last_learning)
    assert_same_checkpoint(
        checkpoint(brain, tmp_path / "continued"),
        checkpoint(restored, tmp_path / "restored"),
    )
    brain.reset()
    assert brain.last_settlement is None


def test_cached_zero_budget_answer_reports_its_fresh_residual_check():
    brain = Brain.compose(
        2, 2, modules=(4,), seed=3, episodic=False, working_memory_amplitude=0,
    )
    observation = [[0.2, -0.1]]
    brain.act(observation, greedy=True)
    previous = brain.last_settlement
    brain.learner.config = replace(brain.learner.config, free_steps=0)
    brain.act(observation, greedy=True)
    report = brain.last_settlement
    assert report is not previous and report["qualified"]
    assert report["steps"] == report["budget"] == 0
    assert report["residual_checks"] == 1
    assert report["stagnation_checks"] == report["damping_halvings"] == 0
