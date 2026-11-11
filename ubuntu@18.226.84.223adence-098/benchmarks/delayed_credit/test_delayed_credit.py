"""The delayed-credit chamber (cadence#283): the worlds, the exact solver, the controls, the
conventional learners and short lives of every kind."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest


def _load(name: str):
    key = f"delayed_credit_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


tk = _load("tasks")
bl = _load("baselines")
ch = _load("chamber")


def _live(task: str, founder: int, delay: int, cue: int, actions) -> tk.Episode:
    corridor = tk.schedule(task, founder, delay, 2)[0][1]
    episode = tk.Episode(task, founder, delay, cue, corridor)
    for action in actions:
        episode.act(action)
    return episode


@pytest.mark.parametrize("task", tk.TASKS)
@pytest.mark.parametrize("delay", [2, 10, 50])
def test_the_decision_is_followed_by_delay_acts_before_its_outcome(task, delay):
    corridor = tk.schedule(task, 0, delay, 2)[0][1]
    episode = tk.Episode(task, 0, delay, 0, corridor)
    assert episode.moments == delay + 1
    rewards = []
    while not episode.done:
        assert episode.reward == 0.0
        episode.act(0)
        rewards.append(episode.reward)
    assert all(r == 0.0 for r in rewards[:-1])
    with pytest.raises(RuntimeError):
        episode.observation()


def test_the_tmaze_pays_the_cues_arm_and_its_corridor_ignores_the_choice():
    for founder in range(4):
        for cue in (0, 1):
            right = tk.right_action(cue, founder)
            good = _live("tmaze", founder, 10, cue, [right] + [0] * 10)
            bad = _live("tmaze", founder, 10, cue, [1 - right] + [1] * 10)
            assert good.reward == tk.RIGHT and bad.reward == tk.WRONG
    corridor = tk.schedule("tmaze", 0, 10, 2)[0][1]
    seen = []
    for choice in (0, 1):
        episode = tk.Episode("tmaze", 0, 10, 0, corridor)
        rows = []
        while not episode.done:
            rows.append(episode.observation())
            episode.act(choice if episode.moment == 0 else 0)
        seen.append(np.array(rows))
    assert np.array_equal(seen[0][1:], seen[1][1:])
    assert (seen[0].sum(axis=1) == 1).all()


def test_the_key_pays_only_through_the_door():
    founder = 2  # the door opens with action 0
    opening = tk.open_action(founder)
    assert opening == 0
    for cue in (0, 1):
        reach = tk.right_action(cue, founder)
        taken = _live("keydoor", founder, 5, cue, [reach, 0, 0, 0, 0, opening])
        assert taken.key and taken.reward == tk.RIGHT
        left = _live("keydoor", founder, 5, cue, [1 - reach, 0, 0, 0, 0, opening])
        assert not left.key and left.reward == tk.WRONG
        shut = _live("keydoor", founder, 5, cue, [reach, 0, 0, 0, 0, 1 - opening])
        assert shut.reward == tk.SHUT
    episode = tk.Episode("keydoor", founder, 5, 0, tk.schedule("keydoor", founder, 5, 2)[0][1])
    assert episode.observation()[tk.POUCH] == 0
    episode.act(tk.right_action(0, founder))
    while episode.moment < episode.moments - 1:
        assert episode.observation()[tk.POUCH] == 1
        episode.act(1)
    assert episode.observation()[tk.DOOR] == 1


def test_the_corridor_shows_distractors_at_the_declared_rate():
    rng = np.random.default_rng(0)
    tokens = np.array(tk.corridor_tokens(rng, 20_000))
    shown = tokens != tk.CORRIDOR
    assert abs(shown.mean() - tk.DISTRACTION) < 0.02
    assert set(tokens[shown]) == set(range(tk.DISTRACTOR, tk.DISTRACTOR + tk.DISTRACTORS))


def test_the_schedule_shows_both_cues_in_every_block_and_is_founder_specific():
    episodes = tk.schedule("tmaze", 0, 10, 64)
    for block in range(32):
        assert {episodes[2 * block][0], episodes[2 * block + 1][0]} == {0, 1}
    assert episodes == tk.schedule("tmaze", 0, 10, 64)
    assert episodes != tk.schedule("tmaze", 1, 10, 64)


@pytest.mark.parametrize("task", tk.TASKS)
def test_the_exact_solver_answers_every_episode_from_what_the_brain_senses(task):
    for founder in range(4):
        for cue, corridor in tk.schedule(task, founder, 10, 32):
            episode = tk.Episode(task, founder, 10, cue, corridor)
            while not episode.done:
                episode.act(tk.solve(task, founder, episode.observation()))
            assert episode.right and episode.reward == tk.RIGHT


def test_yoked_and_shuffled_outcomes_do_not_follow_the_learners_decision():
    yoked = ch.Delivery("yoked", "tmaze", 0, 10)
    shuffled = ch.Delivery("shuffled", "tmaze", 0, 10)
    own = ch.Delivery("none", "tmaze", 0, 10)
    agree = {"yoked": [], "shuffled": []}
    for cue, corridor in tk.schedule("tmaze", 0, 10, 400):
        episode = tk.Episode("tmaze", 0, 10, cue, corridor)
        while not episode.done:
            episode.act(tk.solve("tmaze", 0, episode.observation()))
        assert own(episode) == episode.reward == tk.RIGHT
        agree["yoked"].append(yoked(episode) == tk.RIGHT)
        agree["shuffled"].append(shuffled(episode))
    assert 0.4 < np.mean(agree["yoked"]) < 0.6
    assert set(agree["shuffled"]) == {tk.RIGHT}  # drawn from the learner's own outcomes


@pytest.mark.parametrize("task", tk.TASKS)
def test_the_conventional_learners_are_sized_to_the_brains_parameter_count(task):
    warnings.simplefilter("ignore")
    count = ch.brain_parameters(task)
    for kind in ("lstm", "tdlambda"):
        size = bl.lstm_parameters if kind == "lstm" else bl.elman_parameters
        hidden = bl.matched_hidden(count, tk.INPUTS[task], tk.ACTIONS, kind)
        assert abs(size(tk.INPUTS[task], tk.ACTIONS, hidden) - count) / count < 0.02


def test_the_lstm_learner_counts_its_parameters_as_torch_does_and_its_gradient_is_exact():
    pytest.importorskip("torch")
    learner = bl.LSTMActorCritic(9, 2, 7, 0)
    assert learner.parameter_count == bl.lstm_parameters(9, 2, 7)
    assert bl.lstm_gradient_check() < 1e-4


@pytest.mark.parametrize("kind", ["lstm", "tdlambda"])
def test_a_conventional_learner_acquires_a_short_tmaze(kind):
    if kind == "lstm":
        pytest.importorskip("torch")
    row = ch.run_life({"kind": kind, "task": "tmaze", "delay": 2, "founder": 1,
                       "budgets": [400], "match": 600})
    assert "error" not in row
    assert row["final"]["choice"] == 1.0
    assert row["acted_right_last256"] > 0.7
    assert row["work"]["multiply_accumulates"] > 0


@pytest.mark.parametrize("task", tk.TASKS)
def test_the_solver_and_random_lives_bound_the_reading(task):
    solver = ch.run_life({"kind": "solver", "task": task, "delay": 3, "founder": 0,
                          "budgets": [16]})
    chance = ch.run_life({"kind": "random", "task": task, "delay": 3, "founder": 0,
                          "budgets": [64]})
    assert solver["final"]["choice"] == 1.0 and solver["acted_right_last256"] == 1.0
    assert 0.3 < chance["acted_right_last256"] < 0.7


def test_a_brain_life_reads_on_a_copy_and_reports_its_credit_diagnostics():
    row = ch.run_life({"kind": "brain", "task": "keydoor", "delay": 3, "founder": 0,
                       "budgets": [2, 6]})
    assert "error" not in row
    assert [r["episodes"] for r in row["readings"]] == [2, 6]
    assert row["parameters"] > 0 and row["work"]["multiply_accumulates"] > 0
    diagnostics = row["diagnostics"]
    assert diagnostics["decay_at_outcome"] == pytest.approx(0.72**3)
    assert len(diagnostics["share"]) == 1 and 0 < diagnostics["share"][0] <= 1.0 + 1e-9
    frozen = ch.run_life({"kind": "brain", "task": "tmaze", "delay": 3, "founder": 0,
                          "budgets": [2, 6], "control": "frozen"})
    assert frozen["readings"][0]["choice"] == frozen["readings"][1]["choice"]
    assert frozen["work"]["episodes"] == 0


def test_the_reading_never_changes_the_life():
    import copy

    warnings.simplefilter("ignore")
    brain = ch.make_brain("tmaze", "default", 0)
    brain.step(np.eye(9)[[0]])
    brain.step(np.eye(9)[[2]], reward=[0.0])
    before = copy.deepcopy(brain.learner.brain.efficacy)
    ch._read_brain(brain, "tmaze", 0, 3)
    assert np.array_equal(before, brain.learner.brain.efficacy)
    assert brain.basal_ganglia._pending is not None


def test_the_summary_counts_founders_at_the_gate(tmp_path):
    rows = [{"kind": "brain", "arm": "default", "control": "none", "task": "tmaze",
             "delay": 10, "parameters": 1, "final": {"choice": c, "p_right": 0.5,
                                                     "door": None},
             "acted_right_last256": 0.5, "work": {"per_episode": {"seconds": 1.0,
                                                                  "multiply_accumulates": 2.0}}}
            for c in (1.0, 1.0, 0.5)]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows))
    table = ch.summary(str(path))
    assert "| 2 of 3 |" in table



def test_the_ablated_control_empties_the_decisions_eligibility_after_its_first_transition():
    spec = {"kind": "brain", "arm": "held", "task": "tmaze", "delay": 3, "founder": 0,
            "budgets": [4]}
    intact = ch.run_life(spec)
    ablated = ch.run_life({**spec, "control": "ablated"})
    assert "error" not in ablated
    # the decision's share of the eligibility when the outcome arrives
    assert intact["diagnostics"]["share"][0] > 0.05
    assert ablated["diagnostics"]["share"][0] == 0.0
