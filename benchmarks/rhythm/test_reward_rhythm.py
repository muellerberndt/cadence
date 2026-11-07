"""Reward-rhythm chamber guards: the world's rule, the arms, custody and a smoke run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load_script("rhythm_inputs")
chamber = load_script("reward_rhythm")


@pytest.fixture(scope="module")
def protocol():
    return chamber.load_protocol()


def test_protocol_is_frozen_with_fresh_seeds_and_the_declared_point(protocol):
    declared, digest = protocol
    assert digest == hashlib.sha256(chamber.PROTOCOL_PATH.read_bytes()).hexdigest()
    assert declared["schema"] == chamber.SCHEMA
    assert len(declared["seeds"]["confirmation"]) == 5
    assert not set(declared["seeds"]["confirmation"]) & set(declared["seeds"]["development"])
    point = declared["operating_point"]
    assert point["efference_amplitude"] > 0 and point["efference_decay"] == 0.0
    assert declared["world"]["reward"]["alternate"] > declared["world"]["reward"]["repeat"]
    assert declared["life"]["probe_at"] + declared["life"]["window"] <= declared["life"]["moments"]
    assert declared["gates"]["share"] <= len(declared["seeds"]["confirmation"])


def test_the_world_pays_a_changed_step_and_nothing_else(protocol):
    rule = protocol[0]["world"]["reward"]
    assert chamber.pay(None, 0, rule) == rule["repeat"]
    assert chamber.pay(0, 1, rule) == rule["alternate"]
    assert chamber.pay(1, 1, rule) == rule["repeat"]
    assert chamber.alternation([0, 1, 0, 1]) == 1.0
    assert chamber.alternation([0, 0, 0]) == 0.0
    assert chamber.alternation([0, None, 1]) == 0.0  # a missed step is a repeat
    assert chamber.blocks([1.0, None, 0.0, 1.0], 2) == [1.0, 0.5]


def test_arms_differ_only_as_declared(protocol):
    declared, _ = protocol
    live = chamber.make_brain(0, declared, "live")
    nocopy = chamber.make_brain(0, declared, "nocopy")
    defaults = chamber.make_brain(0, declared, "defaults")
    step = chamber.make_brain(0, declared, "step")
    frozen = chamber.make_brain(0, declared, "frozen")
    lam0 = chamber.make_brain(0, declared, "lambda_control")
    assert live.efference is not None and nocopy.efference is None
    assert live.arousal is not None and step.arousal is None and frozen.arousal is None
    assert live.basal_ganglia.config.eta == declared["operating_point"]["eta"]
    assert live.basal_ganglia.config.eta_critic == declared["operating_point"]["eta_critic"]
    composed = chamber.Brain.compose(inputs.INPUTS, inputs.ACTIONS, modules=(32,), seed=0,
                                     recall=False)
    assert defaults.basal_ganglia.config == composed.basal_ganglia.config
    assert lam0.basal_ganglia.config.lam == declared["operating_point"].get("lam_control", 0.0)
    assert live.efference.amplitude == declared["operating_point"]["efference_amplitude"]
    # the same founder weights behind every brain arm; the brain without the copy holds a
    # subset of the live brain's synapses (the copy's region and projection are appended)
    for other in (defaults, step, frozen, lam0):
        np.testing.assert_array_equal(live.brain.efficacy, other.brain.efficacy)

    def rows(brain):
        c = brain.connectome
        return {
            (int(p), int(q), float(s))
            for p, q, s in zip(
                np.asarray(c.pre), np.asarray(c.post), np.asarray(c.sign), strict=True
            )
        }

    assert rows(nocopy) < rows(live)


def test_tabular_control_learns_the_beat_from_the_one_bit(protocol):
    rule = protocol[0]["world"]["reward"]
    table = chamber.Tabular(3, protocol[0]["tabular"])
    actions, previous, reward = [], None, None
    for _ in range(400):
        action = table.act(reward)
        reward = chamber.pay(previous, action, rule)
        table.outcome(reward)
        actions.append(action)
        previous = action
    assert chamber.alternation(actions[-64:]) >= 0.8


def test_smoke_run_charges_every_arm_and_verifies(protocol, tmp_path):
    out = tmp_path / "run"
    code = chamber.main(
        [
            "--out",
            str(out),
            "--seeds",
            "0",
            "--moments",
            "472",
            "--arms",
            "live",
            "nocopy",
            "frozen",
            "tabular",
            "random",
        ]
    )
    assert code == 0
    valid, reason = chamber.verify(out)
    assert valid, reason
    body = json.loads((out / "summary.json").read_text())["body"]
    assert body["frozen_protocol"] is False
    assert body["declaration"]["overrides"] == {
        "moments": 472,
        "seeds": [0],
        "arms": ["live", "nocopy", "frozen", "tabular", "random"],
    }
    assert body["completed_founders"] == body["planned_founders"]
    runs = {run["arm"]: run for run in body["runs"]}
    assert set(runs) == {"live", "nocopy", "frozen", "tabular", "random"}
    for run in runs.values():
        assert run["final_boundary"] == {
            "previous_action": run["actions"][-1],
            "pending_reward": run["rewards"][-1],
        }
        assert run["work"]["calls_seconds"] > 0
    for arm in ("live", "nocopy", "frozen"):
        run = runs[arm]
        assert len(run["actions"]) == 472 and run["refusals"] == 0
        assert run["continuation"]["actions_equal"] and run["continuation"]["saved_arrays_equal"]
        assert set(run["disturbances"]) == {"pause1", "pause2", "pause4", "distractor"}
        assert run["work"]["routine"] + run["work"]["aroused"] == 472
        assert run["work"]["checkpoints"] == 6  # initial, probe/load, comparison pair, final
        assert run["probe_work"]["calls_seconds"] > 0
        boundary = run["probe_boundary"]
        assert boundary == {
            "previous_action": run["actions"][399],
            "pending_reward": run["rewards"][399],
        }
        for fork in run["disturbances"].values():
            assert fork["feedback"][0] == boundary["pending_reward"]
            assert fork["rewards"][0] == chamber.pay(
                boundary["previous_action"], fork["actions"][0], protocol[0]["world"]["reward"]
            )
        assert (out / f"seed0-{arm}" / "probe.npz").exists()
    assert runs["frozen"]["work"]["aroused"] == 0 and runs["frozen"]["work"]["learning_sweeps"] == 0
    assert runs["frozen"]["alternation"] == 0.0  # a greedy founder holds one action
    assert runs["random"]["aroused_blocks"] is None
    gates = body["aggregate"]["live"]["gates"]
    assert set(gates) == {
        "acquired",
        "learned",
        "greedy",
        "calm",
        "continued",
        "jointly_qualified",
        "admissible",
        "expected_founders",
        "required",
        "passed",
    }
    assert gates["learned"] <= gates["acquired"]
    assert body["aggregate"]["live"]["continuation_equal"] == 1


def test_disturbance_fork_consumes_the_actual_pending_outcome(protocol, tmp_path):
    declared = deepcopy(protocol[0])
    # Nonzero repeat pay makes dropping even a repeated action's outcome observable.
    declared["world"]["reward"] = {"repeat": 0.25, "alternate": 1.0}
    declared["life"]["disturbances"] = {"pre": 2, "post": 3, "pauses": [1]}
    brain = chamber.make_brain(3, declared, "live")
    x = chamber.observation(inputs.KIND_DRIVE)
    previous = int(brain.live(x)[0])
    pending_reward = chamber.pay(None, previous, declared["world"]["reward"])
    probe = brain.save(tmp_path / "pending.npz")
    boundary = {"previous_action": previous, "pending_reward": pending_reward}
    actual = chamber.disturb(probe, "live", "pause1", declared, chamber.Work(), boundary=boundary)
    expected_brain = chamber.Brain.load(probe)
    expected_actions, expected_rewards, expected_feedback = [], [], []
    kinds = [inputs.KIND_DRIVE] * 2 + [inputs.KIND_PAUSE] + [inputs.KIND_DRIVE] * 3
    for kind in kinds:
        expected_feedback.append(pending_reward)
        action = int(expected_brain.live(chamber.observation(kind), reward=[pending_reward])[0])
        pending_reward = chamber.pay(previous, action, declared["world"]["reward"])
        previous = action
        expected_actions.append(action)
        expected_rewards.append(pending_reward)
    assert actual["actions"] == expected_actions
    assert actual["rewards"] == expected_rewards
    assert actual["feedback"] == expected_feedback


@pytest.mark.parametrize("arm", ["live", "step"])
def test_refused_answer_after_feedback_retries_without_crediting_it_twice(
    protocol, monkeypatch, arm
):
    brain = chamber.make_brain(3, protocol[0], arm)
    work = chamber.Work()
    x = chamber.observation(inputs.KIND_DRIVE)
    chamber.moment(brain, arm, x, None, work)
    updates, counted = brain.basal_ganglia.updates, work.counts["learning_sweeps"]
    settled = brain._settled

    def refuse_after_feedback(*args, **kwargs):
        config = brain.learner.config
        brain.learner.config = replace(config, free_steps=0, tolerance=0.0)
        try:
            return settled(*args, **kwargs)
        finally:
            brain.learner.config = config

    monkeypatch.setattr(brain, "_settled", refuse_after_feedback)
    refused = chamber.moment(brain, arm, x, 1.0, work)
    assert refused["action"] is None and refused["retry_reward"] is None
    assert brain.basal_ganglia.updates == updates + 1
    assert work.counts["learning_sweeps"] == counted + brain.last_learning["free_steps"]
    monkeypatch.setattr(brain, "_settled", settled)
    retried = chamber.moment(brain, arm, x, refused["retry_reward"], work)
    assert retried["action"] is not None and work.counts["refusals"] == 1
    assert brain.basal_ganglia.updates == updates + 1


def test_refused_routine_forecast_keeps_the_actual_reward(protocol):
    declared = deepcopy(protocol[0])
    declared["arousal"]["youth"] = 0
    brain = chamber.make_brain(3, declared, "live")
    work = chamber.Work()
    chamber.moment(brain, "live", chamber.observation(inputs.KIND_DRIVE), None, work)
    lived, mood = brain._lived, brain.arousal.to_dict()
    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=0, tolerance=0.0)
    x = chamber.observation(inputs.KIND_DISTRACTOR)
    refused = chamber.moment(brain, "live", x, 1.0, work)
    assert refused["action"] is None and refused["retry_reward"] == 1.0
    assert brain._lived is lived and brain.arousal.to_dict() == mood
    brain.learner.config = config
    retried = chamber.moment(brain, "live", x, refused["retry_reward"], work)
    assert retried["action"] is not None and work.counts["refusals"] == 1


def test_unrelated_error_is_not_hidden_by_a_stale_refusal(protocol):
    brain = chamber.make_brain(3, protocol[0], "live")
    brain.learner.config = replace(brain.learner.config, free_steps=0)
    work = chamber.Work()
    x = chamber.observation(inputs.KIND_DRIVE)
    refused = chamber.moment(brain, "live", x, None, work)
    assert refused["action"] is None and refused["retry_reward"] is None
    with pytest.raises(RuntimeError, match="preceding action"):
        chamber.moment(brain, "live", x, 0.0, work)
    assert work.counts["refusals"] == 1


def test_missed_world_steps_preserve_pending_feedback_without_inventing_credit(
    protocol, monkeypatch, tmp_path
):
    replies = iter([
        {"action": 0, "aroused": True},
        {"action": None, "aroused": None, "retry_reward": 0.0},
        {"action": 1, "aroused": True},
        {"action": None, "aroused": None, "retry_reward": None},
        {"action": 0, "aroused": True},
    ])
    received = []

    def attempt(brain, arm, x, reward, work):
        received.append(reward)
        return next(replies)

    monkeypatch.setattr(chamber, "moment", attempt)
    life = chamber.live_life(
        object(), "live", protocol[0], 0, tmp_path, moments=5, probe_at=6
    )
    assert received == life["feedback"] == [None, 0.0, 0.0, 1.0, None]
    assert life["rewards"] == [0.0, 0.0, 1.0, 0.0, 1.0]
    assert life["final_boundary"] == {"previous_action": 0, "pending_reward": 1.0}


def test_greedy_probe_charges_its_elapsed_calls(protocol, tmp_path, monkeypatch):
    brain = chamber.make_brain(3, protocol[0], "live")
    brain.live(chamber.observation(inputs.KIND_DRIVE))
    path = brain.save(tmp_path / "probe.npz")
    ticks = iter(range(2 * protocol[0]["life"]["greedy_probe"]))
    monkeypatch.setattr(chamber.time, "perf_counter", lambda: next(ticks))
    work = chamber.Work()
    chamber.greedy_probe(path, protocol[0], work)
    assert work.seconds == work.counts["probe_acts"] == protocol[0]["life"]["greedy_probe"]


@pytest.mark.parametrize(
    "bad",
    [
        ["--seeds", "0", "0"],
        ["--arms", "live", "teacher"],
        ["--moments", "100"],
    ],
)
def test_invalid_cli_is_rejected_before_creating_an_attempt(tmp_path, bad):
    out = tmp_path / "never"
    with pytest.raises(SystemExit):
        chamber.main(["--out", str(out), *bad])
    assert not out.exists()


def test_protocol_2_declares_exact_credit_against_reward_1_as_its_control(protocol):
    one, _ = protocol
    two, digest = chamber.load_protocol(ROOT / "protocol-reward-2.json")
    assert digest == hashlib.sha256((ROOT / "protocol-reward-2.json").read_bytes()).hexdigest()
    assert two["schema"] == "steady-rhythm-reward/2"
    assert two["operating_point"]["lam"] == 0.0 and two["operating_point"]["lam_control"] == 0.95
    point_one = {k: v for k, v in one["operating_point"].items() if k not in ("lam", "note")}
    point_two = {
        k: v for k, v in two["operating_point"].items() if k not in ("lam", "lam_control", "note")
    }
    assert point_one == point_two
    assert {k: v for k, v in two["arousal"].items()} == one["arousal"]
    assert two["world"] == one["world"] and two["life"] == one["life"]
    assert not set(two["seeds"]["confirmation"]) & set(one["seeds"]["confirmation"])
    assert not set(two["seeds"]["confirmation"]) & set(two["seeds"]["development"])
    assert (
        two["history"]["reward/1"]["protocol_sha256"]
        == hashlib.sha256(chamber.PROTOCOL_PATH.read_bytes()).hexdigest()
    )
