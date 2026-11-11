"""Receipt admission and arithmetic must not turn partial or mixed successes into proof."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "reward_rhythm_audit_subject", ROOT / "reward_rhythm.py"
)
chamber = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = chamber
spec.loader.exec_module(chamber)


@pytest.fixture
def audit():
    protocol, _ = chamber.load_protocol(ROOT / "protocol-reward-2.json")
    protocol["life"].update(
        moments=8, block=2, window=2, probe_at=2, greedy_probe=4, continuation=2
    )
    protocol["life"]["disturbances"] = {"pre": 2, "post": 4, "pauses": [1]}
    runs = []
    for seed in protocol["seeds"]["confirmation"]:
        for arm in chamber.ARMS:
            brain = arm in chamber.BRAIN_ARMS
            actions = [0] * 8 if arm == "frozen" else [0, 1] * 4
            rewards = [0.0] + [float(a != b) for a, b in zip(actions, actions[1:], strict=False)]
            aroused = [False if brain else None] * 8
            run = {
                "seed": seed,
                "arm": arm,
                "actions": actions,
                "rewards": rewards,
                "feedback": [None] + rewards[:-1],
                "aroused": aroused,
                "belief": [None] * 8,
                "alternation": chamber.alternation(actions),
                "alternation_blocks": chamber.blocks(rewards[1:], 2),
                "aroused_blocks": [0.0] * 4 if brain else None,
                "income_blocks": chamber.blocks(rewards, 2),
                "belief_blocks": [0.0] * 4 if brain else None,
                "window_alternation": chamber.alternation(actions[-2:]),
                "window_aroused": 0.0 if brain else None,
                "window_income": sum(rewards[-2:]) / 2,
                "refusals": 0,
                "work": {"routine": 8 if brain else 0, "aroused": 0, "refusals": 0},
            }
            if brain:
                run.update(
                    probe_boundary={"previous_action": actions[1], "pending_reward": rewards[1]},
                    final_boundary={"previous_action": actions[-1], "pending_reward": rewards[-1]},
                    continuation={
                        "actions_equal": True,
                        "saved_arrays_equal": True,
                        "alternation": chamber.alternation(actions[2:4]),
                    },
                    greedy_probe={
                        "actions": actions[:4],
                        "alternation": chamber.alternation(actions[:4]),
                    },
                    disturbances={},
                )
                for name in ("pause1", "distractor"):
                    # Seven real actions after the saved world boundary, all alternating.
                    fork_actions = [1 - actions[1], actions[1]] * 3 + [1 - actions[1]]
                    run["disturbances"][name] = {
                        "actions": fork_actions,
                        "rewards": [1.0] * 7,
                        "feedback": [rewards[1]] + [1.0] * 6,
                        "pre_alternation": 1.0,
                        "post_alternation": 1.0,
                        "recovered": True,
                    }
            runs.append(run)
    return protocol, runs


def test_gate_requires_the_same_founders_to_meet_every_clause(audit):
    protocol, runs = audit
    live = [run for run in runs if run["arm"] == "live"]
    live[0]["window_alternation"] = 0.0
    live[1]["greedy_probe"]["alternation"] = 0.0
    live[2]["window_aroused"] = 1.0
    live[3]["continuation"]["saved_arrays_equal"] = False
    gates = chamber.aggregate(runs, protocol)["live"]["gates"]
    assert all(gates[key] == 4 for key in ("learned", "greedy", "calm", "continued"))
    assert gates["jointly_qualified"] == 1 and not gates["passed"]


@pytest.mark.parametrize("missing", ["frozen", "one-founder", "custom"])
def test_partial_or_custom_census_cannot_pass(audit, missing):
    protocol, runs = audit
    if missing == "frozen":
        runs = [run for run in runs if run["arm"] != "frozen"]
    elif missing == "one-founder":
        runs = [run for run in runs if run["seed"] != protocol["seeds"]["confirmation"][-1]]
    gates = chamber.aggregate(runs, protocol, admissible=missing != "custom")["live"]["gates"]
    assert not gates["admissible"] and not gates["passed"]
    assert gates["expected_founders"] == 5
    if missing == "frozen":
        assert gates["learned"] == gates["jointly_qualified"] == 0


def test_original_first_protocol_gates_acquisition_while_later_protocols_require_learning(audit):
    protocol, runs = audit
    for run in runs:
        if run["arm"] == "frozen":
            run["window_alternation"] = 1.0
    assert not chamber.aggregate(runs, protocol)["live"]["gates"]["passed"]
    protocol["schema"] = "steady-rhythm-reward/1"
    gates = chamber.aggregate(runs, protocol)["live"]["gates"]
    assert gates["passed"] and gates["jointly_qualified"] == 5 and gates["learned"] == 0


def write_receipt(directory, protocol, runs):
    (directory / "protocol.json").write_text(json.dumps(protocol))
    digest = chamber.sha256(directory / "protocol.json")
    declaration = {
        "instrument_revision": 2,
        "protocol_sha256": digest,
        "overrides": {},
        "seeds": protocol["seeds"]["confirmation"],
        "arms": list(chamber.ARMS),
        "frozen_protocol": True,
    }
    (directory / "declaration.json").write_text(json.dumps(declaration))
    body = {
        "declaration": declaration,
        "protocol": protocol,
        "protocol_sha256": digest,
        "frozen_protocol": True,
        "runs": runs,
        "capped": False,
        "planned_founders": [[run["seed"], run["arm"]] for run in runs],
        "completed_founders": [[run["seed"], run["arm"]] for run in runs],
        "aggregate": chamber.aggregate(runs, protocol),
        "artifacts": {
            name: chamber.sha256(directory / name) for name in ("protocol.json", "declaration.json")
        },
    }
    chamber.Receipt.build(chamber.SCHEMA, body).write(directory / "summary.json")
    return body


@pytest.mark.parametrize(
    "changed",
    [
        "reward",
        "income",
        "greedy",
        "boundary",
        "fork",
        "aggregate",
        "census",
        "protocol",
        "declaration",
    ],
)
def test_resigned_receipt_still_must_match_events_protocol_and_census(audit, tmp_path, changed):
    protocol, runs = audit
    body = write_receipt(tmp_path, protocol, runs)
    valid, reason = chamber.verify(tmp_path)
    assert valid, reason
    run = body["runs"][0]
    if changed == "reward":
        run["rewards"][0] = 1.0
    elif changed == "income":
        run["window_income"] = 0.0
    elif changed == "greedy":
        run["greedy_probe"]["alternation"] = 0.0
    elif changed == "boundary":
        run["probe_boundary"]["pending_reward"] = 0.0
    elif changed == "fork":
        run["disturbances"]["pause1"]["feedback"][0] = 0.0
    elif changed == "aggregate":
        body["aggregate"]["live"]["gates"]["passed"] = False
    elif changed == "census":
        body["runs"].pop()
        body["completed_founders"].pop()
    elif changed == "protocol":
        body["protocol"] = deepcopy(protocol)
        body["protocol"]["gates"]["share"] = 1
    else:
        body["declaration"] = deepcopy(body["declaration"])
        body["declaration"]["seeds"] = [999]
    # A new canonical digest does not make unsupported arithmetic true.
    chamber.Receipt.build(chamber.SCHEMA, body).write(tmp_path / "summary.json")
    valid, reason = chamber.verify(tmp_path)
    assert not valid, reason


def test_legacy_receipt_preserves_historical_labels_without_claiming_corrected_validation(tmp_path):
    raw = chamber.PROTOCOL_PATH.read_bytes()
    (tmp_path / "protocol.json").write_bytes(raw)
    body = {
        "declaration": {},
        "protocol_sha256": chamber.sha256(tmp_path / "protocol.json"),
        "artifacts": {},
        "aggregate": {"historical_gate": "retained, not recomputed"},
    }
    chamber.Receipt.build(chamber.SCHEMA, body).write(tmp_path / "summary.json")
    before = (tmp_path / "summary.json").read_bytes()
    valid, reason = chamber.verify(tmp_path)
    assert (
        valid
        and "legacy receipt" in reason
        and "not validated by the corrected instrument" in reason
    )
    assert (tmp_path / "summary.json").read_bytes() == before


@pytest.mark.parametrize("cap", ["-1", "nan", "inf"])
def test_invalid_time_cap_rejected_before_creating_output(tmp_path, cap):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        chamber.main(["--out", str(output), "--time-cap", cap])
    assert not output.exists()
