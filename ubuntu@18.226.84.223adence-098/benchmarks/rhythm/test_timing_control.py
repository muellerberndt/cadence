"""Regression guards for the separate, spent-founder physical timing control."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("timing_control", ROOT / "timing_control.py")
control = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = control
spec.loader.exec_module(control)
sys.path.pop(0)


@pytest.fixture
def config():
    protocol = json.loads((ROOT / "protocol-timing.json").read_text())
    return {"bounds": protocol["timing_acceptance"]["bounds"], "burners": 10}


def measured(seed=0, name="paced_regular"):
    return {
        "seed": seed,
        "name": name,
        "actions": [[i % 2] * 4 for i in range(8)],
        "missed_deadlines": 0,
        "solve_ms": [1.0] * 8,
        "lateness_ms": [1.0] * 8,
        "wall_seconds": 0.703,
        "elapsed_seconds": 0.71,
        "timing": {
            "due_ms": list(range(0, 800, 100)),
            "begin_ms": list(range(1, 801, 100)),
            "end_ms": list(range(2, 802, 100)),
            "deadline_ms": list(range(100, 900, 100)),
        },
    }


def test_all_sixteen_cases_required_and_no_fresh_or_brain_qualification(config):
    cases = [measured(seed, name) for seed, name in control.PLANNED]
    result = control.evaluate(cases, config, None)
    assert result["competent_physical_control"] and result["passed_cases"] == 16
    assert not result["fresh_confirmation"] and not result["brain_timing_qualified"]
    assert not control.evaluate(cases[:-1], config, None)["competent_physical_control"]
    assert not control.evaluate(cases, config, "interrupted")["competent_physical_control"]
    for corrupt in (cases[1:], cases + cases[:1], cases[:1] + cases[:1] + cases[2:]):
        with pytest.raises(AssertionError):
            control.evaluate(corrupt, config, None)


def test_one_failed_timing_case_cannot_be_averaged_away(config):
    cases = [measured(seed, name) for seed, name in control.PLANNED]
    cases[-1]["timing"]["end_ms"][4] += 40
    result = control.evaluate(cases, config, None)
    assert result["complete"] and result["passed_cases"] == 15
    assert not result["competent_physical_control"]
    assert result["readings"][-1]["timing"]["periods"][0]["max_error_ms"] == 40


def test_physical_scheduler_calls_recurrent_control_once_per_frozen_event(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(control.chamber.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(control.chamber.time, "process_time", lambda: clock[0])
    monkeypatch.setattr(
        control.chamber.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    class Actor:
        def __init__(self):
            self.count = 0

        def act(self, observation):
            assert np.array_equal(observation, np.tile([1, 0, 0, 0], (4, 1)))
            result = np.full(4, self.count % 2)
            self.count += 1
            clock[0] += 0.002
            return result

    actor, work = Actor(), control.ControlWork()
    due = np.array([0, 50, 200, 300, 300], dtype=float)
    observations = np.tile([1, 0, 0, 0], (5, 4, 1))
    result = control.chamber.real_time_run(actor, observations, due, work)
    assert actor.count == 5 and len(work.records) == 5
    assert result["timing"]["due_ms"] == due.tolist()
    np.testing.assert_allclose(result["timing"]["end_ms"], [2, 52, 202, 302, 304])
    assert result["actions"] == [r["answer"] for r in work.records]
    assert control.work_totals(work.records)["action_rows"] == 20


@pytest.fixture
def recorded(tmp_path, config):
    parent = tmp_path / "basis"
    probe = parent / "seed0-efference-every/probe-flipflop.npz"
    probe.parent.mkdir(parents=True)
    model = control.chamber.FlipFlop(0.5)
    model.weights[5:, :] = [[0, 1], [1, 0]]
    model.previous[:] = 1
    model.save(probe)
    x = np.tile([1, 0, 0, 0], (4, 1))
    case = measured()
    np.savez(
        parent / "inputs-seed0.npz",
        **{
            "window/observations": np.stack([x] * 70),
            "cadence/regular/due_ms": np.array(case["timing"]["due_ms"]),
        },
    )
    actions = [model.act(x).tolist() for _ in range(8)]
    assert actions == case["actions"]
    after = tmp_path / "seed0-paced_regular-after.npz"
    model.save(after)
    case.update(
        {
            "burners": 0,
            "records": [
                {"answer": a, "error": None, "cpu_seconds": 0.0001, "wall_seconds": 0.0002}
                for a in actions
            ],
            "checkpoint_io": [
                {"operation": op, "file": p.name, "bytes": p.stat().st_size, "seconds": 0.001}
                for op, p in (("read", probe), ("write", after))
            ],
        }
    )
    case["work"] = control.work_totals(case["records"])
    control.verify_case(case, tmp_path, config)
    return tmp_path, case


@pytest.mark.parametrize(
    "mutation",
    ["schedule", "census", "work", "action", "io", "nan", "rounded", "elapsed", "internal_wall"],
)
def test_verifier_rejects_schedule_action_work_and_custody_changes(recorded, config, mutation):
    directory, original = recorded
    case = copy.deepcopy(original)
    if mutation == "schedule":
        case["timing"]["due_ms"][3] += 1
    elif mutation == "census":
        case["records"].pop()
    elif mutation == "work":
        case["work"]["action_calls"] -= 1
    elif mutation == "action":
        case["actions"][2][0] = 1 - case["actions"][2][0]
        case["records"][2]["answer"] = case["actions"][2]
    elif mutation == "io":
        case["checkpoint_io"][0]["bytes"] += 1
    elif mutation == "nan":
        case["records"][0]["cpu_seconds"] = float("nan")
    elif mutation == "rounded":
        case["solve_ms"] = [0] * 8
    elif mutation == "elapsed":
        case["elapsed_seconds"] = -1
    else:
        case["records"][0]["wall_seconds"] = 0.1
        case["work"] = control.work_totals(case["records"])
    with pytest.raises(AssertionError):
        control.verify_case(case, directory, config)


def test_changed_baseline_receipt_is_refused_before_artifact_use(tmp_path):
    (tmp_path / "basis").mkdir()
    (tmp_path / "basis/summary.json").write_text("{}\n")
    with pytest.raises(AssertionError, match="different baseline"):
        control.basis(tmp_path)


def test_display_rounding_midpoint_preserves_unrounded_acceptance(recorded, config):
    directory, case = recorded
    # Captured first-run cancellation: absolute-clock subtraction rounded down,
    # but the stored relative timestamp subtracts to 1.3965000049211085ms.
    case["timing"]["begin_ms"][2] = 201.3965000049211085
    case["lateness_ms"][2] = 1.396
    case["solve_ms"][2] = 0.603
    control.verify_case(case, directory, config)
    case["lateness_ms"][2] += 0.002
    with pytest.raises(AssertionError):
        control.verify_case(case, directory, config)


def test_failed_policy_call_is_charged_without_invented_completed_arithmetic():
    class Broken:
        def act(self, observation):
            raise RuntimeError("failed during policy call")

    work = control.ControlWork()
    with pytest.raises(RuntimeError):
        work.act(Broken(), np.zeros((4, 4)))
    totals = control.work_totals(work.records)
    assert totals["action_calls"] == totals["failed_calls"] == 1
    assert totals["action_rows"] == 4 and totals["successful_dense_policy_macs"] == 0
    assert totals["wall_seconds"] >= 0 and totals["cpu_seconds"] >= 0


@pytest.mark.parametrize("artifact", ["run-start.json", "seed0-paced_regular-after.npz"])
def test_interrupted_attempt_cannot_be_overwritten(tmp_path, monkeypatch, artifact):
    monkeypatch.setattr(control, "declaration", lambda *args, **kwargs: {})
    (tmp_path / artifact).write_text("preserved")
    with pytest.raises((AssertionError, FileExistsError)):
        control.run(tmp_path)
    assert (tmp_path / artifact).read_text() == "preserved"


def test_failed_case_replay_keeps_successful_partial_calls(recorded):
    directory, case = recorded
    failed = {
        "seed": 0,
        "name": "paced_regular",
        "outer_timing_complete": False,
        "records": case["records"][:3],
        "checkpoint_io": case["checkpoint_io"][:1],
        "elapsed_seconds": 0.1,
    }
    failed["work"] = control.work_totals(failed["records"])
    control.verify_failed(failed, directory)
    failed["records"][1]["answer"][0] = 0
    with pytest.raises(AssertionError, match="saved recurrent policy"):
        control.verify_failed(failed, directory)


@pytest.mark.parametrize("changed", ["producing_source", "source_capsule", "bounds", "freshness"])
def test_declaration_rejects_source_drift_and_resigned_gate_changes(
    tmp_path, monkeypatch, config, changed
):
    (tmp_path / "basis").mkdir()
    (tmp_path / "source").mkdir()
    original = tmp_path / "policy.py"
    capsule = tmp_path / "source/policy.py"
    original.write_text("original policy\n")
    capsule.write_bytes(original.read_bytes())
    source_files = [("source/policy.py", original)]
    monkeypatch.setattr(control, "sources", lambda: source_files)
    env = control.environment()
    baseline = {
        "protocol": {"timing_acceptance": {"bounds": config["bounds"]}},
        "declaration": {**env, "burners": 10},
    }
    monkeypatch.setattr(control, "basis", lambda directory: baseline)
    body = {
        "planned": control.PLANNED,
        "fresh_confirmation": False,
        "baseline_sha256": control.BASE_SHA,
        "protocol_sha256": control.PROTOCOL_SHA,
        "bounds": copy.deepcopy(config["bounds"]),
        "burners": 10,
        "cap_seconds": 180,
        "environment": env,
        "artifacts": {},
    }
    destination = tmp_path / "declaration.json"
    control.Receipt.build(control.SCHEMA + "/declaration", body, source_files).write(destination)
    control.declaration(tmp_path, current=True)
    if changed == "producing_source":
        original.write_text("edited policy\n")
    elif changed == "source_capsule":
        capsule.write_text("edited capsule\n")
    else:
        if changed == "bounds":
            body["bounds"]["missed_deadlines"] = 100
        else:
            body["fresh_confirmation"] = True
        control.Receipt.build(control.SCHEMA + "/declaration", body, source_files).write(
            destination
        )
    with pytest.raises(AssertionError):
        control.declaration(tmp_path, current=True)
