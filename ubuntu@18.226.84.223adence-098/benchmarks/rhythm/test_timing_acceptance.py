"""New timing-protocol guards; historical frozen protocols remain unchanged."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load("rhythm_inputs")
acceptance = load("timing_acceptance")
chamber = load("steady_rhythm")


def test_interval_order_control_is_deterministic_and_does_not_change_observations(tmp_path):
    protocol, _ = inputs.load_protocol(ROOT / "protocol-timing.json")
    frozen = inputs.freeze_inputs(tmp_path / "first.npz", seed=0, protocol=protocol)
    original, _ = inputs.load_protocol(ROOT / "protocol-2.json")
    old = inputs.freeze_inputs(tmp_path / "old.npz", seed=0, protocol=original)
    for key in old:
        np.testing.assert_array_equal(frozen[key], old[key])
    ordered = frozen["cadence/ordered/due_ms"]
    shuffled = frozen["cadence/shuffled_time/due_ms"]
    assert not np.array_equal(ordered, shuffled)
    np.testing.assert_array_equal(np.sort(np.diff(ordered)), np.sort(np.diff(shuffled)))
    repeated = inputs.interval_schedules(0, protocol["cadence"]["events"], [50, 100, 150])
    np.testing.assert_array_equal(shuffled, repeated["shuffled_time"]["due_ms"])


def measured():
    return {
        "actions": [[i % 2] * 4 for i in range(8)],
        "missed_deadlines": 0,
        "timing": {
            "due_ms": list(range(0, 800, 100)),
            "begin_ms": list(range(1, 801, 100)),
            "end_ms": list(range(2, 802, 100)),
            "deadline_ms": list(range(100, 900, 100)),
        },
    }


def test_timing_recomputed_from_unrounded_events_including_period_and_drift():
    value = measured()
    result = acceptance.timing_readings(value)
    assert result["phase_drift_ms"] == 0
    assert result["max_lateness_ms"] == 1
    assert all(p["event_period"] == 2 and p["max_error_ms"] == 0 for p in result["periods"])
    value["timing"]["begin_ms"][4] += 0.123456
    value["timing"]["end_ms"][4] += 0.123456
    result = acceptance.timing_readings(value)
    assert result["phase_drift_ms"] == pytest.approx(0.123456)
    assert result["periods"][0]["max_error_ms"] == pytest.approx(0.123456)


@pytest.mark.parametrize("mutation", ["deadline", "census", "miss", "early", "nan"])
def test_timing_rejects_inconsistent_event_record(mutation):
    value = measured()
    if mutation == "deadline":
        value["timing"]["deadline_ms"][2] += 1
    elif mutation == "census":
        value["actions"].pop()
    elif mutation == "miss":
        value["timing"]["end_ms"][2] = 301
    elif mutation == "early":
        value["timing"]["begin_ms"][2] = 199
    else:
        value["timing"]["end_ms"][2] = float("nan")
    with pytest.raises(AssertionError):
        acceptance.timing_readings(value)


def test_refusals_stay_in_pair_denominator():
    assert acceptance.alternation([[0] * 4, [1] * 4, [-1] * 4, [0] * 4]) == [1 / 3] * 4


def test_confirmation_requires_exact_census_and_no_overrides(monkeypatch):
    protocol, _ = inputs.load_protocol(ROOT / "protocol-timing.json")
    declaration = {
        "seeds": protocol["seeds"]["confirmation"],
        "recipes": protocol["run_recipes"],
        "arms": protocol["run_arms"],
        "cadence_runs": True,
        "frozen_protocol": True,
        "overrides": {},
    }
    runs = [
        {"seed": s, "recipe": r, "arm": a}
        for s in declaration["seeds"]
        for r in declaration["recipes"]
        for a in declaration["arms"]
    ]
    monkeypatch.setattr(acceptance, "founder_gate", lambda run, protocol: {"passed": True})
    assert acceptance.evaluate(runs, protocol, declaration, False)["passed"]
    assert not acceptance.evaluate(runs[:-1], protocol, declaration, False)["passed"]
    assert not acceptance.evaluate(runs + runs[:1], protocol, declaration, False)["passed"]
    assert not acceptance.evaluate(runs, protocol, declaration, True)["passed"]
    declaration["overrides"] = {"burners": 1}
    assert not acceptance.evaluate(runs, protocol, declaration, False)["passed"]
    declaration["overrides"] = {}
    declaration["seeds"] = [0, 1, 2, 3, 4]
    assert not acceptance.evaluate(runs, protocol, declaration, False)["passed"]


def test_short_native_run_charges_restored_fork_untaught_and_custody(tmp_path):
    out = tmp_path / "run"
    assert (
        chamber.main(
            [
                "--out",
                str(out),
                "--protocol",
                str(ROOT / "protocol-timing.json"),
                "--seeds",
                "0",
                "--recipes",
                "efference",
                "--arms",
                "every",
                "--bouts",
                "1",
                "--events-per-bout",
                "2",
                "--window",
                "8",
                "--no-cadence",
            ]
        )
        == 0
    )
    assert chamber.verify(out)[0]
    raw = json.loads((out / "summary.json").read_text())
    body = raw["body"]
    assert not body["timing_acceptance"]["passed"]
    assert body["declaration"]["threads"]["VECLIB_MAXIMUM_THREADS"] == "1"
    run = body["runs"][0]
    assert run["window"]["branches"]["restored"]["work"]["action_attempts"] == 8
    assert run["window"]["branches"]["untaught"]["work"]["action_attempts"] == 13
    assert run["disturbances"]["pause2"]["custody"]["twin_work"]["action_attempts"] == 33
    acceptance.verify_body(body, out)
    altered = copy.deepcopy(body)
    altered["runs"][0]["window"]["branches"]["restored"]["work"]["action_sweeps"] += 32
    (out / "runs.jsonl").write_text(json.dumps(altered["runs"][0]) + "\n")
    with pytest.raises(AssertionError, match="charged work"):
        acceptance.verify_body(altered, out)


def test_actuator_period_and_drift_include_variable_solve_latency():
    value = measured()
    value["timing"]["end_ms"][4] += 40
    reading = acceptance.timing_readings(value)
    assert reading["max_lateness_ms"] == 1
    assert reading["phase_drift_ms"] == 40
    assert reading["periods"][0]["max_error_ms"] == 40
    protocol, _ = inputs.load_protocol(ROOT / "protocol-timing.json")
    assert (
        reading["periods"][0]["max_error_ms"]
        > protocol["timing_acceptance"]["bounds"]["period_error_ms"]
    )


def test_accelerate_thread_limit_is_set_before_numpy_import_and_declared():
    assert "VECLIB_MAXIMUM_THREADS" in chamber.THREAD_VARIABLES
    assert chamber.os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
    source = (ROOT / "steady_rhythm.py").read_text()
    assert source.index('os.environ["VECLIB_MAXIMUM_THREADS"] = "1"') < source.index(
        "import numpy as np"
    )


@pytest.mark.parametrize("bad", [-1, float("nan"), [0.1]])
def test_process_cpu_time_requires_finite_nonnegative_scalar(bad):
    report = {"operation": "act", "call_seconds": 1, "process_cpu_seconds": bad}
    with pytest.raises(AssertionError):
        acceptance.verify_work(chamber.Work().summary(), [report])
