"""Native transfer gates and custody must survive negative and interrupted work."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent


@pytest.fixture
def native(monkeypatch):
    for name in ("extract", "run", "relations", "relation_development", "continual",
                 "online_curriculum", "native_online"):
        spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["native_online"]


def write_fixture(native, folder):
    folder.mkdir()
    arrays, panels = {}, {}
    for offset, (name, count) in enumerate(native.PANEL_COUNTS.items()):
        x = np.zeros((count, 650), dtype=np.float64)
        x[:, 0] = np.arange(count) + offset / 10
        y = np.arange(count, dtype=np.int64)
        arrays[name + "_inputs"], arrays[name + "_labels"] = x, y
        panels[name] = [{"input_sha256": native.harness.row_hash(row), "label": int(label)}
                        for row, label in zip(x, y, strict=True)]
    # A malformed heldout payload must stay unopened by the preflight.
    arrays["heldout_inputs"] = np.array([np.nan])
    np.savez(folder / "school.npz", **arrays)
    (folder / "provenance.json").write_text(json.dumps({"panels": panels}))


def write_reference(native, folder):
    folder.mkdir()
    model = native.harness.make_brain("qualified", 0, native.ARGS)
    protocol = {
        "founder_seeds": [21, 22, 23, 24, 25],
        "model_construction": native.MODEL_CONSTRUCTION,
        "learner_config": model.learner.config.to_dict(),
        "neuron_model": model.brain.neuron_model.to_dict(),
        "library_sources": native.harness.sources(),
        "runtime_precision": native.runtime_precision(),
    }
    native.harness.write_json(folder / "protocol.json", protocol)
    digest = native.harness.sha256(folder / "protocol.json")
    native.harness.write_json(folder / "summary.json", {
        "all_founders_passed": True, "founder_denominator": 5, "recipe_sha256": digest,
        "outcomes": [{"seed": seed, "passed": True, "status": "confirmed",
                      "recipe_sha256": digest, "process_exit_code": 0, "old_passed": True,
                      "development_passed": True, "continuation_passed": True,
                      "heldout_read": True} for seed in protocol["founder_seeds"]],
    })


@pytest.fixture
def capsule(native, monkeypatch, tmp_path):
    fixture, reference, attempt = (tmp_path / name for name in ("fixture", "reference", "attempt"))
    write_fixture(native, fixture)
    write_reference(native, reference)
    protocol = native.prepare(attempt, fixture, reference)
    for name, value in protocol["threads"].items():
        monkeypatch.setenv(name, value)
    return attempt, protocol


def stage_protocol():
    return {
        "seed": 0, "stage_counts": [2, 4, 24],
        "learner_config": {"tolerance": 0.003},
        "positions": {"2": [0, 9], "4": [0, 9, 17, 22], "24": list(range(24))},
        "orders_including_next_lesson": {"2": [0, 9], "4": [0, 9], "24": [0, 1]},
        "lesson_caps": {"2": 1, "4": 1, "24": 1},
        "query_every": {"2": 1, "4": 1, "24": 1},
        "selected_correct": {"2": 2, "4": 4, "24": 18},
        "independent_correct": {"independent_train": 14, "development": 15},
        "teaching_seconds": 7, "worker_seconds": 11, "admission_mib": 144,
        "output_cap_mib": 160, "threads": {"OMP_NUM_THREADS": "1"},
    }


def valid_lesson():
    return {
        "accepted": 1, "failed": False,
        "phases": {name: {"qualified": [True], "full_residual": [0.001],
                          "cache_defect": [0.0]}
                   for name in ("free", "nudged", "opposite")},
        "report": {"total_stagnation_checks": 0},
        "phase_row_sweeps": 3, "reported_residual_checks": 3,
        "reported_row_residual_checks": 3, "independent_residual_checks": 3,
    }


def valid_continuation():
    return {"lessons": [valid_lesson(), valid_lesson()], "continued_arrays_equal": True,
            "accepted_equal": True, "phase_payloads_retained": True}


def fake_stage(native, monkeypatch, *, independent_correct=14, continuation=None):
    class Model:
        def save(self, path):
            path.write_bytes(b"checkpoint")
            return path

    model = Model()
    monkeypatch.setattr(native.harness, "make_brain", lambda *args: model)
    monkeypatch.setattr(native.Brain, "load", lambda path: Model())
    monkeypatch.setattr(native.continual, "graph_lesson", lambda *args: valid_lesson())
    monkeypatch.setattr(native.continual, "graph_checkpoint_continuation",
                        lambda *args: copy.deepcopy(continuation or valid_continuation()))
    calls = []

    def recall(brain, inputs, labels):
        count = len(labels)
        correct = 0 if not calls else count if count < 24 else 18
        if count == 18:
            correct = independent_correct
        calls.append(count)
        return {"correct": correct, "examples": count, "refusals": 0,
                "work": {"calls": count, "row_sweeps": count,
                         "reported_residual_checks": count, "reported_stagnation_checks": 0,
                         "independent_residual_checks": count}}

    monkeypatch.setattr(native.harness, "free_recall", recall)
    data = {name + suffix: np.zeros((count, 650)) if suffix == "_inputs" else np.arange(count)
            for name, count in native.PANEL_COUNTS.items() for suffix in ("_inputs", "_labels")}
    return data, calls


def test_selected_school_success_does_not_hide_independent_failure(native, monkeypatch, tmp_path):
    data, _ = fake_stage(native, monkeypatch, independent_correct=13)
    result = native.stage(tmp_path, 24, data, stage_protocol(), time.monotonic() + 60)
    assert result["selected_passed"] is True
    assert result["continuation_passed"] is True
    assert result["independent_train"]["correct"] == 13
    assert result["development"]["correct"] == 19
    assert result["independent_passed"] is False
    assert result["passed"] is False
    assert result["status"] == "independent_failed"
    assert result["work"]["teacher_calls"] == 3
    assert result["work"]["query_calls"] == 24 * 2 + 18 + 19


@pytest.mark.parametrize("failure", ("refused", "failed", "unqualified", "cache", "nan",
                                     "arrays", "acceptance", "payload"))
def test_continuation_must_accept_qualify_and_agree(native, monkeypatch, tmp_path, failure):
    continuation = valid_continuation()
    if failure == "refused":
        for lesson in continuation["lessons"]:
            lesson["accepted"] = 0
    elif failure == "failed":
        continuation["lessons"][1]["failed"] = True
    elif failure == "unqualified":
        continuation["lessons"][0]["phases"]["opposite"]["qualified"] = [False]
    elif failure in ("cache", "nan"):
        # JSON receipts cannot contain NaN; test that gate directly first.
        phase = continuation["lessons"][0]["phases"]["nudged"]
        phase["cache_defect"] = [0.01 if failure == "cache" else float("nan")]
        if failure == "nan":
            assert native.continuation_passed(continuation, 0.003) is False
            return
    else:
        continuation[{"arrays": "continued_arrays_equal", "acceptance": "accepted_equal",
                      "payload": "phase_payloads_retained"}[failure]] = False
    data, _ = fake_stage(native, monkeypatch, continuation=continuation)
    result = native.stage(tmp_path, 2, data, stage_protocol(), time.monotonic() + 60)
    assert result["selected_passed"] is True
    assert result["continuation_passed"] is False
    assert result["passed"] is False
    assert result["status"] == "continuation_failed"


def test_positive_native_transfer_retains_independent_and_continuation_gates(
    native, monkeypatch, tmp_path
):
    data, _ = fake_stage(native, monkeypatch)
    result = native.stage(tmp_path, 24, data, stage_protocol(), time.monotonic() + 60)
    assert result["passed"] is True
    assert result["independent_passed"] is True
    assert result["continuation_passed"] is True
    assert result["complete"] is True and result["work_census_complete"] is True
    assert len(result["continuation"]["lessons"]) == 2


def test_corrupt_fixture_rejected_before_attempt_is_written(native, tmp_path):
    fixture, reference, attempt = (tmp_path / name for name in ("fixture", "reference", "attempt"))
    write_fixture(native, fixture)
    write_reference(native, reference)
    provenance = json.loads((fixture / "provenance.json").read_text())
    provenance["panels"]["school"][0]["label"] = 35
    (fixture / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="row identity"):
        native.prepare(attempt, fixture, reference)
    assert not attempt.exists()


def test_failed_reference_continuation_rejected_before_writing(native, tmp_path):
    fixture, reference, attempt = (tmp_path / name for name in ("fixture", "reference", "attempt"))
    write_fixture(native, fixture)
    write_reference(native, reference)
    summary = json.loads((reference / "summary.json").read_text())
    summary["outcomes"][2]["continuation_passed"] = False
    native.harness.write_json(reference / "summary.json", summary)
    with pytest.raises(ValueError, match="five-founder"):
        native.prepare(attempt, fixture, reference)
    assert not attempt.exists()


@pytest.mark.parametrize("drift", ("library", "producer", "fixture", "reference", "protocol",
                                   "runtime", "threads", "model"))
def test_worker_preflight_refuses_drift_before_stage_writes(
    native, capsule, monkeypatch, drift
):
    attempt, protocol = capsule
    if drift in ("library", "producer", "fixture", "reference"):
        path = {
            "library": attempt / "source/library/cadence" / next(iter(protocol["library_sources"])),
            "producer": attempt / "source/continual.py",
            "fixture": attempt / "source/fixture/school.npz",
            "reference": attempt / "reference/summary.json",
        }[drift]
        with path.open("ab") as stream:
            stream.write(b"\nchanged")
    elif drift == "protocol":
        protocol["teaching_seconds"] += 1
        native.harness.write_json(attempt / "protocol.json", protocol)
    elif drift == "runtime":
        monkeypatch.setattr(native, "runtime_precision", lambda: {"numpy": "changed"})
    elif drift == "threads":
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
    else:
        original = native.model_identity
        monkeypatch.setattr(native, "model_identity",
                            lambda model: {**original(model), "changed": 1})
    with pytest.raises(ValueError):
        native.worker(attempt)
    assert not list(attempt.glob("selected-*"))
    assert not (attempt / "summary.json").exists()


def test_worker_exception_preserves_current_stage_and_unknown_work(native, monkeypatch, tmp_path):
    data, _ = fake_stage(native, monkeypatch)
    protocol = stage_protocol()
    native.harness.write_json(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(native, "check_sources", lambda *args: None)
    monkeypatch.setattr(native, "load_fixture", lambda *args: data)

    def interrupted(*args):
        raise RuntimeError("interrupted inside first teaching phase")

    monkeypatch.setattr(native.continual, "graph_lesson", interrupted)
    with pytest.raises(RuntimeError, match="inside first"):
        native.worker(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    current = summary["stages"][0]
    assert summary["stage_denominator"] == 3 and len(summary["stages"]) == 3
    assert current["examples"] == 2 and current["passed"] is None
    assert current["work"]["teacher_calls"] == 0
    assert current["current_operation"] == {"kind": "lesson", "lesson": 1, "school_row": 0}
    assert current["work_census_complete"] is False
    assert summary["process_failure"]["kind"] == "worker_exception"
    assert summary["transfer_passed"] is False and summary["complete"] is False


def test_hard_timeout_preserves_current_census_and_uses_protocol_bound(
    native, monkeypatch, tmp_path
):
    protocol = stage_protocol()
    native.harness.write_json(tmp_path / "protocol.json", protocol)
    summary = native.initial_summary(tmp_path, protocol)
    summary["stages"][0].update(status="selected_passed", passed=True, complete=True)
    summary["stages"][1]["status"] = "running"
    summary["current_stage"] = 4
    native.harness.write_json(tmp_path / "summary.json", summary)
    folder = tmp_path / "selected-4"
    folder.mkdir()
    native.harness.write_json(folder / "state.json", {
        "examples": 4, "status": "running", "passed": False, "complete": False,
        "current_operation": {"kind": "lesson", "lesson": 18},
        "work": {"teacher_calls": 17}, "work_census_complete": False,
    })

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 11
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(native.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        native.launch_worker(tmp_path, protocol)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["process_failure"] == {"kind": "worker_timeout", "seconds": 11}
    assert result["stages"][0]["passed"] is True
    assert result["stages"][1]["passed"] is None
    assert result["stages"][1]["work"]["teacher_calls"] == 17
    assert result["stages"][1]["work_census_complete"] is False
    assert result["stages"][2]["status"] == "not_run_process_failure"
    assert result["transfer_passed"] is False


def test_final_artifact_cap_invalidates_an_otherwise_passing_transfer(
    native, monkeypatch, tmp_path
):
    protocol = stage_protocol()
    protocol["output_cap_mib"] = 1
    native.harness.write_json(tmp_path / "protocol.json", protocol)
    summary = native.initial_summary(tmp_path, protocol)
    summary.update(complete=True, status="transfer_passed")
    for row in summary["stages"]:
        row.update(passed=True, complete=True, selected_passed=True, continuation_passed=True,
                   independent_passed=True)
    monkeypatch.setattr(native.online, "tree_bytes", lambda root: 2 * 1024**2)
    result = native.finalize_summary(tmp_path, summary, protocol)
    assert result["output_cap_exceeded"] is True
    assert result["status"] == "output_cap_exceeded"
    assert result["transfer_passed"] is False


def test_final_summary_cannot_promote_missing_independent_gate(native, tmp_path):
    protocol = stage_protocol()
    native.harness.write_json(tmp_path / "protocol.json", protocol)
    summary = native.initial_summary(tmp_path, protocol)
    summary.update(complete=True, status="transfer_passed")
    for row in summary["stages"]:
        row.update(passed=True, complete=True, selected_passed=True, continuation_passed=True)
    summary["stages"][-1]["independent_passed"] = False
    result = native.finalize_summary(tmp_path, summary, protocol)
    assert result["passed"] is False and result["transfer_passed"] is False
    assert result["status"] == "transfer_failed"


def test_corrupt_current_state_does_not_hide_process_failure(native, tmp_path):
    protocol = stage_protocol()
    native.harness.write_json(tmp_path / "protocol.json", protocol)
    summary = native.initial_summary(tmp_path, protocol)
    summary["stages"][0]["status"] = "running"
    native.harness.write_json(tmp_path / "summary.json", summary)
    folder = tmp_path / "selected-2"
    folder.mkdir()
    (folder / "state.json").write_text('{"incomplete')
    result = native.process_failure_summary(tmp_path, protocol, {"kind": "worker_timeout"})
    assert result["stages"][0]["state_readback_failed"] is True
    assert result["stages"][0]["status"] == "process_interrupted"
    assert result["stages"][0]["work_census_complete"] is False
    assert result["process_failure"]["kind"] == "worker_timeout"
    assert result["passed"] is False
