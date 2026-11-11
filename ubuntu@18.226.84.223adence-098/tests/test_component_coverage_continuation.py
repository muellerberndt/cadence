"""Coverage/custody/admission contracts; no candidate solves or learning runs."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def worker(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks/retention"
    for name in (
        "chamber_inputs",
        "actual_outcome_chamber",
        "component_coverage_inputs",
        "component_coverage_continuation",
    ):
        spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["component_coverage_continuation"]


def reading(correct=9):
    return {
        "qualified_free": True,
        "refused": 0,
        "unrun": 0,
        "planned": 33,
        "attempted": 33,
        "world": 0,
        "cues": [
            {
                "cue": cue,
                "correct": correct,
                "prototype_correct": 1,
                "prototype_prediction": cue % 2,
                "predictions": [cue % 2] * 10,
                "margins": [0.5] * 10,
                "memory_values": [[1, 0]] * 10,
                "refused": 0,
                "unrun": 0,
                "obsolete": 10 - correct if cue < 2 else 0,
            }
            for cue in range(3)
        ],
    }


def complete_row(worker):
    row = copy.deepcopy(worker.declared_lives()[1])
    endpoint = {key: reading() for key in ("intact", "graph_only", "initial_graph", "joint_reset")}
    endpoint["uniform"] = {"correct": 16}
    row.update(
        complete=True,
        actual_batches=768,
        executed_batches=768,
        baseline_reproduced=True,
        starting_full_array_parity=True,
        endpoints={name: copy.deepcopy(endpoint) for name in ("initial", *worker.inputs.PHASES)},
        curves={
            phase: {str(n): reading() for n in (0, 32, 64, 128, 256)}
            for phase in worker.inputs.PHASES
        },
        seams=[{"phase": phase, "batch": batch, "passed": True} for phase, batch in worker.SEAMS],
    )
    return row


def test_actual_partial_training_covers_both_directions_without_reusing_held_probes(worker):
    inputs = worker.inputs
    arrays = inputs.frozen_arrays()
    reserved = worker.chamber.inputs.frozen_arrays(0)
    audit = inputs.coverage_audit(arrays, reserved)
    for phase in inputs.PHASES:
        for condition in worker.chamber.inputs.CONDITIONS:
            coverage = audit["phases"][phase]["coverage"][condition]["per_cue_component_counts"]
            assert [(r["full"], r["delete0"], r["delete1"]) for r in coverage] == [
                (448, 224, 224),
                (448, 224, 224),
                (64, 32, 32),
                (128, 0, 0),
            ]
            ids = arrays[phase + "/identities"]
            full = arrays[f"{condition}/{phase}/full-only"]
            candidate = arrays[f"{condition}/{phase}/component-coverage"]
            np.testing.assert_array_equal(full, worker.chamber.inputs.prototypes(condition)[ids])
            assert not np.array_equal(candidate, full)
            np.testing.assert_array_equal(candidate[ids == 3], full[ids == 3])
    # Input preparation does not silently include one of the reserved deletion amplitudes.
    assert audit["new_solves"] == audit["new_outcomes"] == audit["new_learning"] == 0


def test_corrupt_or_changed_input_fixture_refused(worker, tmp_path):
    arrays = worker.inputs.frozen_arrays()
    path = tmp_path / "inputs.npz"
    np.savez_compressed(path, **arrays)
    loaded = worker.inputs.load(path)
    assert not loaded["continued/identities"].flags.writeable
    arrays["orthogonal/continued/component-coverage"][0, 0, 0] += 0.01
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="schedule differs"):
        worker.inputs.load(path)


def test_actual_next_cue_crosses_world_boundary_without_a_fabricated_intermediate(worker):
    arrays = worker.inputs.frozen_arrays()
    for arm in worker.inputs.ARMS:
        got = worker.next_observations(arrays, "continued", "orthogonal", arm, 255)
        np.testing.assert_array_equal(got, arrays[f"orthogonal/revision/{arm}"][0])
        terminal = worker.next_observations(arrays, "restored", "orthogonal", arm, 255)
        np.testing.assert_array_equal(
            terminal, np.repeat(worker.chamber.inputs.prototypes("orthogonal")[[3]], 8, axis=0)
        )


@pytest.mark.parametrize(
    "defect",
    [
        "partial",
        "prototype",
        "missing_curve",
        "missing_control",
        "refused_curve",
        "seam",
        "baseline",
    ],
)
def test_no_false_pass_from_relaxed_or_missing_evidence(worker, defect):
    row = complete_row(worker)
    assert worker.row_gate(row)["passed"]
    if defect == "partial":
        row["endpoints"]["continued"]["intact"]["cues"][0]["correct"] = 8
    elif defect == "prototype":
        row["endpoints"]["restored"]["intact"]["cues"][1]["prototype_correct"] = 0
    elif defect == "missing_curve":
        del row["curves"]["revision"]["64"]
    elif defect == "missing_control":
        del row["endpoints"]["revision"]["joint_reset"]
    elif defect == "refused_curve":
        row["curves"]["continued"]["32"]["qualified_free"] = False
    elif defect == "seam":
        row["seams"][0]["batch"] = 64
    else:
        row["baseline_reproduced"] = False
    assert not worker.row_gate(row)["passed"]


def test_incomplete_cohort_keeps_all_paired_endpoint_denominators(worker):
    rows = worker.declared_lives()
    gate = worker.cohort(rows, "component-coverage")
    assert gate["planned"] == 360 and not gate["complete"] and not gate["passed"]
    assert len(rows) == 8 and sum(row["planned_batches"] for row in rows) == 6144
    assert [(r["condition"], r["exposure"]) for r in rows[::2]] == list(worker.inputs.SPECIMENS)


def test_baseline_requires_exact_predictions_and_memory_values(worker):
    expected = reading()
    assert worker.same_baseline(reading(), expected)
    changed = reading()
    changed["cues"][0]["predictions"][3] = 1
    assert not worker.same_baseline(changed, expected)
    changed = reading()
    changed["cues"][0]["memory_values"][3] = [0.9, 0]
    assert not worker.same_baseline(changed, expected)


@pytest.mark.parametrize(
    "accepted,branch,expected", [(True, False, 1), (False, False, 0), (True, True, 0)]
)
def test_completed_actual_feedback_count_survives_post_call_cap(
    worker, tmp_path, accepted, branch, expected
):
    # Assigned journal records test accounting only; no fabricated Brain moment or action.
    (tmp_path / "protocol.json").write_text("{}\n")
    journal = worker.Journal(tmp_path, worker.fixed_fields(), time.monotonic())
    row = journal.summary["lives"][0]
    row["actual_world"] = {
        "continued": {
            "events": 0,
            "reward_sum": 0.0,
            "non_neutral": 0,
            "correct_non_neutral": 0,
            "actions_per_cue": [[0, 0] for _ in range(4)],
        }
    }
    identities = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    actions = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    reward = worker.chamber.inputs.actual_reward(identities, actions, 0)
    context = {"life": 0, "main_life": True, "arm": "full-only", "phase": "continued"}
    if branch:
        context["branch"] = "reused-witness"

    def cap(**_):
        raise worker.chamber.ResourceLimit("post-call cap")

    journal.bounds = cap
    with pytest.raises(worker.chamber.ResourceLimit, match="post-call cap"):
        journal.completed(
            "actual_feedback",
            context,
            {
                "accepted": accepted,
                "witness": {"identities": identities, "actions": actions, "reward": reward},
            },
            {"actual_feedback_calls": 1},
        )
    assert row["actual_batches"] == row["phases"]["continued"]["actual_batches"] == expected
    assert row["actual_world"]["continued"]["events"] == 0  # Feedback does not execute the world.
    assert journal.totals["actual_feedback_calls"] == 1
    assert journal.summary["unknown_current_work"] is False


def test_world_outcome_survives_post_guard_without_claiming_accepted_feedback(worker, tmp_path):
    # A journal-only accounting fixture, never a fabricated Brain moment/action.
    (tmp_path / "protocol.json").write_text("{}\n")
    journal = worker.Journal(tmp_path, worker.fixed_fields(), time.monotonic())
    row = journal.summary["lives"][0]
    row["actual_world"] = {
        "continued": {
            "events": 0,
            "reward_sum": 0.0,
            "non_neutral": 0,
            "correct_non_neutral": 0,
            "actions_per_cue": [[0, 0] for _ in range(4)],
        }
    }
    ids, actions = np.array([0, 1, 2, 3] * 2), np.array([0, 1, 0, 1, 0, 1, 1, 0])
    witness = {
        "identities": ids,
        "actions": actions,
        "reward": worker.chamber.inputs.actual_reward(ids, actions, 0),
    }
    context = {"life": 0, "main_life": True, "arm": "full-only", "phase": "continued", "batch": 1}

    def cap(**_):
        raise worker.chamber.ResourceLimit("post-world-event cap")

    journal.bounds = cap
    with pytest.raises(worker.chamber.ResourceLimit):
        journal.completed(
            "actual_world_event",
            context,
            {"accepted": True, "witness": witness},
            {"actual_world_outcome_rows": 8},
        )
    assert row["executed_batches"] == 1 and row["actual_batches"] == 0
    assert row["actual_world"]["continued"]["events"] == 8
    np.testing.assert_array_equal(row["pending_outcome"]["witness"]["reward"], witness["reward"])
    assert journal.totals["actual_world_outcome_rows"] == 8


def test_receipt_flush_crossing_time_cap_fails_and_rebinds_summary(worker, tmp_path, monkeypatch):
    clock = [worker.SECONDS - 0.5]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    original_write = worker.chamber.atomic_json
    original_write(
        tmp_path / "summary.json", {"passed": True, "complete": True, "status": "passed"}
    )
    before = worker.chamber.digest(tmp_path / "summary.json")

    def slow_write(path, value):
        original_write(path, value)
        clock[0] += 1

    monkeypatch.setattr(worker.chamber, "atomic_json", slow_write)
    result = worker.close_execution_receipt(tmp_path, 0, 0, {"protocol_sha256": "test"})
    assert result["passed"] is False and result["complete"] is True
    receipt = json.loads((tmp_path / "execution.json").read_text())
    assert receipt["final_resources"]["qualified"] is False
    assert receipt["summary_sha256"] != before
    assert receipt["summary_sha256"] == worker.chamber.digest(tmp_path / "summary.json")


def test_worker_summary_flush_is_part_of_qualified_wall_bound(worker, tmp_path, monkeypatch):
    # Pure result/census fixture, no brain, query, outcome or learning.
    clock = [worker.SECONDS - 0.5]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    rows = []
    for declared in worker.declared_lives():
        row = complete_row(worker)
        row.update({key: declared[key] for key in ("specimen", "condition", "exposure", "arm")})
        for phase in worker.inputs.PHASES:
            for cue in row["endpoints"][phase]["joint_reset"]["cues"]:
                cue["correct"] = 0
        row.update(passed=True, gate=worker.row_gate(row))
        rows.append(row)
    journal = SimpleNamespace(
        root=tmp_path,
        began=0,
        summary={"lives": rows, "process_failure": None, "unknown_current_work": False},
    )
    writes = []

    def persist(**_):
        writes.append(journal.summary["passed"])
        worker.chamber.atomic_json(tmp_path / "summary.json", journal.summary)
        clock[0] += 1

    journal.persist = persist
    result = worker.finish(journal)
    assert writes == [True, False]
    assert result["complete"] and not result["passed"]
    assert result["final_flush_resource_qualified"] is False


def test_corrupt_reference_rejected_before_models_or_execution_marker(
    worker, tmp_path, monkeypatch
):
    helper = tmp_path / "source/component_coverage_continuation.py"
    helper.parent.mkdir()
    helper.write_bytes(Path(worker.__file__).read_bytes())
    bad = tmp_path / "reference/bad.npz"
    bad.parent.mkdir()
    bad.write_bytes(b"corrupt")
    protocol = {
        **worker.fixed_fields(),
        "runtime": {},
        "interpreter_sha256": worker.chamber.digest(Path(sys.executable)),
        "source_files": {
            "source/component_coverage_continuation.py": worker.chamber.digest(helper)
        },
        "reference_files": {"reference/bad.npz": "wrong"},
        "admitted_artifacts": {},
    }
    worker.chamber.atomic_json(tmp_path / "protocol.json", protocol)
    pin = worker.chamber.digest(tmp_path / "protocol.json")
    worker.chamber.atomic_json(
        tmp_path / "admission.json",
        {
            "protocol_sha256": pin,
            "preparation_seconds": 0,
            "final_admission_io_allowance_seconds": 1,
        },
    )
    worker.chamber.atomic_json(
        tmp_path / "job-protocol.json",
        {"protocol_sha256": pin, "helper_sha256": worker.chamber.digest(helper)},
    )
    monkeypatch.setattr(worker.chamber, "runtime", lambda: {})
    monkeypatch.setattr(
        worker.chamber.Brain, "load", lambda *_: pytest.fail("model load before guard")
    )
    with pytest.raises(ValueError, match="frozen source/reference/input changed"):
        worker.launch(tmp_path)
    assert not (tmp_path / "execution-started.json").exists()
