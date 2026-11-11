"""Matched data coverage changes witnesses without changing teacher exposure."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent


@pytest.fixture
def coverage(monkeypatch):
    for name in (
        "extract",
        "run",
        "relations",
        "relation_development",
        "continual",
        "online_curriculum",
        "native_online",
        "single_module_native",
        "single_module_native_24",
        "native_coverage_control",
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["native_coverage_control"]


def panel_fixture(coverage, tmp_path):
    folder = tmp_path / "source/fixture"
    folder.mkdir(parents=True)
    data, panels = {}, {}
    for name, count in (("school", 24), ("independent_train", 18)):
        x = np.zeros((count, 650), dtype=np.float64)
        x[:, 0] = np.arange(count)
        x[:, 1] = 0.1 if name == "school" else 0.2
        y = np.arange(count, dtype=np.int64)
        data[name + "_inputs"], data[name + "_labels"] = x, y
        panels[name] = [
            {
                "frame": i * 12,
                "input_sha256": coverage.native.harness.row_hash(row),
                "label": int(label),
            }
            for i, (row, label) in enumerate(zip(x, y, strict=True))
        ]
    data["development_inputs"] = np.array([np.nan])
    data["heldout_inputs"] = np.array([np.nan])
    np.savez(folder / "school.npz", **data)
    (folder / "provenance.json").write_text(json.dumps({"panels": panels}))
    return list(range(24)) * 43


def test_label_sequence_total_and_action_frequencies_are_identical(coverage, tmp_path):
    order = panel_fixture(coverage, tmp_path)
    schedules = coverage.schedules(tmp_path, order)
    a, b = (schedules[name] for name in coverage.ARMS)
    assert len(a) == len(b) == 1009
    assert [r["label"] for r in a] == [r["label"] for r in b]
    assert [r["school_row"] for r in a] == [r["school_row"] for r in b]
    assert Counter(r["label"] for r in a[:1008]) == Counter(r["label"] for r in b[:1008])
    assert set(Counter(r["label"] for r in a[:1008]).values()) == {42}
    assert all(r["panel"] == "school" for r in a)
    counts = Counter((r["label"], r["panel"]) for r in b[:1008])
    assert all(
        counts[label, panel] == 21
        for label in range(18)
        for panel in ("school", "independent_train")
    )
    assert all(counts[label, "school"] == 42 for label in range(18, 24))
    assert all(r["panel"] != "development" for r in b)
    assert all(a[i] == b[i] for i in range(24))


@pytest.mark.parametrize(
    "change",
    [
        {"teacher_presentations_per_arm": 1024},
        {"presentations_per_action": 43},
        {"whole_issue_passed": True},
        {"founder_seeds": {"selected-only": False, "expanded-train": 0}},
        {"role_change": "independent TRAIN remains untouched"},
    ],
)
def test_coverage_guard_rejects_exposure_or_acceptance_promotion(coverage, tmp_path, change):
    protocol = deepcopy(coverage.FIXED)
    protocol.update(change)
    with pytest.raises(ValueError, match="gene/gate/exposure/bound"):
        coverage.check_sources(tmp_path, protocol)


def test_timeout_recovers_pending_operation_without_claiming_comparison(coverage, tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    folder = tmp_path / "expanded-train"
    folder.mkdir()
    state = {
        "arm": "expanded-train",
        "complete": False,
        "status": "running",
        "completed_lessons": 5,
        "current_operation": {"kind": "lesson"},
        "work_census_complete": False,
    }
    (folder / "state.json").write_text(json.dumps(state))
    coverage.process_failure(tmp_path, {"output_cap_mib": 1}, {"kind": "timeout"})
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["arms"][1] == state
    assert not result["comparison_valid"] and not result["whole_issue_passed"]
    assert not result["complete"]
    assert result["process_failure"]["kind"] == "timeout"


def test_output_violation_is_preserved_and_cannot_count_as_positive(coverage, tmp_path):
    (tmp_path / "retained.bin").write_bytes(b"a" * 1024)
    summary = {"comparison_valid": True, "positive_coverage_effect": True}
    coverage.finish(tmp_path, summary, {"output_cap_mib": 0.0001})
    actual = json.loads((tmp_path / "summary.json").read_text())
    assert actual["output_cap_exceeded"]
    assert not actual["comparison_valid"] and not actual["positive_coverage_effect"]
    assert (tmp_path / "retained.bin").read_bytes() == b"a" * 1024


def test_admission_thread_settings_restore_after_failure(coverage, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    protocol = {"threads": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}
    with pytest.raises(RuntimeError), coverage.admission_threads(protocol):
        assert coverage.os.environ["OMP_NUM_THREADS"] == "1"
        assert coverage.os.environ["MKL_NUM_THREADS"] == "1"
        raise RuntimeError("failed preflight")
    assert coverage.os.environ["OMP_NUM_THREADS"] == "7"
    assert "MKL_NUM_THREADS" not in coverage.os.environ
