"""No numerical calls: information, graph change, baseline custody and gates."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_path as H
import probe as P


@pytest.fixture(autouse=True)
def no_numerical_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No model queries or admissions in head-path tests")

    for name in ("settle", "step", "observe", "observe_batch", "predict"):
        monkeypatch.setattr(P.Brain, name, forbidden)


@pytest.mark.parametrize(
    "kind,states,params",
    [("ordinary", 12, 140), ("observer", 12, 142), ("wide", 20, 304)],
)
def test_only_f_raw_edges_removed_with_all_information_preserved(kind, states, params):
    old = P.make(kind, 3109).inspect()
    new = H.make("head_" + kind, 3109).inspect()
    assert new["patches"] == states
    assert new["patches"] + new["connections"] == params
    assert new["input_samples"] == new["sensor_coverage"] == 13
    assert next(o for o in new["outputs"] if o["name"] == "f")[
        "sensor_coverage_by_coordinate"
    ] == (13,)
    old_edges, new_edges = set(old["edges"]), set(new["edges"])
    removed = old_edges - new_edges
    assert not new_edges - old_edges
    assert removed == {("input", i, states - 1) for i in range(13)}
    assert all(kind == "state" for kind, _, target in new_edges if target == states - 1)
    g = next(p for p in new["populations"] if p["name"] == "nonlinear_head")
    for target in g["indices"]:
        assert {
            source
            for kind, source, dest in new_edges
            if kind == "input" and dest == target
        } == set(range(13))


@pytest.mark.parametrize("kind", ["ordinary", "observer"])
def test_original_control_founder_exact(kind):
    assert H.make("raw_" + kind, 3109).snapshot() == P.make(kind, 3109).snapshot()


def test_settings_dose_and_gates_unchanged():
    for key in (
        "config",
        "updates",
        "batch",
        "checks",
        "live_steps",
        "factual_fit_after_steps",
        "factual_fit_rows",
        "gates",
    ):
        assert H.DESIGN[key] == P.DESIGN[key]
    assert H.DESIGN["arm_seconds"] == 120 and P.DESIGN["arm_seconds"] == 60
    assert H.DESIGN["seeds"] == [3109]
    assert len(H.KINDS) == 5
    assert dict(P.IMPLEMENTATION) == P.EXPECTED_CORE


def fixture(tmp_path, monkeypatch):
    inventory = [
        {"kind": kind, "seed": 3109, "name": kind + "-3109"} for kind in H.KINDS
    ]
    monkeypatch.setattr(H, "bound", lambda _: {"inventory": inventory})
    outcomes = [
        {"name": j["name"], "status": "returned", "returncode": 0} for j in inventory
    ]
    for job in inventory:
        directory = tmp_path / job["name"]
        directory.mkdir()
        H.write(
            directory / "report.json",
            {
                "job": job,
                "status": "complete",
                "source_unchanged": True,
                "same_learning_twins": True,
                "admissions": 132,
                "counts": {"settle": 1296, "observe_batch": 132},
                "unknown_calls": 0,
                "acquired": True,
                "raw_acquisition_exact_parent": True,
                "branches": {
                    "corrected": {
                        "shift_mae": 0.01 if job["kind"] == "head_observer" else 0.03
                    },
                    "routine": {"shift_mae": 0.03},
                },
                "work": {"edge_visits": 123},
            },
        )
    return outcomes


def test_all_five_required_and_partial_veto(tmp_path, monkeypatch):
    outcomes = fixture(tmp_path, monkeypatch)
    assert H.summary(tmp_path, outcomes)["development_quality_gate"]
    bad = copy.deepcopy(outcomes)
    bad[-1]["status"] = "outer_timeout"
    result = H.summary(tmp_path, bad)
    assert not result["complete"] and not result["development_quality_gate"]
    with pytest.raises(ValueError, match="census"):
        H.summary(tmp_path, outcomes[:-1])


def test_own_correction_and_raw_capable_control_not_optional(tmp_path, monkeypatch):
    import json

    outcomes = fixture(tmp_path, monkeypatch)
    path = tmp_path / "raw_ordinary-3109/report.json"
    report = json.loads(path.read_text())
    report["branches"]["corrected"]["shift_mae"] = 0.009
    H.write(path, report)
    assert not H.summary(tmp_path, outcomes)["development_quality_gate"]
    report["branches"]["corrected"]["shift_mae"] = 0.03
    report["raw_acquisition_exact_parent"] = False
    H.write(path, report)
    assert not H.summary(tmp_path, outcomes)["complete"]


def test_acquisition_failure_cannot_promote(tmp_path, monkeypatch):
    import json

    outcomes = fixture(tmp_path, monkeypatch)
    path = tmp_path / "head_wide-3109/report.json"
    report = json.loads(path.read_text())
    report["acquired"] = False
    H.write(path, report)
    result = H.summary(tmp_path, outcomes)
    assert result["complete"] and not result["all_acquired"]
    assert not result["development_quality_gate"]
