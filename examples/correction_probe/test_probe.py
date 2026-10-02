"""No learning/model-query calls: body boundary, topology, schedules and gates."""

import copy
import gzip
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import body
import probe as P


@pytest.fixture(autouse=True)
def no_model_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No numerical calls in fixture tests")

    for name in ("settle", "step", "observe", "observe_batch", "predict"):
        monkeypatch.setattr(P.Brain, name, forbidden)


def test_actor_inputs_exclude_future_and_hidden_state():
    row = body.data()["train"][0]
    actor = copy.deepcopy(body.inputs(row))
    row["actual"][3] = 0.99
    row["actual"][1 - row["visible"]] = -0.99
    row["before"] = [0.7, 0.7]
    row["after"] = [-0.7, 0.7]
    row["coarse_bias"] = [-9, 9]
    assert body.inputs(row) == actor
    assert "f" not in body.facts(row)


def test_factual_labels_only_available_coordinates():
    for row in body.data()["train"]:
        assert set(body.facts(row)) == {f"p{row['visible']}", "c"}
        assert body.facts(row, reveal=True)["f"] == [row["actual"][3]]
        assert body.facts(row)[f"p{row['visible']}"] == [row["actual"][row["visible"]]]


def test_physical_equations_independently_and_causal_prior():
    rows = body.live_rows(123)
    for i, row in enumerate(rows):
        x, y = [
            0.35 * z + 0.6 * a
            for z, a in zip(row["before"], row["command"], strict=True)
        ]
        assert row["after"] == [x, y]
        assert row["actual"] == [
            math.tanh(x + 0.5 * y),
            math.tanh(-0.5 * x + y),
            math.tanh(x - y + 0.25 * x * y),
            math.tanh(0.7 * x - 0.4 * y + 0.35 * x * y),
        ]
        if i:
            assert row["before"] == rows[i - 1]["after"]
            assert row["issued_persistence"] == body.inputs(rows[i - 1])["fine"]
        assert row["coarse_bias"] == ([0.0, 0.0] if i < 16 else [0.22, -0.18])


def test_observed_pair_is_identifying_not_missing_information():
    # After inverse tanh, determinants of (visible P,C) on |x|,|y|<=.7.
    for x in (-0.7, 0.0, 0.7):
        for y in (-0.7, 0.0, 0.7):
            d0 = -1.5 + 0.25 * x - 0.125 * y
            d1 = -0.5 - 0.125 * x - 0.25 * y
            assert d0 < -1.2 and d1 < -0.23
    # Both are monotone along their straight P-level lines; each visible pair
    # uniquely identifies the bounded state. This is not a supplied actor oracle.


@pytest.mark.parametrize(
    "kind,states,parameters",
    [("ordinary", 12, 153), ("observer", 12, 155), ("wide", 20, 317)],
)
def test_layout_counts_and_raw_information(kind, states, parameters):
    info = P.make(kind, 3109).inspect()
    assert info["patches"] == states
    assert info["patches"] + info["connections"] == parameters
    assert info["input_samples"] == info["sensor_coverage"] == 13
    f = next(o for o in info["outputs"] if o["name"] == "f")
    assert f["sensor_coverage_by_coordinate"] == (13,)
    assert sum(k == "residual" for k, _, _ in info["edges"]) == (
        2 if kind == "observer" else 0
    )


def test_schedules_have_matched_declared_exposure():
    for seed in P.DESIGN["preflight_seeds"] + P.DESIGN["screen_seeds"]:
        schedule = P.schedule(seed)
        assert len(schedule) == 128 and all(len(row) == 16 for row in schedule)
        for start in range(0, 128, 32):
            assert sorted(
                i for row in schedule[start : start + 32] for i in row
            ) == list(range(512))


def test_source_pin_exact_public_math():
    assert dict(P.IMPLEMENTATION) == P.EXPECTED_CORE


def test_prepare_and_mutated_founder_reject(tmp_path):
    root = tmp_path / "run"
    P.prepare(root, "preflight")
    assert len(P.bound(root)["inventory"]) == 3
    path = root / "founders.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="data/founder"):
        P.bound(root)


def test_incomplete_case_cannot_pass(tmp_path):
    root = tmp_path / "run"
    P.prepare(root, "preflight")
    p = P.bound(root)
    outcomes = [
        {"name": j["name"], "status": "not_started", "returncode": None}
        for j in p["inventory"]
    ]
    result = P.summary(root, outcomes)
    assert not result["complete"] and not result["all_acquired"]
    assert not result["development_quality_gate"]
    with pytest.raises(ValueError, match="census"):
        P.summary(root, outcomes[:-1])


def test_no_overwrite(tmp_path):
    P.prepare(tmp_path / "run", "preflight")
    with pytest.raises(ValueError, match="Preserve"):
        P.prepare(tmp_path / "run", "preflight")


def test_every_future_record_has_unique_identity_and_finite_values():
    tape = body.data()
    ids = [r["id"] for rows in tape.values() for r in rows]
    assert len(ids) == len(set(ids))
    for rows in tape.values():
        for row in rows:
            assert all(math.isfinite(x) and abs(x) < 1 for x in row["actual"])
    assert len(json.dumps(tape)) < 1_000_000


def test_interrupted_intent_does_not_claim_zero_unknown_work(tmp_path):
    P.write(tmp_path / "inflight.json", {"status": "started", "ordinal": 3})
    with gzip.open(tmp_path / "calls.jsonl.gz", "wt") as stream:
        stream.write(json.dumps({"ordinal": 2, "status": "returned"}) + "\n")
    assert P.pending_unknown(tmp_path)
    with gzip.open(tmp_path / "calls.jsonl.gz", "at") as stream:
        stream.write(json.dumps({"ordinal": 3, "status": "returned"}) + "\n")
    assert not P.pending_unknown(tmp_path)
