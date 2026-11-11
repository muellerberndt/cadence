"""The competitor arms of the odour nursery: declared outside the frozen plan, read like the
other arms, undisturbed by their probes, and carried by verifiable receipts."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("nursery_competitors", HERE / "odour_nursery.py")
nursery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nursery)
SIZES = {"elman": 387, "gru": 1059}  # 4 odours, 2 actions, 16 hidden


@pytest.fixture(scope="module")
def protocol():
    return json.loads((HERE / "protocol.json").read_text())


@pytest.fixture(scope="module")
def short(protocol):
    return {**protocol, "after": 300}


def test_the_competitor_arms_are_declared_outside_the_frozen_plan(protocol):
    assert nursery.COMPETITORS == ("elman", "gru")
    assert not set(nursery.COMPETITORS) & set(nursery.ARMS) and "competitors" not in protocol
    for arm in nursery.COMPETITORS:
        settings = nursery.COMPETITOR_SETTINGS[arm]
        nursery.competitors.check_settings(settings)
        assert settings["gamma"] == 0.0 and settings["lam"] == 0.0  # no bootstrap across trials
        assert nursery.competitor_parameters(arm, settings["hidden"]) == SIZES[arm]
        life = nursery.make_life(arm, protocol, 0, protocol["arousal"])
        assert life.learner.parameter_count == SIZES[arm] and life.learner.cell.name == arm


@pytest.mark.parametrize("arm", nursery.COMPETITORS)
def test_a_competitor_life_reads_like_the_other_arms(short, arm):
    row = nursery.run_life(arm, 0, 300, short)
    assert "error" not in row and len(row["phases"]) == 3
    assert row["learner"]["parameters"] == row["work"]["parameters"] == SIZES[arm]
    for phase in row["phases"]:
        assert sum(phase["visits"]) == 300 and all(c in (0, 1) for c in phase["end_greedy"])
        assert phase["behaviour_approach"] == phase["policy_approach"]  # it samples its policy
        assert 0.0 < phase["executed_probability"] < 1.0
        assert all(0.0 <= v <= 1.0 for v in phase["start_approach"] + phase["end_approach"])
        assert phase["aroused"] == 1.0  # it learns at every trial
    work = row["work"]
    assert work["aroused"] == 901 and work["routine"] == 0 and work["updates"] == 900
    assert work["forward_passes"] == 2 * 901 - 1  # one answer, then a bootstrap and an answer
    assert 36 <= work["probes"] <= 36 + 2 * short["witness_probes"]
    assert work["probe_forward_passes"] == 4 * work["probes"]
    assert work["brains"] == work["checkpoints"] == work["memory_reads"] == 0
    assert work["hidden"] == 16 and work["multiply_accumulates"] > 0
    assert len(work["latency_ms"]["aroused"]) == 3 and work["latency_ms"]["routine"] is None
    body = {"protocol": short, "arms": [arm], "exposures": [300], "seeds": [0], "rows": [row]}
    assert nursery.validate_readings(body) is None


def test_the_probes_do_not_disturb_a_competitor_life(short):
    quick = {**short, "after": 100}
    blind = {**quick, "probe_every": 50, "witness_probes": 0}
    executed = ("final", "whole", "lag", "first_try", "visits", "approaches", "aroused")
    measurement = ("probes", "probe_forward_passes", "latency_ms")
    for arm in nursery.COMPETITORS:
        probed, unprobed = (nursery.run_life(arm, 5, 100, p) for p in (quick, blind))
        for a, b in zip(probed["phases"], unprobed["phases"], strict=True):
            assert [a[key] for key in executed] == [b[key] for key in executed]
            assert a["behaviour_approach"] == b["behaviour_approach"]
        lived = {k: v for k, v in probed["work"].items() if k not in measurement}
        assert lived == {k: v for k, v in unprobed["work"].items() if k not in measurement}
        assert probed["work"]["probes"] > unprobed["work"]["probes"]


def test_a_receipt_with_competitor_arms_verifies_and_leaves_the_other_arms_as_they_are(
    tmp_path, protocol, capsys
):
    path = tmp_path / "receipt.json.gz"
    run = ["--arms", "elman", "gru", "tabular", "--seeds", "0", "--exposures", "100"]
    run += ["--workers", "1", "--out", str(path)]
    assert nursery.main(run) == 0
    agree = "canonical form, digest, sources, arithmetic agree"
    assert nursery.verify(path, current=True) == (True, agree)
    assert nursery.main(["--report", str(path)]) == 0
    stored = json.loads(gzip.decompress(path.read_bytes()))
    body = stored["body"]
    assert body["frozen_protocol"] and body["arms"] == ["elman", "gru", "tabular"]
    assert body["competitors"] == {a: nursery.COMPETITOR_SETTINGS[a] for a in ("elman", "gru")}
    files = [item["path"] for item in stored["source"]["files"]]
    assert files[:2] == ["odour_nursery.py", "competitors.py"]
    for row in body["rows"]:
        assert ("learner" in row) == (row["arm"] in nursery.COMPETITORS)
    tabular = next(r for r in body["rows"] if r["arm"] == "tabular")
    assert tabular["phases"] == nursery.run_life("tabular", 0, 100, protocol)["phases"]

    def rewrite(content):
        path.write_bytes(gzip.compress((nursery.canonical_json(content) + "\n").encode()))

    tampering = (
        ({"parameters": 1}, "a competitor's parameter count differs from its declared size"),
        ({"updates": -1}, "a competitor's counters must be nonnegative integers"),
    )
    for change, reason in tampering:
        edited = json.loads(json.dumps(body))
        edited["rows"][0]["work"].update(change)
        rewrite(nursery.Receipt.build(nursery.SCHEMA, edited, nursery.sources()).to_dict())
        assert nursery.verify(path) == (False, reason)
    capsys.readouterr()
