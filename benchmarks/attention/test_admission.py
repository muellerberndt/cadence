"""Retained failures remain failures; their source and decision custody is checked."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cadence.receipts import Receipt

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("attention_verify", HERE / "verify.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


@pytest.mark.parametrize("budget", [256, 1024])
def test_retained_admission_source_arithmetic_and_stop_decision(budget: int) -> None:
    path = HERE / "results" / f"admission-{budget}.json"
    ok, message = verifier.verify(path)
    assert ok, message
    body = Receipt.read(path).body
    assert not body["passed"] and not any(f["passed"] for f in body["founders"])
    assert body["candidate_rounds_used"] == 0 and not body["confirmation_seeds_used"]


def test_recomputed_gates_reject_optimistic_flags_and_altered_answers() -> None:
    body = Receipt.read(HERE / "results" / "admission-256.json").body
    changed = deepcopy(body)
    changed["passed"] = True
    assert verifier.check(changed) == "aggregate gate differs"
    changed = deepcopy(body)
    changed["founders"][1]["passed"] = True  # 100% answers after an incomplete training run
    assert verifier.check(changed) == "founder gate differs"
    changed = deepcopy(body)
    changed["founders"][0]["evaluation"]["branches"]["intact"]["answers"][0][0] ^= 1
    assert verifier.check(changed) == "behavioral scores differ"


def test_damping_readback_source_custody_and_qualification_flags() -> None:
    path = HERE / "results" / "damping-readback.json"
    receipt = Receipt.read(path)
    sources = verifier._resolved_sources(receipt)
    ok, message = Receipt.verify(path, sources=sources)
    assert ok, message
    for founder in receipt.body["founders"]:
        qualified = max(founder["damped_residual"]) <= founder["tolerance"]
        assert founder["damped_qualified"] == qualified
        assert not qualified
        assert founder["damped_steps"] <= 1024
    assert not receipt.body["all_damped_boundaries_qualified"]


@pytest.mark.parametrize("retained", [None, 0])
def test_historical_numerical_records_accept_only_the_additive_zero(retained, historical) -> None:
    admission = historical
    body = Receipt.read(HERE / "results" / "founder-capacity-development.json").body
    result = deepcopy(body["runs"][0])
    assert "initial_damping" not in result["learning_config"]
    assert "learning_initial_damping" not in result["resolved"]["genes"]
    if retained is not None:
        result["learning_config"]["initial_damping"] = retained
        result["resolved"]["genes"]["learning_initial_damping"] = retained
    assert admission.numerical_check(result, body["protocol"], historical=True) is None
    if retained is None:
        assert admission.numerical_check(result, body["protocol"]) is not None


@pytest.mark.parametrize("section,key", [
    ("learning_config", "initial_damping"), ("genes", "learning_initial_damping")])
@pytest.mark.parametrize("value", [1, False, 0.0, None])
def test_historical_numerical_records_reject_nonzero_or_invalid_policy(
    section, key, value, historical,
):
    admission = historical
    body = Receipt.read(HERE / "results" / "founder-capacity-development.json").body
    result = deepcopy(body["runs"][0])
    record = result["resolved"]["genes"] if section == "genes" else result[section]
    record[key] = value
    assert admission.numerical_check(result, body["protocol"], historical=True) is not None


def test_historical_numerical_records_reject_other_missing_or_new_fields(historical) -> None:
    admission = historical
    body = Receipt.read(HERE / "results" / "founder-capacity-development.json").body
    for section in ("learning_config", "genes"):
        # a real gene, not one of the constants of 0.93.0 the historical reader leaves out
        for mutate in (lambda record: record.pop("actor_eta" if "actor_eta" in record else "eta"),
                       lambda record: record.update(unknown_numerical_policy=0)):
            result = deepcopy(body["runs"][0])
            record = result["resolved"]["genes"] if section == "genes" else result[section]
            mutate(record)
            assert admission.numerical_check(result, body["protocol"], historical=True) is not None


@pytest.fixture(scope="module")
def historical():
    spec = importlib.util.spec_from_file_location(
        "historical_heads_admission", HERE / "heads_admission.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("damage", ["altered", "missing"])
def test_historical_source_resolution_requires_actual_bound_bytes(tmp_path, monkeypatch, damage):
    root, attention = tmp_path / "project", tmp_path / "attention"
    root.mkdir()
    current = root / "source.py"
    current.write_text("original = True\n")
    receipt = Receipt.build("historical-source-test/1", {}, [("source.py", current)])
    digest = receipt.source["files"][0]["sha256"]
    snapshot = attention / "results" / "sources" / (digest + ".txt")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(current.read_bytes())
    current.write_text("original = False\n")
    monkeypatch.setattr(verifier, "HERE", attention)
    monkeypatch.setattr(verifier, "ROOT", root)
    assert verifier._resolved_sources(receipt) == [("source.py", snapshot)]
    path = tmp_path / "receipt.json"
    receipt.write(path)
    assert Receipt.verify(path, sources=verifier._resolved_sources(receipt))[0]
    if damage == "altered":
        snapshot.write_text("changed archive\n")
    else:
        snapshot.unlink()
        current.unlink()
    with pytest.raises(ValueError, match="historical source"):
        verifier._resolved_sources(receipt)
    assert json.loads(path.read_text())["digest"] == receipt.digest
