"""Positive-capacity prerequisites, exact source custody and additional-stage budget."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cadence.receipts import Receipt, canonical_sha256

SPEC = importlib.util.spec_from_file_location(
    "heads_anchored_relevance_tests", Path(__file__).with_name("heads_relevance.py")
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


@pytest.fixture
def prerequisite(tmp_path, monkeypatch):
    declared = subject.anchored_protocol()
    old_digest = declared["prerequisites"]["exposed_receipt_digest"]
    base = json.loads((subject.HERE / "heads-fixed-read-protocol.json").read_text())
    model = tmp_path / "model.py"
    model.write_text("model = 'frozen'\n")
    source = [("src/cadence/model.py", model)]
    source_description = Receipt.build("fixture", {}, source).source
    train_rows = subject.task.panel("train")
    answers = subject.task.targets(train_rows)
    exposed = [{
        "arm": "exposed", "seed": seed, "parameters_preserved": True,
        "training": {"completed": True, "rounds": 512, "wrong_counts": [1] * 512,
                     "work": {"action_refusals": 0, "lesson_refusals": 0}},
        "evaluation": {"train": {"answers": answers.tolist(),
                                  "scores": subject.task.scores(answers, train_rows),
                                  "work": {"action_refusals": 0}}},
    } for seed in (0, 1)]
    old = SimpleNamespace(digest=old_digest, source=source_description,
                          body={"runs": exposed, "protocol": base})
    body = {"passed": True, "masked_historical_reproduced": True,
            "occupied_query_overlap_identical": True, "candidate_rounds_used": 0,
            "confirmation_seeds_used": [],
            "protocol": {"rounds": 512, "batch": 16, "dt": 0.1, "tolerance": 0.003,
                         "free_steps": 2048, "nudged_steps": 2048,
                         "total_read_amplitude": 3.0, "gate": {"trained_floor": 0.95}},
            "reference_receipt_digest": old_digest,
            "previous_capacity_host_seconds": 304.1098535011988,
            "previous_diagnostic_host_seconds": 7.377679918027603,
            "elapsed_host_seconds": 150.0, "cumulative_host_seconds": 461.4875334192264,
            "runs": [{"variant": "anchored", "seed": seed, "empty_anchor": True,
                      "admitted": True} for seed in (0, 1)]}
    base_path = tmp_path / "heads-fixed-read-protocol.json"
    base_path.write_text(json.dumps(base))
    owner = SimpleNamespace(
        PROTOCOL=tmp_path / "empty-boundary-protocol.json", BASE=base_path,
        sources=lambda: source, check=lambda body: None,
        admitted=lambda run, protocol: run["admitted"],
    )
    stage = tmp_path / subject.ANCHORED_PROTOCOL
    stage.write_text(json.dumps(declared))
    monkeypatch.setattr(subject, "empty_owner", lambda: owner)
    monkeypatch.setattr(subject, "anchored_protocol", lambda: declared)
    monkeypatch.setattr(subject.admission, "PROTOCOL", subject.HERE / "heads-protocol.json")
    monkeypatch.setattr(subject.admission, "verify_prior", lambda path: old)
    original_path = subject.admission.protocol_path
    monkeypatch.setattr(subject.admission, "protocol_path", lambda name:
                        stage if name == subject.ANCHORED_PROTOCOL else original_path(name))
    path = tmp_path / "capacity.json"

    def write(changed=None):
        Receipt.build(subject.EMPTY_KIND, body if changed is None else changed, source).write(path)
        return path

    return SimpleNamespace(body=body, old=old, model=model, source=source, owner=owner,
                           protocol=declared, stage=stage, path=path, write=write)


def test_new_capacity_requires_exact_current_sources_and_passed_anchored_founders(prerequisite):
    path = prerequisite.write()
    assert subject.verify_capacity(path).body["passed"]
    assert subject.admission.PROTOCOL == prerequisite.stage
    prerequisite.model.write_text("model = 'changed'\n")
    with pytest.raises(ValueError, match="prerequisite"):
        subject.verify_capacity(path)


@pytest.mark.parametrize("change", ["failed", "historical", "overlap", "missing_founder",
                                    "wrong_flag", "failed_founder", "candidate", "confirmation"])
def test_capacity_failure_or_incomplete_matching_cannot_open_relevance(prerequisite, change):
    body = deepcopy(prerequisite.body)
    if change == "failed":
        body["passed"] = False
    elif change == "historical":
        body["masked_historical_reproduced"] = False
    elif change == "overlap":
        body["occupied_query_overlap_identical"] = False
    elif change == "missing_founder":
        body["runs"].pop()
    elif change == "wrong_flag":
        body["runs"][0]["empty_anchor"] = False
    elif change == "failed_founder":
        body["runs"][0]["admitted"] = False
    elif change == "candidate":
        body["candidate_rounds_used"] = 1
    else:
        body["confirmation_seeds_used"] = [200]
    with pytest.raises(ValueError, match="prerequisite"):
        subject.verify_capacity(prerequisite.write(body))


def test_old_exposed_success_is_recomputed_and_cannot_be_replaced_by_overall_failure(prerequisite):
    path = prerequisite.write()
    prerequisite.old.body["runs"][1]["training"]["work"]["lesson_refusals"] = 1
    with pytest.raises(ValueError, match="exposed acquisition"):
        subject.verify_capacity(path)


def test_old_model_or_task_source_mismatch_blocks_reusing_exposed_evidence(prerequisite):
    path = prerequisite.write()
    prerequisite.old.source["files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="model or task"):
        subject.verify_capacity(path)


def test_capacity_cannot_admit_a_different_sweep_cap_or_acquisition_floor(prerequisite):
    body = deepcopy(prerequisite.body)
    body["protocol"]["free_steps"] = 4096
    with pytest.raises(ValueError, match="recipe differs"):
        subject.verify_capacity(prerequisite.write(body))
    body = deepcopy(prerequisite.body)
    body["protocol"]["gate"]["trained_floor"] = 0.90
    with pytest.raises(ValueError, match="recipe differs"):
        subject.verify_capacity(prerequisite.write(body))


@pytest.mark.parametrize("change", ["linear", "two_rounds", "different_brain", "shared_budget"])
def test_anchored_recipe_has_no_fallback_or_silent_operating_point_change(prerequisite, change):
    path = prerequisite.write()
    if change == "linear":
        prerequisite.protocol["candidate_order"].append("linear")
    elif change == "two_rounds":
        prerequisite.protocol["maximum_candidate_rounds"] = 2
    elif change == "different_brain":
        prerequisite.protocol["brain"]["learning_tolerance"] *= 2
    else:
        prerequisite.protocol["budget"]["additional_stage"] = False
    with pytest.raises(ValueError, match="recipe differs"):
        subject.verify_capacity(path)


def test_exposed_constructor_ignores_only_the_experimental_boundary_flag():
    protocol = subject.anchored_protocol()
    brain = subject.admission.make_brain("exposed", 0, protocol)
    assert brain.recall is None
    assert brain.learner.slot_sizes.tolist() == [4, 4]
    for arm in ("h1", "h2", "h1_wide_ports"):
        candidate = subject.admission.make_brain(arm, 0, protocol)
        assert candidate.empty_anchor and candidate.mode == "agreement"


def test_anchored_protocol_cannot_be_used_as_a_fresh_capacity_campaign(monkeypatch):
    monkeypatch.setattr(subject.admission, "PROTOCOL", subject.HERE / subject.ANCHORED_PROTOCOL)
    with pytest.raises(ValueError, match="heads_relevance"):
        subject.admission.run()


def test_additional_budget_keeps_all_ten_unrun_cells_and_prior_cost(prerequisite, monkeypatch):
    root = prerequisite.path.parent
    (root / "results").mkdir()
    retained = root / "results" / prerequisite.protocol["prerequisites"]["capacity_receipt_file"]
    capacity = Receipt.read(prerequisite.write())
    retained.write_bytes(prerequisite.path.read_bytes())
    monkeypatch.setattr(subject, "HERE", root)
    monkeypatch.setattr(subject.admission, "PROTOCOL", prerequisite.stage)
    monkeypatch.setattr(subject, "verify_capacity", lambda path: capacity)
    monkeypatch.setattr(subject, "anchored_capacity", lambda path: capacity)
    monkeypatch.setattr(subject, "sources", lambda: [])
    # First timer starts this stage; subsequent checks see its entire NEW600s spent.
    clock = iter([0.0])
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(clock, 601.0))
    monkeypatch.setattr(subject, "trial", lambda *a, **k: pytest.fail("expired stage ran a trial"))
    result = subject.run(prerequisite.path).body
    assert len(result["runs"]) == 10
    assert all(row["status"] == "unrun_budget" and not row["passed"] for row in result["runs"])
    assert result["candidate_rounds_used"] == 1 and result["confirmation_seeds_used"] == []
    assert result["budget_scope"] == "additional_relevance_stage"
    assert result["capacity_host_seconds"] == capacity.body["cumulative_host_seconds"]
    assert result["cumulative_host_seconds"] == capacity.body["cumulative_host_seconds"] + 601.0
    assert subject.check(result) is None
    changed = deepcopy(result)
    changed["empty_boundary_host_seconds"] = 0.0
    assert subject.check(changed) == "anchored prerequisite or cumulative budget differs"


def test_candidate_source_union_is_deduplicated_and_binds_retained_capacity(
    prerequisite, monkeypatch,
):
    monkeypatch.setattr(subject.admission, "PROTOCOL", prerequisite.stage)
    monkeypatch.setattr(subject.admission, "sources", lambda: prerequisite.source)
    paths = subject.sources()
    assert len(dict(paths)) == len(paths)
    assert dict(paths)["src/cadence/model.py"] == prerequisite.model
    declared = prerequisite.protocol["prerequisites"]["capacity_receipt_file"]
    assert "benchmarks/attention/results/" + declared in dict(paths)
    assert canonical_sha256(prerequisite.protocol)
