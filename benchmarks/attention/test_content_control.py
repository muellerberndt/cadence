"""Supplied content is unselected, causally editable, and isolated from trace/warm state."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "attention_content_subject", Path(__file__).with_name("content_control.py"))
assert SPEC is not None and SPEC.loader is not None
control = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = control
SPEC.loader.exec_module(control)


def protocol():
    return json.loads(control.PROTOCOL.read_text())


def prepared():
    declared = protocol()
    declared["brain"]["modules"] = [4]  # software contract, not a behavioral reading
    brain = control.make_brain(declared, 0)
    rows = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]])
    prefix, query, _ = control.sr.inputs(rows, np.array([0, 1, 0]), 1)
    for observation in prefix:
        brain.act(observation, greedy=True)
    return brain, prefix, query


def test_query_boundary_preserves_witnessed_facts_but_removes_trace_and_warm_state():
    brain, prefix, _ = prepared()
    before = {name: value.copy() for name, value in brain.recall.arrays().items()}
    assert brain.basal_ganglia.state is not None
    content = control.query_boundary(brain, len(prefix[0]))
    np.testing.assert_array_equal(content, prefix[0] + prefix[1])
    assert brain.basal_ganglia.state is None and not brain.working_memory.trace.any()
    after = brain.recall.arrays()
    for name, value in before.items():
        np.testing.assert_array_equal(after[name], value)
    content[:] = 0
    assert brain.recall.values.any()  # caller owns its supplied-content copy


def test_both_queries_receive_identical_unselected_fact_content_on_recall_ports():
    brain, _, query = prepared()
    content = control.query_boundary(brain, len(query))
    opposite = query.copy()
    opposite[:, 5:7] = opposite[:, 5:7][:, ::-1]
    first = control.supplied_drive(brain, query, content, "recall_ports", 3.0)
    second = control.supplied_drive(brain, opposite, content, "recall_ports", 3.0)
    read = brain.recall.recall_neurons
    np.testing.assert_array_equal(first[:, read], 3.0 * content)
    np.testing.assert_array_equal(first[:, read], second[:, read])
    assert not first[:, brain.sensory_index[:4]].any()
    assert not first[:, brain.sensory_index[7:]].any()
    assert not first[:, brain.connectome.populations["prefrontal"]].any()


def test_matched_sensory_control_supplies_both_facts_on_original_exposed_coordinates():
    brain, _, query = prepared()
    content = control.query_boundary(brain, len(query))
    drive = control.supplied_drive(brain, query, content, "sensory_exposed", 3.0)
    np.testing.assert_array_equal(drive[:, brain.sensory_index[7:11]], 3.0 * content[:, :4])
    assert not drive[:, brain.recall.recall_neurons].any()


@pytest.mark.parametrize("q", [0, 1])
def test_interventions_change_content_only_with_exact_relevant_irrelevant_and_sham_controls(q):
    brain, _, query = prepared()
    content = control.query_boundary(brain, len(query))
    saved = content.copy()
    relevant = control.changed_content(content, q, "relevant_flip")
    irrelevant = control.changed_content(content, q, "irrelevant_flip")
    donor = control.changed_content(content, q, "donor")
    np.testing.assert_array_equal(relevant[:, 2 * q:2 * q + 2],
                                  content[:, 2 * q:2 * q + 2][:, ::-1])
    other = 1 - q
    np.testing.assert_array_equal(relevant[:, 2 * other:2 * other + 2],
                                  content[:, 2 * other:2 * other + 2])
    np.testing.assert_array_equal(irrelevant[:, 2 * q:2 * q + 2],
                                  content[:, 2 * q:2 * q + 2])
    np.testing.assert_array_equal(donor[:, :4], content[:, :4].reshape(-1, 2, 2)[:, :, ::-1]
                                  .reshape(-1, 4))
    np.testing.assert_array_equal(control.changed_content(content, q, "sham"), content)
    assert not control.changed_content(content, q, "erased").any()
    np.testing.assert_array_equal(content, saved)


def test_supplied_answer_does_not_rebuild_its_drive_and_witnesses_only_the_query(monkeypatch):
    brain, _, query = prepared()
    content = control.query_boundary(brain, len(query))
    drive = control.supplied_drive(brain, query, content, "recall_ports", 3.0)
    before = brain.recall.slots

    def forbidden(*args, **kwargs):
        raise AssertionError("the captured drive must not be rebuilt")

    monkeypatch.setattr(brain, "stimulus", forbidden)
    work = control.sr.Work()
    answer = control.act_saved(brain, query, drive, work)
    assert answer is not None and work.actions == 1 and work.action_refusals == 0
    assert brain.recall.slots == before + 1
    np.testing.assert_array_equal(brain.recall.values[:, -1], query)


def ideal_evaluation():
    target = control.sr.FACTS[:, :2]
    conditions = {}
    for name in control.CONDITIONS:
        answer = (1 - target if name in ("relevant_flip", "donor") else
                  np.zeros_like(target) if name == "erased" else target)
        conditions[name] = {"answers": answer.tolist(), "scores": control.sr.scores(answer),
                            "follows_supplied": 1.0,
                            "change_from_intact": float(np.mean(answer != target))}
    return {"completed": True, "original_preserved": True, "conditions": conditions,
            "work": {"action_refusals": 0}}


def test_gate_requires_acquisition_and_each_causal_control():
    training = {"completed": True, "work": {"action_refusals": 0}}
    reading = ideal_evaluation()
    assert control.gate(training, reading, protocol())
    for name, answer in [
        ("intact", np.zeros((8, 2), dtype=int)),
        ("erased", control.sr.FACTS[:, :2]),
        ("donor", control.sr.FACTS[:, :2]),
        ("relevant_flip", control.sr.FACTS[:, :2]),
        ("irrelevant_flip", 1 - control.sr.FACTS[:, :2]),
        ("sham", 1 - control.sr.FACTS[:, :2]),
    ]:
        changed = deepcopy(reading)
        # Deliberately retain optimistic scores: the gate must judge actual answers.
        changed["conditions"][name]["answers"] = answer.tolist()
        assert not control.gate(training, changed, protocol())
    assert not control.gate({"completed": False}, reading, protocol())


def test_confirmation_is_reserved_and_not_a_runner_stage():
    declared = protocol()
    assert declared["development_seeds"] == [0, 1]
    assert declared["confirmation_seeds"] == list(range(300, 320))
    assert not set(declared["development_seeds"]) & set(declared["confirmation_seeds"])


def retained_body():
    path = Path(__file__).with_name("results") / "content-results-development.json"
    return control.Receipt.read(path).body


def test_retained_source_bound_reading_passes_the_stricter_arithmetic_verifier():
    assert control.check(retained_body()) is None


@pytest.mark.parametrize("corruption", [
    "protocol", "confirmation", "founder_order", "matched_parameters", "schedule",
    "forged_score", "optimistic_completion", "hidden_answer_refusal", "hidden_training_refusal",
    "missing_action", "wrong_rows", "wrong_count_rows", "random_seed", "resolved_recipe",
])
def test_strict_verifier_rejects_altered_gates_custody_and_work(corruption):
    body = retained_body()
    run = body["runs"][0]
    evaluation = run["evaluation"]
    if corruption == "protocol":
        body["protocol"]["acquisition_floor"] = 0.0
    elif corruption == "confirmation":
        body["confirmation_seeds_used"] = [300]
    elif corruption == "founder_order":
        body["runs"] = body["runs"][::-1]
    elif corruption == "matched_parameters":
        run["initial_parameters"] = "0" * 64
    elif corruption == "schedule":
        run["schedule_sha256"] = "0" * 64
    elif corruption == "forged_score":
        evaluation["conditions"]["intact"]["scores"]["accuracy"] = 0.75
    elif corruption == "optimistic_completion":
        evaluation["conditions"]["erased"]["answers"] = [[-1, -1]] * 8
    elif corruption == "hidden_answer_refusal":
        evaluation["work"]["action_refusals"] = 1
    elif corruption == "hidden_training_refusal":
        run["training"]["work"]["action_refusals"] = 1
    elif corruption == "missing_action":
        run["training"]["work"]["actions"] -= 1
        run["training"]["work"]["action_rows"] -= 16
    elif corruption == "wrong_rows":
        run["training"]["work"]["presentations"] += 1
    elif corruption == "wrong_count_rows":
        run["training"]["wrong_counts"][0] = 17
    elif corruption == "random_seed":
        random = run["uniform_random"]
        answer = 1 - np.asarray(random["answers"])
        random["answers"], random["scores"] = answer.tolist(), control.sr.scores(answer)
    else:
        run["resolved"]["genes"]["recall_amplitude"] = 3.0
    assert control.check(body) is not None


def test_gate_cannot_hide_refusal_with_a_true_completion_flag():
    run = retained_body()["runs"][0]
    run["evaluation"]["work"]["action_refusals"] = 1
    assert not control.gate(run["training"], run["evaluation"], protocol())


def test_historical_runner_bytes_and_exact_receipt_are_retained():
    receipt = control.Receipt.read(
        Path(__file__).with_name("results") / "content-results-development.json")
    root = Path(__file__).resolve().parents[2]
    files = []
    for entry in receipt.source["files"]:
        snapshot = Path(__file__).with_name("results") / "sources" / (entry["sha256"] + ".txt")
        files.append((entry["path"], snapshot if snapshot.exists() else root / entry["path"]))
    path = Path(__file__).with_name("results") / "content-results-development.json"
    ok, message = control.Receipt.verify(path, sources=files, check=control.check)
    assert ok, message
