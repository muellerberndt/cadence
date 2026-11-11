"""Adversarial checks for claims extracted from confirmation artifacts."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "confirmation_verifier", Path(__file__).with_name("verify_confirmation.py")
)
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)


def task():
    return {
        "task_seed": 11020261003,
        "family_count": 24,
        "family_to_label": np.random.default_rng(np.random.SeedSequence([11020261003, 0]))
        .permutation(36)[:24]
        .tolist(),
        "panel_observations_per_family": {"train": 4, "development": 2, "heldout": 2},
    }


def query(split="development", *, old_only=False, wrong_row=None, refused_row=None):
    families, _, labels = verifier.panel_identity(task(), split, old_only)
    predictions = labels.copy()
    if wrong_row is not None:
        predictions[wrong_row] = (labels[wrong_row] + 1) % 36
    if refused_row is not None:
        predictions[refused_row] = -1
    correct = predictions == labels
    rows = [
        {
            "row": i,
            "label": int(y),
            "prediction": int(a),
            "steps": 32,
            "residual": 1e-5 if a != -1 else 1.0,
            "cache_defect": 0.0,
            "residual_checks": 2,
        }
        for i, (y, a) in enumerate(zip(labels, predictions, strict=True))
    ]
    result = {
        "correct": int(correct.sum()),
        "examples": len(labels),
        "refusals": int((predictions == -1).sum()),
        "predictions": predictions.tolist(),
        "family_credits": sum(bool(np.all(correct[families == f])) for f in np.unique(families)),
        "rows": rows,
        "work": {
            "calls": len(rows),
            "row_sweeps": 32 * len(rows),
            "reported_residual_checks": 2 * len(rows),
            "independent_residual_checks": len(rows),
            "unreported_residual_work": False,
        },
    }
    for name, ids in (("old", verifier.OLD), ("new", set(range(24)) - set(verifier.OLD))):
        present = [f for f in ids if np.any(families == f)]
        if present:
            result[name + "_family_credits"] = sum(
                bool(np.all(correct[families == f])) for f in present
            )
    return result


def test_family_credit_requires_both_independent_observations():
    record = query(wrong_row=0)
    verifier.query_audit(record, verifier.panel_identity(task(), "development"))
    assert record["correct"] == 47
    assert record["family_credits"] == 23
    record["family_credits"] = 24
    with pytest.raises(ValueError, match="family_credits"):
        verifier.query_audit(record, verifier.panel_identity(task(), "development"))


@pytest.mark.parametrize("field,value", [("correct", 49), ("refusals", 1), ("examples", 47)])
def test_query_summary_cannot_override_raw_answers(field, value):
    record = query()
    record[field] = value
    with pytest.raises(ValueError, match=field):
        verifier.query_audit(record, verifier.panel_identity(task(), "development"))


def test_refusal_remains_wrong_and_charged():
    record = query(refused_row=0)
    verifier.query_audit(record, verifier.panel_identity(task(), "development"))
    assert record["refusals"] == 1 and record["correct"] == 47
    record["work"]["row_sweeps"] -= 32
    with pytest.raises(ValueError, match="work"):
        verifier.query_audit(record, verifier.panel_identity(task(), "development"))


def test_accepted_query_cannot_claim_an_unqualified_equilibrium():
    record = query()
    record["rows"][0]["residual"] = 0.004
    with pytest.raises(ValueError, match="equation gate"):
        verifier.query_audit(record, verifier.panel_identity(task(), "development"))


def test_continuing_gate_requires_old_retention_and_enough_new_families():
    reading = {"train": query("train"), "development": query()}
    initial = copy.deepcopy(reading)
    initial["train"]["correct"] = 16
    assert verifier.stage_gate(reading, initial, 128, True)
    assert not verifier.stage_gate(reading, initial, 127, True)
    reading["development"]["old_family_credits"] = 2
    assert not verifier.stage_gate(reading, initial, 128, True)
    reading["development"]["old_family_credits"] = 4
    reading["development"]["new_family_credits"] = 14
    assert not verifier.stage_gate(reading, initial, 128, True)


def failed_campaign(root):
    sources = {
        "library_sources": {"__init__.py": "# frozen library\n"},
        "producing_sources": {"relations.py": "# frozen generator\n"},
    }
    for group, base in (
        ("library_sources", "source/library/cadence"),
        ("producing_sources", "source"),
    ):
        folder = root / base
        folder.mkdir(parents=True, exist_ok=True)
        for name, source in sources[group].items():
            (folder / name).write_text(source)
    fixture = root / "source/fixture"
    fixture.mkdir()
    frozen_task = task()
    (fixture / "protocol.json").write_bytes(verifier.canonical(frozen_task))
    protocol = {
        "schema": verifier.SCHEMAS[1],
        "founder_seeds": list(verifier.SEEDS),
        "founder_denominator": 5,
        "learner_config": {
            "beta": 0.1,
            "centered": True,
            "nudge": "cross_entropy",
            "temperature": 0.2,
            "normalize_floor": 0.001,
            "decay": 0.0,
            "scale_cap": 8.0,
            "eta": 0.1,
            "eta_bias": 0.01,
            "normalize": 0.0,
            "momentum": 0.0,
            "free_steps": 4096,
            "nudged_steps": 4096,
            "qualified": True,
            "damping": 3,
            "tolerance": 0.003,
        },
        "neuron_model": {
            "dt": 1.0,
            "slope": 1.0,
            "threshold": 0.0,
            "gain": 1.0,
            "stimulus_amplitude": 1.0,
            "adaptation": None,
            "leak": 0.1,
        },
        "model_construction": {
            "inputs": 650,
            "actions": 36,
            "modules": [32, 16],
            "observers": [],
            "lateral": -0.5,
        },
        "relations_protocol_sha256": verifier.digest(verifier.canonical(frozen_task)),
        **{
            group: {name: verifier.digest(source.encode()) for name, source in values.items()}
            for group, values in sources.items()
        },
    }
    (root / "protocol.json").write_bytes(verifier.canonical(protocol))
    census = {
        "recipe_sha256": verifier.digest((root / "protocol.json").read_bytes()),
        "founder_denominator": 5,
        "passed": 0,
        "outcomes": [
            {"seed": seed, "status": "process_failed", "passed": False} for seed in verifier.SEEDS
        ],
    }
    (root / "summary.json").write_bytes(verifier.canonical(census))
    return census


def test_process_errors_stay_in_the_complete_denominator_without_fabricated_work(tmp_path):
    failed_campaign(tmp_path)
    result = verifier.verify(tmp_path)
    assert result["schema"] == "cadence.acquisition-demo.v1"
    assert result["passed_founders"] == 0 and result["founder_denominator"] == 5
    assert all(not f["exact_work_available"] for f in result["founders"])
    assert not result["verification_scope"]["historical_trajectory_replayed"]


@pytest.mark.parametrize("mutation", ["missing_seed", "false_count", "running_seed", "source"])
def test_incomplete_or_relabelled_campaign_is_rejected(tmp_path, mutation):
    census = failed_campaign(tmp_path)
    if mutation == "missing_seed":
        census["outcomes"].pop()
    elif mutation == "false_count":
        census["passed"] = 5
    elif mutation == "running_seed":
        census["outcomes"][0]["status"] = "running"
    else:
        (tmp_path / "source/relations.py").write_text("# changed generator\n")
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    with pytest.raises(ValueError):
        verifier.verify(tmp_path)


def test_verified_archive_rejects_member_changes_even_if_outer_hash_is_replaced(tmp_path):
    archive = tmp_path / "seed-1.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        payload = b"changed"
        info = tarfile.TarInfo("seed-1/summary.json")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    archive_hash = verifier.digest(archive.read_bytes())
    custody = tmp_path / "custody"
    custody.mkdir()
    (custody / "seed-1.json").write_text(
        json.dumps(
            {
                "archive_sha256": archive_hash,
                "original_member_sha256": {"seed-1/summary.json": verifier.digest(b"original")},
            }
        )
    )
    with pytest.raises(ValueError, match="member custody"):
        verifier.Artifacts(tmp_path, 1, {"outer_archive_sha256": archive_hash})


def test_archive_cannot_escape_its_artifact_root():
    with pytest.raises(ValueError, match="unsafe"):
        verifier.safe_name("../checkpoint.npz")


def test_demo_panels_keep_correct_rows_distinct_from_whole_family_recall():
    record = {"lesson": 192, "train": query("train"), "development": query(wrong_row=0)}
    result = verifier.demo_reading(record, task(), True)
    assert result["development"]["correct_rows"] == 47
    assert result["development"]["correct_families"] == 23
    assert result["development"]["old_correct"] == 3


def test_uniform_random_control_has_actual_answers_and_a_separate_expectation():
    identity = verifier.panel_identity(task(), "development")
    control = verifier.random_control(identity, 5)
    assert control == verifier.random_control(identity, 5)
    answers = np.asarray(control["predictions"])
    assert control["correct_rows"] == int((answers == identity[2]).sum())
    assert control["total_rows"] == 48
    assert control["expected_row_accuracy"] == 1 / 36
    assert control["row_accuracy_standard_deviation"] > 0
    assert control["measured"] is True
    assert control["work"]["teacher_presentations"] == 0


def candidate_campaign(tmp_path, schema_index):
    census = failed_campaign(tmp_path)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    protocol["schema"] = verifier.SCHEMAS[schema_index]
    first_seed = {2: 6, 3: 11, 4: 16, 5: 21}[schema_index]
    protocol["founder_seeds"] = list(range(first_seed, first_seed + 5))
    protocol["candidate_gene"] = {"eta": 0.05, "eta_bias": 0.005}
    protocol["learner_config"].update(eta=0.05, eta_bias=0.005)
    if schema_index == 3:
        protocol["neuron_model"]["leak"] = 1.0
    if schema_index in (4, 5):
        protocol["model_construction"]["lateral"] = 0.0
    if schema_index == 5:
        protocol["candidate_gene"].update(
            eta=0.003, eta_bias=0.0003, normalize=0.99, momentum=0.0,
            normalize_floor=0.001,
        )
        protocol["learner_config"].update(eta=0.003, eta_bias=0.0003, normalize=0.99)
    protocol["pedagogy_seed"] = 1100301
    for name, count, stage in (("old", 1024, 1), ("mixed", 4096, 2)):
        order = verifier.frozen_order(task(), count + 1, stage, name == "old")
        protocol[name + "_lesson_cap"] = count
        protocol[name + "_order"] = order[:-1]
        protocol[name + "_after_cap_next_row"] = order[-1]
    reference = tmp_path / "reference-development"
    reference.mkdir()
    original_protocol = {
        "learner_config": protocol["learner_config"],
        "neuron_model": protocol["neuron_model"],
        "model_construction": protocol["model_construction"],
        "founder_seeds": [0],
        "old_cap": 1024,
        "mixed_cap": 4096,
    }
    raw = verifier.canonical(original_protocol)
    (reference / "protocol.json").write_bytes(raw)
    original_summary = {
        "passed": True,
        "heldout_read": False,
        "protocol_sha256": verifier.digest(raw),
    }
    (reference / "summary.json").write_bytes(verifier.canonical(original_summary))
    protocol["successful_development_protocol_sha256"] = verifier.digest(raw)
    protocol["successful_development_summary_sha256"] = verifier.digest(
        (reference / "summary.json").read_bytes()
    )
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(protocol))
    census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    for row, seed in zip(census["outcomes"], protocol["founder_seeds"], strict=True):
        row["seed"] = seed
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    return census, protocol, original_protocol, original_summary, raw


@pytest.mark.parametrize("schema_index", [2, 3, 4, 5])
def test_new_recipe_requires_matching_passed_development_and_preserves_five_seeds(
    tmp_path, schema_index
):
    census, protocol, original_protocol, original_summary, raw = candidate_campaign(
        tmp_path, schema_index
    )
    reference = tmp_path / "reference-development"
    result = verifier.verify(tmp_path)
    assert result["verification_scope"]["historical_development_reference_hashes"]
    assert [f["seed"] for f in result["founders"]] == protocol["founder_seeds"]
    reused = copy.deepcopy(protocol)
    reused["founder_seeds"][0] = 0
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(reused))
    reused_census = copy.deepcopy(census)
    reused_census["outcomes"][0]["seed"] = 0
    reused_census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    (tmp_path / "summary.json").write_bytes(verifier.canonical(reused_census))
    with pytest.raises(ValueError, match="reuses a development founder"):
        verifier.verify(tmp_path)
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(protocol))
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    changed = copy.deepcopy(protocol)
    changed["old_lesson_cap"] = 1028
    order = verifier.frozen_order(task(), 1029, 1, True)
    changed["old_order"], changed["old_after_cap_next_row"] = order[:-1], order[-1]
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(changed))
    census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    with pytest.raises(ValueError, match="lesson caps differ"):
        verifier.verify(tmp_path)
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(protocol))
    census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    if schema_index == 3:
        changed_reference = copy.deepcopy(original_protocol)
        changed_reference["neuron_model"]["leak"] = 0.1
        changed_raw = verifier.canonical(changed_reference)
        (reference / "protocol.json").write_bytes(changed_raw)
        changed_summary = dict(original_summary, protocol_sha256=verifier.digest(changed_raw))
        (reference / "summary.json").write_bytes(verifier.canonical(changed_summary))
        changed_protocol = dict(
            protocol,
            successful_development_protocol_sha256=verifier.digest(changed_raw),
            successful_development_summary_sha256=verifier.digest(
                (reference / "summary.json").read_bytes()
            ),
        )
        (tmp_path / "protocol.json").write_bytes(verifier.canonical(changed_protocol))
        census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
        (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
        with pytest.raises(ValueError, match="neuron model differs"):
            verifier.verify(tmp_path)
        (reference / "protocol.json").write_bytes(raw)
        (reference / "summary.json").write_bytes(verifier.canonical(original_summary))
        (tmp_path / "protocol.json").write_bytes(verifier.canonical(protocol))
        census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
        (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    original_summary["passed"] = False
    (reference / "summary.json").write_bytes(verifier.canonical(original_summary))
    with pytest.raises(ValueError, match="reference differs"):
        verifier.verify(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [("normalize", 0.0), ("normalize", 0.98), ("momentum", 0.9),
     ("normalize_floor", 0.0001), ("tolerance", 0.03), ("extra_optimizer", 1)],
)
def test_normalized_candidate_keeps_the_exact_fixed_recipe(tmp_path, field, value):
    census, protocol, *_ = candidate_campaign(tmp_path, 5)
    protocol["learner_config"][field] = value
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(protocol))
    census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    with pytest.raises(ValueError, match="recipe differs|configuration inventory"):
        verifier.verify(tmp_path)


def test_normalized_gene_cannot_disagree_with_the_actual_recipe(tmp_path):
    census, protocol, *_ = candidate_campaign(tmp_path, 5)
    protocol["candidate_gene"]["normalize"] = 0.9
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(protocol))
    census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    with pytest.raises(ValueError, match="declared normalized gene"):
        verifier.verify(tmp_path)


@pytest.mark.parametrize("schema_index", [0, 1, 2, 3, 4])
def test_prior_protocols_do_not_silently_enable_normalization(tmp_path, schema_index):
    if schema_index < 2:
        census = failed_campaign(tmp_path)
        protocol = json.loads((tmp_path / "protocol.json").read_text())
        protocol["schema"] = verifier.SCHEMAS[schema_index]
    else:
        census, protocol, *_ = candidate_campaign(tmp_path, schema_index)
    protocol["learner_config"]["normalize"] = 0.99
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(protocol))
    census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    with pytest.raises(ValueError, match="recipe differs: normalize"):
        verifier.verify(tmp_path)


def test_original_recipe_does_not_silently_change_founders_or_tolerance(tmp_path):
    census = failed_campaign(tmp_path)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    protocol["learner_config"]["tolerance"] = 0.03
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(protocol))
    census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    with pytest.raises(ValueError, match="tolerance"):
        verifier.verify(tmp_path)


@pytest.mark.parametrize(
    "field,value", [("leak", 1.01), ("slope", 2.0), ("threshold", 0.1), ("adaptation", {})]
)
def test_source_bound_leak_candidate_does_not_relax_other_neuron_settings(tmp_path, field, value):
    census = failed_campaign(tmp_path)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    protocol.update(schema=verifier.SCHEMAS[3], candidate_gene={"eta": 0.05, "eta_bias": 0.005})
    protocol["founder_seeds"] = list(range(11, 16))
    protocol["learner_config"].update(eta=0.05, eta_bias=0.005)
    protocol["neuron_model"].update(leak=1.0)
    protocol["neuron_model"][field] = value
    (tmp_path / "reference-development").mkdir()
    (tmp_path / "protocol.json").write_bytes(verifier.canonical(protocol))
    census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    for row, seed in zip(census["outcomes"], protocol["founder_seeds"], strict=True):
        row["seed"] = seed
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    with pytest.raises(ValueError, match="leak gene|neuron rule"):
        verifier.verify(tmp_path)


@pytest.mark.parametrize("lateral", [-0.5, -0.0001, 0.5, float("inf")])
def test_zero_lateral_candidate_requires_its_exact_declared_construction(tmp_path, lateral):
    census = failed_campaign(tmp_path)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    protocol.update(schema=verifier.SCHEMAS[4], candidate_gene={"eta": 0.05, "eta_bias": 0.005})
    protocol["founder_seeds"] = list(range(16, 21))
    protocol["learner_config"].update(eta=0.05, eta_bias=0.005)
    protocol["model_construction"]["lateral"] = lateral
    (tmp_path / "reference-development").mkdir()
    # JSON normally refuses infinities; retain this malformed input to exercise the audit.
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    census["recipe_sha256"] = verifier.digest((tmp_path / "protocol.json").read_bytes())
    for row, seed in zip(census["outcomes"], protocol["founder_seeds"], strict=True):
        row["seed"] = seed
    (tmp_path / "summary.json").write_bytes(verifier.canonical(census))
    with pytest.raises(ValueError, match="construction differs"):
        verifier.verify(tmp_path)
