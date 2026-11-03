"""Independent accounting/fixture guards for the native acquisition protocol."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadence import Brain, BrainState, LearnerConfig

ROOT = Path(__file__).parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extract = load_script("extract")
run = load_script("run")
verifier = load_script("verify")


def test_fixture_panels_are_hash_bound_disjoint_balanced_and_sealed():
    folder = ROOT / "fixture"
    provenance = json.loads((folder / "provenance.json").read_text())
    assert extract.sha256(folder / "school.npz") == provenance["fixture_sha256"]
    with np.load(folder / "school.npz", allow_pickle=False) as data:
        for name, identities in provenance["panels"].items():
            inputs, labels = data[name + "_inputs"], data[name + "_labels"]
            assert inputs.dtype == np.dtype("float64")
            assert inputs.shape == (len(identities), 650)
            assert np.isfinite(inputs).all()
            for row, identity in enumerate(identities):
                assert extract.row_hash(inputs[row]) == identity["input_sha256"]
                assert labels[row] == identity["label"]
        school = {r["parent_row"] for r in provenance["panels"]["school"]}
        independent = {r["parent_row"] for r in provenance["panels"]["independent_train"]}
        assert not school & independent
        assert len(independent) == 18
        assert len(provenance["panels"]["development"]) == 19
        families = provenance["confirmation_families"]
        assert len(families) == 14
        for name in ("confirmation_train", "confirmation_development", "heldout"):
            assert sorted(data[name + "_labels"].tolist()) == families


def test_independent_residual_matches_original_nudge_and_detects_false_cache():
    brain = Brain.compose(
        inputs=2,
        actions=2,
        modules=(3,),
        seed=4,
        learning=LearnerConfig(free_steps=12, nudged_steps=12),
    )
    inputs, labels = np.eye(2), np.array([0, 1])
    drive = brain.stimulus(inputs, memory=False)
    free = brain.learner.free(drive)
    for sign in (0, 1, -1):
        target = brain.learner.targets(labels)
        state = free if sign == 0 else brain.learner.nudged(drive, free, target, sign=sign)
        nudge = None if sign == 0 else brain.learner.nudge_for(target, sign * 0.1)
        residual, cache = run.independent_residual(brain, drive, state, labels, sign)
        np.testing.assert_allclose(
            residual, brain.brain.residual(drive, state, nudge=nudge), atol=2e-14, rtol=2e-14
        )
        np.testing.assert_allclose(cache, 0, atol=2e-14)
    corrupted = BrainState(free.v, free.activation + 0.5, free.adaptation, free.steps)
    _, cache = run.independent_residual(brain, drive, corrupted)
    assert np.all(cache > 0.49)


def test_candidate_arguments_cannot_silently_change_finite_baseline():
    args = SimpleNamespace(rate=0.01, free_steps=17, nudged_steps=19, nudge="cross_entropy",
                           damping=2)
    finite = run.make_brain("finite", 0, args)
    cfg = finite.learner.config
    assert (cfg.free_steps, cfg.nudged_steps, cfg.eta, cfg.nudge) == (
        1024,
        12,
        0.5,
        "cross_entropy",
    )
    candidate = run.make_brain("qualified", 0, args)
    cfg = candidate.learner.config
    assert (cfg.free_steps, cfg.nudged_steps, cfg.eta, cfg.nudge, cfg.qualified) == (
        17,
        19,
        0.01,
        "cross_entropy",
        True,
    )


@pytest.mark.parametrize(
    "gene", ["fixed-lateral-local-rms", "lateral0-local-rms", "lateral0-resting"]
)
def test_effective_genes_match_actual_wiring_bias_and_learning_without_changing_control(gene):
    args = SimpleNamespace(
        rate=0.02, free_steps=17, nudged_steps=19, nudge="cross_entropy", damping=2, gene=gene,
    )
    candidate = run.make_brain("qualified", 0, args)
    effective = run.recipe_configuration("qualified", args)
    assert candidate.learner.config.to_dict() == effective["learning"]
    assert candidate.learner.config.normalize == 0.99
    graph = candidate.brain
    motor_edges = np.isin(graph.connectome.pre, candidate.motor_index) & np.isin(
        graph.connectome.post, candidate.motor_index
    )
    if gene == "fixed-lateral-local-rms":
        assert candidate.learner.config.eta == 0.005
        assert candidate.learner.config.normalize_floor == 0.001
        assert np.any(motor_edges) and not candidate.learner.plastic_synapses[motor_edges].any()
        np.testing.assert_array_equal(graph.bias[candidate.sensory_index], 0.6)
        # A different requested rate is explicitly not a different fixed-gene experiment.
        args.rate = 0.05
        assert run.recipe_configuration("qualified", args) == effective
    else:
        assert candidate.learner.config.eta == args.rate
        assert candidate.learner.config.normalize_floor == 1e-4
        assert not motor_edges.any()
        np.testing.assert_array_equal(graph.bias[candidate.sensory_index], 0.0)
    np.testing.assert_array_equal(graph.bias[candidate.motor_index], 0.0)
    for name in ("module_0", "association"):
        np.testing.assert_array_equal(
            graph.bias[np.asarray(graph.connectome.populations[name])],
            0.5 if gene == "lateral0-resting" else 0,
        )
    finite = run.make_brain("finite", 0, args)
    control = run.make_brain("finite", 0, SimpleNamespace())
    assert run.recipe_configuration("finite", args) == run.recipe_configuration(
        "finite", SimpleNamespace()
    )
    np.testing.assert_array_equal(finite.brain.bias, control.brain.bias)
    np.testing.assert_array_equal(finite.connectome.pre, control.connectome.pre)
    np.testing.assert_array_equal(finite.connectome.post, control.connectome.post)


def test_source_version_is_not_replaced_by_unrelated_installed_metadata(monkeypatch):
    monkeypatch.setattr(run.importlib.metadata, "version", lambda package: "0.0.0-unrelated")
    identity = run.library_identity()
    assert identity["cadence_version"] == run.cadence.__version__
    assert identity["installed_distribution_version"] == "0.0.0-unrelated"
    assert identity["cadence_module"] == str(Path(run.cadence.__file__).resolve())


def test_small_rms_native_transfer_keeps_exact_coupled_rates_and_control():
    args = SimpleNamespace(rate=0.5, free_steps=4096, nudged_steps=4096,
                           nudge="cross_entropy", damping=3, gene="lateral0-small-rms")
    brain = run.make_brain("qualified", 0, args)
    cfg = brain.learner.config
    assert (cfg.eta, cfg.eta_bias, cfg.normalize, cfg.momentum, cfg.normalize_floor) == (
        0.003, 0.0003, 0.99, 0.0, 0.001,
    )
    np.testing.assert_array_equal(brain.brain.bias, 0.0)
    assert brain.learner.plastic_synapses.all() and brain.learner.plastic_neurons.all()
    motor_edges = np.isin(brain.connectome.pre, brain.motor_index) & np.isin(
        brain.connectome.post, brain.motor_index
    )
    assert not motor_edges.any()
    args.rate = 0.05
    assert run.recipe_configuration("qualified", args)["learning"] == cfg.to_dict()
    finite = run.make_brain("finite", 0, args)
    assert (finite.learner.config.eta, finite.learner.config.eta_bias,
            finite.learner.config.momentum, finite.learner.config.normalize) == (
        0.5, 0.02, 0.9, 0.0,
    )


def test_corrupted_source_receipt_refuses_extraction_before_writing(tmp_path):
    witness = tmp_path / "witness.json"
    verification = tmp_path / "verification.json"
    selection = tmp_path / "selection.json"
    witness.write_text(json.dumps({"source": {"inputs": 650, "window": 12}}))
    verification.write_text(json.dumps({"synced": True, "deaths": 0}))
    selection.write_text(json.dumps({"dataset_sha256": "false", "verification_sha256": "false"}))
    with pytest.raises(ValueError, match="parent fixture hashes"):
        extract.extract(witness, verification, selection, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.fixture(scope="module")
def refused_artifact(tmp_path_factory):
    folder = tmp_path_factory.mktemp("native-school") / "refused"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--out",
            str(folder),
            "--recipe",
            "qualified",
            "--updates",
            "3",
            "--check-every",
            "1",
            "--free-steps",
            "1",
            "--nudged-steps",
            "1",
            "--seconds",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,  # the run budgets its own ten seconds; a loaded host is given more
    )
    return folder


def test_refused_artifact_replays_with_zero_accepted_exposure(refused_artifact):
    result = verifier.verify(refused_artifact)
    assert result["passed"] and result["source_bound"]
    assert result["cases"][0]["accepted"] == 0
    assert result["cases"][0]["refused"] == 1


def test_refusal_is_terminal_and_work_is_charged_without_mutating_the_founder(refused_artifact):
    protocol = json.loads((refused_artifact / "protocol.json").read_text())
    receipt = json.loads((refused_artifact / "qualified-2/receipt.json").read_text())
    assert protocol["arguments"]["updates"] == 3
    assert "terminates the stage" in protocol["refusal_policy"]
    assert protocol["effective_recipes"]["qualified"]["learning"] == receipt["learner_config"]
    assert receipt["status"] == "refused_learning" and receipt["attempted_updates"] == 1
    assert receipt["accepted_updates"] == receipt["accepted_row_exposures"] == 0
    assert receipt["work"]["refused_learning_calls"] == 1
    assert receipt["work"]["phase_row_sweeps"] > 0
    with np.load(refused_artifact / "qualified-2/initial.npz", allow_pickle=False) as before:
        with np.load(refused_artifact / "qualified-2/final.npz", allow_pickle=False) as after:
            assert set(before.files) == set(after.files)
            for name in before.files:
                np.testing.assert_array_equal(before[name], after[name], err_msg=name)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("accepted_phase", "accepted qualified phase"),
        ("exposure", "accepted exposure"),
        ("phase_hash", "raw phase hash"),
        ("checkpoint_hash", "final checkpoint hash"),
        ("removed_case", "completeness"),
        ("summary", "summary outcome"),
        ("protocol_hash", "summary protocol hash"),
        ("work", "work accounting"),
        ("missing_work", "mandatory work field"),
        ("effective_rate", "effective recipe differs"),
        ("receipt_rate", "receipt learner configuration differs"),
    ],
)
def test_verifier_rejects_false_green_mutations(refused_artifact, tmp_path, mutation, reason):
    folder = tmp_path / "mutated"
    shutil.copytree(refused_artifact, folder)
    case = folder / "qualified-2"
    receipt = json.loads((case / "receipt.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if mutation == "accepted_phase":
        receipt["updates"][0]["report"]["accepted"] = 1
    elif mutation == "exposure":
        receipt["accepted_row_exposures"] += 1
    elif mutation == "work":
        receipt["work"]["phase_row_sweeps"] += 1
    elif mutation == "missing_work":
        del receipt["work"]["phase_row_sweeps"]
    elif mutation in ("phase_hash", "checkpoint_hash"):
        path = case / ("phases-0001.npz" if mutation == "phase_hash" else "final.npz")
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
    elif mutation == "summary":
        summary["outcomes"][0]["stages"][0]["correct"] += 1
    elif mutation == "protocol_hash":
        summary["protocol_sha256"] = "false"
    elif mutation == "effective_rate":
        protocol = json.loads((folder / "protocol.json").read_text())
        protocol["effective_recipes"]["qualified"]["learning"]["eta"] *= 0.5
        extract.write_json(folder / "protocol.json", protocol)
        summary["protocol_sha256"] = extract.sha256(folder / "protocol.json")
    elif mutation == "receipt_rate":
        receipt["learner_config"]["eta"] *= 0.5
    if mutation == "removed_case":
        shutil.rmtree(case)
    else:
        extract.write_json(case / "receipt.json", receipt)
    extract.write_json(folder / "summary.json", summary)
    with pytest.raises(ValueError, match=reason):
        verifier.verify(folder)


def test_verifier_requires_all_centered_phases_for_an_accepted_lesson(tmp_path):
    folder = tmp_path / "finite"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--out",
            str(folder),
            "--recipe",
            "finite",
            "--updates",
            "1",
            "--check-every",
            "1",
            "--seconds",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,  # the run budgets its own ten seconds; a loaded host is given more
    )
    receipt_path = folder / "finite-2/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    del receipt["updates"][0]["phases"]["opposite"]
    extract.write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="accepted lesson phase census"):
        verifier.verify(folder)
