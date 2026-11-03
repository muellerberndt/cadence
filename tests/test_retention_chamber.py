"""#85 harness gates/custody; no chamber learning campaign is run by these tests."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import cadence as cd


@pytest.fixture
def chamber(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks/retention"
    for name in ("chamber_inputs", "actual_outcome_chamber"):
        spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["actual_outcome_chamber"]


def test_world_rewards_only_the_action_actually_executed(chamber):
    identities = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    actions = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    np.testing.assert_array_equal(chamber.inputs.actual_reward(identities, actions, 0),
                                  [1, 1, 1, 0, 0, 0, -1, 0])
    np.testing.assert_array_equal(chamber.inputs.actual_reward(identities, actions, 1),
                                  [0, 0, 1, 0, 1, 1, -1, 0])
    with pytest.raises(ValueError):
        chamber.inputs.actual_reward(identities, actions.astype(float), 0)


def test_frozen_schedule_and_partial_features_are_genuine(chamber):
    arrays = chamber.inputs.frozen_arrays(0)
    acquisition = arrays["schedule/acquisition"]
    assert acquisition.shape == (256, 8)
    for row in acquisition:
        np.testing.assert_array_equal(np.bincount(row, minlength=4), [2, 2, 2, 2])
    for phase in ("routine", "revision", "restored"):
        schedule = arrays["schedule/" + phase]
        for start in range(0, len(schedule), 16):
            for stream in range(8):
                np.testing.assert_array_equal(np.bincount(schedule[start:start + 16, stream],
                                                           minlength=4), [7, 7, 1, 1])
    for condition in chamber.inputs.CONDITIONS:
        prototypes = arrays[condition + "/prototypes"]
        np.testing.assert_allclose(np.linalg.norm(prototypes, axis=1), 1)
        probes = arrays[condition + "/probes"]
        assert probes.shape == (3, 10, 8)
        for cue in range(3):
            supported = np.count_nonzero(prototypes[cue])
            assert np.all(np.count_nonzero(probes[cue, 3:7], axis=1) == supported // 2)
    correlated = arrays["correlated/prototypes"]
    assert np.dot(correlated[1], correlated[2]) == pytest.approx(0.9)
    # Every deleted-feature correlated B retains evidence distinct from C.
    assert np.all(arrays["correlated/probes"][1, 3:7, 2:4].sum(axis=1) > 0)


@pytest.mark.parametrize("consolidation,rate", [(0.05, 1), (0, 0)])
def test_independent_masked_memory_math_and_input_purity(chamber, consolidation, rate):
    # Assigned-record component fixture; it makes no actual-experience claim.
    memory = cd.SynapticMemory(np.arange(8), np.arange(8, 10),
                               consolidation=consolidation, rate=rate)
    keys = chamber.inputs.prototypes("correlated")[[0, 1, 2, 3, 1, 2, 0, 3]]
    actions, reward = np.arange(8) % 2, np.array([1, 0, 1, 0, -1, 1, 0, 0.0])
    initial_keys = keys.copy()
    for _ in range(3):
        previous = chamber.memory_arrays(memory)
        target, mask = np.zeros((8, 2)), np.zeros((8, 2), bool)
        target[np.arange(8), actions] = reward
        mask[np.arange(8), actions] = True
        memory.observe(keys, target, salience=np.abs(reward), value_mask=mask)
        deviation = chamber.check_memory(previous, memory, keys, actions, reward)
        assert max(deviation.values()) < 1e-12
        np.testing.assert_array_equal(keys, initial_keys)
    assert memory.writes == 24


def test_constructor_preserves_the_ordinary_declared_gene(chamber):
    candidate = chamber.make_brain(0)
    # the chamber froze the working trace of its release (amplitude 3, retention 0.2, scale 12)
    ordinary = cd.Brain.compose(8, 2, modules=(32,), lateral=-0.5, seed=0,
                                working_memory_amplitude=3.0, working_memory_decay=0.2,
                                scales={"trace": 12.0})
    left, right = chamber.model_identity(candidate), chamber.model_identity(ordinary)
    assert left == right
    assert left["learner"]["normalize"] == 0
    assert chamber.capacity(candidate)["C_shape"] == [8, 2]
    assert chamber.capacity(candidate)["strength_capacity_shape"] == [8, 8, 2]


def test_observational_capture_preserves_public_actual_reward_checkpoint(chamber, tmp_path):
    # Smaller, frozen-policy component fixture, not a development brain or recipe probe.
    brain = cd.Brain.compose(8, 2, modules=(4,), seed=9,
                             learning=cd.LearnerConfig(eta=0, eta_bias=0, free_steps=1024,
                                                       nudged_steps=12, tolerance=0.003),
                             reward=cd.ActorCriticConfig(eta=0, eta_bias=0, eta_critic=0,
                                                         eligibility_steps=12))
    copied = cd.Brain.load(brain.save(tmp_path / "original.npz"))
    key = np.eye(8)
    raw_action = brain.act(key)
    brain.learn(np.ones(8), np.ones(8, bool), key)
    with chamber.Capture() as act_work:
        action = copied.act(key)
    np.testing.assert_array_equal(action, raw_action)
    witness = {"key": key, "actions": action, "reward": np.ones(8)}
    with chamber.Capture(witness) as feedback_work:
        copied.learn(np.ones(8), np.ones(8, bool), key)
    assert chamber.checkpoints_equal(brain.save(tmp_path / "plain.npz"),
                                     copied.save(tmp_path / "captured.npz"))
    assert act_work.work["finite_phase_calls"] == 2
    assert act_work.work["solver_row_sweeps"] > 0
    assert feedback_work.work["store_writing_rows"] == 8
    assert len(feedback_work.stores) == 1
    assert all(np.max(phase["cache_defect"]) < 1e-14 for phase in act_work.phases)


def test_capture_rejects_unexecuted_action_values_before_a_write(chamber):
    memory = cd.SynapticMemory(np.arange(8), np.arange(8, 10))
    keys, actions, reward = np.eye(8), np.zeros(8, int), np.ones(8)
    with chamber.Capture({"key": keys, "actions": actions, "reward": reward}):
        with pytest.raises(AssertionError, match="executed-action witness"):
            memory.observe(keys, np.ones((8, 2)), salience=reward,
                           value_mask=np.ones((8, 2), bool))
    assert memory.writes == 0 and not memory.consolidated.any()


def test_saved_pending_private_refusal_and_next_action_contract(chamber, tmp_path):
    # Frozen-policy preservation fixture: no candidate actor/plastic-graph development.
    brain = cd.Brain.compose(8, 2, modules=(32,), seed=3,
                             learning=cd.LearnerConfig(eta=0, eta_bias=0, free_steps=1024,
                                                       nudged_steps=12, tolerance=0.003),
                             reward=cd.ActorCriticConfig(eta=0, eta_bias=0, eta_critic=0,
                                                         eligibility_steps=12))
    root = tmp_path / "custody"
    root.mkdir()
    (root / "protocol.json").write_text("{}\n")
    journal = chamber.Journal(root, {"founders": [0], "worker_seconds": 300,
                                     "output_cap_mib": 64,
                                     "output_reserve_bytes": 2 * 1024**2,
                                     "storage_reconciliation_seconds": 5}, time.monotonic())
    key = np.eye(8)
    with chamber.Capture():
        actions = brain.act(key)
    witness = {"key": key, "actions": actions, "reward": np.ones(8),
               "done": np.ones(8, bool)}
    flags = chamber.checkpoint_seam(journal, brain, witness, key,
                                    "seam", {"fixture": "frozen-policy-actual-outcome"})
    assert flags["passed"]
    assert brain.hippocampus.writes == 0  # Main pending action has not consumed an outcome.
    assert brain.basal_ganglia._pending is not None
    assert journal.totals["deliberate_feedback_refusal_refusals"] == 1
    assert journal.totals["actor_feedback_bookkeeping_updates"] == 2
    assert journal.totals["parameter_changing_feedback_updates"] == 0


def endpoint(correct=9, *, uniform=27, joint=0):
    cues = [{"correct": correct, "prototype_correct": 1, "refused": 0,
             "unrun": 0, "obsolete": 10 - correct} for _ in range(3)]
    return {"intact": {"cues": cues, "refused": 0, "unrun": 0},
            "uniform": {"correct": uniform},
            "joint_reset": {"cues": [{"correct": joint} for _ in range(3)]}}


def good_endpoints():
    return {name: endpoint(uniform=15)
            for name in ("acquired", "routine", "revision", "restored")}


def test_false_retention_and_refused_continuation_cannot_pass(chamber):
    scores = good_endpoints()
    assert chamber.score_gate(scores, [{"passed": True}], complete=True, required=True)["passed"]
    for defect in ("wrong-prototype", "old-route", "refused-probe", "refused-continuation"):
        broken, seams = copy.deepcopy(scores), [{"passed": True}]
        if defect == "wrong-prototype":
            broken["acquired"]["intact"]["cues"][0]["prototype_correct"] = 0
        elif defect == "old-route":
            broken["revision"]["intact"]["cues"][1]["obsolete"] = 2
        elif defect == "refused-probe":
            broken["routine"]["intact"]["cues"][2]["refused"] = 1
        else:
            seams[0]["passed"] = False
        assert not chamber.score_gate(broken, seams, complete=True, required=True)["passed"]
    broken = copy.deepcopy(scores)
    del broken["routine"]
    broken["newborn"] = endpoint()
    assert chamber.score_gate(broken, [{"passed": True}], complete=True,
                               required=True)["reason"] == "incomplete-census"


def test_zero_revision_curve_scores_the_new_world_without_query(chamber):
    reading = {"world": 0, "cues": [{"label": label, "predictions": [label] * 10,
                                      "prototype_prediction": label, "margins": [0.2] * 10,
                                      "correct": 10, "obsolete": 0, "prototype_correct": 1}
                                     for label in (0, 1, 0)]}
    updated = chamber.rescore_previous(reading, 1)
    assert [row["correct"] for row in updated["cues"]] == [0, 0, 10]
    assert [row["obsolete"] for row in updated["cues"]] == [10, 10, 0]
    assert updated["cues"][0]["margins"] == [-0.2] * 10
    assert reading["cues"][0]["correct"] == 10
    assert updated["reused_predictions_zero_new_solves"] is True


def test_baseline_gap_is_cohort_not_each_ten_probe_sample(chamber):
    scores = good_endpoints()
    scores["acquired"]["uniform"]["correct"] = 27  # Same success in this sample.
    assert chamber.score_gate(scores, [{"passed": True}], complete=True, required=True)["passed"]
    measured = chamber.cohort_gate([{"required": True, "endpoints": scores}])
    assert measured["passed"] is True
    assert measured["planned"] == 120
    assert measured["accuracy"] == pytest.approx(0.9)
    for value in scores.values():
        value["uniform"]["correct"] = 27
    assert not chamber.cohort_gate([{"required": True, "endpoints": scores}])["passed"]


def test_refused_causal_baseline_is_not_a_qualified_control(chamber):
    reading = {"qualified_free": True, "planned": 33, "attempted": 33,
               "unrun": 0, "refused": 0}
    assert chamber.probes_complete({"intact": reading, "joint_reset": reading})
    refused = {**reading, "qualified_free": False, "refused": 1}
    assert not chamber.probes_complete({"intact": reading, "joint_reset": refused})


@pytest.fixture
def capsule(chamber, monkeypatch, tmp_path):
    for name in chamber.THREADS:
        monkeypatch.setenv(name, "1")
    root = tmp_path / "prepared"
    protocol = chamber.prepare(root)
    return root, protocol


def test_preparation_has_no_solves_and_preflights_exact_initial_source(chamber, capsule):
    root, protocol = capsule
    assert len(protocol["lives"]) == 6
    assert not (root / "journal.jsonl.gz").exists()
    assert not (root / "summary.json").exists()
    assert protocol["formal_evidence"] == {
        "requested": False, "status": "omitted_not_checked", "source_origin": None, "files": {},
    }
    assert not (root / "formal").exists()
    assert chamber.preflight(root, worker=False) == protocol
    with pytest.raises(ValueError, match="frozen library"):
        chamber.preflight(root, worker=True)


def test_missing_explicit_formal_source_fails_before_construction(chamber, monkeypatch, tmp_path):
    for name in chamber.THREADS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(chamber, "make_brain", lambda *_: pytest.fail("brain before source admission"))
    root = tmp_path / "not-prepared"
    with pytest.raises(FileNotFoundError, match="requested formal source is missing"):
        chamber.prepare(root, formal_source=tmp_path / "absent-formal")
    assert not root.exists()


def test_explicit_formal_snapshots_are_portable_and_source_bound(chamber, monkeypatch, tmp_path):
    for name in chamber.THREADS:
        monkeypatch.setenv(name, "1")
    source = tmp_path / "standalone-source"
    source.mkdir()
    # Synthetic source-byte fixtures, never represented as checked theorems.
    for name in chamber.FORMAL:
        (source / name).write_text("-- source snapshot fixture: " + name + "\n")
    root = tmp_path / "prepared"
    protocol = chamber.prepare(root, formal_source=source)
    evidence = protocol["formal_evidence"]
    assert evidence["status"] == "source_snapshots_bound_not_rechecked"
    assert evidence["source_origin"] == str(source.resolve())
    assert evidence["files"] == {name: chamber.digest(source / name) for name in chamber.FORMAL}
    for name in chamber.FORMAL:
        assert protocol["source_files"]["formal/" + name] == evidence["files"][name]
    # Admission consumes the copied capsule, without relying on the external originals.
    (source / chamber.FORMAL[0]).unlink()
    assert chamber.preflight(root, worker=False) == protocol
    (root / "formal" / chamber.FORMAL[0]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="frozen source/input/initial artifact changed"):
        chamber.preflight(root, worker=False)


def test_formal_cli_selection_is_explicit_and_preparation_only(chamber, monkeypatch, tmp_path):
    selected = []

    def prepare(root, **kwargs):
        selected.append((root, kwargs))
        root.mkdir()
        chamber.atomic_json(root / "protocol.json", {})

    monkeypatch.setattr(chamber, "prepare", prepare)
    root, source = tmp_path / "prepared", tmp_path / "formal"
    chamber.main(["--prepare", "--out", str(root), "--formal-source", str(source)])
    assert selected == [(root.resolve(), {
        "confirmation": False, "reference": None, "formal_source": source,
    })]
    with pytest.raises(SystemExit) as failure:
        chamber.main(["--launch", "--out", str(root), "--formal-source", str(source)])
    assert failure.value.code == 2 and len(selected) == 1


def test_rehashed_formal_claim_cannot_imply_a_proof_recheck(chamber, capsule):
    root, protocol = capsule
    protocol["formal_evidence"]["status"] = "proofs_checked"
    chamber.atomic_json(root / "protocol.json", protocol)
    chamber.atomic_json(root / "admission.json", {
        "protocol_sha256": chamber.digest(root / "protocol.json"),
    })
    with pytest.raises(ValueError, match="formal snapshot status/source binding"):
        chamber.preflight(root, worker=False)


@pytest.mark.parametrize("artifact", ["inputs-0.npz", "source/chamber_inputs.py", "initial-0.npz"])
def test_corrupt_source_inputs_or_initial_rejected_before_learning(chamber, capsule, artifact,
                                                                  monkeypatch):
    root, _ = capsule
    target = root / artifact
    target.write_bytes(target.read_bytes() + b"corrupt")
    calls = []
    monkeypatch.setattr(chamber.Brain, "learn", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="changed"):
        chamber.preflight(root, worker=False)
    assert calls == []


def test_source_guard_rejects_rehashed_changed_caps(chamber, capsule):
    root, protocol = capsule
    protocol["worker_seconds"] = 301
    chamber.atomic_json(root / "protocol.json", protocol)
    chamber.atomic_json(root / "admission.json", {
        "protocol_sha256": chamber.digest(root / "protocol.json"),
    })
    with pytest.raises(ValueError, match="scientific protocol"):
        chamber.preflight(root, worker=False)


def test_hard_failure_preserves_current_and_unrun_denominators(chamber, capsule):
    root, protocol = capsule
    journal = chamber.Journal(root, protocol, time.monotonic())
    journal.summary["lives"][0].update(status="running", actual_batches=7)
    journal.begin("sampled_act", {"life": 0, "batch": 8})
    summary = chamber.failed_process(root, protocol, {"type": "hard_timeout", "seconds": 300})
    assert len(summary["lives"]) == 6
    assert summary["lives"][0]["actual_batches"] == 7
    assert summary["lives"][0]["unknown_current_work"] is True
    assert summary["lives"][1]["status"] == "not_run_process_failure"
    assert not summary["passed"] and not summary["complete"]
    assert summary["current_operation"] == {"kind": "sampled_act", "life": 0, "batch": 8}


def test_limit_arms_unrun_cannot_promote_required_positive_arms(chamber, capsule):
    root, protocol = capsule
    journal = chamber.Journal(root, protocol, time.monotonic())
    for life in journal.summary["lives"]:
        if life["required"]:
            life.update(complete=True, passed=True, endpoints=good_endpoints(),
                        actual_batches=life["planned_batches"])
    result = chamber.finalize(journal)
    assert result["cohort"]["passed"]
    assert result["life_denominator"] == 6 and result["required_denominator"] == 2
    assert result["unrun_batches"] > 0
    assert result["complete"] is False and result["passed"] is False


def test_final_output_bound_includes_receipt_bytes(chamber, capsule, monkeypatch):
    root, protocol = capsule
    journal = chamber.Journal(root, protocol, time.monotonic())
    monkeypatch.setattr(chamber, "tree_bytes", lambda *_: 64 * 1024**2)
    result = chamber.finalize(journal)
    assert result["output_cap_exceeded"] and not result["passed"]
    assert json.loads((root / "summary.json").read_text())["status"] == "output_cap_exceeded"


@pytest.mark.parametrize("reused", [False, True])
def test_completed_store_write_is_counted_when_following_cap_guard_raises(
    chamber, capsule, monkeypatch, reused,
):
    # Assigned-record accounting fixture, not an actual-experience or acquisition claim.
    import gzip
    from types import SimpleNamespace

    root, protocol = capsule
    journal = chamber.Journal(root, protocol, time.monotonic())
    memory = cd.SynapticMemory(np.arange(8), np.arange(8, 10))
    witness = {"key": np.repeat(chamber.inputs.prototypes("orthogonal")[[0]], 8, axis=0),
               "actions": np.zeros(8, dtype=int), "reward": np.ones(8)}
    checks = []

    def cap(*, storage=False):
        checks.append(storage)
        if len(checks) == 2:
            raise chamber.ResourceLimit("test following cap")

    monkeypatch.setattr(journal, "bounds", cap)
    with pytest.raises(chamber.ResourceLimit, match="following cap"):
        chamber.store_write(journal, SimpleNamespace(hippocampus=memory), witness,
                            {"arm": "store-old"}, reused=reused)
    counter = "reused_record_writes" if reused else "factual_donor_record_writes"
    assert checks == [True, True]
    assert memory.writes == 8
    assert journal.totals[counter] == 8
    with gzip.open(root / "journal.jsonl.gz", "rt") as stream:
        record = json.loads(stream.read())
    assert record["body"]["accepted"] is True
    assert record["work"][counter] == 8
    assert journal.summary["unknown_current_work"] is False


def test_refused_store_write_has_no_donor_credit(chamber, capsule, monkeypatch):
    from types import SimpleNamespace

    root, protocol = capsule
    journal = chamber.Journal(root, protocol, time.monotonic())
    memory = cd.SynapticMemory(np.arange(8), np.arange(8, 10))
    witness = {"key": np.repeat(chamber.inputs.prototypes("orthogonal")[[0]], 8, axis=0),
               "actions": np.zeros(8, dtype=int), "reward": np.ones(8)}

    def refuse(*args, **kwargs):
        phase = SimpleNamespace(steps=0, residual=np.ones(8), tolerance=0.003,
                                residual_checks=1, damping_halvings=0, stagnation_checks=0)
        raise chamber.LearningPhaseError("free", {"free": phase})

    monkeypatch.setattr(cd.SynapticMemory, "observe", refuse)
    with pytest.raises(AssertionError, match="refused"):
        chamber.store_write(journal, SimpleNamespace(hippocampus=memory), witness,
                            {"arm": "store-old"})
    assert memory.writes == 0
    assert journal.totals.get("factual_donor_record_writes", 0) == 0
    assert journal.totals["component_store_write_refusals"] == 1


def test_storage_reserve_checked_before_a_new_operation(chamber, capsule):
    root, protocol = capsule
    journal = chamber.Journal(root, protocol, time.monotonic())
    threshold = protocol["output_cap_mib"] * 1024**2 - protocol["output_reserve_bytes"]
    journal.output_bytes = threshold
    with pytest.raises(chamber.ResourceLimit, match="reserve"):
        journal.begin("sampled_act", {"life": 0})
    assert journal.serial == 0
    assert journal.summary["current_operation"] is None


def test_managed_storage_is_exact_without_per_call_tree_scans(chamber, capsule):
    root, protocol = capsule
    journal = chamber.Journal(root, protocol, time.monotonic())
    assert journal.resource_accounting["full_reconciliations"] == 1
    for i in range(20):
        journal.begin("component-accounting", {"fixture": i})
        journal.completed("component-accounting", {"fixture": i}, {"payload": "x" * i})
        assert journal.output_bytes == chamber.tree_bytes(root)
    assert journal.resource_accounting["full_reconciliations"] == 1
    assert journal.resource_accounting["managed_file_size_checks"] == 61


def test_periodic_reconciliation_charges_unmanaged_files(chamber, capsule):
    root, protocol = capsule
    journal = chamber.Journal(root, protocol, time.monotonic())
    (root / "worker.log").write_bytes(b"x" * 512)
    journal.last_storage_reconciliation -= protocol["storage_reconciliation_seconds"]
    journal.bounds(storage=True)
    assert journal.output_bytes == chamber.tree_bytes(root)
    assert journal.resource_accounting["full_reconciliations"] == 2


def test_oversized_final_census_retains_compact_failed_denominator(chamber, capsule):
    root, protocol = capsule
    journal = chamber.Journal(root, protocol, time.monotonic())
    # Synthetic payload size only; no scientific endpoint or graph is executed.
    journal.summary["lives"][0]["endpoints"] = {
        "synthetic-large": "x" * (protocol["output_reserve_bytes"] + 1),
    }
    with pytest.raises(chamber.ResourceLimit, match="complete census"):
        journal.persist(full=True)
    saved = json.loads((root / "summary.json").read_text())
    assert saved["status"] == "census_reserve_exceeded"
    assert saved["complete"] is False and saved["passed"] is False
    assert len(saved["lives"]) == 6
    assert saved["lives"][0]["completed_endpoints"] == ["synthetic-large"]
    assert (root / "summary.json").stat().st_size < protocol["output_reserve_bytes"]


def test_confirmation_recomputes_development_gates_not_mutable_pass_flag(chamber):
    lives = chamber.declared_lives([0])
    for row in lives:
        row.update(complete=True, actual_batches=row["planned_batches"],
                   endpoints=good_endpoints(),
                   seams=[{"passed": True}] * row["planned_pending_seams"])
    protocol = {"schema": chamber.SCHEMA, "mode": "development", "founders": [0],
                "gates": chamber.GATES}
    summary = {"protocol_sha256": "digest", "passed": True, "complete": True,
               "lives": lives, "process_failure": None, "unknown_current_work": False}
    assert chamber.reference_passed(protocol, summary, "digest")
    lives[0]["endpoints"]["revision"]["intact"]["cues"][0]["correct"] = 8
    assert not chamber.reference_passed(protocol, summary, "digest")
