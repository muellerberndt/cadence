"""Guards of the finite recall chamber: protocol, arms, forks, seams and the receipt."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load_script("finite_horizon_inputs")
chamber = load_script("finite_horizon")


@pytest.fixture(autouse=True)
def quiet():
    warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def protocol():
    return chamber.load_protocol()


def test_the_protocol_declares_the_reviewed_recipe_and_fresh_founders(protocol):
    declared, digest = protocol
    assert digest == hashlib.sha256(chamber.PROTOCOL_PATH.read_bytes()).hexdigest()
    assert declared["schema"] == chamber.SCHEMA
    assert declared["seeds"]["founders"] == [301, 302, 303]
    brain = declared["brain"]
    assert (brain["trace_decay"], brain["trace_amplitude"]) == (0.8, 1.0)
    assert (brain["default_decay"], brain["default_amplitude"]) == (
        0.2,
        3.0,
    )  # the composed defaults
    assert declared["learning"]["qualified"] is True and declared["learning"]["nudged_steps"] == 128
    assert declared["arms"] == list(chamber.ARMS)
    assert declared["training"]["repeats"] == 16 and declared["evaluation"]["repeats"] == 24
    assert declared["caps"] == {"seconds": 900, "bytes": 160 * 1024 * 1024}
    assert declared["gates"]["minimum_horizon"] == 1


def test_the_second_protocol_changes_only_the_selected_amplitude_budget_and_founders(protocol):
    declared, _ = protocol
    second = json.loads((ROOT / "protocol-finite-2.json").read_text())
    assert second["seeds"]["founders"] == [304, 305, 306]
    assert second["brain"]["trace_amplitude"] == 0.3 and second["brain"]["trace_decay"] == 0.8
    assert second["training"]["repeats"] == 32 and second["training"]["episodes"] == 384
    assert "selection" in second
    same = {k: v for k, v in second.items() if k not in ("seeds", "brain", "training", "selection")}
    assert same == {k: v for k, v in declared.items() if k not in ("seeds", "brain", "training")}
    assert {k: v for k, v in second["brain"].items() if k != "trace_amplitude"} == {
        k: v for k, v in declared["brain"].items() if k != "trace_amplitude"
    }
    assert second["gates"] == declared["gates"] and second["caps"] == declared["caps"]


def test_the_third_protocol_adds_only_the_surprise_rule_and_fresh_founders():
    second = json.loads((ROOT / "protocol-finite-2.json").read_text())
    third = json.loads((ROOT / "protocol-finite-3.json").read_text())
    assert third["seeds"]["founders"] == [307, 308, 309]
    assert third["training"]["rule"] == "surprise" and "rule" not in second["training"]
    assert {k: v for k, v in third["training"].items() if k not in ("rule", "lesson")} == {
        k: v for k, v in second["training"].items() if k != "lesson"
    }
    assert {k: v for k, v in third.items() if k not in ("seeds", "training", "selection")} == {
        k: v for k, v in second.items() if k not in ("seeds", "training", "selection")
    }


def test_the_arms_differ_only_in_their_declared_trace_genes(protocol):
    declared, _ = protocol
    vanished = chamber.brain_for("vanished", declared, 301)
    default = chamber.brain_for("default", declared, 301)
    history = chamber.brain_for("history", declared, 301)
    assert (vanished.working_memory.decay, vanished.working_memory.amplitude) == (0.8, 1.0)
    assert (default.working_memory.decay, default.working_memory.amplitude) == (0.2, 3.0)
    np.testing.assert_array_equal(vanished.brain.efficacy, history.brain.efficacy)
    np.testing.assert_array_equal(vanished.brain.efficacy, default.brain.efficacy)
    assert vanished.hippocampus is None and vanished.efference is None
    assert vanished.learner.config.qualified and vanished.learner.config.nudged_steps == 128


def test_the_surprise_rule_teaches_only_the_rows_answered_wrong(protocol, tmp_path):
    declared, _ = protocol
    surprise = json.loads(json.dumps(declared))
    surprise["training"]["rule"] = "surprise"
    frozen = chamber.freeze_small(tmp_path / "episodes.npz", 0, {"train": 2, "test": 1})
    began = chamber.time.time()
    seen: list[tuple[int, int]] = []
    real_teach = chamber.Work.teach

    def spy(self, brain, x, labels, *, drive=None):
        seen.append((len(labels), 0 if drive is None else len(drive)))
        return real_teach(self, brain, x, labels, drive=drive)

    chamber.Work.teach = spy
    try:
        every = chamber.train_arm(
            "vanished",
            chamber.brain_for("vanished", declared, 0),
            frozen,
            declared,
            began,
            tmp_path,
        )
        taught_every = list(seen)
        seen.clear()
        gated = chamber.train_arm(
            "vanished",
            chamber.brain_for("vanished", surprise, 0),
            frozen,
            surprise,
            began,
            tmp_path,
        )
    finally:
        chamber.Work.teach = real_teach
    assert every["rule"] == "every" and gated["rule"] == "surprise"
    assert len(taught_every) == every["episodes"] and all(
        n == inputs.STREAMS and d == 0 for n, d in taught_every
    )
    # a lesson under the rule carries only the wrong rows, on the drive their act read
    assert len(seen) <= gated["episodes"] and all(
        0 < n <= inputs.STREAMS and d == n for n, d in seen
    )
    assert gated["work"]["teacher_presentations"] < every["work"]["teacher_presentations"]
    with pytest.raises(ValueError):
        bad = json.loads(json.dumps(declared))
        bad["training"]["rule"] = "always"
        chamber.train_arm(
            "vanished", chamber.brain_for("vanished", bad, 0), frozen, bad, began, tmp_path
        )


def test_the_trace_audit_accepts_the_source_recurrence_and_rejects_a_skipped_update():
    before = {"trace": np.zeros((2, 3)), "last": np.zeros((2, 3)), "cold": np.array([True, True])}
    h = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    after = {"trace": 0.2 * h, "last": h.copy(), "cold": np.array([False, False])}
    assert chamber.audit_trace(before, after, h, 0.8) <= 1e-12
    with pytest.raises(ValueError):
        chamber.audit_trace(before, {**after, "trace": 0.5 * h}, h, 0.8)
    with pytest.raises(ValueError):
        chamber.audit_trace(before, {**after, "cold": np.array([True, False])}, h, 0.8)
    after_default = {"trace": 0.8 * h, "last": h.copy(), "cold": np.array([False, False])}
    assert chamber.audit_trace(before, after_default, h, 0.2) <= 1e-12


def test_a_small_run_scores_forks_seams_and_timing_and_its_receipt_verifies(protocol, tmp_path):
    out = tmp_path / "run"
    code = chamber.main(["--out", str(out), "--seeds", "0", "--repeats", "1", "1"])
    assert code == 0
    valid, reason = chamber.verify(out)
    assert valid, reason
    body = json.loads((out / "summary.json").read_text())["body"]
    assert body["frozen_protocol"] is False and body["declaration"]["repeats"] == {
        "train": 1,
        "test": 1,
    }
    founder = body["founders"][0]
    assert set(founder["arms"]) == set(chamber.ARMS)
    vanished = founder["arms"]["vanished"]
    assert vanished["training"]["episodes"] == len(inputs.TRAIN_CONDITIONS)
    assert len(vanished["trials"]) == len(inputs.TEST_CONDITIONS)
    for trial in vanished["trials"]:
        assert set(trial["branches"]) == set(chamber.CONTROLS)
        assert len(trial["labels"]) == inputs.STREAMS and len(trial["random"]) == inputs.STREAMS
    assert set(vanished["seams"]) == {c.name for c in inputs.TEST_CONDITIONS}
    replacement = vanished["seams"]["replacement-1"]["seams"]
    assert set(replacement) == {"after_cue", "before_replacement", "before_query"}
    assert all(s["answers_equal"] and s["saved_arrays_equal"] for s in replacement.values())
    assert vanished["seams"]["clean-1"]["imagine_leaves_checkpoint"]
    assert vanished["seams"]["clean-1"]["refused_act_leaves_checkpoint"]
    assert vanished["timing"]["equal_arrays"] and vanished["work"]["trace_audits"] > 0
    assert vanished["work"]["trace_audit_max_error"] <= 1e-12
    scores = founder["scores"]
    assert set(scores) == set(chamber.ARMS)
    assert scores["random"]["clean-0"]["answer"]["planned"] == inputs.STREAMS
    assert scores["vanished"]["clean-1"]["shuffled_vs_transplanted"]["planned"] == inputs.STREAMS
    assert founder["history_initial_equal"] is True
    gates = founder["gates"]
    assert set(gates["conditions"]) == {c.name for c in inputs.TEST_CONDITIONS}
    assert -1 <= gates["horizon"] <= 2 and isinstance(gates["closure"], bool)
    assert body["declaration"]["instrument_revision"] == 3
    assert (out / "summary.json").stat().st_size < 1024 * 1024
    assert sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) < 32 * 1024 * 1024
    assert body["unmet_protocol_obligations"] == [] and not body["closure"]
    assert founder["hard_guard"]["completed"] and founder["hard_guard"]["reaped"]
    assert vanished["work"]["retained_trace_records"] == vanished["work"]["trace_audits"]
    assert (
        vanished["work"]["trace_audits"]
        == vanished["work"]["action_attempts"] - vanished["work"]["action_refusals"]
    )
    event_fork = vanished["timing"]["matched_elapsed_events"]
    assert [b["events"] for b in event_fork["branches"]] == [1, 2]
    assert [b["elapsed"] for b in event_fork["branches"]] == [2.0, 2.0]
    assert event_fork["trace_distance"] > 0
    assert len(vanished["trials"][0]["diagnostics"]["paired_trace_distance"]) == 4
    assert (
        body["founders"][0]["arms"]["history"]["trials"][0]["history_traffic"]["buffer_bytes"]
        == 1088
    )
    assert vanished["work"]["imagined_phases"] > 0
    assert vanished["work"]["expected_refusals"] == len(inputs.TEST_CONDITIONS)
    assert vanished["timing"]["work"][0]["action_attempts"] > 0
    assert all(s["reports_equal"] for s in replacement.values())
    # the receipt refuses edited scores
    stored = json.loads((out / "summary.json").read_text())
    stored["body"]["founders"][0]["gates"]["horizon"] = 2
    (out / "summary.json").write_text(json.dumps(stored))
    assert not chamber.verify(out)[0]
    # A new canonical signature still cannot legitimize unsupported arithmetic.
    sources = [(entry["path"], out / entry["path"]) for entry in stored["source"]["files"]]
    chamber.Receipt.build(chamber.SCHEMA, stored["body"], sources).write(out / "summary.json")
    assert not chamber.verify(out)[0]
    stored["body"]["founders"][0]["gates"] = gates
    stored["body"]["founders"][0]["scores"]["vanished"]["clean-0"]["intact"]["correct"] += 1
    chamber.Receipt.build(chamber.SCHEMA, stored["body"], sources).write(out / "summary.json")
    valid, reason = chamber.verify(out)
    assert not valid and "trial scores" in reason
    # Re-signing a changed raw transition cannot manufacture a valid recurrence.
    stored["body"] = body
    chunk = next((out / "founder-0/vanished/training-traces").glob("*.npz"))
    original_chunk = chunk.read_bytes()
    with np.load(chunk, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["after_trace"][0] += 0.1
    np.savez_compressed(chunk, **arrays)
    stored["body"]["artifacts"][chunk.relative_to(out).as_posix()] = chamber.sha256(chunk)
    chamber.Receipt.build(chamber.SCHEMA, stored["body"], sources).write(out / "summary.json")
    valid, reason = chamber.verify(out)
    assert not valid and "accepted-event update" in reason
    chunk.write_bytes(original_chunk)
    body["artifacts"][chunk.relative_to(out).as_posix()] = chamber.sha256(chunk)
    # A diagnostic vector changed behind a re-signed hash still has to match its margins.
    diagnostic = body["founders"][0]["arms"]["vanished"]["trials"][0]["diagnostics"]
    path = out / "founder-0/vanished" / diagnostic["raw"]["path"]
    with np.load(path, allow_pickle=False) as archive:
        vectors = {k: archive[k] for k in archive.files}
    vectors["motor"][0, 0] += 0.5
    np.savez_compressed(path, **vectors)
    diagnostic["raw"]["sha256"] = chamber.sha256(path)
    body["artifacts"][path.relative_to(out).as_posix()] = chamber.sha256(path)
    chamber.Receipt.build(chamber.SCHEMA, body, sources).write(out / "summary.json")
    valid, reason = chamber.verify(out)
    assert not valid and "query diagnostic arithmetic" in reason


@pytest.mark.parametrize("bad", [["--seeds", "0", "0"], ["--repeats", "0", "1"]])
def test_invalid_cli_is_rejected_before_creating_an_attempt(tmp_path, bad):
    out = tmp_path / "never"
    with pytest.raises(SystemExit):
        chamber.main(["--out", str(out), *bad])
    assert not out.exists()


def test_declared_repeat_budget_is_used_without_a_cli_override(tmp_path, monkeypatch):
    declared, _ = chamber.load_protocol(ROOT / "protocol-finite-2.json")
    observed = []

    def inspect(arm, brain, frozen, protocol, began, directory, *, work, progress):
        observed.append((len(frozen["train/condition"]), len(frozen["test/condition"])))
        progress.update(
            episodes=len(frozen["train/condition"]), completed=False, work=work.summary()
        )
        return progress

    monkeypatch.setattr(chamber, "train_arm", inspect)
    result = chamber.run_founder(304, declared, tmp_path / "declared", None)
    assert observed == [(384, 312)] * 3
    assert not result["gates"]["closure"]


def test_every_rule_preserves_the_original_teach_then_act_life(protocol, tmp_path):
    declared = protocol[0]
    frozen = chamber.freeze_small(tmp_path / "episodes.npz", 0, {"train": 1, "test": 1})
    current = chamber.brain_for("vanished", declared, 0)
    reference = chamber.brain_for("vanished", declared, 0)
    result = chamber.train_arm("vanished", current, frozen, declared, chamber.time.time(), tmp_path)
    for index in range(len(frozen["train/condition"])):
        episode = chamber.episode_arrays(frozen, "train", index)
        for event, observation in enumerate(episode["observations"]):
            if event == len(episode["observations"]) - 1:
                reference.learner.step(reference.stimulus(observation), episode["labels"])
            reference.act(observation, greedy=True)
    assert result["completed_episodes"] == result["episodes"] == 12
    assert chamber.same_arrays(
        current.save(tmp_path / "current.npz"), reference.save(tmp_path / "reference.npz")
    )


def test_cap_keeps_completed_work_and_the_full_random_denominator(protocol, tmp_path, monkeypatch):
    actual = chamber.Work.act
    attempts = 0

    def capped(self, *args, **kwargs):
        nonlocal attempts
        answer = actual(self, *args, **kwargs)
        attempts += 1
        if attempts == 2:
            raise chamber.Capped("injected after a completed call")
        return answer

    monkeypatch.setattr(chamber.Work, "act", capped)
    result = chamber.run_founder(0, protocol[0], tmp_path / "capped", {"train": 1, "test": 1})
    assert result["capped"] and not result["gates"]["closure"]
    assert result["arms"]["vanished"]["training"]["work"]["action_attempts"] == 2
    assert len(result["arms"]["random"]["trials"]) == len(inputs.TEST_CONDITIONS)
    assert result["scores"]["history"]["clean-0"]["answer"]["planned"] == inputs.STREAMS


def test_a_missing_control_cannot_improve_the_measured_horizon():
    receipt = json.loads(
        gzip.decompress((ROOT / "results/finite-3-2026-10-08.json.gz").read_bytes())
    )
    founder = receipt["body"]["founders"][2]
    declared = receipt["body"]["protocol"]
    assert chamber.gates_for(founder, declared)["horizon"] == 1
    founder["scores"]["vanished"]["clean-1"]["erased"]["attempted"] = 0
    assert chamber.gates_for(founder, declared)["horizon"] == 0


def test_hard_deadline_reaps_an_artificial_worker_and_keeps_all_denominators(protocol, tmp_path):
    guard = chamber.guarded_process(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"],
        tmp_path / "worker.log",
        0.25,
    )
    assert guard["timed_out"] and guard["reaped"] and not guard["completed"]
    assert guard["elapsed_seconds"] < 10
    with pytest.raises(ProcessLookupError):
        os.kill(guard["pid"], 0)
    founder = chamber.recover_founder(0, protocol[0], tmp_path / "founder", {"train": 1, "test": 1})
    founder["hard_guard"] = guard
    assert founder["capped"] and founder["incomplete_work"]
    assert set(founder["arms"]) == set(chamber.ARMS)
    assert not chamber.gates_for(founder, protocol[0])["closure"]
    for arm, conditions in founder["scores"].items():
        for condition in conditions.values():
            branch = "answer" if arm in ("history", "random") else "intact"
            assert condition[branch]["planned"] == inputs.STREAMS
            if arm != "random":
                assert condition[branch]["correct"] == condition[branch]["attempted"] == 0


def test_query_diagnostics_and_history_traffic_use_observations_not_answers():
    trace = np.array([[1.0, 0.0], [0.0, 1.0]])
    neural = trace * 2
    motor = np.array([[0.7, 0.2], [0.1, 0.8]])
    result = chamber.diagnostics_from_arrays(trace, neural, motor, np.array([0, 1]))
    np.testing.assert_allclose(result["paired_trace_distance"], [np.sqrt(2)])
    np.testing.assert_allclose(result["paired_neural_distance"], [2 * np.sqrt(2)])
    np.testing.assert_allclose(result["motor_margin"], [0.5, 0.7])
    episode = inputs.make_episode(np.random.default_rng(0), inputs.TEST_CONDITIONS[0])
    traffic = chamber.history_traffic(episode.observations)
    assert traffic["buffer_bytes"] == 1088
    assert traffic["query_bytes_transported"] == 1024
    assert traffic["payload_bytes_read"] == traffic["payload_bytes_written"] == 256


def test_packed_trace_records_preserve_exact_values_shapes_and_dtypes(tmp_path):
    records = []
    for index in range(7):
        record = {
            name: np.full((2, 3), index / 10, dtype=np.float64)
            for name in chamber.TRACE_FIELDS
            if name not in ("before_cold", "after_cold", "decay")
        }
        record.update(
            before_cold=np.array([True, False]),
            after_cold=np.array([False, False]),
            decay=np.asarray(0.8),
        )
        records.append(record)
    packed = chamber.pack_traces(records)
    path = tmp_path / "packed.npz"
    np.savez_compressed(path, **packed)
    with np.load(path, allow_pickle=False) as archive:
        restored = chamber.unpack_traces(archive, len(records))
        with pytest.raises(ValueError, match="event census"):
            chamber.unpack_traces(archive, len(records) + 1)
    for before, after in zip(records, restored, strict=True):
        for name in chamber.TRACE_FIELDS:
            assert before[name].dtype == after[name].dtype
            assert before[name].shape == after[name].shape
            assert before[name].tobytes() == after[name].tobytes()
    records[-1]["activation"] = records[-1]["activation"].astype(np.float32)
    with pytest.raises(ValueError, match="promote dtypes"):
        chamber.pack_traces(records)


HISTORY_PROTOCOL = ROOT / "protocol-finite-history.json"


def test_the_history_protocol_adds_the_heard_history_to_the_composed_trace():
    third = json.loads((ROOT / "protocol-finite-3.json").read_text())
    heard, _ = chamber.load_protocol(HISTORY_PROTOCOL)
    kept = ("schema", "reviewed_protocol", "learning", "arms", "training", "evaluation", "caps")
    assert {k: heard[k] for k in kept} == {k: third[k] for k in kept}
    assert heard["gates"] == third["gates"] and heard["seeds"]["founders"] == [0, 1, 304, 305, 306]
    vanished = chamber.brain_for("vanished", heard, 0)
    default = chamber.brain_for("default", heard, 0)
    history = chamber.brain_for("history", heard, 0)
    assert vanished.history.decays == (0.2, 0.8, 0.95) and default.history is None
    assert len(vanished.connectome.populations["history"]) == 3 * inputs.INPUTS
    for brain in (vanished, default):
        assert (brain.working_memory.decay, brain.working_memory.amplitude) == (0.0, 3.0)
        assert brain.describe()["initialization"]["composition"]["working_memory_scale"] == 1.2
    np.testing.assert_array_equal(vanished.brain.efficacy, history.brain.efficacy)
    # the founder part of the vanished brain is the default brain: the history is appended
    added = np.isin(vanished.connectome.pre, vanished.connectome.populations["history"])
    np.testing.assert_array_equal(vanished.brain.efficacy[~added], default.brain.efficacy)
    np.testing.assert_array_equal(vanished.connectome.post[~added], default.connectome.post)


def test_the_forks_of_a_heard_history_erase_and_transplant_that_history(tmp_path):
    heard, _ = chamber.load_protocol(HISTORY_PROTOCOL)
    brain = chamber.brain_for("vanished", heard, 0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        brain.act(rng.integers(0, 2, size=(inputs.STREAMS, inputs.INPUTS)).astype(float))
    checkpoint = brain.save(tmp_path / "query.npz")
    permutation = np.arange(inputs.STREAMS) ^ 1
    seen = {}

    class Spy(chamber.Work):
        def act(self, branch, x, **kwargs):
            seen[len(seen)] = (branch.history.trace.copy(), branch.working_memory.trace.copy())
            return super().act(branch, x, **kwargs)

    query = np.zeros((inputs.STREAMS, inputs.INPUTS))
    query[:, inputs.QUERY.start] = 1.0
    chamber.fork_answers(checkpoint, query, permutation, Spy())
    (erased, erased_trace), (shuffled, shuffled_trace), (reset, _) = seen.values()
    assert not erased.any() and not reset.size
    np.testing.assert_array_equal(erased_trace, brain.working_memory.trace)
    np.testing.assert_array_equal(shuffled, brain.history.trace[permutation])
    np.testing.assert_array_equal(shuffled_trace, brain.working_memory.trace)


def test_a_small_history_run_verifies(tmp_path):
    out = tmp_path / "run"
    code = chamber.main(
        ["--out", str(out), "--protocol", str(HISTORY_PROTOCOL), "--seeds", "0",
         "--repeats", "1", "1"]
    )
    assert code == 0
    valid, reason = chamber.verify(out)
    assert valid, reason
    founder = json.loads((out / "summary.json").read_text())["body"]["founders"][0]
    assert founder["history_initial_equal"] is True
    assert founder["arms"]["vanished"]["timing"]["equal_arrays"]
