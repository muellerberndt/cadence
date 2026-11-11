"""Native history dataflow and audit guards; no full training campaign in unit tests."""

import copy
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "native_history", Path(__file__).with_name("native_history.py")
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def arrays(brain, path):
    with np.load(brain.save(path), allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}


def equal_arrays(first, second):
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def test_source_configuration_keeps_the_original_graph_and_parameters():
    association = native.make_brain(0, "association")
    sensory = native.make_brain(0, "sensory")
    for key, expected in native.initial_arrays(association).items():
        np.testing.assert_array_equal(native.initial_arrays(sensory)[key], expected)
    assert association.connectome.n == sensory.connectome.n == 69
    assert association.working_memory.source == "association"
    assert len(association.working_memory.glow) == 32
    assert sensory.working_memory.source == "sensory"
    np.testing.assert_array_equal(
        sensory.working_memory.glow, sensory.connectome.populations["prefrontal"][:3]
    )
    assert sensory.working_memory.decay == association.working_memory.decay == 0.2
    assert sensory.working_memory.amplitude == association.working_memory.amplitude == 3
    assert sensory.efference is sensory.hippocampus is None
    assert sensory.learner.config == association.learner.config


def test_history_updates_once_after_each_act_and_reads_only_previous_events(monkeypatch):
    brain = native.make_brain(1)
    memory = brain.working_memory
    original, calls = memory.update, []

    def counted(state):
        calls.append(state)
        original(state)

    monkeypatch.setattr(memory, "update", counted)
    expected = np.zeros((1, 3))
    for event in (1, 0, 0, 0, 1):
        x = native.loop.observation(event)
        np.testing.assert_array_equal(brain.stimulus(x)[:, memory.glow], 3 * expected)
        brain.act(x, greedy=True)
        sensed = brain.basal_ganglia.state.activation[:, brain.sensory_index]
        expected = 0.2 * expected + 0.8 * sensed
        np.testing.assert_array_equal(memory.trace, expected)
        np.testing.assert_array_equal(memory.last, sensed)
    assert len(calls) == 5


def test_refused_act_does_not_advance_history_or_saved_life(tmp_path, monkeypatch):
    brain = native.make_brain(0)
    brain.learner.config = replace(brain.learner.config, free_steps=1, tolerance=1e-20)
    before = arrays(brain, tmp_path / "before.npz")
    calls = []
    monkeypatch.setattr(brain.working_memory, "update", lambda state: calls.append(state))
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.act(native.loop.observation(1), greedy=True)
    assert not calls and not brain.last_settlement["qualified"]
    equal_arrays(before, arrays(brain, tmp_path / "after.npz"))


def test_teacher_labels_do_not_enter_the_sensory_trace():
    first, second = native.make_brain(0), native.make_brain(0)
    works = [native.loop.Work(), native.loop.Work()]
    a = native.loop.hear(first, "nocopy", 1, works[0], label=0)
    b = native.loop.hear(second, "nocopy", 1, works[1], label=1)
    assert a == b and sum(w.lessons for w in works) == 1
    for key in ("trace", "last", "cold"):
        np.testing.assert_array_equal(
            getattr(first.working_memory, key), getattr(second.working_memory, key)
        )
    expected = first.basal_ganglia.state.activation[:, first.sensory_index]
    np.testing.assert_array_equal(first.working_memory.trace, 0.8 * expected)


def test_checkpoint_next_learning_and_imagination_keep_native_history(tmp_path):
    brain = native.make_brain(1)
    work = native.loop.Work()
    for event, label in ((1, 0), (0, 0), (0, 0), (0, 1)):
        native.loop.hear(brain, "nocopy", event, work, label=label)
    io = native.Checkpoints()
    path = io.save(brain, tmp_path / "acquired.npz")
    proof = native.preservation(path, tmp_path, io)
    assert proof["passed"] and proof["imagination_unchanged"]
    assert proof["pending_arrays_equal"] and proof["pending_actions_equal"]
    restored = io.load(tmp_path / "pending.npz")
    assert restored.working_memory.source == "sensory"
    assert restored.working_memory.target == "prefrontal/sensory_history"
    assert restored.basal_ganglia._pending is not None
    assert proof["eligibility_sweeps"] > 0 and len(proof["learning_reports"]) == 8


def test_event_lesion_preserves_constant_history():
    protocol = native.load_protocol()
    protocol["play"]["rows"] = 8
    brain = native.make_brain(0, protocol=protocol)
    result = native.assess(brain, native.loop.Work(), protocol, lesion=True)
    constant_activation = float(np.tanh(0.5))  # sensory drive 1 under the held-source model
    for play in result["plays"]:
        kept = np.asarray(play["constant_trace"]).reshape(-1)
        expected = constant_activation * (1 - 0.2 ** np.arange(1, len(kept) + 1))
        np.testing.assert_allclose(kept, expected, rtol=0, atol=1e-15)
        assert np.all(kept > 0)
    np.testing.assert_array_equal(brain.working_memory.trace[:, :2], 0)


def test_preservation_refusal_cannot_pass_even_when_later_continuation_matches(
    tmp_path, monkeypatch
):
    io = native.Checkpoints()
    path = io.save(native.make_brain(0), tmp_path / "acquired.npz")
    original = native.loop.act
    called = False

    def refuse_first(brain, x, work):
        nonlocal called
        config = brain.learner.config
        if not called:
            brain.learner.config = replace(config, free_steps=1, tolerance=1e-20)
            called = True
        try:
            return original(brain, x, work)
        finally:
            brain.learner.config = config

    monkeypatch.setattr(native.loop, "act", refuse_first)
    proof = native.preservation(path, tmp_path, io)
    assert proof["work"]["refused_acts"] == 1
    assert proof["pending_arrays_equal"] and proof["imagination_unchanged"]
    assert not proof["passed"]


def passing_row(seed, protocol):
    row = {
        "seed": seed,
        "status": "ok",
        "preservation": {"passed": True},
        "restored_outputs_equal": True,
        "learning": {"free_steps": 200, "nudged_steps": 50},
    }
    for arm in ("sensory", "untaught", "lesion", "association", "restored"):
        plays = []
        for phase in range(4):
            truth = native.loop.continuation(4, native.loop.pattern(4, 16, phase), 64).tolist()
            plays.append(native.scores(truth if arm == "sensory" else [0] * 64, phase, protocol))
        row[arm] = {
            "plays": plays,
            "work": {"refused_acts": 0, "refused_lessons": 0},
            "phase_reports": [],
        }
    return row


def test_gates_require_every_prime_and_keep_missing_failed_founders():
    protocol = native.load_protocol()
    rows = [passing_row(seed, protocol) for seed in range(6)]
    assert native.stage_gate(rows, list(range(6)), 5, protocol)["passed"]
    damaged = copy.deepcopy(rows[0])
    damaged["sensory"]["plays"][2]["answers"] = [0] * 64
    assert not native.founder_passes(damaged, protocol)  # high mean cannot hide one bad prime
    damaged = copy.deepcopy(rows[0])
    damaged["sensory"]["phase_reports"] = [{"nudged_steps": 50}]
    assert not native.founder_passes(damaged, protocol)
    damaged = copy.deepcopy(rows[0])
    damaged["restored"]["events"] = [{"kind": "act", "steps": 200}]
    assert not native.founder_passes(damaged, protocol)
    damaged = copy.deepcopy(rows[0])
    damaged["lesion"]["plays"] = []
    assert not native.founder_passes(damaged, protocol)
    damaged = copy.deepcopy(rows[0])
    damaged["association"] = {"status": "error"}
    assert native.founder_passes(damaged, protocol)  # old-recipe failure is not a candidate veto
    one_bad_prime = copy.deepcopy(rows)
    one_bad_prime[0]["sensory"]["plays"][2]["answers"] = [0] * 64
    assert native.stage_gate(one_bad_prime, list(range(6)), 5, protocol)["passed"]
    assert not native.stage_gate(rows[:5], list(range(6)), 5, protocol)["passed"]
    assert not native.stage_gate(rows[:5] + [rows[0]], list(range(6)), 5, protocol)["passed"]
    rows[5] = {"seed": 5, "status": "error", "passed": False}
    assert not native.stage_gate(rows, list(range(6)), 5, protocol)["passed"]


def test_cli_requires_a_new_directory(tmp_path):
    with pytest.raises(FileExistsError):
        native.main(["--screen", "--out", str(tmp_path)])
    assert "does not close issue 140" in native.load_protocol()["scope"]


@pytest.fixture(scope="module")
def tiny_receipt(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-audit")
    protocol = native.load_protocol()
    protocol["teaching"].update(rows=8, passes=1)
    protocol["play"].update(prime=4, rows=8)
    path = directory / "protocol-native-history.json"
    path.write_text(json.dumps(protocol))
    output = directory / "receipt"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native, "PROTOCOL", path)
        native.main(["--screen", "--out", str(output)])
    assert native.verify(output)[0]
    return output


def rewrite_receipt(directory, transform):
    receipt = native.Receipt.read(directory / "summary.json")
    transform(receipt.body)
    files = [(item["path"], directory / item["path"]) for item in receipt.source["files"]]
    native.Receipt.build(native.SCHEMA, receipt.body, files).write(directory / "summary.json")


@pytest.mark.parametrize("mutation", ["watch", "teaching", "baseline", "checkpoint", "source"])
def test_verifier_rejects_resealed_but_inconsistent_evidence(tiny_receipt, tmp_path, mutation):
    target = tmp_path / "receipt"
    shutil.copytree(tiny_receipt, target)
    if mutation == "watch":
        rewrite_receipt(
            target,
            lambda body: body["development"][0]["sensory"]["watches"][0].update(agreement=-1.0),
        )
    elif mutation == "teaching":
        rewrite_receipt(
            target,
            lambda body: body["development"][0]["sensory"]["teaching"].update(
                last_pass_agreement=-1.0
            ),
        )
    elif mutation == "baseline":
        rewrite_receipt(
            target, lambda body: body["baselines"]["0"]["random"][0]["own"].update(agreement=-1.0)
        )
    elif mutation == "checkpoint":
        path = target / "development-0/after-imagine.npz"
        with np.load(path, allow_pickle=False) as saved:
            changed = {k: saved[k].copy() for k in saved.files}
        changed["memories/working_memory/trace"] += 0.1
        np.savez(path, **changed)
        rewrite_receipt(
            target,
            lambda body: body["artifacts"].update(
                {path.relative_to(target).as_posix(): native.sha256(path)}
            ),
        )
    else:
        receipt = native.Receipt.read(target / "summary.json")
        files = [(item["path"], target / item["path"]) for item in receipt.source["files"]][1:]
        native.Receipt.build(native.SCHEMA, receipt.body, files).write(target / "summary.json")
    assert not native.verify(target)[0]


def test_independent_baseline_failure_keeps_candidate_and_checkpoint(tmp_path, monkeypatch):
    protocol = native.load_protocol()
    protocol["teaching"].update(rows=8, passes=1)
    protocol["play"].update(prime=4, rows=8)
    assess = native.assess

    def fail_baseline(brain, *args, **kwargs):
        if brain.working_memory.source == "association":
            raise RuntimeError("injected old-recipe failure")
        return assess(brain, *args, **kwargs)

    monkeypatch.setattr(native, "assess", fail_baseline)
    row = native.run_founder(0, tmp_path / "founder", protocol)
    assert row["status"] == "ok" and row["candidate_work_complete"]
    assert not row["work_complete"] and row["preservation"]["passed"]
    assert row["association"]["status"] == "error"
    assert (tmp_path / "founder/association-failed.npz").is_file()
    assert len(row["sensory"]["plays"]) == 4


def test_unexpected_failure_keeps_its_live_context(tmp_path, monkeypatch):
    protocol = native.load_protocol()
    protocol["teaching"].update(rows=8, passes=1)
    protocol["play"].update(prime=4, rows=8)

    def fail(*args, **kwargs):
        raise RuntimeError("injected observation failure")

    monkeypatch.setattr(native, "assess", fail)
    row = native.run_founder(0, tmp_path / "founder", protocol)
    assert row["status"] == "error" and not row["passed"]
    assert row["failure_context"]["stage"] == "untaught"
    assert (tmp_path / "founder/failed-context.npz").is_file()
