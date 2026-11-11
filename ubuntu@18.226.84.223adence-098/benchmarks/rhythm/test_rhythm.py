"""Steady-rhythm chamber guards: frozen inputs, protocol hash, scoring, custody, a smoke run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
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


inputs = load_script("rhythm_inputs")
load_script("timing_acceptance")
chamber = load_script("steady_rhythm")


@pytest.fixture(scope="module")
def protocol():
    return inputs.load_protocol()


def test_protocol_is_frozen_with_fresh_confirmation_seeds_and_declared_recipes(protocol):
    declared, digest = protocol
    assert digest == hashlib.sha256(inputs.PROTOCOL_PATH.read_bytes()).hexdigest()
    assert declared["schema"] == chamber.SCHEMA
    assert len(declared["seeds"]["confirmation"]) == 5
    assert not set(declared["seeds"]["confirmation"]) & set(declared["seeds"]["development"])
    assert tuple(declared["controls"]) == chamber.CONTROLS
    selected, default = declared["recipes"]["selected"], declared["recipes"]["compose_default"]
    assert selected["learning"] == default["learning"]
    assert default["working_memory_decay"] == 0.2 and default["working_memory_amplitude"] == 3.0
    assert selected["working_memory_amplitude"] == 3.0
    assert selected["learning"]["qualified"] is False and selected["learning"]["nudged_steps"] == 12
    assert declared["cadence"]["events"] <= declared["window"]["events"]
    assert declared["cadence"]["disturbed_slot"] < declared["cadence"]["events"] - 1


def test_frozen_inputs_are_deterministic_cued_and_constant_after_the_cue(protocol, tmp_path):
    declared, _ = protocol
    first = inputs.freeze_inputs(tmp_path / "a.npz", seed=7, protocol=declared)
    np.random.default_rng(3).normal(size=100)
    second = inputs.freeze_inputs(tmp_path / "b.npz", seed=7, protocol=declared)
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)
        assert not first[name].flags.writeable
    other = inputs.freeze_inputs(tmp_path / "c.npz", seed=8, protocol=declared)
    assert not np.array_equal(first["teach/cues"], other["teach/cues"])
    teaching = first["teach/observations"]
    kinds = first["teach/kinds"]
    assert teaching.shape == (
        declared["teaching"]["bouts"] * declared["teaching"]["events_per_bout"],
        inputs.ROWS,
        inputs.INPUTS,
    )
    drive_events = teaching[kinds == inputs.KIND_DRIVE]
    assert np.all(drive_events[:, :, inputs.DRIVE] == 1.0)
    assert not drive_events[:, :, inputs.DISTRACTOR :].any()
    for cues in first["teach/cues"]:
        assert sorted(cues) == [0, 0, 1, 1]
    cue_events = teaching[kinds == inputs.KIND_CUE]
    assert np.all(cue_events[:, :, inputs.CUE_A :].sum(axis=2) == 1.0)
    window = first["window/observations"]
    assert window.shape[0] == 1 + declared["window"]["lead"] + declared["window"]["events"]
    assert np.all(window[1:] == window[1:2])
    permutation = first["shuffle/permutation"]
    assert np.all(first["window/cues"][permutation] != first["window/cues"])
    pause = first["disturbance/pause2/observations"]
    pre = declared["disturbances"]["pre"]
    assert not pause[pre : pre + 2].any() and pause[pre + 2 :, :, inputs.DRIVE].all()
    distractor = first["disturbance/distractor/observations"][pre]
    assert distractor[:, inputs.DISTRACTOR].all() and not distractor[:, inputs.DRIVE].any()


def test_slot_schedules_shift_external_phase_without_touching_observations(protocol, tmp_path):
    declared, _ = protocol
    frozen = inputs.freeze_inputs(tmp_path / "a.npz", seed=1, protocol=declared)
    events, slot = declared["cadence"]["events"], declared["cadence"]["disturbed_slot"]
    regular = frozen["cadence/regular/slot"]
    assert list(regular) == list(range(events))
    extra = frozen["cadence/extra/slot"]
    assert len(extra) == events + 1 and list(extra[slot : slot + 2]) == [slot, slot]
    skipped = frozen["cadence/skipped/slot"]
    assert len(skipped) == events - 1 and slot not in skipped
    np.testing.assert_array_equal(
        frozen["cadence/regular/due_ms"], regular * declared["event"]["cadence_ms"]
    )
    for variant in declared["event"]["cadence_variants_ms"]:
        np.testing.assert_array_equal(frozen[f"cadence/regular{variant}/due_ms"], regular * variant)


def test_scoring_counts_alternation_agreement_period_and_refusals():
    anchor = np.array([0, 1, 0, 1])
    perfect = inputs.ideal_alternation(anchor, 16)
    score = chamber.score_window(perfect, anchor, 8)
    assert score["alternation_rate"] == 1.0 and score["agreement"] == 1.0
    assert score["period"] == 2.0 and score["repeats"] == 0 and score["refusals"] == 0
    assert score["blocks"] == [1.0, 1.0]
    held = np.tile(anchor, (16, 1))
    score = chamber.score_window(held, anchor, 8)
    # A held action meets the ideal alternation at every second event.
    assert score["alternation_rate"] == 0.0 and score["agreement"] == 0.5
    assert score["repeats"] == 60 and score["period"] == 1.0
    slipped = perfect.copy()
    slipped[8:] = 1 - slipped[8:]  # one repeat per row at event 8, then perfect alternation
    score = chamber.score_window(slipped, anchor, 8)
    assert score["repeats"] == 4 and score["agreement"] == 0.5
    assert score["blocks"] == [1.0, 1.0]
    refused = perfect.copy()
    refused[3] = chamber.REFUSED
    score = chamber.score_window(refused, anchor, 8)
    assert score["refusals"] == 4 and score["rows"][0]["refusals"] == 1
    assert score["agreement"] == 15 / 16
    assert chamber.recovery_events(perfect) == [0, 0, 0, 0]
    assert chamber.recovery_events(slipped) == [8, 8, 8, 8]
    assert chamber.recovery_events(held) == [None] * 4
    np.testing.assert_array_equal(chamber.last_executed(refused[2:4], anchor), perfect[2])


def test_flipflop_control_learns_alternation_from_its_own_previous_output():
    flip = chamber.FlipFlop(0.5)
    x = inputs.observation(inputs.KIND_DRIVE)
    for _ in range(8):
        labels = 1 - flip.previous
        flip.teach(flip.features(x), labels)
        flip.act(x)
    actions = np.stack([flip.act(x) for _ in range(10)])
    assert np.all(actions[1:] != actions[:-1])
    assert np.all(flip.act(inputs.observation(inputs.KIND_PAUSE)) != actions[-1])


def test_refused_act_is_a_missed_action_that_preserves_state(protocol, tmp_path):
    declared, _ = protocol
    brain = chamber.make_brain(0, declared)
    x = inputs.observation(inputs.KIND_DRIVE)
    work = chamber.Work()
    assert work.act(brain, x) is not None
    before = brain.save(tmp_path / "before.npz")
    from dataclasses import replace

    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=0)
    assert work.act(brain, x) is None
    assert work.action_refusals == 1 and work.action_attempts == 2
    # The configuration is serialized; restore it so the comparison covers state alone.
    brain.learner.config = config
    assert chamber.same_saved_arrays(before, brain.save(tmp_path / "after.npz"))


def test_controls_fork_the_probe_and_the_live_life_is_unchanged(protocol, tmp_path):
    declared, _ = protocol
    brain = chamber.make_brain(0, declared)
    x = inputs.observation(inputs.KIND_DRIVE)
    chamber.Work().act(brain, x)
    chamber.Work().act(brain, x)
    probe = brain.save(tmp_path / "probe.npz")
    permutation = np.array([2, 3, 0, 1])
    for prepare in (chamber.erase_trace, lambda b: chamber.transplant_trace(b, permutation)):
        branch = chamber.Brain.load(probe)
        prepare(branch)
        chamber.Work().act(branch, x)
    assert chamber.same_saved_arrays(probe, brain.save(tmp_path / "after.npz"))
    branch = chamber.Brain.load(probe)
    chamber.erase_trace(branch)
    assert not branch.working_memory.trace.any() and branch.working_memory.cold.all()


def test_smoke_run_preserves_custody_controls_and_denominators(protocol, tmp_path):
    output = tmp_path / "smoke"
    code = chamber.main(
        [
            "--out",
            str(output),
            "--seeds",
            "0",
            "--arms",
            "every",
            "--recipes",
            "selected",
            "--bouts",
            "1",
            "--events-per-bout",
            "3",
            "--window",
            "8",
            "--post",
            "4",
            "--no-cadence",
            "--time-cap",
            "600",
        ]
    )
    assert code == 0
    assert chamber.verify(output)[0]
    body = json.loads((output / "summary.json").read_text())["body"]
    assert body["frozen_protocol"] is False
    assert body["protocol_sha256"] == protocol[1]
    assert body["planned_founders"] == body["completed_founders"] == [[0, "selected", "every"]]
    run = body["runs"][0]
    branches = run["window"]["branches"]
    assert set(branches) == set(chamber.CONTROLS)
    assert run["window"]["continuation"] == {"actions_equal": True, "saved_arrays_equal": True}
    assert run["disturbances"]["pause2"]["custody"]["actions_equal"]
    assert run["disturbances"]["pause2"]["custody"]["saved_arrays_equal"]
    assert branches["static_cold"]["score"]["alternation_rate"] == 0.0
    assert branches["intact"]["actions"] == branches["restored"]["actions"]
    for name in chamber.CONTROLS:
        assert len(branches[name]["actions"]) == 8
    assert run["teaching"]["lessons_attempted"] == 3
    aggregate = body["aggregate"]["selected/every"]
    assert aggregate["founders"] == 1 and aggregate["continuation_equal"]
    assert aggregate["cadence"] is None
    source = output / "source/steady_rhythm.py"
    source.write_text(source.read_text() + "\n# corrupted after the run\n")
    assert not chamber.verify(output)[0]
    with pytest.raises(FileExistsError):
        chamber.main(["--out", str(output)])


@pytest.mark.parametrize(
    "bad",
    [
        ["--seeds", "-1"],
        ["--seeds", "1", "1"],
        ["--arms", "sometimes"],
        ["--recipes", "unknown"],
        ["--window", "4"],
        ["--events-per-bout", "1"],
        ["--decay", "1.0"],
        ["--burners", "0"],
    ],
)
def test_invalid_cli_is_rejected_before_creating_an_attempt(tmp_path, bad):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        chamber.main(["--out", str(output), *bad])
    assert error.value.code == 2 and not output.exists()


# -- rhythm/2: the efference copy as the declared mechanism


@pytest.fixture(scope="module")
def protocol_2():
    return inputs.load_protocol(ROOT / "protocol-2.json")


def test_protocol_2_declares_the_copy_against_the_rhythm_1_control_on_fresh_seeds(
    protocol, protocol_2
):
    one, _ = protocol
    two, digest = protocol_2
    assert digest == hashlib.sha256((ROOT / "protocol-2.json").read_bytes()).hexdigest()
    assert two["schema"] == "steady-rhythm/2"
    assert two["recipes"]["selected"] == one["recipes"]["selected"]
    candidate = dict(two["recipes"]["efference"])
    assert candidate.pop("efference_amplitude") == 3.0 and candidate.pop("efference_decay") == 0.0
    assert candidate == one["recipes"]["selected"]
    assert two["seeds"]["development"] == one["seeds"]["development"]
    assert len(two["seeds"]["confirmation"]) == 5
    assert not set(two["seeds"]["confirmation"]) & set(one["seeds"]["confirmation"])
    assert not set(two["seeds"]["confirmation"]) & set(two["seeds"]["development"])
    for key in ("event", "teaching", "window", "disturbances", "cadence", "controls", "caps"):
        assert two[key] == one[key]


def test_the_efference_recipe_carries_the_copy_and_every_history_control_acts_on_it(
    protocol_2, tmp_path
):
    declared, _ = protocol_2
    control = chamber.make_brain(0, declared, "selected")
    brain = chamber.make_brain(0, declared, "efference")
    assert control.efference is None and brain.efference is not None
    assert brain.efference.decay == 0.0 and brain.efference.amplitude == 3.0
    assert len(brain.connectome.populations["efference"]) == inputs.ACTIONS
    x = inputs.observation(inputs.KIND_DRIVE)
    answer = chamber.Work().act(brain, x)
    np.testing.assert_array_equal(brain.efference.last, np.eye(inputs.ACTIONS)[answer])
    np.testing.assert_array_equal(brain.efference.trace, np.eye(inputs.ACTIONS)[answer])
    chamber.Work().act(brain, x)
    probe = brain.save(tmp_path / "probe.npz")
    erased = chamber.Brain.load(probe)
    chamber.erase_trace(erased)
    for memory in (erased.working_memory, erased.efference):
        assert not memory.trace.any() and memory.cold.all()
    permutation = np.array([2, 3, 0, 1])
    shuffled = chamber.Brain.load(probe)
    chamber.transplant_trace(shuffled, permutation)
    for memory, original in (
        (shuffled.working_memory, brain.working_memory),
        (shuffled.efference, brain.efference),
    ):
        np.testing.assert_array_equal(memory.trace, original.trace[permutation])
        np.testing.assert_array_equal(memory.last, original.last[permutation])
    assert chamber.same_saved_arrays(probe, brain.save(tmp_path / "after.npz"))
