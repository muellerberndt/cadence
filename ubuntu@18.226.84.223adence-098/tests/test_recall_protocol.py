"""The recall instrument must isolate conditions and charge actual attempted work."""

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from cadence import Brain, NeuralGraph


@pytest.fixture
def recall(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "benchmarks/recall"))
    return import_module("vanished_cue")


def saved_arrays(brain, path):
    with np.load(brain.save(path), allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def assert_same_saved(first, second):
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


def tiny_brain(recall, *, qualified=False):
    return recall.make_brain(6, 2, 3, 0.8, 1.0, (4,), qualified)


def frozen(recall, path):
    return recall.freeze_streams(path, seed=3, delay=2, cues=2, distractors=2,
                                 streams=4, lessons=2, test_episodes=3)


def test_frozen_pairs_have_different_cues_and_identical_later_inputs(recall, tmp_path):
    first = frozen(recall, tmp_path / "first.npz")
    np.random.default_rng(31).normal(size=1000)
    second = frozen(recall, tmp_path / "second.npz")
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert not first[name].flags.writeable
    for phase in ("train", "test"):
        labels, obs = first[phase + "_labels"], first[phase + "_observations"]
        for label, steps in zip(labels, obs, strict=True):
            for start in (0, 2):
                assert set(label[start:start + 2]) == {0, 1}
                assert not np.array_equal(steps[0, start], steps[0, start + 1])
                np.testing.assert_array_equal(steps[1:, start], steps[1:, start + 1])
            assert not steps[:, :, -2:].any()  # the vanished-cue input never receives the label
    for labels, permutation in zip(first["test_labels"], first["test_permutations"], strict=True):
        assert sorted(permutation) == list(range(4))
        assert np.all(labels[permutation] != labels)


def test_controls_fork_full_checkpoint_and_do_not_change_the_live_life(recall, tmp_path, monkeypatch):
    brain = tiny_brain(recall)
    brain.act(np.eye(6)[[2, 3]], greedy=True)
    brain.act(np.eye(6)[[0, 0]], greedy=True)
    before = saved_arrays(brain, tmp_path / "anchor.npz")
    original_v = brain.basal_ganglia.state.v.copy()
    original_act = Brain.act
    initial_states = []

    def observed(self, x, **kwargs):
        initial_states.append(None if self.basal_ganglia.state is None
                              else self.basal_ganglia.state.v.copy())
        return original_act(self, x, **kwargs)

    monkeypatch.setattr(Brain, "act", observed)
    forward = recall.probe_controls(tmp_path / "anchor.npz", np.eye(6)[[1, 1]], np.array([1, 0]))
    reverse = recall.probe_controls(tmp_path / "anchor.npz", np.eye(6)[[1, 1]], np.array([1, 0]),
                                    order=tuple(reversed(recall.CONTROLS)))
    for name in recall.CONTROLS:
        assert forward[name]["answers"] == reverse[name]["answers"]
        forward[name]["work"].pop("calls_seconds")
        reverse[name]["work"].pop("calls_seconds")
        assert forward[name]["work"] == reverse[name]["work"]
    assert sum(state is None for state in initial_states) == 2  # only full-reset branches
    for state in initial_states:
        if state is not None:
            np.testing.assert_array_equal(state, original_v)
    assert_same_saved(before, saved_arrays(brain, tmp_path / "after.npz"))


def test_teacher_and_free_work_matches_actual_solver_phases(recall, monkeypatch):
    brain = tiny_brain(recall)
    actual = []
    original = NeuralGraph.settle_batch

    def measured(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        actual.append((len(result.v), result.steps))
        return result

    monkeypatch.setattr(NeuralGraph, "settle_batch", measured)
    work = recall.Work()
    obs = np.eye(6)[np.array([[2, 3], [0, 0], [1, 1]])]
    answers = recall.run_stream(brain, obs, work, teacher=np.array([0, 1]))
    assert answers is not None
    assert work.teacher_attempts == 1 and work.teacher_accepted_presentations == 2
    assert work.action_attempts == 3
    assert work.action_sweeps + work.teacher_sweeps == sum(steps for _, steps in actual)
    assert work.action_row_sweeps + work.teacher_row_sweeps == sum(rows * steps for rows, steps in actual)
    assert work.teacher_sweeps > 0
    assert brain.basal_ganglia._pending is None
    assert brain.basal_ganglia.updates == 0  # no unobserved zero-reward transition
    assert brain.hippocampus is None


def test_teacher_refusal_then_free_refusal_is_charged_without_reset(recall, tmp_path):
    brain = tiny_brain(recall, qualified=True)
    brain.learner.config = replace(brain.learner.config, free_steps=1, nudged_steps=0)
    before = saved_arrays(brain, tmp_path / "before.npz")
    work = recall.Work()
    answer = recall.run_stream(brain, np.eye(6)[np.array([[1, 1]])], work,
                               teacher=np.array([0, 1]))
    assert answer is None
    assert work.teacher_refusals == work.action_refusals == 1
    assert work.teacher_sweeps == 1
    assert work.action_sweeps == 2  # the answer's budget and its rescue
    assert work.teacher_residual_checks > 0 and work.action_residual_checks > 0
    assert work.teacher_accepted_presentations == work.trace_row_updates == 0
    assert work.resets_after_refusal == 0
    assert_same_saved(before, saved_arrays(brain, tmp_path / "after.npz"))


def test_unrelated_runtime_error_cannot_reuse_a_stale_refusal(recall, monkeypatch):
    brain = tiny_brain(recall)
    brain.learner.config = replace(brain.learner.config, free_steps=0)
    x = np.eye(6)[[1, 1]]
    assert recall.Work().act(brain, x) is None

    def unrelated(*args, **kwargs):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(brain, "act", unrelated)
    with pytest.raises(RuntimeError, match="unrelated failure"):
        recall.Work().act(brain, x)


def test_appended_comparator_is_taught_the_same_extra_coordinates_it_will_read(
    recall, tmp_path, monkeypatch,
):
    data = frozen(recall, tmp_path / "streams.npz")
    presentations = []

    def collect(brain, obs, work, teacher=None):
        presentations.append((obs.copy(), teacher.copy()))
        return teacher.copy()

    monkeypatch.setattr(recall, "run_stream", collect)
    recall.train(None, data, appended=True, cues=2)
    assert len(presentations) == len(data["train_labels"])
    for (obs, teacher), raw in zip(presentations, data["train_observations"], strict=True):
        assert not obs[:-1, :, -2:].any()
        np.testing.assert_array_equal(obs[-1, :, -2:], np.eye(2)[teacher])
        np.testing.assert_array_equal(obs[:, :, :-2], raw[:, :, :-2])
    assert not data["train_observations"][:, :, :, -2:].any()


@pytest.mark.parametrize("bad", [
    ["--cues", "0"], ["--cues", "1"], ["--streams", "3"], ["--lessons", "0"],
    ["--test-episodes", "0"], ["--distractors", "-1"], ["--seeds", "-1"],
    ["--delays", "-1"], ["--decays", "nan"], ["--decays", "1"], ["--amplitude", "inf"],
    ["--modules", "0"], ["--modules", "oops"], ["--delays", "1", "1"],
])
def test_invalid_cli_is_rejected_before_creating_an_attempt(recall, tmp_path, bad):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        recall.main(["--out", str(output), *bad])
    assert error.value.code == 2 and not output.exists()


def test_smoke_preserves_sources_inputs_work_and_continuation(recall, tmp_path):
    output = tmp_path / "smoke"
    assert recall.main([
        "--out", str(output), "--cues", "2", "--streams", "2", "--modules", "2",
        "--lessons", "1", "--test-episodes", "1", "--seeds", "0", "--decays", "0.8",
        "--delays", "1", "--finite",
    ]) == 0
    assert recall.verify(output)[0]
    import json

    report = json.loads((output / "summary.json").read_text())["body"]
    models = report["rows"][0]["models"]
    assert set(models) == {"vanished", "appended"}
    for model in models.values():
        assert model["training_completed"] and model["evaluation"]["completed"]
        assert model["training"]["teacher_attempts"] == 1
    evaluation = models["vanished"]["evaluation"]
    assert evaluation["trials"][0]["branches"]["resume_check"]["equal_checkpoint"]
    assert all(score["attempted_rows"] == 2 and score["not_run_rows"] == 0
               for score in evaluation["scores"].values())
    source = output / "source/vanished_cue.py"
    source.write_text(source.read_text() + "\n# corrupted after the run\n")
    assert not recall.verify(output)[0]
    with pytest.raises(FileExistsError):
        recall.main(["--out", str(output)])
