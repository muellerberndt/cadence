"""The spoken-phrase chamber (cadence#303) on synthetic speech: the audio, the phrases and their
targets, the new speakers' disjoint recordings, the baselines' parameter counts, and a short
life of a brain and of a baseline through adaptation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str):
    key = f"spoken_{name}"  # a name of this chamber's own: other benchmarks have their modules
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


audio = _load("audio")
corpora = _load("corpora")
ph = _load("phrases")
ch = _load("chamber")
sp = _load("spoken")


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    out = tmp_path_factory.mktemp("spoken")
    built = ph.build_synthetic(out, train=32, held=4, adapt=8, evaluate=4)
    return out, built


def test_one_second_holds_98_frames_of_40_bands():
    tone = np.sin(2 * np.pi * 1000 * np.arange(audio.RATE) / audio.RATE).astype(np.float32)
    mel = audio.log_mel(tone)
    assert mel.shape == (audio.SECOND, audio.BANDS) == (98, 40)
    loud = int(np.argmax(mel.mean(axis=0)))
    centres = audio._hz(
        np.linspace(audio._mel(audio.F_MIN), audio._mel(audio.F_MAX), audio.BANDS + 2)
    )[1:-1]
    assert abs(centres[loud] - 1000) < 120


def test_resampling_from_48_khz_keeps_speech_and_drops_what_16_khz_cannot_hold():
    t = np.arange(48_000) / 48_000
    kept = audio.resample(np.sin(2 * np.pi * 1000 * t), 48_000)
    dropped = audio.resample(np.sin(2 * np.pi * 11_000 * t), 48_000)
    assert len(kept) == 16_000
    assert 0.6 < np.sqrt(np.mean(kept[200:-200] ** 2)) < 0.75
    assert np.sqrt(np.mean(dropped[200:-200] ** 2)) < 0.01


def test_targets_name_each_word_once_then_end_two_steps_after_the_last():
    target = ph.targets(("go", "stop", "up"), [2, 4, 7])
    assert list(target) == [
        ph.SILENCE,
        ph.WORDS.index("go"),
        ph.SILENCE,
        ph.WORDS.index("stop"),
        ph.SILENCE,
        ph.SILENCE,
        ph.WORDS.index("up"),
        ph.SILENCE,
        ph.END,
        ph.END,
    ]
    assert ph.word_steps([(30, 80), (140, 190)]) == [1, 3]
    assert ph.word_steps([(30, 97), (98, 99)]) == [1, 2]  # never two words at one step
    mel = np.zeros((10 * ph.STEP, audio.BANDS))
    assert ph.window(mel, 3).shape == (ph.WINDOW, audio.BANDS)


def test_built_phrases_hold_their_words_steps_and_seconds(sets):
    out, _ = sets
    train = ph.Phrases(out / "base_train.npz")
    assert len(train) == 32 and set(train.length) == {2, 3}
    for r in range(len(train)):
        steps = int(train.steps[r])
        row = train.target[r, :steps]
        words = row[row < ph.SILENCE]
        assert tuple(words) == train.reference(r)
        assert row[-1] == ph.END and row[-2] == ph.END and (train.target[r, steps:] == -1).all()
        assert abs(train.seconds[r] - audio.samples_for(ph.STEP * (steps + 1)) / audio.RATE) < 1e-9
    held = ph.Phrases(out / "base_test.npz")
    assert set(held.length) == {2, 3, 4}


def test_a_new_speakers_evaluation_never_shares_a_recording_with_adaptation():
    corpus = corpora.Synthetic(seed=0)
    speaker = sorted(corpus.speakers("test"))[0]
    sides = ph.plan_speaker(corpus, speaker, "test", 12, 6, ph.SEED, "speaker")
    adapt = {c for p in sides["adapt"] for c in p.clips}
    evaluate = {c for p in sides["evaluate"] for c in p.clips}
    assert adapt and evaluate and not adapt & evaluate
    assert all(len(set(p.words)) == len(p.words) for p in sides["adapt"] + sides["evaluate"])
    assert {p.speaker for p in sides["adapt"]} == {speaker}


def test_the_base_test_set_leaves_the_new_speakers_out(sets):
    out, built = sets
    held = ph.Phrases(out / "base_test.npz")
    new = built["new_speaker"].split("/", 1)[1]
    dialect = built["dialect"].split("/", 1)[1]
    assert new not in set(held.speaker) and dialect not in set(held.speaker)


def test_the_heard_window_is_an_image_of_bands_by_frames(sets):
    out, _ = sets
    train = ph.Phrases(out / "base_train.npz")
    hearing = ch.Hearing.measured(train)
    x = ch.stream(train, np.array([0]), hearing)
    assert x.shape == (1, int(train.steps[0]), audio.BANDS * ph.WINDOW)
    assert np.abs(x).max() <= ch.CLIP_SIGMA * ch.SCALE + 1e-9
    image = x[0, 0].reshape(audio.BANDS, ph.WINDOW)
    window = ph.window(train.logmel[0].astype(np.float64), 1)
    centred = window - window.mean()
    expected = np.clip((centred - hearing.mean) / hearing.std, -3, 3) * ch.SCALE
    assert np.allclose(image, expected.T)


def test_decode_reads_words_up_to_end_and_edits_count_each_kind():
    acts = np.array([[ph.SILENCE, 3, ph.SILENCE, 5, ph.END, 7]])
    got = ch.decode(acts)[0]
    assert got == {"words": (3, 5), "ended": True}
    assert ch.edits((3, 5), (3, 5)) == (0, 0, 0)
    assert ch.edits((3, 6), (3, 5)) == (1, 0, 0)
    assert ch.edits((3,), (3, 5)) == (0, 0, 1)
    assert ch.edits((3, 4, 5), (3, 5)) == (0, 1, 0)


@pytest.mark.parametrize("kind", ["transformer", "lstm"])
def test_a_baselines_counted_parameters_are_its_tensors(kind):
    pytest.importorskip("torch")
    ct = _load("controls")
    for loss in ("step", "ctc"):
        width = ct.width_for(kind, ph.ACTIONS + (loss == "ctc"), 200_000)
        model = ct.Sequence(kind, loss, width, ph.ACTIONS, seed=0)
        assert model.parameters() == ct.parameters(kind, width, model.outputs)
        assert abs(model.parameters() - 200_000) < 0.05 * 200_000


def test_a_short_life_of_a_brain_and_a_baseline_adapts_and_reads(sets):
    pytest.importorskip("torch")
    out, built = sets
    speaker = built["new_speaker"]
    for name in ("fields", "transformer_ctc"):
        result = sp._run(
            {
                "data": str(out),
                "learner": name,
                "founder": 0,
                "budgets": [32],
                "speakers": [speaker],
                "modes": ["plain", "replay"],
                "rates": [1.0],
                "minutes": [0, 0.25],
                "probe": name == "fields",
            }
        )
        assert "error" not in result, result.get("trace")
        reading = result["readings"][-1]
        assert reading["phrases"] == 32
        assert 0.0 <= reading["test"]["exact"] <= 1.0 and reading["test"]["phrases"] == 12
        curves = {a["mode"]: a["curve"] for a in result["adaptation"]}
        assert [c["minutes"] for c in curves["plain"]] == [0, 0.25]
        assert curves["plain"][0]["phrases"] == 0 and curves["plain"][1]["heard_seconds"] >= 15
        # a reading at 0 minutes is the life before any correction, in both modes
        same = ("word_accuracy", "exact", "substitutions", "insertions", "deletions")
        assert all(curves["plain"][0]["own"][k] == curves["replay"][0]["own"][k] for k in same)
        if name == "fields":
            probe = result["probe"]
            assert {"window", "pools", "pools_drawn", "association"} <= set(probe)
            assert probe["pools"]["width"] == 64 * 9
            assert result["parameters"] > 0 and "at_word" in reading["test"]


def test_reading_a_brain_leaves_its_life_unchanged(sets):
    out, _ = sets
    train = ph.Phrases(out / "base_train.npz")
    hearing = ch.Hearing.measured(train)
    learner = ch.BrainLearner(ch.compose("fields", 0), "fields")
    ch.train(learner, train, ch.groups(train, np.arange(8), 8), hearing)
    before = learner.brain.fields.dictionary.copy()
    efficacy = np.array(learner.brain.brain.efficacy, copy=True)
    ch.read(learner, train, np.arange(8, 16), hearing)
    assert np.array_equal(before, learner.brain.fields.dictionary)
    assert np.array_equal(efficacy, np.asarray(learner.brain.brain.efficacy))


def test_a_held_word_stays_due_over_the_silent_steps_after_it_and_reads_once():
    s, e = ph.SILENCE, ph.END
    target = np.array([[s, 3, s, s, 5, 7, s, e, e, -1]])
    assert (ch.held(target, 1) == target).all()
    assert ch.held(target, 2).tolist() == [[s, 3, 3, s, 5, 7, 7, e, e, -1]]
    assert ch.held(target, 3).tolist() == [[s, 3, 3, 3, 5, 7, 7, e, e, -1]]
    assert ch.decode(ch.held(target, 2), collapse=True)[0] == {"words": (3, 5, 7), "ended": True}
    assert ch.decode(ch.held(target, 2))[0]["words"] == (3, 3, 5, 7, 7)


def test_a_saved_life_adapts_from_where_it_ended(sets, tmp_path):
    pytest.importorskip("torch")
    out, built = sets
    for name in ("fields", "transformer_step"):
        saved = tmp_path / f"{name}-0"
        base = sp._run(
            {"data": str(out), "learner": name, "founder": 0, "budgets": [32], "save": str(saved)}
        )
        assert "error" not in base, base.get("trace")
        again = sp._run(
            {
                "data": str(out),
                "learner": name,
                "founder": 0,
                "budgets": [],
                "load": str(saved),
                "speakers": [built["new_speaker"]],
                "modes": ["plain"],
                "minutes": [0],
            }
        )
        assert "error" not in again, again.get("trace")
        at0 = again["adaptation"][0]["curve"][0]["retention"]
        assert again["readings"] == [] and 0.0 <= at0["exact"] <= 1.0
