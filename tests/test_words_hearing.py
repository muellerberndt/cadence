"""The parrot's ear and voice (cadence.words.hearing): heard moments kept as records beside the
thing in focus, fetched by sound and by focus."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.words.hearing import Ear, Voice, sound

RATE = 16000


def _word(low: float, high: float, seed: int = 0, length: int = 8000) -> np.ndarray:
    """Half a second of two tones one after the other, with silence around them."""
    rng = np.random.default_rng(seed)
    t = np.arange(length) / RATE
    sound = np.where(t < length / (2 * RATE), np.sin(2 * np.pi * low * t),
                     np.sin(2 * np.pi * high * t)) * np.hanning(length)
    silence = np.zeros(1600)
    return np.concatenate([silence, sound, silence]) + 0.001 * rng.standard_normal(length + 3200)


def _noise(seed: int = 0, length: int = 8000) -> np.ndarray:
    """Half a second of noise rising in pitch (a hiss), with silence around it."""
    rng = np.random.default_rng(seed + 100)
    hiss = np.diff(rng.standard_normal(length + 1)) * np.linspace(0.2, 1.0, length)
    silence = np.zeros(1600)
    return np.concatenate([silence, hiss * np.hanning(length), silence]) + 0.001 * \
        rng.standard_normal(length + 3200)


def _taught() -> Ear:
    ear = Ear(5, settle=50, capacity=256)
    first, second = _word(300.0, 900.0), _noise()
    for _ in range(4):  # the fields learn from the first sounds heard
        ear.hear(first, learn=True)
        ear.hear(second, learn=True)
    _, records = ear.listen(first, learn=True)
    ear.teach(records, np.eye(5)[0])
    _, records = ear.listen(second, learn=True)
    ear.teach(records, np.eye(5)[3])
    return ear


def test_the_fields_learn_from_the_first_sounds_and_then_hold() -> None:
    ear = Ear(5, settle=50, capacity=64)
    before = ear.fields.dictionary.copy()
    for _ in range(6):  # about eleven moments of sound a word
        ear.hear(_word(300.0, 900.0), learn=True)
    assert not np.array_equal(before, ear.fields.dictionary)
    assert ear.fields.learning is False and ear.moments >= 50
    held = ear.fields.dictionary.copy()
    ear.hear(_word(700.0, 300.0), learn=True)
    assert np.array_equal(held, ear.fields.dictionary)
    assert ear.records == 0  # hearing alone writes no record


def test_a_sound_brings_back_the_concepts_it_was_heard_with() -> None:
    ear = _taught()
    first, _ = ear.listen(_word(300.0, 900.0, seed=1), learn=False)
    second, _ = ear.listen(_noise(seed=1), learn=False)
    assert int(np.argmax(ear.recall(first, credit=False))) == 0
    assert int(np.argmax(ear.recall(second, credit=False))) == 3


def test_the_focus_chooses_the_word_and_none_for_a_thing_never_heard() -> None:
    ear = _taught()
    record, match = ear.choose(np.eye(5)[3])  # the record heard back names concept 3 most
    assert record >= 0 and match > 0.7
    assert ear.template(record).shape[1] == 40 and len(ear.template(record)) >= 15
    assert ear.choose(np.eye(5)[1]) == (-1, 0.0)
    assert Ear(5, capacity=8).choose(np.eye(5)[0]) == (-1, 0.0)


def test_a_word_is_a_record_and_its_takes_settle_on_their_mean() -> None:
    ear = _taught()
    count = ear.records
    _, records = ear.listen(_word(300.0, 900.0, seed=3), learn=True)
    assert len(records) == 1 and ear.records == count  # one word, told again
    assert ear.counts[records[0]] >= 2


def test_practice_brings_the_voice_toward_the_word_heard() -> None:
    ear = _taught()
    record, _ = ear.choose(np.eye(5)[0])
    voice = Voice(ear, seed=0)
    first = voice.match(voice.say(record), record)
    for _ in range(12):
        voice.attempt(record)
    assert voice.match(voice.say(record), record) > first
    voice.drive = 0.0
    for _ in range(20):
        voice.moment()
    assert voice.bored
    voice.attempt(record)
    assert not voice.bored


def test_the_vocal_tract_sounds_the_frames() -> None:
    frames = np.log(np.full((30, 40), 1e-3)) + np.linspace(0, 2, 40)[None, :]
    samples = sound(frames)
    assert samples.dtype == np.float32 and len(samples) == 29 * 160 + 400


def test_a_test_writes_nothing() -> None:
    ear = _taught()
    count = ear.records
    ear.listen(_word(300.0, 900.0), learn=False)
    assert ear.records == count


def test_the_ear_refuses_settings_it_cannot_hold() -> None:
    with pytest.raises(ValueError):
        Ear(0)
    with pytest.raises(ValueError):
        Ear(5, decays=(1.0,))
    with pytest.raises(ValueError):
        Ear(5, window=4)
