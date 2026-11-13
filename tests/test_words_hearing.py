"""The parrot's ear and voice (cadence.words.hearing): heard moments kept as records beside the
thing in focus, fetched by sound and by focus."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.words.hearing import Ear

RATE = 16000


def _word(low: float, high: float, seed: int = 0, length: int = 8000) -> np.ndarray:
    """Half a second of two tones one after the other, with silence around them."""
    rng = np.random.default_rng(seed)
    t = np.arange(length) / RATE
    sound = np.where(t < length / (2 * RATE), np.sin(2 * np.pi * low * t),
                     np.sin(2 * np.pi * high * t)) * np.hanning(length)
    silence = np.zeros(1600)
    return np.concatenate([silence, sound, silence]) + 0.001 * rng.standard_normal(length + 3200)


def _taught() -> Ear:
    ear = Ear(5, settle=50, capacity=256)
    first, second = _word(300.0, 900.0), _word(1200.0, 500.0)
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
    second, _ = ear.listen(_word(1200.0, 500.0, seed=1), learn=False)
    assert int(np.argmax(ear.recall(first, credit=False))) == 0
    assert int(np.argmax(ear.recall(second, credit=False))) == 3


def test_the_focus_brings_back_the_sound_and_nothing_for_a_thing_never_heard() -> None:
    ear = _taught()
    frames, match = ear.voice(np.eye(5)[3])
    assert frames.shape[1] == 40 and len(frames) > 10 and match > 0.9
    _, match = ear.voice(np.eye(5)[1])
    assert match == 0.0
    empty, match = Ear(5, capacity=8).voice(np.eye(5)[0])
    assert empty.shape == (0, 40) and match == 0.0


def test_a_moment_told_again_revises_its_record_and_a_test_writes_nothing() -> None:
    ear = _taught()
    count = ear.records
    ear.listen(_word(300.0, 900.0), learn=False)
    assert ear.records == count
    ear.listen(_word(300.0, 900.0), learn=True)
    assert ear.records < 2 * count  # most moments of a sound heard again are told again


def test_the_ear_refuses_settings_it_cannot_hold() -> None:
    with pytest.raises(ValueError):
        Ear(0)
    with pytest.raises(ValueError):
        Ear(5, decays=(1.0,))
    with pytest.raises(ValueError):
        Ear(5, window=4)
