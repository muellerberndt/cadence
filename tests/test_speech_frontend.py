"""The speech front end: log-mel energies and the step's time-frequency window."""

import numpy as np
import pytest

from cadence.speech.frontend import Hearing, LogMel, log_mel, mel_bank, windows


def test_the_mel_bank_is_triangles_that_rise_and_fall_once():
    bank = mel_bank(4, rate=8000, fft=16)
    assert bank.shape == (4, 9)
    assert (bank >= 0).all() and bank.max() <= 1.0
    for row in bank:
        peak = int(np.argmax(row))
        assert np.all(np.diff(row[: peak + 1]) >= 0) and np.all(np.diff(row[peak:]) <= 0)


def test_a_tone_lights_the_band_that_holds_its_frequency():
    rate = 16000
    t = np.arange(rate) / rate
    energies = log_mel(np.sin(2 * np.pi * 1000 * t), rate=rate)
    assert energies.shape == (98, 40)  # 1 + (16000 - 400) // 160 frames
    bank = mel_bank(40, rate=rate)
    band = int(np.argmax(bank[:, round(1000 * 512 / rate)]))
    assert int(np.argmax(energies[50])) == band
    assert log_mel(np.zeros(10)).shape == (1, 40)  # shorter than a frame: one padded frame


def test_a_window_is_the_latest_frames_less_its_mean():
    frames = np.arange(10, dtype=float)[:, None] * np.ones((1, 2))  # frame k holds k
    w = windows(frames, window=3, stack=4)
    assert w.shape == (2, 3, 2)  # 10 frames: two whole steps of four
    # step 0 ends after frame 3: frames 1, 2, 3, less their mean 2
    np.testing.assert_allclose(w[0, :, 0], [-1.0, 0.0, 1.0])
    # a window reaching before the audio repeats the first frame: frames 0, 0, 0, 0, 1
    w = windows(frames, window=5, stack=2)
    np.testing.assert_allclose(w[0, :, 0], np.array([0, 0, 0, 0, 1]) - 0.2)


def test_hearing_standardizes_clips_and_lays_out_bands_by_rows():
    rng = np.random.default_rng(0)
    utterances = [rng.normal(size=(40, 3)) for _ in range(4)]
    hearing = Hearing.fit(utterances, window=4, stack=4)
    assert hearing.shape == (3, 4)
    x = hearing.steps(utterances[0])
    assert x.shape == (10, 12) and x.dtype == np.float32
    assert np.abs(x).max() <= 3.0 * 0.3 + 1e-6
    w = windows(utterances[0], window=4, stack=4)
    z = np.clip((w - hearing.mean) / hearing.std, -3, 3) * 0.3
    np.testing.assert_allclose(x[0].reshape(3, 4), z[0].T, rtol=1e-6)  # row = band
    again = Hearing.from_dict(hearing.to_dict())
    np.testing.assert_array_equal(again.steps(utterances[1]), hearing.steps(utterances[1]))
    with pytest.raises(ValueError):
        hearing.steps(rng.normal(size=(40, 5)))
    with pytest.raises(ValueError):
        Hearing(np.zeros(3), np.zeros(3))


def test_the_reader_front_end_turns_samples_into_the_hearings_rows():
    rng = np.random.default_rng(1)
    audio = rng.normal(size=4000)
    hearing = Hearing.fit([log_mel(audio)])
    front = LogMel(hearing)
    assert (front.window, front.hop, front.stack, front.frames) == (400, 160, 4, 24)
    assert front.shape == (40, 24)
    frames = front.log_mel(audio)
    np.testing.assert_array_equal(front.steps(audio), hearing.steps(frames))
    raw = np.repeat(frames[:1], 24, axis=0)[None]  # a window before the audio, unshifted
    np.testing.assert_array_equal(front.rows(raw), hearing.rows(raw - raw.mean()))
