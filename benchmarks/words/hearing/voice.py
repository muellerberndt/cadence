"""The parrot's voice as sound: log-mel frames (the format the ear reads) turned into samples
through the mel filter bank (cadence#329, V1 to V3).

The voice motor gives one frame every 10 ms: the natural log of the power in each of the 40 mel
filters of ``cadence.speech.frontend.mel_bank`` over a 25 ms Hann-windowed frame of 16 kHz
samples (``log_mel``'s format). ``sound`` makes samples whose ``log_mel`` is near those frames:

1. each frame's band powers spread over the spectrum's bins through the filters (each bin the
   filters' powers, weighted by the filter's share of the bin, per unit of filter area);
2. the magnitude spectrum's phase found by Griffin and Lim's iteration (``rounds`` passes of an
   inverse and a forward short-time transform at the front end's frame and hop);
3. the samples scaled to a peak of ``peak``.

``round_trip(frames)`` gives the mean absolute difference of ``log_mel(sound(frames))`` and the
frames over the frames within ``drop`` of the loudest (natural log units). The page's
synthesiser is read against this one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RATE, FRAME, HOP, FFT, MELS = 16000, 400, 160, 512, 40


def _bank() -> np.ndarray:
    from cadence.speech.frontend import mel_bank

    return np.asarray(mel_bank(MELS, rate=RATE, fft=FFT))


def spectrum(frames: Any) -> np.ndarray:
    """Each frame's magnitude spectrum ``(frames, FFT // 2 + 1)`` from its log-mel powers."""
    bank = _bank()
    power = np.maximum(np.exp(np.asarray(frames, dtype=np.float64)) - 1e-6, 0.0)
    area = bank.sum(axis=1, keepdims=True)
    weight = bank / np.maximum(area, 1e-12)  # a filter's power per unit of its area
    cover = np.maximum(bank.sum(axis=0), 1e-12)
    spread = (power @ weight) / cover * (bank.sum(axis=0) > 0)
    return np.asarray(np.sqrt(np.maximum(spread, 0.0)))


def _stft(samples: np.ndarray, count: int) -> np.ndarray:
    window = np.hanning(FRAME)
    padded = np.pad(samples, (0, max(0, (count - 1) * HOP + FRAME - len(samples))))
    index = np.arange(FRAME)[None, :] + HOP * np.arange(count)[:, None]
    return np.asarray(np.fft.rfft(padded[index] * window, FFT))


def _istft(spec: np.ndarray) -> np.ndarray:
    window = np.hanning(FRAME)
    count = len(spec)
    out = np.zeros((count - 1) * HOP + FRAME)
    norm = np.zeros_like(out)
    pieces = np.fft.irfft(spec, FFT)[:, :FRAME] * window
    for t in range(count):
        out[t * HOP: t * HOP + FRAME] += pieces[t]
        norm[t * HOP: t * HOP + FRAME] += window**2
    return np.asarray(out / np.maximum(norm, 1e-8))


def sound(frames: Any, *, rounds: int = 32, peak: float = 0.5, seed: int = 0) -> np.ndarray:
    """Samples at 16 kHz whose log-mel frames are near ``frames`` ``(frames, MELS)``."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != MELS or len(frames) < 2:
        raise ValueError(f"frames is (frames, {MELS}) with 2 frames or more")
    magnitude = spectrum(frames)
    rng = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * rng.random(magnitude.shape))
    samples = _istft(magnitude * phase)
    for _ in range(int(rounds)):
        spec = _stft(samples, len(frames))
        phase = np.exp(1j * np.angle(spec))
        samples = _istft(magnitude * phase)
    top = float(np.abs(samples).max())
    return np.asarray(samples * (peak / top if top > 0 else 1.0), dtype=np.float32)


def round_trip(frames: Any, *, drop: float = 4.0, **settings: Any) -> float:
    """The mean absolute difference, in natural log units, of the log-mel frames of
    ``sound(frames)`` and ``frames`` (each less its frames' mean level, so the level the sound is
    scaled to does not count), over the frames within ``drop`` of the loudest."""
    from cadence.speech.frontend import log_mel

    frames = np.asarray(frames, dtype=np.float64)
    heard = log_mel(sound(frames, **settings).astype(np.float64))
    count = min(len(heard), len(frames))
    a, b = frames[:count], heard[:count]
    energy = np.log(np.exp(a).sum(axis=1))
    loud = energy > energy.max() - drop
    a, b = a[loud], b[loud]
    return float(np.mean(np.abs((a - a.mean()) - (b - b.mean()))))
