"""Audio for the spoken-phrase chamber (cadence#303): 16 kHz samples to log-mel frames.

NumPy only and deterministic. 25 ms frames every 10 ms and 40 mel bands between 20 Hz and
7.6 kHz, so one second of audio holds 98 frames. A recording at another rate is resampled: by
an integer factor through a windowed-sinc low-pass filter (48 kHz to 16 kHz), otherwise by
linear interpolation.
"""

from __future__ import annotations

import hashlib
import io
import wave
from pathlib import Path

import numpy as np

RATE = 16_000
WINDOW = 400  # 25 ms
HOP = 160  # 10 ms
N_FFT = 512
BANDS = 40
F_MIN = 20.0
F_MAX = 7_600.0
LOG_FLOOR = 1e-6
SECOND = 1 + (RATE - WINDOW) // HOP  # 98 frames in one second
TAPS = 32  # half-length of the windowed-sinc filter, in output samples


def read_wav(source: str | Path | bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM as float32 in [-1, 1] and its rate; several channels are averaged."""
    handle = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    with wave.open(handle, "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {8 * width}-bit")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, rate


def write_wav(path: str | Path, pcm: np.ndarray, rate: int = RATE) -> None:
    data = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)


def resample(pcm: np.ndarray, rate: int) -> np.ndarray:
    """``pcm`` at ``rate`` as 16 kHz float32."""
    pcm = np.asarray(pcm, dtype=np.float64)
    if rate == RATE:
        return pcm.astype(np.float32)
    if rate % RATE == 0:  # an integer factor: low-pass below the new Nyquist, then decimate
        factor = rate // RATE
        n = np.arange(-TAPS * factor, TAPS * factor + 1)
        h = np.sinc(n / factor) / factor * np.blackman(len(n))
        h /= h.sum()
        return np.convolve(pcm, h, mode="same")[::factor].astype(np.float32)
    count = int(round(len(pcm) * RATE / rate))
    at = np.arange(count) * (rate / RATE)
    return np.interp(at, np.arange(len(pcm)), pcm).astype(np.float32)


def frame_energies(pcm: np.ndarray) -> np.ndarray:
    """Mean square of each 25 ms frame at the 10 ms hop."""
    pcm = np.ascontiguousarray(pcm, dtype=np.float64)
    if len(pcm) < WINDOW:
        return np.zeros(0)
    n = 1 + (len(pcm) - WINDOW) // HOP
    frames = np.lib.stride_tricks.as_strided(
        pcm, shape=(n, WINDOW), strides=(pcm.strides[0] * HOP, pcm.strides[0])
    )
    return np.asarray(np.mean(frames**2, axis=1))


def loudest_second(pcm: np.ndarray) -> np.ndarray:
    """Exactly one second: the loudest window of longer audio, or shorter audio padded."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if len(pcm) <= RATE:
        return np.pad(pcm, (0, RATE - len(pcm)))
    energy = frame_energies(pcm)
    window = np.convolve(energy, np.ones(SECOND), mode="valid")
    start = min(int(np.argmax(window)) * HOP, len(pcm) - RATE)
    return pcm[start : start + RATE]


def _mel(f: np.ndarray | float) -> np.ndarray:
    return np.asarray(2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0))


def _hz(m: np.ndarray | float) -> np.ndarray:
    return np.asarray(700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0))


def filterbank() -> np.ndarray:
    """Triangular mel filters with unit peaks, (bands, n_fft // 2 + 1)."""
    edges = _hz(np.linspace(_mel(F_MIN), _mel(F_MAX), BANDS + 2))
    bins = np.fft.rfftfreq(N_FFT, d=1.0 / RATE)
    bank = np.zeros((BANDS, len(bins)))
    for b in range(BANDS):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        rising = (bins - lo) / max(mid - lo, 1e-9)
        falling = (hi - bins) / max(hi - mid, 1e-9)
        bank[b] = np.clip(np.minimum(rising, falling), 0.0, 1.0)
    return bank


_HANN = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WINDOW) / WINDOW)
_BANK = filterbank()


def log_mel(pcm: np.ndarray) -> np.ndarray:
    """Log-mel frames of audio of any length, (frames, 40) float32."""
    pcm = np.ascontiguousarray(pcm, dtype=np.float64)
    if len(pcm) < WINDOW:
        return np.zeros((0, BANDS), np.float32)
    n = 1 + (len(pcm) - WINDOW) // HOP
    frames = np.lib.stride_tricks.as_strided(
        pcm, shape=(n, WINDOW), strides=(pcm.strides[0] * HOP, pcm.strides[0])
    )
    spectrum = np.fft.rfft(frames * _HANN, n=N_FFT, axis=1)
    power = (spectrum.real**2 + spectrum.imag**2) / N_FFT
    return np.log(power @ _BANK.T + LOG_FLOOR).astype(np.float32)


def samples_for(frames: int) -> int:
    """Samples whose log-mel holds ``frames`` frames."""
    return WINDOW + HOP * (frames - 1)


def pcm_sha256(pcm: np.ndarray) -> str:
    return hashlib.sha256((np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
