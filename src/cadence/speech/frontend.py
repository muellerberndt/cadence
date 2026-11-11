"""Hearing speech: log-mel energies and the time-frequency window a brain reads at each step.

Audio at ``rate`` samples a second becomes log-mel energies of 25 ms frames every 10 ms
(``log_mel``). A step of 40 ms (four frames) hears the latest ``window`` frames, 240 ms by
default, as an image of ``mels`` rows (bands) by ``window`` columns (frames), each window less
its own mean, standardized per band on training windows, clipped and scaled (``Hearing``). A
brain reads it with ``Brain.compose(inputs=(mels, window), ..., learned_fields=features)``.
``LogMel`` joins the two for a reader that hears samples as they arrive
(``cadence.speech.stream``).

The defaults are those of the letters chamber (``benchmarks/letters``), where this front end was
measured on read speech.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RATE = 16000  # samples a second
FRAME = 400  # samples of a frame: 25 ms
HOP = 160  # samples between frames: 10 ms
FFT = 512  # points of a frame's spectrum
MELS = 40  # mel bands
STACK = 4  # frames to a step: 40 ms
WINDOW = 24  # frames a step hears: 240 ms


def mel_bank(mels: int = MELS, *, rate: int = RATE, fft: int = FFT) -> np.ndarray:
    """Triangular filters evenly spaced on the mel scale from 0 Hz to half the rate:
    ``(mels, fft // 2 + 1)``."""
    if mels < 1 or rate <= 0 or fft < 2:
        raise ValueError("mel_bank needs a positive band count, rate and spectrum size")

    def mel(hz: np.ndarray) -> np.ndarray:
        return np.asarray(2595.0 * np.log10(1.0 + hz / 700.0))

    edges = 700.0 * (10 ** (np.linspace(mel(np.array(0.0)), mel(np.array(rate / 2)),
                                        mels + 2) / 2595.0) - 1.0)
    bins = np.fft.rfftfreq(fft, 1.0 / rate)
    bank = np.zeros((mels, len(bins)))
    for m in range(mels):
        low, mid, high = edges[m], edges[m + 1], edges[m + 2]
        rise = (bins - low) / max(mid - low, 1e-9)
        fall = (high - bins) / max(high - mid, 1e-9)
        bank[m] = np.clip(np.minimum(rise, fall), 0.0, None)
    return bank


def log_mel(audio: Any, *, rate: int = RATE, mels: int = MELS, frame: int = FRAME,
            hop: int = HOP, fft: int = FFT, bank: np.ndarray | None = None) -> np.ndarray:
    """Log-mel energies of Hann-windowed frames: ``(frames, mels)``, one frame every ``hop``
    samples. Audio shorter than a frame is padded with zeros to one frame."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("audio is a finite one-dimensional array of samples")
    if frame < 1 or hop < 1 or fft < frame:
        raise ValueError("frame and hop must be positive and fft at least the frame")
    if bank is None:
        bank = mel_bank(mels, rate=rate, fft=fft)
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    count = 1 + (len(audio) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    frames = audio[index] * np.hanning(frame)[None, :]
    power = np.abs(np.fft.rfft(frames, fft)) ** 2
    return np.asarray(np.log(power @ bank.T + 1e-6))


def windows(frames: Any, *, window: int = WINDOW, stack: int = STACK) -> np.ndarray:
    """The ``window`` latest frames at the end of each step of ``stack`` frames, the first frame
    repeated before the audio began: ``(steps, window, mels)``, each window less its own mean.
    A trailing part step is dropped."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or len(frames) == 0:
        raise ValueError("frames is a (frames, mels) array of at least one frame")
    if window < 1 or stack < 1:
        raise ValueError("window and stack must be positive")
    steps = len(frames) // stack
    padded = np.concatenate([np.repeat(frames[:1], window, axis=0), frames])
    ends = stack * (np.arange(steps) + 1) + window
    index = ends[:, None] - window + np.arange(window)[None, :]
    out = padded[index]
    return np.asarray(out - out.mean(axis=(1, 2), keepdims=True))


@dataclass
class Hearing:
    """The step's image: each window standardized per band by ``mean`` and ``std`` (from
    training windows, ``fit``), clipped at ``clip`` and scaled by ``scale``, flattened band by
    band to ``mels * window`` values, the layout of ``Brain.compose(inputs=(mels, window))``."""

    mean: np.ndarray
    std: np.ndarray
    window: int = WINDOW
    stack: int = STACK
    clip: float = 3.0
    scale: float = 0.3
    mels: int = field(init=False)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.std = np.asarray(self.std, dtype=np.float64).reshape(-1)
        if self.mean.shape != self.std.shape or len(self.mean) == 0:
            raise ValueError("mean and std hold one value per band")
        if not (np.isfinite(self.mean).all() and np.isfinite(self.std).all()) or (
                self.std <= 0).any():
            raise ValueError("mean must be finite and std finite and positive")
        if self.window < 1 or self.stack < 1 or self.clip <= 0 or self.scale <= 0:
            raise ValueError("window, stack, clip and scale must be positive")
        self.mels = len(self.mean)

    @property
    def shape(self) -> tuple[int, int]:
        """The image a brain reads: ``(mels, window)``."""
        return (self.mels, self.window)

    @classmethod
    def fit(cls, utterances: Iterable[Any], *, window: int = WINDOW, stack: int = STACK,
            clip: float = 3.0, scale: float = 0.3, floor: float = 1e-3) -> Hearing:
        """Per-band mean and standard deviation over the windows of ``utterances`` (each a
        ``(frames, mels)`` array of ``log_mel``), the deviation at least ``floor``."""
        total = square = None
        count = 0
        for frames in utterances:
            w = windows(frames, window=window, stack=stack)
            if total is None or square is None:
                total, square = np.zeros(w.shape[2]), np.zeros(w.shape[2])
            total += w.sum(axis=(0, 1))
            square += (w**2).sum(axis=(0, 1))
            count += w.shape[0] * w.shape[1]
        if total is None or square is None or count == 0:
            raise ValueError("fit needs at least one utterance of a whole step")
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean**2, 1e-6))
        return cls(mean, np.maximum(std, floor), window=window, stack=stack, clip=clip,
                   scale=scale)

    def steps(self, frames: Any) -> np.ndarray:
        """The observations of an utterance's steps: ``(steps, mels * window)`` float32."""
        return self.rows(windows(frames, window=self.window, stack=self.stack))

    def rows(self, held: Any) -> np.ndarray:
        """Observations of windows ``(steps, window, mels)`` of log-mel frames: each less its
        own mean, standardized, clipped, scaled and laid out band by band."""
        w = np.asarray(held, dtype=np.float64)
        if w.ndim != 3 or w.shape[1] != self.window or w.shape[2] != self.mels:
            raise ValueError(f"windows are (steps, {self.window}, {self.mels})")
        w = w - w.mean(axis=(1, 2), keepdims=True)
        z = np.clip((w - self.mean) / self.std, -self.clip, self.clip) * self.scale
        return np.ascontiguousarray(z.transpose(0, 2, 1)).reshape(len(z), -1).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "window": self.window,
                "stack": self.stack, "clip": self.clip, "scale": self.scale}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hearing:
        return cls(np.asarray(data["mean"]), np.asarray(data["std"]),
                   window=int(data.get("window", WINDOW)), stack=int(data.get("stack", STACK)),
                   clip=float(data.get("clip", 3.0)), scale=float(data.get("scale", 0.3)))


class LogMel:
    """Samples in, a brain's rows out: ``log_mel`` at named settings and a ``Hearing``. Its
    ``window`` and ``hop`` are a frame's length and stride in samples, ``stack`` the frames of a
    step and ``frames`` the frames a step hears, the front end of ``cadence.speech.stream``."""

    def __init__(self, hearing: Hearing, *, rate: int = RATE, frame: int = FRAME, hop: int = HOP,
                 fft: int = FFT) -> None:
        self.hearing = hearing
        self.rate, self.window, self.hop, self.fft = int(rate), int(frame), int(hop), int(fft)
        self.stack, self.frames = hearing.stack, hearing.window
        self.bank = mel_bank(hearing.mels, rate=self.rate, fft=self.fft)

    @property
    def shape(self) -> tuple[int, int]:
        """The image a brain reads: ``inputs=`` of ``Brain.compose``."""
        return self.hearing.shape

    def log_mel(self, samples: Any) -> np.ndarray:
        return log_mel(samples, rate=self.rate, frame=self.window, hop=self.hop, fft=self.fft,
                       bank=self.bank)

    def rows(self, held: Any) -> np.ndarray:
        return self.hearing.rows(held)

    def steps(self, samples: Any) -> np.ndarray:
        """The rows of a whole utterance's steps."""
        return self.hearing.steps(self.log_mel(samples))
