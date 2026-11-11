"""A streaming reader: audio samples in as they arrive, text out as the brain says it.

``StreamReader`` holds one stream of a trained recogniser. ``push(samples)`` takes the next
samples of 16 kHz audio, in blocks of any size; as each 10 ms frame completes, the front end
turns it into log-mel energies, and as each step's frames complete (four frames, 40 ms), the
brain hears the step's window and acts. ``push`` returns the text the actions add, read through
the letter decoder (repeats and blanks collapsed); ``finish()`` returns the whole transcript,
read through a word-list decoder where one is given. ``latency()`` reports the seconds each
step took on this machine.

The reader hears exactly what the front end makes of a whole utterance: the windows it builds
from frames as they arrive are the windows the front end builds from the utterance at once,
the first frame repeated before the utterance began.

The brain acts from where it stands: ``StreamReader`` resets it at the start and between
utterances (``reset``). Its learned fields keep learning at every act unless the caller holds
them (``brain.fields.learning = False``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

__all__ = ["Frontend", "StreamReader"]


class Frontend(Protocol):
    """What the reader needs of a front end.

    ``window`` and ``hop`` are a frame's length and its stride in samples, ``stack`` the frames
    of one step and ``frames`` the frames a step's window holds; ``log_mel(samples)`` returns the
    log-mel energies ``(n, bands)`` of the whole frames of ``samples``; ``rows(windows)`` turns
    windows ``(steps, frames, bands)`` into the brain's rows ``(steps, inputs)``."""

    window: int
    hop: int
    stack: int
    frames: int

    def log_mel(self, samples: np.ndarray) -> np.ndarray: ...

    def rows(self, windows: np.ndarray) -> np.ndarray: ...


class StreamReader:
    """One stream of ``brain`` heard through ``frontend``.

    ``letters`` reads a sequence of actions as text (the letter decoder: repeats and blanks
    collapsed); ``words``, optional, reads the log beliefs of a whole utterance as text (a
    word-list decoder) for ``finish``. ``temperature`` divides the motor activity before the
    softmax of the log beliefs (the brain's own when None)."""

    def __init__(
        self,
        brain: Any,
        frontend: Frontend,
        letters: Callable[[np.ndarray], str],
        *,
        words: Callable[[np.ndarray], str] | None = None,
        temperature: float | None = None,
    ) -> None:
        self.brain, self.frontend = brain, frontend
        self.letters, self.words = letters, words
        if temperature is None:
            temperature = float(brain.describe()["genes"]["temperature"])
        self.temperature = float(temperature)
        self._motor = np.asarray(brain.connectome.populations["motor"], dtype=np.int64)
        self.timings: list[float] = []
        self.reset()

    def reset(self) -> None:
        """Begin a new utterance: the brain at rest, no samples, frames or actions held."""
        self.brain.reset()
        self._samples = np.zeros(0, dtype=np.float64)
        self._frames: list[np.ndarray] = []  # the frames a later window may read
        self._first: np.ndarray | None = None
        self._count = 0  # frames made so far
        self._steps = 0
        self.actions: list[int] = []
        self.beliefs: list[np.ndarray] = []
        self._said = ""

    # -- hearing

    def push(self, samples: Any) -> str:
        """Hear the next samples; return the text they add to the transcript (the whole
        transcript so far where the letter decoder revised what it had read)."""
        block = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not np.isfinite(block).all():
            raise ValueError("audio samples must be finite")
        self._samples = np.concatenate([self._samples, block])
        f = self.frontend
        whole = 0 if len(self._samples) < f.window else 1 + (len(self._samples) - f.window) // f.hop
        if whole:
            made = np.asarray(f.log_mel(self._samples[: f.window + f.hop * (whole - 1)]))
            if len(made) != whole:
                raise ValueError("the front end made another number of frames than it was given")
            if self._first is None:
                self._first = made[0]
            self._frames.extend(made)
            self._count += whole
            self._samples = self._samples[f.hop * whole :]
            while (self._steps + 1) * f.stack <= self._count:
                self._step()
        return self._new_text()

    def finish(self) -> str:
        """The utterance's transcript: the word-list decoder's where one is given, else the
        letter decoder's."""
        if self.words is not None and self.beliefs:
            return self.words(np.stack(self.beliefs))
        return self.letters(np.asarray(self.actions, dtype=np.int64))

    def latency(self) -> dict[str, float]:
        """Seconds per step on this machine: median and 95th percentile, and seconds per second
        of audio at the median."""
        if not self.timings:
            return {}
        f = self.frontend
        median = float(np.median(self.timings))
        return {
            "step_median_s": median,
            "step_p95_s": float(np.percentile(self.timings, 95)),
            "seconds_per_audio_second": median / (f.stack * f.hop / 16000.0),
        }

    # -- one step

    def _step(self) -> None:
        f = self.frontend
        assert self._first is not None
        end = (self._steps + 1) * f.stack  # the step's window ends at this frame (exclusive)
        start = end - f.frames
        held_from = self._count - len(self._frames)
        window = [self._frames[i - held_from] if i >= 0 else self._first for i in range(start, end)]
        ticked = time.perf_counter()
        row = np.asarray(f.rows(np.stack(window)[None]))
        action = int(np.asarray(self.brain.act(row, greedy=True)).reshape(-1)[0])
        drive = np.asarray(self.brain.basal_ganglia.state.activation).reshape(-1)[self._motor]
        drive = drive / self.temperature
        top = drive.max()
        self.timings.append(time.perf_counter() - ticked)
        self.beliefs.append(drive - top - np.log(np.sum(np.exp(drive - top))))
        self.actions.append(action)
        self._steps += 1
        # frames no later window reads leave the reader
        keep_from = (self._steps + 1) * f.stack - f.frames
        drop = min(max(0, keep_from - held_from), len(self._frames))
        del self._frames[:drop]

    def _new_text(self) -> str:
        text = self.letters(np.asarray(self.actions, dtype=np.int64))
        new = text[len(self._said) :] if text.startswith(self._said) else text
        self._said = text
        return new
