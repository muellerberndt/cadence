"""A streaming reader: audio samples in as they arrive, text out as the brain says it.

``StreamReader`` holds one stream of a trained recogniser. ``push(samples)`` takes the next
samples of 16 kHz audio, in blocks of any size. As each 10 ms frame completes, the front end
turns it into log-mel energies (``frontend.log_mel``); as each 40 ms step's frames complete, the
brain hears the step's window (``Hearing``) and acts. ``push`` returns the text the actions add,
read by the letter decoder (repeats and blanks collapsed, ``decode.collapse``); ``finish()``
returns the whole transcript, read against a word list (``decode.WordBeam``) where one is given.
``latency()`` reports the seconds each step took on this machine.

The reader hears what the front end makes of the whole utterance at once
(``Hearing.steps(log_mel(audio))``): the same windows, the first frame repeated before the
utterance began, step by step as the frames arrive.

The brain acts from where it stands: the reader resets it at the start and at ``reset`` (a new
utterance). Its learned fields keep learning at every act unless the caller holds them
(``brain.fields.learning = False``).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .decode import Alphabet, WordBeam, collapse
from .frontend import FRAME, HOP, RATE, Hearing, log_mel, mel_bank

__all__ = ["StreamReader"]


class StreamReader:
    """One stream of ``brain`` heard through ``hearing``.

    ``alphabet`` names the brain's symbols (``Alphabet()``: blank, ``a``-``z``, the apostrophe
    and space); ``words``, optional, reads the whole utterance's beliefs for ``finish``;
    ``temperature`` divides the motor activity in the log beliefs (the brain's own when None)."""

    def __init__(
        self,
        brain: Any,
        hearing: Hearing,
        *,
        alphabet: Alphabet | None = None,
        words: WordBeam | None = None,
        temperature: float | None = None,
    ) -> None:
        self.brain, self.hearing = brain, hearing
        self.alphabet = alphabet or Alphabet()
        self.words = words
        if temperature is None:
            temperature = float(brain.describe()["genes"]["temperature"])
        self.temperature = float(temperature)
        self._motor = np.asarray(brain.connectome.populations["motor"], dtype=np.int64)
        if len(self._motor) != self.alphabet.size:
            raise ValueError(
                f"the brain says {len(self._motor)} symbols where the alphabet holds "
                f"{self.alphabet.size}"
            )
        self._bank = mel_bank(hearing.mels)
        # a step reads its window from the latest whole steps of frames
        stack = hearing.stack
        self._span = -(-hearing.window // stack) * stack
        self.timings: list[float] = []
        self.reset()

    def reset(self) -> None:
        """Begin a new utterance: the brain at rest, no samples, frames or actions held."""
        self.brain.reset()
        self._samples = np.zeros(0, dtype=np.float64)
        self._frames: list[np.ndarray] = []  # the latest frames a later window may read
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
        whole = 0 if len(self._samples) < FRAME else 1 + (len(self._samples) - FRAME) // HOP
        if whole:
            made = log_mel(
                self._samples[: FRAME + HOP * (whole - 1)], mels=self.hearing.mels, bank=self._bank
            )
            if self._first is None:
                self._first = made[0]
            self._frames.extend(made)
            self._count += whole
            self._samples = self._samples[HOP * whole :]
            while (self._steps + 1) * self.hearing.stack <= self._count:
                self._step()
        return self._new_text()

    def text(self) -> str:
        """The letter decoder's transcript so far."""
        return self.alphabet.decode(collapse(self.actions))

    def finish(self) -> str:
        """The utterance's transcript: the word-list decoder's where one is given, else the
        letter decoder's."""
        if self.words is not None and self.beliefs:
            return self.words.decode(np.stack(self.beliefs))
        return self.text()

    def latency(self) -> dict[str, float]:
        """Seconds per step on this machine: the median and the 95th percentile, and seconds per
        second of audio at the median."""
        if not self.timings:
            return {}
        median = float(np.median(self.timings))
        step = self.hearing.stack * HOP / RATE
        return {
            "step_median_s": median,
            "step_p95_s": float(np.percentile(self.timings, 95)),
            "seconds_per_audio_second": median / step,
        }

    # -- one step

    def _step(self) -> None:
        assert self._first is not None
        end = (self._steps + 1) * self.hearing.stack  # the step's frames end here (exclusive)
        held_from = self._count - len(self._frames)
        span = [
            self._frames[i - held_from] if i >= 0 else self._first
            for i in range(end - self._span, end)
        ]
        ticked = time.perf_counter()
        row = self.hearing.steps(np.stack(span))[-1:]
        action = int(np.asarray(self.brain.act(row, greedy=True)).reshape(-1)[0])
        drive = np.asarray(self.brain.basal_ganglia.state.activation).reshape(-1)[self._motor]
        drive = drive / self.temperature
        top = drive.max()
        self.timings.append(time.perf_counter() - ticked)
        self.beliefs.append(drive - top - np.log(np.sum(np.exp(drive - top))))
        self.actions.append(action)
        self._steps += 1
        # frames no later window reads leave the reader
        keep_from = (self._steps + 1) * self.hearing.stack - self._span
        del self._frames[: min(max(0, keep_from - held_from), len(self._frames))]

    def _new_text(self) -> str:
        text = self.text()
        new = text[len(self._said) :] if text.startswith(self._said) else text
        self._said = text
        return new
