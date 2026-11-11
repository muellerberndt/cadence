"""The streaming reader (cadence.speech.stream): frames and steps as samples arrive."""

from __future__ import annotations

import numpy as np

from cadence import Brain
from cadence.speech.stream import StreamReader

RATE, WINDOW, HOP, STACK, FRAMES, BANDS = 16000, 400, 160, 4, 12, 8


class Toy:
    """A small front end: log energies of 8 bands of 25 ms frames every 10 ms; a step's rows are
    its window of frames, less the window's mean, laid out bands by frames."""

    window, hop, stack, frames = WINDOW, HOP, STACK, FRAMES

    def __init__(self) -> None:
        self.heard: list[np.ndarray] = []

    def log_mel(self, samples: np.ndarray) -> np.ndarray:
        count = 1 + (len(samples) - WINDOW) // HOP
        index = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
        power = np.abs(np.fft.rfft(samples[index] * np.hanning(WINDOW), 64)) ** 2
        return np.log(power[:, 1 : 1 + 4 * BANDS].reshape(count, BANDS, 4).sum(axis=2) + 1e-6)

    def rows(self, windows: np.ndarray) -> np.ndarray:
        w = windows - windows.mean(axis=(1, 2), keepdims=True)
        out = np.ascontiguousarray(0.1 * w.transpose(0, 2, 1)).reshape(len(w), -1)
        self.heard.append(out)
        return out


def whole(front: Toy, audio: np.ndarray) -> np.ndarray:
    """The rows of a whole utterance at once, the first frame repeated before it began."""
    frames = front.log_mel(audio)
    steps = len(frames) // STACK
    padded = np.concatenate([np.repeat(frames[:1], FRAMES, axis=0), frames])
    index = STACK * (np.arange(steps) + 1)[:, None] + np.arange(FRAMES)[None, :]
    w = padded[index]
    w = w - w.mean(axis=(1, 2), keepdims=True)
    return np.ascontiguousarray(0.1 * w.transpose(0, 2, 1)).reshape(steps, -1)


def letters(actions: np.ndarray) -> str:
    out, previous = [], 0
    for a in actions:
        if a and a != previous:
            out.append("abc"[int(a) - 1])
        previous = int(a)
    return "".join(out)


def test_the_reader_hears_what_the_front_end_makes_of_the_whole_utterance():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(RATE + 1234) * np.sin(np.linspace(0, 30, RATE + 1234))
    brain = Brain.compose(BANDS * FRAMES, 4, seed=0, recall=False, backend="cpu")
    front = Toy()
    reader = StreamReader(brain, front, letters, temperature=1.0)
    said = ""
    at = 0
    for size in (37, 1000, 1, 399, 5000, 3, 12000):
        said += reader.push(audio[at : at + size])
        at += size
    said += reader.push(audio[at:])
    expected = whole(Toy(), audio)
    assert len(reader.actions) == len(expected)
    assert np.allclose(np.concatenate(front.heard), expected)
    brain.reset()
    acted = [int(brain.act(row[None], greedy=True)[0]) for row in expected]
    assert reader.actions == acted
    assert said == letters(np.asarray(acted)) == reader.finish()
    assert len(reader.beliefs) == len(acted)
    assert np.allclose([np.exp(b).sum() for b in reader.beliefs], 1.0)
    latency = reader.latency()
    assert latency["step_median_s"] > 0 and latency["seconds_per_audio_second"] > 0
    reader.reset()
    assert reader.actions == [] and reader.push(audio[:100]) == ""


def test_finish_reads_the_beliefs_through_a_word_decoder_where_one_is_given():
    brain = Brain.compose(BANDS * FRAMES, 4, seed=1, recall=False, backend="cpu")
    seen: list[tuple[int, int]] = []

    def words(beliefs: np.ndarray) -> str:
        seen.append(beliefs.shape)
        return "word"

    reader = StreamReader(brain, Toy(), letters, words=words, temperature=1.0)
    reader.push(np.random.default_rng(1).standard_normal(4000))
    assert reader.finish() == "word" and seen == [(len(reader.actions), 4)]
