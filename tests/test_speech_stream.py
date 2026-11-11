"""The streaming reader (cadence.speech.stream): frames and steps as samples arrive."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import Brain
from cadence.speech.decode import Alphabet, collapse
from cadence.speech.frontend import Hearing, log_mel
from cadence.speech.stream import StreamReader

MELS, WINDOW = 8, 12
ALPHABET = Alphabet("ab")  # blank, a, b, space: four symbols


def hearing() -> Hearing:
    return Hearing(np.zeros(MELS), np.ones(MELS), window=WINDOW)


def ears(seed: int) -> Brain:
    return Brain.compose((MELS, WINDOW), ALPHABET.size, seed=seed, learned_fields=4,
                         recall=False, backend="cpu")


def audio(seed: int, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(16000 * seconds)
    return rng.standard_normal(n) * np.abs(np.sin(np.linspace(0, 20, n)))


def test_the_reader_hears_what_the_front_end_makes_of_the_whole_utterance():
    sound = audio(0, 1.07)
    brain = ears(0)
    brain.fields.learning = False
    reader = StreamReader(brain, hearing(), alphabet=ALPHABET)
    said, at = "", 0
    for size in (37, 1000, 1, 399, 5000, 3, 4000):
        said += reader.push(sound[at : at + size])
        at += size
    said += reader.push(sound[at:])
    rows = hearing().steps(log_mel(sound, mels=MELS))
    assert len(reader.actions) == len(rows)
    brain.reset()
    acted = [int(brain.act(row[None], greedy=True)[0]) for row in rows]
    assert reader.actions == acted
    assert said == ALPHABET.decode(collapse(acted)) == reader.finish() == reader.text()
    assert len(reader.beliefs) == len(acted)
    assert np.allclose([np.exp(b).sum() for b in reader.beliefs], 1.0)
    latency = reader.latency()
    assert latency["step_median_s"] > 0 and latency["seconds_per_audio_second"] > 0
    reader.reset()
    assert reader.actions == [] and reader.push(sound[:100]) == ""


def test_finish_reads_the_beliefs_against_a_word_list_where_one_is_given():
    class Words:
        def __init__(self) -> None:
            self.seen: list[tuple[int, ...]] = []

        def decode(self, beliefs: np.ndarray) -> str:
            self.seen.append(beliefs.shape)
            return "ab"

    words = Words()
    reader = StreamReader(ears(1), hearing(), alphabet=ALPHABET, words=words)  # type: ignore[arg-type]
    reader.push(audio(1, 0.5))
    assert reader.finish() == "ab" and words.seen == [(len(reader.actions), ALPHABET.size)]


def test_a_brain_of_another_symbol_count_is_refused():
    with pytest.raises(ValueError):
        StreamReader(ears(2), hearing())
