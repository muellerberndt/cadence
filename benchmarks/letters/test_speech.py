"""The read-speech adapter, without the corpus; the corpus test skips where it is absent."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("speech", _here / "speech.py")
assert _spec is not None and _spec.loader is not None
sp = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("speech", sp)
_spec.loader.exec_module(sp)


def test_steps_are_forty_milliseconds_of_log_mel() -> None:
    bank = sp.mel_bank(40)
    assert bank.shape == (40, 257) and (bank >= 0).all()
    second = np.sin(np.linspace(0, 2000 * np.pi, sp.RATE))
    steps = sp.steps_of(second, bank)
    assert steps.shape == (24, 160) and np.isfinite(steps).all()


def test_text_round_trips_as_symbols() -> None:
    symbols = sp.encode("Don't  STOP, now")
    assert sp.decode(symbols.tolist()) == "don't stop now"
    assert symbols.max() < sp.SYMBOLS and (symbols > 0).all()
    path = sp.even_path(10, symbols[:3])
    assert sp.lt.collapse(path) == symbols[:3].tolist()


def test_the_reading_counts_characters_and_words() -> None:
    text = sp.encode("ab cd")
    held = [(np.zeros((8, 4)), text)]
    perfect = [np.array([1, 1, 0, 2, 28, 3, 0, 4])]
    assert sp.read(perfect, held)["characters"] == 1.0
    spaced = [np.array([1, 1, 0, 2, 0, 3, 0, 4])]
    out = sp.read(spaced, held)
    assert out["words"] < 1.0 and out["words_lost_on_spacing_alone"] == 1.0


def test_the_located_terms_read_one_utterance() -> None:
    text = sp.encode("ab")
    logp = np.log(np.full((4, sp.SYMBOLS), 0.01))
    for t, symbol in enumerate([1, 1, 2, 0]):
        logp[t, symbol] = 0.0
    said = sp.locate([np.array([1, 1, 2, 0])], [logp], [(np.zeros((4, 1)), text)])
    assert said["letter_from_sound"] == 1.0 and said["emission_given_letter"] == 1.0
    assert said["word_given_letters"] == 1.0
    silent = sp.locate([np.zeros(4, dtype=np.int64)], [logp], [(np.zeros((4, 1)), text)])
    assert silent["letter_from_sound"] == 1.0 and silent["emission_given_letter"] == 0.0


ROOT = os.environ.get("LIBRISPEECH_ROOT")


@pytest.mark.skipif(not ROOT or not (Path(ROOT) / "LibriSpeech" / "dev-clean").exists(),
                    reason="LibriSpeech is not on this machine (LIBRISPEECH_ROOT)")
def test_the_corpus_lists_utterances() -> None:
    listing = sp.utterances(Path(str(ROOT)), "dev-clean")
    assert len(listing) == 2703
    assert listing[0][1].exists()
