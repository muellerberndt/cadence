"""Synthetic voices for the hearing chamber (cadence#329): nonce words and questions spoken aloud.

Every word of the words chamber is a nonce word drawn for its founder, so no recording of it
exists: the chamber has it spoken by synthetic voices. A voice is a speaker of a multi-speaker
Piper model (VITS, exported to ONNX), run here with ``onnxruntime``. The chamber builds each
word's phonemes itself from its consonant-vowel syllables (``phonemes``), so no phonemizer runs
and every voice says the same phonemes. The models and their licences:

| Model | Speakers | Trained on | Licence of the data |
| --- | --- | --- | --- |
| ``en_US-libritts_r-medium`` | 904 | LibriTTS-R | CC BY 4.0 |
| ``en_GB-vctk-medium`` | 109 | VCTK 0.92 | CC BY 4.0 |
| ``en_US-arctic-medium`` | 18 | CMU ARCTIC | free use, modification and redistribution |

The models come from ``huggingface.co/rhasspy/piper-voices`` (MIT); ``SUMS`` holds the sha256 of
each file the chamber read. The audio is made on the box that runs the lives and kept in the
lane's bucket, never in git.

A voice here is ``(model, speaker)``; ``say(text, voice, rng)`` gives 16 kHz samples with the
voice's own pace drawn per utterance (``length_scale`` from ``PACE``), the model's own noise
scales, and a level drawn from ``LEVEL`` (decibels below full scale, of the utterance's peak).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RATE = 16000
MODELS = {"libritts": "en_US-libritts_r-medium", "vctk": "en_GB-vctk-medium",
          "arctic": "en_US-arctic-medium"}
SUMS = {  # sha256 of each model file as read for the declared runs (filled from the box)
}
CONSONANTS = {"b": "b", "d": "d", "f": "f", "g": "ɡ", "k": "k", "l": "l", "m": "m", "n": "n",
              "p": "p", "r": "ɹ", "s": "s", "t": "t", "v": "v", "z": "z"}
VOWELS = {"a": "ɑː", "e": "eɪ", "i": "iː", "o": "oʊ", "u": "uː"}
PACE = (0.85, 1.2)  # length_scale drawn per utterance: slower above 1
LEVEL = (-12.0, -1.0)  # the utterance's peak, decibels below full scale


def phonemes(word: str) -> str:
    """A nonce word of consonant-vowel syllables as the IPA the voices were trained on, stress
    on the first syllable's vowel as the voices' phonemizer marks it: ``"bazu"`` is
    ``"bˈɑːzuː"``."""
    if len(word) % 2 or not word:
        raise ValueError(f"a nonce word is consonant-vowel syllables: {word!r}")
    out = []
    for k in range(0, len(word), 2):
        c, v = word[k], word[k + 1]
        if c not in CONSONANTS or v not in VOWELS:
            raise ValueError(f"a nonce word is consonant-vowel syllables: {word!r}")
        out.append(CONSONANTS[c] + ("ˈ" if k == 0 else "") + VOWELS[v])  # noqa: RUF001
    return "".join(out)


def sentence(words: list[str], *, question: bool) -> str:
    """The phonemes of words said together: a space between words, ``?`` or ``.`` at the end."""
    return " ".join(phonemes(w) for w in words) + ("?" if question else ".")


@dataclass(frozen=True)
class Voice:
    model: str  # a key of MODELS
    speaker: int

    def __str__(self) -> str:
        return f"{self.model}:{self.speaker}"


class Voices:
    """The synthetic voices of the models in ``folder`` (each ``<name>.onnx`` with its
    ``<name>.onnx.json``)."""

    def __init__(self, folder: Any, *, threads: int = 1):
        import onnxruntime as ort

        self.folder = Path(folder)
        self.sessions: dict[str, Any] = {}
        self.configs: dict[str, dict[str, Any]] = {}
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        for key, name in MODELS.items():
            path = self.folder / f"{name}.onnx"
            if not path.exists():
                continue
            self.configs[key] = json.loads((self.folder / f"{name}.onnx.json").read_text())
            self.sessions[key] = ort.InferenceSession(str(path), options,
                                                      providers=["CPUExecutionProvider"])

    def speakers(self, model: str) -> int:
        return int(self.configs[model].get("num_speakers", 1))

    def ids(self, model: str, text: str) -> list[int]:
        """Piper's phoneme ids: the start, each phoneme then the pad, the end."""
        table = self.configs[model]["phoneme_id_map"]
        out = list(table["^"]) + list(table["_"])
        for symbol in text:
            if symbol not in table:
                raise ValueError(f"the voice {model} has no phoneme {symbol!r}")
            out += list(table[symbol]) + list(table["_"])
        return out + list(table["$"])

    def say(self, text: str, voice: Voice, rng: np.random.Generator) -> np.ndarray:
        """16 kHz samples of ``text`` (phonemes, as ``sentence`` makes them) in ``voice``."""
        from scipy.signal import resample_poly

        config = self.configs[voice.model]
        inference = config.get("inference", {})
        ids = np.array([self.ids(voice.model, text)], dtype=np.int64)
        pace = float(rng.uniform(*PACE))
        scales = np.array([inference.get("noise_scale", 0.667), pace * inference.get(
            "length_scale", 1.0), inference.get("noise_w", 0.8)], dtype=np.float32)
        feed = {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                "scales": scales}
        if self.speakers(voice.model) > 1:
            feed["sid"] = np.array([int(voice.speaker)], dtype=np.int64)
        audio = np.asarray(self.sessions[voice.model].run(None, feed)[0], dtype=np.float64)
        audio = audio.reshape(-1)
        rate = int(config["audio"]["sample_rate"])
        if rate != RATE:
            g = np.gcd(rate, RATE)
            audio = resample_poly(audio, RATE // g, rate // g)
        peak = float(np.abs(audio).max())
        level = 10 ** (float(rng.uniform(*LEVEL)) / 20.0)
        return np.asarray(audio * (level / peak if peak > 0 else 1.0), dtype=np.float32)
