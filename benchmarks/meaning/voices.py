"""The demo's sounds (cadence#334): questions in trainer voices, and the parrot's takes of words.

    python voices.py <model.onnx> <out_dir> '<json: {"questions": [...], "words": [...]}>'

A trainer's voice is a speaker of the Piper model ``en_US-libritts_r-medium`` (code: Piper,
used as a tool; voices: LibriTTS-R, CC BY 4.0). Each question is said by a trainer, at 22.05 kHz.
Each word the parrot may say is the trainer's own take of that word, said alone, kept at 8 kHz:
the parrot's voice says a learned word by replaying the take its memory record keeps.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

TRAINERS = {"anna": 12, "ben": 300}  # LibriTTS-R speaker ids of the two trainers
PARROT_RATE = 8000


def say(voice, text: str, speaker: int) -> tuple[np.ndarray, int]:
    from piper.config import SynthesisConfig

    chunks = list(voice.synthesize(text, syn_config=SynthesisConfig(speaker_id=speaker)))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    return audio, int(chunks[0].sample_rate)


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def resample(audio: np.ndarray, rate: int, target: int) -> np.ndarray:
    n = int(round(len(audio) * target / rate))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)


def main(argv: list[str]) -> None:
    from piper import PiperVoice

    model, out = argv[1], Path(argv[2])
    spec = json.loads(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(model)
    for i, text in enumerate(spec.get("questions", [])):
        name = "anna" if i % 2 == 0 else "ben"
        audio, rate = say(voice, text, TRAINERS[name])
        write(out / f"q{i:02d}.wav", audio, rate)
    for w in spec.get("words", []):
        audio, rate = say(voice, w, TRAINERS["anna"])
        write(out / f"take_{w}.wav", resample(audio, rate, PARROT_RATE), PARROT_RATE)
    (out / "voices.json").write_text(json.dumps(
        {"model": "en_US-libritts_r-medium", "trainers": TRAINERS, "parrot_rate": PARROT_RATE,
         "licence": "LibriTTS-R voices, CC BY 4.0; Piper used as a tool", **spec}, indent=1))


if __name__ == "__main__":
    main(sys.argv)
