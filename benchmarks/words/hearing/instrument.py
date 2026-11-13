"""The voice's measuring instrument (cadence#329, V1, V2, V4): an independent speech recogniser,
Whisper (``openai-whisper``, MIT, model ``base.en``), outside the brain's path. It reads what the
parrot's voice said, as sound made by the library's vocal tract (``cadence.words.hearing.sound``).

Every word is the founder's nonce word, so the instrument reads two ways:

- **choice:** which of the founder's words Whisper finds most likely, by its own probability of
  each word's spelling (capitalized, with a full stop) given the sound; a word said is right
  where that word is the one meant. The word error rate is the share of words said wrong.
- **transcript:** Whisper's free transcript, its letters against the word's spelling, as the
  character error rate (``jiwer``).

The same instrument reads the trainer voices' own takes of the words (the probes heard by the
parrot, ``hearing.script``): its ceiling on these nonce words.

Usage: ``instrument.py <seed> <voice.npz> <audio.npz> <out.json> [<wav folder> <count>]``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import world  # noqa: E402


class Whisper:
    """Whisper as an instrument over a closed set of words."""

    def __init__(self, words: list[str], model: str = "base.en"):
        import torch
        import whisper

        torch.set_num_threads(1)
        self.torch, self.whisper = torch, whisper
        self.model = whisper.load_model(model, device="cpu")
        self.tokens = whisper.tokenizer.get_tokenizer(multilingual=False)
        self.words = list(words)
        prefix = list(self.tokens.sot_sequence_including_notimestamps)
        rows = [prefix + self.tokens.encode(" " + w.capitalize() + ".") + [self.tokens.eot]
                for w in self.words]
        width = max(len(r) for r in rows)
        self.inputs = torch.full((len(rows), width - 1), self.tokens.eot, dtype=torch.long)
        self.targets = torch.full((len(rows), width - 1), -1, dtype=torch.long)
        for i, row in enumerate(rows):
            self.inputs[i, : len(row) - 1] = torch.tensor(row[:-1])
            self.targets[i, len(prefix) - 1: len(row) - 1] = torch.tensor(row[len(prefix):])

    def read(self, samples: np.ndarray) -> tuple[str, str]:
        """The word chosen among the founder's words, and the free transcript."""
        torch, whisper = self.torch, self.whisper
        audio = whisper.pad_or_trim(torch.from_numpy(np.asarray(samples, dtype=np.float32)))
        mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
        with torch.no_grad():
            features = self.model.embed_audio(mel)
            logits = self.model.logits(self.inputs, features.expand(len(self.words), -1, -1))
            logp = torch.log_softmax(logits, dim=-1)
            keep = self.targets >= 0
            picked = logp.gather(2, self.targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            scores = (picked * keep).sum(dim=1)
        chosen = self.words[int(torch.argmax(scores))]
        text = self.model.transcribe(np.asarray(samples, dtype=np.float32), language="en",
                                     temperature=0.0, fp16=False,
                                     condition_on_previous_text=False)["text"]
        return chosen, str(text)


def _letters(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalpha())


def measure(seed: int, voice_path: str, audio_path: str, *, wavs: str | None = None,
            count: int = 0, most: int = 150) -> dict[str, Any]:
    import hearing
    import jiwer
    from scipy.io import wavfile

    from cadence.words.hearing import sound

    lex = world.lexicon(seed)
    words = (lex.kind + lex.colour + lex.place + list(lex.question.values()) + [lex.none]
             + lex.foils)
    instrument = Whisper(words)
    held = np.load(voice_path)
    ends = np.cumsum(held["lengths"])
    items = [(str(m), held["frames"][e - n: e], str(w))
             for m, n, e, w in zip(held["measure"], held["lengths"], ends, held["meant"],
                                   strict=True)]
    out: dict[str, Any] = {"seed": seed, "model": "base.en", "words": len(words)}
    rows: dict[str, list[tuple[bool, float]]] = {}
    examples: list[tuple[str, str, np.ndarray]] = []
    for measure_name, frames, meant in items:
        kept = rows.setdefault(measure_name, [])
        if len(kept) >= most or not meant:
            continue
        if len(frames) < 2:
            kept.append((False, 1.0))
            continue
        samples = sound(frames, rounds=32)
        chosen, text = instrument.read(samples)
        kept.append((chosen == meant, float(jiwer.cer(meant, _letters(text) or "-"))))
        if measure_name in ("V1", "V2", "V4_late") and len(examples) < count:
            examples.append((measure_name, meant, samples))
    sounds = hearing.load(audio_path)
    events = hearing.script(seed)
    probes = [(i, e[1]) for i, e in enumerate(events) if e[0] == "probe"][:most]
    for i, meant in probes:
        chosen, text = instrument.read(sounds[i])
        rows.setdefault("trainers", []).append(
            (chosen == meant, float(jiwer.cer(meant, _letters(text) or "-"))))
    for name, kept in rows.items():
        out[f"{name}_n"] = len(kept)
        out[f"{name}_wer"] = 1.0 - float(np.mean([ok for ok, _ in kept])) if kept else None
        out[f"{name}_cer"] = float(np.mean([c for _, c in kept])) if kept else None
    if wavs:
        folder = Path(wavs)
        folder.mkdir(parents=True, exist_ok=True)
        english = lex.english()
        trainers = {meant: sounds[i] for i, meant in probes}
        for k, (name, meant, samples) in enumerate(examples):
            tag = f"{seed}_{k:02d}_{name}_{meant}_{english.get(meant, meant).replace(' ', '-')}"
            wavfile.write(str(folder / f"{tag}_parrot.wav"), 16000, samples)
            if meant in trainers:
                wavfile.write(str(folder / f"{tag}_trainer.wav"), 16000,
                              np.asarray(trainers[meant], dtype=np.float32))
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1])
    result = measure(seed, sys.argv[2], sys.argv[3],
                     wavs=sys.argv[5] if len(sys.argv) > 5 else None,
                     count=int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    Path(sys.argv[4]).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
