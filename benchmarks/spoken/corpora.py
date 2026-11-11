"""The corpora of the spoken-phrase chamber (cadence#303), each as a list of word clips.

- **Speech Commands v0.02** (Warden 2018, CC BY 4.0): 105,829 one-second clips of 35 words by
  2,618 speakers, with the official speaker-disjoint split (``validation_list.txt`` and
  ``testing_list.txt``) and six background-noise recordings. The base vocabulary and the new
  speakers.
- **AudioMNIST** (Becker, Ackermann, Lapuschkin, Mueller, Samek 2018; MIT licence): 30,000
  recordings of the digits zero to nine by 60 speakers, 50 of each digit per speaker at 48 kHz,
  with each speaker's reported accent. Ten words shared with Speech Commands.
- **Common Voice, single-word target segment** (Mozilla, CC0): the digits, ``yes`` and ``no``
  with each speaker's reported accent, as MP3 (decoded through ``ffmpeg`` where it is
  installed). Twelve words shared with Speech Commands.
- **Synthetic speech** for the tests: each word a chain of three voiced tones whose pitch and
  formants a speaker shifts, so the tests need no download.

A ``Corpus`` holds its clips and returns any clip's samples at 16 kHz; ``noise(split)`` returns
the background recordings a phrase's bed is cut from.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    key = f"spoken_{name}"  # a name of this chamber's own: other benchmarks have their modules
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


audio = _load("audio")

WORDS: tuple[str, ...] = (
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
)
DIGITS: tuple[str, ...] = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
SHARED_COMMON_VOICE: tuple[str, ...] = (*DIGITS, "yes", "no")
SPLITS = ("train", "validation", "test")
BACKGROUND = "_background_noise_"
SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
AUDIOMNIST_URL = "https://github.com/soerenab/AudioMNIST"
COMMON_VOICE_URL = "https://mozilladatacollective.com/datasets/cmkzhp64p00wlno07elrmt20y"
# The share of each background recording a split's beds are cut from, so no bed crosses splits.
NOISE_REGION = {"train": (0.0, 0.8), "validation": (0.8, 0.9), "test": (0.9, 1.0)}


@dataclass(frozen=True)
class Clip:
    """One recording of one word: ``key`` names it within its corpus."""

    key: str
    word: str
    speaker: str
    split: str
    accent: str = ""


class Corpus:
    """A list of word clips and the means to read them."""

    name = "corpus"

    def __init__(self, clips: Iterable[Clip]) -> None:
        self.clips = list(clips)

    def pcm(self, clip: Clip) -> np.ndarray:  # pragma: no cover - each corpus reads its own
        raise NotImplementedError

    def noise(self, split: str) -> list[tuple[str, np.ndarray]]:
        """Background recordings (name, 16 kHz samples): the region ``split``'s beds come from."""
        raise NotImplementedError

    def speakers(self, split: str | None = None) -> dict[str, dict[str, list[Clip]]]:
        """Speaker to word to clips, of one split or all."""
        table: dict[str, dict[str, list[Clip]]] = {}
        for clip in self.clips:
            if split is None or clip.split == split:
                table.setdefault(clip.speaker, {}).setdefault(clip.word, []).append(clip)
        return table


class SpeechCommands(Corpus):
    name = "speech_commands_v0.02"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        held: dict[str, str] = {}
        for split, listing in (("validation", "validation_list.txt"), ("test", "testing_list.txt")):
            for line in (self.root / listing).read_text().split():
                held[line.strip()] = split
        clips = []
        for word in WORDS:
            folder = self.root / word
            if not folder.is_dir():
                raise FileNotFoundError(f"missing word folder {folder}")
            for wav in sorted(folder.glob("*.wav")):
                key = f"{word}/{wav.name}"
                speaker = wav.stem.split("_nohash_")[0]
                clips.append(Clip(key, word, speaker, held.get(key, "train")))
        super().__init__(clips)
        self._noise: list[tuple[str, np.ndarray]] | None = None

    def pcm(self, clip: Clip) -> np.ndarray:
        pcm, rate = audio.read_wav(self.root / clip.key)
        return audio.loudest_second(audio.resample(pcm, rate))

    def noise(self, split: str) -> list[tuple[str, np.ndarray]]:
        if self._noise is None:
            files = sorted((self.root / BACKGROUND).glob("*.wav"))
            if not files:
                raise FileNotFoundError(f"no background recordings under {self.root / BACKGROUND}")
            self._noise = []
            for f in files:
                pcm, rate = audio.read_wav(f)
                self._noise.append((f.name, audio.resample(pcm, rate)))
        lo, hi = NOISE_REGION[split]
        return [(name, pcm[int(lo * len(pcm)) : int(hi * len(pcm))]) for name, pcm in self._noise]


class AudioMNIST(Corpus):
    """``root`` is a clone of the AudioMNIST repository (its ``data`` folder)."""

    name = "audiomnist"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        data = self.root / "data"
        meta = json.loads((data / "audioMNIST_meta.txt").read_text())
        clips = []
        for speaker in sorted(meta):
            accent = str(meta[speaker]["accent"]).strip().lower()
            for wav in sorted((data / speaker).glob("*.wav")):
                digit, who, _ = wav.stem.split("_")
                if who != speaker:
                    raise ValueError(f"{wav} is filed under speaker {speaker}")
                clips.append(
                    Clip(f"{speaker}/{wav.name}", DIGITS[int(digit)], speaker, "test", accent)
                )
        super().__init__(clips)
        self.meta = meta

    def pcm(self, clip: Clip) -> np.ndarray:
        pcm, rate = audio.read_wav(self.root / "data" / clip.key)
        return audio.loudest_second(audio.resample(pcm, rate))


class CommonVoiceSingleWord(Corpus):
    """``root`` is a language folder of the single-word target segment (``validated.tsv`` and
    ``clips/``). Only the twelve words shared with Speech Commands are kept."""

    name = "common_voice_single_word"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        clips = []
        with open(self.root / "validated.tsv", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                word = row.get("sentence", "").strip().lower()
                if word not in SHARED_COMMON_VOICE:
                    continue
                accent = (row.get("accents") or row.get("accent") or "").strip().lower()
                clips.append(Clip(row["path"], word, row["client_id"], "test", accent))
        super().__init__(clips)

    def pcm(self, clip: Clip) -> np.ndarray:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("reading Common Voice's MP3 clips needs ffmpeg on the path")
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "quiet",
                "-i",
                str(self.root / "clips" / clip.key),
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                str(audio.RATE),
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return audio.loudest_second(pcm)


# ------------------------------------------------------------------------------ synthetic speech

SYNTH_RATE = audio.RATE


def _word_plan(word: str) -> list[tuple[float, float, float]]:
    """Three voiced segments per word, each (first formant, second formant, share of the word)."""
    seed = int.from_bytes(word.encode()[:8].ljust(8, b"\0"), "little") % (2**32)
    rng = np.random.default_rng(seed)
    shares = rng.dirichlet((4.0, 4.0, 4.0))
    return [(float(rng.uniform(300, 900)), float(rng.uniform(900, 2600)), float(s)) for s in shares]


def synth_word(word: str, *, pitch: float, formant: float, length: float, seed: int) -> np.ndarray:
    """One second of a synthetic word: voiced segments at the speaker's pitch, formants scaled by
    ``formant``, ``length`` seconds long, centred, with a little breath noise."""
    rng = np.random.default_rng(seed)
    n = int(length * SYNTH_RATE)
    out = np.zeros(n)
    at = 0
    plan = _word_plan(word)
    for i, (f1, f2, share) in enumerate(plan):
        m = n - at if i == len(plan) - 1 else int(share * n)
        t = np.arange(m) / SYNTH_RATE
        jitter = 1.0 + 0.02 * rng.standard_normal()
        segment = np.zeros(m)
        for harmonic in range(1, 24):
            f = harmonic * pitch * jitter
            if f > 7000:
                break
            gain = np.exp(-(((f - f1 * formant) / 160.0) ** 2)) + 0.6 * np.exp(
                -(((f - f2 * formant) / 220.0) ** 2)
            )
            segment += gain * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        out[at : at + m] = segment
        at += m
    envelope = np.clip(np.sin(np.pi * np.arange(n) / max(n - 1, 1)), 0.0, None) ** 0.5
    out = out * envelope + 0.01 * rng.standard_normal(n)
    out = 0.3 * out / (np.abs(out).max() + 1e-9)
    clip = np.zeros(SYNTH_RATE)
    start = (SYNTH_RATE - n) // 2
    clip[start : start + n] = out
    return clip.astype(np.float32)


class Synthetic(Corpus):
    """Synthetic speakers of a vocabulary: ``speakers`` per split for ``train``, ``validation`` and
    ``test``, ``takes`` recordings of each word each; ``dialect`` speakers (split ``test``, accent
    ``synthetic``) shift the formants further than any training speaker."""

    name = "synthetic"

    def __init__(
        self,
        words: tuple[str, ...] = WORDS[:6],
        *,
        speakers: int = 4,
        takes: int = 6,
        dialect: int = 2,
        seed: int = 0,
    ) -> None:
        self.words = words
        rng = np.random.default_rng(seed)
        self.voices: dict[str, tuple[float, float, float]] = {}
        clips = []
        for split in SPLITS:
            for s in range(speakers):
                name = f"{split[:2]}{s}"
                self.voices[name] = (
                    float(rng.uniform(90, 220)),
                    float(rng.uniform(0.9, 1.1)),
                    float(rng.uniform(0.45, 0.65)),
                )
                clips += [
                    Clip(f"{name}/{w}/{k}", w, name, split) for w in words for k in range(takes)
                ]
        for s in range(dialect):
            name = f"di{s}"
            self.voices[name] = (
                float(rng.uniform(90, 220)),
                float(rng.uniform(1.25, 1.35)),
                float(rng.uniform(0.3, 0.4)),
            )
            clips += [
                Clip(f"{name}/{w}/{k}", w, name, "test", "synthetic")
                for w in words
                for k in range(takes)
            ]
        super().__init__(clips)
        self.seed = seed

    def pcm(self, clip: Clip) -> np.ndarray:
        pitch, formant, length = self.voices[clip.speaker]
        take = int(clip.key.rsplit("/", 1)[1])
        seed = int.from_bytes(clip.key.encode()[-8:].rjust(8, b"\0"), "little") % (2**32)
        return synth_word(
            clip.word,
            pitch=pitch * (1.0 + 0.03 * (take % 3)),
            formant=formant,
            length=length,
            seed=seed + self.seed,
        )

    def noise(self, split: str) -> list[tuple[str, np.ndarray]]:
        rng = np.random.default_rng(1000 + SPLITS.index(split) + self.seed)
        white = rng.standard_normal(SYNTH_RATE * 20)
        pink = np.cumsum(white) * 0.002
        pink -= np.convolve(pink, np.ones(400) / 400, mode="same")
        return [("synthetic_noise.wav", (0.05 * white + pink).astype(np.float32))]
