"""Phrases of one speaker's words, heard as a stream of half-second steps (cadence#303).

A phrase is two to four word clips of one speaker in a row over a bed of background noise, the
second run's construction of cadence#183 (its repaired ``end``): each clip keeps ``PAD`` frames
around its extent (the frames from its first to its last above a tenth of its peak energy) with
a raised-cosine edge; the first extent starts ``LEAD`` frames into the stream and ``INTERVAL``
silent frames separate two extents (at most 0.85 s); the bed's loudest frame lies at least 3 dB
below the floor of the phrase's weakest word.

The brain hears the stream in steps of ``STEP`` frames (0.49 s): at step ``k`` the window of the
latest ``WINDOW`` frames, ``[49 (k - 1), 49 (k + 1))``. Targets per step: a word at the first step
whose window holds its last extent frame; ``end`` two steps after the last word, and once more
on the step that closes the stream; silence elsewhere.

Sets:
- ``base_train``: the base corpus's training speakers, lengths 2 and 3;
- ``base_validation``, ``base_test``: its held-out speakers, lengths 2, 3 and 4 (4 never taught);
  the speakers chosen as new speakers are left out of ``base_test``, which is the retention panel;
- for each new speaker or dialect speaker: ``adapt`` (labelled phrases heard one at a time, in
  order, up to the most minutes read) and ``evaluate`` (that speaker's other recordings): the
  speaker's clips of each word are dealt to the two sides, so no recording is on both.

    PYTHONPATH=src python benchmarks/spoken/phrases.py build --speech-commands /data/sc \\
        --audiomnist /data/audiomnist/repo --out /data/phrases --workers 30
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
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
corpora = _load("corpora")

FORMAT = "cadence/spoken-phrases/1"
SEED = 30311
STEP = 49  # frames per step: half a second
WINDOW = 2 * STEP  # frames the brain hears at a step: the latest second
LEAD = (25, 45)
INTERVAL = (50, 85)
PAD = 25
FADE = 160  # samples of each kept clip's raised-cosine edge: 10 ms
END_AFTER = 2
TAIL = 1
FLOOR = 0.1  # an extent's frames are above this share of the clip's peak energy
BED_MARGIN = 0.5  # the bed's loudest frame below this share of the weakest word's floor: 3 dB
BED_GAIN_LOG10 = (-3.0, -1.5)
LONGEST = 4
WORDS = corpora.WORDS
SILENCE = len(WORDS)  # 35: the step carries no word
END = len(WORDS) + 1  # 36: the phrase is over
ACTIONS = len(WORDS) + 2
LABELS: tuple[str, ...] = (*WORDS, "_silence_", "end")


@dataclass(frozen=True)
class Phrase:
    """A planned phrase: one speaker's clips in order and the cut of its bed."""

    set: str
    speaker: str
    clips: tuple[str, ...]
    words: tuple[str, ...]
    noise: str
    noise_start: int
    noise_gain: float
    accent: str = ""

    @property
    def id(self) -> str:
        body = "|".join(
            [
                self.set,
                self.speaker,
                *self.clips,
                self.noise,
                str(self.noise_start),
                f"{self.noise_gain:.6g}",
            ]
        )
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def extent(pcm: np.ndarray) -> tuple[int, int, float]:
    """First and last frame above the floor, and the clip's peak frame energy."""
    energy = audio.frame_energies(pcm)
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 0:
        return 0, max(len(energy) - 1, 0), 0.0
    above = np.flatnonzero(energy >= FLOOR * peak)
    return int(above[0]), int(above[-1]), peak


def word_steps(spans: list[tuple[int, int]]) -> list[int]:
    """Each word's target step (1-based): the first whose window holds its last extent frame."""
    steps: list[int] = []
    for _, end in spans:
        k = max(1, int(np.ceil(end / STEP)) - 1)
        steps.append(max(k, steps[-1] + 1) if steps else k)
    return steps


def targets(words: tuple[str, ...], steps: list[int]) -> np.ndarray:
    last = steps[-1]
    target = np.full(last + END_AFTER + TAIL, SILENCE, np.int16)
    for word, k in zip(words, steps, strict=True):
        target[k - 1] = WORDS.index(word)
    target[last + END_AFTER - 1 :] = END
    return target


def window(logmel: np.ndarray, step: int) -> np.ndarray:
    """The frames heard at 1-based ``step``: ``[STEP (step - 1), STEP (step + 1))``."""
    start = STEP * (step - 1)
    return logmel[..., start : start + WINDOW, :]


def render(
    corpus: Any, phrase: Phrase, rng: np.random.Generator, bed: np.ndarray
) -> dict[str, Any]:
    """The phrase's log-mel stream, its targets and its seconds."""
    pcms = [corpus.pcm(clip) for clip in phrase_clips(corpus, phrase)]
    found = [extent(p) for p in pcms]
    lead = int(rng.integers(LEAD[0], LEAD[1] + 1))
    gaps = [int(g) for g in rng.integers(INTERVAL[0], INTERVAL[1] + 1, size=len(pcms) - 1)]
    spans, at = [], lead
    for i, (first, last, _) in enumerate(found):
        spans.append((at, at + last - first + 1))
        at = spans[-1][1] + (gaps[i] if i < len(gaps) else 0)
    ks = word_steps(spans)
    target = targets(phrase.words, ks)
    frames = STEP * (len(target) + 1)
    total = audio.samples_for(frames)
    start = min(phrase.noise_start, max(0, len(bed) - total))
    cut = bed[start : start + total].astype(np.float64)
    cut = np.pad(cut, (0, total - len(cut)))
    gain = phrase.noise_gain
    loudest = float(audio.frame_energies(cut * gain).max())
    limit = BED_MARGIN * min(FLOOR * peak for _, _, peak in found)
    if loudest >= limit > 0:
        gain *= float(np.sqrt(limit / loudest)) * (1 - 1e-6)
    pcm = cut * gain
    fade = 0.5 - 0.5 * np.cos(np.pi * (np.arange(FADE) + 0.5) / FADE)
    for clip_pcm, (first, last, _), (begin, _) in zip(pcms, found, spans, strict=True):
        a = max(0, first - PAD)
        z = min(audio.SECOND, last + PAD + 1)
        kept = clip_pcm[audio.HOP * a : min(len(clip_pcm), audio.HOP * (z - 1) + audio.WINDOW)]
        kept = kept.astype(np.float64)
        n = min(FADE, len(kept) // 2)
        kept[:n] *= fade[:n]
        kept[len(kept) - n :] *= fade[:n][::-1]
        origin = audio.HOP * (begin - first + a)
        end = min(len(pcm), origin + len(kept))
        pcm[origin:end] += kept[: end - origin]
    pcm = np.clip(pcm, -1.0, 1.0).astype(np.float32)
    mel = audio.log_mel(pcm)[:frames]
    return {
        "mel": mel.astype(np.float16),
        "target": target,
        "steps": len(target),
        "frames": frames,
        "seconds": total / audio.RATE,
        "sha": audio.pcm_sha256(pcm),
        "word_steps": ks,
        "bed_capped": bool(gain < phrase.noise_gain),
    }


def phrase_clips(corpus: Any, phrase: Phrase) -> list[Any]:
    index = _clip_index(corpus)
    return [index[key] for key in phrase.clips]


_INDEX: dict[int, dict[str, Any]] = {}


def _clip_index(corpus: Any) -> dict[str, Any]:
    if id(corpus) not in _INDEX:
        _INDEX[id(corpus)] = {clip.key: clip for clip in corpus.clips}
    return _INDEX[id(corpus)]


# ---------------------------------------------------------------------------------- planning


def _bed(corpus: Any, split: str, rng: np.random.Generator, length: int) -> tuple[str, int, float]:
    beds = corpus.noise(split)
    name, pcm = beds[int(rng.integers(len(beds)))]
    total = audio.samples_for(STEP * (_most_steps(length) + 1))
    start = int(rng.integers(0, max(1, len(pcm) - total)))
    return name, start, float(10 ** rng.uniform(*BED_GAIN_LOG10))


def _most_steps(length: int) -> int:
    longest = LEAD[1] + length * audio.SECOND + (length - 1) * INTERVAL[1]
    return int(np.ceil(longest / STEP)) + END_AFTER + TAIL


def plan_base(
    corpus: Any,
    split: str,
    name: str,
    counts: dict[int, int],
    seed: int,
    exclude: frozenset[str] | set[str] = frozenset(),
) -> list[Phrase]:
    """Phrases of ``split``'s speakers: a clip drawn uniformly, then further clips of its speaker
    with words not yet in the phrase."""
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    table = {s: w for s, w in corpus.speakers(split).items() if s not in exclude}
    pool = [c for s in sorted(table) for w in sorted(table[s]) for c in table[s][w]]
    out = []
    for length in sorted(counts):
        made = 0
        while made < counts[length]:
            first = pool[int(rng.integers(len(pool)))]
            words = table[first.speaker]
            if len(words) < length:
                continue
            others = [w for w in sorted(words) if w != first.word]
            chosen = [first] + [
                words[w][int(rng.integers(len(words[w])))]
                for w in rng.choice(others, length - 1, replace=False)
            ]
            order = rng.permutation(length)
            picked = [chosen[i] for i in order]
            noise, start, gain = _bed(corpus, split, rng, length)
            out.append(
                Phrase(
                    name,
                    first.speaker,
                    tuple(c.key for c in picked),
                    tuple(c.word for c in picked),
                    noise,
                    start,
                    gain,
                    first.accent,
                )
            )
            made += 1
    return out


def plan_speaker(
    corpus: Any, speaker: str, split: str, adapt: int, evaluate: int, seed: int, tag: str
) -> dict[str, list[Phrase]]:
    """A new speaker's ``adapt`` and ``evaluate`` phrases, lengths 2 and 3 alternating, from
    disjoint recordings: of each word's clips in a shuffled order, two in three go to ``adapt``
    and one to ``evaluate``. A side whose clips run out deals them again in a fresh order (the
    build's summary counts ``clips`` against ``distinct_clips``)."""
    rng = np.random.default_rng([seed, int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16)])
    words = corpus.speakers()[speaker]
    sides: dict[str, list[Any]] = {"adapt": [], "evaluate": []}
    for word in sorted(words):
        clips = [words[word][int(i)] for i in rng.permutation(len(words[word]))]
        offset = int(rng.integers(3))
        for j, clip in enumerate(clips):
            sides["evaluate" if (j + offset) % 3 == 0 else "adapt"].append(clip)
    out: dict[str, list[Phrase]] = {}
    for side, count in (("adapt", adapt), ("evaluate", evaluate)):
        deck: list[Any] = []
        phrases = []
        for i in range(count):
            length = 2 + i % 2
            picked: list[Any] = []
            while len(picked) < length:
                if not deck:
                    deck = [sides[side][int(j)] for j in rng.permutation(len(sides[side]))]
                at = next(
                    (j for j, c in enumerate(deck) if c.word not in {p.word for p in picked}), None
                )
                if at is None:
                    deck = deck + [sides[side][int(j)] for j in rng.permutation(len(sides[side]))]
                    continue
                picked.append(deck.pop(at))
            noise, start, gain = _bed(corpus, split, rng, length)
            phrases.append(
                Phrase(
                    f"{tag}/{side}",
                    speaker,
                    tuple(c.key for c in picked),
                    tuple(c.word for c in picked),
                    noise,
                    start,
                    gain,
                    picked[0].accent,
                )
            )
        out[side] = phrases
    return out


# ---------------------------------------------------------------------------------- the sets


class Phrases:
    """A built set in memory: log-mel streams, per-step targets and the phrase records."""

    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as z:
            self.logmel = z["logmel"]
            self.target = z["target"].astype(np.int64)
            self.steps = z["steps"].astype(np.int64)
            self.length = z["length"].astype(np.int64)
            self.words = z["words"].astype(np.int64)
            self.speaker = z["speaker"]
            self.seconds = z["seconds"].astype(np.float64)
        self.path = Path(path)

    def __len__(self) -> int:
        return len(self.steps)

    def reference(self, row: int) -> tuple[int, ...]:
        return tuple(int(w) for w in self.words[row, : self.length[row]])

    @classmethod
    def build(
        cls, corpus: Any, phrases: list[Phrase], path: str | Path, split: str, seed: int = SEED
    ) -> dict[str, Any]:
        """Render ``phrases`` (beds from ``split``'s region) into ``path`` (.npz); a summary."""
        beds = dict(corpus.noise(split))
        built = []
        for i, phrase in enumerate(phrases):
            rng = np.random.default_rng([seed, i, int(phrase.id, 16) % (2**31)])
            built.append(render(corpus, phrase, rng, beds[phrase.noise]))
        return cls.save(path, phrases, built)

    @staticmethod
    def save(
        path: str | Path, phrases: list[Phrase], built: list[dict[str, Any]]
    ) -> dict[str, Any]:
        n = len(built)
        frames = max(b["frames"] for b in built)
        steps = max(b["steps"] for b in built)
        logmel = np.full((n, frames, audio.BANDS), np.log(audio.LOG_FLOOR), np.float16)
        target = np.full((n, steps), -1, np.int16)
        words = np.full((n, LONGEST), -1, np.int16)
        for i, (p, b) in enumerate(zip(phrases, built, strict=True)):
            logmel[i, : b["frames"]] = b["mel"]
            target[i, : b["steps"]] = b["target"]
            words[i, : len(p.words)] = [WORDS.index(w) for w in p.words]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            logmel=logmel,
            target=target,
            words=words,
            length=np.array([len(p.words) for p in phrases], np.int16),
            steps=np.array([b["steps"] for b in built], np.int16),
            seconds=np.array([b["seconds"] for b in built]),
            speaker=np.array([p.speaker for p in phrases]),
            uid=np.array([p.id for p in phrases]),
        )
        records = [
            {
                **asdict(p),
                "id": p.id,
                "sha": b["sha"],
                "steps": b["steps"],
                "bed_capped": b["bed_capped"],
            }
            for p, b in zip(phrases, built, strict=True)
        ]
        path.with_suffix(".json").write_text(json.dumps(records))
        clips = [c for p in phrases for c in p.clips]
        return {
            "phrases": n,
            "speakers": len({p.speaker for p in phrases}),
            "lengths": {
                str(k): sum(len(p.words) == k for p in phrases)
                for k in sorted({len(p.words) for p in phrases})
            },
            "seconds": round(float(sum(b["seconds"] for b in built)), 2),
            "clips": len(clips),
            "distinct_clips": len(set(clips)),
            "beds_capped": int(sum(b["bed_capped"] for b in built)),
            "npz_sha256": audio.sha256_file(path),
        }


def _render_chunk(
    args: tuple[str, dict[str, str], str, list[tuple[int, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    kind, roots, split, chunk = args
    corpus = open_corpus(kind, roots)
    beds = dict(corpus.noise(split))
    out = []
    for i, record in chunk:
        phrase = Phrase(
            **{**record, "clips": tuple(record["clips"]), "words": tuple(record["words"])}
        )
        rng = np.random.default_rng([SEED, i, int(phrase.id, 16) % (2**31)])
        out.append(render(corpus, phrase, rng, beds[phrase.noise]))
    return out


def open_corpus(kind: str, roots: dict[str, str]) -> Any:
    if kind == "speech_commands":
        return corpora.SpeechCommands(roots["speech_commands"])
    if kind == "audiomnist":
        return _Bedded(
            corpora.AudioMNIST(roots["audiomnist"]),
            corpora.SpeechCommands(roots["speech_commands"]),
        )
    if kind == "common_voice":
        return _Bedded(
            corpora.CommonVoiceSingleWord(roots["common_voice"]),
            corpora.SpeechCommands(roots["speech_commands"]),
        )
    raise ValueError(kind)


class _Bedded:
    """A corpus of clips whose beds are cut from another corpus's background recordings."""

    def __init__(self, clips: Any, beds: Any) -> None:
        self.inner, self.beds = clips, beds
        self.clips, self.name = clips.clips, clips.name

    def pcm(self, clip: Any) -> np.ndarray:
        return np.asarray(self.inner.pcm(clip))

    def noise(self, split: str) -> list[tuple[str, np.ndarray]]:
        return list(self.beds.noise(split))

    def speakers(self, split: str | None = None) -> dict[str, dict[str, list[Any]]]:
        return dict(self.inner.speakers(split))


def build_set(
    kind: str, roots: dict[str, str], phrases: list[Phrase], path: Path, split: str, workers: int
) -> dict[str, Any]:
    records = [(i, {k: v for k, v in asdict(p).items()}) for i, p in enumerate(phrases)]
    size = max(8, len(records) // (workers * 6))
    chunks = [(kind, roots, split, records[i : i + size]) for i in range(0, len(records), size)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            built = [b for part in pool.map(_render_chunk, chunks) for b in part]
    else:
        built = [b for chunk in chunks for b in _render_chunk(chunk)]
    return Phrases.save(path, phrases, built)


def fresh_speakers(corpus: Any, split: str, count: int, least: int) -> list[str]:
    """The ``count`` speakers of ``split`` with the most clips, each with at least ``least``."""
    table = corpus.speakers(split)
    ranked = sorted(table, key=lambda s: (-sum(len(c) for c in table[s].values()), s))
    return [s for s in ranked if sum(len(c) for c in table[s].values()) >= least][:count]


def build(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = {"speech_commands": args.speech_commands}
    if args.audiomnist:
        roots["audiomnist"] = args.audiomnist
    if args.common_voice:
        roots["common_voice"] = args.common_voice
    base = corpora.SpeechCommands(args.speech_commands)
    started = time.time()
    tarball = Path(args.speech_commands).parent / "speech_commands_v0.02.tar.gz"
    summary: dict[str, Any] = {
        "format": FORMAT,
        "seed": SEED,
        "labels": list(LABELS),
        "rules": {
            "step": STEP,
            "window": WINDOW,
            "lead": LEAD,
            "interval": INTERVAL,
            "pad": PAD,
            "end_after": END_AFTER,
            "tail": TAIL,
            "floor": FLOOR,
            "bed_margin": BED_MARGIN,
            "bed_gain_log10": BED_GAIN_LOG10,
        },
        "corpora": {
            "speech_commands": {
                "url": corpora.SPEECH_COMMANDS_URL,
                "licence": "CC BY 4.0",
                "tarball_sha256": audio.sha256_file(tarball) if tarball.exists() else None,
                "clips": len(base.clips),
                "speakers": len(base.speakers()),
                "per_split": {s: sum(c.split == s for c in base.clips) for s in corpora.SPLITS},
            }
        },
        "sets": {},
    }
    new = fresh_speakers(base, "test", args.speakers, args.least)
    summary["new_speakers"] = new
    plans: dict[str, tuple[str, list[Phrase], str]] = {
        "base_train": (
            "speech_commands",
            plan_base(base, "train", "base_train", {2: args.train // 2, 3: args.train // 2}, SEED),
            "train",
        ),
        "base_validation": (
            "speech_commands",
            plan_base(
                base,
                "validation",
                "base_validation",
                {2: args.held, 3: args.held, 4: args.held},
                SEED,
            ),
            "validation",
        ),
        "base_test": (
            "speech_commands",
            plan_base(
                base,
                "test",
                "base_test",
                {2: args.held, 3: args.held, 4: args.held},
                SEED,
                exclude=set(new),
            ),
            "test",
        ),
    }
    for speaker in new:
        sides = plan_speaker(
            base, speaker, "test", args.adapt, args.evaluate, SEED, f"speaker/{speaker}"
        )
        for side, phrases in sides.items():
            plans[f"speaker/{speaker}/{side}"] = ("speech_commands", phrases, "test")
    if args.audiomnist:
        inner = corpora.AudioMNIST(args.audiomnist)
        digits = _Bedded(inner, base)  # beds from the base corpus's background recordings
        group = [
            s
            for s in sorted(inner.meta)
            if str(inner.meta[s]["accent"]).strip().lower() == args.accent
        ]
        summary["corpora"]["audiomnist"] = {
            "url": corpora.AUDIOMNIST_URL,
            "licence": "MIT",
            "commit": (Path(args.audiomnist).parent / "commit.txt").read_text().strip()
            if (Path(args.audiomnist).parent / "commit.txt").exists()
            else None,
            "clips": len(digits.clips),
            "speakers": len(inner.meta),
            "accent": args.accent,
            "accent_speakers": len(group),
        }
        summary["dialect_speakers"] = group
        for speaker in group:
            sides = plan_speaker(
                digits, speaker, "test", args.adapt, args.evaluate, SEED, f"dialect/{speaker}"
            )
            for side, phrases in sides.items():
                plans[f"dialect/{speaker}/{side}"] = ("audiomnist", phrases, "test")
    for name, (kind, phrases, split) in plans.items():
        path = out / f"{name.replace('/', '__')}.npz"
        if path.exists() and not args.again:
            continue
        summary["sets"][name] = build_set(kind, roots, phrases, path, split, args.workers)
        print(name, summary["sets"][name]["phrases"], round(time.time() - started), "s", flush=True)
    summary["seconds"] = round(time.time() - started, 1)
    manifest = out / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {"sets": {}}
    summary["sets"] = {**previous.get("sets", {}), **summary["sets"]}
    manifest.write_text(json.dumps(summary, indent=1))
    return summary


def build_synthetic(
    out: str | Path,
    *,
    train: int = 64,
    held: int = 8,
    adapt: int = 12,
    evaluate: int = 6,
    seed: int = 0,
) -> dict[str, Any]:
    """The chamber's sets from synthetic speech (``corpora.Synthetic``), for the tests: one new
    speaker from the test split and one dialect speaker."""
    corpus = corpora.Synthetic(seed=seed)
    out = Path(out)
    accents = {c.speaker: c.accent for c in corpus.clips if c.split == "test"}
    new = sorted(s for s, a in accents.items() if not a)[0]
    dialect = sorted(s for s, a in accents.items() if a)[0]
    sets = {
        "base_train": (
            plan_base(corpus, "train", "base_train", {2: train // 2, 3: train // 2}, SEED),
            "train",
        ),
        "base_validation": (
            plan_base(corpus, "validation", "base_validation", {2: held, 3: held, 4: held}, SEED),
            "validation",
        ),
        "base_test": (
            plan_base(
                corpus,
                "test",
                "base_test",
                {2: held, 3: held, 4: held},
                SEED,
                exclude={new, dialect},
            ),
            "test",
        ),
    }
    for tag, speaker in ((f"speaker/{new}", new), (f"dialect/{dialect}", dialect)):
        for side, planned in plan_speaker(
            corpus, speaker, "test", adapt, evaluate, SEED, tag
        ).items():
            sets[f"{tag}/{side}"] = (planned, "test")
    summary = {
        name: Phrases.build(corpus, planned, out / f"{name.replace('/', '__')}.npz", split)
        for name, (planned, split) in sets.items()
    }
    return {"sets": summary, "new_speaker": f"speaker/{new}", "dialect": f"dialect/{dialect}"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--speech-commands", required=True)
    b.add_argument("--audiomnist", default=None)
    b.add_argument("--common-voice", default=None)
    b.add_argument("--accent", default="german")
    b.add_argument("--out", required=True)
    b.add_argument("--train", type=int, default=24_000)
    b.add_argument("--held", type=int, default=1_000, help="held-out phrases per length")
    b.add_argument("--speakers", type=int, default=24, help="new speakers from the test split")
    b.add_argument("--least", type=int, default=100, help="clips a new speaker holds at least")
    b.add_argument("--adapt", type=int, default=84, help="adaptation phrases per new speaker")
    b.add_argument("--evaluate", type=int, default=16, help="evaluation phrases per new speaker")
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    b.add_argument("--again", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        summary = build(args)
        print(json.dumps({k: v for k, v in summary.items() if k != "sets"}, indent=1))


if __name__ == "__main__":
    main()
