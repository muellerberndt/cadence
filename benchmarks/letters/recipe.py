"""The speech recipe end to end, from the library's pieces (docs/speech.md).

Trains a recogniser from (audio, transcript) pairs, reads it, adapts it to a speaker from
corrections and runs it on a stream:

1. **hear:** every utterance becomes log-mel frames (``speech.frontend.log_mel``) and the
   40 ms steps a brain hears (``Hearing.fit`` on the training utterances, ``Hearing.steps``);
2. **compose:** ``Brain.compose(hearing.shape, 29, learned_fields=64, modules=(256,),
   fields_motor=1.0, learning_surprise=False)``, the actor's rates at zero (the composition the
   letters chamber measured best, issue 304);
3. **train:** utterances side by side as streams, ``batch`` at a time, each batch from rest, one
   lesson a step (``step(..., teacher=labels)``); the labels are each utterance's forced
   alignment through a frozen copy's beliefs, renewed every ``align_every`` batches
   (``speech.align``), the first batches on an even split;
4. **read:** held-out utterances, the letter decoder (``collapse``) and, with a word list, the
   beam over the list (``WordBeam``): characters and words;
5. **adapt:** half of a new speaker's utterances as corrections heard beside three training
   utterances at weight .25 (``speech.correct.Rehearsal``), the speaker's other half and the
   held-out speakers read before and after (on LibriSpeech the dev-clean speaker with the most
   utterances, kept apart from the held-out ones);
6. **stream:** one held-out utterance heard 100 ms at a time (``speech.stream.StreamReader``),
   the text as it arrives and the seconds per step.

Corpus: LibriSpeech (CC BY 4.0, https://www.openslr.org/12), a directory holding its
``LibriSpeech/<split>`` folders (``--splits``, ``train-clean-100`` by default, and ``dev-clean``),
read with ``soundfile``;
with ``--words`` only utterances whose every word is on the list. ``--synthetic`` builds a tiny
corpus of tone words instead, so the recipe runs without a download (the test does).

    python benchmarks/letters/recipe.py --librispeech DIR --words benchmarks/letters/words-2000.txt
    python benchmarks/letters/recipe.py --synthetic --utterances 24 --fields 8 --width 32
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

RATE = 16000


# ------------------------------------------------------------------------------- the corpus


def catalogue(root: Path, split: str, listed: set[str] | None) -> list[tuple[str, Path, str]]:
    """(identity, audio path, transcript) of a split's utterances, on the list where one is
    given, in the corpus's order."""
    found = []
    for listing in sorted((root / "LibriSpeech" / split).rglob("*.trans.txt")):
        for line in listing.read_text().splitlines():
            identity, _, text = line.partition(" ")
            if listed is not None and not all(w in listed for w in text.lower().split()):
                continue
            found.append((identity, listing.parent / f"{identity}.flac", text))
    return found


def librispeech(root: Path, split: str, listed: set[str] | None, limit: int,
                longest: float = 12.0, seed: int | None = None,
                speaker: str | None = None) -> list[dict[str, Any]]:
    """Up to ``limit`` utterances of a split: audio, transcript and speaker; on the list where
    one is given; one speaker's where ``speaker`` is given; a seeded draw where ``seed`` is
    given, else the corpus's order; at most ``longest`` seconds."""
    import soundfile

    found = [f for f in catalogue(root, split, listed)
             if speaker is None or f[0].split("-")[0] == speaker]
    if seed is not None:
        found = [found[int(k)] for k in np.random.default_rng(seed).permutation(len(found))]
    out = []
    for identity, path, text in found:
        audio, rate = soundfile.read(path)
        if rate != RATE or len(audio) > longest * RATE:
            continue
        out.append({"audio": np.asarray(audio, dtype=np.float64), "text": text.lower(),
                    "speaker": identity.split("-")[0], "identity": identity})
        if len(out) == limit:
            break
    return out


TONES = {"a": 500.0, "g": 900.0, "n": 1300.0, "o": 1700.0, "t": 2300.0}
SYNTHETIC_WORDS = ("go", "no", "on", "to", "at", "tag")


def synthetic(count: int, seed: int, speakers: int = 3, first: int = 0) -> list[dict[str, Any]]:
    """Utterances of one to three tone words: each letter 80 ms of its tone, a speaker shifting
    every tone, 120 ms of quiet between words; speakers ``first`` onwards."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        speaker = first + k % speakers
        shift = 1.0 + 0.05 * speaker
        words = [SYNTHETIC_WORDS[int(i)] for i in rng.integers(len(SYNTHETIC_WORDS),
                                                               size=int(rng.integers(1, 4)))]
        parts = [np.zeros(int(0.1 * RATE))]
        for word in words:
            for letter in word:
                t = np.arange(int(0.08 * RATE)) / RATE
                parts.append(0.3 * np.sin(2 * np.pi * TONES[letter] * shift * t))
            parts.append(np.zeros(int(0.12 * RATE)))
        sound = np.concatenate(parts) + 0.003 * rng.standard_normal(sum(map(len, parts)))
        out.append({"audio": sound, "text": " ".join(words), "speaker": f"s{speaker}"})
    return out


# ---------------------------------------------------------------------------------- reading


def edits(reference: list[Any], hypothesis: list[Any]) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hypothesis, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1]


def score(said: list[str], references: list[str]) -> dict[str, float]:
    """Characters and words: one less the edits per reference character (or word)."""
    chars = sum(len(r) for r in references)
    words = sum(len(r.split()) for r in references)
    return {
        "characters": 1.0 - sum(edits(list(r), list(s)) for r, s in zip(references, said,
                                                                         strict=True)) / chars,
        "words": 1.0 - sum(edits(r.split(), s.split()) for r, s in zip(references, said,
                                                                       strict=True)) / words,
    }


# --------------------------------------------------------------------------------- the recipe


def padded(rows: list[np.ndarray]) -> np.ndarray:
    """Utterances side by side, each after its end holding its last step: (steps, rows, inputs)."""
    longest = max(len(x) for x in rows)
    return np.stack([np.concatenate([x, np.repeat(x[-1:], longest - len(x), axis=0)])
                     for x in rows], axis=1)


def frozen(brain: Any, folder: Path) -> Any:
    """A copy of the brain at rest that a reading or an alignment may use up."""
    from cadence import Brain

    brain.reset()
    return Brain.load(brain.save(folder / "copy.npz"))


def transcribe(brain: Any, rows: list[np.ndarray], alphabet: Any, words: Any, batch: int,
               folder: Path) -> tuple[list[str], list[str]]:
    """What a frozen copy says of each utterance: the letter decoder's and the word list's."""
    from cadence.speech.align import log_beliefs
    from cadence.speech.decode import collapse

    copy = frozen(brain, folder)
    letters, listed = [], []
    for k in range(0, len(rows), batch):
        for logp in log_beliefs(copy, rows[k : k + batch]):
            letters.append(alphabet.decode(collapse(np.argmax(logp, axis=1))))
            listed.append(words.decode(logp) if words is not None else letters[-1])
    return letters, listed


def run(args: argparse.Namespace) -> dict[str, Any]:
    from cadence import Brain
    from cadence.speech.align import Alignments, forced_path, log_beliefs
    from cadence.speech.correct import Rehearsal
    from cadence.speech.decode import Alphabet, WordBeam
    from cadence.speech.frontend import Hearing, log_mel
    from cadence.speech.stream import StreamReader

    started = time.monotonic()
    alphabet = Alphabet()
    listed = None
    words = None
    if args.words:
        lines = [line.split() for line in Path(args.words).read_text().splitlines() if line.strip()]
        listed = {w[0] for w in lines}
        words = WordBeam([w[0] for w in lines], counts=[float(w[1]) for w in lines]
                         if all(len(w) > 1 for w in lines) else None)
    if args.synthetic:
        train = synthetic(args.utterances, args.seed)
        held = synthetic(args.held, args.seed + 1)
        adapt = synthetic(args.adapt, args.seed + 2, speakers=1, first=3)  # a speaker unheard
        if words is None:
            words = WordBeam(list(SYNTHETIC_WORDS))
    else:
        root = Path(args.librispeech)
        train = [u for split in args.splits.split(",")
                 for u in librispeech(root, split, listed, args.utterances)][: args.utterances]
        held = librispeech(root, "dev-clean", listed, args.held, seed=args.seed)
        # the corrected speaker: the dev-clean speaker with the most such utterances, apart
        counts: dict[str, int] = {}
        for identity, _, _ in catalogue(root, "dev-clean", listed):
            counts[identity.split("-")[0]] = counts.get(identity.split("-")[0], 0) + 1
        speaker = max(sorted(counts), key=lambda s: counts[s])
        held = [u for u in held if u["speaker"] != speaker]
        adapt = librispeech(root, "dev-clean", listed, args.adapt, speaker=speaker)

    # 1. hear
    for u in train + held + adapt:
        u["frames"] = log_mel(u["audio"])
        u["symbols"] = alphabet.encode(u["text"])
    hearing = Hearing.fit([u["frames"] for u in train])
    for u in train + held + adapt:
        u["rows"] = hearing.steps(u["frames"])
    def fits(u: dict[str, Any]) -> bool:
        repeats = int(np.sum(u["symbols"][1:] == u["symbols"][:-1]))
        return len(u["symbols"]) > 0 and len(u["rows"]) >= len(u["symbols"]) + repeats

    train, held = [u for u in train if fits(u)], [u for u in held if fits(u)]
    adapt = [u for u in adapt if fits(u)]
    audio_seconds = sum(len(u["audio"]) for u in train) / RATE
    print(json.dumps({"train": len(train), "held": len(held), "adapt": len(adapt),
                      "hours": audio_seconds / 3600}), flush=True)

    # 2. compose
    brain = Brain.compose(hearing.shape, alphabet.size, seed=args.seed, backend="cpu",
                          learned_fields=args.fields, modules=(args.width,), fields_motor=1.0,
                          learning_surprise=False, actor_eta=0.0, actor_eta_bias=0.0,
                          actor_eta_critic=0.0)
    report: dict[str, Any] = {"parameters": int(brain.parameters()), "passes": []}

    # 3. train
    folder = Path(tempfile.mkdtemp(prefix="recipe-"))
    alignments = Alignments()
    rng = np.random.default_rng(args.seed)
    lengths = np.array([len(u["rows"]) for u in train])
    trained = 0.0
    for p in range(args.passes):
        order = rng.permutation(len(train))
        order = order[np.argsort(lengths[order], kind="stable")]
        groups = [order[k : k + args.batch] for k in range(0, len(order), args.batch)]
        groups = [groups[int(k)] for k in rng.permutation(len(groups))]
        for n, group in enumerate(groups):
            keys = [(p, int(i)) for i in group]
            if n % args.align_every == 0:  # the next batches' paths, from a frozen copy
                upcoming = [(p, int(i)) for g in groups[n : n + args.align_every] for i in g]
                if p == 0 and n == 0:
                    for key in upcoming:
                        u = train[key[1]]
                        alignments.even(key, len(u["rows"]), u["symbols"])
                else:
                    copy = frozen(brain, folder)
                    rows = [train[i]["rows"] for _, i in upcoming]
                    for key, logp in zip(upcoming, log_beliefs(copy, rows), strict=True):
                        alignments.realign(key, logp, train[key[1]]["symbols"])
            x = padded([train[int(i)]["rows"] for i in group])
            labels = alignments.lesson(keys, len(x))
            ticked = time.monotonic()
            brain.reset()
            for t in range(len(x)):
                brain.step(x[t], teacher=labels[t])
            trained += time.monotonic() - ticked
        said, by_list = transcribe(brain, [u["rows"] for u in held], alphabet, words,
                                   args.batch, folder)
        reading = {"pass": p + 1, **score(said, [u["text"] for u in held]),
                   "words_on_the_list": score(by_list, [u["text"] for u in held])["words"],
                   "training_seconds_per_audio_hour": trained / (p + 1) / audio_seconds * 3600,
                   "sample": said[:2]}
        report["passes"].append(reading)
        print(json.dumps(reading), flush=True)

    # 5. adapt: a new speaker's utterances as corrections beside training utterances
    corrections, readings = adapt[: len(adapt) // 2], adapt[len(adapt) // 2 :]
    if corrections and readings:
        bank = Rehearsal(rows=3, weight=0.25, seed=args.seed)
        copy = frozen(brain, folder)
        sample = rng.choice(len(train), min(len(train), 64), replace=False)
        for i, logp in zip(sample, log_beliefs(copy, [train[int(i)]["rows"] for i in sample]),
                           strict=True):
            bank.remember(train[int(i)]["rows"], forced_path(logp, train[int(i)]["symbols"]))

        def reading_of(group: list[dict[str, Any]]) -> float:
            said_ = transcribe(brain, [u["rows"] for u in group], alphabet, None, args.batch,
                               folder)[0]
            return score(said_, [u["text"] for u in group])["characters"]

        before = {"speaker": reading_of(readings), "others": reading_of(held)}
        for u in corrections:
            copy = frozen(brain, folder)
            logp = log_beliefs(copy, [u["rows"]])[0]
            bank.correct(brain, u["rows"], forced_path(logp, u["symbols"]))
        after = {"speaker": reading_of(readings), "others": reading_of(held)}
        report["adapt"] = {"speaker": adapt[0]["speaker"], "corrections": len(corrections),
                           "characters_before": before, "characters_after": after}
        print(json.dumps(report["adapt"]), flush=True)

    # 6. stream one held-out utterance, 100 ms at a time
    brain.reset()
    copy = frozen(brain, folder)
    if copy.fields is not None:
        copy.fields.learning = False
    reader = StreamReader(copy, hearing, words=words)
    sound = held[0]["audio"]
    arrived = [reader.push(sound[k : k + RATE // 10]) for k in range(0, len(sound), RATE // 10)]
    report["stream"] = {"heard": held[0]["text"], "said": "".join(arrived),
                        "finish": reader.finish(), **reader.latency()}
    print(json.dumps(report["stream"]), flush=True)
    report["seconds"] = time.monotonic() - started
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--librispeech", help="a directory holding LibriSpeech/<split> folders")
    source.add_argument("--synthetic", action="store_true", help="tone words, no download")
    parser.add_argument("--words", help="a word list, one word (and its count) a line")
    parser.add_argument("--utterances", type=int, default=400, help="training utterances")
    parser.add_argument("--splits", default="train-clean-100",
                        help="LibriSpeech training splits, comma-separated, in order")
    parser.add_argument("--held", type=int, default=60, help="held-out utterances")
    parser.add_argument("--adapt", type=int, default=24,
                        help="utterances of a new speaker: half corrections, half read")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32, help="utterances side by side")
    parser.add_argument("--align-every", type=int, default=8, help="batches between alignments")
    parser.add_argument("--fields", type=int, default=64, help="learned_fields")
    parser.add_argument("--width", type=int, default=256, help="the association's neurons")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="a JSON file for the report")
    args = parser.parse_args(argv)
    report = run(args)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
