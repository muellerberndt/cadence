"""Read speech in the letters chamber: LibriSpeech utterances heard one 40 ms step per moment.

The adapter reads LibriSpeech (CC BY 4.0, https://www.openslr.org/12) from a local directory
that holds its ``LibriSpeech/<split>`` folders; nothing is downloaded here, and the tests skip
where the data is absent. At every 40 ms step the brain hears the log-mel energies of that step
(4 frames of 10 ms, ``mels`` bands each, normalized per band on the training utterances) and
says one of 29 symbols: blank, ``a``-``z``, the apostrophe and space.

The teacher knows each utterance's text and nothing of its timing. Its lessons: each utterance's
steps are first split evenly among its characters (spaces included); every ``align_every``
utterances a frozen copy of the brain answers the next utterances greedily, and each one's steps
go to its characters by CTC's forced alignment through the copy's log beliefs (``ctc_path``:
blank may fill any step between characters and parts a letter from the same letter after it).
Each step is taught the symbol on its path. The path is the teacher's, computed outside the
brain, and the brain learns by its own rule.

Readings are the chamber's CTC reading on held-out utterances: characters, inserted, deleted and
substituted characters per character, words, the share of word errors that are insertions, and
the words lost with every letter right (a space said wrong) against the words lost on a letter.

    python benchmarks/letters/speech.py prepare --root DIR --out CACHE
    python benchmarks/letters/speech.py life --cache CACHE --protocol P --seed S --out DIR
    python benchmarks/letters/speech.py control --model lstm --mode online --cache CACHE ...
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("letters", _here / "letters.py")
assert _spec is not None and _spec.loader is not None
lt = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("letters", lt)
_spec.loader.exec_module(lt)
sys.path.insert(0, str(_here))
import vocabulary as vc  # noqa: E402

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"
SPACE = len(ALPHABET) + 1  # 28
SYMBOLS = len(ALPHABET) + 2  # blank, the letters and the apostrophe, space
RATE, WINDOW, HOP, STACK = 16000, 400, 160, 4
STEP_SECONDS = STACK * HOP / RATE  # 0.04
TRAINING = ("train-clean-100", "train-clean-360")  # the word list's and the corpus's splits


def word_matcher(protocol: dict[str, Any]) -> Any:
    """The list a protocol names (``word_list``, one word and its count a line, beside this
    file), as the matcher of the list-matched reading; None without one."""
    if "word_list" not in protocol:
        return None
    lines = (_here / protocol["word_list"]).read_text().splitlines()
    return vc.Matcher([line.split()[0] for line in lines if line.strip()])


# ------------------------------------------------------------------------------ the sounds


def mel_bank(mels: int, *, rate: int = RATE, fft: int = 512) -> np.ndarray:
    """Triangular filters evenly spaced on the mel scale, (mels, fft // 2 + 1)."""
    def mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    edges = 700.0 * (10 ** (np.linspace(mel(np.array(0.0)), mel(np.array(rate / 2)),
                                        mels + 2) / 2595.0) - 1.0)
    bins = np.fft.rfftfreq(fft, 1.0 / rate)
    bank = np.zeros((mels, len(bins)))
    for m in range(mels):
        low, mid, high = edges[m], edges[m + 1], edges[m + 2]
        rise = (bins - low) / max(mid - low, 1e-9)
        fall = (high - bins) / max(high - mid, 1e-9)
        bank[m] = np.clip(np.minimum(rise, fall), 0.0, None)
    return bank


def steps_of(audio: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Log-mel energies of 25 ms frames every 10 ms, four frames to a 40 ms step:
    (steps, 4 * mels)."""
    energies = energies_of(audio, bank)
    steps = len(energies) // STACK
    return energies[: steps * STACK].reshape(steps, STACK * bank.shape[0]).astype(np.float32)


def energies_of(audio: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Log-mel energies of 25 ms frames every 10 ms: (frames, mels)."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < WINDOW:
        audio = np.pad(audio, (0, WINDOW - len(audio)))
    count = 1 + (len(audio) - WINDOW) // HOP
    index = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
    frames = audio[index] * np.hanning(WINDOW)[None, :]
    power = np.abs(np.fft.rfft(frames, 512)) ** 2
    return np.log(power @ bank.T + 1e-6)


def encode(text: str) -> np.ndarray:
    """A transcript as symbols: letters and the apostrophe 1 to 27, single spaces 28."""
    words = "".join(c if c in ALPHABET else " " for c in text.lower()).split()
    out: list[int] = []
    for k, word in enumerate(words):
        if k:
            out.append(SPACE)
        out.extend(ALPHABET.index(c) + 1 for c in word)
    return np.asarray(out, dtype=np.int64)


def decode(symbols: list[int]) -> str:
    chars = [" " if s == SPACE else ALPHABET[s - 1] for s in symbols]
    return " ".join("".join(chars).split())


def utterances(root: Path, split: str) -> list[tuple[str, Path, str]]:
    """(identity, audio path, transcript) of every utterance of a LibriSpeech split, sorted."""
    out = []
    for listing in sorted((root / "LibriSpeech" / split).rglob("*.trans.txt")):
        for line in listing.read_text().splitlines():
            identity, _, text = line.partition(" ")
            out.append((identity, listing.parent / f"{identity}.flac", text))
    return out


def prepare(root: Path, out: Path, *, mels: int = 40, train: int = 4000, held: int = 300,
            longest: float = 12.0, seed: int = 0) -> dict[str, Any]:
    """Cache the steps of ``train`` utterances of train-clean-100 and ``held`` of dev-clean,
    drawn at random among those of at most ``longest`` seconds, normalized per band on the
    training steps."""
    import soundfile  # the adapter's only audio dependency

    bank = mel_bank(mels)
    rng = np.random.default_rng(seed)
    sets: dict[str, Any] = {}
    for name, split, n in (("train", "train-clean-100", train), ("held", "dev-clean", held)):
        chosen: list[tuple[str, np.ndarray, np.ndarray]] = []
        for k in rng.permutation(len(listing := utterances(root, split))):
            identity, path, text = listing[int(k)]
            audio, rate = soundfile.read(path)
            if rate != RATE or len(audio) > longest * RATE:
                continue
            steps, symbols = steps_of(audio, bank), encode(text)
            repeats = int(np.sum(symbols[1:] == symbols[:-1]))
            if len(steps) < len(symbols) + repeats:  # too fast for one step a character
                continue
            chosen.append((identity, steps, symbols))
            if len(chosen) == n:
                break
        sets[name] = chosen
    stacked = np.concatenate([x for _, x, _ in sets["train"]])
    mean, scale = stacked.mean(0), stacked.std(0) + 1e-6
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"mels": mels, "longest": longest, "seed": seed}
    for name, chosen in sets.items():
        np.savez_compressed(
            out / f"{name}.npz",
            steps=np.concatenate([(x - mean) / scale for _, x, _ in chosen]).astype(np.float32),
            lengths=np.array([len(x) for _, x, _ in chosen]),
            text=np.concatenate([t for _, _, t in chosen]),
            text_lengths=np.array([len(t) for _, _, t in chosen]),
            identities=np.array([i for i, _, _ in chosen]),
        )
        meta[name] = {"utterances": len(chosen), "steps": int(sum(len(x) for _, x, _ in chosen))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load(cache: Path, name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The cached (steps, symbols) of each utterance; a window cache (``prepare_listed``) gives
    each step the image its window makes, flattened band by band."""
    meta = json.loads((cache / "meta.json").read_text())
    if "window" in meta:
        return load_windows(cache, name, meta)
    with np.load(cache / f"{name}.npz") as data:
        x, text = data["steps"], data["text"]
        a = np.concatenate([[0], np.cumsum(data["lengths"])])
        b = np.concatenate([[0], np.cumsum(data["text_lengths"])])
    return [(x[a[i] : a[i + 1]], text[b[i] : b[i + 1]]) for i in range(len(a) - 1)]


def inputs_of(cache: Path) -> int | tuple[int, int]:
    """What a brain hears from a cache: a vector width, or a window's (bands, frames) image."""
    meta = json.loads((cache / "meta.json").read_text())
    if "window" in meta:
        return (int(meta["mels"]), int(meta["window"]))
    with np.load(cache / "held.npz") as data:
        return int(data["steps"].shape[1])


SCALE, CLIP_SIGMA = 0.3, 3.0  # the standardized window, as the spoken chamber hears it


def windows_of(frames: np.ndarray, window: int) -> np.ndarray:
    """The ``window`` latest 10 ms frames at each 40 ms step, the utterance's first frame
    repeated before it began: (steps, window, mels), each window less its own mean."""
    steps = len(frames) // STACK
    padded = np.concatenate([np.repeat(frames[:1], window, axis=0), frames])
    ends = STACK * (np.arange(steps) + 1) + window
    index = ends[:, None] - window + np.arange(window)[None, :]
    out = padded[index]
    return out - out.mean(axis=(1, 2), keepdims=True)


def load_windows(cache: Path, name: str, meta: dict[str, Any]) -> list[tuple[np.ndarray,
                                                                         np.ndarray]]:
    mean, std = np.asarray(meta["mean"]), np.asarray(meta["std"])
    window = int(meta["window"])
    with np.load(cache / f"{name}.npz") as data:
        frames, text = data["frames"], data["text"]
        a = np.concatenate([[0], np.cumsum(data["lengths"])])
        b = np.concatenate([[0], np.cumsum(data["text_lengths"])])
    out = []
    for i in range(len(a) - 1):
        w = windows_of(frames[a[i] : a[i + 1]].astype(np.float64), window)
        z = np.clip((w - mean) / std, -CLIP_SIGMA, CLIP_SIGMA) * SCALE
        image = np.ascontiguousarray(z.transpose(0, 2, 1)).reshape(len(z), -1)
        out.append((image.astype(np.float32), text[b[i] : b[i + 1]]))
    return out


def prepare_listed(root: Path, out: Path, listed: list[str], *, mels: int = 40, window: int = 24,
                   held: int = 300, longest: float = 12.0, seed: int = 0,
                   training: tuple[str, ...] = TRAINING) -> dict[str, Any]:
    """Cache the utterances of the ``training`` splits whose every word is on ``listed`` (all of
    them, at most ``longest`` seconds, in the corpus's order) and ``held`` such utterances of
    dev-clean drawn at random, as 10 ms log-mel frames; a step hears its ``window`` latest
    frames, standardized per band on the training windows."""
    import soundfile

    on = set(listed)
    bank = mel_bank(mels)
    rng = np.random.default_rng(seed)
    sets: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for name, splits in (("train", training), ("held", ("dev-clean",))):
        listing = [u for split in splits for u in utterances(root, split)
                   if vc.on_list(u[2], on)]
        if name == "held":
            listing = [listing[int(k)] for k in rng.permutation(len(listing))]
        chosen = []
        for identity, path, text in listing:
            audio, rate = soundfile.read(path)
            if rate != RATE or len(audio) > longest * RATE:
                continue
            frames, symbols = energies_of(audio, bank), encode(text)
            repeats = int(np.sum(symbols[1:] == symbols[:-1]))
            if len(frames) // STACK < len(symbols) + repeats:
                continue
            chosen.append((identity, frames.astype(np.float32), symbols))
            if name == "held" and len(chosen) == held:
                break
        sets[name] = chosen
    total, square, count = np.zeros(mels), np.zeros(mels), 0
    for _, frames, _ in sets["train"]:
        w = windows_of(frames.astype(np.float64), window)
        total += w.sum(axis=(0, 1))
        square += (w**2).sum(axis=(0, 1))
        count += w.shape[0] * w.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 1e-6))
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"mels": mels, "window": window, "longest": longest, "seed": seed,
                            "listed": len(listed), "mean": mean.tolist(),
                            "std": np.maximum(std, 1e-3).tolist()}
    for name, chosen in sets.items():
        np.savez_compressed(
            out / f"{name}.npz",
            frames=np.concatenate([f for _, f, _ in chosen]),
            lengths=np.array([len(f) for _, f, _ in chosen]),
            text=np.concatenate([t for _, _, t in chosen]),
            text_lengths=np.array([len(t) for _, _, t in chosen]),
            identities=np.array([i for i, _, _ in chosen]),
        )
        meta[name] = {"utterances": len(chosen),
                      "steps": int(sum(len(f) // STACK for _, f, _ in chosen)),
                      "seconds": float(sum(len(f) for _, f, _ in chosen) * HOP / RATE)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# -------------------------------------------------------------------------------- readings


def even_path(steps: int, text: np.ndarray) -> np.ndarray:
    """Each step's symbol on an even split of the steps among the characters."""
    n = len(text)
    return text[np.minimum((np.arange(steps) * n) // max(steps, 1), n - 1)]


def read(actions: list[np.ndarray], held: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """The CTC reading of every held-out utterance, pooled."""
    chars = {"substitutions": 0, "deletions": 0, "insertions": 0, "errors": 0}
    words = {"substitutions": 0, "deletions": 0, "insertions": 0, "errors": 0}
    total_chars = total_words = exact = spelled_lost = 0
    for a, (_, text) in zip(actions, held, strict=True):
        reference, hypothesis = decode(text.tolist()), decode(lt.collapse(a))
        c = lt.edits(list(reference), list(hypothesis))
        w = lt.edits(reference.split(), hypothesis.split())
        for k in chars:
            chars[k] += c[k]
            words[k] += w[k]
        total_chars += len(reference)
        total_words += len(reference.split())
        exact += int(reference == hypothesis)
        # words lost although every letter of the utterance's letters is in place: a space
        letters_right = lt.edits(list(reference.replace(" ", "")),
                                 list(hypothesis.replace(" ", "")))
        spelled_lost += w["errors"] if letters_right["errors"] == 0 else 0
    word_errors = max(words["errors"], 1)
    return {
        "characters": 1.0 - chars["errors"] / total_chars,
        **{f"{k}_per_character": chars[k] / total_chars for k in
           ("substitutions", "deletions", "insertions")},
        "words": 1.0 - words["errors"] / total_words,
        "insertion_share": words["insertions"] / word_errors,
        "deletion_share": words["deletions"] / word_errors,
        "exact": exact / len(held),
        "words_lost_on_spacing_alone": spelled_lost / word_errors,
        "utterances": len(held),
    }


def locate(actions: list[np.ndarray], logps: list[np.ndarray],
           held: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Each held-out character's steps by CTC's forced alignment of the reference through the
    learner's own beliefs, and on them: the letter from the sound (the symbol it prefers over
    the others but blank is the character, on one of its steps), its emission given the letter
    (said on one of its steps), and the word given its letters (a word whose every character
    was said, said whole by the CTC reading)."""
    identified = said = known_words = whole_given_known = 0
    characters = 0
    for a, logp, (_, text) in zip(actions, logps, held, strict=True):
        if len(logp) < len(text):
            continue
        states = lt.ctc_path(logp, text, states_out=True)
        preferred = np.argmax(logp[:, 1:], axis=1) + 1
        ok = np.zeros(len(text), dtype=bool)
        for k, symbol in enumerate(text.tolist()):
            steps = states == 2 * k + 1
            found = bool(np.any(preferred[steps] == symbol))
            spoken = bool(np.any(a[steps] == symbol))
            identified += found
            said += found and spoken
            ok[k] = spoken
            characters += 1
        reference, hypothesis = decode(text.tolist()), decode(lt.collapse(a))
        matched = lt.edits(reference.split(), hypothesis.split())["matched"]
        start = 0
        for w, word in enumerate(reference.split()):
            if ok[start : start + len(word)].all():
                known_words += 1
                whole_given_known += int(matched[w])
            start += len(word) + 1
    return {
        "letter_from_sound": identified / max(characters, 1),
        "emission_given_letter": said / max(identified, 1),
        "word_given_letters": whole_given_known / max(known_words, 1),
        "words_with_every_letter_said": known_words,
    }


# ----------------------------------------------------------------------------------- brain


def beliefs(brain: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A brain's greedy actions and log beliefs at each step of one utterance, from rest."""
    brain.reset()
    motor = np.asarray(brain.connectome.populations["motor"], dtype=np.int64)
    temperature = float(brain.describe()["genes"]["temperature"])
    actions = np.zeros(len(x), dtype=np.int64)
    out = np.zeros((len(x), len(motor)))
    for t, row in enumerate(x):
        actions[t] = int(brain.act(row[None], greedy=True)[0])
        drive = np.asarray(brain.basal_ganglia.state.activation).reshape(-1)[motor] / temperature
        out[t] = drive - drive.max() - np.log(np.sum(np.exp(drive - drive.max())))
    return actions, out


def life(cache: Path, protocol: dict[str, Any], seed: int, out: Path,
         limit: int | None = None) -> dict[str, Any]:
    """One brain hears the training utterances in a seeded order, one step a moment, each from
    rest; its lessons follow the teacher's alignment; a saved copy reads the held-out
    utterances every ``probe_every`` utterances. The brain saved at each reading and the record
    so far let a stopped life continue from its last reading, the teacher re-aligning the next
    utterances on it as it would have."""
    import cadence as cd

    train, held = load(cache, "train"), load(cache, "held")
    order = np.random.default_rng([seed, 7]).permutation(len(train))[:limit]
    every, probe_every = int(protocol["align_every"]), int(protocol["probe_every"])
    brain = cd.Brain.compose(inputs_of(cache), SYMBOLS, backend="cpu", seed=seed,
                             **protocol["brain"]["genes"])
    matcher = word_matcher(protocol)
    clock = {"training": 0.0, "reading": 0.0}
    record: dict[str, Any] = {"seed": seed, "parameters": brain.parameters(), "probes": [],
                              "alignments": [], "describe": brain.describe()}
    paths = {int(i): even_path(len(train[i][0]), train[i][1]) for i in order[:every]}
    started, steps, begin = time.monotonic(), 0, 0
    partial = out / f"speech-brain-{seed}.partial.json.gz"
    if partial.exists():  # a life a lost machine stopped continues from its last reading
        record = lt.read_record(partial)
        clock = dict(record["probes"][-1].get("clock", clock))
        begin = int(record["probes"][-1]["utterances"])
        steps = int(record["probes"][-1]["steps"])
        started -= float(record["probes"][-1]["seconds"])
        brain = cd.Brain.load(out / f"speech-{seed}-u{begin}.npz")
        record.setdefault("resumed", []).append(begin)
    for n, i in enumerate(order):
        if n < begin:
            continue
        i = int(i)
        if n and n % every == 0:  # the teacher re-aligns the next utterances on a frozen copy
            copy = cd.Brain.load(brain.save(out / f"speech-{seed}-align.npz"))
            changed = []
            for j in order[n : n + every]:
                j = int(j)
                _, logp = beliefs(copy, train[j][0])
                new = lt.ctc_path(logp, train[j][1])
                old = even_path(len(new), train[j][1])
                changed.append(float(np.mean(new != old)))
                paths[j] = new
            record["alignments"].append({"utterances": n,
                                         "moved_from_even": float(np.mean(changed))})
        x, y = train[i]
        ticked = time.monotonic()
        brain.reset()
        for t in range(len(x)):
            brain.step(x[t][None], teacher=[int(paths[i][t])])
        clock["training"] += time.monotonic() - ticked
        steps += len(x)
        paths.pop(i, None)
        if (n + 1) % probe_every == 0 or n + 1 == len(order):
            path = brain.save(out / f"speech-{seed}-u{n + 1}.npz")
            copy = cd.Brain.load(path)
            ticked = time.monotonic()
            answered = [beliefs(copy, x_) for x_, _ in held]
            read_seconds = time.monotonic() - ticked
            actions = [a for a, _ in answered]
            result = {**read(actions, held), **locate(actions, [b for _, b in answered], held)}
            if matcher is not None:
                result.update(vc.held_words(actions, held, matcher, decode))
            result["training_seconds_per_audio_second"] = clock["training"] / (steps * STEP_SECONDS)
            result["reading_seconds_per_audio_second"] = read_seconds / (
                sum(len(x_) for x_, _ in held) * STEP_SECONDS)
            record["probes"].append({"utterances": n + 1, "steps": steps, "clock": dict(clock),
                                     "seconds": time.monotonic() - started, "score": result,
                                     "sample": [decode(lt.collapse(a)) for a in actions[:3]]})
            lt.write_record(partial, record)
            print(json.dumps({"seed": seed, "utterances": n + 1,
                              "characters": round(result["characters"], 3),
                              "words": round(result["words"], 3),
                              "elapsed": round(time.monotonic() - started)}), flush=True)
    return record


def padded(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Utterances of a batch side by side, each after its end holding its last step: (steps,
    rows, inputs) and the mask of real steps (steps, rows)."""
    longest = max(len(x) for x in rows)
    out = np.stack([np.concatenate([x, np.repeat(x[-1:], longest - len(x), axis=0)])
                    for x in rows], axis=1)
    mask = np.arange(longest)[:, None] < np.array([len(x) for x in rows])[None, :]
    return out, mask


def batch_beliefs(brain: Any, rows: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """``beliefs`` of several utterances heard side by side as streams of one brain, from rest."""
    x, mask = padded(rows)
    brain.reset()
    motor = np.asarray(brain.connectome.populations["motor"], dtype=np.int64)
    temperature = float(brain.describe()["genes"]["temperature"])
    actions = np.zeros(mask.shape, dtype=np.int64)
    logp = np.zeros((*mask.shape, len(motor)))
    for t in range(len(x)):
        actions[t] = np.asarray(brain.act(x[t], greedy=True)).reshape(-1)
        drive = np.asarray(brain.basal_ganglia.state.activation)[:, motor] / temperature
        top = drive.max(axis=1, keepdims=True)
        logp[t] = drive - top - np.log(np.sum(np.exp(drive - top), axis=1, keepdims=True))
    return [(actions[mask[:, r], r], logp[mask[:, r], r]) for r in range(len(rows))]


def batches(order: np.ndarray, lengths: np.ndarray, size: int) -> list[np.ndarray]:
    """The seeded order cut into batches of ``size`` utterances of similar length: each block of
    sixteen batches sorted by length, then cut."""
    out = []
    block = 16 * size
    for start in range(0, len(order), block):
        part = order[start : start + block]
        part = part[np.argsort(lengths[part], kind="stable")]
        out.extend(part[k : k + size] for k in range(0, len(part), size))
    return out


def confusions(logps: list[np.ndarray],
               held: list[tuple[np.ndarray, np.ndarray]]) -> list[list[int]]:
    """For each held-out character on the reference's forced path through the learner's own
    beliefs, the symbol other than blank it prefers most often over the character's steps:
    counts[reference symbol - 1][preferred symbol - 1], 28 by 28."""
    counts = np.zeros((SYMBOLS - 1, SYMBOLS - 1), dtype=np.int64)
    for logp, (_, text) in zip(logps, held, strict=True):
        if len(logp) < len(text):
            continue
        states = lt.ctc_path(logp, text, states_out=True)
        preferred = np.argmax(logp[:, 1:], axis=1)
        for k, symbol in enumerate(text.tolist()):
            votes = np.bincount(preferred[states == 2 * k + 1], minlength=SYMBOLS - 1)
            counts[symbol - 1, int(np.argmax(votes))] += 1
    return [[int(v) for v in row] for row in counts]


def write_atomic(path: Path, record: dict[str, Any]) -> None:
    """A record written whole or not at all: beside its place first, then moved there."""
    beside = path.with_name(path.name + ".tmp")
    lt.write_record(beside, record)
    os.replace(beside, path)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_slot(brain: Any, out: Path, seed: int, index: int, state: dict[str, Any]) -> Path:
    """The brain at rest between two batches in one of two alternating files, with the state of
    its life beside it (batches heard, steps, clock, the file's sha256): what a stopped life
    continues from."""
    path = out / f"speech-{seed}-align-{'ab'[index % 2]}.npz"
    beside = path.with_name(path.name.replace(".npz", ".tmp.npz"))
    brain.save(beside)
    digest = sha256_of(beside)
    os.replace(beside, path)
    side = path.with_suffix(".json")
    side.with_name(side.name + ".tmp").write_text(json.dumps({**state, "sha256": digest}))
    os.replace(side.with_name(side.name + ".tmp"), side)
    return path


def latest_slot(out: Path, seed: int) -> tuple[Path, dict[str, Any]] | None:
    """The later of the two saved brains whose file matches its recorded sha256."""
    best: tuple[Path, dict[str, Any]] | None = None
    for slot in "ab":
        path = out / f"speech-{seed}-align-{slot}.npz"
        side = path.with_suffix(".json")
        if not (path.exists() and side.exists()):
            continue
        try:
            state = json.loads(side.read_text())
        except ValueError:
            continue
        if sha256_of(path) != state.get("sha256"):
            continue
        if best is None or state["batches"] > best[1]["batches"]:
            best = (path, state)
    return best


def life_batched(cache: Path, protocol: dict[str, Any], seed: int, out: Path,
                 stop_after: int | None = None) -> dict[str, Any] | None:
    """``life`` with ``batch`` utterances heard side by side as streams of one brain: each batch
    from rest, an utterance after its end holding its last step and taught blank there; the
    teacher re-aligns every ``align_every`` batches on a frozen copy; a reading every
    ``probe_every`` batches; ``passes`` passes over the training utterances, each in its own
    seeded order.

    The brain is saved at rest at every re-alignment (``save_slot``), and a stopped life
    continues from the latest of those saves: it re-aligns there as the life did and hears the
    same batches. ``stop_after`` ends the life after that many batches, as a lost machine would
    (None is returned). ``place`` in the protocol names where the brain settles (the keywords
    of ``Brain.compose`` and ``Brain.load``: ``backend``, ``device``, ``precision``); without
    it the CPU.

    ``teacher`` in the protocol names the lesson: ``"path"`` (as without it) teaches each step
    the symbol on the forced alignment, the first batches on an even split; ``"soft"`` teaches
    each step the soft path through the brain's beliefs (``cadence.speech.align``'s
    ``soft_path``: the share of the transcript's paths that say each symbol at the step), from
    the first batch on, through the newborn's beliefs there, each moment a ``soft_step``: the
    lesson of the distribution and a greedy act. The soft teacher reads the beliefs from the
    brain itself, at rest between two batches with its learned fields held (its parameters
    stay as they were), every ``align_every`` batches, and the brain is saved at rest every
    ``save_every`` batches (a multiple of ``align_every``; ``align_every`` without it).
    ``sharpness`` in the protocol is the soft path's (1 without it): one number, or one per
    pass, the last held for the passes after.
    ``rate`` in the protocol multiplies the lesson's two rates (``eta`` and ``eta_bias`` as
    composed; 1 without it), one number or one per pass in the same form: the brain is retuned
    where the multiplier changes, and keeps its optimizer's history."""
    import cadence as cd
    from cadence.speech.align import soft_path, soft_step

    train, held = load(cache, "train"), load(cache, "held")
    size = int(protocol["batch"])
    lengths = np.array([len(x) for x, _ in train])
    groups = [g for p in range(int(protocol.get("passes", 1)))  # each pass in its own order
              for g in batches(np.random.default_rng([seed, 7, p]).permutation(len(train))
                               if p else np.random.default_rng([seed, 7]).permutation(len(train)),
                               lengths, size)]
    every, probe_every = int(protocol["align_every"]), int(protocol["probe_every"])
    place = dict(protocol.get("place", {"backend": "cpu"}))  # where the brain settles
    brain = cd.Brain.compose(inputs_of(cache), SYMBOLS, seed=seed, **place,
                             **protocol["brain"]["genes"])
    matcher = word_matcher(protocol)
    clock = {"training": 0.0, "reading": 0.0, "aligning": 0.0}
    record: dict[str, Any] = {"seed": seed, "parameters": brain.parameters(), "probes": [],
                              "alignments": [], "describe": brain.describe(), "batch": size}
    soft = protocol.get("teacher", "path") == "soft"
    if protocol.get("teacher", "path") not in ("path", "soft"):
        raise ValueError("a protocol's teacher is 'path' or 'soft'")
    save_every = int(protocol.get("save_every", every))
    if save_every < 1 or save_every % every:
        raise ValueError("a protocol's save_every is a multiple of its align_every")
    per_pass = len(groups) // max(int(protocol.get("passes", 1)), 1)

    def scheduled(name: str, batch: int) -> float:
        """A protocol's number for a batch: the number itself, or its pass's of a list."""
        value = protocol.get(name, 1.0)
        if isinstance(value, (int, float)):
            return float(value)
        return float(value[min(batch // max(per_pass, 1), len(value) - 1)])

    def sharpness(batch: int) -> float:
        return scheduled("sharpness", batch)

    composed = brain.learner.config  # the lesson's rates as composed: what ``rate`` multiplies
    composed_bias = (composed.eta_bias if composed.eta_bias is not None else composed.eta / 10)
    rated = [1.0]

    def retuned(batch: int) -> None:
        """The lesson's rates at the batch's multiplier, the optimizer's history kept."""
        factor = scheduled("rate", batch)
        if factor != rated[0]:
            brain.retune(learning=dataclasses.replace(
                brain.learner.config, eta=composed.eta * factor,
                eta_bias=composed_bias * factor))
            rated[0] = factor

    # each batch's paths, by the batch: an utterance heard in two passes takes a path in each
    paths: dict[int, list[np.ndarray]] = {} if soft else {
        m: [even_path(len(train[int(i)][0]), train[int(i)][1]) for i in groups[m]]
        for m in range(min(every, len(groups)))
    }

    def realign(copy: Any, first: int, last: int) -> list[float]:
        moved = []
        for m in range(first, min(last, len(groups))):
            g = groups[m]
            paths[m] = []
            for i, (_, logp) in zip(g, batch_beliefs(copy, [train[int(i)][0] for i in g]),
                                    strict=True):
                text = train[int(i)][1]
                new = (soft_path(logp, text, sharpness=sharpness(m)) if soft
                       else lt.ctc_path(logp, text))
                best = np.argmax(new, axis=1) if soft else new
                moved.append(float(np.mean(best != even_path(len(best), text))))
                paths[m].append(new)
        return moved

    steps, begin = 0, 0
    partial = out / f"speech-brain-{seed}.partial.json.gz"
    saved = latest_slot(out, seed)
    if saved is not None:  # a stopped life: from its latest saved re-alignment
        begin, steps = int(saved[1]["batches"]), int(saved[1]["steps"])
        clock = {**clock, **saved[1]["clock"]}
        brain = cd.Brain.load(saved[0], **place)
        if partial.exists():
            earlier = lt.read_record(partial)
            record["probes"] = [q for q in earlier["probes"] if q["batches"] <= begin]
            record["alignments"] = [a for a in earlier["alignments"] if a["batches"] < begin]
            record["resumed"] = list(earlier.get("resumed", []))
        record.setdefault("resumed", []).append(begin)
        loaded = brain.learner.config  # the rates the save holds
        rated[0] = loaded.eta / composed.eta
    for n, group in enumerate(groups):
        if n < begin:
            continue
        if stop_after is not None and n >= stop_after:
            return None
        retuned(n)
        if soft and n and n % save_every == 0:
            brain.reset()  # a batch begins from rest: the save holds no stream
            save_slot(brain, out, seed, n // save_every, {
                "batches": n, "steps": steps, "clock": dict(clock)})
        if soft and n % every == 0:  # the brain's own beliefs, at rest, its fields held
            ticked = time.monotonic()
            brain.reset()
            moving = brain.fields is not None and brain.fields.learning
            if moving:
                brain.fields.learning = False
            moved = realign(brain, n, n + every)
            if moving:
                brain.fields.learning = True
            clock["aligning"] += time.monotonic() - ticked
            record["alignments"].append({"batches": n, "moved_from_even": float(np.mean(moved))})
        if not soft and n and n % every == 0:
            ticked = time.monotonic()
            brain.reset()  # a batch begins from rest: the save holds no stream
            copy = cd.Brain.load(save_slot(brain, out, seed, n // every, {
                "batches": n, "steps": steps, "clock": dict(clock)}), **place)
            moved = realign(copy, n, n + every)
            clock["aligning"] += time.monotonic() - ticked
            record["alignments"].append({"batches": n, "moved_from_even": float(np.mean(moved))})
        x, mask = padded([train[int(i)][0] for i in group])
        if soft:  # a distribution per step, blank after an utterance's end
            targets = np.zeros((*mask.shape, SYMBOLS))
            targets[:, :, 0] = 1.0
            for r, path_r in enumerate(paths[n]):
                targets[: len(path_r), r] = path_r
        else:
            labels = np.zeros(mask.shape, dtype=np.int64)
            for r, path_r in enumerate(paths[n]):
                labels[: len(path_r), r] = path_r
        ticked = time.monotonic()
        brain.reset()
        for t in range(len(x)):
            if soft:  # the lesson's pull, then a greedy act; the fields step once, at the act
                soft_step(brain, x[t], targets[t])
            else:
                brain.step(x[t], teacher=labels[t])
        clock["training"] += time.monotonic() - ticked
        steps += int(mask.sum())
        paths.pop(n, None)
        if (n + 1) % probe_every == 0 or n + 1 == len(groups):
            brain.reset()
            copy = cd.Brain.load(brain.save(out / f"speech-{seed}-reading.npz"), **place)
            ticked = time.monotonic()
            answered = []
            for k in range(0, len(held), size):
                answered += batch_beliefs(copy, [x_ for x_, _ in held[k : k + size]])
            read_seconds = time.monotonic() - ticked
            clock["reading"] += read_seconds
            actions = [a for a, _ in answered]
            logps = [b for _, b in answered]
            result = {**read(actions, held), **locate(actions, logps, held)}
            if matcher is not None:
                result.update(vc.held_words(actions, held, matcher, decode))
            audio = sum(len(x_) for x_, _ in held) * STEP_SECONDS
            result["training_seconds_per_audio_second"] = clock["training"] / (steps * STEP_SECONDS)
            result["reading_seconds_per_audio_second"] = read_seconds / audio
            probe: dict[str, Any] = {
                "batches": n + 1, "utterances": int(sum(len(g) for g in groups[: n + 1])),
                "steps": steps, "clock": dict(clock), "score": result,
                "sample": [decode(lt.collapse(a)) for a in actions[:3]]}
            if protocol.get("transcripts"):  # what a later decoding or a word prior reads
                probe["transcripts"] = [decode(lt.collapse(a)) for a in actions]
                probe["confusions"] = confusions(logps, held)
            record["probes"].append(probe)
            write_atomic(partial, record)
            print(json.dumps({"seed": seed, "batches": n + 1,
                              "characters": round(result["characters"], 3),
                              "words": round(result["words"], 3),
                              "matched": round(result.get("words_matched", float("nan")), 3),
                              "training": round(clock["training"])}), flush=True)
    return record


# -------------------------------------------------------------------------------- controls


def control(cache: Path, model: str, mode: str, seed: int, parameters: int,
            protocol: dict[str, Any], epochs: int = 30,
            out: Path | None = None, stop_after: int | None = None) -> dict[str, Any] | None:
    """An LSTM or causal transformer at the brain's parameter count, trained by CTC on each
    utterance's text (one Adam step per utterance): ``online`` for the protocol's ``passes`` in
    the brain's orders, ``offline`` in shuffled epochs to the stopping rule (at most
    ``offline_epochs``, else ``epochs``); read the same way, after every pass with
    ``read_each_pass`` and at the end without. With ``out`` the learner is saved after every
    pass and a stopped run continues from there (``stop_after`` passes ends it as a lost machine
    would). ``controls_device`` in the protocol names where torch runs the learner (the CPU
    without it); its first weights are drawn on the CPU from the seed wherever it runs."""
    import controls as ct
    import torch

    torch.set_num_threads(1)
    device = str(protocol.get("controls_device", "cpu"))
    train, held = load(cache, "train"), load(cache, "held")
    inputs = train[0][0].shape[1]
    matcher = word_matcher(protocol)
    width = ct.matched_width(model, parameters, inputs, SYMBOLS)
    learner = ct.Learner(model, "ctc", width, inputs, SYMBOLS, seed, ct.RATES[mode],
                         device=device)
    order = np.random.default_rng([seed, 7]).permutation(len(train))
    record: dict[str, Any] = {"arm": f"{model}-ctc-{mode}", "seed": seed, "width": width,
                              "parameters": ct.count(learner.net), "probes": []}
    each = bool(protocol.get("read_each_pass"))

    def reading(n: int, passes_heard: int) -> None:
        actions, logps = [], []
        ticked = time.monotonic()
        with torch.no_grad():
            for x, _ in held:
                logits = learner.net(torch.from_numpy(x[None]).to(device))[0]
                actions.append(np.argmax(logits.cpu().numpy(), -1))
                logps.append(torch.log_softmax(logits, -1).cpu().numpy().astype(np.float64))
        read_seconds = time.monotonic() - ticked
        score = {**read(actions, held), **locate(actions, logps, held)}
        if matcher is not None:
            score.update(vc.held_words(actions, held, matcher, decode))
        audio = sum(len(x) for x, _ in held) * STEP_SECONDS
        score["training_seconds_per_audio_second"] = learner.work["seconds"] / (
            learner.work["frames"] * STEP_SECONDS)
        score["reading_seconds_per_audio_second"] = read_seconds / audio
        probe: dict[str, Any] = {"utterances": n, "passes": passes_heard,
                                 "work": dict(learner.work), "score": score}
        if protocol.get("transcripts"):
            probe["transcripts"] = [decode(lt.collapse(a)) for a in actions]
            probe["confusions"] = confusions(logps, held)
        record["probes"].append(probe)

    passes = (int(protocol.get("passes", 1)) if mode == "online"
              else int(protocol.get("offline_epochs", epochs)))
    best, waited, first = float("inf"), 0, 0
    state = None if out is None else out / f"speech-{record['arm']}-{seed}.state.pt"
    if state is not None and state.exists():  # a stopped run: from its last whole pass
        kept = torch.load(state, weights_only=False, map_location=device)
        learner.net.load_state_dict(kept["net"])
        learner.opt.load_state_dict(kept["opt"])
        learner.work = dict(kept["work"])
        best, waited, first = kept["best"], kept["waited"], kept["passes"]
        record["probes"] = kept["probes"]
        record["resumed"] = [*kept.get("resumed", []), first]
    heard, stopped = first, waited >= ct.PATIENCE
    while heard < passes and not stopped:
        if stop_after is not None and heard >= stop_after:
            return None
        total = 0.0
        for i in (order if heard == 0 else np.random.default_rng(
                [seed, 7, heard] if mode == "online" else [seed, heard]).permutation(len(train))):
            x, y = train[int(i)]
            total += learner.update(x, y)
        heard += 1
        if mode == "offline":
            if total < best - 1e-4:
                best, waited = total, 0
            else:
                waited += 1
            stopped = waited >= ct.PATIENCE
        if each:
            reading(len(train) * heard, heard)
        if state is not None:
            beside = state.with_name(state.name + ".tmp")
            torch.save({"net": learner.net.state_dict(), "opt": learner.opt.state_dict(),
                        "work": dict(learner.work), "best": best, "waited": waited,
                        "passes": heard, "probes": record["probes"],
                        "resumed": record.get("resumed", [])}, beside)
            os.replace(beside, state)
    if not each:
        reading(len(train) * heard, heard)
    record["passes"] = heard
    record["stopped_by_rule"] = bool(stopped)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "prepare-listed", "words", "life",
                                            "control"))
    parser.add_argument("--list", type=Path, help="the word list (prepare-listed, words)")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=("lstm", "transformer"))
    parser.add_argument("--mode", choices=("online", "offline"))
    parser.add_argument("--parameters", type=int)
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--limit", type=int, help="the first training utterances only")
    parser.add_argument("--splits", help="training splits of prepare-listed, comma-separated")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "words":
        args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        print(json.dumps(prepare(args.root, args.out, train=args.train)))
        return 0
    if args.command == "words":  # the list: the most frequent words of the training transcripts
        total = vc.counts(args.root, TRAINING)
        args.out.write_text("".join(f"{w} {total[w]}\n" for w in vc.top(total, 2000)))
        return 0
    if args.command == "prepare-listed":
        listed = [line.split()[0] for line in args.list.read_text().splitlines() if line.strip()]
        splits = tuple(args.splits.split(",")) if args.splits else TRAINING
        meta = prepare_listed(args.root, args.out, listed, window=args.window, training=splits)
        meta["training"] = list(splits)
        (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: v for k, v in meta.items() if k not in ("mean", "std")}))
        return 0
    assert args.cache is not None and args.protocol is not None
    protocol = json.loads(args.protocol.read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "life" and int(protocol.get("batch", 1)) > 1:
        record = life_batched(args.cache, protocol, args.seed, args.out)
        assert record is not None
        final = args.out / f"speech-{args.seed}-reading.npz"
        if final.exists():  # the brain of the last reading: the life's end
            os.replace(final, args.out / f"speech-{args.seed}-final.npz")
        write_atomic(args.out / f"speech-brain-{args.seed}.json.gz", record)
    elif args.command == "life":
        record = life(args.cache, protocol, args.seed, args.out, args.limit)
        lt.write_record(args.out / f"speech-brain-{args.seed}.json.gz", record)
    else:
        record = control(args.cache, args.model, args.mode, args.seed, args.parameters, protocol,
                         out=args.out if protocol.get("read_each_pass") else None)
        assert record is not None
        write_atomic(args.out / f"speech-{record['arm']}-{args.seed}.json.gz", record)
        print(json.dumps({"arm": record["arm"], "seed": args.seed,
                          **{k: round(v, 3) for k, v in record["probes"][-1]["score"].items()
                             if isinstance(v, float)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
