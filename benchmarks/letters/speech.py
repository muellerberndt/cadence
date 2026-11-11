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

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"
SPACE = len(ALPHABET) + 1  # 28
SYMBOLS = len(ALPHABET) + 2  # blank, the letters and the apostrophe, space
RATE, WINDOW, HOP, STACK = 16000, 400, 160, 4


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
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < WINDOW:
        audio = np.pad(audio, (0, WINDOW - len(audio)))
    count = 1 + (len(audio) - WINDOW) // HOP
    index = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
    frames = audio[index] * np.hanning(WINDOW)[None, :]
    power = np.abs(np.fft.rfft(frames, 512)) ** 2
    energies = np.log(power @ bank.T + 1e-6)
    steps = count // STACK
    return energies[: steps * STACK].reshape(steps, STACK * bank.shape[0]).astype(np.float32)


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
    """The cached (steps, symbols) of each utterance."""
    with np.load(cache / f"{name}.npz") as data:
        x, text = data["steps"], data["text"]
        a = np.concatenate([[0], np.cumsum(data["lengths"])])
        b = np.concatenate([[0], np.cumsum(data["text_lengths"])])
    return [(x[a[i] : a[i + 1]], text[b[i] : b[i + 1]]) for i in range(len(a) - 1)]


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
    brain = cd.Brain.compose(train[0][0].shape[1], SYMBOLS, backend="cpu", seed=seed,
                             **protocol["brain"]["genes"])
    record: dict[str, Any] = {"seed": seed, "parameters": brain.parameters(), "probes": [],
                              "alignments": [], "describe": brain.describe()}
    paths = {int(i): even_path(len(train[i][0]), train[i][1]) for i in order[:every]}
    started, steps, begin = time.monotonic(), 0, 0
    partial = out / f"speech-brain-{seed}.partial.json.gz"
    if partial.exists():  # a life a lost machine stopped continues from its last reading
        record = lt.read_record(partial)
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
        brain.reset()
        for t in range(len(x)):
            brain.step(x[t][None], teacher=[int(paths[i][t])])
        steps += len(x)
        paths.pop(i, None)
        if (n + 1) % probe_every == 0 or n + 1 == len(order):
            path = brain.save(out / f"speech-{seed}-u{n + 1}.npz")
            copy = cd.Brain.load(path)
            answered = [beliefs(copy, x_) for x_, _ in held]
            actions = [a for a, _ in answered]
            result = {**read(actions, held), **locate(actions, [b for _, b in answered], held)}
            record["probes"].append({"utterances": n + 1, "steps": steps,
                                     "seconds": time.monotonic() - started, "score": result,
                                     "sample": [decode(lt.collapse(a)) for a in actions[:3]]})
            lt.write_record(partial, record)
            print(json.dumps({"seed": seed, "utterances": n + 1,
                              "characters": round(result["characters"], 3),
                              "words": round(result["words"], 3),
                              "elapsed": round(time.monotonic() - started)}), flush=True)
    return record


# -------------------------------------------------------------------------------- controls


def control(cache: Path, model: str, mode: str, seed: int, parameters: int,
            protocol: dict[str, Any], epochs: int = 30) -> dict[str, Any]:
    """An LSTM or causal transformer at the brain's parameter count, trained by CTC on each
    utterance's text (one Adam step per utterance), one pass or offline to the stopping rule,
    read the same way."""
    import controls as ct
    import torch

    torch.set_num_threads(1)
    train, held = load(cache, "train"), load(cache, "held")
    inputs = train[0][0].shape[1]
    width = ct.matched_width(model, parameters, inputs, SYMBOLS)
    learner = ct.Learner(model, "ctc", width, inputs, SYMBOLS, seed, ct.RATES[mode])
    order = np.random.default_rng([seed, 7]).permutation(len(train))
    record: dict[str, Any] = {"arm": f"{model}-ctc-{mode}", "seed": seed, "width": width,
                              "parameters": ct.count(learner.net), "probes": []}

    def reading(n: int) -> None:
        actions, logps = [], []
        with torch.no_grad():
            for x, _ in held:
                logits = learner.net(torch.from_numpy(x[None]))[0]
                actions.append(np.argmax(logits.numpy(), -1))
                logps.append(torch.log_softmax(logits, -1).numpy().astype(np.float64))
        record["probes"].append({"utterances": n, "work": dict(learner.work),
                                 "score": {**read(actions, held), **locate(actions, logps, held)}})

    passes = 1 if mode == "online" else epochs
    best, waited = float("inf"), 0
    for epoch in range(passes):
        total = 0.0
        for i in (order if epoch == 0 else np.random.default_rng([seed, epoch]).permutation(
                len(train))):
            x, y = train[int(i)]
            total += learner.update(x, y)
        if mode == "offline":
            if total < best - 1e-4:
                best, waited = total, 0
            else:
                waited += 1
            if waited >= ct.PATIENCE:
                break
    reading(len(train) * (epoch + 1))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "life", "control"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=("lstm", "transformer"))
    parser.add_argument("--mode", choices=("online", "offline"))
    parser.add_argument("--parameters", type=int)
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--limit", type=int, help="the first training utterances only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.root, args.out, train=args.train)))
        return 0
    assert args.cache is not None and args.protocol is not None
    protocol = json.loads(args.protocol.read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "life":
        record = life(args.cache, protocol, args.seed, args.out, args.limit)
        lt.write_record(args.out / f"speech-brain-{args.seed}.json.gz", record)
    else:
        sys.path.insert(0, str(_here))
        record = control(args.cache, args.model, args.mode, args.seed, args.parameters, protocol)
        lt.write_record(args.out / f"speech-{record['arm']}-{args.seed}.json.gz", record)
        print(json.dumps({"arm": record["arm"], "seed": args.seed,
                          **{k: round(v, 3) for k, v in record["probes"][-1]["score"].items()
                             if isinstance(v, float)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
