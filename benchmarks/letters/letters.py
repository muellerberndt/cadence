"""The letters chamber: letters heard as sounds, said once per sound, and words formed from them.

A synthetic speaker spells words from a lexicon over an alphabet of ``letters`` (16). A letter's
sound is a fixed vector of ``features`` (16) values in [0, 1] that lasts one to four frames
(``sound_frames``), every frame with Gaussian noise; silence is zero. No word holds a letter twice
back to back, so every sound boundary inside a word is a change of letter. One to three silent
frames follow each word. ``held_out`` words of the lexicon are never said in training; every
letter they hold is said in the trained words.

The brain says one of ``letters + 2`` symbols at every frame: blank (0), a letter (1 to
``letters``) or space (``letters + 1``). Its lesson at a frame is the symbol due there:

- ``onset`` (the supplied alignment): a letter at the first frame of its sound, space at the first
  silent frame after a word, blank on every other frame;
- ``even``: a word's letters spread evenly over the word's frames and space at the middle of its
  silence, the timing a speech instrument supplies when it knows where each word lies and not
  where each letter's sound begins;
- ``own`` (the brain's own alignment): a letter from its sound's first frame until the brain's
  act says it, blank on the frames of the sound after that, and space likewise in the silence
  after a word;
- ``even_frames``: a word's frames split evenly among its letters, each letter taught on every
  frame of its share, space on every silent frame after the word;
- ``forced``: as ``even_frames`` at first; every ``align_every`` words a frozen copy of the brain
  answers the next words greedily, and each word's frames are re-assigned to its known letters by
  the monotonic best path through the copy's beliefs (``forced_path``), each letter taught on
  every frame of its path. The path is the teacher's, computed outside the brain;
- ``frames``: a letter on every frame of its sound and space on every silent frame after a word,
  the letter's identity with no decision when to say it;
- ``delay=d`` moves each due symbol ``d`` frames later, and never onto or before the symbol before
  it.

``motor="split"`` gives the brain two motor slots that settle together: the symbol it hears
(taught as ``frames``) and, apart, whether to say it (taught 1 where the arm's lessons put a
symbol, 0 elsewhere); it says the first slot's symbol where the second says 1.

Readings are the instrument's, the same for every learner, on a held-out stream answered greedily
without learning:

- **collapsed**: the CTC reading, adjacent repeats read once and blanks dropped; characters
  (1 - character error rate over letters and spaces), insertions, deletions and substitutions per
  reference character, words (1 - word error rate), exact word recall on trained and on held-out
  words;
- **said**: blanks dropped and nothing merged, so a letter said on two steps is a letter said
  twice: the reading of "a letter said once per sound";
- **held steps**: for each due symbol, the steps it is said from its due frame up to the next due
  frame (one is said once; two or more is held; zero is missed);
- **stages**, each read with the stage before it supplied: the letter from the sound (the letter
  the motor prefers at its due frame among the letters alone), its emission given the letter (said
  at its due frame, not said on the frames after), and the word given its letters (a word said
  whole among the words whose every letter the motor preferred at its due frame).

``python benchmarks/letters/letters.py --arm default --seed 0 --out <dir>`` runs one life and writes
its record; ``--report <dir>`` tabulates a directory.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

SCHEMA = "letters/1"
PROTOCOL = Path(__file__).with_name("protocol.json")
BLANK = 0


def load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    protocol: dict[str, Any] = json.loads(path.read_bytes())
    for key in ("world", "training_words", "probe_every", "evaluation_words", "brain", "arms"):
        if key not in protocol:
            raise ValueError(f"protocol lacks {key}")
    return protocol


# ----------------------------------------------------------------------------------- world


def rng_for(seed: int, purpose: str) -> np.random.Generator:
    """The independent random stream of one seed for one purpose."""
    purposes = ("lexicon", "training", "evaluation", "held_out", "learner")
    return np.random.default_rng([int(seed), purposes.index(purpose)])


def space_symbol(letters: int) -> int:
    return letters + 1


def symbols(letters: int) -> int:
    """Blank, the letters and space."""
    return letters + 2


def make_lexicon(
    seed: int,
    *,
    letters: int = 16,
    features: int = 16,
    words: int = 40,
    held_out: int = 8,
    word_letters: tuple[int, int] = (3, 5),
) -> dict[str, Any]:
    """The letters' sounds and the words of one seed.

    ``sounds`` (letters, features) are uniform in [0, 1]. Each word is a distinct string of
    ``word_letters`` letters with no letter twice back to back. The last ``held_out`` words drawn
    are never said in training; the trained words hold every letter.
    """
    rng = rng_for(seed, "lexicon")
    if letters < 2 or held_out >= words:
        raise ValueError("a lexicon needs two letters and a trained word")
    sounds = rng.random((letters, features))
    while True:
        spelled: list[tuple[int, ...]] = []
        while len(spelled) < words:
            length = int(rng.integers(word_letters[0], word_letters[1] + 1))
            word = [int(rng.integers(letters))]
            while len(word) < length:
                letter = int(rng.integers(letters))
                if letter != word[-1]:
                    word.append(letter)
            if tuple(word) not in spelled:
                spelled.append(tuple(word))
        trained = spelled[: words - held_out]
        if len({letter for word in trained for letter in word}) == letters:
            break
    return {
        "seed": int(seed),
        "sounds": sounds,
        "words": [list(w) for w in spelled],
        "trained": list(range(words - held_out)),
        "held_out": list(range(words - held_out, words)),
    }


def spelling(lexicon: dict[str, Any], word: int) -> str:
    """A word as text: letter k is the k-th lowercase letter."""
    return "".join(chr(ord("a") + letter) for letter in lexicon["words"][word])


def make_stream(
    lexicon: dict[str, Any],
    words: int,
    rng: np.random.Generator,
    *,
    pool: str = "trained",
    noise: float = 0.1,
    sound_frames: tuple[int, int] = (1, 4),
    silence: tuple[int, int] = (1, 3),
) -> dict[str, Any]:
    """A stream of ``words`` words drawn uniformly from ``pool`` (``trained`` or ``all``), each
    after one to three silent frames, the last followed by its own silence.

    ``frames`` (T, features); ``track`` (T,) the letter of each frame, -1 for silence;
    ``sounds`` the due symbols in order as (symbol, first frame, last frame); ``said`` the words in
    order; ``text`` the reference, the words joined by single spaces."""
    letters, features = lexicon["sounds"].shape
    choices = lexicon["trained"] if pool == "trained" else list(range(len(lexicon["words"])))
    said = [int(choices[i]) for i in rng.integers(len(choices), size=words)]
    track: list[int] = []
    sounds: list[tuple[int, int, int]] = []
    track.extend([-1] * int(rng.integers(silence[0], silence[1] + 1)))
    for word in said:
        for letter in lexicon["words"][word]:
            length = int(rng.integers(sound_frames[0], sound_frames[1] + 1))
            sounds.append((letter + 1, len(track), len(track) + length - 1))
            track.extend([letter] * length)
        gap = int(rng.integers(silence[0], silence[1] + 1))
        sounds.append((space_symbol(letters), len(track), len(track) + gap - 1))
        track.extend([-1] * gap)
    held = np.asarray(track, dtype=np.int64)
    clean = np.where(held[:, None] >= 0, lexicon["sounds"][np.maximum(held, 0)], 0.0)
    frames = clean + rng.normal(0.0, noise, clean.shape)
    return {
        "frames": frames,
        "track": held,
        "sounds": sounds,
        "said": said,
        "text": " ".join(spelling(lexicon, w) for w in said),
        "held_out": [w in set(lexicon["held_out"]) for w in said],
    }


def due(stream: dict[str, Any], *, delay: int = 0, even: bool = False) -> np.ndarray:
    """The frame each sound's symbol is due: its first frame plus ``delay``, at least one frame
    after the symbol before it, within the stream. With ``even`` a word's letters are spread
    evenly over the word's frames whatever their sounds' lengths, letter k of n at
    ``first + floor((k + 1/2) * frames / n)``, and a space at the middle of its silence: the
    timing a speech instrument supplies when it knows only where each word lies."""
    frames = len(stream["track"])
    sounds = stream["sounds"]
    starts = np.array([start for _, start, _ in sounds], dtype=np.int64)
    if even:
        k = 0
        while k < len(sounds):
            j = k
            while j < len(sounds) and stream["track"][sounds[j][1]] >= 0:
                j += 1  # sounds k..j-1 are one word's letters, sound j its silence
            first, last = sounds[k][1], sounds[j - 1][2]
            n = j - k
            for i in range(n):
                starts[k + i] = first + int((i + 0.5) * (last - first + 1) / n)
            if j < len(sounds):
                starts[j] = (sounds[j][1] + sounds[j][2]) // 2
            k = j + 1
    out = np.zeros(len(sounds), dtype=np.int64)
    previous = -1
    for k in range(len(sounds)):
        previous = min(max(int(starts[k]) + delay, previous + 1), frames - 1)
        out[k] = previous
    return out


def labels(stream: dict[str, Any], mode: str = "onset", *, delay: int = 0) -> np.ndarray:
    """The symbol taught at every frame (``onset``, ``even`` or ``frames``; module docstring)."""
    y = np.zeros(len(stream["track"]), dtype=np.int64)
    if mode in ("onset", "even"):
        times = due(stream, delay=delay, even=mode == "even")
        for (symbol, _, _), t in zip(stream["sounds"], times, strict=True):
            y[t] = symbol
    elif mode == "frames":
        for symbol, start, end in stream["sounds"]:
            y[start + delay : end + 1 + delay] = symbol
    elif mode in ("even_frames", "forced"):  # ``forced`` starts from the even split
        for word in words_of(stream):
            first, last = word["frames"]
            n = len(word["letters"])
            for k, symbol in enumerate(word["letters"]):
                length = last - first
                y[first + k * length // n : first + (k + 1) * length // n] = symbol
            y[word["silence"][0] : word["silence"][1]] = word["space"]
    else:
        raise ValueError(f"unknown label mode {mode}")
    return y


def words_of(stream: dict[str, Any]) -> list[dict[str, Any]]:
    """Each word of a stream as a teacher who knows where words lie sees it: its letters' symbols
    in order, its frames ``[first, last)`` and its silence ``[first, last)``, and the space."""
    out: list[dict[str, Any]] = []
    held: list[tuple[int, int, int]] = []
    for symbol, start, end in stream["sounds"]:
        if stream["track"][start] >= 0:
            held.append((symbol, start, end))
            continue
        out.append({"letters": [h[0] for h in held], "frames": (held[0][1], held[-1][2] + 1),
                    "silence": (start, end + 1), "space": symbol})
        held = []
    return out


def forced_path(scores: np.ndarray) -> np.ndarray:
    """The monotonic best path of a fixed sequence through frames: ``scores`` (frames, n) holds
    the log belief in the sequence's k-th symbol at each frame; every symbol takes at least one
    frame, in order. Returns the index into the sequence at each frame (Viterbi, as CTC's forced
    alignment without blanks)."""
    frames, n = scores.shape
    if frames < n:
        raise ValueError("a sequence needs a frame per symbol")
    best = np.full((frames, n), -np.inf)
    came = np.zeros((frames, n), dtype=bool)  # True: entered k at this frame from k - 1
    best[0, 0] = scores[0, 0]
    for t in range(1, frames):
        stay = best[t - 1]
        enter = np.concatenate([[-np.inf], best[t - 1, :-1]])
        came[t] = enter > stay
        best[t] = np.maximum(stay, enter) + scores[t]
    path = np.zeros(frames, dtype=np.int64)
    k = n - 1
    for t in range(frames - 1, -1, -1):
        path[t] = k
        if t and came[t, k]:
            k -= 1
    return path


def ctc_path(scores: np.ndarray, sequence: np.ndarray, *, states_out: bool = False) -> np.ndarray:
    """CTC's forced alignment without a backward pass: the symbol at each frame on the best path
    of ``sequence`` through ``scores`` (frames, symbols), the log beliefs, where blank (0) may
    fill any frame between symbols and must part a symbol from the same symbol after it. With
    ``states_out`` the path's state at each frame instead: ``2k + 1`` for the sequence's k-th
    symbol, even for a blank."""
    frames = len(scores)
    states = np.zeros(2 * len(sequence) + 1, dtype=np.int64)
    states[1::2] = sequence
    n = len(states)
    skip = np.zeros(n, dtype=bool)  # a symbol state reachable from two states back
    skip[3::2] = states[3::2] != states[1:-2:2]
    best = np.full((frames, n), -np.inf)
    back = np.zeros((frames, n), dtype=np.int64)
    best[0, 0] = scores[0, states[0]]
    if n > 1:
        best[0, 1] = scores[0, states[1]]
    for t in range(1, frames):
        stay = best[t - 1]
        one = np.concatenate([[-np.inf], best[t - 1, :-1]])
        two = np.where(skip, np.concatenate([[-np.inf, -np.inf], best[t - 1, :-2]]), -np.inf)
        choice = np.argmax(np.stack([stay, one, two]), axis=0)
        best[t] = np.choose(choice, [stay, one, two]) + scores[t, states]
        back[t] = choice
    k = n - 1 if n == 1 or best[-1, n - 1] >= best[-1, n - 2] else n - 2
    path = np.zeros(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = k if states_out else states[k]
        k -= int(back[t, k])
    return path


def forced_labels(stream: dict[str, Any], beliefs: np.ndarray, words: range,
                  y: np.ndarray) -> float:
    """Re-align ``words`` of the stream in ``y`` from ``beliefs`` (frames, symbols), the log
    belief of a frozen copy of the brain in each symbol at each frame: each word's frames go to
    its known letters by the monotonic best path; its silence stays space. Returns the share of
    those words' letter frames whose new label is the letter heard there."""
    spans = words_of(stream)
    right = total = 0
    for i in words:
        word = spans[i]
        first, last = word["frames"]
        sequence = np.asarray(word["letters"], dtype=np.int64)
        path = forced_path(beliefs[first:last][:, sequence])
        y[first:last] = sequence[path]
        right += int(np.sum(y[first:last] == stream["track"][first:last] + 1))
        total += last - first
    return right / max(total, 1)


# --------------------------------------------------------------------------------- reading


def collapse(actions: np.ndarray) -> list[int]:
    """The CTC reading: adjacent repeats once, blanks dropped."""
    out: list[int] = []
    previous = -1
    for a in np.asarray(actions, dtype=np.int64).tolist():
        if a != previous and a != BLANK:
            out.append(a)
        previous = a
    return out


def said(actions: np.ndarray) -> list[int]:
    """Every step said: blanks dropped, nothing merged."""
    return [a for a in np.asarray(actions, dtype=np.int64).tolist() if a != BLANK]


def text_of(sequence: list[int], letters: int) -> str:
    space = space_symbol(letters)
    chars = [" " if s == space else chr(ord("a") + s - 1) for s in sequence]
    return " ".join("".join(chars).split())


def edits(reference: list[Any], hypothesis: list[Any]) -> dict[str, Any]:
    """Levenshtein alignment: substitutions, deletions, insertions and, per reference item,
    whether it was matched exactly."""
    n, m = len(reference), len(hypothesis)
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = reference[i - 1]
        row, above = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(above[j] + 1, row[j - 1] + 1, above[j - 1] + (ri != hypothesis[j - 1]))
    i, j = n, m
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    matched = [False] * n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            if reference[i - 1] == hypothesis[j - 1]:
                matched[i - 1] = True
            else:
                counts["substitutions"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            counts["deletions"] += 1
            i -= 1
        else:
            counts["insertions"] += 1
            j -= 1
    return {**counts, "errors": int(d[n, m]), "matched": matched}


def reading(text: str, reference: str) -> dict[str, Any]:
    """Characters and words of one transcript against the reference."""
    chars = edits(list(reference), list(text))
    words_ref, words_hyp = reference.split(), text.split()
    words = edits(words_ref, words_hyp)
    total = max(len(reference), 1)
    return {
        "characters": 1.0 - chars["errors"] / total,
        "insertions": chars["insertions"] / total,
        "deletions": chars["deletions"] / total,
        "substitutions": chars["substitutions"] / total,
        "words": 1.0 - words["errors"] / max(len(words_ref), 1),
        "word_insertions": words["insertions"],
        "word_matched": words["matched"],
        "exact": text == reference,
    }


def score(
    actions: np.ndarray,
    stream: dict[str, Any],
    letters: int,
    *,
    preferred: np.ndarray | None = None,
    delay: int = 0,
    even: bool = False,
) -> dict[str, Any]:
    """Every reading of one answered stream (module docstring). ``preferred`` (T,) is the letter
    or space the motor prefers at each frame among the symbols other than blank, for the stage
    readings; ``delay`` places the due frames as the lessons did."""
    a = np.asarray(actions, dtype=np.int64)
    reference = stream["text"]
    collapsed = reading(text_of(collapse(a), letters), reference)
    strict = reading(text_of(said(a), letters), reference)
    held_out = np.asarray(stream["held_out"], dtype=bool)
    matched = np.asarray(collapsed.pop("word_matched"), dtype=bool)
    strict.pop("word_matched")
    times = due(stream, delay=delay, even=even)
    ends = np.append(times[1:], len(a))
    held = np.array(
        [int(np.sum(a[t:e] == s)) for (s, _, _), t, e in zip(stream["sounds"], times, ends,
                                                               strict=True)]
    )
    letter_sound = np.array([s != space_symbol(letters) for s, _, _ in stream["sounds"]])
    # separate runs of the symbol in its window: two or more read twice even after the collapse
    runs = np.array(
        [sum(1 for v in collapse(a[t:e]) if v == s)
         for (s, _, _), t, e in zip(stream["sounds"], times, ends, strict=True)]
    )
    out: dict[str, Any] = {
        "collapsed": collapsed,
        "said": strict,
        "trained_words": float(matched[~held_out].mean()) if (~held_out).any() else None,
        "held_out_words": float(matched[held_out].mean()) if held_out.any() else None,
        "held_steps": {
            "letters_mean": float(held[letter_sound].mean()),
            "letters_once": float(np.mean(held[letter_sound] == 1)),
            "letters_held": float(np.mean(held[letter_sound] >= 2)),
            "letters_missed": float(np.mean(held[letter_sound] == 0)),
            "letters_apart": float(np.mean(runs[letter_sound] >= 2)),
            "spaces_mean": float(held[~letter_sound].mean()),
        },
    }
    lengths = np.array([e - b + 1 for _, b, e in stream["sounds"]])
    first = np.array([True] + [s == space_symbol(letters) for s, _, _ in stream["sounds"][:-1]])
    out["by_length"] = {
        str(n): {"sounds": int(np.sum(letter_sound & (lengths == n))),
                 "once": float(np.mean(held[letter_sound & (lengths == n)] == 1)),
                 "missed": float(np.mean(held[letter_sound & (lengths == n)] == 0)),
                 "held": float(np.mean(held[letter_sound & (lengths == n)] >= 2))}
        for n in sorted(set(lengths[letter_sound].tolist()))
    }
    out["by_place"] = {
        place: {"once": float(np.mean(held[letter_sound & mask] == 1)),
                "missed": float(np.mean(held[letter_sound & mask] == 0)),
                "held": float(np.mean(held[letter_sound & mask] >= 2))}
        for place, mask in (("first", first), ("later", ~first))
    }
    symbol = np.array([s for s, _, _ in stream["sounds"]])
    at_due = a[times]
    after = [a[t + 1 : e] for t, e in zip(times, ends, strict=True)]
    out["emission"] = {
        "at_due": float(np.mean(at_due != BLANK)),
        "right_at_due": float(np.mean(at_due == symbol)),
        "after_due_said": float(
            np.mean(np.concatenate([x != BLANK for x in after if len(x)]))
        ) if any(len(x) for x in after) else 0.0,
    }
    if preferred is not None:
        pref = np.asarray(preferred, dtype=np.int64)[times]
        identified = pref == symbol
        out["stages"] = {
            "letter_from_sound": float(identified[letter_sound].mean()),
            "space_from_silence": float(identified[~letter_sound].mean()),
            "emission_given_letter": float(np.mean(at_due[identified] == symbol[identified])),
        }
        # the word given its letters: words whose letters were all identified at their due frames
        word_ok, k = [], 0
        for _ in stream["said"]:
            spelled = []
            while letter_sound[k]:
                spelled.append(bool(identified[k]))
                k += 1
            k += 1  # the space
            word_ok.append(all(spelled))
        known = np.asarray(word_ok, dtype=bool)
        out["stages"]["words_identified"] = float(known.mean())
        out["stages"]["word_given_letters"] = (
            float(matched[known].mean()) if known.any() else None
        )
    return out


# ------------------------------------------------------------------------------ brain arms


def world_settings(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    w = protocol["world"]
    lexicon = {k: (tuple(w[k]) if isinstance(w[k], list) else w[k])
               for k in ("letters", "features", "words", "held_out", "word_letters")}
    stream = {k: (tuple(w[k]) if isinstance(w[k], list) else w[k])
              for k in ("noise", "sound_frames", "silence")}
    return lexicon, stream


def world(seed: int, protocol: dict[str, Any], sizes: dict[str, int]) -> dict[str, Any]:
    lexicon_kw, stream_kw = world_settings(protocol)
    lexicon = make_lexicon(seed, **lexicon_kw)
    train = make_stream(lexicon, sizes["training_words"], rng_for(seed, "training"), **stream_kw)
    test = make_stream(lexicon, sizes["evaluation_words"], rng_for(seed, "evaluation"),
                       pool="all", **stream_kw)
    return {"lexicon": lexicon, "train": train, "test": test}


def arm_settings(arm: str, protocol: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["arms"][arm]
    return {
        "genes": {**protocol["brain"]["genes"], **spec.get("genes", {})},
        "labels": spec.get("labels", "onset"),
        "delay": int(spec.get("delay", 0)),
        "motor": spec.get("motor", "one"),
        "align_every": int(spec.get("align_every", 0)),
    }


def compose(arm: str, seed: int, protocol: dict[str, Any]) -> Any:
    import cadence as cd

    letters = int(protocol["world"]["letters"])
    settings = arm_settings(arm, protocol)
    genes = settings["genes"]
    if settings["motor"] == "split":  # the symbol heard and, apart, whether to say it
        return cd.Brain.compose(int(protocol["world"]["features"]), symbols(letters) + 2,
                                slots=[symbols(letters), 2], backend="cpu", seed=seed, **genes)
    return cd.Brain.compose(int(protocol["world"]["features"]), symbols(letters), backend="cpu",
                            seed=seed, **genes)


def brain_answers(brain: Any, x: np.ndarray, *, record: bool = False) -> dict[str, Any]:
    """A brain's greedy answers to a stream without learning (the caller passes a loaded and
    reset copy): actions, the preferred symbol other than blank, and with ``record`` the settled
    activation of every population at every frame. A brain of two motor slots (``split``) says
    the first slot's symbol where the second slot says 1, and blank elsewhere."""
    populations = brain.connectome.populations
    motor = np.asarray(populations["motor"], dtype=np.int64)
    split = len(brain.learner.slot_sizes) == 2
    width = int(brain.learner.slot_sizes[0])
    actions, preferred = [], []
    states: dict[str, list[np.ndarray]] = {name: [] for name in populations}
    for row in x:
        act = np.asarray(brain.act(row[None], greedy=True)).reshape(-1)
        actions.append(int(act[0]) if not split or int(act[1]) == 1 else BLANK)
        activation = np.asarray(brain.basal_ganglia.state.activation).reshape(-1)
        preferred.append(int(np.argmax(activation[motor][1:width])) + 1)
        if record:
            for name, index in populations.items():
                states[name].append(activation[np.asarray(index, dtype=np.int64)])
    out: dict[str, Any] = {
        "actions": np.asarray(actions, dtype=np.int64),
        "preferred": np.asarray(preferred, dtype=np.int64),
    }
    if record:
        out["states"] = {name: np.asarray(v) for name, v in states.items() if v and len(v[0])}
    return out


def probe(features: np.ndarray, target: np.ndarray, *, ridge: float = 1e-2) -> float:
    """A linear read of ``target`` from ``features``: ridge least squares on one-hot targets,
    fitted on the first 70% of the rows and scored on the rest."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=np.int64)
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    cut = int(0.7 * len(x))
    mean, scale = x[:cut].mean(0), x[:cut].std(0) + 1e-8
    z = np.hstack([(x - mean) / scale, np.ones((len(x), 1))])
    onehot = (y[:cut, None] == classes[None]).astype(float)
    w = np.linalg.solve(z[:cut].T @ z[:cut] + ridge * len(z[:cut]) * np.eye(z.shape[1]),
                        z[:cut].T @ onehot)
    guess = classes[np.argmax(z[cut:] @ w, axis=1)]
    return float(np.mean(guess == y[cut:]))


def probes(states: dict[str, np.ndarray], stream: dict[str, Any], letters: int) -> dict[str, Any]:
    """What the settled populations hold at each frame of a letter's sound: the letter, whether
    the frame is its sound's first, and the letter of the frame before. Fitted on the stream's
    first 70% of letter frames, scored on the rest; chance beside each."""
    track = stream["track"]
    first = np.zeros(len(track), dtype=bool)
    for symbol, start, _ in stream["sounds"]:
        if symbol != space_symbol(letters):
            first[start] = True
    previous = np.concatenate([[-1], track[:-1]])
    rows = track >= 0
    out: dict[str, Any] = {
        "chance": {
            "letter": float(np.max(np.bincount(track[rows])) / rows.sum()),
            "onset": float(max(first[rows].mean(), 1 - first[rows].mean())),
            "previous": float(np.max(np.bincount(previous[rows] + 1)) / rows.sum()),
        }
    }
    for name, value in states.items():
        out[name] = {
            "letter": probe(value[rows], track[rows]),
            "onset": probe(value[rows], first[rows].astype(np.int64)),
            "previous": probe(value[rows], previous[rows]),
        }
    together = np.hstack([states[n] for n in states if n != "motor"])
    out["all_but_motor"] = {
        "letter": probe(together[rows], track[rows]),
        "onset": probe(together[rows], first[rows].astype(np.int64)),
        "previous": probe(together[rows], previous[rows]),
    }
    return out


def brain_life(arm: str, seed: int, protocol: dict[str, Any], out: Path,
               sizes: dict[str, int]) -> dict[str, Any]:
    """One continuing life: every training frame through ``Brain.step(frame,
    teacher=symbol)``; every ``probe_every`` words a saved copy, loaded and reset, answers the
    held-out stream greedily."""
    import cadence as cd

    settings = arm_settings(arm, protocol)
    letters = int(protocol["world"]["letters"])
    w = world(seed, protocol, sizes)
    train, test = w["train"], w["test"]
    own = settings["labels"] == "own"
    y = labels(train, "onset" if own else settings["labels"], delay=settings["delay"])
    forced = settings["labels"] == "forced"
    spans = words_of(train)
    realign = ({spans[w]["frames"][0]: w for w in range(0, len(spans), settings["align_every"])
                if w > 0} if forced else {})
    alignments: list[dict[str, Any]] = []
    # what the brain said in training: per sound, the frame of its first saying and its count
    first_said = np.full(len(train["sounds"]), -1, dtype=np.int64)
    said_count = np.zeros(len(train["sounds"]), dtype=np.int64)
    window_start = 0
    # with ``own`` lessons: the sound each frame belongs to, and whether the brain has said it
    sound_of = np.full(len(train["track"]), -1, dtype=np.int64)
    for k, (_, start, end) in enumerate(train["sounds"]):
        sound_of[start : end + 1] = k
    said_sound = np.zeros(len(train["sounds"]), dtype=bool)
    split = settings["motor"] == "split"
    heard = labels(train, "frames")  # the split motor's first slot: the symbol heard
    brain = compose(arm, seed, protocol)
    x = np.asarray(train["frames"], dtype=float)
    record: dict[str, Any] = {
        "arm": arm, "seed": seed, "settings": settings, "parameters": brain.parameters(),
        "describe": brain.describe(), "probes": [], "alignments": alignments,
    }
    word_ends = {start for symbol, start, _ in train["sounds"] if symbol == space_symbol(letters)}
    words = 0
    lessons = 0
    started = time.monotonic()
    for t in range(len(x)):
        if t in realign:  # the teacher re-aligns the next words on a frozen copy's beliefs
            w0 = realign[t]
            w1 = min(w0 + settings["align_every"], len(spans))
            end = spans[w1 - 1]["silence"][1]
            path = brain.save(out / f"{arm}-{seed}-align.npz")
            copy = cd.Brain.load(path)
            copy.reset()
            path.unlink()
            beliefs = copy_beliefs(copy, x[t:end])
            full = np.zeros((len(x), beliefs.shape[1]))
            full[t:end] = beliefs
            alignments.append({"words": w0, "right": forced_labels(train, full, range(w0, w1), y)})
        label = int(y[t])
        if own and sound_of[t] >= 0:
            k = int(sound_of[t])
            label = BLANK if said_sound[k] else int(train["sounds"][k][0])
        if split:
            issued = brain.step(x[t][None], teacher=[[int(heard[t]), int(label != BLANK)]])
            issued = issued[:, 0] if int(issued[0, 1]) == 1 else np.array([BLANK])
        else:
            issued = brain.step(x[t][None], teacher=[label])
        if own and sound_of[t] >= 0 and int(issued[0]) == train["sounds"][sound_of[t]][0]:
            said_sound[sound_of[t]] = True
        if sound_of[t] >= 0 and int(issued[0]) == train["sounds"][sound_of[t]][0]:
            k = int(sound_of[t])
            said_count[k] += 1
            if first_said[k] < 0:
                first_said[k] = t - train["sounds"][k][1]
        lessons += int(brain.last_learning.get("demonstration_accepted_presentations", 1)) \
            if brain.last_learning else 0
        if t in word_ends:
            words += 1
            if words % sizes["probe_every"] == 0 or words == sizes["training_words"]:
                final = words == sizes["training_words"]
                path = brain.save(out / f"{arm}-{seed}-w{words}.npz")
                copy = cd.Brain.load(path)
                copy.reset()
                answered = brain_answers(copy, test["frames"], record=final)
                result = score(answered["actions"], test, letters,
                               preferred=answered["preferred"], delay=settings["delay"],
                               even=settings["labels"] == "even")
                entry = {"words": words, "frames": t + 1, "lessons": lessons,
                         "seconds": time.monotonic() - started, "score": result,
                         "training": said_in_training(train, first_said, said_count,
                                                      window_start, t, letters)}
                window_start = t + 1
                if final:
                    entry["probes"] = probes(answered["states"], test, letters)
                    entry["actions"] = answered["actions"].tolist()
                    entry["sample"] = sample(answered["actions"], test, letters)
                record["probes"].append(entry)
                if not final:
                    path.unlink()
                print(json.dumps({"arm": arm, "seed": seed, "words": words,
                                  "chars": round(result["collapsed"]["characters"], 3),
                                  "said_chars": round(result["said"]["characters"], 3),
                                  "held": round(result["held_steps"]["letters_mean"], 2),
                                  "elapsed": round(time.monotonic() - started)}), flush=True)
    return record


def copy_beliefs(brain: Any, x: np.ndarray) -> np.ndarray:
    """A frozen copy's log belief in each symbol at each frame of a stream, answered greedily:
    the log-softmax of its settled motor activations at the learner's temperature."""
    motor = np.asarray(brain.connectome.populations["motor"], dtype=np.int64)
    temperature = float(brain.describe()["genes"]["temperature"])
    out = np.zeros((len(x), len(motor)))
    for t, row in enumerate(x):
        brain.act(row[None], greedy=True)
        drive = np.asarray(brain.basal_ganglia.state.activation).reshape(-1)[motor] / temperature
        out[t] = drive - drive.max() - np.log(np.sum(np.exp(drive - drive.max())))
    return out


def said_in_training(stream: dict[str, Any], first_said: np.ndarray, said_count: np.ndarray,
                     start: int, end: int, letters: int) -> dict[str, Any]:
    """Of the letter sounds heard in frames ``[start, end]``: the share first said at each frame
    of the sound (0 its first) or never, and the mean frames said."""
    keep = np.array([s != space_symbol(letters) and start <= b and e <= end
                     for s, b, e in stream["sounds"]])
    if not keep.any():
        return {}
    offsets = first_said[keep]
    return {
        "first_said": {str(k): float(np.mean(offsets == k)) for k in range(4)},
        "never": float(np.mean(offsets < 0)),
        "said_frames": float(said_count[keep].mean()),
    }


def sample(
    actions: np.ndarray, stream: dict[str, Any], letters: int, n: int = 12
) -> dict[str, str]:
    """The first ``n`` words of the stream: the reference, the collapsed reading and the
    steps as said (``_`` blank, ``|`` space)."""
    space = space_symbol(letters)
    silences = [start for s, start, _ in stream["sounds"] if s == space]
    stop = silences[n - 1] if len(silences) >= n else len(actions)
    a = np.asarray(actions[: stop + 1])
    space = space_symbol(letters)
    steps = "".join("_" if v == BLANK else "|" if v == space else chr(ord("a") + v - 1) for v in a)
    return {
        "reference": " ".join(stream["text"].split()[:n]),
        "collapsed": text_of(collapse(a), letters),
        "steps": steps,
    }


# ---------------------------------------------------------------------------------- report


def read_record(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(gzip.decompress(path.read_bytes()))
    return data


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.write_bytes(gzip.compress(json.dumps(record, sort_keys=True).encode(), mtime=0))


def report(directory: Path) -> str:
    rows = []
    for path in sorted(directory.glob("*.json.gz")):
        r = read_record(path)
        if not r.get("probes"):
            continue
        s = r["probes"][-1]["score"]
        st = s.get("stages", {})
        rows.append(
            f"| {r['arm']} | {r['seed']} | {r['parameters']:,} | {r['probes'][-1]['words']:,} | "
            f"{s['collapsed']['characters']:.3f} | {s['collapsed']['insertions']:.3f} | "
            f"{s['collapsed']['words']:.3f} | {s['trained_words']:.2f} | "
            f"{s['held_out_words']:.2f} | {s['said']['characters']:.3f} | "
            f"{s['said']['insertions']:.3f} | {s['held_steps']['letters_mean']:.2f} | "
            f"{s['held_steps']['letters_once']:.2f} | "
            f"{st.get('letter_from_sound', float('nan')):.2f} | "
            f"{st.get('emission_given_letter', float('nan')):.2f} | "
            f"{s['emission']['after_due_said']:.2f} |"
        )
    head = (
        "| Arm | Founder | Parameters | Words heard | Characters | Insertions per char | Words | "
        "Trained words | Held-out words | Said: characters | Said: insertions | Held steps | "
        "Said once | Letter from sound | Emission given letter | Said after due |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- |"
    )
    return head + "\n" + "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--words", type=int, help="training words, in place of the protocol's")
    parser.add_argument("--probe-every", type=int, help="words between readings")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.report:
        print(report(args.report))
        return 0
    protocol = load_protocol(args.protocol)
    sizes = {k: int(protocol[k]) for k in ("training_words", "probe_every", "evaluation_words")}
    if args.smoke:
        sizes = {"training_words": 20, "probe_every": 10, "evaluation_words": 10}
    if args.words:
        sizes["training_words"] = args.words
    if args.probe_every:
        sizes["probe_every"] = args.probe_every
    assert args.out is not None and args.arm is not None
    args.out.mkdir(parents=True, exist_ok=True)
    record = brain_life(args.arm, args.seed, protocol, args.out, sizes)
    write_record(args.out / f"{args.arm}-{args.seed}.json.gz", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
