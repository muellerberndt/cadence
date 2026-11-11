"""A closed word list for the letters chamber's read speech, and words read matched to it.

The list is the most frequent words of LibriSpeech's own training transcripts (CC BY 4.0), so the
list and the audio share one source and one licence. An utterance belongs to the list's corpus
when every word it holds is on the list. The words a learner says are read twice, the same for
every learner: against the open lexicon (as said), and matched to the list, each said word
replaced by the list word nearest to it in letters (fewest edits, the more frequent word on a
tie). The match is an output stage of the instrument, like the CTC reading, and never part of a
learner's answer path.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("letters", _here / "letters.py")
assert _spec is not None and _spec.loader is not None
lt = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("letters", lt)
_spec.loader.exec_module(lt)


def words_of(text: str) -> list[str]:
    """A transcript's words as the chamber spells them: lowercase letters and apostrophes."""
    return "".join(c if c.isalpha() or c == "'" else " " for c in text.lower()).split()


def counts(root: Path, splits: tuple[str, ...]) -> Counter[str]:
    """Word counts over the transcripts of LibriSpeech ``splits`` under ``root``."""
    total: Counter[str] = Counter()
    for split in splits:
        for listing in sorted((root / "LibriSpeech" / split).rglob("*.trans.txt")):
            for line in listing.read_text().splitlines():
                total.update(words_of(line.partition(" ")[2]))
    return total


def top(total: Counter[str], n: int = 2000) -> list[str]:
    """The ``n`` most frequent words, most frequent first, ties in alphabetical order."""
    return [w for w, _ in sorted(total.items(), key=lambda item: (-item[1], item[0]))[:n]]


def on_list(text: str, listed: set[str]) -> bool:
    """Whether every word of a transcript is on the list."""
    words = words_of(text)
    return bool(words) and all(w in listed for w in words)


def distance(a: str, b: str) -> int:
    """Edits (insertions, deletions, substitutions) between two words."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (ca != cb))
    return row[-1]


class Matcher:
    """Each said word to the nearest list word: fewest edits, the earlier (more frequent) list
    word on a tie; a word on the list is itself."""

    def __init__(self, listed: list[str]) -> None:
        self.listed = list(listed)
        self.rank = {w: k for k, w in enumerate(self.listed)}
        self.cache: dict[str, str] = {}
        self.lengths = np.array([len(w) for w in self.listed], dtype=np.int64)
        self.codes = np.zeros((len(self.listed), int(self.lengths.max()) + 1), dtype=np.int64)
        for k, w in enumerate(self.listed):
            self.codes[k, : len(w)] = [ord(c) for c in w]

    def distances(self, word: str) -> np.ndarray:
        """Edits from ``word`` to every list word, in the list's order (``distance`` of each,
        computed over the whole list at once)."""
        width = self.codes.shape[1]
        row = np.tile(np.arange(width, dtype=np.int64), (len(self.listed), 1))
        for i, c in enumerate(word, 1):
            cost = (self.codes[:, : width - 1] != ord(c)).astype(np.int64)
            new = np.empty_like(row)
            new[:, 0] = i
            for j in range(1, width):
                new[:, j] = np.minimum(np.minimum(row[:, j] + 1, new[:, j - 1] + 1),
                                       row[:, j - 1] + cost[:, j - 1])
            row = new
        return np.asarray(row[np.arange(len(self.listed)), self.lengths])

    def __call__(self, word: str) -> str:
        if word in self.rank:
            return word
        if word not in self.cache:
            self.cache[word] = self.listed[int(np.argmin(self.distances(word)))]
        return self.cache[word]

    def place(self, said: str, meant: str) -> int:
        """Where the list word ``meant`` stands among the list words nearest ``said``: 0 where
        it is the nearest (fewest edits, the more frequent word first on a tie)."""
        d = self.distances(said)
        k = self.rank[meant]
        return int(np.sum((d < d[k]) | ((d == d[k]) & (np.arange(len(d)) < k))))


def align(reference: list[str], hypothesis: list[str]) -> list[tuple[int | None, int | None]]:
    """The fewest-edits alignment of two word sequences as pairs of indexes: a reference word
    with the said word in its place, with None where it was dropped, and None with a said word
    that stands for no reference word."""
    n, m = len(reference), len(hypothesis)
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1]))
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def candidates(hypotheses: list[str], references: list[str], matcher: Matcher,
               within: tuple[int, ...] = (1, 5, 20)) -> dict[str, Any]:
    """What a word prior would have to choose among. Each reference word, by the said word in
    its place: said right; among the ``within`` list words nearest the said word (the room a
    prior choosing among that many candidates has); further; or dropped. Shares of the
    reference words, and the said words that stand for no reference word per reference word."""
    counts = {"right": 0, "further": 0, "dropped": 0, **{f"within_{k}": 0 for k in within}}
    total = extra = 0
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        said, ref = hypothesis.split(), reference.split()
        for i, j in align(ref, said):
            if i is None:
                extra += 1
                continue
            total += 1
            if j is None:
                counts["dropped"] += 1
            elif said[j] == ref[i]:
                counts["right"] += 1
            elif ref[i] not in matcher.rank:
                counts["further"] += 1
            else:
                place = matcher.place(said[j], ref[i])
                for k in within:
                    if place < k:
                        counts[f"within_{k}"] += 1
                        break
                else:
                    counts["further"] += 1
    out: dict[str, Any] = {k: v / max(total, 1) for k, v in counts.items()}
    out["extra_per_reference_word"] = extra / max(total, 1)
    out["reference_words"] = total
    return out


def read_words(hypotheses: list[str], references: list[str],
               matcher: Matcher) -> dict[str, Any]:
    """Words against the open lexicon and matched to the list, pooled over utterances: one minus
    the word error rate, and the share of utterances said exactly."""
    open_errors = matched_errors = total = 0
    open_exact = matched_exact = 0
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        said, ref = hypothesis.split(), reference.split()
        matched = [matcher(w) for w in said]
        open_errors += lt.edits(ref, said)["errors"]
        matched_errors += lt.edits(ref, matched)["errors"]
        open_exact += int(said == ref)
        matched_exact += int(matched == ref)
        total += len(ref)
    n = max(len(references), 1)
    return {
        "words_open": 1.0 - open_errors / max(total, 1),
        "words_matched": 1.0 - matched_errors / max(total, 1),
        "exact_open": open_exact / n,
        "exact_matched": matched_exact / n,
    }


def held_words(actions: list[np.ndarray], held: list[tuple[np.ndarray, np.ndarray]],
               matcher: Matcher, decode: Any) -> dict[str, Any]:
    """``read_words`` of a learner's answers to the held-out utterances (``decode`` maps a symbol
    list to text, as ``speech.decode``)."""
    hypotheses = [decode(lt.collapse(a)) for a in actions]
    references = [decode(text.tolist()) for _, text in held]
    return read_words(hypotheses, references, matcher)
