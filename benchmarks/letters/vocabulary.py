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

    def __call__(self, word: str) -> str:
        if word in self.rank:
            return word
        if word not in self.cache:
            best = min(self.listed, key=lambda w: (distance(word, w), self.rank[w]))
            self.cache[word] = best
        return self.cache[word]


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
