"""Reading text from a brain's per-step letters.

A recogniser says one symbol at each step: blank (0), a letter, or space. ``collapse`` reads a
stream of said symbols as text, adjacent repeats once and blanks dropped, the reading of CTC.
``WordBeam`` reads the per-step letter beliefs against a word list: a beam over a trie of the
list, so that every word it writes is a list word. Both are readings of what the brain said or
believed; neither changes the brain.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

BLANK = 0
ALPHABET = "abcdefghijklmnopqrstuvwxyz'"  # symbols 1 to 27; space is 28


@dataclass(frozen=True)
class Alphabet:
    """Symbols of a transcript: blank 0, the ``letters`` 1 to ``len(letters)``, space after."""

    letters: str = ALPHABET

    @property
    def space(self) -> int:
        return len(self.letters) + 1

    @property
    def size(self) -> int:
        """Symbols with blank and space: a recogniser's action count."""
        return len(self.letters) + 2

    def encode(self, text: str) -> np.ndarray:
        """A transcript as symbols: lowercase, characters off the alphabet read as spaces, single
        spaces between words."""
        words = "".join(c if c in self.letters else " " for c in text.lower()).split()
        out: list[int] = []
        for k, word in enumerate(words):
            if k:
                out.append(self.space)
            out.extend(self.letters.index(c) + 1 for c in word)
        return np.asarray(out, dtype=np.int64)

    def decode(self, symbols: Iterable[int]) -> str:
        """Symbols as text, blanks dropped, words parted by single spaces."""
        chars = []
        for s in symbols:
            s = int(s)
            if s == self.space:
                chars.append(" ")
            elif 1 <= s <= len(self.letters):
                chars.append(self.letters[s - 1])
        return " ".join("".join(chars).split())


def collapse(symbols: Iterable[int], *, blank: int = BLANK) -> list[int]:
    """The CTC reading of said symbols: adjacent repeats once, blanks dropped."""
    out: list[int] = []
    previous: int | None = None
    for s in symbols:
        s = int(s)
        if s != previous and s != blank:
            out.append(s)
        previous = s
    return out


def greedy(log_beliefs: Any, alphabet: Alphabet | None = None) -> str:
    """The text of the most believed symbol at each step, collapsed."""
    alphabet = alphabet or Alphabet()
    beliefs = np.asarray(log_beliefs)
    return alphabet.decode(collapse(np.argmax(beliefs, axis=1)))


@dataclass
class _Node:
    children: dict[int, _Node] = field(default_factory=dict)
    word: str | None = None  # the list word that ends here
    prior: float = 0.0  # its log prior


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    top = max(a, b)
    return top + math.log(math.exp(a - top) + math.exp(b - top))


class WordBeam:
    """A beam over a trie of a word list (CTC prefix beam search with every word a list word).

    ``words`` is the list; ``counts`` (optional, one per word) gives each word a log prior of
    ``prior_weight * log(count / total)``, added when the word is closed; ``word_bonus`` is added
    per word, so the beam trades words against letters. ``beam`` hypotheses are kept at each step
    and only the ``top`` most believed symbols of a step are tried. A hypothesis closes a word by
    saying space after it or at the end; ``decode`` returns the best whose words are all closed,
    and the best open one with its last word dropped where none closes.
    """

    def __init__(self, words: Sequence[str], *, counts: Sequence[float] | None = None,
                 alphabet: Alphabet | None = None, beam: int = 16, top: int = 8,
                 prior_weight: float = 0.0, word_bonus: float = 0.0) -> None:
        self.alphabet = alphabet or Alphabet()
        if beam < 1 or top < 1:
            raise ValueError("beam and top must be positive")
        if counts is not None and len(counts) != len(words):
            raise ValueError("counts holds one count per word")
        self.beam, self.top = int(beam), int(top)
        self.word_bonus = float(word_bonus)
        self.root = _Node()
        total = float(sum(counts)) if counts is not None else 0.0
        for k, word in enumerate(words):
            symbols = self.alphabet.encode(word)
            if len(symbols) == 0 or (symbols == self.alphabet.space).any():
                raise ValueError(f"list word {word!r} is not one word of the alphabet")
            node = self.root
            for s in symbols.tolist():
                node = node.children.setdefault(s, _Node())
            node.word = self.alphabet.decode(symbols)
            if counts is not None:
                if counts[k] <= 0:
                    raise ValueError("counts must be positive")
                node.prior = float(prior_weight) * math.log(counts[k] / total)

    def decode(self, log_beliefs: Any) -> str:
        """The list words best read from ``(steps, symbols)`` log beliefs (blank 0, the letters,
        space ``alphabet.space``)."""
        beliefs = np.asarray(log_beliefs, dtype=np.float64)
        if beliefs.ndim != 2 or beliefs.shape[1] != self.alphabet.size:
            raise ValueError(f"log beliefs are (steps, {self.alphabet.size})")
        space, blank = self.alphabet.space, BLANK
        # a hypothesis: (said symbols) -> [log p ending in blank, log p ending in a symbol,
        # trie node, closed words' score]
        start: tuple[int, ...] = ()
        beams: dict[tuple[int, ...], list[Any]] = {start: [0.0, -math.inf, self.root, 0.0]}
        for row in beliefs:
            order = np.argsort(row)[::-1][: self.top]
            tried = set(int(s) for s in order) | {blank}
            nxt: dict[tuple[int, ...], list[Any]] = {}

            def add(key: tuple[int, ...], pb: float, pnb: float, node: _Node, score: float,
                    nxt: dict[tuple[int, ...], list[Any]] = nxt) -> None:
                if key in nxt:
                    entry = nxt[key]
                    entry[0] = _logsumexp(entry[0], pb)
                    entry[1] = _logsumexp(entry[1], pnb)
                else:
                    nxt[key] = [pb, pnb, node, score]

            for key, (pb, pnb, node, score) in beams.items():
                total = _logsumexp(pb, pnb)
                last = key[-1] if key else None
                for s in tried:
                    p = float(row[s])
                    if s == blank:
                        add(key, total + p, -math.inf, node, score)
                        continue
                    if s == last:  # the same symbol again: held (no new symbol) ...
                        add(key, -math.inf, pnb + p, node, score)
                        if s == space:
                            continue  # a space is said once between words
                        child = node.children.get(s)
                        if child is not None:  # ... or a doubled letter after a blank
                            add(key + (s,), -math.inf, pb + p, child, score)
                        continue
                    if s == space:
                        if node is self.root or node.word is None:
                            continue  # a space only closes a list word
                        add(key + (s,), -math.inf, total + p, self.root,
                            score + node.prior + self.word_bonus)
                        continue
                    child = node.children.get(s)
                    if child is not None:
                        add(key + (s,), -math.inf, total + p, child, score)
            ranked = sorted(nxt.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]) + kv[1][3],
                            reverse=True)
            beams = dict(ranked[: self.beam])
        best_closed: tuple[float, tuple[int, ...]] | None = None
        best_open: tuple[float, tuple[int, ...]] | None = None
        for key, (pb, pnb, node, score) in beams.items():
            value = _logsumexp(pb, pnb) + score
            if node is self.root or node.word is not None:
                if node is not self.root:
                    value += node.prior + self.word_bonus
                if best_closed is None or value > best_closed[0]:
                    best_closed = (value, key)
            elif best_open is None or value > best_open[0]:
                best_open = (value, key)
        if best_closed is not None:
            return self.alphabet.decode(best_closed[1])
        assert best_open is not None
        said = list(best_open[1])
        while said and said[-1] != space:  # the open word is dropped
            said.pop()
        return self.alphabet.decode(said)
