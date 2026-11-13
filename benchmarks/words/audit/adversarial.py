"""Adversarial questions and scenes (cadence#326 audit).

Questions changed, asked of the brain at every test question of the founder's life:

- ``shuffled``: its words in a drawn order; the world's answer, and ``same``: the brain's own
  answer to the question in order.
- ``extra_untaught``: a nonce word never heard put in at a drawn place; the world's answer.
- ``false_colour``: "where is the <colour> <kind>" with a colour the thing does not have (W1
  where questions); ``follows``: "I do not know"; ``same``: the place of the kind, the colour
  word left unread.
- ``true_colour``: "where is the <colour> <kind>" with the thing's own colour; the world's
  answer.
- ``question_words_only``: the content word taken out; ``follows``: "I do not know".
- ``content_only``: the content word alone; the kinds of word said (``categories``).
- ``extra_kind``: the word of another kind in the room added at the end (W1); whose answer was
  said (``categories``).
- ``read_swapped``: a where question's thing asked its colour and the reverse (W1); the world's
  answer to the question the words ask.

Scenes the world never builds, on the brain the life left (``after``):

- ``close_<d>``: two things of different kinds and colours ``d`` metres apart (the world keeps
  0.45 m), each asked where it is and its colour, what has its colour, what is at its place;
- ``same_kind``: two things of one kind in two colours and two places around the parrot, asked
  where the kind is (both places), its colour (both colours) and what has each colour;
- ``crowded``: rooms of the eight trained kinds, every question form;
- ``test_after_new_words``: rooms drawn as the test's rooms are, every question form, asked
  after the new words' namings (the chamber asks W1 before them).

    python adversarial.py <founder> <out> [short]
"""

from __future__ import annotations

import math
import sys
from typing import Any

import common
import numpy as np
import truth
import world
from common import Harness, Probe, scored


def untaught(h: Harness) -> str:
    """A nonce form the founder's lexicon has never heard."""
    known = set(h.brain.lexicon.forms) | set(world.lexicon(h.seed).foils)
    rng = np.random.default_rng(np.random.SeedSequence([h.seed, 326, 78]))
    while True:
        form = "".join(str(rng.choice(list(world.CONSONANTS))) + str(rng.choice(list(world.VOWELS)))
                       for _ in range(3))
        if form not in known:
            return form


def category(h: Harness, said: list[str]) -> str:
    lex = h.lex
    if said == [lex.none]:
        return "none"
    if not said or said == [""]:
        return "nothing"
    for name, group in (("kind", lex.kind), ("colour", lex.colour), ("place", lex.place)):
        if all(w in group for w in said):
            return name
    return "mixed"


class Shuffled(Probe):
    name = "shuffled"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        order = q.text.split()
        for _ in range(5):
            order = [order[int(i)] for i in h.rng.permutation(len(order))]
            if order != q.text.split():
                break
        return scored(h, q, h.brain.ask(" ".join(order)), base=said)


class ExtraUntaught(Probe):
    name = "extra_untaught"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        parts = q.text.split()
        parts.insert(int(h.rng.integers(len(parts) + 1)), untaught(h))
        return scored(h, q, h.brain.ask(" ".join(parts)), base=said)


class Colour(Probe):
    """A where question with a colour word before the kind's word."""

    def __init__(self, true: bool):
        self.true = true
        self.name = "true_colour" if true else "false_colour"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        if not gate.startswith("W1") or q.form != "where":
            return None
        assert h.room is not None
        thing = next(t for t in h.room.things if t.kind == q.kind)
        colour = thing.colour if self.true else int(
            h.rng.choice([c for c in range(world.COLOURS) if c != thing.colour]))
        w = h.lex.question
        text = f"{w['where']} {w['is']} {w['the']} {h.lex.colour[colour]} {h.lex.kind[q.kind]}"
        out = h.brain.ask(text)
        asked = world.Question(text, "where", q.answer if self.true else [h.lex.none],
                               q.about, q.kind)
        result = scored(h, asked, out, predicted=asked.answer, base=said)
        return result


class QuestionWordsOnly(Probe):
    name = "question_words_only"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        asked = common.question_words(h.lex)
        text = " ".join(w for w in q.text.split() if w in asked)
        return scored(h, q, h.brain.ask(text), predicted=[h.lex.none])


class ContentOnly(Probe):
    name = "content_only"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        text = " ".join(common.content(h.lex, q.text))
        if not text:
            return None
        out = h.brain.ask(text)
        result = scored(h, q, out, base=said)
        result["category"] = category(h, out)
        return result


class ExtraKind(Probe):
    name = "extra_kind"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        if not gate.startswith("W1"):
            return None
        assert h.room is not None
        others = [t for t in h.room.things if t.kind != q.kind]
        if not others:
            return None
        other = others[int(h.rng.integers(len(others)))]
        out = h.brain.ask(f"{q.text} {h.lex.kind[other.kind]}")
        second = truth.answer(h.lex, truth.triples(h.room),
                              world.Question(q.text, q.form, [], q.about, other.kind))
        result = scored(h, q, out, base=said)
        first_ok, second_ok = world.right(q, out), sorted(out) == sorted(second)
        result["category"] = ("both" if first_ok and second_ok else "first" if first_ok
                              else "second" if second_ok else "other")
        return result


class ReadSwapped(Probe):
    """The question form asked with the other form's words: a where question's thing asked
    "what colour is", and a colour question's thing asked "where is"."""

    name = "read_swapped"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        if not gate.startswith("W1"):
            return None
        assert h.room is not None
        swapped = (world.colour_of if q.form == "where" else world.where)(h.lex, h.room, q.kind)
        return scored(h, swapped, h.brain.ask(swapped.text))


def probes() -> list[Probe]:
    return [Shuffled(), ExtraUntaught(), Colour(False), Colour(True), QuestionWordsOnly(),
            ContentOnly(), ExtraKind(), ReadSwapped()]


# -- scenes the world never builds

def _place(rng: np.random.Generator, near: np.ndarray | None = None, d: float = 0.0,
           away: list[np.ndarray] = ()) -> np.ndarray:
    for _ in range(10000):
        if near is None:
            r, a = rng.uniform(0.8, 1.6), rng.uniform(-math.pi, math.pi)
            p = np.array([r * math.cos(a), r * math.sin(a)])
        else:
            a = rng.uniform(-math.pi, math.pi)
            p = near + d * np.array([math.cos(a), math.sin(a)])
            if not 0.6 <= float(np.linalg.norm(p)) <= 1.8:
                continue
        if all(np.linalg.norm(p - q) >= 0.45 for q in away):
            return p
    raise RuntimeError("no place")


def _ask(h: Harness, book: dict[str, Any], name: str, q: world.Question) -> None:
    said = h.brain.ask(q.text)
    common.tally(book, name, scored(h, q, said))


def close_rooms(h: Harness, rng: np.random.Generator, d: float, rooms: int,
                book: dict[str, Any]) -> None:
    for _ in range(rooms):
        kinds = [int(k) for k in rng.choice(world.TRAINED, size=4, replace=False)]
        colours = [int(rng.choice([c for c in range(world.COLOURS) if c != h.unseen[k]]))
                   for k in kinds]
        if colours[1] == colours[0]:
            colours[1] = next(c for c in range(world.COLOURS)
                              if c not in (colours[0], h.unseen[kinds[1]]))
        a = _place(rng)
        b = _place(rng, a, d)
        rest = [_place(rng, away=[a, b])]
        rest.append(_place(rng, away=[a, b, rest[0]]))
        places = [a, b, *rest]
        r = world.Room([world.Thing(k, c, p) for k, c, p in zip(kinds, colours, places,
                                                              strict=True)],
                       heading=float(rng.uniform(-math.pi, math.pi)))
        h.room = r
        common.look_around(h.brain.map, h.brain.body, r)
        lex = h.lex
        for t in r.things[:2]:
            _ask(h, book, "where", world.where(lex, r, t.kind))
            _ask(h, book, "colour", world.colour_of(lex, r, t.kind))
            _ask(h, book, "by_colour", world.by_colour(lex, r, t.colour))
            _ask(h, book, "by_place", world.by_place(lex, r, world.place_of(r.heading, t.place)))


def same_kind_rooms(h: Harness, rng: np.random.Generator, rooms: int,
                    book: dict[str, Any]) -> None:
    lex, w = h.lex, h.lex.question
    for _ in range(rooms):
        heading = float(rng.uniform(-math.pi, math.pi))
        k = int(rng.integers(world.TRAINED))
        options = [c for c in range(world.COLOURS) if c != h.unseen[k]]
        c1, c2 = (int(c) for c in rng.choice(options, size=2, replace=False))
        for _ in range(1000):
            a, b = _place(rng), _place(rng)
            if (np.linalg.norm(a - b) >= 0.8
                    and world.place_of(heading, a) != world.place_of(heading, b)):
                break
        others = [int(j) for j in rng.choice([j for j in range(world.TRAINED) if j != k],
                                             size=2, replace=False)]
        rest = [_place(rng, away=[a, b])]
        rest.append(_place(rng, away=[a, b, rest[0]]))
        things = [world.Thing(k, c1, a), world.Thing(k, c2, b)] + [
            world.Thing(j, int(rng.choice([c for c in range(world.COLOURS)
                                           if c not in (c1, c2, h.unseen[j])])), p)
            for j, p in zip(others, rest, strict=True)]
        r = world.Room(things, heading=heading)
        h.room = r
        common.look_around(h.brain.map, h.brain.body, r)
        both_places = sorted(lex.place[world.place_of(heading, p)] for p in (a, b))
        both_colours = sorted(lex.colour[c] for c in (c1, c2))
        _ask(h, book, "where_both", world.Question(
            f"{w['where']} {w['is']} {w['the']} {lex.kind[k]}", "where", both_places, "several",
            k))
        _ask(h, book, "colour_both", world.Question(
            f"{w['what']} {w['colour']} {w['is']} {w['the']} {lex.kind[k]}", "colour",
            both_colours, "several", k))
        for c in (c1, c2):
            _ask(h, book, "by_colour", world.by_colour(lex, r, c))
        held = h.brain.map.things()
        book.setdefault("map_holds_two", {"n": 0, "ok": 0})
        book["map_holds_two"]["n"] += 1
        book["map_holds_two"]["ok"] += int(sum(1 for e in held if e["kind"] == k
                                               and e["place"] is not None) == 2)


def crowded_rooms(h: Harness, rng: np.random.Generator, rooms: int, book: dict[str, Any]) -> None:
    import words

    trained = list(range(world.TRAINED))
    for _ in range(rooms):
        r = world.room(rng, trained, world.TRAINED, avoid=h.unseen)
        h.room = r
        words.look_around(h.brain, r)
        for q in words.room_questions(h.lex, r, rng, absent=[], foils=[]):
            _ask(h, book, words.gate_of(q), q)


def test_rooms_again(h: Harness, rng: np.random.Generator, rooms: int,
                     book: dict[str, Any]) -> None:
    """Rooms drawn as the test's rooms are, asked every question form, on the brain the life
    left: after the new words' namings, where the chamber asks W1 before them."""
    import words

    trained = list(range(world.TRAINED))
    for _ in range(rooms):
        r = world.room(rng, trained, int(rng.integers(4, 7)), avoid=h.unseen)
        h.room = r
        words.look_around(h.brain, r)
        for q in words.room_questions(h.lex, r, rng, absent=trained, foils=list(range(6, 12))):
            _ask(h, book, words.gate_of(q), q)


def scenes(h: Harness) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([h.seed, 326, 79]))
    rooms = 4 if h.settings.get("test_rooms", 30) < 10 else 16
    out: dict[str, Any] = {}
    for d in (0.1, 0.2, 0.3, 0.6):
        book: dict[str, Any] = {}
        close_rooms(h, rng, d, rooms, book)
        out[f"close_{d}"] = common.shares(book)
    book = {}
    same_kind_rooms(h, rng, rooms, book)
    held = book.pop("map_holds_two")
    out["same_kind"] = common.shares(book)
    out["same_kind"]["map_holds_two"] = held["ok"] / max(held["n"], 1)
    out["same_kind"]["examples"] = {g: r["examples"] for g, r in book.items()}
    book = {}
    crowded_rooms(h, rng, rooms, book)
    out["crowded"] = common.shares(book)
    out["crowded"]["examples"] = {g: r["examples"] for g, r in book.items()}
    book = {}
    test_rooms_again(h, rng, 2 * rooms, book)
    out["test_after_new_words"] = common.shares(book)
    out["test_after_new_words"]["examples"] = {g: r["examples"] for g, r in book.items()}
    return out


def main(argv: list[str]) -> None:
    seed, out, settings = common.arguments(argv)
    result = common.replay(seed, probes(), settings, after=scenes)
    print(common.write(out, "adversarial", seed, result))


if __name__ == "__main__":
    main(sys.argv)
