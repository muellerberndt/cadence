"""The grammar of the parrot's world (cadence#334, the grammar rung): questions and answers about
a room, generated with depth as a knob.

A sentence is what the parrot hears, one word form a moment, beside a meaning that only the
teacher, the scorer and the probes read (``Sentence``): the brain is given the forms alone. The
same record serves a chamber of sentences elsewhere (the whole-sentence work of cadence#308): a
list of forms and, per form, the role it plays in the meaning.

**Phrases** (a description picks out things in the room):

- an attribute: a colour, "round", "big" or "small";
- a place around the body: "behind you", "next to you" (near);
- a relation to another description: "left of", "right of", "in front of", "behind", "next to"
  (relations of relations: "the cup left of the ball behind you");
- joined conditions: "and", "or", "not".

A clause inside a phrase ("the ball that was behind you") joins with the rung of time, where a
thing's place before it was carried is in the records (cadence#326 W8).

**Questions**: where is X, what is <description>, what colour is X, is X round, how many
<colour> things are there, is there X, what is not <colour>, and the word-order pairs ("what is
left of the ball" against "what is the ball left of").

**Answers** are short sentences: "the ball is behind me", "two red balls", "one ball" (number
agreement: a plural marker after the noun), "yes", "no", "i do not know".

``depth`` counts the modifiers of the description a question uses (an attribute, a place or a
relation each adds one; a relation adds its own description's depth). Every phrase is checked
against the room: a description used in a question picks out exactly the things the answer
names.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import meaning_world as world
import numpy as np

RELATIONS = ("left_of", "right_of", "in_front_of", "behind_of", "next_to")
NUMBERS = ("zero", "one", "two", "three", "four", "five")
SIZES = ("big", "small")
GRAMMAR_WORDS = ("what", "is", "are", "the", "you", "me", "and", "or", "not", "that", "was",
                 "will", "where", "colour", "which", "how", "many", "there", "a", "of", "it",
                 "yes", "no", "i_do_not_know", "plural", "thing", "left", "right", "in", "front",
                 "behind_rel", "next", "to", "round") + NUMBERS + SIZES


@dataclass
class Sentence:
    """What is heard (``words``, the founder's forms) and, for the teacher and the probes only,
    the role of each word and the meaning."""

    words: list[str]
    roles: list[str]
    meaning: dict[str, Any] = field(default_factory=dict)


@dataclass
class Item:
    """A question, its answer and what they test."""

    question: Sentence
    answer: Sentence
    form: str
    depth: int
    pair: str | None = None  # the key two word-order twins share


class Words:
    """A founder's nonce forms for the grammar: the world's words and the grammar's own."""

    def __init__(self, seed: int):
        self.lex = world.lexicon(seed)
        taken = self.lex.every()
        forms = world.nonce(seed, len(GRAMMAR_WORDS), salt=world.SALT + 1)
        forms = [f for f in forms if f not in taken]
        while len(forms) < len(GRAMMAR_WORDS):  # a collision with the base lexicon is redrawn
            more = world.nonce(seed + 7919 * len(forms), 1, salt=world.SALT + 2)[0]
            if more not in taken and more not in forms:
                forms.append(more)
        self.g = dict(zip(GRAMMAR_WORDS, forms, strict=True))

    def __call__(self, name: str) -> str:
        return self.g[name]


# -- what the room holds, in the parrot's frame


def frame(r: world.Room, t: world.Thing) -> np.ndarray:
    """A thing's place in the parrot's frame: ahead (x) and to the left (y), metres."""
    c, s = math.cos(-r.heading), math.sin(-r.heading)
    x, y = float(t.place[0]), float(t.place[1])
    return np.array([c * x - s * y, s * x + c * y])


def related(r: world.Room, a: world.Thing, b: world.Thing, rel: str) -> bool:
    """Whether ``a`` stands in ``rel`` to ``b``, as the parrot sees them."""
    if a is b:
        return False
    pa, pb = frame(r, a), frame(r, b)
    d = pa - pb
    if rel == "left_of":
        return bool(d[1] > 0.3)
    if rel == "right_of":
        return bool(d[1] < -0.3)
    if rel == "in_front_of":  # between the parrot and b
        return bool(np.linalg.norm(pa) < np.linalg.norm(pb) - 0.3 and abs(
            world.direction(0.0, pa) - world.direction(0.0, pb)) < 0.6)
    if rel == "behind_of":
        return bool(np.linalg.norm(pa) > np.linalg.norm(pb) + 0.3 and abs(
            world.direction(0.0, pa) - world.direction(0.0, pb)) < 0.6)
    if rel == "next_to":
        return bool(np.linalg.norm(d) < 0.7)
    raise ValueError(rel)


def size_of(t: world.Thing) -> str:
    return "big" if t.width >= 0.16 else "small"


# -- descriptions: a filter over things, its words, its depth


@dataclass
class Description:
    test: Callable[[world.Thing], bool]
    words: list[str]
    roles: list[str]
    depth: int
    meaning: dict[str, Any]


def attribute(w: Words, kind: str, value: int | str) -> Description:
    lex = w.lex
    if kind == "colour":
        c = int(value)
        return Description(lambda t: t.colour == c, [lex.colour[c]], ["colour"], 1,
                           {"colour": world.COLOUR_NAMES[c]})
    if kind == "round":
        return Description(world.is_round, [lex.attribute["round"]], ["shape"], 1,
                           {"shape": "round"})
    if kind == "size":
        s = str(value)
        return Description(lambda t: size_of(t) == s, [w(s)], ["size"], 1, {"size": s})
    raise ValueError(kind)


def body_place(w: Words, r: world.Room, place: int | str) -> Description:
    lex = w.lex
    if place == "near":
        return Description(world.near, [w("next"), w("to"), w("you")], ["place", "place", "place"],
                           1, {"place": "near"})
    q = int(place)
    return Description(lambda t: world.place_of(r.heading, t.place) == q,
                       [lex.place[q], w("you")], ["place", "place"], 1,
                       {"place": world.PLACE_NAMES[q]})


def relation_words(w: Words, rel: str) -> list[str]:
    return {"left_of": [w("left"), w("of")], "right_of": [w("right"), w("of")],
            "in_front_of": [w("in"), w("front"), w("of")], "behind_of": [w("behind_rel"), w("of")],
            "next_to": [w("next"), w("to")]}[rel]


def noun_phrase(w: Words, kind: int, mods: list[Description]) -> Description:
    """"the <attributes> <noun> <places and relations>": attributes before the noun, places and
    relations after it, as English orders them."""
    lex = w.lex
    pre = [m for m in mods if m.roles[0] in ("colour", "shape", "size")]
    post = [m for m in mods if m not in pre]
    words, roles = [w("the")], ["det"]
    for m in pre:
        words += m.words
        roles += m.roles
    words.append(lex.kind[kind])
    roles.append("noun")
    for m in post:
        words += m.words
        roles += m.roles
    tests = [m.test for m in mods]
    return Description(lambda t: t.kind == kind and all(f(t) for f in tests), words, roles,
                       sum(m.depth for m in mods),
                       {"kind": world.KIND_NAMES[kind], "mods": [m.meaning for m in mods]})


def relation(w: Words, r: world.Room, rel: str, ground: Description) -> Description:
    """"<rel> <ground>": a thing in ``rel`` to the one thing ``ground`` picks out."""
    grounds = [t for t in r.things if ground.test(t)]
    if len(grounds) != 1:
        raise ValueError("a relation's ground picks out one thing")
    g = grounds[0]
    words = relation_words(w, rel) + ground.words
    roles = ["relation"] * len(relation_words(w, rel)) + ground.roles
    return Description(lambda t: related(r, t, g, rel), words, roles, 1 + ground.depth,
                       {"relation": rel, "ground": ground.meaning})


def join(w: Words, op: str, a: Description, b: Description | None = None) -> Description:
    if op == "not":
        return Description(lambda t: not a.test(t), [w("not")] + a.words, ["not"] + a.roles,
                           a.depth + 1, {"not": a.meaning})
    assert b is not None
    if op == "and":
        return Description(lambda t: a.test(t) and b.test(t), a.words + [w("and")] + b.words,
                           a.roles + ["and"] + b.roles, a.depth + b.depth,
                           {"and": [a.meaning, b.meaning]})
    return Description(lambda t: a.test(t) or b.test(t), a.words + [w("or")] + b.words,
                       a.roles + ["or"] + b.roles, a.depth + b.depth,
                       {"or": [a.meaning, b.meaning]})


def modifiers(w: Words, r: world.Room, t: world.Thing, rng: np.random.Generator,
              depth: int) -> list[Description]:
    """Up to ``depth`` modifiers true of ``t``: attributes, a body place, or a relation to
    another thing described in turn (a relation of a relation when depth allows)."""
    out: list[Description] = []
    budget = depth
    options = ["colour", "round", "size", "place", "relation"]
    for _ in range(20):
        if budget <= 0:
            break
        choice = str(rng.choice(options))
        if choice == "colour" and not any(m.roles == ["colour"] for m in out):
            out.append(attribute(w, "colour", t.colour))
            budget -= 1
        elif choice == "round" and world.is_round(t) and not any(
                m.roles == ["shape"] for m in out):
            out.append(attribute(w, "round", 0))
            budget -= 1
        elif choice == "size" and not any(m.roles == ["size"] for m in out):
            out.append(attribute(w, "size", size_of(t)))
            budget -= 1
        elif choice == "place" and not any(m.roles[0] == "place" for m in out):
            out.append(body_place(w, r, "near" if world.near(t) and rng.random() < 0.3
                                  else world.place_of(r.heading, t.place)))
            budget -= 1
        elif choice == "relation" and budget >= 1 and not any(
                m.roles[0] == "relation" for m in out):
            others = [o for o in r.things if o is not t]
            rng.shuffle(others)
            for o in others:
                rels = [rel for rel in RELATIONS if related(r, t, o, rel)]
                if not rels:
                    continue
                ground = describe(w, r, o, rng, budget - 1, unique=True)
                if ground is None:
                    continue
                out.append(relation(w, r, str(rng.choice(rels)), ground))
                budget -= out[-1].depth
                break
    return out


def describe(w: Words, r: world.Room, t: world.Thing, rng: np.random.Generator, depth: int, *,
             unique: bool = True) -> Description | None:
    """A noun phrase for ``t`` of at most ``depth`` modifiers that picks out ``t`` alone (with
    ``unique``), or None."""
    for _ in range(30):
        np_ = noun_phrase(w, t.kind, modifiers(w, r, t, rng, depth))
        picked = [o for o in r.things if np_.test(o)]
        if not unique or picked == [t]:
            return np_
    return None


# -- questions and their answers


def _s(words: list[str], roles: list[str], meaning: dict[str, Any] | None = None) -> Sentence:
    return Sentence(list(words), list(roles), dict(meaning or {}))


def counted(w: Words, n: int, kind: int) -> Sentence:
    """"one ball", "two balls": the number word, the noun, and the plural marker past one."""
    lex = w.lex
    words, roles = [w(NUMBERS[min(n, 5)]), lex.kind[kind]], ["number", "noun"]
    if n != 1:
        words.append(w("plural"))
        roles.append("plural")
    return _s(words, roles, {"count": n, "kind": world.KIND_NAMES[kind]})


def items(w: Words, r: world.Room, rng: np.random.Generator, *, depth: int,
          forms: tuple[str, ...] | None = None) -> list[Item]:
    """Every question form the room supports at ``depth``, each with its answer."""
    lex = w.lex
    out: list[Item] = []
    things = list(r.things)
    allowed = set(forms) if forms else None

    def want(form: str) -> bool:
        return allowed is None or form in allowed

    for t in things:
        d = describe(w, r, t, rng, depth)
        if d is None:
            continue
        noun = lex.kind[t.kind]
        q = world.place_of(r.heading, t.place)
        if want("where"):
            out.append(Item(_s([w("where"), w("is")] + d.words, ["q", "verb"] + d.roles,
                               {"ask": "where", "of": d.meaning}),
                            _s([w("the"), noun, w("is"), lex.place[q], w("me")],
                               ["det", "noun", "verb", "place", "place"],
                               {"place": world.PLACE_NAMES[q]}), "where", d.depth))
        if want("colour"):
            out.append(Item(_s([w("what"), w("colour"), w("is")] + d.words,
                               ["q", "q", "verb"] + d.roles, {"ask": "colour", "of": d.meaning}),
                            _s([lex.colour[t.colour]], ["colour"],
                               {"colour": world.COLOUR_NAMES[t.colour]}), "colour", d.depth))
        if want("is_round"):
            yes = world.is_round(t)
            out.append(Item(_s([w("is")] + d.words + [lex.attribute["round"]],
                               ["verb"] + d.roles + ["shape"], {"ask": "is_round",
                                                                "of": d.meaning}),
                            _s([w("yes" if yes else "no")], ["truth"], {"truth": yes}),
                            "is_round", d.depth))
        # what is <predicate>: the noun's modifiers alone pick the thing out
        mods = [m for m in modifiers(w, r, t, rng, max(depth, 1))]
        if want("what") and mods:
            pred = mods[0]
            for m in mods[1:]:
                pred = join(w, "and", pred, m)
            if [o for o in things if pred.test(o)] == [t]:
                out.append(Item(_s([w("what"), w("is")] + pred.words, ["q", "verb"] + pred.roles,
                                   {"ask": "what", "of": pred.meaning}),
                                _s([w("the"), noun], ["det", "noun"],
                                   {"kind": world.KIND_NAMES[t.kind]}), "what", pred.depth))
        # word-order twins: "what is left of the X" against "what is the X left of"
        if want("order"):
            for o in things:
                if o is t:
                    continue
                for rel in ("left_of", "right_of"):
                    if not related(r, o, t, rel):
                        continue
                    left = [x for x in things if related(r, x, t, rel)]
                    right = [x for x in things if related(r, t, x, rel)]
                    if len(left) != 1 or len(right) != 1:
                        continue
                    key = f"{t.kind}:{rel}"
                    rw = relation_words(w, rel)
                    out.append(Item(_s([w("what"), w("is")] + rw + [w("the"), noun],
                                       ["q", "verb"] + ["relation"] * len(rw) + ["det", "noun"],
                                       {"ask": "what", "of": {"relation": rel, "ground": noun}}),
                                    _s([w("the"), lex.kind[left[0].kind]], ["det", "noun"]),
                                    "order", 1, pair=key))
                    out.append(Item(_s([w("what"), w("is"), w("the"), noun] + rw,
                                       ["q", "verb", "det", "noun"] + ["relation"] * len(rw),
                                       {"ask": "what", "of": {"figure": noun, "relation": rel}}),
                                    _s([w("the"), lex.kind[right[0].kind]], ["det", "noun"]),
                                    "order", 1, pair=key))
    if want("how_many"):  # how many <colour> things are there
        for c in sorted({t.colour for t in things}):
            n = sum(t.colour == c for t in things)
            out.append(Item(_s([w("how"), w("many"), lex.colour[c], w("thing"), w("plural"),
                                w("are"), w("there")],
                               ["q", "q", "colour", "noun", "plural", "verb", "there"],
                               {"ask": "count", "of": {"colour": world.COLOUR_NAMES[c]}}),
                            _s([w(NUMBERS[min(n, 5)])], ["number"], {"count": n}), "how_many", 1))
    if want("is_there"):
        present = {t.kind for t in things}
        for k in range(world.TRAINED):
            yes = k in present
            out.append(Item(_s([w("is"), w("there"), w("a"), lex.kind[k]],
                               ["verb", "there", "det", "noun"],
                               {"ask": "exists", "of": {"kind": world.KIND_NAMES[k]}}),
                            _s([w("yes" if yes else "no")], ["truth"], {"truth": yes}),
                            "is_there", 0))
    if want("not"):
        for c in sorted({t.colour for t in things}):
            rest = [t for t in things if t.colour != c]
            if len(rest) == 1:
                d = join(w, "not", attribute(w, "colour", c))
                out.append(Item(_s([w("what"), w("is")] + d.words, ["q", "verb"] + d.roles,
                                   {"ask": "what", "of": d.meaning}),
                                _s([w("the"), lex.kind[rest[0].kind]], ["det", "noun"]),
                                "not", d.depth))
    return out


def phrases_heard(item: Item) -> set[tuple[str, str]]:
    """The adjacent pairs of words a question holds: a test item counts as a combination never
    heard together where one of its pairs no practice question held."""
    words = item.question.words
    return {(a, b) for a, b in zip(words, words[1:], strict=False)}
