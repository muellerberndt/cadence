"""Several things of one kind, answers that name several things, and how many: the part
``cadence.words.several.Several`` of a ``Words`` brain (#328)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadence.space.map import Map
from cadence.words import Lexicon, Words
from cadence.words.several import Several

KINDS, COLOURS = ["ball", "cup", "box"], ["red", "blue", "gold"]
PLACES = ["front", "right", "behind", "left"]
NUMBERS, TRUTHS, NOTHING = ["zero", "one", "two", "three"], ["no", "yes"], "i do not know"


def _room(brain: Words, rng: np.random.Generator) -> list[tuple[int, np.ndarray, int]]:
    """Two balls of two colours, and a cup and a box each held or not, seen at once."""
    kinds = [0, 0] + [k for k in (1, 2) if rng.random() < 0.7]
    turn = rng.uniform(0.0, 2 * math.pi)
    balls = [int(c) for c in rng.permutation(3)[:2]]
    things = [(k, 1.3 * np.array([math.cos(turn + i * 2 * math.pi / len(kinds)),
                                  math.sin(turn + i * 2 * math.pi / len(kinds))]),
               balls[i] if k == 0 else int(rng.integers(3))) for i, k in enumerate(kinds)]
    brain.map.reset()
    brain.map.moment(seen=[(k, p, {"colour": c}) for k, p, c in things])
    brain.body.heading = float(rng.uniform(-math.pi, math.pi))
    return things


def _questions(brain: Words, things: list[tuple[int, np.ndarray, int]],
               stage: int) -> list[tuple[str, str | list[str]]]:
    """The teacher's questions about a room and their answers, in the teacher's order: what and
    where, then how many as well, then whether there is one as well."""
    out: list[tuple[str, str | list[str]]] = []
    place = {i: PLACES[brain.body.place(p)] for i, (_, p, _) in enumerate(things)}
    for i, (k, _, c) in enumerate(things):
        out.append((f"where is the {COLOURS[c]} {KINDS[k]}", place[i]))
    for k in range(3):
        held = [i for i, t in enumerate(things) if t[0] == k]
        out.append((f"where is the {KINDS[k]}", sorted(place[i] for i in held) or NOTHING))
        if stage >= 2:
            out.append((f"how many {KINDS[k]} are there", NUMBERS[len(held)]))
        if stage >= 3:
            out.append((f"is there a {KINDS[k]}", TRUTHS[int(bool(held))]))
    for c in range(3):
        if stage >= 2:
            many = sum(t[2] == c for t in things)
            out.append((f"how many {COLOURS[c]} things are there", NUMBERS[many]))
        if stage >= 3:
            there = any(t[0] == 0 and t[2] == c for t in things)
            out.append((f"is there a {COLOURS[c]} ball", TRUTHS[int(there)]))
    out.append(("where is the zorp", NOTHING))
    return out


@pytest.fixture(scope="module")
def raised() -> Words:
    """A brain named and practised in rooms of two balls and a cup and a box."""
    rng = np.random.default_rng(3)
    brain = Words(Map((-2.0, 2.0, -2.0, 2.0), (24, 24), units=3, values={"colour": 3}),
                  parts=[Several()])
    for _ in range(40):  # naming: the parrot attends to a thing and hears one word
        things = _room(brain, rng)
        for _ in range(4):
            k, p, c = things[int(rng.integers(len(things)))]
            brain.body.heading = float(rng.uniform(-math.pi, math.pi))
            u = rng.random()
            word = KINDS[k] if u < 0.5 else COLOURS[c] if u < 0.75 else PLACES[
                brain.body.place(p)]
            brain.name(word, p)
    for stage in (1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3):
        things = _room(brain, rng)
        for question, answer in _questions(brain, things, stage):
            brain.practice(question, answer)
    return brain


def test_several_of_one_kind_told_apart_counted_and_found(raised: Words) -> None:
    brain = raised
    rng = np.random.default_rng(11)
    for _ in range(3):  # rooms never seen
        things = _room(brain, rng)
        for question, answer in _questions(brain, things, 3):
            wanted = [answer] if isinstance(answer, str) else answer
            assert sorted(brain.ask(question)) == sorted(wanted), question
    assert brain.ask("how many zorps are there") == [NOTHING]


def test_number_words_are_learned_from_answers_alone(raised: Words) -> None:
    lexicon = raised.lexicon
    for n, word in enumerate(NUMBERS[:3]):
        assert lexicon.say(("number", n)) == word
        assert int(np.argmax(lexicon.hear(word))) == lexicon.index("number", n)
    assert [lexicon.say(("truth", t)) for t in (0, 1)] == TRUTHS
    assert lexicon.say(("none", 0)) == NOTHING
    # no naming showed them: their synapses to a thing's concepts hold nothing
    for word in NUMBERS[:3] + TRUTHS:
        heard = lexicon.hear(word)
        assert max(heard[lexicon.groups[g]].max() for g in ("kind", "colour", "place")) < 0.02
    # the question's words carry the read: how many asks a count, is there asks a truth
    reads = raised.hear(raised._words("how many cup are there")).reads
    assert max(reads, key=lambda g: reads[g]) == "number"
    reads = raised.hear(raised._words("is there a red ball")).reads
    assert max(reads, key=lambda g: reads[g]) == "truth"
    part = raised.parts[0]
    assert part.name_words(raised, raised._words("is there a red ball")) == {
        "kind": 0, "colour": 0}


def test_with_the_lexicon_at_zero_it_says_nothing(raised: Words) -> None:
    arrays = raised.lexicon.arrays()
    arrays["efficacy"] = np.zeros_like(arrays["efficacy"])
    mute = Words(raised.map, parts=[Several()], body=raised.body,
                 lexicon=Lexicon.from_arrays(raised.lexicon.settings, arrays))
    assert mute.ask("how many ball are there") == [""]
    assert mute.ask("where is the red ball") == [""]


def test_the_part_adds_its_concepts_and_reads() -> None:
    held = Map((-1.0, 1.0, -1.0, 1.0), (8, 8), units=2, values={"colour": 2})
    brain = Words(held, parts=[Several(numbers=4)])
    assert brain.answers == ["kind", "colour", "place", "number", "truth"]
    sizes = {g: s.stop - s.start for g, s in brain.lexicon.groups.items()}
    assert sizes["number"] == 4 and sizes["truth"] == 2 and sizes["read"] == 5
    assert Several(numbers=4).together(9)["number"].tolist() == [0, 0, 0, 1]
    with pytest.raises(ValueError, match="counts 2 or more"):
        Several(numbers=1)
