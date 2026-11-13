"""Relations between things in words learned from questions answered (``cadence.words.relations``,
#327)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadence.space.map import Map
from cadence.words import Words
from cadence.words.relations import RELATION_GROUPS, Relations

QUARTERS = ("fobe", "lume", "kosa", "riza")  # nonce words: in front, left, behind, right
KINDS = ("dako", "mipu", "sefa", "lobi")


def test_vector_cells() -> None:
    part = Relations()
    ahead = part.code([0.0, 0.0], [1.0, 0.0], 0.0)
    assert set(ahead) == set(RELATION_GROUPS)
    assert int(np.argmax(ahead["direction"])) == 0  # straight ahead of where the parrot faces
    left = part.code([0.5, 0.5], [0.5, 1.5], 0.0)
    assert int(np.argmax(left["direction"])) == 2  # a quarter turn counter-clockwise: its left
    turned = part.code([0.5, 0.5], [0.5, 1.5], math.pi / 2)
    assert int(np.argmax(turned["direction"])) == 0  # the parrot turned to face it
    near, far = part.code([0, 0], [0.4, 0], 0.0), part.code([0, 0], [2.5, 0], 0.0)
    assert int(np.argmax(near["distance"])) < int(np.argmax(far["distance"]))
    assert all(0.0 < v.min() and v.max() <= 1.0 for v in (*ahead.values(), *far.values()))
    with pytest.raises(ValueError, match="direction cells"):
        Relations(directions=2)
    with pytest.raises(ValueError, match="nearest"):
        Relations(nearest=2.0, farthest=1.0)


def _brain(seed: int = 0) -> tuple[Words, Relations, np.random.Generator]:
    part = Relations()
    held = Map((-2.0, 2.0, -2.0, 2.0), (24, 24), units=len(KINDS), seed=seed)
    brain = Words(held, parts=[part], capacity=32, shares={"kind": 0.4, "place": 0.3})
    return brain, part, np.random.default_rng(seed)


def _room(brain: Words, rng: np.random.Generator) -> list[np.ndarray]:
    """Four things around the parrot, one a quarter, each well inside its quarter."""
    turn = float(rng.uniform(-math.pi, math.pi))
    places = []
    for k in range(4):
        a = turn + k * math.pi / 2 + float(rng.uniform(-0.5, 0.5))
        places.append(float(rng.uniform(0.7, 1.7)) * np.array([math.cos(a), math.sin(a)]))
    order = rng.permutation(4)
    brain.map.reset()
    brain.map.moment(seen=[(int(k), places[int(k)]) for k in order])
    brain.body.heading = turn
    return places


def test_a_relation_word_from_answered_questions_said_in_both_directions() -> None:
    brain, part, rng = _brain()
    for _ in range(12):  # the kinds, named
        places = _room(brain, rng)
        for k in range(4):
            brain.name(KINDS[k], places[k])
    part.name_self(brain, "you")
    for _ in range(16):  # "where is the X from you?", answered with its quarter's word
        _room(brain, rng)
        for k in range(4):
            brain.practice(f"where is the {KINDS[k]} from you", QUARTERS[k])
    for _ in range(4):  # "what is R of you?", answered with the thing, for two of the words
        _room(brain, rng)
        for k in range(2):
            brain.practice(f"what is {QUARTERS[k]} you", KINDS[k])
    places = _room(brain, rng)
    for k in range(4):
        assert brain.ask(f"where is the {KINDS[k]} from you") == [QUARTERS[k]]
    # the other direction: from the relation word to the thing, for the two words never asked
    # so as well
    for k in range(4):
        assert brain.ask(f"what is {QUARTERS[k]} you") == [KINDS[k]], k
    assert part.last["relation"] == [brain.lexicon.unit(QUARTERS[3])]
    # each word names the direction cells: its distance cells, which its lessons varied, fall
    # away
    for word in QUARTERS:
        heard = brain.lexicon.hear(word)
        groups = brain.lexicon.groups
        assert part._group_of(brain, heard) == "direction"
        assert heard[groups["direction"]].max() > heard[groups["distance"]].max()
    # a question with no relation word and one thing is left to the senses
    brain.ask(f"where is the {KINDS[0]}")
    assert part.last["relation"] == [] and "roles" not in part.last


def test_the_relation_groups_at_zero_say_no_relation() -> None:
    brain, part, rng = _brain(1)
    places = _room(brain, rng)
    for k in range(4):
        brain.name(KINDS[k], places[k])
    part.name_self(brain, "you")
    code = part.code(brain.body.at, places[0], brain.body.heading)
    assert part.describe(brain, code) == {"direction": None, "distance": None}


def test_a_new_relation_word_taught_through_known_words() -> None:
    """"What is riza you?" answered with the thing: the question's only new word takes the
    meaning the answer shows, and is said in the other direction."""
    brain, part, rng = _brain(2)
    for _ in range(12):
        places = _room(brain, rng)
        for k in range(4):
            brain.name(KINDS[k], places[k])
    part.name_self(brain, "you")
    for _ in range(16):  # three of the quarter words, from "where is the X from you?"
        _room(brain, rng)
        for k in range(3):
            brain.practice(f"where is the {KINDS[k]} from you", QUARTERS[k])
    for _ in range(4):  # the frame "what is R you?", on the words known
        _room(brain, rng)
        for k in range(3):
            brain.practice(f"what is {QUARTERS[k]} you", KINDS[k])
    assert brain.lexicon.unit(QUARTERS[3]) is None
    for _ in range(4):  # the fourth word, only through "what is riza you?"
        _room(brain, rng)
        brain.practice(f"what is {QUARTERS[3]} you", KINDS[3])
    _room(brain, rng)
    assert part._group_of(brain, brain.lexicon.hear(QUARTERS[3])) == "direction"
    assert brain.ask(f"what is {QUARTERS[3]} you") == [KINDS[3]]
    assert brain.ask(f"where is the {KINDS[3]} from you") == [QUARTERS[3]]
