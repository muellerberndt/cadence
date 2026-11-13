"""Concepts learned from the senses and from naming (``cadence.words.concepts``, #326): the born
senses, a region's concepts and their lesson, and a parrot that learns where its teacher's colour
names change."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadence.space.map import Map
from cadence.words import Words
from cadence.words.concepts import Circle, Concepts, Features, LearnedWords, Region

TAU = 2.0 * math.pi
SHAPE = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))


def test_the_born_senses() -> None:
    circle = Circle(24)
    x = circle([0.0, math.pi])
    assert x.shape == (2, 24) and x[0, 0] == 1.0 and x[1, 12] == 1.0
    assert np.allclose(circle([TAU - 0.01]), circle([-0.01]))  # the circle closes
    assert (x[0] > 0).sum() <= 5  # an angle lights the inputs within two spacings
    shapes = Features(SHAPE, per=8)
    y = shapes([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    assert y.shape == (2, 512) and y.max() == 1.0
    # two things that differ in one feature alone light different inputs
    assert float(y[0] @ y[1]) < 0.05 * float(y[0] @ y[0])


def test_a_region_takes_a_concept_where_none_explains_and_learns_a_boundary() -> None:
    region = Region(Circle(24), 4, rate=0.3)
    assert region.concept(0.3) is None and region.explained(0.3) == (None, False)
    a = region.recruit(0.0)
    assert a == 0 and region.concept(0.0) == 0
    assert abs(float(region.settle(0.0)[0, 0]) - region.level) < 1e-6
    assert region.explained(0.1)[1] and not region.explained(math.pi)[1]
    b = region.recruit(1.6)
    assert region.concept(1.6) == b
    # a split concept wins its own reading
    c = region.recruit(0.05)
    assert region.concept(0.05) == c
    # lessons: the teacher's names change at 1.2 radians, between the two first concepts
    teacher = Region(Circle(24), 2, rate=0.3)
    teacher.recruit(0.2)
    teacher.recruit(2.2)
    rng = np.random.default_rng(0)
    for _ in range(300):
        h = float(rng.uniform(0.0, 2.6))
        teacher.lesson(h, 0 if h < 1.2 else 1)
    sweep = np.linspace(0.6, 1.8, 121)
    switch = sweep[int(np.argmax(teacher.winners(sweep) == 1))]
    assert abs(switch - 1.2) < math.radians(4.0)  # not the midpoint of the first readings


def _room(colours: int = 4) -> Map:
    return Map((-2.0, 2.0, -2.0, 2.0), (16, 16), units=6, values={"colour": colours})


def test_learned_concepts_are_the_senses_of_words() -> None:
    held = _room()
    concepts = Concepts(held, SHAPE, places=6)
    assert concepts.groups == {"kind": 6, "colour": 4, "place": 6}
    brain = LearnedWords(held, concepts)
    assert brain.senses is concepts and brain.body is concepts.body
    assert {g: brain.lexicon.groups[g].stop - brain.lexicon.groups[g].start
            for g in ("kind", "colour", "place")} == concepts.groups
    # the eyes report raw readings; the map takes each thing's kind concept, recruited from the
    # senses, and holds no colour concept before any colour word names one
    brain.look([(np.array([0.2, 0.2, 0.2]), np.array([1.0, 0.0]), 0.5),
                (np.array([0.8, 0.8, 0.8]), np.array([0.0, 1.0]), 2.0)])
    assert concepts.regions["kind"].used == 2 and concepts.regions["colour"].used == 0
    described = concepts.describe(np.array([1.0, 0.0]))
    assert described is not None and described["kind"].argmax() == 0
    assert described["colour"].max() == 0 and described["place"].max() == 0
    assert concepts.describe(np.array([-1.0, -1.0])) is None
    with pytest.raises(ValueError, match="colour values"):
        Concepts(Map((-2.0, 2.0, -2.0, 2.0), (16, 16), units=6), SHAPE)
    with pytest.raises(ValueError, match="own map"):
        LearnedWords(_room(), concepts)


COLOURS = ("ruka", "beto", "gani", "sefo")


def _life(seed: int, edge: float, namings: int = 320) -> tuple[LearnedWords, Concepts]:
    """A parrot named two kinds and four colours: its teacher's colour names change at 0,
    ``edge``, 3.0 and 4.6 radians of the hue; every thing in view, in front."""
    rng = np.random.default_rng(seed)
    held = _room()
    concepts = Concepts(held, SHAPE, places=4)
    brain = LearnedWords(held, concepts, seed=seed)
    shapes = {"dako": np.array([0.2, 0.3, 0.4]), "mipu": np.array([0.7, 0.6, 0.5])}
    edges = np.array([0.0, edge, 3.0, 4.6])
    for _ in range(namings):
        kind = str(rng.choice(list(shapes)))
        hue = float(rng.uniform(0.0, TAU))
        place = np.array([1.2, float(rng.uniform(-0.6, 0.6))])
        held.reset()
        brain.look([(shapes[kind] + rng.normal(0.0, 0.03, 3), place, hue)])
        colour = COLOURS[int(np.searchsorted(edges, hue, side="right")) - 1]
        brain.name(kind if rng.random() < 0.4 else colour, place)
    concepts.learning = False
    return brain, concepts


@pytest.mark.parametrize("edge", [1.0, 1.9])
def test_a_colour_concept_ends_where_its_teacher_says(edge: float) -> None:
    brain, concepts = _life(0, edge)
    assert all(concepts.group_of(brain.lexicon.unit(w)) == "colour" for w in COLOURS)
    assert concepts.group_of(brain.lexicon.unit("dako")) == "kind"
    assert concepts.regions["colour"].used == len(COLOURS)  # one concept a colour word
    sweep = np.linspace(edge - 0.6, edge + 0.6, 121)
    said = [brain.lexicon.say(("colour", concepts.percept(np.zeros(3), h)[1])) for h in sweep]
    switch = sweep[[w == "beto" for w in said].index(True)]
    assert abs(switch - edge) < math.radians(8.0)
    assert all(w == "ruka" for w, h in zip(said, sweep, strict=True) if h < edge - 0.15)
    assert all(w == "beto" for w, h in zip(said, sweep, strict=True) if h > edge + 0.15)
    # what the map holds is the concepts' own: a question about the thing is answered
    held = brain.map
    held.reset()
    place = np.array([1.2, 0.3])
    brain.look([(np.array([0.2, 0.3, 0.4]), place, edge / 2)])
    described = concepts.describe(place)
    assert described is not None
    assert brain.lexicon.say(("colour", int(described["colour"].argmax()))) == "ruka"
    assert brain.lexicon.say(("kind", int(described["kind"].argmax()))) == "dako"


def test_the_senses_alone_learn_no_boundary_of_the_teachers() -> None:
    held = _room(colours=8)
    concepts = Concepts(held, SHAPE, places=4, naming=False, vigilance=0.15)
    brain = Words(held, senses=concepts, body=concepts.body)
    rng = np.random.default_rng(1)
    for _ in range(40):
        hue = float(rng.uniform(0.0, TAU))
        held.moment(seen=concepts.perceive([(np.array([0.5, 0.5, 0.5]), np.array([1.0, 0.0]),
                                             hue)]))
    assert concepts.regions["colour"].used >= 3  # concepts from the hue alone
    assert concepts.regions["colour"].lessons == 0
    assert brain.lexicon.groups["colour"].stop - brain.lexicon.groups["colour"].start == 8
