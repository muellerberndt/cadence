"""Words for things: the lexicon learned from naming and the brain that answers about its room
(``cadence.words``, #326)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadence.space.map import Map
from cadence.words import PLACES, Body, Lexicon, Words


def _named(seed: int = 0, namings: int = 300) -> tuple[Lexicon, list[str], list[str]]:
    """A lexicon taught by namings in which a word names a kind or a colour of the thing
    attended, the other concept present by chance."""
    rng = np.random.default_rng(seed)
    lexicon = Lexicon({"kind": 4, "colour": 3, "none": 1})
    kinds, colours = ["dako", "mipu", "sefa", "lobi"], ["ruka", "beto", "gani"]
    for _ in range(namings):
        k, c = int(rng.integers(4)), int(rng.integers(3))
        word = kinds[k] if rng.random() < 0.6 else colours[c]
        lexicon.name(word, {("kind", k): 0.5, ("colour", c): 0.5})
    return lexicon, kinds, colours


def test_a_word_keeps_what_its_namings_share() -> None:
    lexicon, kinds, colours = _named()
    for k, word in enumerate(kinds):
        assert int(np.argmax(lexicon.hear(word))) == lexicon.index("kind", k)
        assert lexicon.say(("kind", k)) == word
    for c, word in enumerate(colours):
        assert int(np.argmax(lexicon.hear(word))) == lexicon.index("colour", c)
        assert lexicon.say(("colour", c)) == word
    assert lexicon.unit("dako") is not None and lexicon.unit("never") is None
    # a word never heard settles where silence does: it names nothing
    assert np.array_equal(lexicon.hear("never"), lexicon.hear([]))
    assert lexicon.hear("never").max() < 0.02
    assert lexicon.say(("none", 0)) is None  # a concept never named has no word


def test_three_namings_bind_a_new_word() -> None:
    lexicon, _, _ = _named(1)
    for _ in range(3):
        lexicon.name("tovez", {("kind", 3): 0.5, ("colour", 1): 0.3})
    heard = lexicon.hear("tovez")
    assert int(np.argmax(heard)) == lexicon.index("kind", 3) and heard.max() > 0.1


def test_the_lexicon_as_arrays() -> None:
    lexicon, kinds, _ = _named(2, namings=60)
    again = Lexicon.from_arrays(lexicon.settings, lexicon.arrays())
    assert again.forms == lexicon.forms and again.lessons == lexicon.lessons
    assert np.allclose(again.hear(kinds[0]), lexicon.hear(kinds[0]))
    assert all(np.array_equal(again.counts[k], lexicon.counts[k]) for k in lexicon.counts)
    lexicon.name(kinds[1], {("kind", 1): 0.5})  # the next lesson: the same share in both
    again.name(kinds[1], {("kind", 1): 0.5})
    assert np.allclose(again.hear(kinds[1]), lexicon.hear(kinds[1]))
    with pytest.raises(ValueError, match="letters"):
        lexicon.name("Dako!", {("kind", 0): 1.0})
    with pytest.raises(ValueError, match="no concept"):
        lexicon.index("kind", 9)


def test_the_body_places() -> None:
    body = Body((0.0, 0.0), heading=0.0)
    assert [PLACES[body.place(p)] for p in ([1, 0], [0, -1], [-1, 0], [0, 1])] == [
        "front", "right", "behind", "left"]
    body.heading = math.pi / 2
    assert PLACES[body.place([1, 0])] == "right"
    flags = body.cells(2, np.array([[0.0, -1.0], [0.0, 1.0]]))
    assert flags.tolist() == [True, False]


def test_a_brain_answers_about_its_room() -> None:
    held = Map((-2.0, 2.0, -2.0, 2.0), (24, 24), units=3, values={"colour": 2})
    brain = Words(held)
    kinds, colours, places = ["dako", "mipu", "sefa"], ["ruka", "beto"], [
        "fobe", "riza", "kosa", "lume"]
    rng = np.random.default_rng(0)

    def room() -> list[tuple[int, np.ndarray, int]]:
        """Three things at drawn places around the robot, each in a drawn colour."""
        turn = rng.uniform(0.0, 2 * math.pi)
        things = [(k, 1.2 * np.array([math.cos(turn + 2.1 * k), math.sin(turn + 2.1 * k)]),
                   int(rng.integers(2))) for k in range(3)]
        held.reset()
        held.moment(seen=[(k, p, {"colour": c}) for k, p, c in things])
        return things

    for _ in range(40):  # namings: a thing's kind, colour or place around the robot
        things = room()
        for _ in range(4):
            k, p, c = things[int(rng.integers(3))]
            brain.body.heading = float(rng.uniform(-math.pi, math.pi))
            u = rng.random()
            word = kinds[k] if u < 0.5 else (colours[c] if u < 0.75 else
                                              places[brain.body.place(p)])
            brain.name(word, p)
    brain.body.heading = 0.0
    for _ in range(4):  # practice: where a thing is, its colour, a word never named
        things = room()
        k, p, c = things[int(rng.integers(3))]
        brain.practice(f"where is {kinds[k]}", places[brain.body.place(p)])
        brain.practice(f"what colour is {kinds[k]}", colours[c])
        brain.practice("where is zuzo", "i do not know")
    things = room()
    for k, p, c in things:
        assert brain.ask(f"where is {kinds[k]}") == [places[brain.body.place(p)]]
        assert brain.ask(f"what colour is {kinds[k]}") == [colours[c]]
    assert brain.ask("where is novu") == ["i do not know"]  # a word never named
    plain = Words(Map((-1.0, 1.0, -1.0, 1.0)))  # a map with no values: kinds and places
    assert set(plain.senses.groups) == {"kind", "place"} and plain.answers == ["kind", "place"]


def test_pistachios_teach_what_a_question_asks_for() -> None:
    """Answers never told: a right answer earns a reward of 1, and the words of asking learn
    the read the brain tried; with the reward withheld no read is learned."""
    kinds, colours, places = ["dako", "mipu", "sefa"], ["ruka", "beto"], [
        "fobe", "riza", "kosa", "lume"]

    def raised(rewarded: bool) -> tuple[Words, list]:
        held = Map((-2.0, 2.0, -2.0, 2.0), (24, 24), units=3, values={"colour": 2})
        brain = Words(held, seed=1)
        rng = np.random.default_rng(1)

        def room() -> list:
            turn = rng.uniform(0.0, 2 * math.pi)
            things = [(k, 1.2 * np.array([math.cos(turn + 2.1 * k), math.sin(turn + 2.1 * k)]),
                       int(rng.integers(2))) for k in range(3)]
            held.reset()
            brain.look(seen=[(k, p, {"colour": c}) for k, p, c in things])
            return things

        for _ in range(40):
            things = room()
            for _ in range(4):
                k, p, c = things[int(rng.integers(3))]
                brain.body.heading = float(rng.uniform(-math.pi, math.pi))
                u = rng.random()
                brain.name(kinds[k] if u < 0.5 else colours[c] if u < 0.75 else
                           places[brain.body.place(p)], p)
        brain.body.heading = 0.0
        for _ in range(30):
            things = room()
            k, p, c = things[int(rng.integers(3))]
            for question, right in ((f"where is {kinds[k]}", places[brain.body.place(p)]),
                                    (f"what colour is {kinds[k]}", colours[c])):
                said = brain.ask(question, explore=True)
                if rewarded:
                    brain.reward(1.0 if said == [right] else 0.0)
        return brain, room()

    brain, things = raised(True)
    for k, p, c in things:
        assert brain.ask(f"where is {kinds[k]}") == [places[brain.body.place(p)]]
        assert brain.ask(f"what colour is {kinds[k]}") == [colours[c]]
    # a word that names a kind keeps no read: its reads settle as silence does
    reads = brain.lexicon.groups["read"]
    assert brain.lexicon.hear("dako")[reads].max() < brain.floor
    assert np.allclose(brain.lexicon.hear("dako")[reads], brain.lexicon.hear([])[reads],
                       atol=0.02)
    withheld, _ = raised(False)
    assert max(withheld.hear(["where", "is", "dako"]).reads.values()) < withheld.floor


def test_a_heard_form_with_no_letters() -> None:
    """The sound of a word takes a word neuron of its own beside the written form's, which the
    motor never says; one concept holds both."""
    lexicon, kinds, _ = _named(3, namings=120)
    unit = lexicon.enter("~7", written=False)
    assert not lexicon.written[unit] and not lexicon.letters[unit].any()
    for _ in range(3):
        lexicon.name(["~7"], {("kind", 2): 0.5})
    assert int(np.argmax(lexicon.hear(["~7"]))) == lexicon.index("kind", 2)
    assert lexicon.say(("kind", 2)) == kinds[2]  # the motor says the written form
    again = Lexicon.from_arrays(lexicon.settings, lexicon.arrays())
    assert not again.written[unit] and again.say(("kind", 2)) == kinds[2]
    with pytest.raises(ValueError, match="outside the letters"):
        lexicon.enter("zuzo", written=False)  # a form of letters is a written form


def test_the_voice_chooses_a_heard_word() -> None:
    """The parrot's voice replays a word's heard sound: the lexicon chooses which word."""
    lexicon, kinds, _ = _named(4, namings=120)
    unit = lexicon.enter("~3", written=False)
    for _ in range(3):
        lexicon.name(["~3"], {("kind", 1): 0.5})
    assert lexicon.choose(("kind", 1), among="heard") == unit
    assert lexicon.forms[lexicon.choose(("kind", 1))] == kinds[1] == lexicon.say(("kind", 1))
    assert lexicon.choose(("kind", 3), among="heard") is None  # no heard form names it
    with pytest.raises(ValueError, match="among"):
        lexicon.choose(("kind", 1), among="sung")


def test_a_pistachio_makes_a_mark_a_link() -> None:
    """Three factors: a word heard in focus leaves a mark that fades; a pistachio makes the marks
    of its moment lasting links, in both directions; a word whose pistachio is withheld keeps
    none."""
    lexicon = Lexicon({"kind": 3, "colour": 2})
    for _ in range(4):  # heard with no pistachio: the marks fade
        lexicon.mark("dako", {("kind", 0): 0.6, ("colour", 1): 0.4})
        for _ in range(12):
            lexicon.fade()
    assert not lexicon.marks and lexicon.hear("dako").max() < 0.02
    assert lexicon.say(("kind", 0)) is None
    for k, colour in ((0, 1), (0, 0), (0, 1)):  # heard, then a pistachio: a link
        lexicon.mark("dako", {("kind", k): 0.6, ("colour", colour): 0.4})
        assert lexicon.confirm(1.0) == 1
    lexicon.mark("mipu", {("kind", 1): 0.6, ("colour", 0): 0.4})  # its pistachio withheld
    for _ in range(12):
        lexicon.fade()
    assert int(np.argmax(lexicon.hear("dako"))) == lexicon.index("kind", 0)
    assert lexicon.say(("kind", 0)) == "dako" and lexicon.say(("kind", 1)) is None
    assert lexicon.hear("mipu").max() < 0.02
