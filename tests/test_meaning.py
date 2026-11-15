"""cadence.meaning: the born senses and the deep composition as genes (cadence#334)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadence import genes
from cadence.meaning import (
    HAND_SET,
    SPACE,
    Circle,
    Eyes,
    Forms,
    Line,
    Seen,
    compose,
    modules,
    observation,
    random_genome,
)


def seen(direction: float = 0.0, salient: bool = False) -> Seen:
    return Seen(hue=1.0, saturation=0.9, lightness=0.5, shape=np.full(6, 0.5), width=0.12,
                direction=direction, distance=1.0, salient=salient)


def test_a_circle_wraps_and_a_line_clips():
    circle = Circle(12)
    assert np.allclose(circle(0.0), circle(2.0 * math.pi))
    assert int(np.argmax(circle(math.pi / 2))) == 3
    line = Line(5, 0.0, 1.0)
    assert np.allclose(line(-3.0), line(0.0)) and int(np.argmax(line(1.0))) == 4


def test_a_scene_is_slots_of_tuned_inputs_and_the_order_is_kept():
    eyes = Eyes(slots=3)
    a, b = seen(0.0), seen(math.pi, salient=True)
    scene = eyes.scene([a, b])
    assert scene.shape == (eyes.width,)
    w = eyes.slot_width
    assert np.allclose(scene[:w], eyes.slot(a)) and np.allclose(scene[w:2 * w], eyes.slot(b))
    assert not scene[2 * w:].any()
    assert scene[2 * w - 1] == 1.0 and scene[w - 1] == 0.0  # salience is the slot's last input
    with pytest.raises(ValueError):
        eyes.scene([a] * 4)


def test_a_heard_form_has_one_code_per_founder_and_none_is_silence():
    forms = Forms(width=32, active=4, seed=7)
    code = forms("bado")
    assert np.allclose(code, forms("bado")) and np.isclose(np.linalg.norm(code), 1.0)
    assert not forms(None).any() and forms.heard_as(code) == "bado"
    assert not np.allclose(Forms(width=32, active=4, seed=8)("bado"), code)


def test_observation_is_the_scene_then_the_heard_code():
    eyes, forms = Eyes(slots=2), Forms(width=16, active=2, seed=0)
    x = observation(eyes, forms, [seen()], "kito")
    assert x.shape == (eyes.width + 16,) and np.allclose(x[eyes.width:], forms("kito"))


def test_the_hand_set_composition_is_deep_with_apical_heads_and_a_voice():
    assert modules(HAND_SET) == (128, 128, 128)
    brain = compose(HAND_SET, inputs=40, words=6, seed=0)
    layout = brain.describe()["layout"]["composition"]
    assert layout["modules"] == [128, 128, 128] and layout["apical"] == 0.25
    assert brain.recall is not None and brain.recall.heads == 2
    assert len(brain.motor_index) == 7  # six words and silence
    action = brain.step(np.random.default_rng(0).random((2, 40)))
    assert action.shape[0] == 2


def test_a_voice_and_a_gaze_are_two_slots():
    brain = compose({"depth": 1, "module_width": 16, "association": 16, "recall": 0},
                    inputs=10, words=4, gaze=8, seed=1)
    assert brain.step(np.zeros((1, 10))).shape == (1, 2)


def test_every_gene_of_the_space_mutates_to_a_genome_that_composes():
    rng = np.random.default_rng(3)
    mutate = genes(SPACE, rate=1.0)
    genome = dict(HAND_SET)
    for _ in range(3):
        genome = mutate(genome, rng)
        small = {**genome, "module_width": 16, "association": 16, "recall_capacity": 8}
        compose(small, inputs=12, words=3, seed=0)
    drawn = random_genome(rng)
    assert set(drawn) == set(SPACE)
    compose({**drawn, "module_width": 16, "association": 16}, inputs=12, words=3, seed=0)


def test_the_host_recalls_a_word_only_where_the_nearest_records_agree():
    """The page host's read of the word records is the chamber's: the nearest records agree and
    the nearest lies at ``match`` or above. A record at a cosine near 1 answers only with its
    neighbours' agreement (keys of one seen thing lie near .96 whatever was asked)."""
    from cadence.longterm import LongTermMemory
    from cadence.meaning.host import MeaningParrot

    width = 8
    host = MeaningParrot.__new__(MeaningParrot)
    host.width = width
    host.spec = {"records": {"agree": 3, "match": 0.85}}
    host.records = LongTermMemory(width=16, value_width=width, capacity=32, cosine=True,
                                  revise=0.9999, seed=0)
    thing = np.zeros(16)
    thing[:5] = 1.0  # what is seen of one thing: the long part of every key about it
    words = np.eye(width)

    def key(asked):
        out = thing.copy()
        out[8 + asked] = 1.0  # the last word heard: the short part
        return out

    for asked, said in ((0, 1), (1, 2), (2, 3)):  # three questions about it, three answers
        host.records.write(key(asked), words[said])
    cue = key(0)
    nearest = host.records.read(cue, heads=1, credit=False).first()
    held = np.asarray(host.records.keys[nearest], dtype=float)
    assert float(held @ cue) / (np.linalg.norm(held) * np.linalg.norm(cue)) > 0.95
    assert not host.recall(cue).any()  # the neighbours hold other words: nothing is recalled
    agreed = LongTermMemory(width=16, value_width=width, capacity=32, cosine=True,
                            revise=0.9999, seed=0)
    for asked in (0, 1, 2):  # the same keys, one word
        agreed.write(key(asked), words[4])
    host.records = agreed
    np.testing.assert_array_equal(host.recall(cue), words[4])

