"""The symbols chamber's world and readings, small."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chamber as ch  # noqa: E402
from world import KINDS, WORDS, a_thing  # noqa: E402


def test_the_trainer_hears_every_take_as_its_word():
    things, ear, words = ch.setting()
    rng = np.random.default_rng(0)
    assert all(words.heard(words.take(word, rng)) == word for word in WORDS for _ in range(3))
    assert words.heard(np.zeros((1, ear.width))) is None
    assert things.width == 56 and ear.width == 640


def test_a_named_thing_comes_back_from_its_name_and_not_without_records():
    things, ear, words = ch.setting()
    rng = np.random.default_rng(1)
    brain = ch.compose(0, things, ear)
    ch.listen(brain, words, rng, reps=1)
    ch.naming(brain, things, words, rng, rounds=2)
    back = brain.complete({"ear": words.take("banana", rng)})["eyes"]
    assert ch.kind_of(back, things) == "banana"
    brain.bound.store.values[:] = 0.0
    assert ch.kind_of(brain.complete({"ear": words.take("banana", rng)})["eyes"], things) is None


def test_a_reward_teaches_a_name_and_the_controls_run():
    things, ear, words = ch.setting()
    rng = np.random.default_rng(2)
    brain = ch.compose(0, things, ear)
    ch.listen(brain, words, rng, reps=1)
    items = [(a_thing(rng, "cup", "white"), None, "cup")]
    paid = ch.exchanges(brain, things, words, rng, items, paid=True)
    assert paid["exchanges"][0] is not None and paid["exchanges"][0] <= ch.BUDGET
    assert ch.table_control(items, rng)["exchanges"][0] is not None
    assert len(ch.delta_control(brain, things, words, items, rng)["exchanges"]) == 1
    assert set(KINDS) >= {"cup", "apple", "banana", "key"}
