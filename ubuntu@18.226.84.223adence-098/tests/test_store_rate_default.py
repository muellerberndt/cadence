"""A new brain's store averages its outcomes (``memory_rate=0.2``, the owner's ruling of 10 November
after #169's Amendment 3); ``memory_rate=1.0`` writes as 0.96 did."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd


def composed(**options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Brain.compose(4, 2, seed=7, backend="cpu", **options)


def test_a_new_brains_store_writes_a_fifth_of_the_way():
    brain = composed()
    assert brain.hippocampus.rate == 0.2 and brain.describe()["genes"]["memory_rate"] == 0.2
    here = np.array([[1.0, 0.0, 0.0, 0.0]])
    taken = brain.step(here)
    brain.step(here, reward=[1.0])
    assert np.isclose(brain.hippocampus.recall(here)[0, taken[0]], 0.28)  # C's .1 and .2 of the rest


def test_the_old_write_is_named_and_the_others_keep_their_rates():
    last = composed(memory_rate=1.0)
    here = np.array([[1.0, 0.0, 0.0, 0.0]])
    taken = last.step(here)
    last.step(here, reward=[1.0])
    assert np.isclose(last.hippocampus.recall(here)[0, taken[0]], 1.0)
    assert composed(memory_choice=True).hippocampus.rate == 0.5
    assert composed(recall=False).hippocampus.rate == 1.0
    assert composed(recall_founder=True).hippocampus.rate == 1.0
    assert "memory_rate" not in composed(episodic=False).describe()["genes"]


@pytest.mark.parametrize("rate", [0.2, 1.0])
def test_a_saved_brain_keeps_its_rate(rate, tmp_path):
    brain = composed(memory_rate=rate)
    assert cd.Brain.load(brain.save(tmp_path / "b")).hippocampus.rate == rate
