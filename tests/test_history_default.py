"""The default composition hears the moment before (``history_decays=(0.0,)``, the owner's ruling of 10
November): a new brain with the memory holds one heard block of decay 0 with its own carried gain, and
``history_decays=()`` composes the brain of 0.96.1."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd


def composed(**options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Brain.compose(6, 3, seed=0, backend="cpu", **options)


def test_a_new_brain_hears_the_moment_before():
    brain = composed()
    assert brain.history is not None and brain.history.decays == (0.0,)
    assert brain.describe()["genes"]["history_decays"] == [0.0]
    assert set(brain.carried_gains) == {"trace", "history_0", "read"}
    x = np.eye(6)[[2]] * 0.7
    brain.step(x)
    brain.step(np.eye(6)[[4]])
    np.testing.assert_array_equal(brain.history.trace, np.eye(6)[[4]])  # the moment before, as heard
    assert brain.history.trace.shape == (1, 6)


def test_no_history_composes_the_brain_without_it():
    plain = composed(history_decays=())
    assert plain.history is None and "history_0" not in plain.carried_gains
    assert "history" not in plain.connectome.populations
    heard = composed()
    extra = heard.parameters() - plain.parameters()
    synapses = len(heard.connectome.pre) - len(plain.connectome.pre)
    neurons = heard.connectome.n - plain.connectome.n
    assert extra == synapses + neurons + 1  # its synapses, its neurons' biases and its gain


@pytest.mark.parametrize("options", [{"recall": False}, {"recall_founder": True},
                                     {"history_decays": (0.3, 0.8)}],
                         ids=["no memory", "the founder word", "named decays"])
def test_the_default_history_is_the_composed_memorys_and_a_name_wins(options):
    brain = composed(**options)
    if "history_decays" in options:
        assert brain.history.decays == (0.3, 0.8)
    else:
        assert brain.history is None
