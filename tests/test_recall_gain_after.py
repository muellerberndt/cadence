"""``recall_gain_after`` and ``recall_gain_next``: the value path released once the read has
formed, on the next ports."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def _brain(**options):
    options = {"recall_gain": 1.0, "recall_tokens": list(range(6)), **options}
    return cd.Brain.compose(12, 6, backend="cpu", recall=True, seed=3, **options)


def _lessons(brain, count, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        brain.reset()
        x = np.eye(12)[rng.integers(6, size=2)]
        brain.lesson(x, rng.integers(6, size=2))
        brain.act(x, greedy=True)


def test_the_gains_are_held_at_zero_and_released_on_the_next_ports():
    brain = _brain(recall_gain_after=3, recall_gain_next=10.0)
    assert not brain.recall.gains.any()
    _lessons(brain, 2)
    assert brain.recall.gain_lessons == 2 and not brain.recall.gains.any()
    _lessons(brain, 1, seed=1)
    assert brain.recall.gain_lessons == 3
    np.testing.assert_array_equal(brain.recall.gains, [0.0, 10.0])


def test_without_a_delay_the_next_ports_start_at_their_gain():
    np.testing.assert_array_equal(_brain(recall_gain_next=4.0).recall.gains, [0.0, 4.0])


def test_at_their_founders_the_gains_live_as_without_them():
    plain, named = _brain(), _brain(recall_gain_after=0, recall_gain_next=0.0)
    _lessons(plain, 4)
    _lessons(named, 4)
    np.testing.assert_array_equal(plain.recall.gains, named.recall.gains)
    assert "recall_gain_after" not in named.describe()["genes"]


def test_the_count_is_saved_and_the_release_continues_after_a_load(tmp_path):
    brain = _brain(recall_gain_after=3, recall_gain_next=10.0)
    _lessons(brain, 2)
    loaded = cd.Brain.load(brain.save(tmp_path / "delayed"))
    assert loaded.recall.gain_lessons == 2 and loaded.recall.gain_after == 3
    for b in (brain, loaded):
        _lessons(b, 1, seed=1)
    np.testing.assert_array_equal(brain.recall.gains, loaded.recall.gains)
    np.testing.assert_array_equal(loaded.recall.gains, [0.0, 10.0])


@pytest.mark.parametrize("options, match", [
    ({"recall_gain": 0.0, "recall_gain_after": 3}, "need gain"),
    ({"recall_gain_after": -1}, "nonnegative integer"),
    ({"recall_gain_after": 2.5}, "nonnegative integer"),
    ({"recall_gain_next": -1.0}, "gain_next"),
])
def test_bad_settings_are_refused(options, match):
    with pytest.raises(ValueError, match=match):
        _brain(**options)
