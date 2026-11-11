"""Direct tests for cadence.memory: the persistent matrix, the fading residual, and their gates."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.memory import SynapticMemory

KEY = np.array([[1.0, 0.0, 0.0]])
VALUE = np.array([[0.5, -0.25]])


def memory(**kwargs: object) -> SynapticMemory:
    return SynapticMemory(np.arange(3), np.arange(3, 5), **kwargs)  # type: ignore[arg-type]


def test_starts_with_empty_persistent_synapses() -> None:
    m = memory()
    assert m.consolidated.shape == (3, 2)
    assert not m.consolidated.any()
    assert m.writes == 0


def test_rejects_hebbian_rule_and_bad_consolidation() -> None:
    with pytest.raises(ValueError, match="delta"):
        memory(rule="hebb")
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="consolidation"):
            memory(consolidation=bad)


def test_one_observation_is_recalled_exactly_and_partly_consolidated() -> None:
    m = memory(consolidation=0.05)
    m.reset(1)
    m.observe(KEY, VALUE)
    np.testing.assert_allclose(m.recall(KEY), VALUE)
    np.testing.assert_allclose(m.consolidated[0], 0.05 * VALUE[0])
    assert m.writes == 1


def test_reset_keeps_persistent_weights_and_clear_erases_them() -> None:
    m = memory(consolidation=0.5)
    m.reset(1)
    m.observe(KEY, VALUE)
    before = m.consolidated.copy()
    m.reset(1)
    np.testing.assert_array_equal(m.consolidated, before)
    np.testing.assert_allclose(m.strength[0], before)  # the residual is gone, the matrix stays
    m.clear()
    assert not m.consolidated.any() and not m.strength.any()


def test_repetition_accumulates_in_the_slow_matrix() -> None:
    m = memory(consolidation=0.1)
    m.reset(1)
    slow = []
    for _ in range(20):
        m.observe(KEY, VALUE)
        slow.append(m.consolidated[0].copy())
    assert np.linalg.norm(slow[-1] - VALUE[0]) < np.linalg.norm(slow[0] - VALUE[0])


def test_salience_raises_the_slow_rate_up_to_one() -> None:
    plain, salient = memory(consolidation=0.1), memory(consolidation=0.1)
    plain.observe(KEY, VALUE)
    salient.observe(KEY, VALUE, salience=np.array([1e6]))
    np.testing.assert_allclose(salient.consolidated[0], VALUE[0])  # rate capped at one
    assert np.abs(salient.consolidated).sum() > np.abs(plain.consolidated).sum()


def test_masked_value_components_are_never_taught() -> None:
    m = memory(consolidation=1.0)
    m.observe(KEY, VALUE, value_mask=np.array([[True, False]]))
    assert m.consolidated[0, 0] == pytest.approx(0.5)
    assert m.consolidated[0, 1] == 0.0


def test_gated_or_empty_rows_do_not_write() -> None:
    m = memory()
    m.observe(KEY, VALUE, write=np.array([False]))
    m.observe(np.zeros((1, 3)), VALUE)
    assert m.writes == 0 and not m.consolidated.any()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"salience": np.array([-1.0])}, "salience"),
        ({"salience": np.array([1.0, 1.0])}, "salience"),
        ({"write": np.array([1])}, "write"),
        ({"value_mask": np.array([1, 0])}, "value_mask"),
    ],
)
def test_observe_rejects_malformed_gates(kwargs: dict[str, np.ndarray], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        memory().observe(KEY, VALUE, **kwargs)  # type: ignore[arg-type]


def test_to_dict_names_the_consolidating_kind() -> None:
    d = memory(consolidation=0.2).to_dict()
    assert d["kind"] == "consolidating"
    assert d["consolidation"] == 0.2
    assert d["persistent_parameters"] == 6
