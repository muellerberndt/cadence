"""Direct tests for cadence.neuron: NeuronModel and Adaptation."""

import numpy as np
import pytest

from cadence.neuron import Adaptation, NeuronModel

# ------------------------------------------------------------------ Adaptation


def test_adaptation_stores_finite_params() -> None:
    a = Adaptation(tau_steps=10.0, strength=0.5)
    assert a.tau_steps == 10.0 and a.strength == 0.5


def test_adaptation_rejects_non_positive_tau() -> None:
    with pytest.raises(ValueError, match="tau_steps"):
        Adaptation(tau_steps=0.0, strength=0.5)


def test_adaptation_rejects_negative_strength() -> None:
    with pytest.raises(ValueError, match="strength"):
        Adaptation(tau_steps=5.0, strength=-0.1)


def test_adaptation_rejects_infinite_params() -> None:
    with pytest.raises(ValueError):
        Adaptation(tau_steps=float("inf"), strength=0.5)


# ------------------------------------------------------------------ NeuronModel construction


def test_default_neuron_model_is_finite() -> None:
    m = NeuronModel()
    assert np.isfinite(m.rest_emission)
    assert 0.0 <= m.rest_emission < 1.0


def test_rest_emission_is_the_raw_sigmoid_at_rest() -> None:
    assert NeuronModel(threshold=0.0).rest_emission == pytest.approx(0.5)
    m = NeuronModel(slope=4.0, threshold=1.5)
    assert m.rest_emission == pytest.approx(1.0 / (1.0 + np.exp(6.0)))


def test_neuron_model_rejects_saturating_slope_threshold() -> None:
    with pytest.raises(ValueError, match="saturate"):
        NeuronModel(slope=1e9, threshold=1e9)


# ------------------------------------------------------------------ activation


def test_activation_is_exactly_zero_at_rest_and_bounded_by_one() -> None:
    m = NeuronModel(threshold=1.0, slope=1.0)
    assert float(m.activation(np.array([0.0]))[0]) == 0.0
    high = m.activation(np.array([1e3]))
    assert float(high[0]) == pytest.approx(1.0)


def test_activation_above_threshold_is_positive() -> None:
    m = NeuronModel(threshold=0.0, slope=2.0, leak=0.0)
    v = np.array([2.0])
    assert float(m.activation(v)[0]) > 0.0


def test_activation_below_threshold_with_leak_is_negative() -> None:
    m = NeuronModel(threshold=0.0, slope=1.0, leak=0.1)
    v = np.array([-3.0])
    assert float(m.activation(v)[0]) < 0.0


def test_activation_below_threshold_no_leak_is_zero() -> None:
    m = NeuronModel(threshold=0.0, slope=1.0, leak=0.0)
    v = np.array([-5.0])
    assert float(m.activation(v)[0]) == 0.0


# ------------------------------------------------------------------ slope_at


def test_slope_at_matches_a_finite_difference_above_rest() -> None:
    m = NeuronModel(threshold=1.5, slope=4.0)
    v, h = np.array([1.2]), 1e-6
    numeric = (m.activation(v + h) - m.activation(v - h)) / (2 * h)
    np.testing.assert_allclose(m.slope_at(v), numeric, rtol=1e-5)


def test_slope_at_is_largest_at_threshold_and_zero_below_rest_without_leak() -> None:
    m = NeuronModel(threshold=1.5, slope=4.0, leak=0.0)
    at, far = m.slope_at(np.array([1.5])), m.slope_at(np.array([10.0]))
    assert float(at[0]) > float(far[0]) > 0.0
    assert float(m.slope_at(np.array([-1.0]))[0]) == 0.0
    assert float(m.replace(leak=0.1).slope_at(np.array([-1.0]))[0]) > 0.0


# ------------------------------------------------------------------ replace / serialisation


def test_replace_changes_only_named_fields() -> None:
    m = NeuronModel(slope=2.0, threshold=1.0)
    m2 = m.replace(slope=3.0)
    assert m2.slope == 3.0
    assert m2.threshold == m.threshold


def test_to_dict_round_trips_through_replace() -> None:
    m = NeuronModel(slope=1.5, threshold=0.5, leak=0.05)
    d = m.to_dict()
    assert d["slope"] == 1.5
    assert d["threshold"] == 0.5
    assert NeuronModel(**d) == m
    adapted = m.replace(adaptation=Adaptation(tau_steps=5.0, strength=0.2))
    assert adapted.to_dict()["adaptation"] == {"tau_steps": 5.0, "strength": 0.2}
