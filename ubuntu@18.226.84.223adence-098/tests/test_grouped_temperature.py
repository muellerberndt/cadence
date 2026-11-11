"""Local nudges differentiate the actual temperature of each competing motor slot."""

import numpy as np
import pytest

import cadence as cd


def test_group_temperatures_and_fractional_masks_match_an_independent_score_gradient():
    activity = np.array([0.3, -0.2, 0.5, 0.7, 0.1, -0.1])
    groups = np.array([-1, 4, 4, 9, 9, 9])
    temperature = np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8])
    mask = np.array([0.0, 1.0, 1.0, 0.25, 0.25, 0.25])
    target = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    nudge = cd.Nudge(target, mask, 0.1, temperature, groups=groups)

    def log_policy(s):
        total = 0.0
        for members, chosen, heat in (([1, 2], 0, 0.2), ([3, 4, 5], 2, 0.8)):
            logits = s[members] / heat
            maximum = logits.max()
            total += logits[chosen] - maximum - np.log(np.exp(logits - maximum).sum())
        return total

    numeric = np.zeros(6)
    for index in range(6):
        perturbation = np.eye(6)[index] * 1e-6
        numeric[index] = (log_policy(activity + perturbation)
                          - log_policy(activity - perturbation)) / 2e-6
    np.testing.assert_allclose(nudge.drive(activity), 0.1 * 0.2 * numeric, atol=1e-11)
    temperature[:] = 7.0  # caller mutation cannot change the admitted solve
    np.testing.assert_array_equal(nudge.softmax_temperature, [0.2, 0.2, 0.2, 0.8, 0.8, 0.8])


@pytest.mark.parametrize("temperature", [
    [0.2, 0.4], [[0.2, 0.2, 0.2]], [0.2, 0.0, 0.2],
    [0.2, np.inf, 0.2], [0.2, np.nan, 0.2], [0.2, 0.2, 0.4],
])
def test_group_temperature_rejects_wrong_shape_nonpositive_or_mixed_group(temperature):
    with pytest.raises(ValueError, match="softmax_temperature"):
        cd.Nudge(np.ones(3), np.ones(3), 0.1, np.array(temperature))


@pytest.mark.parametrize("backend", ["cpu", "torch", "mlx"])
@pytest.mark.parametrize("cold", [0.2, float(np.nextafter(0.0, 1.0))])
def test_grouped_nudges_match_reference_settlement_and_residual(backend, cold):
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    connectome = cd.layered(3, 4, 5, density=1.0, seed=7)
    model = cd.learning_neuron_model(dt=1.0)
    reference = cd.NeuralGraph(connectome, model)
    options = {"device": "cpu", "precision": "float64"} if backend == "torch" else {}
    graph = cd.NeuralGraph(connectome, model, backend=backend, **options)
    outputs = np.asarray(connectome.populations["output"])
    groups = np.full(connectome.n, -1)
    groups[outputs[:2]], groups[outputs[2:]] = 4, 9
    temperatures = np.full(connectome.n, 0.8)
    temperatures[outputs[:2]] = cold
    mask = np.zeros(connectome.n)
    mask[outputs[:2]], mask[outputs[2:]] = 1.0, 0.25
    target = np.zeros((2, connectome.n))
    target[:, outputs[[0, 4]]] = 1.0
    nudge = cd.Nudge(target, mask, 0.001, temperatures, groups=groups)
    drive = np.random.default_rng(11).normal(size=(2, connectome.n))
    # A trajectory explicitly selects the independent NumPy stepping loop.
    expected = reference.settle_batch(drive, steps=24, nudge=nudge, trajectory=True)
    actual = graph.settle_batch(drive, steps=24, nudge=nudge)
    tolerance = 1e-6 if backend == "mlx" else 1e-12
    np.testing.assert_allclose(actual.activation, expected.activation, atol=tolerance, rtol=0)
    np.testing.assert_allclose(
        graph.residual(drive, actual, nudge=nudge),
        reference.residual(drive, expected, nudge=nudge), atol=tolerance, rtol=0,
    )
    assert np.isfinite(actual.activation).all()
