"""A neural graph's check and its gain read refuse what its settle refuses: a nudge that does not
fit the drive, a drive of another width, a drive whose rows are not the state's. Nothing is read
past an array or broadcast into rows of another row's drive."""

from __future__ import annotations

import numpy as np
import pytest
from places import PLACES, place

import cadence as cd
from cadence import brain as brain_module
from cadence.brain import Apical, Nudge

KERNELS = pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "numpy-loop"])


def kernels(monkeypatch, compiled: bool) -> None:
    """The compiled CPU kernels as this run has them, or the NumPy loop in their place."""
    if compiled and not brain_module._FUSED:
        pytest.skip("CADENCE_FUSED=0 reads the NumPy loop alone")
    monkeypatch.setattr(brain_module, "_FUSED", compiled)


def misfits(n: int, batch: int) -> dict[str, Nudge]:
    target = np.zeros(n)
    target[-1] = 1.0
    mask = np.zeros(n)
    mask[-2:] = 1.0
    return {
        "weight of fewer rows": Nudge(target, mask, 0.5, weight=np.ones(batch - 2)),
        "weight of more rows": Nudge(target, mask, 0.5, weight=np.ones(batch + 2)),
        "target of other rows": Nudge(np.tile(target, (2, 1)), mask, 0.5),
        "anchor of other rows": Nudge(
            target, mask, 0.5, anchor=np.zeros((2, n)), anchor_gain=np.ones(n)
        ),
        "a nudge of another graph": Nudge(np.array([1.0]), np.array([1.0]), 0.5),
    }


@KERNELS
@pytest.mark.parametrize("name", PLACES)
def test_a_check_refuses_every_nudge_its_settle_refuses(monkeypatch, compiled, name) -> None:
    kernels(monkeypatch, compiled)
    connectome = cd.layered(4, 6, 2, density=1.0, seed=1)
    graph = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=0.5), **place(name))
    n, batch = connectome.n, 4
    drive = np.random.default_rng(3).normal(size=(batch, n))
    state = graph.settle_batch(drive, steps=20)
    for label, nudge in misfits(n, batch).items():
        with pytest.raises(ValueError) as settled:
            graph.settle_batch(drive, steps=1, state=state, nudge=nudge)
        with pytest.raises(ValueError) as checked:
            graph.residual(drive, state, nudge=nudge)
        assert str(checked.value) == str(settled.value), label


@KERNELS
def test_a_float_vector_of_another_width_is_refused_not_read_as_neurons(
    monkeypatch, compiled
) -> None:
    kernels(monkeypatch, compiled)
    connectome = cd.layered(4, 4, 2, density=1.0, seed=1)
    graph = cd.NeuralGraph(connectome, cd.NeuronModel(stimulus_amplitude=3.0))
    n = connectome.n
    for vector in (
        np.eye(n - 2)[5],
        np.array([0.0, 1.0] * (n // 2 + 1)),
        np.array([0.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 2.0]),
    ):
        with pytest.raises(ValueError, match="drive must have one column per neuron"):
            graph.stimulus_vector(vector)
        with pytest.raises(ValueError, match="drive must have one column per neuron"):
            graph.settle(vector, steps=3)
    # what the forms the documentation names give is as it was
    expected = np.zeros(n)
    expected[[0, 5]] = 3.0
    np.testing.assert_array_equal(graph.stimulus_vector([0, 5]), expected)
    np.testing.assert_array_equal(graph.stimulus_vector(np.array([0, 5])), expected)
    np.testing.assert_array_equal(graph.stimulus_vector({0: 1.0, 5: 1.0}), expected)
    dense = np.linspace(0.0, 1.0, n)
    np.testing.assert_array_equal(graph.stimulus_vector(dense), dense)
    np.testing.assert_array_equal(graph.stimulus_vector([]), np.zeros(n))


def compartment_graph(adaptation: cd.Adaptation | None) -> cd.NeuralGraph:
    rng = np.random.default_rng(11)
    n = 8
    pre, post = np.nonzero(rng.random((n, n)) < 0.6)
    keep = pre != post
    connectome = cd.Connectome.from_synapses(
        n, pre=pre[keep], post=post[keep], populations={"low": range(0, 4), "high": range(4, 8)}
    )
    groups = np.where(np.arange(n) < 4, 0, 1)
    apical = Apical(rng.random(connectome.synapses) < 0.5, groups, fraction=0.8)
    model = cd.NeuronModel(dt=0.5, slope=1.0, threshold=0.0, gain=0.2, leak=0.1,
                           adaptation=adaptation)
    return cd.NeuralGraph(connectome, model, efficacy=rng.normal(size=connectome.synapses),
                          apical=apical)


@KERNELS
@pytest.mark.parametrize("adaptation", [None, cd.Adaptation(tau_steps=4.0, strength=0.7)])
def test_a_gain_read_refuses_a_drive_whose_rows_are_not_the_states(
    monkeypatch, compiled, adaptation
) -> None:
    kernels(monkeypatch, compiled)
    graph = compartment_graph(adaptation)
    drive = np.random.default_rng(5).normal(size=(3, graph.connectome.n))
    three = graph.settle_batch(drive, steps=10)
    one = graph.settle_batch(drive[:1], steps=10)
    for given, state in ((drive[:1], three), (drive[0], three), (drive, one), (drive[:2], three)):
        with pytest.raises(ValueError, match="state batch does not match the drive batch"):
            graph.residual(given, state)
        with pytest.raises(ValueError, match="state batch does not match the drive batch"):
            graph.apical_gain(given, state)
    with pytest.raises(ValueError, match="drive must have one column per neuron"):
        graph.apical_gain(drive[:, :-1], three)
    # a drive of the state's rows reads as it did, one row given as a vector included
    gains = graph.apical_gain(drive, three)
    assert gains.shape == (3, 2)
    np.testing.assert_array_equal(graph.apical_gain(drive[0], one), graph.apical_gain(drive[:1], one))
