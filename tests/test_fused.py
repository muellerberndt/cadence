"""Direct tests for cadence.fused: every compiled kernel against the NumPy arithmetic it replaces."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module

pytest.importorskip("numba")

from cadence import fused  # noqa: E402


def dense_brain(seed: int, adaptation: cd.Adaptation | None = None) -> cd.NeuralGraph:
    """Input, hidden and output ranges with reciprocal hidden/output synapses: a blocked layout."""
    rng = np.random.default_rng(seed)
    inputs, hidden, outputs = 5, 7, 3
    pre, post = [], []
    for i in range(inputs):
        for h in range(hidden):
            pre.append(i)
            post.append(inputs + h)
    for h in range(hidden):
        for o in range(outputs):
            pre += [inputs + h, inputs + hidden + o]
            post += [inputs + hidden + o, inputs + h]
    n = inputs + hidden + outputs
    connectome = cd.Connectome.from_synapses(
        n,
        pre=np.array(pre),
        post=np.array(post),
        sign=rng.choice([-1.0, 1.0], size=len(pre)) * rng.uniform(0.2, 1.0, size=len(pre)),
        populations={
            "input": range(0, inputs),
            "hidden": range(inputs, inputs + hidden),
            "output": range(inputs + hidden, n),
        },
    )
    model = cd.learning_neuron_model(dt=1.0).replace(adaptation=adaptation)
    return cd.NeuralGraph(connectome, model, bias=rng.normal(0.0, 0.3, n))


def numpy_settle(brain: cd.NeuralGraph, drive: np.ndarray, **kw: object) -> cd.BrainState:
    original = brain_module._FUSED
    brain_module._FUSED = False
    try:
        return brain.settle_batch(drive, **kw)  # type: ignore[arg-type]
    finally:
        brain_module._FUSED = original


def test_available_reports_numba() -> None:
    assert fused.available()
    assert set(fused.__all__) == {"available", "fused_residual", "fused_settle"}


@pytest.mark.parametrize("adaptation", [None, cd.Adaptation(tau_steps=15.0, strength=0.3)])
@pytest.mark.parametrize("tolerance", [None, 1e-6])
def test_fused_settle_matches_the_numpy_loop(
    adaptation: cd.Adaptation | None, tolerance: float | None
) -> None:
    brain = dense_brain(2, adaptation)
    assert brain._blocks is not None and brain_module._FUSED
    n = brain.connectome.n
    drive = np.random.default_rng(0).normal(0.0, 1.5, (4, n))
    out = np.array(brain.connectome.populations["output"])
    mask = np.zeros(n)
    mask[out] = 1.0
    target = np.zeros((4, n))
    target[:, out[0]] = 1.0
    keep = np.ones(n)
    keep[out[1]] = 0.0
    for kw in (
        dict(),
        dict(nudge=cd.Nudge(target, mask, 0.4)),
        dict(mask=keep),
        dict(mask=keep, nudge=cd.Nudge(target, mask, 0.2, softmax_temperature=0.3)),
    ):
        compiled = brain.settle_batch(drive, steps=25, tolerance=tolerance, **kw)
        expected = numpy_settle(brain, drive, steps=25, tolerance=tolerance, **kw)
        assert compiled.steps == expected.steps
        np.testing.assert_allclose(compiled.v, expected.v, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(compiled.activation, expected.activation, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(compiled.adaptation, expected.adaptation, rtol=1e-12, atol=1e-12)


def test_fused_settle_from_rest_and_warm_agree_with_numpy() -> None:
    brain = dense_brain(4)
    n = brain.connectome.n
    drive = np.random.default_rng(1).normal(0.0, 1.0, (2, n))
    warm = brain.settle_batch(drive, steps=5)
    compiled = brain.settle_batch(drive, steps=10, state=warm)
    expected = numpy_settle(brain, drive, steps=10, state=warm)
    np.testing.assert_allclose(compiled.v, expected.v, rtol=1e-12, atol=1e-12)


def phases(seed: int, batch: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s_minus = rng.uniform(0.0, 1.0, (batch, n))
    return s_minus + rng.normal(0.0, 0.05, (batch, n)), s_minus


def edge_contrast(s_plus: np.ndarray, s_minus: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    a_plus, a_minus = s_plus[:, pre], s_minus[:, pre]
    b_plus, b_minus = s_plus[:, post], s_minus[:, post]
    return a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus


def test_contrast_mean_matches_numpy() -> None:
    n, batch, span = 9, 6, 0.4
    rng = np.random.default_rng(3)
    pre, post = rng.integers(0, n, 20), rng.integers(0, n, 20)
    s_plus, s_minus = phases(3, batch, n)
    edges, neurons = fused.contrast_mean(s_plus, s_minus, pre, post, span)
    np.testing.assert_allclose(edges, edge_contrast(s_plus, s_minus, pre, post).mean(0) / span)
    np.testing.assert_allclose(neurons, (s_plus - s_minus).mean(0) / span)


def test_trace_step_decays_accumulates_and_weights_by_dopamine() -> None:
    n, batch, edges, span, decay = 7, 3, 12, 0.5, 0.8
    rng = np.random.default_rng(5)
    pre, post = rng.integers(0, n, edges), rng.integers(0, n, edges)
    s_plus, s_minus = phases(5, batch, n)
    trace = rng.normal(0.0, 1.0, (batch, edges))
    trace_bias = rng.normal(0.0, 1.0, (batch, n))
    delta = rng.normal(0.0, 1.0, batch)
    expected_trace = decay * trace + edge_contrast(s_plus, s_minus, pre, post) / span
    expected_bias = decay * trace_bias + (s_plus - s_minus) / span
    step_scale, step_bias = fused.trace_step(
        trace, trace_bias, decay, s_plus, s_minus, pre, post, span, delta
    )
    np.testing.assert_allclose(trace, expected_trace)  # updated in place
    np.testing.assert_allclose(trace_bias, expected_bias)
    np.testing.assert_allclose(step_scale, (delta[:, None] * expected_trace).mean(0))
    np.testing.assert_allclose(step_bias, (delta[:, None] * expected_bias).mean(0))


def test_fused_residual_is_zero_only_at_equilibrium() -> None:
    brain = dense_brain(6)
    n = brain.connectome.n
    drive = np.random.default_rng(2).normal(0.0, 1.0, (2, n))
    eq = brain.equilibrate(drive, budget=400, chunk=10, tolerance=1e-9)
    assert eq.converged.all()
    early = brain.settle_batch(drive, steps=1)
    assert (brain.residual(drive, eq.state) < 1e-8).all()
    assert (brain.residual(drive, early) > 1e-6).all()


def test_a_settle_of_no_sweeps_publishes_the_activations_without_a_launch(monkeypatch) -> None:
    brain = dense_brain(7, cd.Adaptation(tau_steps=15.0, strength=0.3))
    n = brain.connectome.n
    drive = np.random.default_rng(3).normal(0.0, 1.5, (3, n))
    warm = brain.settle_batch(drive, steps=9)
    keep = np.ones(n)
    keep[2] = 0.0
    launched = []
    monkeypatch.setattr(fused, "_launch", lambda *args: launched.append(args))
    m = brain.neuron_model
    for kw in (dict(), dict(mask=keep), dict(state=warm), dict(state=warm, mask=keep)):
        still = brain.settle_batch(drive, steps=0, **kw)
        assert not launched and still.steps == 0
        source = warm if "state" in kw else cd.BrainState(np.zeros((3, n)), None, np.zeros((3, n)), 0)
        np.testing.assert_array_equal(still.v, source.v)
        np.testing.assert_array_equal(still.adaptation, source.adaptation)
        assert still.v is not source.v and still.adaptation is not source.adaptation
        expected = np.empty((3, n))
        for b in range(3):
            fused._activation(source.v[b], m.slope, m.threshold, m.rest_emission, m.leak, expected[b])
        if "mask" in kw:
            expected *= keep
        np.testing.assert_array_equal(still.activation, expected)
        np.testing.assert_array_equal(still.activity_change, np.zeros(3))


def test_a_held_gain_of_another_shape_is_refused_at_no_sweeps_too() -> None:
    brain = cd.Brain.compose(10, 4, modules=(12, 9), seed=3, apical=0.25)
    graph = brain.brain
    drive = brain.stimulus(np.ones((3, 10)))
    free = graph.settle_batch(drive, steps=20)
    held = graph.apical_gain(drive, free)
    wrong = cd.Nudge(np.zeros(graph.connectome.n), np.zeros(graph.connectome.n), 0.1,
                     apical_gain=held[:2])
    with pytest.raises(ValueError, match="apical_gain"):
        graph.settle_batch(drive, steps=0, state=free, nudge=wrong)


def test_a_settle_without_a_mask_is_the_settle_under_a_mask_of_ones() -> None:
    """The kernels read no mask where none is given; a mask of ones gives the same bits."""
    brain = dense_brain(8, cd.Adaptation(tau_steps=12.0, strength=0.2))
    n = brain.connectome.n
    drive = np.random.default_rng(4).normal(0.0, 1.5, (2, n))
    out = np.array(brain.connectome.populations["output"])
    mask = np.zeros(n)
    mask[out] = 1.0
    target = np.zeros((2, n))
    target[:, out[1]] = 1.0
    nudge = cd.Nudge(target, mask, 0.3, softmax_temperature=0.2)
    ones = np.ones(n)
    for kw in (dict(), dict(nudge=nudge), dict(tolerance=1e-5)):
        plain = brain.settle_batch(drive, steps=17, **kw)
        under = brain.settle_batch(drive, steps=17, mask=ones, **kw)
        assert plain.steps == under.steps
        for name in ("v", "activation", "adaptation", "activity_change"):
            np.testing.assert_array_equal(getattr(plain, name), getattr(under, name))
        np.testing.assert_array_equal(
            brain.residual(drive, plain, nudge=kw.get("nudge")),
            brain.residual(drive, under, nudge=kw.get("nudge"), mask=ones),
        )


def test_the_arrays_a_nudge_a_layout_and_no_nudge_give_the_kernels_are_made_once() -> None:
    brain = dense_brain(9)
    n = brain.connectome.n
    out = np.array(brain.connectome.populations["output"])
    mask = np.zeros(n)
    mask[out] = 1.0
    target = np.zeros((4, n))
    target[:, out[0]] = 1.0
    nudge = cd.Nudge(target, mask, 0.3, softmax_temperature=0.2, weight=np.ones(4))
    first = fused._nudge_arrays(nudge, 4, n)
    assert all(a is b for a, b in zip(first, fused._nudge_arrays(nudge, 4, n), strict=True)
               if isinstance(a, np.ndarray))
    with pytest.raises(ValueError):
        fused._nudge_arrays(nudge, 2, n)  # another batch is not the held one
    none = fused._nudge_arrays(None, 4, n)
    assert all(a is b for a, b in zip(none, fused._nudge_arrays(None, 4, n), strict=True)
               if isinstance(a, np.ndarray))
    assert fused._nudge_arrays(None, 3, n)[6].shape == (3,)
    layout, flat = brain.layout, brain._blocks
    plain = fused._compartment(None, layout, flat, None, 4, n)
    again = fused._compartment(None, layout, flat, None, 2, n)
    assert plain[0] is again[0] and plain[4] is again[4] and plain[1] is flat
    assert fused._fed_ranges(layout) is fused._fed_ranges(layout)
    # a settle and a check read them and leave them as they were made
    kept = [a.copy() for a in (*first, *none, *plain) if isinstance(a, np.ndarray)]
    drive = np.random.default_rng(5).normal(0.0, 1.0, (4, n))
    state = brain.settle_batch(drive, steps=15, nudge=nudge)
    brain.residual(drive, state, nudge=nudge)
    brain.residual(drive, brain.settle_batch(drive, steps=15))
    for before, after in zip(kept, (a for a in (*first, *none, *plain) if isinstance(a, np.ndarray)),
                             strict=True):
        np.testing.assert_array_equal(before, after)


def test_the_tied_step_gives_numpys_step_efficacy_and_weights_and_their_finiteness() -> None:
    rng = np.random.default_rng(6)
    count = 3000
    partner = np.full(count, -1, dtype=np.int64)
    pairs = rng.permutation(count)[:2000].reshape(2, 1000)
    partner[pairs[0]], partner[pairs[1]] = pairs[1], pairs[0]
    delta, efficacy = rng.normal(size=count), rng.normal(size=count) * 1.5
    gain_pre = rng.uniform(0.01, 0.05, size=count)
    step, scale, weights, finite = fused.tied_step(delta, partner, efficacy, gain_pre, 2.0)
    expected = delta.copy()
    tied = partner >= 0
    expected[tied] = 0.5 * (delta[tied] + delta[partner[tied]])
    np.testing.assert_array_equal(step, expected)
    np.testing.assert_array_equal(scale, np.clip(efficacy + expected, -2.0, 2.0))
    np.testing.assert_array_equal(weights, gain_pre * scale)
    assert finite == (True, True, True)
    inf_efficacy = efficacy.copy()
    inf_efficacy[7] = np.inf  # the bound keeps it finite
    _, bounded, _, finite = fused.tied_step(delta, partner, inf_efficacy, gain_pre, 2.0)
    assert bounded[7] == 2.0 and finite == (True, True, True)
    inf_gain = gain_pre.copy()
    inf_gain[9] = np.inf
    assert fused.tied_step(delta, partner, efficacy, inf_gain, 2.0)[3] == (True, True, False)
    nan_delta = delta.copy()
    nan_delta[5] = np.nan
    _, _, _, finite = fused.tied_step(nan_delta, partner, efficacy, gain_pre, 2.0)
    assert not finite[0]


def test_a_compiled_update_takes_the_passs_arrays_as_with_parameters_would() -> None:
    brain = cd.Brain.compose(6, 3, modules=(24,), seed=4)
    graph = brain.brain
    learner = brain.learner
    rng = np.random.default_rng(2)
    step = rng.normal(size=brain.connectome.synapses) * 0.01
    step_bias = rng.normal(size=brain.connectome.n) * 0.01
    tied = step.copy()
    paired = learner.reverse >= 0
    tied[paired] = 0.5 * (step[paired] + step[learner.reverse[paired]])
    cap = learner.config.scale_cap
    expected = graph.with_parameters(
        efficacy=np.clip(graph.efficacy + tied, -cap, cap), bias=graph.bias + step_bias,
    )
    learner.apply(step, step_bias)
    updated = learner.brain
    assert updated is not graph
    np.testing.assert_array_equal(updated.efficacy, expected.efficacy)
    np.testing.assert_array_equal(updated.bias, expected.bias)
    np.testing.assert_array_equal(updated.weights, expected.weights)
    np.testing.assert_array_equal(updated._blocks, expected._blocks)
    with pytest.raises(ValueError, match="one finite entry per synapse"):
        graph._with_update(graph.efficacy, graph.weights, graph.bias, (False, True))
    with pytest.raises(ValueError, match="effective synaptic weights must be finite"):
        graph._with_update(graph.efficacy, graph.weights, graph.bias, (True, False))
    with pytest.raises(ValueError, match="bias must have one finite entry"):
        graph._with_update(graph.efficacy, graph.weights, np.full(brain.connectome.n, np.nan),
                           (True, True))
