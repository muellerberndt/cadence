"""The compiled residual check agrees with the NumPy residual, term by term."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module

pytest.importorskip("numba")
pytestmark = pytest.mark.skipif(not brain_module._FUSED, reason="CADENCE_FUSED=0 reads the NumPy loop alone")


def three_ranges(seed: int, adaptation: cd.Adaptation | None = None) -> cd.NeuralGraph:
    """Input, hidden and output populations: three blocks, reciprocal hidden/output synapses."""
    rng = np.random.default_rng(seed)
    inputs, hidden, outputs = 6, 8, 4
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
    sign = rng.choice([-1.0, 1.0], size=len(pre)) * rng.uniform(0.2, 1.0, size=len(pre))
    connectome = cd.Connectome.from_synapses(
        n,
        pre=np.array(pre),
        post=np.array(post),
        sign=sign,
        populations={
            "input": range(0, inputs),
            "hidden": range(inputs, inputs + hidden),
            "output": range(inputs + hidden, n),
        },
    )
    model = cd.learning_neuron_model(dt=1.0).replace(adaptation=adaptation)
    return cd.NeuralGraph(connectome, model, bias=rng.normal(0.0, 0.3, n))


def reference(brain: cd.NeuralGraph, drive, state, **kw) -> np.ndarray:
    original = brain_module._FUSED
    brain_module._FUSED = False
    try:
        return brain.residual(drive, state, **kw)
    finally:
        brain_module._FUSED = original


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("adaptation", [None, cd.Adaptation(tau_steps=20.0, strength=0.4)])
def test_fused_residual_matches_numpy_with_nudges_and_masks(batch: int, adaptation) -> None:
    brain = three_ranges(3, adaptation)
    assert brain._blocks is not None and brain_module._FUSED
    n = brain.connectome.n
    rng = np.random.default_rng(batch)
    drive = rng.normal(0.0, 2.0, (batch, n))
    state = brain.settle_batch(drive, steps=7)
    out_index = np.array(brain.connectome.populations["output"])
    mask = np.zeros(n)
    mask[out_index] = 1.0
    groups = np.full(n, -1, dtype=np.int64)
    groups[out_index[:2]] = 0
    groups[out_index[2:]] = 1
    target = np.zeros((batch, n))
    target[:, out_index[0]] = 1.0
    target[:, out_index[3]] = 1.0
    plain = cd.Nudge(target, mask, 0.3, weight=rng.uniform(-1.0, 1.0, batch))
    softmax = cd.Nudge(
        target,
        mask,
        0.3,
        softmax_temperature=0.2,
        groups=groups,
        weight=rng.uniform(0.5, 1.5, batch),
    )
    ungrouped = cd.Nudge(target, mask, 0.3, softmax_temperature=0.5)
    shared_mask = np.ones(n)
    shared_mask[out_index[1]] = 0.0
    row_mask = np.ones((batch, n))
    row_mask[0, out_index[2]] = 0.0
    row_mask[-1, 2] = 0.0
    cases = [
        dict(),
        dict(nudge=plain),
        dict(nudge=softmax),
        dict(nudge=ungrouped),
        dict(mask=shared_mask),
        dict(mask=row_mask, nudge=softmax),
        dict(mask=shared_mask[None, :], nudge=plain),
    ]
    for kw in cases:
        fused = brain.residual(drive, state, **kw)
        expected = reference(brain, drive, state, **kw)
        assert fused.shape == (batch,)
        np.testing.assert_allclose(fused, expected, rtol=1e-12, atol=1e-12)
    # a masked state that has not decayed to rest carries the projection's error
    ablated = brain.settle_batch(drive, steps=3, mask=shared_mask)
    np.testing.assert_allclose(
        brain.residual(drive, state, mask=shared_mask),
        reference(brain, drive, state, mask=shared_mask),
    )
    np.testing.assert_allclose(
        brain.residual(drive, ablated, mask=shared_mask),
        reference(brain, drive, ablated, mask=shared_mask),
    )


def test_fused_residual_is_infinite_where_the_state_is_not_finite() -> None:
    brain = three_ranges(5)
    n = brain.connectome.n
    drive = np.ones((2, n))
    state = brain.settle_batch(drive, steps=4)
    v = np.array(state.v, copy=True)
    v[1, 0] = np.nan
    broken = cd.BrainState(v, state.activation, state.adaptation, state.steps)
    out = brain.residual(drive, broken)
    assert np.isfinite(out[0]) and np.isinf(out[1])
    np.testing.assert_allclose(out, reference(brain, drive, broken))


def test_equilibrate_reports_the_same_residual_through_the_fused_check() -> None:
    brain = three_ranges(7)
    n = brain.connectome.n
    drive = np.random.default_rng(1).normal(0.0, 1.0, (3, n))
    eq = brain.equilibrate(drive, budget=400, chunk=10, tolerance=1e-8)
    assert eq.converged.all()
    np.testing.assert_allclose(
        eq.residual, reference(brain, drive, eq.state), rtol=1e-12, atol=1e-14
    )
