"""The block transport: layouts, cached products, frozen ranges, and the block contrast."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.blocks import BlockTransport, block_contrast, layout


def test_layout_cuts_at_contiguous_sets_and_pairs_only_wired_ranges() -> None:
    w = cd.layered(6, 4, 3, density=1.0, seed=0)
    lay = layout(w)
    # input->hidden, hidden->output, output->hidden, and the (zero) lateral block among the outputs
    assert lay.ranges == 3 and lay.pairs == 4
    assert lay.size == 6 * 4 + 4 * 3 + 3 * 4 + 3 * 3
    assert lay.sources() == [0]
    flat = lay.flat(np.arange(1, w.edges + 1, dtype=float))
    assert np.count_nonzero(flat) == w.edges
    dense = np.zeros((w.n, w.n))
    dense[w.pre, w.post] = np.arange(1, w.edges + 1)
    for k, block in enumerate(lay.blocks(flat)):
        a0, a1, b0, b1 = lay.bounds(k)
        assert np.array_equal(block, dense[a0:a1, b0:b1])
    scattered = w.with_sets(odd=[1, 3, 5])  # not one contiguous run: ignored
    assert layout(scattered).ranges == 3
    bare = cd.Wiring(w.n, w.pre, w.post, w.count, w.sign)  # no sets: one block, the full matrix
    assert layout(bare).ranges == 1 and layout(bare).pairs == 1 and layout(bare).size == w.n**2
    assert layout(cd.Wiring.from_edges(3, pre=[], post=[])).pairs == 0


def test_block_transport_matches_the_full_matrix_and_reuses_still_ranges() -> None:
    w = cd.layered(5, 4, 3, density=1.0, seed=1)
    engine = cd.Settlement(w, cd.learning_rule(dt=1.0))
    lay = engine.layout
    transport = BlockTransport(lay, lay.flat(engine.weights))
    rng = np.random.default_rng(0)
    s = rng.random((3, w.n))
    assert np.allclose(transport.inbox(s), s @ engine.dense())
    s2 = s.copy()
    s2[:, 5:] = rng.random((3, w.n - 5))  # the inputs did not move
    first = [c is not None and c.copy() for c in transport.cache]
    out = transport.inbox(s2)
    assert np.allclose(out, s2 @ engine.dense())
    assert np.array_equal(transport.cache[0], first[0])  # the input block was reused


@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_fused_kernel_agrees_with_the_loop_with_frozen_inputs(dt: float) -> None:
    w = cd.layered(7, 5, 3, density=1.0, seed=2)
    rule = cd.learning_rule(dt=dt)
    engine = cd.Settlement(w, rule)
    drive = engine.clamp_levels(np.random.default_rng(3).random((4, w.n)) * 0.5)
    fused = engine.settle_batch(drive, steps=40)
    loop = engine.settle_batch(drive, steps=40, trajectory=True)  # the NumPy loop
    assert (
        np.array_equal(fused.activation, loop.activation)
        or np.abs(fused.activation - loop.activation).max() < 1e-15
    )
    out = list(w.sets["output"])
    mask = np.zeros(w.n)
    mask[out] = 1.0
    target = np.zeros((4, w.n))
    target[:, out] = 0.7
    nudge = cd.Nudge(target, mask, 0.1, softmax_temperature=0.2)
    a = engine.settle_batch(drive, steps=20, state=fused, nudge=nudge)
    b = engine.settle_batch(drive, steps=20, state=loop, nudge=nudge, trajectory=True)
    assert np.abs(a.activation - b.activation).max() < 1e-15
    assert cd.conformance(engine, drive[0], steps=30)["max_abs_deviation"] < 1e-12


def test_block_contrast_matches_the_gram_matrix() -> None:
    w = cd.layered(6, 5, 4, density=1.0, seed=4)
    lay = layout(w)
    rng = np.random.default_rng(5)
    s_plus, s_minus = rng.random((3, w.n)), rng.random((3, w.n))
    s_minus[:, :6] = s_plus[:, :6]  # clamped inputs are the same in both phases
    gram = (s_plus.T @ s_plus - s_minus.T @ s_minus)[w.pre, w.post]
    assert np.allclose(block_contrast(lay, s_plus, s_minus), gram, atol=1e-12)


def test_layout_survives_with_parameters_and_learning() -> None:
    w = cd.layered(8, 6, 3, density=1.0, seed=6)
    engine = cd.Settlement(w, cd.learning_rule(dt=1.0))
    changed = engine.with_parameters(bias=np.full(w.n, 0.01))
    assert changed.layout is engine.layout and changed.to_dict()["layout"]["blocks"] == 4
    learner = cd.Learner(engine, w.sets["output"], cd.LearnerConfig(eta=1.0, beta=0.1))
    drive = engine.clamp_levels(np.random.default_rng(7).random((5, w.n)) * 0.5)
    learner.step(drive, np.array([0, 1, 2, 0, 1]))
    assert learner.engine.layout is engine.layout


@pytest.mark.skipif("torch" not in cd.available_backends(), reason="torch not installed")
def test_torch_precision_option_and_block_transport_agree_with_cpu() -> None:
    w = cd.layered(9, 7, 3, density=1.0, seed=8)
    rule = cd.learning_rule(dt=1.0)
    drive = cd.Settlement(w, rule).clamp_levels(np.random.default_rng(9).random((3, w.n)) * 0.5)
    cpu = cd.Settlement(w, rule).settle_batch(drive, steps=30)
    single = cd.Settlement(w, rule, backend="torch", device="cpu", precision="float32")
    double = cd.Settlement(w, rule, backend="torch", device="cpu", precision="float64")
    assert np.abs(single.settle_batch(drive, steps=30).activation - cpu.activation).max() < 1e-5
    assert np.abs(double.settle_batch(drive, steps=30).activation - cpu.activation).max() < 1e-12
    assert single.with_parameters(bias=np.zeros(w.n)).precision == "float32"
    with pytest.raises(ValueError):
        cd.Settlement(w, rule, backend="torch", device="cpu", precision="half")
