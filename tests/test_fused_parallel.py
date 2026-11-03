"""The fused kernels on Numba's threads, the apical compartment in the kernel, and the compiled
per-synapse passes of a lesson: each against the arithmetic it must reproduce."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module
from cadence.blocks import block_contrast

numba = pytest.importorskip("numba")

from cadence import fused  # noqa: E402

THREADS = sorted({1, 2, 4, min(8, int(numba.config.NUMBA_NUM_THREADS))} & set(
    range(1, int(numba.config.NUMBA_NUM_THREADS) + 1)))


@contextmanager
def build(parallel: bool, threads: int = 1) -> Iterator[None]:
    """Run the fused kernels on one build at ``threads`` Numba threads."""
    chosen, threads_on, previous = fused._in_parallel, fused._threads, numba.get_num_threads()
    fused._in_parallel = lambda batch, neurons: parallel  # type: ignore[assignment]
    fused._threads = lambda: parallel  # type: ignore[assignment]
    numba.set_num_threads(threads)
    try:
        yield
    finally:
        fused._in_parallel = chosen  # type: ignore[assignment]
        fused._threads = threads_on  # type: ignore[assignment]
        numba.set_num_threads(previous)


@contextmanager
def numpy_loop() -> Iterator[None]:
    original = brain_module._FUSED
    brain_module._FUSED = False
    try:
        yield
    finally:
        brain_module._FUSED = original


def same(a: cd.BrainState, b: cd.BrainState) -> bool:
    return (
        a.steps == b.steps
        and np.array_equal(a.v, b.v)
        and np.array_equal(a.activation, b.activation)
        and np.array_equal(a.adaptation, b.adaptation)
        and np.array_equal(a.activity_change, b.activity_change)
    )


def graphs() -> Iterator[tuple[str, cd.Brain, cd.NeuralGraph]]:
    """A founder, the same with adaptation, and a deep brain with the apical compartment
    (with and without adaptation): every branch of both row kernels."""
    founder = cd.Brain.compose(10, 4, modules=(12, 8), seed=1)
    yield "founder", founder, founder.brain
    g = founder.brain
    adapted = cd.NeuralGraph(
        g.connectome, g.neuron_model.replace(adaptation=cd.Adaptation(tau_steps=7.0, strength=0.4)),
        efficacy=g.efficacy, bias=g.bias,
    )
    yield "adapted", founder, adapted
    deep = cd.Brain.compose(10, 4, modules=(12, 9, 7), seed=2, apical=0.25, observers=(3,))
    yield "apical", deep, deep.brain
    g = deep.brain
    yield "apical_adapted", deep, cd.NeuralGraph(
        g.connectome, g.neuron_model.replace(adaptation=cd.Adaptation(tau_steps=9.0, strength=0.3)),
        efficacy=g.efficacy, bias=g.bias, apical=g.apical,
    )


def settings(
    graph: cd.NeuralGraph, drive: np.ndarray, rng: np.random.Generator
) -> Iterator[dict[str, object]]:
    """Free and nudged settles: plain, weighted softmax over groups, an anchor, a held apical
    gain; no mask, a shared and a per-row mask; from rest and warm; stopping and fixed."""
    batch, n = drive.shape
    motor = np.array(graph.connectome.populations["motor"])
    target = np.zeros((batch, n))
    target[np.arange(batch), motor[rng.integers(0, len(motor), batch)]] = 1.0
    mask = np.zeros(n)
    mask[motor] = 1.0
    groups = np.full(n, -1)
    groups[motor] = np.arange(len(motor)) % 2
    anchor_gain = np.zeros(n)
    anchor_gain[motor] = 0.3
    with numpy_loop():
        free = graph.settle_batch(drive, steps=200, tolerance=1e-9)
    held = graph.apical_gain(drive, free)
    nudges: list[cd.Nudge | None] = [
        None,
        cd.Nudge(target, mask, 0.4, apical_gain=held),
        cd.Nudge(target, mask, 0.3, softmax_temperature=0.5, groups=groups,
                 weight=rng.normal(size=batch), apical_gain=held),
        cd.Nudge(target, mask, 0.3, softmax_temperature=0.5),
        cd.Nudge(target, mask, 0.2, anchor=rng.uniform(size=(batch, n)), anchor_gain=anchor_gain),
    ]
    shared = np.ones(n)
    shared[motor[0]] = 0.0
    shared[2] = 0.5
    masks = [None, shared, (rng.uniform(size=(batch, n)) > 0.1).astype(float)]
    for nudge, keep, (tolerance, state) in itertools.product(
        nudges, masks, ((1e-7, None), (None, free))
    ):
        yield {"steps": 40, "nudge": nudge, "mask": keep, "tolerance": tolerance, "state": state}


@pytest.mark.parametrize("batch", [1, 5])
def test_the_parallel_build_settles_and_checks_to_the_serial_bits(batch: int) -> None:
    rng = np.random.default_rng(batch)
    for _, brain, graph in graphs():
        drive = brain.stimulus(rng.normal(size=(batch, 10)))
        for kw in settings(graph, drive, rng):
            with build(parallel=False):
                serial = graph.settle_batch(drive, **kw)  # type: ignore[arg-type]
                check = graph.residual(drive, serial, nudge=kw["nudge"], mask=kw["mask"])  # type: ignore[arg-type]
            for threads in THREADS:
                with build(parallel=True, threads=threads):
                    rows = graph.settle_batch(drive, **kw)  # type: ignore[arg-type]
                    again = graph.residual(drive, serial, nudge=kw["nudge"], mask=kw["mask"])  # type: ignore[arg-type]
                assert same(rows, serial)
                np.testing.assert_array_equal(again, check)


def test_the_kernel_picks_the_serial_build_for_one_row_a_small_sweep_or_one_thread() -> None:
    previous = numba.get_num_threads()
    work = fused._PARALLEL_WORK
    try:
        assert not fused._in_parallel(1, 10 * work)
        assert not fused._in_parallel(2, work // 2 - 1)
        numba.set_num_threads(1)
        assert not fused._in_parallel(64, work)
        assert not fused._elements_in_parallel(10 * fused._PARALLEL_ELEMENTS)
        if numba.config.NUMBA_NUM_THREADS > 1:
            numba.set_num_threads(2)
            assert fused._in_parallel(2, work // 2) and fused._in_parallel(64, work)
            assert fused._elements_in_parallel(fused._PARALLEL_ELEMENTS)
            assert not fused._elements_in_parallel(fused._PARALLEL_ELEMENTS - 1)
    finally:
        numba.set_num_threads(previous)


@pytest.mark.parametrize("batch", [1, 6])
def test_the_fused_apical_compartment_is_the_numpy_loop(batch: int) -> None:
    rng = np.random.default_rng(10 + batch)
    for name, brain, graph in graphs():
        if graph.apical is None:
            continue
        assert graph._blocked and brain_module._FUSED
        drive = brain.stimulus(rng.normal(size=(batch, 10)))
        for kw in settings(graph, drive, rng):
            fused_state = graph.settle_batch(drive, **kw)  # type: ignore[arg-type]
            with numpy_loop():
                reference = graph.settle_batch(drive, **kw)  # type: ignore[arg-type]
            assert fused_state.steps == reference.steps, name
            for field in ("v", "activation", "adaptation"):
                np.testing.assert_allclose(
                    getattr(fused_state, field), getattr(reference, field), rtol=0, atol=1e-12
                )
            np.testing.assert_allclose(
                fused_state.activity_change, reference.activity_change, rtol=1e-12, atol=1e-12
            )
            check = graph.residual(drive, fused_state, nudge=kw["nudge"], mask=kw["mask"])  # type: ignore[arg-type]
            with numpy_loop():
                expected = graph.residual(drive, fused_state, nudge=kw["nudge"], mask=kw["mask"])  # type: ignore[arg-type]
            np.testing.assert_allclose(check, expected, rtol=1e-12, atol=1e-12)


def test_a_held_gain_of_another_shape_is_refused_by_the_fused_kernel() -> None:
    brain = cd.Brain.compose(10, 4, modules=(12, 9), seed=3, apical=0.25)
    graph = brain.brain
    drive = brain.stimulus(np.ones((3, 10)))
    free = graph.settle_batch(drive, steps=20)
    held = graph.apical_gain(drive, free)
    n = graph.connectome.n
    wrong = cd.Nudge(np.zeros(n), np.zeros(n), 0.1, apical_gain=held[:2])
    with pytest.raises(ValueError, match="apical_gain"):
        graph.settle_batch(drive, steps=2, state=free, nudge=wrong)
    with pytest.raises(ValueError, match="apical_gain"):
        graph.residual(drive, free, nudge=wrong)


def test_the_pool_rms_is_numpys_pairwise_sum_to_the_bit() -> None:
    rng = np.random.default_rng(0)
    frames, left = np.empty((3, 64), dtype=np.int64), np.empty(64)
    for count in [*range(1, 300), 511, 512, 1000, 1024, 4097, 70_001]:
        x = rng.normal(size=count + 9) * 10.0 ** rng.integers(-3, 4, size=count + 9)
        for lo in (0, 9):
            part = x[lo : lo + count]
            assert fused._square_sum(x, lo, count, frames, left) == np.add.reduce(part * part)


def test_the_compiled_lesson_passes_are_numpys_to_the_bit() -> None:
    rng = np.random.default_rng(1)
    size, edges = 5000, 3000
    index = rng.permutation(size)[:edges].astype(np.int64)
    values = rng.normal(size=edges)
    expected = np.zeros(size)
    expected[index] = values
    np.testing.assert_array_equal(fused.scatter(size, index, values), expected)
    np.testing.assert_array_equal(fused.gather(expected, index, 3.7), expected[index] / 3.7)
    group = rng.integers(0, 11, edges)
    group[group == 4] = 5  # an empty group divides by one
    power = np.bincount(group, weights=values * values, minlength=12)
    count = np.maximum(np.bincount(group, minlength=12), 1)
    np.testing.assert_array_equal(
        fused.relative_floor(values, group, 12, 0.1, 1e-12),
        values + 0.1 * np.sqrt(power / count)[group] + 1e-12,
    )
    # and on Numba's threads, over a pass long enough to take them
    big = fused._PARALLEL_ELEMENTS
    index = rng.permutation(big).astype(np.int64)
    values = rng.normal(size=big)
    group = rng.integers(0, 40, big)
    with build(parallel=True, threads=THREADS[-1]):
        assert fused._elements_in_parallel(big)
        scattered = fused.scatter(big, index, values)
        floor = fused.relative_floor(values, group, 40, 0.1, 1e-12)
    expected = np.zeros(big)
    expected[index] = values
    np.testing.assert_array_equal(scattered, expected)
    power = np.bincount(group, weights=values * values, minlength=40)
    count = np.maximum(np.bincount(group, minlength=40), 1)
    np.testing.assert_array_equal(floor, values + 0.1 * np.sqrt(power / count)[group] + 1e-12)


def test_layout_flat_and_block_contrast_keep_their_numpy_bits() -> None:
    brain = cd.Brain.compose(10, 4, modules=(12, 8), seed=4)
    graph = brain.brain
    layout = graph.layout
    rng = np.random.default_rng(2)
    weights = rng.normal(size=graph.connectome.synapses)
    s_plus, s_minus = rng.uniform(size=(5, graph.connectome.n)), rng.uniform(size=(5, graph.connectome.n))
    compiled_flat = layout.flat(weights)
    compiled_contrast = block_contrast(layout, s_plus, s_minus, 2.5)
    with numpy_loop():
        assert not fused.compiled()
        np.testing.assert_array_equal(compiled_flat, layout.flat(weights))
        np.testing.assert_array_equal(compiled_contrast, block_contrast(layout, s_plus, s_minus, 2.5))
        np.testing.assert_array_equal(
            block_contrast(layout, s_plus, s_minus) / 2.5, block_contrast(layout, s_plus, s_minus, 2.5)
        )


def test_the_apical_blocks_are_found_once_per_compartment_and_layout() -> None:
    brain = cd.Brain.compose(10, 4, modules=(12, 9, 7), seed=5, apical=0.25)
    graph = brain.brain
    block, mask = graph.layout.block_of_edge, graph.apical.synapses
    basal, top_down = brain_module._apical_blocks(graph.apical, graph.layout)
    assert basal == sorted(int(k) for k in np.unique(block[~mask]))
    assert top_down == sorted(int(k) for k in np.unique(block[mask]))
    rebuilt = graph.with_parameters(efficacy=graph.efficacy * 1.1)
    assert rebuilt._apical_parts()[3] is graph._apical_parts()[3]  # the same lists, not refound


@pytest.mark.parametrize("apical", [0.0, 0.25])
def test_lessons_learn_the_same_bits_on_one_thread_and_on_many(apical: float) -> None:
    def run(parallel: bool, threads: int) -> list[np.ndarray]:
        brain = cd.Brain.compose(
            12, 5, modules=(16, 12, 12), seed=6, apical=apical, teaching="deep",
            learning_nudged_steps=16, episodic=False,
        )
        rng = np.random.default_rng(7)
        with build(parallel, threads):
            for _ in range(4):
                brain.lesson(rng.uniform(size=(8, 12)), rng.integers(0, 5, 8))
        learner = brain.learner
        return [brain.brain.efficacy, brain.brain.bias, learner.second_moment, learner.velocity]

    serial = run(False, 1)
    for threads in THREADS:
        for a, b in zip(run(True, threads), serial, strict=True):
            np.testing.assert_array_equal(a, b)
