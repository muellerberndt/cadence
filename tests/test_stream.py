"""Owned state as a clamp: the stateful wiring, the echo, and a task that needs carried state."""

from __future__ import annotations

import dataclasses

import numpy as np

import cadence as cd


def test_stateful_wiring_has_a_context_source_range_into_the_hidden_owners() -> None:
    w, tie = cd.stateful(5, 2, 3, 4, 5, seed=0)
    assert w.n == 2 * 5 + 4 + 2 * 3 + 4 + 5
    ctx, hid = w.sets["context"], w.sets["hidden"]
    assert len(ctx) == len(hid) == 4 and len(tie) == w.edges
    degree = w.in_degree()
    assert degree[list(ctx)].sum() == 0  # context owners hear nothing
    assert all(degree[i] >= 4 for i in hid)  # every hidden owner hears every context owner
    engine = cd.Settlement(w, cd.learning_rule(dt=1.0))
    lay = engine.layout
    assert 1 in lay.sources()  # the context range is a source of the block transport
    assert cd.conformance(engine, list(w.sets["input"])[:2], steps=20)["max_abs_deviation"] < 1e-12


def test_echo_decays_toward_the_hidden_activation_and_enters_the_clamp() -> None:
    w, _ = cd.stateful(3, 1, 2, 3, 3, seed=1)
    engine = cd.Settlement(w, cd.learning_rule(dt=1.0))
    echo = cd.Echo(w, decay=0.5)
    drive = np.zeros((2, w.n))
    drive[:, 0] = 1.0
    state = engine.settle_batch(echo.clamp(drive), steps=30)
    echo.update(state)
    h = state.activation[:, list(w.sets["hidden"])]
    assert np.allclose(echo.trace, 0.5 * h)
    echo.update(state)
    assert np.allclose(echo.trace, 0.75 * h)
    clamped = echo.clamp(drive)
    assert np.allclose(clamped[:, list(w.sets["context"])], echo.trace)
    assert clamped[:, 0].sum() == 2.0  # the input clamp is untouched


def test_carried_state_learns_what_no_window_can_see() -> None:
    """Predict the symbol seen one input ago from a window of one: impossible without state."""
    rng = np.random.default_rng(0)
    v, streams, length = 4, 64, 60
    seq = rng.integers(0, v, (streams, length))
    w, tie = cd.stateful(v, 1, 4, 24, v, seed=0)
    cfg = cd.LearnerConfig(
        eta=2.0, beta=0.1, temperature=0.1, tolerance=3e-3, nudged_steps=12, free_steps=60
    )
    learner = cd.Learner(
        cd.Settlement(w, cd.learning_rule(dt=1.0)), w.sets["output"], cfg, tie_groups=tie
    )
    echo = cd.Echo(w, decay=0.0)  # the previous equilibrium, undiluted

    def run(learn: bool) -> float:
        echo.reset(streams)
        hits = total = 0
        for t in range(1, length):
            drive = np.zeros((streams, w.n))
            drive[np.arange(streams), seq[:, t]] = 1.0
            target = seq[:, t - 1]
            drive = echo.clamp(drive)
            if learn:
                learned, _ = learner.step(drive, target)
                free = learned.free
            else:
                free = learner.free(drive)
            if t > 5:
                hits += int(
                    (free.activation[:, learner.output_index].argmax(axis=1) == target).sum()
                )
                total += streams
            echo.update(free)
        return hits / total

    for epoch in range(6):
        learner.config = dataclasses.replace(cfg, eta=2.0 * 0.8**epoch)
        run(learn=True)
    learner.config = dataclasses.replace(cfg, tolerance=1e-4)
    assert run(learn=False) > 0.6  # chance is 0.25 and a window of one gives exactly chance
