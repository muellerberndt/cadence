"""Check the working trace against its independently written recurrence.

For ``focus = 0``, one update is ``trace' = decay * trace + (1 - decay) * h``,
``last' = h`` and ``cold' = false``, with ``h`` read before any write. Iterating
with a zero source gives ``decay ** k * trace``; with a constant source it gives
``h + decay ** k * (trace - h)``. Induction gives these identities and the
arbitrary-source weighted sum. For ``0 <= decay <= 1``, the update is a convex
combination, so its magnitude is at most ``max(abs(trace), abs(h))``; bounded
sources therefore preserve that bound. Rows evolve independently.

These checks compare NumPy execution of ``Trace.update``, ``reset``, ``keep``
and ``stimulate`` with the reference below at ``focus = 0`` and ``focus = 1``,
for several batch sizes, decays and random inputs. They check reset, row and
checkpoint boundaries, plus stimulation of the selected columns only. Focused
updates weight source motion before writing ``last``: reversing that order
incorrectly makes a warm row pure decay. A mutation harness checks that wrong
source rows, decay, update order, aliasing and reset/keep/stimulation semantics
are rejected.

The checks concern the trace, not settlement, learning, recall or behavior of
the brain reading it, and not the torch device path. They trust NumPy float64
arithmetic, use exact equality for equal operation order, and allow relative
slack ``1e-12`` for closed forms evaluated differently. This is executable
reference checking, not formal verification of floating-point Python.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

EPS = 1e-9  # the additive guard of the movement weight in ``Trace.update``
DECAYS = (0.0, 0.3, 0.5, 0.9, 0.999)
BATCHES = (1, 2, 5)
CLOSED_FORM = dict(rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# The independent recurrence reference


def reference_step(decay: float, trace: np.ndarray, h: np.ndarray) -> np.ndarray:
    """``decay * trace + (1 - decay) * h`` coordinatewise."""
    return decay * trace + (1.0 - decay) * h


def movement_weight(focus: float, h: np.ndarray, last: np.ndarray) -> np.ndarray:
    """movement since ``last`` over the row mean."""
    moved = np.abs(h - last)
    return (moved / (moved.mean(axis=1, keepdims=True) + EPS)) ** focus


def reference_update(
    decay: float,
    focus: float,
    trace: np.ndarray,
    last: np.ndarray,
    cold: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``update`` at ``focus = 0`` and ``focusedUpdate`` at ``focus > 0``; the inputs are not written."""
    hypotheses(decay, trace, last, h)
    if focus:
        weight = movement_weight(focus, h, last)
        weight[cold] = 1.0
        new_trace = decay * trace + (1.0 - decay) * weight * h
    else:
        new_trace = reference_step(decay, trace, h)
    return new_trace, h.copy(), np.zeros(len(h), dtype=bool)


def hypotheses(decay: float, *arrays: np.ndarray) -> None:
    """Assert the recurrence hypotheses on every compared input."""
    assert 0.0 <= decay < 1.0, decay
    for array in arrays:
        assert np.isfinite(array).all()


# ---------------------------------------------------------------------------
# A small real connectome and its states


def connectome() -> cd.Connectome:
    """Three inputs, three context, two embedding, three hidden and three output neurons."""
    w, _ = cd.stateful(3, 1, 2, 3, 3, seed=1)
    return w


def settled(w: cd.Connectome, batch: int, seed: int) -> cd.BrainState:
    brain = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0))
    drive = np.random.default_rng(seed).random((batch, w.n)) * 0.6
    return brain.settle_batch(drive, steps=20)


def synthetic(activation: np.ndarray) -> cd.BrainState:
    """A state whose activation is the given array, shared with the caller."""
    return cd.BrainState(
        v=activation.copy(), activation=activation, adaptation=np.zeros_like(activation), steps=1
    )


def source_of(trace: cd.Trace, state: cd.BrainState) -> np.ndarray:
    return np.array(np.atleast_2d(state.activation)[:, trace.hidden])


def same(trace: cd.Trace, expected: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """The runtime arrays equal the reference arrays exactly, dtype included."""
    for actual, wanted in zip((trace.trace, trace.last, trace.cold), expected, strict=True):
        assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
        assert np.array_equal(actual, wanted)


# ---------------------------------------------------------------------------
# The bridge checks; each raises AssertionError on a mismatch


def check_one_step_focus_zero() -> None:
    """``update_trace``, ``update_last``, ``update_cold`` on settled states, two moments each."""
    w = connectome()
    for batch in BATCHES:
        for decay in DECAYS:
            trace = cd.Trace(w, decay=decay)
            held = (np.zeros((0, 3)), np.zeros((0, 3)), np.ones(0, dtype=bool))
            for seed in (1, 2):
                state = settled(w, batch, seed)
                h = source_of(trace, state)
                if len(held[0]) != batch:
                    held = (np.zeros((batch, 3)), np.zeros((batch, 3)), np.ones(batch, dtype=bool))
                held = reference_update(decay, 0.0, *held, h)
                trace.update(state)
                same(trace, held)


def check_one_step_focus_one() -> None:
    """``focusedUpdate``: a cold first moment weighs one, a warm second moment reads ``last``."""
    w = connectome()
    for batch in BATCHES:
        for decay in DECAYS:
            trace = cd.Afterglow(w, decay=decay, focus=1.0, target="context")
            held = (np.zeros((batch, 3)), np.zeros((batch, 3)), np.ones(batch, dtype=bool))
            for seed in (3, 4, 5):
                state = settled(w, batch, seed)
                h = source_of(trace, state)
                held = reference_update(decay, 1.0, *held, h)
                trace.update(state)
                same(trace, held)


def _primed(w: cd.Connectome, decay: float, batch: int, seed: int) -> cd.Trace:
    """A trace after one moment of random activation: a nonzero, warm starting point."""
    trace = cd.Trace(w, decay=decay)
    trace.update(synthetic(np.random.default_rng(seed).normal(size=(batch, w.n))))
    return trace


def check_k_step_zero_source() -> None:
    """``zero_source_closed_form``: ``k`` silent moments leave ``decay ** k`` of the trace."""
    w = connectome()
    for batch in BATCHES:
        for decay in DECAYS:
            trace = _primed(w, decay, batch, 10)
            x0 = trace.trace.copy()
            hypotheses(decay, x0)
            for k in range(1, 13):
                trace.update(synthetic(np.zeros((batch, w.n))))
                np.testing.assert_allclose(trace.trace, decay**k * x0, **CLOSED_FORM)
            assert np.array_equal(trace.last, np.zeros((batch, 3)))


def check_k_step_constant_source() -> None:
    """``constant_source_closed_form``: the remainder from ``h`` decays geometrically."""
    w = connectome()
    for batch in BATCHES:
        for decay in DECAYS:
            trace = _primed(w, decay, batch, 11)
            x0 = trace.trace.copy()
            activation = np.random.default_rng(12).normal(size=(batch, w.n))
            h = activation[:, trace.hidden]
            hypotheses(decay, x0, h)
            for k in range(1, 13):
                trace.update(synthetic(activation))
                np.testing.assert_allclose(trace.trace, h + decay**k * (x0 - h), **CLOSED_FORM)
                np.testing.assert_allclose(trace.trace - h, decay**k * (x0 - h), **CLOSED_FORM)


def check_k_step_closed_form() -> None:
    """``evolve_closed_form`` over a declared horizon of twelve random moments, and
    ``abs_evolve_le``: the trace stays within the bound of its inputs."""
    w = connectome()
    rng = np.random.default_rng(13)
    for batch in BATCHES:
        for decay in DECAYS:
            trace = _primed(w, decay, batch, 14)
            x0 = trace.trace.copy()
            sources = []
            for k in range(1, 13):
                activation = rng.normal(size=(batch, w.n))
                sources.append(activation[:, trace.hidden])
                hypotheses(decay, x0, *sources)
                trace.update(synthetic(activation))
                expected = decay**k * x0 + (1.0 - decay) * sum(
                    decay**i * sources[k - 1 - i] for i in range(k)
                )
                np.testing.assert_allclose(trace.trace, expected, **CLOSED_FORM)
                bound = np.maximum(np.abs(x0), np.max(np.abs(sources), axis=0))
                assert (np.abs(trace.trace) <= bound * (1.0 + 1e-12)).all()


def check_one_step_bound() -> None:
    """``abs_step_le_max`` on random inputs, including negative ones."""
    rng = np.random.default_rng(15)
    for decay in DECAYS:
        trace, h = rng.normal(size=(5, 7)), rng.normal(size=(5, 7))
        hypotheses(decay, trace, h)
        bound = np.maximum(np.abs(trace), np.abs(h))
        assert (np.abs(reference_step(decay, trace, h)) <= bound * (1.0 + 1e-12)).all()


def check_row_independence() -> None:
    """``updateRows_row_independent``: row two of the update reads row two only."""
    w = connectome()
    rng = np.random.default_rng(16)
    for decay in DECAYS:
        first = rng.normal(size=(5, w.n))
        second = rng.normal(size=(5, w.n))
        second[2] = first[2]
        a, b = cd.Trace(w, decay=decay), cd.Trace(w, decay=decay)
        a.update(synthetic(first))
        b.update(synthetic(second))
        assert np.array_equal(a.trace[2], b.trace[2]) and np.array_equal(a.last[2], b.last[2])
        assert not np.array_equal(a.trace[1], b.trace[1])
        third = rng.normal(size=(5, w.n))
        fourth = rng.normal(size=(5, w.n))
        fourth[2] = third[2]
        a.update(synthetic(third))
        b.update(synthetic(fourth))
        assert np.array_equal(a.trace[2], b.trace[2]) and np.array_equal(a.last[2], b.last[2])


def check_reset_rows() -> None:
    """``reset_selected``, ``reset_unselected``, ``reset_all`` on warm, nonzero rows."""
    w = connectome()
    trace = _primed(w, 0.5, 5, 17)
    trace.update(synthetic(np.random.default_rng(18).normal(size=(5, w.n))))
    before = (trace.trace.copy(), trace.last.copy(), trace.cold.copy())
    assert not trace.cold.any() and (trace.trace != 0).all() and (trace.last != 0).all()
    trace.reset(5, rows=np.array([1, 3]))
    for r in (1, 3):
        assert np.array_equal(trace.trace[r], np.zeros(3))
        assert np.array_equal(trace.last[r], np.zeros(3))
        assert trace.cold[r]
    for r in (0, 2, 4):
        assert np.array_equal(trace.trace[r], before[0][r])
        assert np.array_equal(trace.last[r], before[1][r])
        assert trace.cold[r] == before[2][r]
    trace.reset(5)
    assert np.array_equal(trace.trace, np.zeros((5, 3)))
    assert np.array_equal(trace.last, np.zeros((5, 3)))
    assert trace.cold.all() and trace.cold.dtype == bool
    trace.reset(3, rows=np.array([0]))  # another batch size: everything starts cold
    assert trace.trace.shape == (3, 3) and trace.cold.all()


def check_keep_rows() -> None:
    """``keep_apply`` and ``keep_update``: the kept rows carry their arrays, and keeping then
    updating equals updating then keeping."""
    w = connectome()
    rows = np.array([0, 2, 4])
    for decay in DECAYS:
        trace = _primed(w, decay, 5, 19)
        before = (trace.trace.copy(), trace.last.copy(), trace.cold.copy())
        trace.keep(rows)
        same(trace, (before[0][rows], before[1][rows], before[2][rows]))
        activation = np.random.default_rng(20).normal(size=(5, w.n))
        a = _primed(w, decay, 5, 19)
        a.update(synthetic(activation))
        a.keep(rows)
        b = _primed(w, decay, 5, 19)
        b.keep(rows)
        b.update(synthetic(activation[rows]))
        same(a, (b.trace, b.last, b.cold))


def check_checkpoint_round_trip() -> None:
    """A generic brain's working memory survives save and load byte for byte and continues
    identically."""
    rng = np.random.default_rng(21)
    g = cd.Brain.build(6, 3, hidden=16, working_memory=True, seed=1)
    g.step(rng.random((4, 6)))
    g.step(rng.random((4, 6)), reward=rng.normal(size=4))
    assert g.working_memory is not None and (g.working_memory.trace != 0).any()
    with tempfile.TemporaryDirectory() as folder:
        h = cd.Brain.load(g.save(Path(folder) / "generic"))
    assert h.working_memory is not None
    for name in ("trace", "last", "cold"):
        saved, loaded = getattr(g.working_memory, name), getattr(h.working_memory, name)
        assert saved.dtype == loaded.dtype and saved.shape == loaded.shape
        assert saved.tobytes() == loaded.tobytes()
    assert h.working_memory.decay == g.working_memory.decay
    assert h.working_memory.focus == g.working_memory.focus
    activation = rng.normal(size=(4, g.connectome.n))
    g.working_memory.update(synthetic(activation))
    h.working_memory.update(synthetic(activation.copy()))
    same(h.working_memory, (g.working_memory.trace, g.working_memory.last, g.working_memory.cold))


def check_stimulate() -> None:
    """``stimulate`` writes amplitude times trace into the target columns of a copy and nothing
    else; another batch size reads zero."""
    w = connectome()
    rng = np.random.default_rng(22)
    trace = cd.Trace(w, decay=0.5, amplitude=2.5)
    trace.update(synthetic(rng.normal(size=(4, w.n))))
    drive = rng.normal(size=(4, w.n))
    original = drive.copy()
    out = trace.stimulate(drive)
    assert np.array_equal(drive, original) and not np.shares_memory(out, drive)
    assert np.array_equal(out[:, trace.glow], 2.5 * trace.trace)
    others = np.setdiff1d(np.arange(w.n), trace.glow)
    assert np.array_equal(out[:, others], drive[:, others])
    other = trace.stimulate(rng.normal(size=(3, w.n)))
    assert np.array_equal(other[:, trace.glow], np.zeros((3, 3)))


def check_last_is_the_source_read_before_the_write() -> None:
    """``update_last`` and ``focusedUpdate_twice``: ``last`` owns the source read at the call,
    whatever the caller writes into its state storage afterwards."""
    w = connectome()
    for batch in (1, 3):
        for focus in (0.0, 1.0):
            trace = cd.Trace(w, decay=0.5, focus=focus)
            activation = np.random.default_rng(23).normal(size=(batch, w.n))
            state = synthetic(activation)
            read = activation[:, trace.hidden].copy()
            trace.update(state)
            activation[:, trace.hidden] += 1.0  # the caller reuses its state storage
            assert np.array_equal(trace.last, read)
            held = (trace.trace.copy(), read, np.zeros(batch, dtype=bool))
            expected = reference_update(0.5, focus, *held, activation[:, trace.hidden].copy())
            trace.update(state)
            same(trace, expected)


CHECKS: dict[str, Callable[[], None]] = {
    "one_step_focus_zero": check_one_step_focus_zero,
    "one_step_focus_one": check_one_step_focus_one,
    "k_step_zero_source": check_k_step_zero_source,
    "k_step_constant_source": check_k_step_constant_source,
    "k_step_closed_form": check_k_step_closed_form,
    "one_step_bound": check_one_step_bound,
    "row_independence": check_row_independence,
    "reset_rows": check_reset_rows,
    "keep_rows": check_keep_rows,
    "checkpoint_round_trip": check_checkpoint_round_trip,
    "stimulate": check_stimulate,
    "last_is_the_source_read_before_the_write": check_last_is_the_source_read_before_the_write,
}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_runtime_satisfies_the_bridge_check(name: str) -> None:
    CHECKS[name]()


# ---------------------------------------------------------------------------
# The mutation harness


def _mutant_update(variant: str) -> Callable[[cd.Trace, cd.BrainState], None]:
    """``Trace.update`` with one deliberate defect; ``control`` is the runtime statement."""

    def update(self: cd.Trace, state: cd.BrainState) -> None:
        h = self._source_activation(state)
        if variant == "source_target_columns":
            h = np.ascontiguousarray(np.atleast_2d(state.activation)[:, self._glow_columns])
        elif variant == "source_shifted_row":
            h = np.roll(h, 1, axis=0)
        if len(self.trace) != len(h):
            self.reset(len(h))
        if variant == "order_last_before_weight":
            self.last = h.copy()
        if self.focus:
            moved = np.abs(h - self.last)
            weight = (moved / (moved.mean(axis=1, keepdims=True) + 1e-9)) ** self.focus
            if variant != "cold_rows_not_weighted_one":
                weight[self.cold] = 1.0
            if variant == "decay_without_complement":
                self.trace = self.decay * self.trace + weight * h
            elif variant == "decay_swapped":
                self.trace = (1.0 - self.decay) * self.trace + self.decay * weight * h
            elif variant == "decay_after_blend":
                self.trace = self.decay * (self.trace + (1.0 - self.decay) * weight * h)
            else:
                self.trace = self.decay * self.trace + (1.0 - self.decay) * weight * h
        else:
            if variant == "decay_without_complement":
                self.trace = self.decay * self.trace + h
            elif variant == "decay_swapped":
                self.trace = (1.0 - self.decay) * self.trace + self.decay * h
            elif variant == "decay_after_blend":
                self.trace = self.decay * (self.trace + (1.0 - self.decay) * h)
            else:
                self.trace = self.decay * self.trace + (1.0 - self.decay) * h
        if variant == "last_aliased":
            self.last = h
        elif variant != "order_last_before_weight":
            self.last = h.copy()
        if variant != "cold_not_cleared":
            self.cold[:] = False

    return update


def _mutant_reset(variant: str) -> Callable[..., None]:
    def reset(self: cd.Trace, batch: int, rows: np.ndarray | None = None) -> None:
        if rows is None or len(self.trace) != batch:
            self.trace = np.zeros((batch, len(self.hidden)))
            self.last = np.zeros((batch, len(self.hidden)))
            self.cold = np.ones(batch, dtype=bool)
        elif variant == "reset_every_row":
            self.trace[:] = 0.0
            self.last[:] = 0.0
            self.cold[:] = True
        else:
            self.trace[rows] = 0.0
            if variant != "reset_keeps_last":
                self.last[rows] = 0.0
            if variant != "reset_rows_stay_warm":
                self.cold[rows] = True

    return reset


def _mutant_keep(self: cd.Trace, rows: np.ndarray) -> None:
    self.trace, self.last = self.trace[rows], self.last[rows]


def _mutant_stimulate(variant: str) -> Callable[[cd.Trace, np.ndarray], np.ndarray]:
    def stimulate(self: cd.Trace, drive: np.ndarray) -> np.ndarray:
        out = np.array(drive, dtype=float)
        target = (
            self._hidden_columns if variant == "stimulate_source_columns" else self._glow_columns
        )
        scale = 1.0 if variant == "stimulate_without_amplitude" else self.amplitude
        out[:, target] = scale * self.trace if len(self.trace) == len(out) else 0.0
        return out

    return stimulate


MUTANTS: dict[str, tuple[str, Callable[..., object]]] = {
    "decay_without_complement": ("update", _mutant_update("decay_without_complement")),
    "decay_swapped": ("update", _mutant_update("decay_swapped")),
    "decay_after_blend": ("update", _mutant_update("decay_after_blend")),
    "source_target_columns": ("update", _mutant_update("source_target_columns")),
    "source_shifted_row": ("update", _mutant_update("source_shifted_row")),
    "order_last_before_weight": ("update", _mutant_update("order_last_before_weight")),
    "cold_rows_not_weighted_one": ("update", _mutant_update("cold_rows_not_weighted_one")),
    "cold_not_cleared": ("update", _mutant_update("cold_not_cleared")),
    "last_aliased": ("update", _mutant_update("last_aliased")),
    "reset_keeps_last": ("reset", _mutant_reset("reset_keeps_last")),
    "reset_rows_stay_warm": ("reset", _mutant_reset("reset_rows_stay_warm")),
    "reset_every_row": ("reset", _mutant_reset("reset_every_row")),
    "keep_without_cold": ("keep", _mutant_keep),
    "stimulate_source_columns": ("stimulate", _mutant_stimulate("stimulate_source_columns")),
    "stimulate_without_amplitude": ("stimulate", _mutant_stimulate("stimulate_without_amplitude")),
}


def failing_checks() -> list[str]:
    """The names of the bridge checks that fail under the current ``Trace`` methods."""
    failed = []
    for name, check in CHECKS.items():
        try:
            check()
        except (AssertionError, IndexError, ValueError):
            failed.append(name)
    return failed


def test_the_control_template_passes_every_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutant template with no defect is the runtime statement: no check fails."""
    monkeypatch.setattr(cd.Trace, "update", _mutant_update("control"))
    assert failing_checks() == []
    monkeypatch.setattr(cd.Trace, "reset", _mutant_reset("control"))
    monkeypatch.setattr(cd.Trace, "stimulate", _mutant_stimulate("control"))
    assert failing_checks() == []


def test_every_mutant_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each deliberate defect fails at least one bridge check; the table names the killers."""
    rows = []
    for name, (attribute, function) in MUTANTS.items():
        with monkeypatch.context() as patched:
            patched.setattr(cd.Trace, attribute, function)
            rows.append((name, attribute, failing_checks()))
    width = max(len(name) for name, _, _ in rows)
    print(f"\n{'mutant':<{width}}  method     killed by")
    for name, attribute, killers in rows:
        print(f"{name:<{width}}  {attribute:<9}  {', '.join(killers) or 'SURVIVED'}")
    survivors = [name for name, _, killers in rows if not killers]
    print(f"mutation score {len(rows) - len(survivors)}/{len(rows)}")
    assert not survivors, survivors
