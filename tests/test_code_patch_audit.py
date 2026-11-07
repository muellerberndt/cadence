"""Adversarial numerical and mutation contracts of the learned code's local rules."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def region() -> cd.LearnedCode:
    return cd.LearnedCode(4, 8, 3, cd.learning_neuron_model(), seed=2)


def snapshot(code: cd.LearnedCode) -> tuple[dict[str, np.ndarray], object, dict[str, int]]:
    return ({name: value.copy() for name, value in code.arrays().items()},
            code.graph, code.work.copy())


def unchanged(code: cd.LearnedCode, saved: tuple) -> None:
    tables, graph, work = saved
    for name, value in tables.items():
        np.testing.assert_array_equal(getattr(code, name), value)
    assert code.graph is graph and code.work == work


@pytest.mark.parametrize("bad", ["context_width", "activation_rows", "free_rows", "nan"])
def test_invalid_local_learning_rows_preserve_every_table_and_the_graph(bad):
    code = region()
    x, a, z, free = np.ones((2, 4)), np.full((2, 8), 0.5), np.ones((2, 3)), None
    if bad == "context_width":
        z = np.ones((2, 4))
    elif bad == "activation_rows":
        a = a[:1]
    elif bad == "free_rows":
        free = a[:1]
    else:
        z[0, 0] = np.nan
    before = snapshot(code)
    with pytest.raises(ValueError, match="finite rows|one row per event"):
        code.learn(x, a, z, free=free)
    unchanged(code, before)


def test_a_nonfinite_predictor_proposal_rolls_back_an_earlier_dictionary_update():
    code = region()
    code.weights[:] = 1e308  # finite tables, but this context's prediction overflows
    before = snapshot(code)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="predictor must remain finite"):
            code.learn(np.ones((2, 4)), np.full((2, 8), 0.5), np.ones((2, 3)))
    unchanged(code, before)


def test_finite_large_dictionary_proposals_keep_unit_columns_and_checkpoint_validity():
    code = region()
    with np.errstate(over="ignore", invalid="ignore"):
        code.learn(np.full((1, 4), 1e160), np.full((1, 8), 0.5), np.zeros((1, 3)))
    expected = np.full((4, 8), 0.5)  # the overwhelmingly positive, equal input coordinates
    np.testing.assert_allclose(code.dictionary, expected)
    np.testing.assert_allclose(np.linalg.norm(code.dictionary, axis=0), 1.0)
    restored = region()
    restored.take(**code.arrays())
    np.testing.assert_array_equal(restored.dictionary, code.dictionary)


def test_exactly_cancelled_column_keeps_its_existing_unit_direction():
    code = region()
    code.dictionary[:, 0] = np.array([1.0, 0.0, 0.0, 0.0])
    code.install()
    before = code.dictionary.copy()
    activation = np.zeros((8, 8))
    activation[:, 0] = 1.0
    # At rate 1/8 these eight identical residuals cancel unit zero's whole column.
    code.learn(np.zeros((8, 4)), activation, np.zeros((8, 3)))
    np.testing.assert_allclose(code.dictionary, before, atol=1e-16)
    np.testing.assert_allclose(np.linalg.norm(code.dictionary, axis=0), 1.0)
    assert code.work["updates"] == 1


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_finite_code_vectors_keep_their_direction_when_the_squared_norm_is_not_representable(scale):
    code = region()
    activation = np.zeros((2, code.units))
    activation[0, :2] = [scale, 2.0 * scale]
    activation[1] = -scale
    key = code.code_of(activation)
    expected = np.zeros_like(activation)
    expected[0, :2] = np.array([1.0, 2.0]) / np.sqrt(5.0)
    np.testing.assert_allclose(key, expected)


def test_global_step_bound_covers_a_steeper_negative_activation_branch():
    model = cd.NeuronModel(dt=1.0, leak=1.0)
    code = cd.LearnedCode(1, 8, 2, model, seed=2)
    v = np.full((1, 8), -1e-10)
    bound = code.rest_stability()
    assert bound >= code.stability(v)
    step = model.dt / 2**bound
    # Independent Jacobian including the standing boundary's restoring term.
    slopes = np.diag(model.slope_at(v)[0])
    coupling = model.gain * code.lateral() - code.top_down * np.eye(code.units)
    jacobian = np.eye(code.units) + step * (-np.eye(code.units) + coupling @ slopes)
    assert np.max(np.abs(np.linalg.eigvals(jacobian))) < 1.0


@pytest.mark.parametrize("bad", [-1, True, 1.5, 2000, 10**100])
def test_invalid_step_changes_leave_the_graph_and_step_consistent(bad):
    code = region()
    graph, halvings = code.graph, code.halvings
    with pytest.raises(ValueError, match="nonnegative integer|remain positive"):
        code.restep(bad)
    assert code.graph is graph and code.halvings == halvings


def test_an_answer_reports_all_code_phase_sweeps_including_the_failed_first_attempt():
    brain = cd.Brain.compose(1, 2, modules=(4,), learned_code=32, seed=2,
                             episodic=False, learning_free_steps=32, recall=False)
    code = brain.learned_code
    before = code.work["sweeps"]
    brain.act([[1.0]], greedy=True)
    report = brain.last_settlement
    assert code.work["capped"] >= 1  # the bound retries this genuinely oscillatory code
    assert report["qualified"]
    assert report["code_steps"] + report["context_free_steps"] == code.work["sweeps"] - before


def test_a_code_fallback_reports_total_work_without_rewriting_the_final_state(monkeypatch):
    brain = cd.Brain.compose(1, 2, modules=(4,), learned_code=32, seed=2,
                             episodic=False, learning_free_steps=32, recall=False)
    code = brain.learned_code
    settle = code.settle
    attempts = []

    def observed(*args, **kwargs):
        result = settle(*args, **kwargs)
        attempts.append(result)
        return result

    monkeypatch.setattr(code, "settle", observed)
    phase = brain._settle_code(np.array([[1.0]]), context=False, report=False)
    assert len(attempts) == 2 and not attempts[0].qualified.all() and phase.qualified.all()
    assert phase.steps == sum(attempt.steps for attempt in attempts) == code.work["sweeps"]
    assert phase.residual_checks == sum(attempt.residual_checks for attempt in attempts)
    assert phase.state is attempts[-1].state
    np.testing.assert_array_equal(phase.residual, attempts[-1].residual)
