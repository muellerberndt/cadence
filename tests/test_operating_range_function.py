"""Functional response/retention for the published optional initialization.

The historical <=0 activity-fraction assertion and strict xfail remain intact in
`test_audit_bias.py`. Signed activity is not a dead-neuron predicate. This assay
protects one declared teacher-assisted relation task, not default improvement,
all-builder cognition, actual-reward behavior or native movie-action transfer.
"""

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

import cadence as cd


def test_below_rest_signed_response_has_independent_nonzero_derivative():
    model = cd.learning_neuron_model()
    voltage = np.array([-2.0, -0.7, -0.1])
    rest = 1 / (1 + np.exp(model.slope * model.threshold))
    sigmoid = 1 / (1 + np.exp(-model.slope * (voltage - model.threshold)))
    expected = model.leak * (sigmoid - rest) / rest
    expected_slope = model.leak * model.slope * sigmoid * (1 - sigmoid) / rest
    np.testing.assert_allclose(model.activation(voltage), expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(model.slope_at(voltage), expected_slope, rtol=1e-13, atol=1e-15)
    assert np.all(expected < 0) and np.all(expected_slope > 0)
    epsilon = 1e-6
    numerical = (model.activation(voltage + epsilon) - model.activation(voltage - epsilon)) / (
        2 * epsilon
    )
    np.testing.assert_allclose(numerical, expected_slope, rtol=1e-8, atol=1e-11)
    # Exact silence belongs to the explicitly rectified control, not every negative value.
    np.testing.assert_array_equal(model.replace(leak=0).activation(voltage), 0)


def _same_arrays(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("resting_bias", [0.0, 0.5])
def test_optional_resting_bias_acquires_retains_and_continues_same_graph(
    resting_bias, tmp_path, monkeypatch
):
    benchmark_dir = Path(__file__).parents[1] / "benchmarks/acquisition"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    helpers = {name: import_module(name) for name in ("relations", "online_curriculum", "run")}
    for name, helper in helpers.items():
        assert Path(helper.__file__).resolve() == (benchmark_dir / f"{name}.py").resolve()
    relations = helpers["relations"]
    curriculum = helpers["online_curriculum"].curriculum
    independent_residual = helpers["run"].independent_residual
    brain = cd.Brain.compose(
        650, 36, modules=(32, 16), observers=(), lateral=0, seed=0, resting_bias=resting_bias
    )
    brain.learner.config = replace(
        brain.learner.config,
        eta=0.003,
        eta_bias=0.0003,
        normalize=0.99,
        normalize_floor=0.001,
        momentum=0.0,
        beta=0.1,
        centered=True,
        free_steps=4096,
        nudged_steps=4096,
        qualified=True,
        damping=3,
        tolerance=0.003,
        temperature=0.2,
        nudge="cross_entropy",
    )
    panels = {name: relations.load_panel(name) for name in ("train", "development")}
    before = brain.brain.bias.copy()
    uniform = np.random.default_rng(10620261004).integers(
        0, 36, len(panels["development"]["labels"])
    )
    uniform_correct = int((uniform == panels["development"]["labels"]).sum())
    accepted = queries = teacher_sweeps = query_sweeps = 0
    negative_response_seen = False

    def teach(model, inputs, labels):
        nonlocal accepted, teacher_sweeps, negative_response_seen
        drive = model.stimulus(inputs, memory=False)
        graph, wire = model.brain, model.connectome
        weights = (
            graph.neuron_model.gain * wire.count * graph.efficacy * np.exp(graph.log_gain[wire.pre])
        )
        bias = graph.bias.copy()
        states, report = model.learner.step(drive, labels)
        assert report["accepted"] == 1
        accepted += 1
        teacher_sweeps += report["total_row_sweeps"]
        for phase, sign in ((states.free, 0), (states.nudged, 1), (states.opposite, -1)):
            residual, cache = independent_residual(
                model, drive, phase, labels, sign, weights=weights, bias=bias
            )
            assert np.max(residual) <= 0.003 and np.max(cache) <= 0.003
        _, contrast = model.learner.contrast(states.free, states.nudged, states.opposite)
        members = np.concatenate([wire.populations[name] for name in ("module_0", "association")])
        negative_response_seen |= bool(
            np.any((states.free.activation[0, members] < 0) & (np.abs(contrast[members]) > 1e-9))
        )

    def read(model, panel):
        nonlocal queries, query_sweeps
        predicted = []
        qualified = type(model)._qualified

        def audited_qualified(self, drive, state=None, *, operation="answer"):
            result = qualified(self, drive, state, operation=operation)
            residual, cache = independent_residual(self, drive, result)
            assert np.max(residual) <= 0.003 and np.max(cache) <= 0.003
            return result

        # Audit the returned public solve; do not run another endpoint or change its answer.
        with monkeypatch.context() as checks:
            checks.setattr(type(model), "_qualified", audited_qualified)
            for row in panel["inputs"]:
                predicted.append(int(model.predict(row[None])[0]))
                report = model.last_settlement
                assert report["qualified"]
                queries += 1
                query_sweeps += report["steps"]
        predicted = np.asarray(predicted)
        credits = {
            int(family)
            for family in np.unique(panel["families"])
            if np.all(
                predicted[panel["families"] == family]
                == panel["labels"][panel["families"] == family]
            )
        }
        return {
            "predictions": predicted,
            "correct": int((predicted == panel["labels"]).sum()),
            "credits": credits,
        }

    stage_lessons = []
    final_panel = None
    for stage, cap, every in (("old4", 1024, 32), ("mixed", 4096, 96)):
        stage_panels = {
            name: {
                key: value[np.isin(panel["families"], relations.OLD_FAMILIES)]
                if stage == "old4"
                else value
                for key, value in panel.items()
            }
            for name, panel in panels.items()
        }
        initial_correct = read(brain, stage_panels["train"])["correct"]
        order = curriculum(stage_panels["train"], cap + 1, 1 if stage == "old4" else 2)
        for number, row in enumerate(order[:-1], 1):
            teach(
                brain,
                stage_panels["train"]["inputs"][[row]],
                stage_panels["train"]["labels"][[row]],
            )
            if number % every and number != cap:
                continue
            train, development = (
                read(brain, stage_panels[name]) for name in ("train", "development")
            )
            if stage == "old4":
                passed = train["correct"] == 16 and development["correct"] == 8
            else:
                passed = (
                    number >= 128
                    and len(train["credits"]) >= 18
                    and development["correct"] >= 36
                    and len(development["credits"] & set(relations.OLD_FAMILIES)) >= 3
                    and len(development["credits"] & set(relations.NEW_FAMILIES)) >= 15
                )
            if passed and train["correct"] > initial_correct:
                stage_lessons.append(number)
                break
        else:
            pytest.fail(f"{stage} failed its unchanged {cap}-lesson acquisition/retention bound")
        final_panel = stage_panels["train"]
    assert accepted == sum(stage_lessons)
    assert development["correct"] > uniform_correct
    if resting_bias == 0.0:
        assert negative_response_seen  # below-rest responses participate in actual local teaching
    assert teacher_sweeps > 0 and query_sweeps > 0 and queries > 0
    assert not np.array_equal(brain.brain.bias, before)
    assert brain.hippocampus.writes == 0
    saved = brain.save(tmp_path / "acquired.npz")
    restored = cd.Brain.load(saved)
    assert restored.resting_bias == resting_bias
    _same_arrays(saved, restored.save(tmp_path / "loaded.npz"))
    np.testing.assert_array_equal(
        read(brain, final_panel)["predictions"], read(restored, final_panel)["predictions"]
    )
    next_row = order[stage_lessons[-1]]
    for model in (brain, restored):
        teach(model, final_panel["inputs"][[next_row]], final_panel["labels"][[next_row]])
    _same_arrays(
        brain.save(tmp_path / "continued.npz"), restored.save(tmp_path / "loaded-continued.npz")
    )
    # The documented option is an initial plastic bias; load never reapplies that initializer.
    assert accepted == sum(stage_lessons) + 2
