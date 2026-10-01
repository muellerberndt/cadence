"""Conditional activity continuation without learning or hidden clamp retention."""

import json
import math

import pytest

from cadence import Brain, Cortex


def layout():
    cortex = Cortex(seed=17, tolerance=1e-7)
    signal = cortex.input("signal", shape=1)
    past = cortex.column("past", patches=2, inputs=signal)
    observer = cortex.observer("observer", patches=2, observes=past)
    actual = cortex.output("actual", shape=1, reads=past)
    cortex.output("alias", shape=1, reads=past)
    cortex.output("future", shape=2, reads=observer)
    return cortex.build(), actual, past, observer


@pytest.mark.parametrize(
    "clamps",
    [
        {},
        {"targets": {"actual": [0.7]}},
        {"interventions": {"observer": [-0.3, 0.4]}},
        {
            "targets": {"actual": [0.7]},
            "interventions": {"observer": [-0.3, 0.4]},
        },
        {
            "targets": {"actual": [0.7], "alias": [0.7]},
            "interventions": {"past": [0.7, -0.2]},
        },
    ],
)
def test_step_matches_pure_solve_and_changes_only_live_activity(clamps):
    brain, *_ = layout()
    inputs = {"signal": [0.25]}
    assert brain.observe(inputs, {"future": [0.2, -0.1]}, event_id=7)["accepted"]
    before = brain.snapshot()
    pure = brain.settle(inputs, **clamps)
    assert pure["qualified"] and brain.snapshot() == before
    continued = brain.step(inputs, **clamps)
    assert continued == {**pure, "accepted": True}
    assert brain.state == tuple(pure["state"])
    expected = json.loads(before)
    expected["state"] = list(brain.state)
    assert json.loads(brain.snapshot()) == expected
    restored = Brain.from_snapshot(brain.snapshot())
    assert restored.snapshot() == brain.snapshot()

    # Retaining diagnostic activity neither steals nor rewrites the learning ID.
    saved = brain.snapshot()
    retry = brain.observe(inputs, {"future": [0.2, -0.1]}, event_id=7, budget=0)
    assert retry["duplicate"]
    assert brain.snapshot() == saved
    following = brain.observe(inputs, {"future": [0.1, -0.2]})
    assert following["accepted"] and following["event_id"] == 8


def test_owned_handles_and_consistent_overlapping_clamps_are_accepted():
    brain, actual, past, _ = layout()
    result = brain.step(
        {"signal": [0.25]},
        targets={actual: [0.6]},
        interventions={past: [0.6, -0.4]},
    )
    assert result["accepted"]
    assert brain.state[:2] == (0.6, -0.4)


def test_refused_partial_clamp_preserves_entire_continuation():
    brain, *_ = layout()
    before = brain.snapshot()
    result = brain.step({"signal": [0.8]}, targets={"actual": [0.7]}, budget=0)
    assert not result["qualified"] and not result["accepted"]
    assert result["outputs"]["actual"] == (0.7,)  # Refused diagnostic only.
    assert brain.snapshot() == before


@pytest.mark.parametrize(
    "clamps",
    [
        {"targets": {"missing": [0.2]}},
        {"targets": {"actual": [math.nan]}},
        {"targets": {"actual": [1.01]}},
        {"targets": {"actual": [0.1, 0.2]}},
        {"targets": {"actual": [0.2], "alias": [0.3]}},
        {"interventions": {"past": [0.2]}},
        {"interventions": {"past": [0.2, math.inf]}},
        {"interventions": {"missing": [0.2]}},
        {
            "targets": {"actual": [0.2]},
            "interventions": {"past": [0.3, 0.4]},
        },
        {"targets": {"actual": [0.2]}, "budget": -1},
    ],
)
def test_invalid_or_conflicting_clamps_preserve_entire_continuation(clamps):
    brain, *_ = layout()
    before = brain.snapshot()
    with pytest.raises(ValueError):
        brain.step({"signal": [0.25]}, **clamps)
    assert brain.snapshot() == before


@pytest.mark.parametrize("kind", ["output", "population"])
def test_foreign_clamp_handles_are_rejected_without_mutation(kind):
    brain, *_ = layout()
    _, foreign_output, foreign_population, _ = layout()
    kwargs = (
        {"targets": {foreign_output: [0.2]}}
        if kind == "output"
        else {"interventions": {foreign_population: [0.2, 0.3]}}
    )
    before = brain.snapshot()
    with pytest.raises(ValueError, match="foreign"):
        brain.step({"signal": [0.25]}, **kwargs)
    assert brain.snapshot() == before


def test_next_unclamped_step_requalifies_and_releases_the_previous_goal():
    cortex = Cortex(seed=19, tolerance=1e-8)
    sensor = cortex.input("signal", shape=1)
    patch = cortex.column("prediction", patches=1, inputs=sensor)
    cortex.output("future", shape=1, reads=patch)
    brain = cortex.build()
    inputs = {"signal": [0.4]}

    # With no eligible free coordinate, this is boundary retention, not inference.
    held = brain.step(inputs, targets={"future": [0.8]}, budget=0)
    assert held["accepted"] and brain.state == (0.8,)
    saved = brain.snapshot()
    assert not brain.step(inputs, budget=0)["accepted"]
    assert brain.snapshot() == saved

    pure = Brain.from_snapshot(saved).settle(inputs)
    released = brain.step(inputs)
    assert released == {**pure, "accepted": True}
    expected = math.tanh(brain.weights[0] * 0.4) / (1 + brain.config["state_prior"])
    assert brain.state[0] == pytest.approx(expected, abs=1e-8)
    assert abs(brain.state[0] - 0.8) > 0.5
    assert brain.inspect()["admissions"] == 0
