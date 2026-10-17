"""Contracts of the token-binding instrument."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from cadence import Brain

HERE = Path(__file__).resolve().parent


def _instrument():
    spec = importlib.util.spec_from_file_location("token_binding", HERE / "token_binding.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tb = _instrument()


def test_holdout_keeps_every_value_and_name_in_training() -> None:
    for seed in range(6):
        held = tb.holdout(seed)
        assert held.shape == (tb.TRAIN_NAMES, tb.VALUES)
        assert (held.sum(axis=0) == 1).all()  # each value held out for exactly one name
        assert (~held).any(axis=1).all() and (~held).any(axis=0).all()


def test_training_trials_avoid_held_pairs_and_panels_use_them() -> None:
    held = tb.holdout(3)
    rng = np.random.default_rng(0)
    trials = tb.draw(rng, 400, held)
    assert not held[trials["names"], trials["values"]].any()
    queried = tb.draw(rng, 200, held, queried_held=True)
    rows = np.arange(200)
    names = queried["names"][rows, queried["query"]]
    assert held[names, tb.answers(queried)].all()
    for i in range(200):
        assert len(set(queried["names"][i])) == queried["names"].shape[1]
    new = tb.draw(rng, 50, held, new_name=True)
    assert (new["names"][np.arange(50), new["query"]] == tb.NAMES - 1).all()


def test_the_query_repeats_the_name_token_of_its_fact() -> None:
    held = tb.holdout(0)
    trials = tb.draw(np.random.default_rng(1), 8, held)
    seq = tb.events(trials)
    query = seq[-1]
    rows = np.arange(8)
    asked = trials["names"][rows, trials["query"]]
    fact = np.stack([seq[trials["query"][i]][i] for i in rows])
    shared = (query > 0) & (fact > 0)
    assert (shared.sum(axis=1) == 1).all()
    assert shared[rows, tb.coordinate("name") + asked].all()
    assert (query[:, tb.coordinate("value") :] == 0).all()


def test_without_the_recall_region_the_brain_is_the_composed_one() -> None:
    built = tb.ContentBrain.build(4)
    composed = Brain.compose(tb.INPUTS, tb.VALUES, modules=(tb.WIDTH,), lateral=tb.LATERAL, seed=4)
    a, b = built.connectome, composed.connectome
    assert a.n == b.n and a.populations.keys() == b.populations.keys()
    np.testing.assert_array_equal(a.pre, b.pre)
    np.testing.assert_array_equal(a.post, b.post)
    np.testing.assert_array_equal(built.brain.efficacy, composed.brain.efficacy)
    np.testing.assert_array_equal(built.brain.bias, composed.brain.bias)
    x = tb.events(tb.draw(np.random.default_rng(2), 4, tb.holdout(4)))[0]
    np.testing.assert_array_equal(built.act(x, greedy=True), composed.act(x, greedy=True))


def test_the_recall_region_adds_neurons_after_the_composed_ones() -> None:
    plain = tb.ContentBrain.build(5)
    with_recall = tb.ContentBrain.build(5, content=dict(tb.CONTENT))
    n = plain.connectome.n
    assert with_recall.connectome.n == n + tb.INPUTS
    np.testing.assert_array_equal(
        with_recall.connectome.populations["recall"], np.arange(n, n + tb.INPUTS)
    )


def test_the_query_recalls_the_event_that_shares_its_name() -> None:
    brain = tb.ContentBrain.build(0, content=dict(tb.CONTENT))
    held = tb.holdout(0)
    trials = tb.draw(np.random.default_rng(3), 6, held)
    seq = tb.events(trials)
    brain.reset()
    for x in seq[:-1]:
        brain.act(x, greedy=True)
    assert brain.content is not None
    recalled = brain.content.recall(seq[-1])
    rows = np.arange(6)
    value = tb.coordinate("value") + tb.answers(trials)
    other = trials["values"][rows, 1 - trials["query"]]
    np.testing.assert_allclose(recalled[rows, value], 1.0)
    differs = other != tb.answers(trials)
    assert (np.abs(recalled[rows, tb.coordinate("value") + other])[differs] < 1e-12).all()
    asked = tb.coordinate("name") + trials["names"][rows, trials["query"]]
    np.testing.assert_allclose(recalled[rows, asked], 1.0)
    brain.reset()
    assert brain.content.recall(seq[-1]).shape == (6, tb.INPUTS)
    assert not np.any(brain.content.recall(seq[-1]))


def test_lessons_judged_by_the_act_teach_only_differing_rows() -> None:
    brain = tb.ContentBrain.build(1, content=dict(tb.CONTENT), learning_surprise=True)
    held = tb.holdout(1)
    trials = tb.draw(np.random.default_rng(4), tb.BATCH, held)
    work = tb.new_work()
    assert tb.graph_round(brain, trials, work)
    assert work["lessons"] == 1 and work["lesson_rows"] == tb.BATCH
    assert 0 <= work["rows_taught"] <= tb.BATCH


def test_readings_leave_the_life_unchanged() -> None:
    brain = tb.ContentBrain.build(2, content=dict(tb.CONTENT), learning_surprise=True)
    held = tb.holdout(2)
    rng = np.random.default_rng(5)
    for _ in range(3):
        tb.graph_round(brain, tb.draw(rng, tb.BATCH, held), tb.new_work())
    before = brain.brain.efficacy.copy()
    tb.read_graph(brain, 2, held)
    np.testing.assert_array_equal(brain.brain.efficacy, before)


def test_a_restated_name_asks_for_its_later_value() -> None:
    held = tb.holdout(2)
    trials = tb.draw(np.random.default_rng(7), 60, held, facts=3, restated=True)
    names, values = trials["names"], trials["values"]
    assert (names[:, 0] == names[:, 2]).all() and (names[:, 1] != names[:, 0]).all()
    assert (values[:, 0] != values[:, 2]).all() and (trials["query"] == 2).all()
    assert (tb.answers(trials) == values[:, 2]).all()
    assert not held[names, values].any()


def test_the_delta_store_favours_the_later_statement() -> None:
    held = tb.holdout(2)
    trials = tb.draw(np.random.default_rng(8), 12, held, facts=3, restated=True)
    seq = tb.events(trials)
    rows = np.arange(12)
    later = tb.coordinate("value") + trials["values"][:, 2]
    earlier = tb.coordinate("value") + trials["values"][:, 0]
    for rule in ("hebb", "delta"):
        brain = tb.ContentBrain.build(0, content=dict(tb.CONTENT, rule=rule))
        brain.reset()
        for x in seq[:-1]:
            brain.act(x, greedy=True)
        recalled = brain.content.recall(seq[-1])
        if rule == "hebb":
            np.testing.assert_allclose(recalled[rows, later], recalled[rows, earlier])
        else:
            # the delta residual also subtracts what a middle fact with the same value token
            # predicted, so the claim holds where the middle value differs from the later one
            clean = trials["values"][:, 1] != trials["values"][:, 2]
            assert clean.sum() >= 6
            assert (recalled[rows, later] > recalled[rows, earlier])[clean].all()


def test_the_gate_reads_all_three_declared_criteria() -> None:
    def reading(intact: float, erased: float, follows: float | None) -> dict:
        held = {"intact": intact, "content_erased": erased}
        if follows is not None:
            held["follows_donor"] = follows
        return {"panels": {"held_out": held}}

    assert tb.gate(reading(0.9, 0.1, 0.9))
    assert not tb.gate(reading(0.7, 0.1, 0.9))
    assert not tb.gate(reading(0.9, 0.3, 0.9))
    assert not tb.gate(reading(0.9, 0.1, 0.7))
    assert tb.gate({"panels": {"held_out": {"intact": 0.9, "trace_erased": 0.1}}})

