"""Contracts of the fresh-binding instrument (``benchmarks/relevance/bindings.py``): the
encoding is the selection instrument's, facts are drawn fresh from the training pairs, the
held-out split keeps every single value in training, the answer is the queried fact, the act
rule teaches the rows the brain's own act got wrong, and the readings fork the trained brain
and reach its working trace alone."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "benchmarks" / "relevance"
sys.path.insert(0, str(HERE))

import bindings as bd  # noqa: E402
import selective_recall as sr  # noqa: E402

try:
    import cadence  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    pytest.skip("Cadence is not importable", allow_module_level=True)


def decode(row: np.ndarray, bits: int) -> tuple[str, int]:
    """The named fact a prefix event shows, read from its lit coordinates alone."""
    lit = np.flatnonzero(row)
    assert len(lit) == bits and (lit < 4 * bits).all()
    names = {"A" if c < 2 * bits else "B" for c in lit}
    assert len(names) == 1  # one named fact per event
    position = (lit % (2 * bits)) // 2
    assert sorted(position.tolist()) == list(range(bits))  # one coordinate per bit
    return names.pop(), int(sum(int(c % 2) << int(i) for c, i in zip(lit, position, strict=True)))


def test_one_bit_encoding_is_the_selection_instruments() -> None:
    for exposed in (False, True):
        prefix, query, target = bd.inputs(sr.CASES, 1, exposed=exposed)
        expected = sr.inputs(sr.CASES[:, :3], sr.CASES[:, 3], bd.NEUTRAL_EVENTS, exposed=exposed)
        assert len(prefix) == len(expected[0])
        for ours, theirs in zip(prefix, expected[0], strict=True):
            np.testing.assert_array_equal(ours, theirs)
        np.testing.assert_array_equal(query, expected[1])
        np.testing.assert_array_equal(target[:, 0], expected[2])


@pytest.mark.parametrize("bits", [3, 4])
def test_events_show_the_drawn_facts_and_the_answer_is_the_queried_fact(bits: int) -> None:
    rng = np.random.default_rng(bits)
    n = 64
    rows = np.column_stack(
        [
            rng.integers(0, 2**bits, n),
            rng.integers(0, 2**bits, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
        ]
    ).astype(np.int64)
    prefix, query, target = bd.inputs(rows, bits)
    assert len(prefix) == 2 + bd.NEUTRAL_EVENTS
    for i, row in enumerate(rows):
        first, second = decode(prefix[0][i], bits), decode(prefix[1][i], bits)
        assert {first[0], second[0]} == {"A", "B"}
        assert (first[0] == "A") == (row[2] == 0)  # order 0 shows A first
        facts = dict([first, second])
        assert facts["A"] == row[0] and facts["B"] == row[1]
        asked = facts["A" if row[3] == 0 else "B"]
        assert target[i].tolist() == [(asked >> j) & 1 for j in range(bits)]
    for neutral in prefix[2:]:
        assert (np.flatnonzero(neutral[0]) == [4 * bits]).all()
        assert np.unique(neutral, axis=0).shape[0] == 1
    assert (query.sum(axis=1) == 1).all()  # the trace arm's query names a fact, nothing else
    np.testing.assert_array_equal(query.argmax(axis=1), 4 * bits + 1 + rows[:, 3])


@pytest.mark.parametrize("bits", [3, 4])
def test_the_arms_differ_only_in_the_facts_the_query_carries(bits: int) -> None:
    rng = np.random.default_rng(7)
    rows = np.column_stack(
        [
            rng.integers(0, 2**bits, 32),
            rng.integers(0, 2**bits, 32),
            rng.integers(0, 2, 32),
            rng.integers(0, 2, 32),
        ]
    ).astype(np.int64)
    trace = bd.inputs(rows, bits)
    exposed = bd.inputs(rows, bits, exposed=True)
    for ours, theirs in zip(trace[0], exposed[0], strict=True):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(trace[2], exposed[2])
    np.testing.assert_array_equal(trace[1][:, : 4 * bits + 3], exposed[1][:, : 4 * bits + 3])
    assert not trace[1][:, 4 * bits + 3 :].any()
    shown = trace[0][0] + trace[0][1]  # both facts, whatever the order or the query
    np.testing.assert_array_equal(exposed[1][:, 4 * bits + 3 :], shown[:, : 4 * bits])


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("seed", list(range(12)))
def test_holdout_split_keeps_every_single_value_in_training(bits: int, seed: int) -> None:
    held = bd.holdout(seed, bits)
    values = 2**bits
    assert held.shape == (values, values) and held.sum() == values * values // 4
    np.testing.assert_array_equal(held, bd.holdout(seed, bits))  # fixed by the founder seed
    # every value of each fact keeps three quarters of its partners in training
    assert (held.sum(axis=1) == values // 4).all() and (held.sum(axis=0) == values // 4).all()
    assert (~held).any(axis=1).all() and (~held).any(axis=0).all()
    training = np.argwhere(~held)
    for i in range(bits):  # every combination of one bit of A and the same bit of B is taught
        combos = {(int(a >> i) & 1, int(b >> i) & 1) for a, b in training}
        assert combos == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_holdout_splits_differ_between_founders() -> None:
    assert len({bd.holdout(s, 3).tobytes() for s in range(12)}) > 1


def test_trials_are_fresh_draws_of_the_training_pairs() -> None:
    held = bd.holdout(3, 3)
    drawn = np.concatenate(list(bd.trials(3, 3, 64, held)))
    assert drawn.shape == (64 * bd.BATCH, 4)
    assert not held[drawn[:, 0], drawn[:, 1]].any()  # never a held-out pair
    pairs = {(int(a), int(b)) for a, b in drawn[:, :2]}
    assert pairs == {(int(a), int(b)) for a, b in np.argwhere(~held)}  # every training pair
    assert set(drawn[:, 2].tolist()) == {0, 1} and set(drawn[:, 3].tolist()) == {0, 1}
    rounds = drawn.reshape(64, bd.BATCH, 4)
    assert len({r.tobytes() for r in rounds}) == 64  # no round repeats another
    shorter = np.concatenate(list(bd.trials(3, 3, 8, held)))
    np.testing.assert_array_equal(shorter, drawn[: 8 * bd.BATCH])  # a budget is a prefix
    other = np.concatenate(list(bd.trials(4, 3, 8, held)))
    assert not np.array_equal(other, shorter)


def test_donors_complement_both_facts_in_the_same_order() -> None:
    panel = bd.histories(3)
    donor = bd.donors(panel, 3)
    np.testing.assert_array_equal(panel[donor, :2], 7 - panel[:, :2])
    np.testing.assert_array_equal(panel[donor, 2], panel[:, 2])
    np.testing.assert_array_equal(donor[donor], np.arange(len(panel)))
    expected = bd.truth(panel, 3)
    assert (expected[donor] != expected).all()  # every bit of the queried fact differs


def test_scores_and_gate_separate_selection_from_a_table_and_a_constant() -> None:
    bits = 3
    held = bd.holdout(5, bits)
    panel = bd.histories(bits)
    expected = bd.truth(panel, bits)
    on_held = held[panel[:, 0], panel[:, 1]]
    constant = np.zeros_like(expected)
    table = expected.copy()
    table[on_held] = 1 - expected[on_held]  # right on the trained pairs alone
    donor = expected[bd.donors(panel, bits)]
    selection = {"intact": expected, "erased": constant, "transplant": donor}
    scores = bd.score(selection, held, bits)
    for panel_name in ("training", "held_out"):
        assert scores["intact"][panel_name]["exact"] == 1.0
        assert scores["intact"][panel_name]["per_bit"] == 1.0
        assert scores["erased"][panel_name]["per_bit"] == 0.5  # balanced panels
        assert scores["transplant"][panel_name]["follow_exact"] == 1.0
        assert scores["transplant"][panel_name]["stay_exact"] == 0.0
    assert scores["erased"]["held_out"]["exact"] == 2.0**-bits
    assert scores["intact"]["held_out"]["cases"] == 2 * 2 * held.sum()
    assert bd.gate({"completed": True, "scores": scores}, bits)
    tabled = bd.score({"intact": table, "erased": constant, "transplant": donor}, held, bits)
    assert tabled["intact"]["training"]["exact"] == 1.0
    assert tabled["intact"]["held_out"]["exact"] == 0.0
    assert not bd.gate({"completed": True, "scores": tabled}, bits)
    leaky = bd.score({"intact": expected, "erased": expected, "transplant": donor}, held, bits)
    assert not bd.gate({"completed": True, "scores": leaky}, bits)  # erased must be at chance
    assert not bd.gate({"completed": False, "scores": scores}, bits)
    swapped = bd.score(
        {"intact": expected[:, ::-1], "erased": constant, "transplant": donor}, held, bits
    )
    assert swapped["intact"]["held_out"]["other_fact_unequal"] == 1.0


def spy_lessons(brain: object) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    original = brain.learner.step  # type: ignore[attr-defined]

    def step(drive, labels, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        state, report = original(drive, labels, *args, **kwargs)
        calls.append(
            {
                "drive": np.array(drive, copy=True),
                "labels": np.array(labels, copy=True),
                "answer": kwargs.get("answer"),
                "report": report,
            }
        )
        return state, report

    brain.learner.step = step  # type: ignore[attr-defined]
    return calls


@pytest.mark.parametrize("rule", ["act", "every"])
def test_lessons_read_the_pre_query_stimulus_and_the_act_rule_teaches_its_wrong_rows(
    rule: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(bd.BRAINS, "m8", [8])
    bits = 3
    held = bd.holdout(1, bits)
    rows = next(bd.trials(1, bits, 1, held))
    twin = bd.compose("m8", rule, 1, bits)
    prefix, query, target = bd.inputs(rows, bits)
    twin.reset()
    for observation in prefix:
        twin.act(observation, greedy=True)
    drive = twin.stimulus(query)
    answer = np.asarray(twin.act(query, greedy=True)).reshape(len(rows), bits)
    wrong = (answer != target).any(axis=1)
    brain = bd.compose("m8", rule, 1, bits)
    assert brain.learner.config.surprise == (rule == "act")
    calls = spy_lessons(brain)
    work = sr.Work()
    got, report, stop = bd.graph_round(brain, work, rows, bits, exposed=False, rule=rule)
    assert stop is None and len(calls) == 1 and work.lessons == 1
    np.testing.assert_array_equal(got, answer)  # the act the lesson is judged by
    np.testing.assert_allclose(calls[0]["drive"], drive)  # the trace the prefix built
    np.testing.assert_array_equal(calls[0]["labels"], target)
    assert report is calls[0]["report"]
    if rule == "act":
        np.testing.assert_array_equal(calls[0]["answer"], answer)
        assert report["accepted_presentations"] == wrong.sum()
    else:
        assert calls[0]["answer"] is None
        assert report["accepted_presentations"] == len(rows)


def test_readings_fork_the_trained_brain_and_lesions_reach_the_trace_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cadence import Brain

    monkeypatch.setitem(bd.BRAINS, "m8", [8])
    bits = 3
    held = bd.holdout(2, bits)
    brain = bd.compose("m8", "act", 2, bits)
    work = sr.Work()
    for rows in bd.trials(2, bits, 2, held):
        bd.graph_round(brain, work, rows, bits, exposed=False, rule="act")
    updates = brain.learner.updates
    reading = bd.graph_reading(brain, bits, held, sr.Work(), exposed=False)  # raises if changed
    assert reading["completed"] and brain.learner.updates == updates
    panel = bd.histories(bits)
    donor = bd.donors(panel, bits)
    walker = Brain.load(brain.save(tmp_path / "trained.npz"))
    walker.reset()
    prefix, _, _ = bd.inputs(bd.asked(panel, 0), bits)
    for observation in prefix:
        walker.act(observation, greedy=True)
    trace = walker.working_memory.trace.copy()
    warm = np.array(walker.basal_ganglia.state.v, copy=True)
    assert np.abs(trace).max() > 0
    for condition, expected in (
        ("intact", trace),
        ("erased", np.zeros_like(trace)),
        ("transplant", trace[donor]),
    ):
        memory = walker.working_memory
        saved = {k: getattr(memory, k).copy() for k in ("trace", "last", "cold")}
        bd.lesion(memory, condition, donor)
        np.testing.assert_array_equal(memory.trace, expected)
        np.testing.assert_array_equal(walker.basal_ganglia.state.v, warm)  # warm state kept
        for k, v in saved.items():
            setattr(memory, k, v)
    answers = np.asarray(reading["answers"]["intact"])
    assert answers.shape == (len(panel), 2, bits) and (answers >= 0).all()


def test_gru_gradients_match_finite_differences() -> None:
    bits = 3
    model = bd.Supervised(bits, 0, hidden=6)
    rng = np.random.default_rng(0)
    for value in model.parameters().values():
        value += rng.normal(0.0, 0.3, value.shape)  # a readout away from zero
    rows = np.column_stack(
        [rng.integers(0, 8, 4), rng.integers(0, 8, 4), rng.integers(0, 2, 4), rng.integers(0, 2, 4)]
    ).astype(np.int64)
    prefix, query, target = bd.inputs(rows, bits)
    events = [*prefix, query]
    _, grads, _ = model.gradients(events, target)
    for name in ("W", "b", "Wz", "Uz", "bz", "Wr", "Ur", "br", "Wn", "Un", "bn"):
        value = model.parameters()[name]
        for index in [tuple(rng.integers(0, s) for s in value.shape) for _ in range(3)]:
            keep = value[index]
            value[index] = keep + 1e-6
            up = model.gradients(events, target)[0]
            value[index] = keep - 1e-6
            down = model.gradients(events, target)[0]
            value[index] = keep
            assert grads[name][index] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-8)


def test_the_query_credit_path_holds_the_state_entering_the_query_fixed() -> None:
    import copy

    bits = 3
    query_only = bd.Supervised(bits, 1, hidden=6, through="query")
    rng = np.random.default_rng(1)
    for value in query_only.parameters().values():
        value += rng.normal(0.0, 0.3, value.shape)
    whole = copy.deepcopy(query_only)
    whole.through = "trial"
    rows = next(bd.trials(1, bits, 1, bd.holdout(1, bits)))
    prefix, query, target = bd.inputs(rows, bits)
    loss, grads, _ = query_only.gradients([*prefix, query], target)
    entering, _ = whole.run(prefix)
    fixed, expected, _ = whole.gradients([query], target, start=entering)
    assert loss == pytest.approx(fixed)
    for name, value in expected.items():
        np.testing.assert_allclose(grads[name], value, rtol=1e-12, atol=1e-15)
    full = whole.gradients([*prefix, query], target)[1]
    assert not np.allclose(full["Wn"], grads["Wn"])  # the trial's gradient reaches the prefix
    with pytest.raises(ValueError):
        bd.Supervised(bits, 1, through="prefix")


def test_gru_learns_one_round_and_its_reading_lesions_its_state() -> None:
    bits = 3
    held = bd.holdout(0, bits)
    model = bd.Supervised(bits, 0)
    rows = next(bd.trials(0, bits, 1, held))
    prefix, query, target = bd.inputs(rows, bits)
    before = model.gradients([*prefix, query], target)[0]
    model.lesson([*prefix, query], target)
    assert model.gradients([*prefix, query], target)[0] < before
    reading = bd.gru_reading(model, bits, held, exposed=False)
    panel = bd.histories(bits)
    erased = np.asarray(reading["answers"]["erased"])
    for q in (0, 1):
        _, alone, _ = bd.inputs(bd.asked(panel, q), bits)
        h, _ = model.run([alone])  # the query from a zero state: the erased reading
        np.testing.assert_array_equal(erased[:, q], model.answer(h))


@pytest.mark.parametrize(
    "learner,rule", [("graph", "act"), ("graph", "every"), ("gru", "trial"), ("gru", "query")]
)
def test_a_reading_along_a_life_equals_a_life_of_that_budget(
    learner: str, rule: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(bd.BRAINS, "m8", [8])
    spec = {
        "learner": learner,
        "brain": "m8",
        "arm": "trace",
        "rule": rule,
        "seed": 4,
        "bits": 3,
        "gru": {"hidden": 8},
    }
    along = bd.job({**spec, "budgets": [1, 2]})
    alone = bd.job({**spec, "budgets": [2]})
    assert along["readings"]["2"]["answers"] == alone["readings"]["2"]["answers"]
    assert along["training"]["completed"] and set(along["readings"]) == {"1", "2"}
    assert along["held_out_pairs"] == np.argwhere(bd.holdout(4, 3)).tolist()
    log = along["training"]["log"]
    assert len(log["taught_rows"]) == len(log["lessons_exhausted"]) == 1  # one partial window
    assert 0 <= log["lessons_exhausted"][0] <= 2 and 0 <= log["taught_rows"][0] <= 2 * bd.BATCH
