"""The sequence benchmarks (cadence#232, #234, #236, #237, #238, #239): the tasks, the arms as
the library receives them, a short life of every reading, and readings that never change the
life they read."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(name: str):
    key = f"sequence_{name}"  # the name width.py loads the battery under
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


bt = _load("battery")
wd = _load("width")
od = _load("order")

SMALL = {"lessons": 3, "tests": 2, "module": 16}


def _tokens(moments):
    return np.stack([np.argmax(m.x, axis=1) for m in moments], axis=1)


def test_look_back_answers_the_cued_place_at_the_test_lengths():
    task = bt.LookBack(lengths=(5, 7), test_lengths=(20, 24))
    rng = np.random.default_rng(0)
    for train, (lo, hi) in ((True, (5, 7)), (False, (20, 24))):
        moments = task.episode(rng, train)
        story, query = _tokens(moments[:-1]), moments[-1]
        length = story.shape[1]
        assert lo <= length <= hi
        cues = np.argmax(query.x, axis=1) - task.vocab
        place = np.array([{"first": 0, "last": length - 1, "second": 1}[task.cues[c]]
                          for c in cues])
        assert np.array_equal(query.label, story[np.arange(bt.BATCH), place])
        assert np.array_equal(query.age, length - 1 - place)
    task.ask = ("last",)
    assert {c for _ in range(4) for c in task.episode(rng, True)[-1].condition} == {"last"}


def test_copying_answers_each_token_and_feeds_back_the_answer():
    task = bt.Repeat(unique=True, lengths=(4, 4))
    moments = task.episode(np.random.default_rng(1), True)
    story = _tokens(moments[1:5])
    assert all(len(set(row)) == 4 for row in story)  # unique tokens
    go, rest = moments[5], moments[6:]
    assert np.array_equal(go.label, story[:, 0]) and not go.feedback
    assert [m.feedback for m in rest] == [True] * 3
    assert all(np.array_equal(m.label, story[:, j + 1]) for j, m in enumerate(rest))


def test_ago_and_before_answer_from_the_story():
    rng = np.random.default_rng(2)
    ago = bt.Ago(lengths=(8, 8)).episode(rng, True)
    story, question = _tokens(ago[:-1]), ago[-1]
    asked = np.argmax(question.x[:, : bt.STORY], axis=1)
    place = np.array([list(row).index(t) for row, t in zip(story, asked, strict=True)])
    assert np.array_equal(question.label, np.searchsorted(bt.BUCKETS, 7 - place, side="right"))
    before = bt.Before(lengths=(8, 8)).episode(rng, True)
    story = _tokens(before[:-2])
    a = np.argmax(before[-2].x[:, : bt.STORY], axis=1)
    b = np.argmax(before[-1].x[:, : bt.STORY], axis=1)
    told = [(list(row).index(x), list(row).index(y)) for row, x, y in zip(story, a, b, strict=True)]
    assert np.array_equal(before[-1].label, [int(i < j) for i, j in told])
    assert before[-2].label is None


def test_two_hops_ask_the_end_of_a_chain():
    task = bt.TwoHop(think=False)
    moments = task.episode(np.random.default_rng(3), True)
    pairs = _tokens(moments[:-1])
    query = moments[-1]
    for row in range(bt.BATCH):
        follows = dict(zip(pairs[row, ::2], pairs[row, 1::2], strict=True))
        a = int(np.argmax(query.x[row]))
        assert follows[follows[a]] == query.label[row]


@pytest.mark.parametrize("arm", list(bt.ARMS))
def test_every_arm_reaches_the_brain(arm):
    brain = bt.make_brain(bt.TwoHop(), arm, 0, module=16)
    genes = brain.describe()["genes"]
    for name, value in bt.ARMS[arm].items():
        if name == "recall":
            assert brain.recall is not None
            continue
        assert name in genes, name
        got = genes[name]
        assert (list(got) if isinstance(value, list) else got) == value, name


def test_a_short_life_reads_every_reading():
    look = bt.run_life({"task": "lookback", "arm": "recipe", "founder": 0, "erase": True,
                        "task_args": {"lengths": [5, 6], "test_lengths": [9, 10]}, **SMALL})
    assert set(look["test"]["conditions"]) <= {"first", "last", "second"}
    assert set(look["erased"]) == set(look["test"]) and "teacher_forced" not in look
    copy = bt.run_life({"task": "repeat", "arm": "recipe_relative", "founder": 0, **SMALL})
    assert 0.0 <= copy["teacher_forced"]["accuracy"] <= 1.0
    phased = bt.run_life({"task": "lookback", "arm": "rehearsal", "founder": 0,
                          "phases": [["first"], ["last"]], **SMALL})
    assert phased["phases"] == [["first"], ["last"]] and phased["lessons"] == 3
    assert phased["settings"]["learning_rehearsal"] == 1.0
    assert phased["refused_lessons"] == 0
    with pytest.raises(ValueError, match="look back's cue groups"):
        bt.run_life({"task": "ago", "arm": "when", "founder": 0, "phases": [["first"]],
                     **SMALL})


def test_a_life_is_the_same_twice_and_the_summary_spans_its_founders():
    spec = {"task": "ago", "arm": "when", **SMALL}
    rows = [bt.run_life({**spec, "founder": f}) for f in (0, 1, 0)]
    for row in rows:
        row.pop("seconds")
    assert rows[0] == rows[2]
    (cell,) = bt.summarize(rows[:2])
    assert cell["founders"] == [0, 1] and len(cell["test"]["accuracy"]) == 2


def test_fan_in_rates_scale_a_wide_brain_to_the_narrow_one():
    task = wd.TASKS["associative"]()
    narrow = wd.compose(task, 64, 0, True)
    wide = wd.compose(task, 128, 0, True, {"learning_surprise_share": True})
    assert wide.learner.config.surprise_share
    rates = wd.fan_in_rates(wide, narrow)
    assert rates.shape == (wide.connectome.synapses,)
    assert rates.max() <= 1.0 and rates.min() < 1.0
    assert np.all(wd.fan_in_rates(narrow, narrow) == 1.0)


def test_a_short_width_life_reads_its_checkpoints():
    row = wd.run_life({"task": "lookup", "arm": "relative", "width": 32, "founder": 0,
                       "lessons": 4, "checkpoints": [2]})
    assert [p["lessons"] for p in row["curve"]] == [2, 4]
    last = row["curve"][-1]
    assert 0.0 <= last["test"] <= 1.0 and 0.0 <= last["motor_at_bound"] <= 1.0
    assert last["refused_acts"] == 0 and last["sweeps_per_act"] > 0
    (cell,) = wd.summarize([row])
    assert cell["lessons"] == [2, 4] and cell["held_out"][1][0] == round(last["test"], 3)


def test_the_ordered_world_writes_two_values_and_asks_one():
    for condition in od.CONDITIONS:
        for value in (0, 1):
            events = np.concatenate(od.events(condition, value))
            first = condition.startswith("ask-first")
            writes = np.flatnonzero(events[:, od.WRITE])
            assert np.flatnonzero(events[:, od.START]).tolist() == [0]
            assert writes[1] - writes[0] == (2 if condition.endswith("-gap") else 1)
            asked = events[writes[0] if first else writes[1], od.PAYLOAD]
            assert np.array_equal(asked, od.VALUES[value])
            assert not events[writes[1] + 1:, od.PAYLOAD].any()  # no value after the writes
            assert events[-1, od.QUERY_FIRST if first else od.QUERY] == 1.0
            assert events[-1, od.QUERY if first else od.QUERY_FIRST] == 0.0


def test_the_schedule_is_balanced_in_blocks():
    plan = od.schedule(24, np.random.default_rng(4))
    cells = {(c, v) for c in od.CONDITIONS for v in (0, 1)}
    assert all(set(plan[i:i + 8]) == cells for i in range(0, 24, 8))


def test_a_life_through_live_and_its_readings_never_change_it(monkeypatch):
    monkeypatch.setattr(od, "EVALUATION", 8)
    tiny = {"modules": [8]}
    row = od.run_life({"arm": "position_address", "founder": 1001, "settings": tiny,
                       "budgets": [4, 8]})
    assert [r["episodes"] for r in row["readings"]] == [4, 8] and row["ended"] is None
    assert len(row["training"]) == 1 and row["refused"] == 0
    brain = od.make_brain("position_address", 1001, tiny)
    for x in od.events("ask-first", 1):
        brain.live(x)
    before = (np.asarray(brain.brain.efficacy).copy(), brain.decision_id)
    od.read(brain, 1001, greedy=False)
    assert np.array_equal(np.asarray(brain.brain.efficacy), before[0])
    assert brain.decision_id == before[1]


def test_a_life_taught_by_lessons_learns_at_the_query(monkeypatch):
    monkeypatch.setattr(od, "EVALUATION", 8)
    tiny = {"modules": [8]}
    row = od.run_life({"arm": "recipe", "founder": 1001, "teach": "lesson", "settings": tiny,
                       "budgets": [4]})
    assert row["teach"] == "lesson" and row["refused"] == 0
    assert row["settings"]["recall_anchor"] and row["readings"][0]["episodes"] == 4
    brain = od.make_brain("founder", 1001, tiny)
    before = np.asarray(brain.brain.efficacy).copy()
    assert od.taught(brain, "ask-first", 1) in (0, 1)
    assert not np.array_equal(np.asarray(brain.brain.efficacy), before)
