"""The long-context instruments (cadence#268): the tasks answer what they say, the terms sum to the
attention, a read never changes the life it reads, and a short life of each reading runs."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(name: str):
    key = f"long_context_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


rc = _load("reach")
nt = _load("nets")
ct = _load("cost")
B = rc.B


def _tokens(moments, vocab):
    x = np.stack([m.x for m in moments], axis=1)
    out = np.argmax(x, axis=2)
    out[x[:, :, vocab:].sum(axis=2) > 0] = -1
    return out


def test_a_dense_cue_answers_the_token_k_moments_back_and_never_a_cue():
    task = rc.DenseNBack(ks=(5, 9, 13), lessons=20)
    moments = task.episode(np.random.default_rng(0), True)
    assert len(moments) == task.warm + 4 * 20
    tokens = _tokens(moments, task.vocab)
    rows = np.arange(B.BATCH)
    answered = 0
    for t, m in enumerate(moments):
        if m.label is None:
            continue
        answered += 1
        assert t >= task.warm and t % 4 == 3
        ks = m.age + 1
        assert set(ks) <= {5, 9, 13}
        np.testing.assert_array_equal(m.label, tokens[rows, t - ks])
        assert (tokens[rows, t - ks] >= 0).all()
        assert m.condition == [f"k{k}" for k in ks]
    assert answered == 20
    with pytest.raises(ValueError, match="odd"):
        rc.DenseNBack(ks=(4,))


def test_look_back_names_its_length_and_the_same_token_story_holds_one_token():
    task = rc.ReachLookBack(test_lengths=(30, 30))
    moments = task.episode(np.random.default_rng(1), False)
    assert len(moments) == 31
    assert {c.split("@")[1] for c in moments[-1].condition} == {"30"}
    same = rc.ReachLookBack(test_lengths=(30, 30), same=True)
    moments = same.episode(np.random.default_rng(1), False)
    story = np.stack([np.argmax(m.x, axis=1) for m in moments[:-1]], axis=1)
    assert (story == story[:, :1]).all()
    np.testing.assert_array_equal(moments[-1].label, story[:, 0])
    assert moments[-1].condition[0].startswith("same-")


def test_the_dense_story_answers_each_place_at_every_length():
    task = rc.DenseLookBack(length=40)
    moments = task.episode(np.random.default_rng(2), True)
    tokens = _tokens(moments, task.vocab)
    rows = np.arange(B.BATCH)
    for t, m in enumerate(moments):
        if m.label is None:
            continue
        pos = t - 1 - m.age
        np.testing.assert_array_equal(m.label, tokens[rows, pos])
        names = [c.split("@")[0] for c in m.condition]
        for r, name in enumerate(names):
            assert pos[r] == {"first": 0, "second": 1, "last": t - 1}[name]


def test_a_periodic_answer_names_its_period_and_repeat():
    task = rc.ReachPeriodic(periods=(8,), repeats=4, variation=0.0)
    moments = task.episode(np.random.default_rng(3), False)
    tokens = np.stack([np.argmax(m.x, axis=1) for m in moments], axis=1)
    for t, m in enumerate(moments):
        if m.label is None:
            assert t < 8
            continue
        assert m.condition == [f"P8:r{t // 8}"] * B.BATCH
        assert m.period == 8
        np.testing.assert_array_equal(m.label, tokens[:, t + 1 - 8])


def _short_brain(arm="Fd", **spec):
    task = rc.DenseNBack(ks=(3, 5), lessons=6)
    return task, rc.compose(task, {"arm": arm, "founder": 0, "capacity": 32, "position": 16,
                                   "module": 16, **spec})


@pytest.mark.parametrize("arm", ["Fd", "FdL"])
def test_the_terms_sum_to_the_attention(arm):
    task, brain = _short_brain(arm)
    rng = np.random.default_rng(4)
    brain.reset()
    for m in task.episode(rng, True):
        if m.label is not None:
            brain.lesson(m.x, m.label)
            split = rc.terms(brain.recall, m.x)
            brain.act(m.x, greedy=True)
            total = sum(split.values())
            q = np.exp(total - total.max(axis=1, keepdims=True))
            q /= q.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(q, brain.recall.attended_heads[0], atol=1e-12)
        else:
            brain.act(m.x, greedy=True)


def test_a_frozen_read_leaves_the_life_as_it_was():
    task, brain = _short_brain()
    rng = np.random.default_rng(5)
    brain.reset()
    for m in task.episode(rng, True):
        if m.label is not None:
            brain.lesson(m.x, m.label)
        brain.act(m.x, greedy=True)
    twin = copy.deepcopy(brain)
    check = {"checked": 0, "worst": 0.0}
    cells, work = rc.frozen_read(brain, task, np.random.default_rng(6), 1, check)
    assert set(cells) == {"k3", "k5"} and work["answers"] == 6
    for a, b in zip(brain.recall.position_maps[0], twin.recall.position_maps[0], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(brain.recall.keys, twin.recall.keys)
    np.testing.assert_array_equal(brain.learner.brain.efficacy, twin.learner.brain.efficacy)


def test_a_right_slot_is_counted_as_held_only_while_the_capacity_holds_it():
    task = rc.DenseNBack(ks=(21,), lessons=4)
    brain = rc.compose(task, {"arm": "Fd", "founder": 0, "capacity": 8, "position": 8,
                              "module": 16})
    check = {"checked": 0, "worst": 0.0}
    cells, _ = rc.frozen_read(brain, task, np.random.default_rng(7), 1, check)
    assert cells["k21"]["held"] == 0.0 and cells["k21"]["mass_value"] == 0.0


def test_a_short_life_reads_and_the_network_reads_the_same_conditions():
    spec = {"tag": "t", "task": "nback_dense", "task_args": {"ks": [3, 5], "lessons": 6},
            "arm": "FdL", "founder": 0, "capacity": 32, "position": 16, "module": 16, "answers": 12,
            "checkpoints": [0.5], "reads": [{"task": "nback_dense", "always": True,
                                             "task_args": {"ks": [3, 5], "lessons": 6}}]}
    row = rc.run_life(spec)
    assert [r["at"] for r in row["reads"]] == [6, 12]
    assert row["work"]["answers"] == 12 and row["check"]["worst"] < 1e-9
    assert row["parameters"] > 0 and row["reads"][-1]["conditions"]["k3"]["n"] > 0
    life = nt.run_life({"tag": "n", "model": "transformer", "width": 8, "founder": 0, **{
        k: spec[k] for k in ("task", "task_args", "capacity", "position")},
        "context": 16, "stride": 8, "length": 128, "at_answers": [2], "max_answers": 6,
        "max_seconds": 30, "reads": spec["reads"]})
    assert set(life["readings"][-1]["conditions"]["nback_dense"]) == {"k3", "k5"}


@pytest.mark.parametrize("model", ["transformer", "lstm"])
def test_a_network_sees_only_its_window_and_its_answers(model):
    assert list(nt.slices(10, 4, 2, np.ones(10, dtype=bool))) == [
        (0, 0, 2), (0, 2, 4), (0, 4, 6), (2, 6, 8), (4, 8, 10)]
    answered = np.zeros(10, dtype=bool)
    answered[7] = True
    assert list(nt.slices(10, 4, 2, answered)) == [(2, 6, 8)]
    net = nt.build(model, 5, 3, 8, 0, 32)
    assert net.macs(4) > net.macs(2) > 0
    assert nt.matched_width(model, 5, 3, nt.count(net)) == 8


def test_a_cost_cell_reads_its_parts():
    row = ct.run_cell({"inputs": 8, "streams": 1, "capacity": 8, "position": 8, "acts": 3,
                       "lessons": 2})
    assert row["slots"] == 8 and row["act_ms"] >= row["act_read_ms"] + row["act_write_ms"]
    assert row["slot_bytes_per_stream"] == (3 * 8 * 8 + 1) * 8
