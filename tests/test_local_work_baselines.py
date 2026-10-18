"""The local-work instrument's amendment 3: recurrent baselines, the per-slot lesson, the
damped fallback and the relearning reading."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "benchmarks" / "local_work"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lw():
    return _load("local_work", FOLDER / "local_work.py")


@pytest.fixture(scope="module")
def rec(lw):
    return _load("local_work_recurrent", FOLDER / "recurrent.py")


def _loss(net, xs, labels):
    h = np.zeros((len(xs[0]), net.hidden))
    for x in xs:
        h, _ = net._step(x, h)
    p = net._probabilities(h)
    rows = np.arange(len(h))[:, None]
    slots = np.arange(net.slots)[None, :]
    return float(-np.log(p[rows, slots, labels]).mean())


def test_truncated_backward_matches_finite_differences(rec):
    net = rec.SkillGRU(4, 2, 2, 3, 7, truncation=4)
    rng = np.random.default_rng(1)
    xs = [rng.normal(size=(2, 4)) for _ in range(3)]
    labels = np.array([[0, 1], [1, 1]])
    net.reset(2)
    for x in xs[:-1]:
        net.answer(x)
    h, cache = net._step(xs[-1], net.h)
    net.history.append(cache)
    grads = net.gradients(labels, net._probabilities(h), h)
    eps = 1e-6
    for name, value in net.params().items():
        for index in [(0,) * value.ndim, tuple(s - 1 for s in value.shape)]:
            keep = value[index]
            value[index] = keep + eps
            up = _loss(net, xs, labels)
            value[index] = keep - eps
            down = _loss(net, xs, labels)
            value[index] = keep
            assert grads[name][index] == pytest.approx((up - down) / (2 * eps), abs=1e-7)


def test_work_counts_and_determinism(rec):
    net = rec.SkillGRU(8, 2, 2, 5, 0)
    assert net.step_macs == 3 * 5 * (8 + 5) + 2 * 2 * 5
    assert net.macs() == {"input": 120, "recurrent": 75, "readout": 20}
    first = rec.train_gru(2, 4, 0, budget=40)[1]
    second = rec.train_gru(2, 4, 0, budget=40)[1]
    first.pop("seconds")
    second.pop("seconds")
    assert first == second
    assert first["lessons"] <= 40 and first["learning_macs"] > 0


def test_a_lesson_that_is_not_a_moment_keeps_the_stream(rec):
    net = rec.SkillGRU(4, 1, 2, 3, 2)
    net.answer(np.eye(4)[[0]])
    state, frames = net.h.copy(), len(net.history)
    weights = net.W.copy()
    net.lesson(np.eye(4)[[1]], np.array([[1]]), advance=False)
    assert np.array_equal(net.h, state) and len(net.history) == frames
    assert not np.array_equal(net.W, weights)


def test_the_oracle_router_steps_only_what_changed(rec, lw):
    bank, record = rec.train_bank(2, 3, 0, budget=40)
    maps = lw.skill_maps(2, 0, 4, 2)
    schedule = [{"index": i, "kind": "settle-hold", "episode": -1, "cue0": 0} for i in range(40)]
    schedule.append({"index": 40, "kind": "input", "episode": 0, "cue0": 1})
    moments = rec.live_bank(bank, schedule, maps)
    assert moments[0]["stepped"] == [0, 1]
    change = moments[-1]
    assert 0 in change["stepped"]
    assert change["macs"] == sum(bank.modules[k].step_macs for k in change["stepped"])
    # under a constant input the modules come to rest and stop being stepped
    assert moments[-2]["stepped"] == [] and moments[-2]["macs"] == 0
    assert record["parameters"] == bank.parameter_count


def test_a_cadence_schedule_replays_through_the_gru(rec, lw):
    case = lw.run_case(lw.CaseSpec(
        "sparse", (8, 8), 0, lessons=20, probes=1, relation_probes=1, hold=8, check="cone"
    ))
    schedule = rec.schedule_from_case(case)
    assert len(schedule) == len(case["moments"])
    assert [m["kind"] for m in schedule] == [m["kind"] for m in case["moments"]]
    relation = [m for m in schedule if m["kind"] == "relation"]
    assert relation and relation[0]["relation"] == 0
    net, _ = rec.train_gru(2, 4, 0, budget=20)
    moments = rec.live_gru(net, schedule, lw.skill_maps(2, 0, 4, 2))
    assert {m["macs"] for m in moments} == {net.step_macs}
    lesson = next(m for m in moments if m["kind"] == "relation")
    assert lesson["lesson_changed_parameters"] > 0 and lesson["lesson_macs"] > 0
    summary = rec.life_summary(moments)
    assert summary["input"]["change_macs"] == net.step_macs
    assert summary["moments"] == len(moments)
    json.dumps(summary)


def test_the_slot_lesson_is_the_library_lesson_for_one_skill(lw):
    brain = lw.build_brain("sparse", (8,), 0)
    lw.acquire(brain, lw.skill_maps(1, 0, 4, 2), 0, d=4, lessons=20)
    brain.reset()
    x = lw.observations(np.array([[1], [2]]), 4)
    library, slot = lw.clone(brain), lw.clone(brain)
    library.lesson(x, np.array([0, 1]))
    lw.slot_lesson(slot, x, np.array([0, 1]), 0)
    assert np.array_equal(library.brain.efficacy, slot.brain.efficacy)
    assert np.array_equal(library.brain.bias, slot.brain.bias)


def test_a_fixed_slot_lesson_leaves_the_other_blocks_alone(lw):
    widths = (8, 8)
    brain = lw.build_brain("sparse", widths, 0, momentum=0.0)
    lw.acquire(brain, lw.skill_maps(2, 0, 4, 2), 0, d=4, lessons=20)
    brain.reset()
    blocks = lw.block_of(brain, widths, 4, 2)
    outside = blocks[brain.connectome.post] != 0
    x = lw.observations(np.array([[1, 2], [3, 0]]), 4)
    fixed, library = lw.clone(brain), lw.clone(brain)
    before = np.array(brain.brain.efficacy)
    lw.apply_lesson(fixed, x, np.array([[0, 1], [1, 0]]), "slot-fixed")
    lw.apply_lesson(library, x, np.array([[0, 1], [1, 0]]), "library")
    moved = np.asarray(fixed.brain.efficacy) - before
    assert np.count_nonzero(moved[outside]) == 0 and np.count_nonzero(moved[~outside]) > 0
    assert np.array_equal(np.asarray(fixed.brain.bias)[blocks != 0],
                          np.asarray(brain.brain.bias)[blocks != 0])
    assert np.count_nonzero((np.asarray(library.brain.efficacy) - before)[outside]) > 0
    with pytest.raises(ValueError, match="lesson"):
        lw.apply_lesson(fixed, x, np.array([[0, 1], [1, 0]]), "half")


def test_a_stalled_repair_goes_to_the_library_settle(lw):
    brain = lw.build_brain("sparse", (8, 8), 0)
    lw.acquire(brain, lw.skill_maps(2, 0, 4, 2), 0, d=4, lessons=20)
    brain.reset()
    brain.act(lw.observations(np.zeros((1, 2), int), 4), greedy=True)
    state, drive = brain.basal_ganglia.state, brain.basal_ganglia._drive
    tolerance = float(brain.learner.config.tolerance)
    calls = []

    def damped(d, s):
        calls.append(1)
        return brain._equilibrate(d, s, budget=1024, tolerance=tolerance)

    repair = lw.LocalRepair(
        brain.brain, state, drive, tolerance=tolerance, max_rounds=1, fallback=damped
    )
    drive2 = brain.stimulus(lw.observations(np.array([[1, 0]]), 4))
    settled, counts = repair.repair(drive2)
    assert calls == [1] and counts.fallbacks == 1 and counts.stalled_round == 1
    assert counts.qualified and counts.fallback_contacts > 0
    assert counts.contacts() == (
        counts.messages + counts.check_contacts + counts.fallback_contacts
    )
    assert float(brain.brain.residual(drive2, settled)[0]) <= tolerance
    with pytest.raises(ValueError, match="patience"):
        lw.LocalRepair(brain.brain, state, drive, tolerance=tolerance, patience=0)


def test_the_relearning_reading_records_each_lesson(lw):
    spec = lw.CaseSpec("sparse", (8, 8), 0, lessons=20, momentum=0.0)
    result = lw.run_relearning(spec, budget=20)
    assert set(result["arms"]) == set(lw.LESSONS)
    fixed = result["arms"]["slot-fixed"]
    assert fixed["footprint_mean"]["synapses_outside_block0"] == 0.0
    assert result["arms"]["library"]["footprint_mean"]["synapses_outside_block0"] > 0
    assert len(fixed["accuracy"]) == 2 and fixed["lessons"] <= 20
    assert lw.CaseSpec("sparse", (8, 8), 0, lesson="slot-fixed", fallback=64).name.endswith(
        "-slotfixed-fb64"
    )
