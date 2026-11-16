"""The recall strength chamber (cadence#336) in short: its readings hold their shape on a bank
of 32 slots beside the same brain without the option."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAPACITY = 32


def _load():
    key = "recall_strength_chamber"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / "strength.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


ch = _load()


def share_held(reads, span):
    return float(reads[0][span][:, 0].mean())


def test_a_fact_told_again_is_held_by_its_tellings_and_the_plain_bank_holds_a_capacity():
    spans = (0.5, 2.0, 8.0)
    deep = ch.dose(0, "strength", CAPACITY, doses=(1, 2, 16), spans=spans, facts=2)
    assert [share_held(deep[1], span) for span in spans] == [1.0, 0.0, 0.0]
    assert [share_held(deep[2], span) for span in spans] == [1.0, 1.0, 0.0]
    assert [share_held(deep[16], span) for span in spans] == [1.0, 1.0, 1.0]
    # read right wherever it is held: the read is the plain memory's
    assert deep[2][0][2.0][:, 1].all() and deep[16][0][8.0][:, 1].all()
    # told twice: about 8 ln(2 / 0.9) capacities, 6.4
    assert 5.0 < float(np.median(deep[2][1])) < 8.0 and np.isinf(deep[16][1]).all()
    # the plain bank holds a capacity of moments, a fact told four times as one told once
    plain = ch.dose(0, "plain", CAPACITY, doses=(1, 4), spans=spans[:2], facts=2)
    for n in (1, 4):
        assert [share_held(plain[n], span) for span in spans[:2]] == [1.0, 0.0]
        assert (plain[n][1] <= 1.0).all()


def test_one_struck_moment_is_kept_at_once_where_the_brain_reads_its_surprise():
    spans = (0.5, 4.0)
    deep = ch.one_shot(0, "strength", CAPACITY, spans=spans)
    assert deep["tellings"]["struck"] > 8.0 > deep["tellings"]["plain"]
    assert share_held(deep["struck"], 4.0) == 1.0 and np.isinf(deep["struck"][1]).all()
    plain = ch.one_shot(0, "plain", CAPACITY, spans=spans)
    assert plain["tellings"]["struck"] > 8.0 and share_held(plain["struck"], 4.0) == 0.0


def test_a_fact_the_reads_fetch_by_its_parts_is_held_and_an_unread_one_leaves():
    spans = (0.5, 6.0)
    deep, left = ch.use(0, "strength", CAPACITY, spans=spans)
    by_name, by_part, never = (deep[6.0][k] for k in range(3))
    assert by_part.tolist() == [1.0, 1.0] and never[0] == 0.0 and np.isinf(left[1])
    # asked by its name alone the read is shared with the slots of the asking moments
    assert left[2] <= left[0] < 6.0
    plain, gone = ch.use(0, "plain", CAPACITY, spans=spans)
    assert plain[6.0][:, 0].tolist() == [0.0, 0.0, 0.0] and (gone <= 1.0).all()


def test_the_bank_gives_up_recent_moments_for_the_ones_it_keeps_and_never_its_newest_half():
    for familiar in (0.0, 0.25):
        plain = ch.recent(0, "plain", CAPACITY, life=6.0, familiar=familiar)
        deep = ch.recent(0, "strength", CAPACITY, life=6.0, familiar=familiar)
        assert plain["recent held"] == 1.0 and plain["oldest slot"] < 1.0
        assert 0.5 <= deep["recent held"] < 1.0 and deep["oldest slot"] > 1.0
        assert plain["kept slots"] == 0.0
    # two facts told again at every eighth moment or so pass the kept depth within the life
    deep = ch.recent(0, "strength", CAPACITY, life=6.0, familiar=0.25, known=2)
    assert deep["kept slots"] == 2.0


def test_the_taught_reading_the_paid_life_and_the_timing_run():
    answers = ch.taught(0, "strength", 16, lessons=12, gaps=(0.25,), trials=2)
    assert set(answers) == {0.25} and 0.0 <= answers[0.25] <= 1.0
    for arm in ch.ARMS:
        right = ch.paid(0, arm, 3, episodes=4, capacity=8)
        assert right.shape == (4,) and set(right.tolist()) <= {0.0, 1.0}
    assert ch.moment_time("strength", 16, moments=20) > 0.0
