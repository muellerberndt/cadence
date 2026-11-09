"""The crossing chamber's rows, supports and readings (benchmarks/attention/crossing.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load() -> object:
    spec = importlib.util.spec_from_file_location("crossing_chamber", HERE / "crossing.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["crossing_chamber"] = module
    spec.loader.exec_module(module)
    return module


cr = _load()


def test_the_answer_is_the_value_of_the_fact_the_crossing_names() -> None:
    rows = cr.keyed(np.random.default_rng(0), n=64, think=True)
    for r in rows:
        assert r.k in (0, 1) and r.fact == r.c ^ r.k
        assert r.answer == r.values[r.fact] and r.values[0] != r.values[1]
        assert len(r.pre) == 5 and r.think
        heard = [int(np.argmax(x)) for x in r.pre]
        assert heard[0] == cr.NAMES[r.first] and heard[1] == r.values[r.first]
        assert heard[4] == cr.gr.OPENS[r.c]
        assert int(np.argmax(r.question)) == cr.gr.MATCH[cr.gr.OPENS[r.k]]
    assert {(r.c, r.k) for r in rows} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    direct = cr.keyed(np.random.default_rng(1), n=32, direct=1.0)
    assert all(r.k == -1 and int(np.argmax(r.question)) == cr.NAMES[r.fact] for r in direct)


def test_the_fact_slots_follow_the_order_witnessed() -> None:
    brain = cr.make_brain("Fd", 0)
    rows = cr.keyed(np.random.default_rng(2), n=8)
    for r in rows:
        first, second = cr.fact_slots(brain, r)
        start = 1 if brain.recall.boundary else 0
        assert {first, second} == {start, start + 2}
        assert (first == start) == (r.first == 0)


def test_a_frozen_cortex_moves_only_the_readout_and_the_reads_synapses() -> None:
    brain = cr.make_brain("Fd", 0)
    support = cr.Support(brain, ["frozen_cortex"])
    before = np.array(brain.learner.brain.efficacy)
    bias = np.array(brain.learner.brain.bias)
    rng = np.random.default_rng(0)
    work = cr.new_work()
    for _ in range(3):
        cr.teach(brain, cr.keyed(rng, direct=1 / 3, think=True), work, support)
    after = np.array(brain.learner.brain.efficacy)
    assert np.array_equal(after[support.mask], before[support.mask])
    assert np.array_equal(np.asarray(brain.learner.brain.bias)[support.neurons],
                          bias[support.neurons])
    assert not np.array_equal(after[~support.mask], before[~support.mask])


def test_a_unit_cortex_is_what_the_maps_read() -> None:
    brain = cr.make_brain("Fd", 0)
    support = cr.Support(brain, ["cortex_unit"])
    rows = cr.keyed(np.random.default_rng(3), n=4, think=True)
    cortex = cr.cortex_at_address(brain, rows, support)
    assert np.allclose(np.linalg.norm(cortex, axis=1), 1.0)


def test_a_newborn_cortex_holds_the_crossing_and_its_reading_is_bounded() -> None:
    brain = cr.make_brain("Fd", 0)
    out = cr.decode(brain, np.random.default_rng(1), cr.Support(brain), True)
    assert out["c"] > 0.9 and out["k"] > 0.9
    assert out["crossing"] > 0.9
    assert out["crossing_term"] > 0 and out["spread"] > 0


def test_a_life_reads_its_panels_and_leaves_the_brain_as_taught() -> None:
    row = cr.run_life({"arm": "Fd", "founder": 0, "batches": 2, "panel_batches": 1,
                       "decode": False})
    last = row["readings"][-1]
    assert last["batches"] == 2 and last["work"]["lessons"] == 2
    intact = last["panels"]["intact"]
    assert 0.0 <= intact["right"] <= 1.0 and set(intact["conditions"]) <= {
        "c0k0", "c0k1", "c1k0", "c1k1"}
    assert {"erase_relevant", "erase_other", "memory_off", "direct"} <= set(last["panels"])


def test_frozen_all_keeps_the_reads_synapses_into_the_cortex_as_well() -> None:
    brain = cr.make_brain("Fd", 0)
    cortex, every = cr.Support(brain, ["frozen_cortex"]), cr.Support(brain, ["frozen_all"])
    assert every.mask.sum() > cortex.mask.sum()
    assert np.array_equal(every.mask & cortex.mask, cortex.mask)
