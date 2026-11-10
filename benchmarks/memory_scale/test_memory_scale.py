"""The memory-scale instruments (cadence#288): the bank reads, writes, keeps and revises as it
says, the references answer what they say, and a short run of each instrument holds its shape."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(name: str):
    key = f"memory_scale_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


bk = _load("bank")
wd = _load("world")
rf = _load("references")
rt = _load("retrieval")
rn = _load("retention")
ct = _load("cost")
us = _load("used")
lb = _load("library")


def _filled(n=400, width=64, **settings):
    rng = np.random.default_rng(0)
    recs = wd.records(rng, n, width, width)
    bank = bk.Bank(width, width, n, **settings)
    slots = [bank.write(recs.address[i], recs.content[i])[0] for i in range(n)]
    return recs, bank, np.asarray(slots)


# -- the world


def test_records_come_in_families_at_the_declared_cosine_and_cues_do_what_they_say():
    rng = np.random.default_rng(1)
    recs = wd.records(rng, 800, 128, 128)
    np.testing.assert_allclose(np.linalg.norm(recs.address, axis=1), 1.0)
    assert (recs.family >= 0).sum() == 800 // 4 and len(recs.themes) == 25
    rows = recs.members(3)
    assert len(rows) == wd.FAMILY
    inside = recs.address[rows] @ recs.address[rows].T
    assert abs(inside[np.triu_indices(wd.FAMILY, 1)].mean() - wd.SHARE) < 0.03
    lone = recs.address[recs.singletons[:50]]
    assert abs((lone @ lone.T)[np.triu_indices(50, 1)]).max() < 0.5
    a = recs.address[0]
    np.testing.assert_array_equal(wd.cue(rng, a, "exact"), a)
    noisy = [float(wd.cue(rng, a, "noisy") @ a) for _ in range(200)]
    assert abs(np.mean(noisy) - 1 / np.sqrt(2)) < 0.03
    partial = wd.cue(rng, a, "partial")
    assert (partial == 0).sum() == 64 and np.array_equal(partial[partial != 0], a[partial != 0])
    assert (recs.family[wd.targets(rng, recs, "similar", 40)] >= 0).all()
    assert (recs.family[wd.targets(rng, recs, "noisy", 40)] < 0).all()
    cue, wanted = wd.head_trial(rng, recs, "superposed", 4)
    assert len(wanted) == 4 and (recs.address[wanted] @ cue > 0.3).all()
    cue, wanted = wd.head_trial(rng, recs, "family", 4)
    assert len(wanted) == wd.FAMILY and (recs.address[wanted] @ cue > 0.8).all()
    sparse = wd.codes(rng, 5, 100, "sparse")
    assert ((sparse > 0).sum(axis=1) == 10).all() and (sparse >= 0).all()


def test_a_stream_uses_its_facts_by_rank_and_drift_reshuffles_them():
    stream = wd.Stream(facts=400, moments=40_000, seed=3)
    again = wd.Stream(facts=400, moments=40_000, seed=3)
    np.testing.assert_array_equal(stream.fact, again.fact)
    counts = np.bincount(stream.fact, minlength=400)
    assert counts[stream.top()].mean() > 8 * counts[stream.bottom()].mean()
    assert counts[stream.order[0]] > counts[stream.order[10]] > counts[stream.order[200]]
    assert 0.03 < stream.changed.mean() < 0.07
    drift = wd.Stream(facts=400, moments=40_000, seed=3, drift=True)
    assert not np.array_equal(drift.top(late=True), drift.top(late=False))
    late = np.bincount(drift.fact[20_000:], minlength=400)
    assert late[drift.top(late=True)].mean() > 4 * late[drift.top(late=False)].mean()


# -- the bank


def test_a_record_reads_back_whole_by_its_key_with_and_without_the_coarse_stage():
    for settings in ({"candidates": 0}, {"candidates": 16, "expansion": 512, "winners": 16}):
        recs, bank, slots = _filled(temperature=0.05, **settings)
        assert bank.coarse == bool(settings["candidates"]) and bank.count == 400
        for i in (0, 57, 399):
            read = bank.read(recs.address[i], credit=False)
            assert read.first() == slots[i]
            np.testing.assert_allclose(bank.value(read), recs.content[i], atol=0.05)
            if not recs.family[i] >= 0:
                assert read.mass(int(slots[i])) > 0.95


def test_the_fine_stage_over_candidates_is_the_dense_read_restricted_to_them():
    recs, coarse, _ = _filled(temperature=0.1, candidates=32, expansion=512, winners=16)
    _, dense, _ = _filled(temperature=0.1, candidates=0)
    cue = wd.cue(np.random.default_rng(5), recs.address[11], "noisy")
    a, b = coarse.read(cue, credit=False), dense.read(cue, credit=False)
    assert len(a.slots) == 32 and len(b.slots) == 400
    full = np.zeros(400)
    full[b.slots] = b.weights[0]
    np.testing.assert_allclose(a.weights[0], full[a.slots] / full[a.slots].sum(), atol=1e-12)


def test_the_address_lists_hold_exactly_the_slots_whose_address_holds_each_unit():
    rng = np.random.default_rng(2)
    bank = bk.Bank(24, 4, 60, candidates=8, expansion=128, winners=8, revise=0.9)
    for _ in range(500):  # replacements and revisions churn the lists
        key = wd.codes(rng, 1, 24)[0]
        slot, _ = bank.write(key, np.zeros(4))
        bank.moment()
        if rng.random() < 0.3:
            again, revised = bank.write(key, np.ones(4))
            assert revised and again == slot
    assert bank.count == 60
    listed = {u: set(bank.lists[u, : bank.counts[u]].tolist()) for u in range(bank.expansion)}
    for slot in range(bank.count):
        units = bank.address(bank.keys[slot])
        assert set(bank.units[slot][bank.units[slot] >= 0].tolist()) == set(units.tolist())
        for u in units:
            assert slot in listed[int(u)]
            assert bank.lists[u, bank.places[slot][list(bank.units[slot]).index(u)]] == slot
    assert sum(len(v) for v in listed.values()) == int((bank.units[: bank.count] >= 0).sum())


def test_a_key_told_again_revises_its_record_and_a_similar_key_takes_its_own_slot():
    rng = np.random.default_rng(4)
    bank = bk.Bank(64, 8, 32, candidates=0, temperature=0.0)
    key = wd.codes(rng, 1, 64)[0]
    slot, revised = bank.write(key, np.eye(8)[0])
    assert not revised and bank.count == 1
    again, revised = bank.write(key, np.eye(8)[1])
    assert revised and again == slot and bank.count == 1
    np.testing.assert_array_equal(bank.value(bank.read(key)), np.eye(8)[1])
    assert bank.deep[slot] == 3.0  # two writes and one read; no moment has passed
    sibling = wd.unit(0.8 * key + 0.6 * wd.codes(rng, 1, 64)[0])  # cosine about .8
    other, revised = bank.write(sibling, np.eye(8)[2])
    assert not revised and other != slot and bank.count == 2
    appended = bk.Bank(64, 8, 32, candidates=0, revise=0.0)
    appended.write(key, np.eye(8)[0])
    assert appended.write(key, np.eye(8)[1]) == (1, False)


def test_a_full_bank_replaces_the_idle_shallow_slot_and_keeps_the_used_and_the_deep():
    rng = np.random.default_rng(6)
    keys = wd.codes(rng, 40, 64)
    bank = bk.Bank(64, 4, 8, candidates=0, temperature=0.0, idle=4.0, hold=1e9)
    for i in range(8):
        bank.write(keys[i], np.zeros(4))
        bank.moment()
    for _ in range(12):  # slot 2 becomes deep
        bank.read(keys[2])
    for i in range(8, 30):  # slot 5 stays in use; new records keep arriving
        bank.read(keys[5])
        slot, _ = bank.write(keys[i], np.zeros(4))
        assert slot not in (2, 5)
        bank.moment()
    assert bank.faded[5] < 1e-3 and bank.deep[2] > 12
    for i in range(30, 40):  # left alone, the deep slot outlasts every shallow one written since
        slot, _ = bank.write(keys[i], np.zeros(4))
        assert slot != 2
        bank.moment()
    ring = bk.Bank(64, 4, 8, candidates=0, fade=False)
    order = [ring.write(keys[i], np.zeros(4))[0] for i in range(20)]
    assert order == [i % 8 for i in range(20)]
    recent_only = bk.Bank(64, 4, 8, candidates=0, depth=float("inf"), idle=4.0)
    for i in range(8):
        recent_only.write(keys[i], np.zeros(4))
        recent_only.moment()
    assert recent_only.write(keys[20], np.zeros(4))[0] == 0  # the slot idle longest


def test_a_probe_leaves_the_traces_and_a_read_deepens_what_it_attends():
    recs, bank, slots = _filled(n=50, candidates=0, temperature=0.05)
    for _ in range(3):
        bank.moment()
    assert (bank.deep[:50] < 1.0).all() and (bank.faded[:50] > 0).all()  # the depth leaks
    deep, faded = bank.deep.copy(), bank.faded.copy()
    bank.read(recs.address[7], credit=False)
    np.testing.assert_array_equal(bank.deep, deep)
    np.testing.assert_array_equal(bank.faded, faded)
    assert not bank.recent.any()  # a write marks no recent use
    read = bank.read(recs.address[7])
    at = int(slots[7])
    assert bank.deep[at] == pytest.approx(deep[at] + read.mass(at))
    assert bank.recent[at] == pytest.approx(read.mass(at))
    assert bank.faded[at] == pytest.approx(faded[at] * (1 - read.mass(at)))


def test_heads_read_in_turn_whole_at_temperature_zero_and_by_shares_above_it():
    rng = np.random.default_rng(7)
    recs = wd.records(rng, 256, 64, 64)
    hard = bk.Bank(64, 64, 256, temperature=0.0, candidates=0)
    soft = bk.Bank(64, 64, 256, temperature=0.05, candidates=0)
    for i in range(256):
        hard.write(recs.address[i], recs.content[i])
        soft.write(recs.address[i], recs.content[i])
    cue, members = wd.head_trial(rng, recs, "family", 4)
    read = hard.read(cue, heads=4, credit=False)
    firsts = [read.first(h) for h in range(4)]
    assert len(set(firsts)) == 4 and set(firsts) <= set(members.tolist())
    assert all(w.sum() == 1.0 and w.max() == 1.0 for w in read.weights)
    shares = soft.read(cue, heads=4, credit=False)
    total = np.sum(shares.weights, axis=0)
    assert total.max() <= 1.0 + 1e-9  # no slot is read more than once in full
    np.testing.assert_allclose([w.sum() for w in shares.weights], 1.0)
    assert hard.read(cue, heads=2, credit=False).weights[1].sum() == 1.0
    empty = bk.Bank(8, 8, 4)
    assert empty.read(np.ones(8), heads=2).first(1) == -1


def test_bank_settings_are_checked():
    with pytest.raises(ValueError, match="positive"):
        bk.Bank(0, 4, 8)
    with pytest.raises(ValueError, match="depth"):
        bk.Bank(4, 4, 8, depth=0.0)
    with pytest.raises(ValueError, match="idle"):
        bk.Bank(4, 4, 8, idle=1.0)
    with pytest.raises(ValueError, match="hold"):
        bk.Bank(4, 4, 8, hold=0.5)
    assert bk.Bank(4, 4, 8).hold == 64.0 and bk.Bank(4, 4, 32).idle == 2.0
    with pytest.raises(ValueError, match="winners"):
        bk.Bank(4, 4, 80, candidates=8, expansion=16, winners=32)
    assert bk.Bank(4, 4, 8, candidates=64).coarse is False  # every slot fits the fine stage


# -- the references


def test_exact_returns_the_nearest_records_in_order():
    rng = np.random.default_rng(8)
    keys = wd.codes(rng, 300, 32)
    store = rf.Exact(keys)
    scores = keys @ keys[17]
    np.testing.assert_array_equal(store.top(keys[17], 5), np.argsort(-scores)[:5])
    assert store.top(keys[17])[0] == 17 and len(store.top(keys[17], 999)) == 300


def test_each_cache_evicts_by_its_policy_and_revises_in_place():
    lru, fifo, lfu = (rf.Cache(2, p, facts=10) for p in ("lru", "fifo", "lfu"))
    for cache in (lru, fifo, lfu):
        assert cache.tell(0, 0) and cache.tell(1, 0)
        assert cache.ask(0, 0) and cache.ask(0, 0)  # fact 0 is used, fact 1 is not
        assert not cache.tell(0, 1) and cache.holds(0, 1) and not cache.holds(0, 0)
        assert cache.tell(2, 0) and cache.writes == 3
    assert lru.holds(0) and not lru.holds(1)  # the least recently used went
    assert not fifo.holds(0) and fifo.holds(1)  # the first in went
    assert lfu.holds(0) and not lfu.holds(1)  # the least used went
    assert not lru.ask(1, 0) and not lru.ask(0, 0) and lru.ask(0, 1)
    random = rf.Cache(3, "random", facts=50, seed=1)
    for fact in range(40):
        random.tell(fact, 0)
    assert random.held == 3 and (random.version >= 0).sum() == 3
    assert sorted(random.members) == np.flatnonzero(random.version >= 0).tolist()
    with pytest.raises(ValueError, match="policy"):
        rf.Cache(2, "newest", facts=4)


# -- the instruments


def test_the_library_arm_writes_one_slot_per_record_and_reads_them_by_address():
    rng = np.random.default_rng(9)
    recs = wd.records(rng, 40, 32, 32)
    slots = lb.Slots(0, 32, 32, 40)
    for i in range(40):
        slots.write(recs.address[i], recs.content[i])
    recall = slots.recall
    assert recall.slots == 41 and slots.slot(0) == 1 and slots.slot(39) == 40
    np.testing.assert_array_equal(recall.keys[0, slots.slot(5), :32], recs.address[5])
    np.testing.assert_array_equal(recall.values[0, slots.slot(5), 32:], recs.content[5])
    p = slots.read(recs.address[5])[0]
    assert int(np.argmax(p)) == slots.slot(5) and p.sum() == pytest.approx(1.0)
    two = lb.Slots(0, 32, 32, 40, heads=2)
    for i in range(40):
        two.write(recs.address[i], recs.content[i])
    first, second = two.read_heads(recs.address[5], 2)
    np.testing.assert_allclose(first, two.read(recs.address[5])[0])
    assert (first + second).max() <= 1.0 + 1e-9


def test_a_retrieval_cell_scores_every_arm_and_the_exact_store_is_the_ceiling():
    rows = rt.run_cell({
        "founder": 0, "n": 200, "width": 64, "trials": 40, "head_trials": 10,
        "arms": ["exact", "library",
                 {"name": "bank", "settings": {"temperature": 0.0, "candidates": 32,
                                               "expansion": 512, "winners": 16}}],
    })
    by = {(r["arm"], r["cue"], r["heads"]): r for r in rows}
    assert {r["arm"] for r in rows} == {"exact", "library", "bank"}
    assert by["exact", "exact", 1]["found"] == 1.0
    assert by["library", "exact", 1]["first"] == 1.0  # the content read at birth
    assert by["library", "exact", 1]["found"] < 0.5  # spread at a new brain's temperature
    assert by["bank", "exact", 1]["found"] == 1.0 and by["bank", "exact", 1]["held"] == 1.0
    for cue in wd.CUES:
        assert by["bank", cue, 1]["first"] <= by["exact", cue, 1]["first"] + 1e-9
        assert by["bank", cue, 1]["candidate"] >= by["bank", cue, 1]["first"]
    assert by["bank", "family", 4]["found"] == 1.0
    assert by["library", "family", 4]["relevant"] <= 0.5  # soft heads share one first choice
    plan = {"founders": [0, 1], "n": [100, 1000], "arms": ["exact", {"name": "b", "max_n": 100}]}
    todo = rt.cells(plan)
    assert len(todo) == 4 and todo[0]["n"] == 1000 and todo[0]["arms"] == ["exact"]


def test_the_ring_bank_reads_as_the_first_in_first_out_cache_and_fading_keeps_more():
    spec = {"founder": 1, "budget": 32, "moments_per_slot": 40}
    ring = rn.run_cell({**spec, "arm": {"name": "ring", "settings": {
        "temperature": 0.0, "candidates": 0, "fade": False}}})
    fifo = rn.run_cell({**spec, "arm": "fifo"})
    assert [r["writes"] for r in ring] == [r["writes"] for r in fifo]
    assert [r["hits"] for r in ring] == [r["hits"] for r in fifo]
    assert [r["top"] for r in ring] == [r["top"] for r in fifo]
    fade = rn.run_cell({**spec, "arm": {"name": "fade", "settings": {
        "temperature": 0.0, "candidates": 0}}})
    assert len(fade) == rn.CHECKPOINTS and fade[-1]["moment"] == 32 * 40
    assert fade[-1]["writes"] < fifo[-1]["writes"] and fade[-1]["hits"] > fifo[-1]["hits"]
    assert fade[-1]["stale"] == 0.0
    append = rn.run_cell({**spec, "arm": {"name": "append", "settings": {
        "temperature": 0.0, "candidates": 0, "revise": 0.0}}})
    assert append[-1]["hits"] < fade[-1]["hits"]  # copies crowd the slots
    drift = rn.run_cell({**spec, "drift": True, "arm": "lru"})
    assert np.isnan(drift[0]["left"]) and 0.0 <= drift[-1]["left"] <= 1.0
    assert len(rn.cells({"founders": [0], "budgets": [8, 64], "drift": [False, True],
                         "arms": ["lru", {"name": "x", "max_budget": 8}]})) == 6


def test_a_cost_cell_times_every_arm():
    rows = ct.run_cell({
        "founder": 0, "n": 64, "width": 16, "acts": 5, "reads": 10,
        "arms": ["moment", "exact", {"name": "bank", "settings": {"candidates": 0}}],
    })
    by = {r["arm"]: r for r in rows}
    assert by["moment"]["slots"] == 65 and by["moment"]["rest_ms"] > 0
    assert by["moment"]["map_bytes"] == 8 * (32 * 32 + 16 * 32 + 32)
    assert by["bank"]["read_ms"] > 0 and by["bank"]["write_ms"] > 0
    assert by["exact"]["bytes_per_record"] == 8 * 16


def test_the_logit_terms_sum_to_the_read_and_a_question_writes_no_slot():
    import cadence as cd

    brain = cd.Brain.compose(inputs=16 + us.TOKENS, actions=us.TOKENS, modules=(16,), seed=0,
                             backend="cpu", episodic=False, recall_capacity=41)
    rng = np.random.default_rng(10)
    report = us.nursery(brain, rng, 16, rounds=2)
    assert 0.0 <= report["nursery"] <= 1.0
    row = us.read_at(brain, rng, 16, 40, questions=6)
    assert brain.recall.slots == 41 and brain.recall._private is False
    assert set(row["margins"]) == set(us.TERMS) and 0.0 <= row["used"] <= 1.0
    x = us.question(wd.codes(rng, 1, 16))
    parts = us.logit_terms(brain, x)
    brain.recall.stimulate(np.zeros((1, brain.connectome.n)), brain.observation_code(x))
    logits = sum(parts[name][0] for name in us.TERMS)
    p = np.exp(logits - logits.max())
    np.testing.assert_allclose(p / p.sum(), brain.recall.attended_heads[0][0], atol=1e-9)


def test_the_transformer_reference_reads_a_window_and_scores_by_decoding():
    pytest.importorskip("torch")
    nt = _load("nets")
    rng = np.random.default_rng(11)
    tokens, contents, wanted, asked = nt.batch(rng, 12, 3, 16, 16, ("exact",))
    assert tokens.shape == (3, 13, 33) and contents.shape == (3, 12, 16)
    assert (tokens[:, :12, -1] == 0).all() and (tokens[:, 12, -1] == 1).all()
    for b in range(3):
        np.testing.assert_allclose(tokens[b, 12, :16], tokens[b, wanted[b], :16])
        assert not tokens[b, 12, 16:32].any()
    rows = nt.run_cell({"founder": 0, "n": 12, "width": 16, "steps": 4, "rows": 2, "model": 16,
                        "trials": 4, "read_n": [20]})
    assert [r["n"] for r in rows] == [12] * 5 + [20] * 5 and rows[0]["taught_n"] == 12
    assert all(0.0 <= r["right"] <= 1.0 for r in rows)


def test_a_moment_fades_every_slot_by_the_declared_law():
    rng = np.random.default_rng(12)
    for depth in (2.0, float("inf")):
        bank = bk.Bank(8, 2, 300, candidates=0, revise=0.0, depth=depth)
        for _ in range(300):
            bank.write(rng.standard_normal(8), np.zeros(2))
        bank.recent[:] = rng.random(300) * 0.05
        bank.deep[:] = rng.random(300) * 6
        bank.faded[:] = rng.random(300)
        recent, deep, faded = bank.recent.copy(), bank.deep.copy(), bank.faded.copy()
        bank.moment()
        law = 1.0 if not np.isfinite(depth) else 1.0 + (deep / depth) ** bk.ORDER
        idle = 1.0 / (1.0 + (recent / bk.FLOOR) ** 2)
        np.testing.assert_allclose(bank.faded, faded + idle / law, rtol=1e-12)
        np.testing.assert_allclose(bank.recent, recent * (1 - 1 / bank.idle), rtol=1e-12)
        np.testing.assert_allclose(bank.deep, deep * (1 - 1 / bank.hold), rtol=1e-12)
    ring = bk.Bank(8, 2, 4, candidates=0, fade=False)
    ring.write(np.ones(8), np.zeros(2))
    ring.moment()
    assert ring.faded[0] == 0.0 and ring.deep[0] == 1.0


def test_the_address_units_are_used_about_equally():
    bank = bk.Bank(32, 2, 4000, candidates=64, expansion=512, winners=16, revise=0.0)
    rng = np.random.default_rng(13)
    np.testing.assert_allclose(np.linalg.norm(bank.projection, axis=0), 1.0)
    for key in wd.codes(rng, 4000, 32):
        bank.write(key, np.zeros(2))
    assert bank.counts.sum() == 4000 * 16
    assert bank.counts.max() < 2 * bank.counts.mean() and bank.lists.shape[1] < 2.5 * 125


def test_the_declared_plans_parse_into_cells_on_the_declared_founders():
    import json

    plans = {p.stem: json.loads(p.read_text()) for p in (HERE / "protocol").glob("*.json")}
    assert set(plans) == {"retrieval", "retrieval-wide", "used", "retention", "cost",
                          "cost-width", "cost-device", "nets"}
    assert all(set(plan["founders"]) <= set(range(2880, 2886)) for plan in plans.values())
    assert len(rt.cells(plans["retrieval"])) == 24 and len(rt.cells(plans["retrieval-wide"])) == 9
    assert len(us.cells(plans["used"])) == 24
    todo = rn.cells(plans["retention"])
    assert len(todo) == 312 and sum(c["budget"] == 10_000 for c in todo) == 72
    assert len(ct.cells(plans["cost"])) == 12 and len(ct.cells(plans["cost-width"])) == 4
    candidate = next(a for a in plans["retrieval"]["arms"] if a != "exact"
                     and not isinstance(a, str) and a["name"] == "bank")["settings"]
    bank = bk.Bank(64, 64, 1000, **candidate)
    assert (bank.temperature, bank.candidates, bank.expansion, bank.winners) == (0.0, 256, 8192, 64)
    assert (bank.depth, bank.hold, bank.idle, bank.revise) == (2.0, 8000.0, 62.5, 0.9)
    for name in ("temperature", "candidates", "expansion", "winners", "depth", "hold", "idle"):
        assert getattr(bk.Bank(64, 64, 1000), name) == getattr(bank, name)  # the defaults


def test_the_plan_read_after_the_declaration_names_the_library_founder():
    import json

    plan = json.loads((HERE / "after" / "cost.json").read_text())
    assert len(ct.cells(plan)) == 12
    arms = {a["name"]: a["settings"] for a in plan["arms"] if not isinstance(a, str)}
    memory = bk.Bank(64, 64, 10_000, **arms["memory"])
    declared = bk.Bank(64, 64, 10_000, **arms["bank"])
    assert memory.direct == 16_384 and memory.coarse is False  # every slot scored
    assert declared.direct == 0 and declared.coarse is True  # the declared arm keeps its address
    assert bk.Bank(64, 64, 100_000, **arms["memory"]).coarse is True
    assert bk.Bank(64, 64, 100, **arms["memory-float32"]).keys.dtype == np.float32


def test_the_declared_receipts_hold_the_verdicts_the_readme_states():
    sm = _load("summarize")
    gates = sm.gates(HERE / "results")
    passed = {name for name, gate in gates.items() if gate["passes"]}
    assert passed == {"R1", "R2", "R3", "U1", "K1", "K2", "K3", "writes", "C2"}
    assert gates["C1"]["passes"] is False and gates["C1"]["founders"] == 0
    assert all(gates[name]["founders"] == 6 for name in ("R1", "R2", "R3", "U1", "K1", "K2", "K3"))
    assert gates["R2"]["values"] == [1.0, 1.0, 1.0, 0.999, 1.0, 1.0]
    assert gates["K1"]["values"] == [0.98, 0.965, 0.97, 0.9625, 0.9675, 0.9725]
    assert min(gates["K2"]["values"]) >= 0.97 and gates["writes"]["least"] >= 14
    assert max(gates["C2"]["values"]) < min(gates["C2"]["budget"])
    assert min(gates["C1"]["values"]) > 2 * max(gates["C1"]["budget"])
    readme = (HERE / "README.md").read_text()
    for cell in (".963 to .980", ".971 to .981", ".782 to .823", ".986 to .997", ".245 to .251"):
        assert cell in readme
    retrieval = sm.rows(HERE / "results" / "retrieval.jsonl")
    assert len(retrieval) == 24 * 66 and {r["founder"] for r in retrieval} == set(sm.FOUNDERS)
    library = sm.by_founder(retrieval, "first", arm="library", n=10_000, cue="exact", heads=1)
    spread = sm.by_founder(retrieval, "mass", arm="library", n=10_000, cue="exact", heads=1)
    assert library == [1.0] * 6 and max(spread) < 0.01  # the right first choice in a crowd
    assert "| `fade`, the candidate | .963 to .980 |" in readme
    assert sm.span([0.963, 0.98]) == "0.963 to 0.980" and sm.span([1.0, 1.0], 2) == "1.00"


# -- the second round: keys made by ``cadence.Message``


def test_a_fact_s_key_and_value_are_what_a_message_hears():
    mn = _load("meanings")
    lexicon = mn.Lexicon(nouns=32, verbs=8)
    assert (lexicon.forms, lexicon.abouts, lexicon.inputs) == (64, 512, 64 + 8 + 32 + 8 + 3)
    assert mn.Lexicon().abouts == 131_072 and mn.fan_in(mn.Lexicon()) == 821
    agent = lexicon.noun(np.array([3, 35]), "AGENT")  # one stem, singular and plural
    assert agent.sum(axis=1).tolist() == [2.0, 3.0] and agent[1, lexicon.inputs - 3] == 1.0
    assert lexicon.noun_word(np.array([35]))[0, 35] == 1.0
    assert lexicon.act(np.array([5])).sum() == 1
    coder = mn.Coder(lexicon, 280, seed=3)
    assert coder.message.to_dict()["units"] == 280 and coder.message.to_dict()["phases"] == 4
    assert coder.message.separator.fan_in == mn.fan_in(lexicon) == round(2 * 115 * 16 / 70)
    rng = np.random.default_rng(0)
    found = mn.facts(rng, coder, 300)
    assert len(set(zip(found.agent.tolist(), found.verb.tolist(), strict=True))) == 300
    moments = [lexicon.act(found.verb), lexicon.noun(found.agent, "AGENT")]
    key, _ = mn.held(coder.message, moments)
    _, state = mn.held(coder.message, [*moments, lexicon.noun(found.patient, "PATIENT")])
    np.testing.assert_array_equal(key, found.key)  # the union of the act's and the agent's codes
    np.testing.assert_array_equal(state, found.value)  # the held message of the whole fact
    other, _ = mn.held(coder.message, moments[::-1])  # one meaning in the other order
    np.testing.assert_array_equal(other, found.key)
    assert found.key.shape == (300, 280) and found.value.shape == (300, 1120)
    assert set(np.unique(found.key)) <= {0.0, 1.0} and found.key.sum(axis=1).max() <= 10
    assert np.mean(coder.said(found.value) == found.patient) > 0.9
    assert coder.said(np.zeros((1, 1120)))[0] == -1
    with pytest.raises(ValueError, match="agent-verb pairs"):
        mn.facts(rng, coder, 513)


def test_cues_of_a_held_meaning_do_what_they_say():
    mn = _load("meanings")
    coder = mn.Coder(mn.Lexicon(nouns=32, verbs=8), 280, seed=1)
    rng = np.random.default_rng(1)
    found = mn.facts(rng, coder, 200)
    key = found.key[7]
    np.testing.assert_array_equal(mn.cue(rng, coder, found, 7, "exact"), key)
    noisy = mn.cue(rng, coder, found, 7, "noisy")
    assert noisy.sum() == key.sum() and noisy @ key == key.sum() - round(mn.NOISE * key.sum())
    partial = mn.cue(rng, coder, found, 7, "partial")
    assert (partial <= key).all() and partial.sum() < key.sum()
    reworded = mn.cue(rng, coder, found, 7, "reworded")
    assert reworded.sum() > 0 and reworded @ key < key.sum() / 2  # a word's code is its own
    alone = coder.about_agent(found.agent[[7]])[0]
    assert alone.sum() <= 5 and alone @ key == alone.sum()  # the agent's code lies in the key
    siblings = np.flatnonzero((found.agent == found.agent[7]) & (np.arange(200) != 7))
    assert all(found.key[s] @ alone == alone.sum() for s in siblings)
    with pytest.raises(ValueError, match="cue"):
        mn.cue(rng, coder, found, 7, "similar")
    narrow = mn.Coder(mn.Lexicon(nouns=32, verbs=8), 70, seed=1, fan=16)
    assert narrow.agent.shape == (64, 70) and narrow.message.separator.fan_in >= 16


def test_a_meaning_key_cell_scores_the_exact_store_and_the_memory():
    mk = _load("meaning_keys")
    np.testing.assert_array_equal(mk.latest_best(np.array([1.0, 3.0, 3.0, 2.0]), 3), [2, 1, 3])
    rows = mk.run_cell({
        "founder": 0, "n": 400, "units": 280, "nouns": 32, "verbs": 16, "trials": 60,
        "head_trials": 20,
        "arms": ["exact", "memory", {"name": "memory-same", "settings": {"revise": 0.999}},
                 {"name": "memory-direct", "settings": {"direct": "all"}}],
    })
    by = {(r["arm"], r["cue"], r["heads"]): r for r in rows}
    assert {r["arm"] for r in rows} == {"exact", "memory", "memory-same", "memory-direct"}
    assert by["exact", "exact", 1]["found"] >= 0.95 and by["exact", "reworded", 1]["found"] <= 0.1
    assert by["exact", "noisy", 1]["single"] <= by["exact", "noisy", 1]["found"] + 1e-9
    assert by["memory", "exact", 1]["found"] >= 0.9 and by["memory", "exact", 1]["coarse"] is False
    assert by["memory-direct", "exact", 1]["settings"]["direct"] == 400
    # a key recognised only where it is the same key keeps every fact whose key is its own
    assert by["memory-same", "exact", 1]["held"] >= by["memory", "exact", 1]["held"]
    same = by["memory-same", "exact", 1]["held"]
    assert same == pytest.approx(rows[0]["distinct_keys"], abs=1e-4)
    assert ("exact", "agent", 4) in by and 0.0 <= by["memory", "agent", 4]["found"] <= 1.0
    assert rows[0]["keys"] == "message" and rows[0]["fan_in"] == round(2 * 131 * 16 / 70)


def test_retention_takes_facts_keyed_by_held_meanings():
    spec = {"founder": 1, "budget": 48, "moments_per_slot": 40, "keys": "message", "units": 280}
    memory = rn.run_cell({**spec, "arm": {"name": "memory", "settings": {"direct": None}}})
    lru = rn.run_cell({**spec, "arm": "lru"})
    assert memory[-1]["keys"] == "message" and memory[-1]["width"] == 280
    assert 0.9 < memory[-1]["distinct_keys"] <= 1.0 and memory[-1]["stale"] == 0.0
    assert memory[-1]["top"] >= lru[-1]["top"] - 0.1 and memory[-1]["revised_facts"] > 0
    append = rn.run_cell({**spec, "arm": {"name": "append",
                                          "settings": {"direct": None, "revise": 0.0}}})
    assert append[-1]["writes"] > memory[-1]["writes"]  # every statement takes a slot


def test_the_second_round_s_plans_parse_on_their_declared_founders():
    import json

    mk = _load("meaning_keys")
    plans = {p.stem: json.loads(p.read_text()) for p in (HERE / "protocol-2").glob("*.json")}
    assert set(plans) == {"cost", "meaning-keys", "meaning-retention", "meaning-retention-70"}
    assert plans["cost"]["founders"] == [2886, 2887, 2888]
    assert all(plans[name]["founders"] == list(range(2890, 2896)) for name in plans
               if name != "cost")
    assert len(ct.cells(plans["cost"])) == 12
    store = next(a for a in plans["cost"]["arms"] if not isinstance(a, str)
                 and a["name"] == "memory")["settings"]
    as_it_stands = bk.Bank(64, 64, 10_000, **store)
    assert as_it_stands.direct == 16_384 and as_it_stands.coarse is False
    todo = mk.cells(plans["meaning-keys"])
    assert len(todo) == 54 and {c["units"] for c in todo} == {280, 70}
    names = [[a if isinstance(a, str) else a["name"] for a in c["arms"]] for c in todo]
    assert all(("memory-direct" in n) == (c["n"] == 100_000)
               for n, c in zip(names, todo, strict=True))
    assert len(rn.cells(plans["meaning-retention"])) == 96
    assert plans["meaning-retention"]["units"] == 280
    assert plans["meaning-retention-70"]["units"] == 70


def test_the_floor_of_a_read_that_scores_every_slot_is_timed():
    rows = ct.run_cell({"founder": 0, "n": 64, "width": 16, "acts": 5, "reads": 10,
                        "arms": ["floor", "floor32"]})
    assert [r["arm"] for r in rows] == ["floor", "floor32"] and rows[0]["read_ms"] > 0
    assert rows[0]["bytes_per_record"] == 2 * rows[1]["bytes_per_record"]


def test_a_fact_told_under_its_whole_episode_s_key_keeps_its_old_copy_when_it_changes():
    import json

    spec = {"founder": 2, "budget": 48, "moments_per_slot": 40, "units": 280,
            "arm": {"name": "memory", "settings": {"direct": None}}}
    turn = rn.run_cell({**spec, "keys": "message"})
    whole = rn.run_cell({**spec, "keys": "message-whole"})
    assert turn[-1]["stale"] == 0.0 and whole[-1]["stale"] > 0.2
    assert whole[-1]["revised"] < turn[-1]["revised"] - 0.2
    assert whole[-1]["writes"] > turn[-1]["writes"]  # every change takes a slot
    plan = json.loads((HERE / "after" / "whole-keys.json").read_text())
    assert plan["keys"] == "message-whole" and plan["founders"] == [0, 1, 2]
    assert len(rn.cells(plan)) == 9


def test_the_second_declaration_s_receipts_hold_the_verdicts_the_readme_states():
    sm = _load("summarize")
    gates = sm.gates2(HERE / "results")
    passed = {name for name, gate in gates.items() if gate["passes"]}
    assert passed == {"C1b", "M1", "M4", "M5", "M6"}
    assert gates["C1b"]["founders"] == 3 and gates["M2"]["founders"] == 1
    assert gates["M3"]["founders"] == 3
    assert max(gates["C1b"]["values"]) < min(gates["C1b"]["budget"])
    assert max(gates["C1b"]["floor"]) < min(gates["C1b"]["values"])  # the floor lies below
    assert gates["M1"]["values"] == [0.998, 0.997, 0.996, 0.997, 0.997, 1.0]
    assert gates["M2"]["values"] == [0.867, 0.908, 0.88, 0.852, 0.863, 0.891]
    keys = sm.pick(sm.rows(HERE / "results" / "meaning-keys.jsonl"), units=280, fan_in=821)
    assert {r["founder"] for r in keys} == set(sm.SECOND)
    for n in (1000, 10_000, 100_000):  # the exact store misses with the memory: the key binds
        store = sm.by_founder(keys, "found", sm.SECOND, arm="memory", n=n, cue="noisy", heads=1)
        exact = sm.by_founder(keys, "found", sm.SECOND, arm="exact", n=n, cue="noisy", heads=1)
        assert len(store) == len(exact) == 6
        assert abs(float(np.mean(store)) - float(np.mean(exact))) < 0.02
    reworded = [r["found"] for r in keys if r["cue"] == "reworded" and r["arm"] == "memory"]
    assert max(reworded) <= 0.01
    apart = sm.by_founder(keys, "held", sm.SECOND, arm="memory", n=100_000, cue="exact", heads=1)
    same = sm.by_founder(keys, "held", sm.SECOND, arm="memory-same", n=100_000, cue="exact",
                         heads=1)
    assert all(b > a for a, b in zip(apart, same, strict=True))
    readme = (HERE / "README.md").read_text()
    for cell in (".852 to .908", ".961 to .991", ".968 to .978", ".084 to .093", ".593 to .707"):
        assert cell in readme
    whole = sm.rows(HERE / "results" / "dev-whole-keys.jsonl")
    ends = [max(sm.pick(whole, arm="memory", founder=f), key=lambda r: r["moment"])
            for f in (0, 1, 2)]
    assert all(r["stale"] > 0.5 and r["revised"] < 0.3 for r in ends)


# -- the third round: the brain writes and reads its own records


def test_the_brain_records_chamber_tells_asks_and_scores():
    br = _load("brain_records")
    world = br.World(0)
    assert world.inputs == 388 and world.turn == 387 and world.fan_in() == 177
    assert len(world.nursery) == br.NURSERY and not set(world.nursery) & set(world.pool)
    about = world.pool[:3]
    fact = world.episode(about, np.array([1, 2, 3]))
    question = world.episode(about, None)
    assert len(fact) == len(question) == 4 and fact[2][:, 387].all() and not question[3].any()
    for a, b in zip(fact[:3], question[:3], strict=True):
        np.testing.assert_array_equal(a, b)  # a question hears what its fact did, to the turn
    rows = br.run_cell({"founder": 0, "arm": "records", "n": [12], "rounds": 4, "questions": 2,
                        "capacity": 64})
    row = rows[0]
    assert row["records"] == [12] * br.STREAMS and row["counts"]["revisions"] >= br.STREAMS
    for kind in ("unrevised", "revised"):
        assert row[kind]["questions"] == 2 * br.STREAMS and 0 <= row[kind]["used"] <= 1
    assert row["unrevised"]["found"] == 1.0 and row["revised"]["found"] == 1.0
    assert row["revised"]["stale"] == 0.0
    plain = br.run_cell({"founder": 0, "arm": "plain", "n": [12], "rounds": 2, "questions": 1})
    assert "records" not in plain[0] and plain[0]["unrevised"]["found"] == 0.0


def test_the_cost_cell_fills_records_with_the_keys_the_brain_s_message_gives():
    import cadence as cd

    br = _load("brain_records")
    world = br.World(1)
    brain = br.make_brain(world, 1, 8)
    about = world.pool[:4]
    episode = world.episode(about, np.array([0, 1, 2, 3]))
    brain.reset()
    for x in episode:
        brain.act(x, greedy=True)
    fields = ("inputs", "phases", "units", "winners", "fan_in", "seed")
    coder = cd.Message(**{k: brain.message.to_dict()[k] for k in fields})
    coder.reset(4)
    coder.hear(episode[0])
    coder.hear(episode[1])
    np.testing.assert_array_equal(coder.key(), brain.message.key())
    rows = br.cost_cell({"founder": 1, "n": [0, 20], "streams": 2, "episodes": 2})
    assert [r["n"] for r in rows] == [0, 20] and rows[0]["records"] is None
    assert rows[1]["records"][0] >= 19 and rows[1]["question_ms"] > 0


def test_the_third_declaration_s_plans_parse():
    import json

    br = _load("brain_records")
    plans = {p.stem: json.loads(p.read_text()) for p in (HERE / "protocol-3").glob("*.json")}
    assert set(plans) == {"records", "cost", "cost-device"}
    assert plans["records"]["founders"] == list(range(2900, 2906))
    assert plans["cost"]["founders"] == plans["cost-device"]["founders"] == [2906, 2907, 2908]
    todo = br.cells(plans["records"])
    assert len(todo) == 18 and {c["arm"] for c in todo} == {"records", "plain", "records_h2"}
    assert plans["cost-device"]["device"] == "cuda"


def test_the_third_declaration_s_receipts_hold_the_verdicts_the_readme_states():
    sm = _load("summarize")
    gates = sm.gates3(HERE / "results")
    assert {name for name, gate in gates.items() if gate["passes"]} == {
        "B1", "B2", "B3", "B4", "B5", "C3", "D1"}
    assert gates["B2"]["founders"] == 5 and min(gates["B2"]["values"]) == 0.98
    assert max(gates["C3"]["values"]) < 2 and max(gates["D1"]["values"]) < 2
    table = sm.rows(HERE / "results" / "third-records.jsonl")
    assert len(table) == 36 and {r["founder"] for r in table} == set(sm.THIRD)
    for row in sm.pick(table, arm="records"):
        for kind in ("unrevised", "revised"):
            assert abs(row[kind]["used"] - row[kind]["found"]) <= 0.005 + 1e-9
        assert row["revised"]["stale"] == 0.0
    assert all(r["unrevised"]["found"] == 0.0 for r in sm.pick(table, arm="plain"))
    readme = (HERE / "README.md").read_text()
    for cell in ("| B2 | found, unrevised facts, 10,000 records | .99 | .980 to 1.000, 5 of 6 |",
                 "1.82 to 1.83", "| one core | 16 | 10,000 | 11.675 | 27.043 |"):
        assert cell in readme
