"""``LongTermMemory`` (cadence#288): a record reads back by its key through the sparse address
and the recall read, several heads return different records, a full memory keeps what is used
and replaces what faded, a key told again revises its record, and a saved memory continues as
the one it was saved from."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.longterm import LongTermMemory, LongTermRead


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def codes(rng, n, width):
    return unit(rng.standard_normal((n, width)))


def filled(n=400, width=64, **settings):
    rng = np.random.default_rng(0)
    keys, values = codes(rng, n, width), codes(rng, n, width)
    memory = LongTermMemory(width, width, n, **settings)
    slots = np.array([memory.write(keys[i], values[i])[0] for i in range(n)])
    return keys, values, memory, slots


def test_the_class_is_exported_with_its_declared_founders():
    assert cd.LongTermMemory is LongTermMemory and cd.LongTermRead is LongTermRead
    assert {"LongTermMemory", "LongTermRead"} <= set(cd.__all__)
    memory = LongTermMemory(64, 8, 1000)
    assert memory.to_dict() == {
        "width": 64, "value_width": 8, "capacity": 1000, "temperature": 0.0, "candidates": 256,
        "expansion": 8192, "winners": 64, "depth": 2.0, "idle": 62.5, "hold": 8000.0,
        "revise": 0.9, "fade": True, "seed": 0, "direct": 16384, "dtype": "float64",
    }
    assert memory.count == 0 and memory.writes == 0
    assert LongTermMemory(**memory.to_dict()).to_dict() == memory.to_dict()
    # up to twice the address's units every slot is scored; past them the address picks
    assert memory.coarse is False and LongTermMemory(8, 8, 16_384).coarse is False
    assert LongTermMemory(8, 8, 16_385).coarse is True
    assert LongTermMemory(8, 8, 1000, direct=0).coarse is True
    assert LongTermMemory(8, 8, 200, direct=0).coarse is False  # 256 candidates hold every slot
    assert LongTermMemory(8, 8, 1000, direct=0, candidates=0).coarse is False
    assert LongTermMemory(8, 8, 32).idle == 2.0
    narrow = LongTermMemory(8, 8, 32, dtype="float32")
    assert narrow.keys.dtype == np.float32 and narrow.to_dict()["dtype"] == "float32"
    assert LongTermMemory(**narrow.to_dict()).values.dtype == np.float32
    assert narrow.nbytes < LongTermMemory(8, 8, 32).nbytes


def test_a_record_reads_back_whole_by_its_key_with_and_without_the_coarse_stage():
    for settings in ({"candidates": 0},
                     {"candidates": 16, "expansion": 512, "winners": 16, "direct": 0}):
        keys, values, memory, slots = filled(**settings)
        assert memory.coarse == bool(settings["candidates"]) and memory.count == 400
        for i in (0, 57, 399):
            read = memory.read(keys[i], credit=False)
            assert read.first() == slots[i] and read.mass(int(slots[i])) == 1.0
            np.testing.assert_array_equal(memory.value(read), values[i])
        noisy = unit(keys[5] + 0.4 * codes(np.random.default_rng(1), 1, 64)[0])
        assert memory.read(noisy, credit=False).first() == slots[5]
        partial = keys[9].copy()
        partial[::2] = 0.0
        assert memory.read(partial, credit=False).first() == slots[9]


def test_the_fine_stage_over_candidates_is_the_dense_read_restricted_to_them():
    keys, _, coarse, _ = filled(temperature=0.1, candidates=32, expansion=512, winners=16,
                                direct=0)
    _, _, dense, _ = filled(temperature=0.1, candidates=0)
    cue = unit(keys[11] + 0.5 * codes(np.random.default_rng(5), 1, 64)[0])
    a, b = coarse.read(cue, credit=False), dense.read(cue, credit=False)
    assert len(a.slots) == 32 and len(b.slots) == 400
    full = np.zeros(400)
    full[b.slots] = b.weights[0]
    np.testing.assert_allclose(a.weights[0], full[a.slots] / full[a.slots].sum(), atol=1e-12)
    np.testing.assert_allclose(b.weights[0].sum(), 1.0)
    logits = dense.keys @ cue / 0.1  # the recall memory's content read
    p = np.exp(logits - logits.max())
    np.testing.assert_allclose(b.weights[0], p / p.sum(), atol=1e-12)


def test_the_address_lists_hold_exactly_the_slots_whose_address_holds_each_unit():
    rng = np.random.default_rng(2)
    memory = LongTermMemory(24, 4, 60, candidates=8, expansion=128, winners=8, direct=0)
    np.testing.assert_allclose(np.linalg.norm(memory.projection, axis=0), 1.0)
    for _ in range(500):  # replacements and revisions churn the lists
        key = codes(rng, 1, 24)[0]
        slot, _ = memory.write(key, np.zeros(4))
        memory.moment()
        if rng.random() < 0.3:
            again, revised = memory.write(key, np.ones(4))
            assert revised and again == slot
    assert memory.count == 60
    listed = {u: set(memory.lists[u, : memory.counts[u]].tolist())
              for u in range(memory.expansion)}
    for slot in range(memory.count):
        units = memory.address(memory.keys[slot])
        assert set(memory.units[slot][memory.units[slot] >= 0].tolist()) == set(units.tolist())
        for u in units:
            assert slot in listed[int(u)]
            at = memory.places[slot][list(memory.units[slot]).index(u)]
            assert memory.lists[u, at] == slot
    assert sum(len(v) for v in listed.values()) == int((memory.units[: memory.count] >= 0).sum())


def test_a_key_told_again_revises_its_record_and_a_similar_key_takes_its_own_slot():
    rng = np.random.default_rng(4)
    memory = LongTermMemory(64, 8, 32, candidates=0)
    key = codes(rng, 1, 64)[0]
    slot, revised = memory.write(key, np.eye(8)[0])
    assert not revised and memory.count == 1
    again, revised = memory.write(key, np.eye(8)[1])
    assert revised and again == slot and memory.count == 1 and memory.writes == 2
    np.testing.assert_array_equal(memory.value(memory.read(key)), np.eye(8)[1])
    assert memory.deep[slot] == 3.0  # two writes and one read; no moment has passed
    sibling = unit(0.8 * key + 0.6 * codes(rng, 1, 64)[0])  # cosine about .8
    other, revised = memory.write(sibling, np.eye(8)[2])
    assert not revised and other != slot and memory.count == 2
    appended = LongTermMemory(64, 8, 32, candidates=0, revise=0.0)
    appended.write(key, np.eye(8)[0])
    assert appended.write(key, np.eye(8)[1]) == (1, False)


def test_a_full_memory_replaces_the_idle_shallow_slot_and_keeps_the_used_and_the_deep():
    rng = np.random.default_rng(6)
    keys = codes(rng, 40, 64)
    memory = LongTermMemory(64, 4, 8, candidates=0, idle=4.0, hold=1e9)
    for i in range(8):
        memory.write(keys[i], np.zeros(4))
        memory.moment()
    for _ in range(12):  # slot 2 becomes deep
        memory.read(keys[2])
    for i in range(8, 30):  # slot 5 stays in use; new records keep arriving
        memory.read(keys[5])
        slot, _ = memory.write(keys[i], np.zeros(4))
        assert slot not in (2, 5)
        memory.moment()
    assert memory.faded[5] < 1e-3 and memory.deep[2] > 12
    for i in range(30, 40):  # left alone, the deep slot outlasts every shallow one written since
        slot, _ = memory.write(keys[i], np.zeros(4))
        assert slot != 2
        memory.moment()
    ring = LongTermMemory(64, 4, 8, candidates=0, fade=False)
    assert [ring.write(keys[i], np.zeros(4))[0] for i in range(20)] == [i % 8 for i in range(20)]
    recent_only = LongTermMemory(64, 4, 8, candidates=0, depth=float("inf"), idle=4.0)
    for i in range(8):
        recent_only.write(keys[i], np.zeros(4))
        recent_only.moment()
    assert recent_only.write(keys[20], np.zeros(4))[0] == 0  # the slot idle longest


def test_a_moment_fades_every_slot_by_the_law_and_a_read_deepens_what_it_attends():
    rng = np.random.default_rng(12)
    for depth in (2.0, float("inf")):
        memory = LongTermMemory(8, 2, 300, candidates=0, revise=0.0, depth=depth)
        for _ in range(300):
            memory.write(rng.standard_normal(8), np.zeros(2))
        assert not memory.recent.any()  # a write marks no recent use
        memory.recent[:] = rng.random(300) * 0.05
        memory.deep[:] = rng.random(300) * 6
        memory.faded[:] = rng.random(300)
        recent, deep, faded = memory.recent.copy(), memory.deep.copy(), memory.faded.copy()
        memory.moment()
        law = 1.0 if not np.isfinite(depth) else 1.0 + (deep / depth) ** LongTermMemory.ORDER
        idle = 1.0 / (1.0 + (recent / LongTermMemory.FLOOR) ** 2)
        np.testing.assert_allclose(memory.faded, faded + idle / law, rtol=1e-12)
        np.testing.assert_allclose(memory.recent, recent * (1 - 1 / memory.idle), rtol=1e-12)
        np.testing.assert_allclose(memory.deep, deep * (1 - 1 / memory.hold), rtol=1e-12)
    keys, _, memory, slots = filled(n=50, candidates=0, temperature=0.05)
    memory.moment()
    deep, faded = memory.deep.copy(), memory.faded.copy()
    memory.read(keys[7], credit=False)  # a probe
    np.testing.assert_array_equal(memory.deep, deep)
    np.testing.assert_array_equal(memory.faded, faded)
    read = memory.read(keys[7])
    at = int(slots[7])
    assert memory.deep[at] == pytest.approx(deep[at] + read.mass(at))
    assert memory.recent[at] == pytest.approx(read.mass(at))
    assert memory.faded[at] == pytest.approx(faded[at] * (1 - read.mass(at)))


def test_heads_read_in_turn_whole_at_temperature_zero_and_by_shares_above_it():
    rng = np.random.default_rng(7)
    theme = codes(rng, 1, 64)[0]
    keys = codes(rng, 256, 64)
    family = np.arange(8)
    keys[family] = unit(np.sqrt(0.8) * theme + np.sqrt(0.2) * keys[family])
    hard = LongTermMemory(64, 4, 256, candidates=0)
    soft = LongTermMemory(64, 4, 256, candidates=0, temperature=0.05)
    for key in keys:
        hard.write(key, np.zeros(4))
        soft.write(key, np.zeros(4))
    read = hard.read(theme, heads=4, credit=False)
    firsts = [read.first(h) for h in range(4)]
    assert len(set(firsts)) == 4 and set(firsts) <= set(family.tolist())
    assert all(w.sum() == 1.0 and w.max() == 1.0 for w in read.weights)
    shares = soft.read(theme, heads=4, credit=False)
    assert np.sum(shares.weights, axis=0).max() <= 1.0 + 1e-9  # no slot is read twice in full
    np.testing.assert_allclose([w.sum() for w in shares.weights], 1.0)
    assert len({shares.first(h) for h in range(4)}) < 4  # soft heads share first choices
    one = LongTermMemory(8, 8, 4, candidates=0)
    one.write(np.ones(8), np.ones(8))
    alone = one.read(np.ones(8), heads=2)
    assert alone.first(0) == 0 and alone.first(1) == -1  # nothing is left for the second head
    assert LongTermMemory(8, 8, 4).read(np.ones(8)).first() == -1
    assert not one.value(LongTermMemory(8, 8, 4).read(np.ones(8))).any()


def test_a_saved_memory_continues_as_the_one_it_was_saved_from():
    rng = np.random.default_rng(8)
    keys, values = codes(rng, 300, 32), codes(rng, 300, 6)
    for settings in ({"candidates": 16, "expansion": 256, "winners": 12, "direct": 0},
                     {"candidates": 0}, {"candidates": 0, "fade": False}):
        first = LongTermMemory(32, 6, 80, seed=3, **settings)
        for i in range(150):
            first.read(keys[rng.integers(i + 1)])
            first.write(keys[i], values[i])
            first.moment()
        saved = first.arrays()
        second = LongTermMemory(**first.to_dict())
        second.restore(saved)
        assert (second.count, second.writes) == (first.count, first.writes)
        for i in range(150, 300):
            cue = keys[rng.integers(i)]
            a, b = first.read(cue, heads=2), second.read(cue, heads=2)
            np.testing.assert_array_equal(a.slots, b.slots)
            np.testing.assert_array_equal(a.weights[1], b.weights[1])
            assert first.write(keys[i], values[i]) == second.write(keys[i], values[i])
            first.moment()
            second.moment()
        for name, array in first.arrays().items():
            np.testing.assert_array_equal(array, second.arrays()[name], err_msg=name)
        assert saved["keys"] is not first.keys and first.nbytes == second.nbytes
        with pytest.raises(ValueError, match="hold"):
            second.restore({"keys": saved["keys"]})
        with pytest.raises(ValueError, match="another shape"):
            LongTermMemory(32, 6, 81, seed=3, **settings).restore(saved)


def test_settings_and_rows_are_checked():
    for bad, match in (
        ({"width": 0}, "width"), ({"capacity": True}, "capacity"),
        ({"temperature": -0.1}, "temperature"), ({"candidates": -1}, "candidates"),
        ({"revise": 1.5}, "revise"), ({"depth": 0.0}, "depth"), ({"idle": 1.0}, "idle"),
        ({"hold": 0.5}, "hold"), ({"expansion": 16, "winners": 32, "candidates": 8}, "winners"),
        ({"direct": -1}, "direct"), ({"expansion": 0}, "expansion"), ({"dtype": int}, "dtype"),
    ):
        settings = {"width": 16, "value_width": 4, "capacity": 80, **bad}
        with pytest.raises(ValueError, match=match):
            LongTermMemory(**settings)
    memory = LongTermMemory(16, 4, 8)
    with pytest.raises(ValueError, match="key"):
        memory.write(np.ones(15), np.ones(4))
    with pytest.raises(ValueError, match="value"):
        memory.write(np.ones(16), np.full(4, np.nan))
    with pytest.raises(ValueError, match="cue"):
        memory.read(np.ones((1, 16)))
    with pytest.raises(ValueError, match="heads"):
        memory.read(np.ones(16), heads=0)
    assert memory.count == 0 and memory.writes == 0  # a refused write changes nothing


def test_under_a_budget_the_memory_keeps_the_facts_used_often_and_revises_the_changed():
    """A short stream of the retention instrument's shape: facts used by a Zipf law, asked and
    then told, one in twenty changed before an occurrence."""
    rng = np.random.default_rng(9)
    facts, budget, moments = 240, 60, 6000
    address, content = codes(rng, facts, 64), codes(rng, facts, 16)
    weight = 1.0 / np.arange(1, facts + 1)
    order = rng.choice(facts, size=moments, p=weight / weight.sum())
    held = {}
    for fade in (True, False):
        memory = LongTermMemory(64, 16, budget, candidates=0, fade=fade)
        truth = content.copy()
        owner = np.full(budget, -1)
        change = np.random.default_rng(10)
        for f in order:
            if change.random() < 0.05:
                truth[f] = codes(change, 1, 16)[0]
            memory.read(address[f])
            slot, _ = memory.write(address[f], truth[f])
            owner[slot] = f
            memory.moment()
        top = np.arange(facts // 10)
        current = [
            owner[memory.read(address[f], credit=False).first()] == f
            and np.array_equal(memory.value(memory.read(address[f], credit=False)), truth[f])
            for f in top
        ]
        held[fade] = float(np.mean(current))
        assert len(set(owner.tolist())) == budget  # no fact holds two slots
    assert held[True] >= 0.95 and held[True] > held[False] + 0.1


def test_a_brain_answers_from_what_the_memory_beside_it_reads():
    """The use beside a brain: the memory's read of a moment's address is held on input
    coordinates of its own, and lessons teach the brain to answer from it."""
    rng = np.random.default_rng(11)
    width, tokens, streams = 16, 4, 8
    brain = cd.Brain.compose(inputs=width + 2 * tokens, actions=tokens, modules=(32,), seed=0,
                             backend="cpu", recall=False, episodic=False)

    def moment(address, token, read):
        x = np.zeros((len(address), width + 2 * tokens))
        x[:, :width] = address
        if token is not None:
            x[np.arange(len(address)), width + token] = 1.0
        x[:, width + tokens:] = read
        return x

    def episode(teach):
        memory = LongTermMemory(width, tokens, 64, candidates=0)
        address = codes(rng, streams * 4, width).reshape(streams, 4, width)
        token = rng.integers(tokens, size=(streams, 4))
        brain.reset()
        for f in range(4):
            read = np.stack([memory.value(memory.read(a)) for a in address[:, f]])
            brain.act(moment(address[:, f], token[:, f], read), greedy=True)
            for a, t in zip(address[:, f], token[:, f], strict=True):
                memory.write(a, np.eye(tokens)[t])
        asked = rng.integers(4, size=streams)
        rows = np.arange(streams)
        read = np.stack([memory.value(memory.read(a)) for a in address[rows, asked]])
        x = moment(address[rows, asked], None, read)
        if teach:
            brain.lesson(x, token[rows, asked])
        said = np.asarray(brain.act(x, greedy=True)).reshape(streams, -1)[:, 0]
        return float(np.mean(said == token[rows, asked]))

    for _ in range(150):
        episode(teach=True)
    assert np.mean([episode(teach=False) for _ in range(10)]) >= 0.9
