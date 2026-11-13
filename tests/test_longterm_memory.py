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
        "cosine": False, "pairs": 0, "apart": 0,
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
                     {"candidates": 0}, {"candidates": 0, "fade": False},
                     {"candidates": 0, "cosine": True},
                     {"candidates": 16, "expansion": 256, "winners": 12, "direct": 0,
                      "cosine": True}):
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
        np.testing.assert_array_equal(first.norms, second.norms)  # as their writes gave them
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


def test_a_key_another_holds_reads_its_own_record_by_cosine():
    small, large = np.zeros(12), np.zeros(12)
    small[:4], large[:6] = 1.0, 1.0
    for cosine, first in ((False, 0), (True, 1)):
        memory = LongTermMemory(12, 3, 8, revise=0.999, cosine=cosine)
        memory.write(large, np.array([1.0, 0.0, 0.0]))
        memory.write(small, np.array([0.0, 1.0, 0.0]))
        assert memory.count == 2  # at cosine .82 another record
        # the dot product ties the contained key with the key that holds it, the lowest slot
        # first; the cosine reads the key's own record
        assert memory.read(small, credit=False).first() == first
        assert memory.read(large, credit=False).first() == 0
    told = LongTermMemory(12, 3, 8, revise=0.999, cosine=True)
    told.write(large, np.array([1.0, 0.0, 0.0]))
    told.write(small, np.array([0.0, 1.0, 0.0]))
    assert told.write(small, np.array([0.0, 0.0, 1.0])) == (1, True)  # its own record revised
    np.testing.assert_array_equal(told.values[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(told.norms[:2], [np.sqrt(6.0), 2.0])
    assert LongTermMemory(12, 3, 8).norms.shape == (0,)  # held by a cosine memory alone


# -- the address of pairs, heads with cues of their own, chains and keys held apart (cadence#310)

PAIRED = {"expansion": 512, "winners": 32, "pairs": 8, "candidates": 64, "direct": 0,
          "revise": 0.999, "fade": False, "cosine": True, "dtype": "float32"}


def paired(n=3000, width=32, **settings):
    rng = np.random.default_rng(3)
    keys = codes(rng, n, width).astype(np.float32)
    values = codes(rng, n, 8).astype(np.float32)
    memory = LongTermMemory(width, 8, n + 8, **{**PAIRED, **settings})
    return rng, keys, values, memory


def test_the_address_of_pairs_reads_an_exact_cue_and_most_noisy_ones_as_the_scan_does():
    rng, keys, values, memory = paired()
    slots = memory.fill(keys, values)
    assert memory.coarse and memory.pairs == 8 and memory.count == 3000
    np.testing.assert_array_equal(slots, np.arange(3000))
    scan = LongTermMemory(32, 8, 3008, candidates=0, fade=False, cosine=True, dtype="float32")
    scan.fill(keys, values)
    assert not scan.coarse
    for i in range(0, 3000, 50):  # an exact cue shares every pair with its record
        read = memory.read(keys[i], credit=False)
        assert read.first() == i and len(read.slots) <= 64
        np.testing.assert_array_equal(memory.value(read), values[i])
    noisy = [unit(keys[i] + 0.6 * rng.standard_normal(32) / np.sqrt(32)) for i in range(300)]
    through = np.array([memory.read(c, credit=False).first() for c in noisy])
    every = np.array([scan.read(c, credit=False).first() for c in noisy])
    assert np.mean(every == np.arange(300)) > 0.97
    assert np.mean(through == every) > 0.9  # the address costs a few of the scan's finds


def test_fill_files_records_as_writes_without_recognition_do():
    _, keys, values, memory = paired()
    _, _, _, one_by_one = paired()
    memory.fill(keys, values)
    for key, value in zip(keys, values, strict=True):
        one_by_one.write(key, value, recognised=-1)
    for name in ("cells", "fills", "keys", "values", "deep", "recent", "faded", "norms"):
        np.testing.assert_array_equal(getattr(memory, name), getattr(one_by_one, name))
    assert memory.dropped == one_by_one.dropped and memory.writes == one_by_one.writes == 3000
    with pytest.raises(ValueError, match="empty slots"):
        memory.fill(keys[:9], values[:9])
    with pytest.raises(ValueError, match="rows of"):
        memory.fill(keys[:2, :5], values[:2])


def test_the_compiled_coarse_stage_names_the_slots_the_numpy_read_names():
    from cadence import longterm

    rng, keys, values, memory = paired()
    memory.fill(keys, values)
    cues = [unit(keys[i] + rng.standard_normal(32) / np.sqrt(32)) for i in range(40)]
    named = [memory._named(memory.address(c)) for c in cues]
    try:
        LongTermMemory.COMPILED = False
        plain = [memory._named(memory.address(c)) for c in cues]
    finally:
        LongTermMemory.COMPILED = True
    for a, b in zip(named, plain, strict=True):
        np.testing.assert_array_equal(a, b)
        assert len(a) <= 64 and np.all(np.diff(a) > 0)
    if longterm._COUNT_PAIRS is not None:  # the scratch is left clean for the next read
        assert memory._hits is not None and not memory._hits.any()


def test_a_write_under_the_address_of_pairs_recognises_revises_and_refiles():
    _, keys, values, memory = paired()
    memory.fill(keys, values)
    slot, revised = memory.write(keys[7], values[9])  # the key told again
    assert (slot, revised) == (7, True) and memory.count == 3000
    np.testing.assert_array_equal(memory.value(memory.read(keys[7], credit=False)), values[9])
    slot, revised = memory.write(keys[11], values[3], recognised=5)  # slot 5 takes key 11
    assert (slot, revised) == (5, True)
    assert memory.read(keys[5], credit=False).first() != 5  # its old pairs no longer name it
    assert memory.read(keys[11], heads=2, credit=False).first(0) == 5  # the lower of two equals
    other = unit(keys[7] + 0.5 * keys[8])
    slot, revised = memory.write(other, values[1])
    assert (slot, revised) == (3000, False) and memory.read(other, credit=False).first() == 3000


def test_a_saved_memory_with_the_address_of_pairs_continues(tmp_path):
    _, keys, values, memory = paired(apart=6)
    memory.fill(keys, values, sources=np.eye(3000, 6, dtype=np.float32) + 1.0)
    assert set(memory.arrays()) >= {"cells", "fills", "signs", "state"}
    again = LongTermMemory(**memory.to_dict())
    again.restore(memory.arrays())
    assert again.cells.ctypes.data % 64 == 0  # each cell's row on a memory line of its own
    for i in (0, 17, 2999):
        assert again.read(keys[i], credit=False).first() == i
    np.testing.assert_array_equal(again.signs, memory.signs)
    assert again.nbytes == memory.nbytes > memory.keys.nbytes + memory.cells.nbytes


def test_heads_read_in_turn_each_with_a_cue_of_its_own():
    for settings in ({}, {"candidates": 0, "direct": None}):
        _, keys, values, memory = paired(**settings)
        memory.fill(keys, values)
        read = memory.read_each(np.stack([keys[1], keys[2], keys[1]]), credit=False)
        assert [read.first(h) for h in range(3)][:2] == [1, 2]
        assert read.first(2) not in (1, 2)  # the third head reads what the first two left
        np.testing.assert_array_equal(memory.value(read, 1), values[2])
        before = memory.deep[[1, 2]].copy()
        memory.read_each(keys[[1, 2]])
        np.testing.assert_array_equal(memory.deep[[1, 2]], before + 1.0)
    with pytest.raises(ValueError, match="per head"):
        memory.read_each(keys[1])
    soft = LongTermMemory(32, 8, 64, temperature=0.05, candidates=0)
    for key, value in zip(keys[:64], values[:64], strict=True):
        soft.write(key, value)
    shares = soft.read_each(np.stack([keys[4], keys[4]]), credit=False).weights
    assert shares[0][4] > 0.9 and shares[1][4] < 0.1  # a slot is read in full once


def test_a_chain_lays_each_value_on_the_next_cue():
    rng = np.random.default_rng(5)
    entities, relations = codes(rng, 40, 8), codes(rng, 3, 8)
    memory = LongTermMemory(16, 8, 200, cosine=True)

    def after(entity, relation):
        return (entity * 7 + relation + 1) % 40

    for e in range(40):
        for r in range(3):
            memory.write(np.concatenate([entities[e], relations[r]]), entities[after(e, r)])
    links = np.stack([np.concatenate([np.zeros(8), relations[r]]) for r in (1, 2)])
    reads = memory.chain(np.concatenate([entities[3], relations[0]]), links, credit=False)
    first = after(3, 0)
    second = after(first, 1)
    assert [int(np.argmax(entities @ memory.value(r))) for r in reads] == [
        first, second, after(second, 2)]
    assert len(memory.chain(np.zeros(16), links, credit=False)) == 3  # a zero cue ties: slot 0
    with pytest.raises(ValueError, match="must fit"):
        memory.chain(np.zeros(16), links, at=9)
    assert len(LongTermMemory(16, 8, 4).chain(np.ones(16), links)) == 1  # nothing read: ends


def test_a_source_tells_two_facts_with_one_key_apart():
    rng = np.random.default_rng(7)
    key, values, sources = unit(rng.standard_normal(16)), codes(rng, 4, 8), \
        rng.standard_normal((3, 20))
    plain = LongTermMemory(16, 8, 100, cosine=True, revise=0.999)
    assert plain.write(key, values[0]) == (0, False) and plain.write(key, values[1]) == (0, True)
    memory = LongTermMemory(16, 8, 100, cosine=True, revise=0.999, apart=20)
    assert memory.write(key, values[0], source=sources[0]) == (0, False)
    assert memory.write(key, values[1], source=sources[1]) == (1, False)  # another fact
    assert memory.write(key, values[2], source=sources[0]) == (0, True)  # the first, told again
    assert memory.count == 2
    assert memory.read(key, source=sources[0], credit=False).first() == 0
    assert memory.read(key, source=sources[1], credit=False).first() == 1
    np.testing.assert_array_equal(
        memory.value(memory.read(key, source=sources[1], credit=False)), values[1])
    assert memory.read(key, credit=False).first() == 0  # without a source: the lower of equals
    assert memory.write(key, values[3]) == (0, True)  # and a write revises its first choice
    with pytest.raises(ValueError, match="apart"):
        plain.read(key, source=sources[0])
    with pytest.raises(ValueError, match="one finite row of 20"):
        memory.write(key, values[0], source=sources[0][:5])


def test_the_new_settings_are_checked_and_off_by_default():
    assert LongTermMemory(8, 8, 32).pairs == 0 and LongTermMemory(8, 8, 32).apart == 0
    for settings in ({"pairs": 1}, {"pairs": 65}, {"pairs": -2}, {"apart": -1}, {"pairs": True}):
        with pytest.raises(ValueError):
            LongTermMemory(8, 8, 32, **settings)
    small = LongTermMemory(8, 8, 32, pairs=8)  # under ``direct`` slots every slot is scored
    assert not small.coarse and not hasattr(small, "cells")


# -- a table that grows, records on disk, a read that ends at its record (cadence#310)


def growing(n=3000, width=32, **settings):
    """``paired``'s records in a store whose table grows; the capacity is a bound."""
    rng = np.random.default_rng(3)
    keys = codes(rng, n, width).astype(np.float32)
    values = codes(rng, n, 8).astype(np.float32)
    memory = LongTermMemory(width, 8, 4 * n, **{**PAIRED, "grow": True, **settings})
    return rng, keys, values, memory


def table(memory):
    """The cells a growing table holds."""
    return memory.cells[: memory._level + memory._split]


def test_a_table_that_grows_holds_what_its_records_call_for_and_reads_as_the_fixed_one():
    rng, keys, values, memory = growing()
    assert memory.grow and memory._level + memory._split == 64 and memory.entries == 0
    empty = memory.nbytes
    memory.fill(keys[:1500], values[:1500])
    half, cells = memory.nbytes, memory._level + memory._split
    load = memory.GROW_LOAD
    assert -(-memory.entries // load) <= cells <= -(-(memory.entries + memory.dropped) // load)
    memory.fill(keys[1500:], values[1500:])
    # the bytes follow the records held, not the bound on the slots
    assert memory.count == 3000 and (memory.nbytes - empty) / (half - empty) == pytest.approx(
        2.0, rel=0.05)
    assert memory._level + memory._split == pytest.approx(2 * cells, rel=0.05)
    # every entry lies in a cell that holds it, and the cells past the table are untouched
    assert int((table(memory) != memory.EMPTY).sum()) == memory.entries
    assert memory.entries + memory.dropped == 3000 * 28
    assert not memory.cells[memory._level + memory._split :][:64].any()
    _, _, _, fixed = paired()
    fixed.fill(keys, values)
    for i in range(0, 3000, 50):  # an exact cue shares every pair with its record
        read = memory.read(keys[i], credit=False)
        assert read.first() == i and len(read.slots) <= 64
        np.testing.assert_array_equal(memory.value(read), values[i])
    noisy = [unit(keys[i] + 0.6 * rng.standard_normal(32) / np.sqrt(32)) for i in range(300)]
    grown = np.array([memory.read(c, credit=False).first() for c in noisy])
    sized = np.array([fixed.read(c, credit=False).first() for c in noisy])
    assert np.mean(grown == np.arange(300)) > 0.9 and np.mean(grown == sized) > 0.95


def test_a_write_splits_the_cells_its_entries_call_for_and_none_rebuilds_the_table():
    _, keys, values, memory = growing()
    most, held = 0, memory._level + memory._split
    for i in range(3000):
        slot, revised = memory.write(keys[i], values[i])
        assert (slot, revised) == (i, False)
        now = memory._level + memory._split
        most, held = max(most, now - held), now
    # a record of 28 pairs calls for five cells of six entries: no write split more
    load = memory.GROW_LOAD
    assert most <= -(-28 // load)
    assert -(-memory.entries // load) <= held <= -(-(memory.entries + memory.dropped) // load)
    assert memory.entries + memory.dropped == 3000 * 28
    assert int((table(memory) != memory.EMPTY).sum()) == memory.entries
    for i in range(0, 3000, 7):
        assert memory.read(keys[i], credit=False).first() == i
    # the rounds ended on the way: the table went through several doublings
    assert memory._level >= 64 * 2 ** 7


def test_the_compiled_and_the_numpy_forms_make_the_same_growing_table():
    from cadence import _pairs

    made = []
    for compiled in (True, False):
        try:
            LongTermMemory.COMPILED = compiled
            rng, keys, values, memory = growing()
            memory.fill(keys[:1000], values[:1000])              # an empty table laid out
            memory.fill(keys[1000:2000], values[1000:2000])      # a table with entries split
            for i in range(2000, 2300):                          # writes one by one
                memory.write(keys[i], values[i])
            memory.write(keys[2400], values[3], recognised=5)    # slot 5 leaves and files again
            cues = [unit(keys[i] + rng.standard_normal(32) / np.sqrt(32)) for i in range(40)]
            made.append((memory, [memory._named(memory.address(c)) for c in cues]))
        finally:
            LongTermMemory.COMPILED = True
    (a, named), (b, plain) = made
    assert (a._level, a._split, a.entries, a.dropped) == (b._level, b._split, b.entries,
                                                         b.dropped)
    np.testing.assert_array_equal(table(a), table(b))
    np.testing.assert_array_equal(a.filed[:2300], b.filed[:2300])
    for x, y in zip(named, plain, strict=True):
        np.testing.assert_array_equal(x, y)
    drive = np.random.default_rng(5).standard_normal((50, 300)).astype(np.float32)
    drive[3, :] = -1.0  # no unit driven above zero
    drive[4, 10:] = -1.0  # fewer than asked
    drive[5, 20:40] = 9.0  # equal drives: the lower units win
    plain_top = _pairs.top_units(drive, 8)
    assert (plain_top[3] == -1).all() and (plain_top[4] >= 0).sum() <= 8
    np.testing.assert_array_equal(plain_top[5], np.arange(20, 28))
    if _pairs._TOP is not None:
        out = np.zeros((50, 8), dtype=np.int32)
        _pairs._TOP(drive, np.zeros(50, dtype=np.float32), np.zeros(300, dtype=np.float32), 8,
                    out)
        np.testing.assert_array_equal(out, plain_top)
        mean = np.linspace(-1, 1, 50).astype(np.float32)
        sums = np.random.default_rng(6).standard_normal(300).astype(np.float32)
        _pairs._TOP(drive, mean, sums, 8, out)  # centred as the address of pairs centres a key
        np.testing.assert_array_equal(out, _pairs.top_units(drive - mean[:, None] * sums, 8))


def test_a_write_under_a_growing_table_recognises_revises_and_refiles():
    _, keys, values, memory = growing()
    memory.fill(keys, values)
    entries = memory.entries
    slot, revised = memory.write(keys[7], values[9])  # the key told again
    assert (slot, revised) == (7, True) and memory.count == 3000 and memory.entries == entries
    np.testing.assert_array_equal(memory.value(memory.read(keys[7], credit=False)), values[9])
    slot, revised = memory.write(keys[11], values[3], recognised=5)  # slot 5 takes key 11
    assert (slot, revised) == (5, True)
    assert memory.read(keys[5], credit=False).first() != 5  # its old pairs no longer name it
    assert memory.read(keys[11], heads=2, credit=False).first(0) == 5  # the lower of two equals
    np.testing.assert_array_equal(memory.filed[5], memory.filed[11])
    assert int((table(memory) != memory.EMPTY).sum()) == memory.entries
    other = unit(keys[7] + 0.5 * keys[8])
    slot, revised = memory.write(other, values[1])
    assert (slot, revised) == (3000, False) and memory.read(other, credit=False).first() == 3000


def test_fills_that_leave_their_records_unfiled_share_one_pass_over_the_table():
    _, keys, values, whole = growing()
    _, _, _, parts = growing()
    whole.fill(keys, values)
    parts.fill(keys[:1200], values[:1200], file=False)
    assert parts.count == 1200 and parts.entries == 0  # written, and not yet under their pairs
    parts.fill(keys[1200:], values[1200:], file=False)
    parts.file()
    np.testing.assert_array_equal(table(parts), table(whole))
    assert (parts.entries, parts.dropped) == (whole.entries, whole.dropped)
    _, _, _, written = growing()
    written.fill(keys[:2999], values[:2999], file=False)
    written.write(keys[2999], values[2999])  # a write files what the fills left
    assert written.read(keys[10], credit=False).first() == 10
    assert written.read(keys[2999], credit=False).first() == 2999


def test_a_saved_memory_with_a_growing_table_continues_and_a_copy_is_its_own():
    from copy import deepcopy

    _, keys, values, memory = growing(apart=6)
    memory.fill(keys, values, sources=np.eye(3000, 6, dtype=np.float32) + 1.0)
    kept = memory.arrays()
    assert set(kept) >= {"cells", "filed", "signs", "state", "table"}
    assert len(kept["keys"]) == 3000 and len(kept["cells"]) == memory._level + memory._split
    again = LongTermMemory(**memory.to_dict())
    assert again.grow and memory.to_dict()["grow"] is True and "path" not in memory.to_dict()
    again.restore(kept)
    np.testing.assert_array_equal(table(again), table(memory))
    for i in (0, 17, 2999):
        assert again.read(keys[i], credit=False).first() == i
    assert again.nbytes == memory.nbytes
    assert again.write(unit(keys[1] + keys[2]), values[0]) == (3000, False)
    twin = deepcopy(memory)
    twin.write(unit(keys[3] + keys[4]), values[0])
    assert twin.count == 3001 and memory.count == 3000 and twin.entries > memory.entries
    assert int((table(memory) != memory.EMPTY).sum()) == memory.entries
    with pytest.raises(ValueError, match="another shape"):
        LongTermMemory(**{**memory.to_dict(), "pairs": 6}).restore(kept)


def test_a_store_on_disk_continues_from_its_files(tmp_path):
    from copy import deepcopy

    rng, keys, values, _ = growing()
    folder = tmp_path / "records"
    settings = {**PAIRED, "grow": True, "path": folder}
    store = LongTermMemory(32, 8, 1 << 17, **settings)  # a bound far past the records
    store.fill(keys[:2000], values[:2000])
    assert store.to_dict()["path"] == str(folder) and store.count == 2000
    files = {f.name for f in folder.iterdir()}
    assert files >= {"store.json", "state.npy", "keys.npy", "values.npy", "cells.npy",
                     "filed.npy", "projection.npy", "norms.npy"}
    # the files' lengths are the bound's; the blocks on the disk follow the records
    assert (folder / "keys.npy").stat().st_size > (1 << 17) * 32 * 4
    assert store.nbytes < (folder / "keys.npy").stat().st_size
    with pytest.raises(ValueError, match="holds a store"):
        LongTermMemory(32, 8, 1 << 17, **settings)
    with pytest.raises(TypeError, match="one store"):
        deepcopy(store)
    cells = table(store).copy()
    store.close()
    again = LongTermMemory.open(folder)
    assert (again.count, again.writes, again.entries) == (2000, 2000, store.entries)
    assert again.to_dict() == store.to_dict()
    np.testing.assert_array_equal(table(again), cells)
    np.testing.assert_array_equal(again.projection, store.projection)
    for i in range(0, 2000, 40):
        read = again.read(keys[i], credit=False)
        assert read.first() == i
        np.testing.assert_array_equal(again.value(read), values[i])
    for i in range(2000, 3000):  # records added to the opened store, one by one
        again.write(keys[i], values[i])
    again.flush()
    third = LongTermMemory.open(folder)  # a second mapping of the same files
    assert third.count == 3000 and third.read(keys[2999], credit=False).first() == 2999
    noisy = [unit(keys[i] + 0.6 * rng.standard_normal(32) / np.sqrt(32)) for i in range(100)]
    _, _, _, memory = growing()
    memory.fill(keys[:2000], values[:2000])
    for i in range(2000, 3000):
        memory.write(keys[i], values[i])
    for cue in noisy:  # the store on disk reads as the same store in memory
        assert third.read(cue, credit=False).first() == memory.read(cue, credit=False).first()
    third._cold = True  # a store whose reads wait for the disk fetches its rows by threads
    for cue in noisy:
        assert third.read(cue, credit=False).first() == memory.read(cue, credit=False).first()
        third._cold = True
    third.close()
    again.close()
    with pytest.raises(ValueError, match="holds no store"):
        LongTermMemory.open(tmp_path / "nothing")
    scan = LongTermMemory(32, 8, 500, candidates=0, cosine=True, path=tmp_path / "scan")
    scan.write(keys[0], values[0].astype(np.float64))
    scan.close()
    assert LongTermMemory.open(tmp_path / "scan").read(keys[0], credit=False).first() == 0
    sized = LongTermMemory(32, 8, 3008, **{**PAIRED, "path": tmp_path / "sized"})
    sized.fill(keys, values)  # the fixed table of pairs in files
    sized.close()
    sized = LongTermMemory.open(tmp_path / "sized")
    _, _, _, fixed = paired()
    fixed.fill(keys, values)
    np.testing.assert_array_equal(sized.cells, fixed.cells)
    assert all(sized.read(c, credit=False).first() == fixed.read(c, credit=False).first()
               for c in noisy)
    sized.close()


def test_a_read_with_sure_ends_at_its_record_and_reads_any_other_cue_in_full():
    rng, keys, values, plain = growing()
    _, _, _, memory = growing(sure=0.999)
    plain.fill(keys, values)
    memory.fill(keys, values)
    np.testing.assert_array_equal(table(memory), table(plain))
    read_units = []
    found = memory._found
    memory._found = lambda units: read_units.append(len(units)) or found(units)
    for i in range(0, 3000, 30):
        read = memory.read(keys[i], credit=False)
        assert read.first() == i == plain.read(keys[i], credit=False).first()
        assert len(read.slots) <= memory.CHECK
        np.testing.assert_array_equal(memory.value(read), values[i])
    assert set(read_units) == {8}  # the pairs of the cue's eight most driven units, no more
    read_units.clear()
    noisy = [unit(keys[i] + 0.6 * rng.standard_normal(32) / np.sqrt(32)) for i in range(200)]
    for cue in noisy:  # no record holds a noisy cue so closely: the whole read, the same choice
        assert memory.read(cue, credit=False).first() == plain.read(cue, credit=False).first()
    assert max(read_units) == 32 and read_units.count(32) == 200
    read_units.clear()
    two = memory.read(keys[3], heads=2, credit=False)  # several heads read in full
    assert two.first(0) == 3 and read_units == [32]
    slot, revised = memory.write(keys[40], values[41])  # a write recognises through it too
    assert (slot, revised) == (40, True)
    near = unit(keys[9] + 0.01 * rng.standard_normal(32) / np.sqrt(32))
    assert memory.read(near, credit=False).first() == 9


def test_the_growing_table_and_the_store_on_disk_are_checked_and_off_by_default(tmp_path):
    plain = LongTermMemory(8, 8, 32)
    assert plain.grow is False and plain.path is None and plain.sure == 0.0
    assert not {"grow", "path", "sure"} & set(plain.to_dict())
    for settings in ({"grow": True}, {"grow": 1, "pairs": 8, "direct": 0},
                     {"sure": 0.9}, {"sure": 2.0, "pairs": 8, "direct": 0},
                     {"sure": True, "pairs": 8, "direct": 0}):
        with pytest.raises(ValueError):
            LongTermMemory(8, 8, 3000, **settings)
    with pytest.raises(ValueError, match="address of pairs"):
        LongTermMemory(8, 8, 30000, path=tmp_path / "units")  # the address of units, on disk
    with pytest.raises(ValueError, match="under 2\\*\\*28"):
        LongTermMemory(8, 8, 1 << 28, pairs=8, grow=True, direct=0)


def test_keys_of_few_active_coordinates_are_filed_by_their_weights_alone():
    from cadence import _pairs

    rng = np.random.default_rng(7)
    keys = np.zeros((2000, 96), dtype=np.float32)
    for row in keys:
        row[rng.choice(96, size=int(rng.integers(5, 9)), replace=False)] = 1.0
    values = codes(rng, 2000, 8).astype(np.float32)
    settings = {**PAIRED, "grow": True, "expansion": 1024}
    memory = LongTermMemory(96, 8, 8000, **settings)
    memory.fill(keys, values)
    for i in range(0, 2000, 20):
        assert np.array_equal(memory.keys[memory.read(keys[i], credit=False).first()], keys[i])
    # the units a fill files a key under are the units its own write files it under
    drive = memory._drive(keys[:300])
    whole = _pairs.top_units(drive, 8)
    kept = np.take_along_axis(drive, np.maximum(whole, 0), axis=1)
    apart = (np.sort(drive, axis=1)[:, -9] < kept.min(axis=1) - 1e-5) & (whole >= 0).all(axis=1)
    assert apart.sum() > 250  # rows whose eighth unit is clear of the ninth
    np.testing.assert_array_equal(memory.filed[:300][apart], whole[apart])
    for i in np.flatnonzero(apart)[:50]:
        np.testing.assert_array_equal(memory._row(keys[i]), memory.filed[i])
