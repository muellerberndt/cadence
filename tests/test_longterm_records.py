"""``Brain.compose(longterm=records)`` (cadence#288): the brain keys its records on its own held
message at a turn, writes the moment after the turn only where it says more than the record the
key finds, reads its records before the settle onto regions the association reads, and keeps
them across episodes. Without it a brain is the brain without the option."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.longterm import LongTermRecords

INPUTS = 15  # agents 0-5, verbs 6-9, patients 10-13, the turn 14
TURN = 14


def moment(*on: int, streams: int = 1) -> np.ndarray:
    x = np.zeros((streams, INPUTS))
    x[:, list(on)] = 1.0
    return x


def brain(**settings):
    return cd.Brain.compose(
        INPUTS, 4, modules=(32,), seed=0, backend="cpu", message=70, message_boundary=TURN,
        message_boundary_holds=True, **settings,
    )


def tell(b, agent: int, verb: int, patient: int) -> None:
    b.reset()
    for x in (moment(agent), moment(6 + verb), moment(TURN), moment(10 + patient)):
        b.act(x, greedy=True)


def ask(b, agent: int, verb: int) -> int:
    b.reset()
    for x in (moment(agent), moment(6 + verb), moment(TURN)):
        b.act(x, greedy=True)
    return int(np.asarray(b.act(np.zeros((1, INPUTS)), greedy=True)).ravel()[0])


def test_without_the_option_the_brain_is_the_brain_without_it():
    plain = brain()
    named = brain(longterm=None)
    assert plain.longterm is None and named.longterm is None
    assert "longterm" not in plain.memories and "longterm" not in plain.connectome.populations
    assert plain.connectome.n == named.connectome.n
    np.testing.assert_array_equal(plain.connectome.pre, named.connectome.pre)
    for b in (plain, named):
        b.reset()
    for x in (moment(0), moment(6), moment(TURN), moment(10), np.zeros((1, INPUTS))):
        a, c = plain.act(x, greedy=True), named.act(x, greedy=True)
        np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(plain.basal_ganglia.state.activation,
                                  named.basal_ganglia.state.activation)


def test_the_option_adds_one_region_per_head_and_its_projection_last():
    plain, held = brain(), brain(longterm=32, longterm_heads=2)
    populations = held.connectome.populations
    width = held.message.inputs
    assert list(populations)[-2:] == ["longterm", "longterm_1"]
    assert len(populations["longterm"]) == len(populations["longterm_1"]) == width
    assert held.connectome.n == plain.connectome.n + 2 * width
    names = list(held.memories)
    assert names.index("longterm") == names.index("message") + 1  # read after the message
    kept = held.longterm
    assert isinstance(kept, LongTermRecords) and kept.capacity == 32 and kept.heads == 2
    assert (kept.match, kept.revise, kept.amplitude) == (0.75, 0.999, 3.0)


def test_the_option_is_refused_without_a_held_message_from_a_turn():
    with pytest.raises(ValueError, match="message=units"):
        cd.Brain.compose(INPUTS, 4, seed=0, backend="cpu", longterm=16)
    with pytest.raises(ValueError, match="message_boundary"):
        cd.Brain.compose(INPUTS, 4, seed=0, backend="cpu", message=70, longterm=16)
    with pytest.raises(ValueError, match="needs long-term records"):
        cd.Brain.compose(INPUTS, 4, seed=0, backend="cpu", longterm_heads=2)
    with pytest.raises(TypeError, match="unknown longterm setting"):
        brain(longterm=8, longterm_width=3)
    with pytest.raises(ValueError, match="cosine"):
        brain(longterm=8, longterm_match=1.5)
    b = brain(longterm=8)
    with pytest.raises(ValueError):
        b.retune(longterm_match=0.5)


def test_a_fact_is_written_once_told_again_agreed_and_a_change_revised():
    b = brain(longterm=16)
    kept = b.longterm
    tell(b, 0, 0, 2)
    store = kept.stores[0]
    assert store.count == 1 and kept.counts["writes"] == 1
    key = b.message.key()[0]
    np.testing.assert_array_equal(store.keys[0], key)  # the message at the turn
    np.testing.assert_array_equal(store.values[0], moment(12)[0])  # the moment after it
    tell(b, 0, 0, 2)
    assert store.count == 1 and kept.counts["agreed"] == 1 and kept.counts["writes"] == 1
    tell(b, 0, 0, 3)
    assert store.count == 1 and kept.counts["revisions"] == 1
    np.testing.assert_array_equal(store.values[0], moment(13)[0])
    tell(b, 1, 0, 3)  # another agent: another key, another record
    assert store.count == 2
    ask(b, 0, 0)  # a question writes nothing: the moment after its turn is silent
    assert store.count == 2 and kept.counts["writes"] == 3


def test_the_read_holds_the_record_its_key_finds_after_the_turn_and_silence_before():
    b = brain(longterm=16)
    tell(b, 0, 1, 2)
    tell(b, 3, 2, 1)
    b.reset()
    columns = b.connectome.populations["longterm"]
    for x in (moment(3), moment(8)):
        drive = b.stimulus(x)
        assert not drive[0, columns].any()  # no turn yet: the region is silent
        b.act(x, greedy=True)
    b.act(moment(TURN), greedy=True)
    drive = b.stimulus(np.zeros((1, INPUTS)))
    np.testing.assert_allclose(drive[0, columns], 3.0 * moment(11)[0])
    assert b.longterm.read_slots[0, 0] == 1
    b.reset()
    for x in (moment(5), moment(9), moment(TURN)):  # a key no record holds
        b.act(x, greedy=True)
    assert not b.stimulus(np.zeros((1, INPUTS)))[0, columns].any()


def test_records_survive_reset_and_each_stream_keeps_its_own():
    b = brain(longterm=16)
    b.reset()
    tells = [moment(0, streams=2), moment(6, streams=2), moment(TURN, streams=2)]
    patient = np.zeros((2, INPUTS))
    patient[0, 10] = patient[1, 13] = 1.0
    for x in (*tells, patient):
        b.act(x, greedy=True)
    stores = b.longterm.stores
    assert [s.count for s in stores] == [1, 1]
    np.testing.assert_array_equal(stores[0].values[0], patient[0])
    np.testing.assert_array_equal(stores[1].values[0], patient[1])
    b.reset()
    assert [s.count for s in b.longterm.stores] == [1, 1]
    b.longterm.clear()
    assert b.longterm.stores == []


def test_heads_read_in_turn_onto_their_own_regions():
    b = brain(longterm=16, longterm_heads=2, longterm_match=0.0)
    tell(b, 0, 0, 1)
    tell(b, 0, 1, 2)
    b.reset()
    for x in (moment(0), moment(6), moment(TURN)):
        b.act(x, greedy=True)
    drive = b.stimulus(np.zeros((1, INPUTS)))
    first = drive[0, b.connectome.populations["longterm"]]
    second = drive[0, b.connectome.populations["longterm_1"]]
    np.testing.assert_allclose(first, 3.0 * moment(11)[0])  # the fact itself
    np.testing.assert_allclose(second, 3.0 * moment(12)[0])  # the next record, the other
    assert len(set(b.longterm.read_slots[0].tolist())) == 2


def test_an_imagined_branch_reads_the_records_and_writes_none():
    b = brain(longterm=16)
    tell(b, 0, 0, 1)
    before = (b.longterm.stores[0].count, dict(b.longterm.counts),
              b.longterm.stores[0].deep.copy())
    b.reset()
    b.imagine([moment(2), moment(7), moment(TURN), moment(12)])
    assert b.longterm.stores[0].count == before[0] and b.longterm.counts == before[1]
    np.testing.assert_array_equal(b.longterm.stores[0].deep, before[2])


def nursery(b, rng: np.random.Generator, rounds: int) -> None:
    """Facts about the pairs of an even sum, each told and then a told one asked and taught."""
    pairs = [(a, v) for a in range(6) for v in range(4) if (a + v) % 2 == 0]
    truth: dict[tuple[int, int], int] = {}
    for _ in range(rounds):
        a, v = pairs[int(rng.integers(len(pairs)))]
        truth[(a, v)] = int(rng.integers(4))
        tell(b, a, v, truth[(a, v)])
        a, v = list(truth)[int(rng.integers(len(truth)))]
        b.reset()
        for x in (moment(a), moment(6 + v), moment(TURN)):
            b.act(x, greedy=True)
        b.lesson(np.zeros((1, INPUTS)), [truth[(a, v)]])
        b.act(np.zeros((1, INPUTS)), greedy=True)


def fresh(b, rng: np.random.Generator) -> float:
    """Facts about the pairs no lesson asked (an odd sum), told once, then asked."""
    truth = {(a, v): int(rng.integers(4)) for a in range(6) for v in range(4) if (a + v) % 2}
    for (a, v), p in truth.items():
        tell(b, a, v, p)
    return float(np.mean([ask(b, a, v) == p for (a, v), p in truth.items()]))


def test_a_brain_answers_from_its_own_records_on_facts_no_lesson_asked():
    held, plain = brain(longterm=64), brain()
    for b in (held, plain):
        nursery(b, np.random.default_rng(0), 300)
    right = fresh(held, np.random.default_rng(1))
    assert right >= 0.85
    assert fresh(plain, np.random.default_rng(1)) <= 0.5  # the answer is nowhere to be read


def _format(path) -> str:
    import json

    with np.load(path, allow_pickle=False) as archive:
        return str(json.loads(str(archive["brain"]))["format"])


def test_a_saved_brain_with_records_continues_as_the_one_it_was_saved_from(tmp_path):
    first = brain(longterm=32, longterm_heads=2)
    rng = np.random.default_rng(3)
    for _ in range(6):
        tell(first, int(rng.integers(6)), int(rng.integers(4)), int(rng.integers(4)))
    first.reset()
    for x in (moment(1), moment(7), moment(TURN)):  # saved in the middle of a question
        first.act(x, greedy=True)
    path = first.save(tmp_path / "records.npz")
    assert _format(path) == "cadence-brain/4"
    second = cd.Brain.load(path)
    kept, taken = first.longterm, second.longterm
    assert taken is not None and taken.message is second.message
    assert taken.to_dict() == kept.to_dict()
    np.testing.assert_array_equal(taken.since, kept.since)
    for a, b in zip(kept.stores, taken.stores, strict=True):
        for name, value in a.arrays().items():
            np.testing.assert_array_equal(value, b.arrays()[name], err_msg=name)
    for x in (np.zeros((1, INPUTS)), moment(2), moment(8), moment(TURN), moment(11)):
        np.testing.assert_array_equal(first.act(x, greedy=True), second.act(x, greedy=True))
        np.testing.assert_array_equal(first.longterm.read_slots, second.longterm.read_slots)
    assert first.longterm.counts == second.longterm.counts
    for a, b in zip(first.longterm.stores, second.longterm.stores, strict=True):
        for name, value in a.arrays().items():
            np.testing.assert_array_equal(value, b.arrays()[name], err_msg=name)


def test_a_brain_without_records_keeps_its_format_and_a_named_file_needs_its_records(tmp_path):
    import json

    plain = brain()
    for x in (moment(0), moment(6), moment(TURN), moment(10)):
        plain.act(x, greedy=True)
    assert _format(plain.save(tmp_path / "plain.npz")) == "cadence-brain/1"
    held = brain(longterm=8)
    tell(held, 0, 0, 1)
    path = held.save(tmp_path / "held.npz", compressed=False)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive}
    saved = json.loads(str(data["brain"]))
    saved["format"] = "cadence-brain/1"  # a records file under another name is refused
    data["brain"] = np.array(json.dumps(saved))
    np.savez(tmp_path / "renamed.npz", **data)
    with pytest.raises(ValueError, match="cadence-brain/4"):
        cd.Brain.load(tmp_path / "renamed.npz")
    saved["format"] = "cadence-brain/4"
    del data["memories/longterm/store/0/keys"]
    data["brain"] = np.array(json.dumps(saved))
    np.savez(tmp_path / "torn.npz", **data)
    with pytest.raises(ValueError, match="missing saved long-term records"):
        cd.Brain.load(tmp_path / "torn.npz")


def _lived(batched: bool, streams: int, capacity: int, heads: int):
    """A brain of ``streams`` streams that tells, revises and asks facts past its capacity, its
    records read, written and faded as one bank or store by store."""
    b = brain(longterm=capacity, longterm_heads=heads)
    b.longterm.BATCHED = batched  # this brain's records alone
    rng = np.random.default_rng(7)
    acts, reads = [], []
    for episode in range(40):
        b.reset()
        agent, verb = rng.integers(6, size=streams), rng.integers(4, size=streams)
        patient = rng.integers(4, size=streams)
        asking = episode % 3 == 2
        for t in range(4):
            x = np.zeros((streams, INPUTS))
            rows = np.arange(streams)
            if t == 0:
                x[rows, agent] = 1.0
            elif t == 1:
                x[rows, 6 + verb] = 1.0
            elif t == 2:
                x[:, TURN] = 1.0
            elif not asking:
                x[rows, 10 + patient] = 1.0
            acts.append(np.asarray(b.act(x, greedy=True)))
            reads.append(b.longterm.read_slots.copy())
    return b, acts, reads


def test_the_bank_reads_writes_and_fades_as_the_stores_one_by_one():
    one, acts_one, reads_one = _lived(False, 3, 6, 2)
    bank, acts_bank, reads_bank = _lived(True, 3, 6, 2)
    assert bank.longterm.batched and not one.longterm.batched
    for a, b in zip(acts_one, acts_bank, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(reads_one, reads_bank, strict=True):
        np.testing.assert_array_equal(a, b)
    assert one.longterm.counts == bank.longterm.counts
    assert one.longterm.counts["revisions"] > 0 and all(
        s.count == 6 for s in bank.longterm.stores)  # past the capacity: slots replaced
    for a, b in zip(one.longterm.stores, bank.longterm.stores, strict=True):
        n = a.count
        assert n == b.count and a.writes == b.writes
        for name in ("keys", "values", "deep", "recent", "faded"):
            np.testing.assert_array_equal(getattr(a, name)[:n], getattr(b, name)[:n],
                                          err_msg=name)


def test_the_bank_follows_the_stores_through_save_load_keep_and_clear(tmp_path):
    first, _, _ = _lived(True, 2, 8, 1)
    path = first.save(tmp_path / "bank.npz")
    second = cd.Brain.load(path)
    kept = second.longterm
    assert kept.batched
    for s, store in enumerate(kept.stores):
        assert np.shares_memory(store.deep, kept._deep)
        held = store.keys[: store.count].T.astype(np.uint8)
        np.testing.assert_array_equal(kept._index[s, :, : store.count], held)
    for on in (1, 7, TURN, 12):
        x = np.zeros((2, INPUTS))
        x[:, on] = 1.0
        np.testing.assert_array_equal(first.act(x, greedy=True), second.act(x, greedy=True))
    for a, b in zip(first.longterm.stores, second.longterm.stores, strict=True):
        np.testing.assert_array_equal(a.deep[: a.count], b.deep[: b.count])
    second.longterm.keep(np.array([1]))
    assert len(second.longterm.stores) == 1 and second.longterm._index.shape[0] == 1
    second.longterm.clear()
    assert second.longterm.stores == [] and second.longterm._deep.shape == (0, 8)


def test_a_write_reads_its_store_once():
    from cadence.longterm import LongTermMemory

    b = brain(longterm=16)
    tell(b, 0, 0, 1)
    calls = {"read": 0}
    original = LongTermMemory.read

    def counted(self, *args, **kwargs):
        calls["read"] += 1
        return original(self, *args, **kwargs)

    LongTermMemory.read = counted
    try:
        tell(b, 0, 0, 2)  # a revision: the read before the settle is the write's recognition
        tell(b, 1, 1, 3)  # a new fact
    finally:
        LongTermMemory.read = original
    assert calls["read"] == 0  # the bank reads; no store reads one by one
    assert b.longterm.counts["revisions"] == 1 and b.longterm.stores[0].count == 2


def test_the_records_read_a_contained_key_s_own_record_as_a_bank_and_one_by_one():
    picks = []
    for batched in (True, False):
        kept = brain(longterm=8).longterm
        kept.BATCHED = batched
        kept._streams(1)
        small, large = np.zeros(kept.width), np.zeros(kept.width)
        small[:4], large[:6] = 1.0, 1.0
        kept.record(0, large, moment(10)[0])
        kept.record(0, small, moment(11)[0])
        assert kept.stores[0].count == 2 and kept.batched == batched
        if batched:
            picks.append(int(kept._choices(np.array([0]), small[None, :], 1)[0, 0]))
        else:
            picks.append(kept.stores[0].read(small, credit=False).first())
    assert picks == [1, 1]  # by cosine; the dot product ties them and reads slot 0


# -- turns in a conversation (``longterm_turn``, #305)


def turns_brain(**settings):
    return cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, backend="cpu", message=70,
                            longterm=16, longterm_turn=TURN, **settings)


def say(b, *moments):
    for on in moments:
        b.act(moment(*on) if isinstance(on, tuple) else moment(on), greedy=True)


def test_turns_need_a_message_and_a_channel_and_name_their_settings():
    b = turns_brain()
    kept = b.longterm
    assert kept.turn == TURN and kept.rare == 0.03 and b.message.boundary is None
    assert kept.to_dict()["turn"] == TURN and "turn" not in brain(longterm=16).longterm.to_dict()
    with pytest.raises(ValueError, match="message"):
        cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, longterm=16, longterm_turn=TURN)
    with pytest.raises(ValueError, match="longterm_turn"):
        cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, message=70, longterm=16,
                         longterm_turn=INPUTS)
    with pytest.raises(ValueError, match="longterm_rare"):
        turns_brain(longterm_rare=0.0)
    with pytest.raises(ValueError, match="needs long-term records"):
        cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, message=70, longterm_turn=TURN)


def test_an_episode_writes_a_record_at_every_turn_and_keeps_its_recall():
    b = turns_brain(longterm_rare=1.0)  # every unit is content
    b.reset()
    say(b, 0, 6, TURN, 10, 1, 7, TURN, 11)  # two facts in one episode
    store = b.longterm.stores[0]
    assert store.count == 2 and b.longterm.counts["writes"] == 2
    assert np.array_equal(store.values[0], moment(10)[0]) and np.array_equal(
        store.values[1], moment(11)[0])
    code = b.message.code
    first = np.maximum(code(moment(0)), code(moment(6)))[0]
    assert np.array_equal(store.keys[0], first)  # the moments since the episode began
    second = np.maximum(code(moment(1)), code(moment(7)))[0]
    assert np.array_equal(store.keys[1], second)  # since the value before: not 10, not 0 or 6
    say(b, 0, 6, TURN)  # the first fact asked again in the same episode
    drive = b.stimulus(np.zeros((1, INPUTS)))
    assert np.allclose(drive[0, b.connectome.populations["longterm"]], 3.0 * moment(10)[0])
    say(b, 0)  # the moment after the question's turn says another agent: written as a fact
    assert store.count == 2 and b.longterm.counts["revisions"] == 1


def test_a_turn_s_key_holds_the_rare_units_alone():
    b = turns_brain()
    for _ in range(400):  # verbs 0, 1 and 2 are heard often; the agents rarely
        say(b, 6, 7, 8)
    kept = b.longterm
    code = b.message.code
    say(b, 0, 6, TURN, 10)
    key = kept.stores[0].keys[0]
    verb = code(moment(6))[0] * (kept.heard / kept.moments > kept.rare)
    assert key.any() and not (key * verb).any()  # the frequent verb's units drop out
    say(b, 0, 7, 8, TURN, 12)  # the same agent in other words: the record revised
    assert kept.stores[0].count == 1 and kept.counts["revisions"] == 1
    assert np.array_equal(kept.stores[0].values[0], moment(12)[0])


def test_a_saved_brain_with_turns_continues_as_the_one_it_was_saved_from(tmp_path):
    first = turns_brain(longterm_rare=1.0)
    say(first, 0, 6, TURN, 10, 1)  # saved inside a window
    second = cd.Brain.load(first.save(tmp_path / "turns.npz"))
    assert second.longterm.turn == TURN and second.longterm.moments == first.longterm.moments
    for on in (7, TURN, 11, 0, 6, TURN, 13):
        x = moment(on)
        np.testing.assert_array_equal(first.act(x, greedy=True), second.act(x, greedy=True))
    for name in ("window", "key", "heard"):
        np.testing.assert_array_equal(getattr(first.longterm, name),
                                      getattr(second.longterm, name))
    a, b = first.longterm.stores[0], second.longterm.stores[0]
    assert a.count == b.count == 2
    np.testing.assert_array_equal(a.keys[:2], b.keys[:2])


def test_an_imagined_branch_with_turns_writes_nothing():
    b = turns_brain(longterm_rare=1.0)
    say(b, 0, 6)
    branch = b.longterm.branch()
    branch.hear(moment(TURN))
    branch.hear(moment(10))
    assert branch.stores[0].count == 0 and b.longterm.stores[0].count == 0
    assert b.longterm.since[0] == -1 and b.longterm.window.any()


# -- records read by what tells them apart (``longterm_distinct``, #305)


def distinct_brain(**settings):
    return turns_brain(longterm_distinct=0.5, longterm_heads=2, longterm_match=0.5, **settings)


def test_distinct_needs_turns_and_names_its_setting():
    b = distinct_brain()
    assert b.longterm.distinct == 0.5 and b.longterm.to_dict()["distinct"] == 0.5
    assert "distinct" not in turns_brain().longterm.to_dict()
    with pytest.raises(ValueError, match="longterm_turn"):
        cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, message=70, message_boundary=TURN,
                         message_boundary_holds=True, longterm=16, longterm_distinct=0.5)


def test_a_question_reads_the_latest_record_of_what_it_names():
    b = distinct_brain()
    b.reset()
    say(b, 0, 6, TURN, 10)   # agent 0 with verb 0: patient 10
    say(b, 1, 6, TURN, 11)   # agent 1
    say(b, 0, 7, TURN, 12)   # agent 0 again, other words: a record of its own
    kept = b.longterm
    assert kept.stores[0].count == 3 and len(kept.codebook) >= 5
    say(b, 0, 8, TURN)       # a question naming agent 0 in words no fact used
    drive = b.stimulus(np.zeros((1, INPUTS)))
    region = b.connectome.populations["longterm"]
    assert np.allclose(drive[0, region], 3.0 * moment(12)[0])  # the latest telling
    assert kept.read_slots[0, 0] == 2


def test_a_second_head_chains_from_what_the_first_record_is_about():
    b = distinct_brain()
    b.reset()
    say(b, 0, 6, TURN, 10)   # agent 0 goes to place 10
    say(b, 1, 7, TURN, 11)   # agent 1 goes to place 11
    say(b, 0, 8, TURN, 13)   # agent 0 takes object 13
    say(b, 0, 6, TURN, 12)   # agent 0 goes to place 12
    say(b, 13, TURN)         # where is object 13?
    drive = b.stimulus(np.zeros((1, INPUTS)))
    kept = b.longterm
    assert kept.read_slots[0, 0] == 2  # the object's record
    assert kept.read_slots[0, 1] == 3  # agent 0's latest other record
    assert np.allclose(drive[0, b.connectome.populations["longterm_1"]], 3.0 * moment(12)[0])


def test_a_saved_brain_with_distinct_records_continues_as_the_one_saved(tmp_path):
    first = distinct_brain()
    say(first, 0, 6, TURN, 10, 1, 7, TURN, 11, 0)
    second = cd.Brain.load(first.save(tmp_path / "distinct.npz"))
    for on in (8, TURN, 12, 1, TURN, 0, 13):
        x = moment(on)
        np.testing.assert_array_equal(first.act(x, greedy=True), second.act(x, greedy=True))
    for name in ("codebook", "sets", "stamps", "wset", "cue"):
        np.testing.assert_array_equal(getattr(first.longterm, name),
                                      getattr(second.longterm, name), err_msg=name)
    branch = second.longterm.branch()
    branch.hear(moment(9))
    assert len(second.longterm.codebook) == len(first.longterm.codebook)


# -- the record that holds the most, and what the question asks beyond it (#311)


def best_brain(**settings):
    return turns_brain(longterm_distinct=0.1, longterm_heads=2, longterm_match=0.5,
                       longterm_best=True, **settings)


def test_best_and_rest_need_distinct_and_name_their_settings():
    b = best_brain(longterm_rest=True)
    assert b.longterm.best and b.longterm.rest
    assert b.longterm.to_dict()["best"] is True and b.longterm.to_dict()["rest"] is True
    assert "best" not in distinct_brain().longterm.to_dict()
    assert b.describe()["genes"].get("longterm_best") is True
    with pytest.raises(ValueError, match="longterm_distinct"):
        turns_brain(longterm_best=True)
    with pytest.raises(ValueError, match="boolean"):
        turns_brain(longterm_distinct=0.1, longterm_best=1)


def test_the_best_read_takes_the_record_of_subject_and_relation():
    b = best_brain()
    b.reset()
    say(b, 0, 6, TURN, 10)   # agent 0, relation 6: 10
    say(b, 0, 7, TURN, 11)   # agent 0, relation 7: 11
    say(b, 1, 6, TURN, 12)   # agent 1, relation 6: 12
    say(b, 2, 6, TURN, 13)   # agent 2, relation 6: 13 (the latest record of relation 6)
    say(b, 0, 6, 9, TURN)    # relation 6 of agent 0, with a word no record holds
    drive = b.stimulus(np.zeros((1, INPUTS)))
    assert b.longterm.read_slots[0, 0] == 0
    assert np.allclose(drive[0, b.connectome.populations["longterm"]], 3.0 * moment(10)[0])
    latest = distinct_brain()  # the latest record above the match: agent 2's
    latest.reset()
    say(latest, 0, 6, TURN, 10, 0, 7, TURN, 11, 1, 6, TURN, 12, 2, 6, TURN, 13, 0, 6, 9, TURN)
    latest.stimulus(np.zeros((1, INPUTS)))
    assert latest.longterm.read_slots[0, 0] != 0


def test_the_best_read_takes_the_latest_telling_of_a_revised_fact():
    b = best_brain()
    b.reset()
    say(b, 0, 6, TURN, 10, 1, 6, TURN, 12, 0, 6, TURN, 11)  # agent 0's fact told again
    say(b, 0, 6, TURN)
    drive = b.stimulus(np.zeros((1, INPUTS)))
    assert b.longterm.read_slots[0, 0] == 2
    assert np.allclose(drive[0, b.connectome.populations["longterm"]], 3.0 * moment(11)[0])


def test_rest_reads_the_second_fact_of_a_chain_and_nothing_for_one_fact():
    b = best_brain(longterm_rest=True)
    b.reset()
    say(b, 0, 6, TURN, 10)   # agent 0, relation 6: 10
    say(b, 1, 6, TURN, 11)   # agent 1, relation 6: 11
    say(b, 10, 7, TURN, 12)  # 10, relation 7: 12
    say(b, 11, 7, TURN, 13)  # 11, relation 7: 13
    say(b, 0, 6, 7, TURN)    # relation 7 of relation 6 of agent 0
    drive = b.stimulus(np.zeros((1, INPUTS)))
    kept = b.longterm
    assert kept.read_slots[0, 0] == 2 and kept.read_slots[0, 1] == 0  # the answer on head 0
    assert np.allclose(drive[0, b.connectome.populations["longterm"]], 3.0 * moment(12)[0])
    assert np.allclose(drive[0, b.connectome.populations["longterm_1"]], 3.0 * moment(10)[0])
    say(b, ())               # the answer's silent moment writes nothing
    say(b, 1, 6, TURN)       # one fact: the first record holds the whole question
    drive = b.stimulus(np.zeros((1, INPUTS)))
    assert kept.read_slots[0, 0] == 1 and kept.read_slots[0, 1] == -1
    assert not drive[0, b.connectome.populations["longterm_1"]].any()


def moment_of(inputs: int, on: int) -> np.ndarray:
    x = np.zeros((1, inputs))
    x[0, on] = 1.0
    return x


def test_the_records_lists_grow_past_a_dense_table_and_read_the_same():
    """Thousands of records over thousands of entries: each read takes the record the dense
    table's arithmetic takes."""
    inputs, turn = 64, 63
    b = cd.Brain.compose(inputs, 4, modules=(16,), seed=0, backend="cpu", message=70,
                         longterm=3000, longterm_turn=turn, longterm_distinct=0.1,
                         longterm_heads=2, longterm_match=0.5, longterm_best=True)
    kept = b.longterm
    rng = np.random.default_rng(0)
    b.reset()
    for _ in range(2500):  # a subject of three channels, a relation, the value
        x = np.zeros((1, inputs))
        x[0, rng.choice(48, 3, replace=False)] = 1.0
        kept.hear(x)
        for on in (48 + int(rng.integers(8)), turn, 56 + int(rng.integers(7))):
            kept.hear(moment_of(inputs, on))
    assert kept.stores[0].count == 2500 and kept.entries > 2000
    values = {kept._entries[np.asarray(kept.message.code(moment_of(inputs, on))[0],
                                       dtype=np.uint8).tobytes()] for on in range(56, 63)}
    sets = np.zeros((2500, kept.entries))
    for slot, held in kept._members[0].items():
        sets[slot, held] = 1.0
    for slot in rng.choice(2500, 20, replace=False):
        cue = np.array([e for e in kept._members[0][int(slot)] if e not in values])
        weight = kept.weights(0) * np.isin(np.arange(kept.entries), cue)
        weight = weight * (weight >= 0.1 * weight.max() - 1e-12)
        held = sets @ weight / weight.sum()
        able = held >= max(0.5, held.max() - 1e-9) - 1e-12
        want = int(np.argmax(np.where(able, kept.stamps[0, :2500], -1.0)))
        assert kept._read_distinct(0, cue) == want


# -- each stream's store through the address of pairs (cadence#310)

PAIRS = {"longterm_pairs": 8, "longterm_expansion": 256, "longterm_winners": 32,
         "longterm_candidates": 8, "longterm_fade": False}


def test_records_read_the_same_through_the_address_of_pairs_as_through_the_scan(tmp_path):
    scanned, paired = brain(longterm=600), brain(longterm=600, **PAIRS)
    assert scanned.longterm.batched and not paired.longterm.batched
    assert paired.longterm.address() == {name[len("longterm_"):]: value
                                         for name, value in PAIRS.items()}
    facts = [(agent, verb, (agent * 3 + verb) % 4) for agent in range(6) for verb in range(4)]
    for b in (scanned, paired):
        for fact in facts:
            tell(b, *fact)
        tell(b, 2, 1, 0)  # a fact told again with another patient revises its record
    store = paired.longterm.stores[0]
    assert store.coarse and store.pairs == 8 and store.count == 24 > store.candidates
    assert not store.fade and scanned.longterm.stores[0].count == 24
    assert paired.longterm.counts == scanned.longterm.counts
    for agent, verb, _ in facts:
        ask(scanned, agent, verb)
        ask(paired, agent, verb)
        np.testing.assert_array_equal(paired.longterm.read_slots, scanned.longterm.read_slots)
    np.testing.assert_array_equal(store.values[:24], scanned.longterm.stores[0].values[:24])
    path = tmp_path / "paired.npz"
    paired.save(path)
    loaded = cd.Brain.load(path)
    assert loaded.longterm.address() == paired.longterm.address()
    np.testing.assert_array_equal(loaded.longterm.stores[0].cells, store.cells)
    for agent, verb, _ in facts[:6]:
        ask(paired, agent, verb)
        ask(loaded, agent, verb)
        np.testing.assert_array_equal(loaded.longterm.read_slots, paired.longterm.read_slots)


def test_the_store_settings_are_named_checked_and_off_by_default():
    assert brain(longterm=16).longterm.address() == {}
    assert brain(longterm=16, longterm_pairs=0, longterm_fade=True).longterm.address() == {}
    with pytest.raises(ValueError, match="at least 2"):
        brain(longterm=16, longterm_pairs=1)
    with pytest.raises(ValueError, match="boolean"):
        brain(longterm=16, longterm_fade=1)
    with pytest.raises(ValueError):
        brain(longterm=16, longterm_winners=0)
    with pytest.raises(ValueError, match="needs long-term records"):
        cd.Brain.compose(INPUTS, 4, seed=0, backend="cpu", longterm_pairs=8)


def test_rest_with_one_head_holds_the_answering_record_alone():
    b = turns_brain(longterm_distinct=0.1, longterm_match=0.5, longterm_best=True,
                    longterm_rest=True)
    assert b.longterm.heads == 1 and "longterm_1" not in b.connectome.populations
    b.reset()
    say(b, 0, 6, TURN, 10, 1, 6, TURN, 11, 10, 7, TURN, 12, 11, 7, TURN, 13)
    say(b, 0, 6, 7, TURN)    # relation 7 of relation 6 of agent 0
    drive = b.stimulus(np.zeros((1, INPUTS)))
    assert b.longterm.read_slots[0, 0] == 2
    assert np.allclose(drive[0, b.connectome.populations["longterm"]], 3.0 * moment(12)[0])
    say(b, ())
    say(b, 1, 6, TURN)       # one fact
    b.stimulus(np.zeros((1, INPUTS)))
    assert b.longterm.read_slots[0, 0] == 1


def test_entries_saved_as_tables_load_as_the_lists_they_were():
    """A file that keeps the records' entries as tables (0.97.1's ``sets``, ``wset``, ``cue``)
    loads into the lists."""
    first = distinct_brain()
    say(first, 0, 6, TURN, 10, 1, 7, TURN, 11, 0, 8)
    kept = first.longterm
    saved = kept.arrays()
    tables = {k: v for k, v in saved.items()
              if k not in ("held", "window_entries", "cue_entries")}
    tables.update(sets=kept.sets, wset=kept.wset, cue=kept.cue)
    second = distinct_brain()
    second.longterm.take(tables, len(kept.stores), kept.counts)
    for name in ("sets", "wset", "cue", "codebook", "stamps"):
        np.testing.assert_array_equal(getattr(second.longterm, name), getattr(kept, name),
                                      err_msg=name)


# -- records at any size: a table that grows, files, a read that ends at its record, and facts
# told apart by what was heard (cadence#310)

FACTS = [(agent, verb, (agent * 3 + verb) % 4) for agent in range(6) for verb in range(4)]


def test_records_that_grow_read_as_the_sized_table_s_and_save_under_their_own_format(tmp_path):
    sized = brain(longterm=600, **PAIRS)
    grown = brain(longterm=600, **PAIRS, longterm_grow=True, longterm_sure=0.999)
    assert grown.longterm.address() == {
        **{name[len("longterm_"):]: value for name, value in PAIRS.items()},
        "grow": True, "sure": 0.999}
    for b in (sized, grown):
        for fact in FACTS:
            tell(b, *fact)
        tell(b, 2, 1, 0)  # told again with another patient: revised in place
    store = grown.longterm.stores[0]
    assert store.grow and store.sure == 0.999 and store.count == 24 > store.candidates
    assert grown.longterm.counts == sized.longterm.counts
    assert int((store.cells[: store._level + store._split] != store.EMPTY).sum()) == store.entries
    for agent, verb, _ in FACTS:
        ask(sized, agent, verb)
        ask(grown, agent, verb)
        np.testing.assert_array_equal(grown.longterm.read_slots, sized.longterm.read_slots)
    assert _format(sized.save(tmp_path / "sized.npz")) == "cadence-brain/4"
    path = grown.save(tmp_path / "grown.npz")
    assert _format(path) == "cadence-brain/5"  # a name 0.99.0 and earlier refuse
    loaded = cd.Brain.load(path)
    assert loaded.longterm.to_dict() == grown.longterm.to_dict()
    again = loaded.longterm.stores[0]
    assert (again._level, again._split, again.entries) == (store._level, store._split,
                                                           store.entries)
    for b in (grown, loaded):
        tell(b, 5, 3, 2)  # the loaded brain's table grows as the saved one's does
    for agent, verb, _ in FACTS[:8] + [(5, 3, 2)]:
        ask(grown, agent, verb)
        ask(loaded, agent, verb)
        np.testing.assert_array_equal(loaded.longterm.read_slots, grown.longterm.read_slots)
    np.testing.assert_array_equal(again.cells[: again._level + again._split],
                                  store.cells[: store._level + store._split])


def test_records_on_disk_stay_in_their_files_and_a_checkpoint_names_them(tmp_path):
    import json

    folder = tmp_path / "records"
    settings = {**PAIRS, "longterm_grow": True, "longterm_sure": 0.999}
    memory = brain(longterm=600, **settings)
    disk = brain(longterm=600, **settings, longterm_path=folder)
    assert disk.longterm.address()["path"] == str(folder)
    for b in (memory, disk):
        for fact in FACTS:
            tell(b, *fact)
    assert (folder / "stream-0" / "keys.npy").exists()
    for agent, verb, _ in FACTS:
        ask(memory, agent, verb)
        ask(disk, agent, verb)
        np.testing.assert_array_equal(disk.longterm.read_slots, memory.longterm.read_slots)
    path = disk.save(tmp_path / "disk.npz", compressed=False)
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive)
        saved = json.loads(str(archive["brain"]))
    assert saved["format"] == "cadence-brain/5"
    assert "memories/longterm/store/0/state" in names
    assert not {n for n in names if n.endswith(("store/0/keys", "store/0/values"))}
    assert saved["memories"]["longterm"]["folders"] == ["stream-0"]
    loaded = cd.Brain.load(path)  # the records are the files': the loaded brain continues them
    assert loaded.longterm.stores[0].count == 24
    tell(loaded, 5, 3, 2)
    loaded.longterm.stores[0].flush()
    for agent, verb, _ in FACTS[:6]:
        ask(disk, agent, verb)
        ask(loaded, agent, verb)
        np.testing.assert_array_equal(loaded.longterm.read_slots, disk.longterm.read_slots)
    # a brain made over the directory continues its records; other settings are refused
    over = brain(longterm=600, **settings, longterm_path=folder)
    ask(over, 0, 0)
    assert over.longterm.stores[0].count == 24 and over.longterm.read_slots[0, 0] == 0
    other = brain(longterm=700, **settings, longterm_path=folder)
    with pytest.raises(ValueError, match="other settings"):
        ask(other, 0, 0)
    unopened = brain(longterm=600, **settings, longterm_path=folder)
    unopened.longterm.clear()  # the records and their files, of streams it never opened too
    assert not (folder / "stream-0").exists() and unopened.longterm.stores == []
    tell(unopened, 1, 1, 3)
    assert unopened.longterm.stores[0].count == 1  # a new store in the emptied directory
    unopened.longterm.clear()
    with pytest.raises(ValueError, match="the files of stream 0"):
        cd.Brain.load(path)


def test_facts_whose_keys_are_one_key_are_told_apart_by_what_was_heard(tmp_path):
    one = np.zeros(70)
    one[[3, 11, 29, 40, 55]] = 1.0  # every episode's held message gives this key

    def same_key(b):
        b.message.key = lambda: np.tile(one, (len(b.message.trace), 1))
        return b

    plain = same_key(brain(longterm=64))
    apart = same_key(brain(longterm=64, longterm_apart=True))
    assert apart.longterm.address() == {"apart": True} and not apart.longterm.batched
    for b in (plain, apart):
        for fact in FACTS:
            tell(b, *fact)
    assert plain.longterm.stores[0].count == 1  # one key: one record, the last telling's
    kept = apart.longterm
    store = kept.stores[0]
    assert store.apart == INPUTS and store.count == 24 and kept.counts["revisions"] == 0
    assert store.same == store.SIGN  # the heard moments are told exactly: the signature whole
    for agent, verb, patient in FACTS:  # each question reads the record of its own fact
        ask(apart, agent, verb)
        slot = int(kept.read_slots[0, 0])
        np.testing.assert_array_equal(store.values[slot], moment(10 + patient)[0])
    tell(apart, 2, 1, 0)  # a fact told again with another patient revises its own record
    tell(apart, 4, 0, FACTS[16][2])  # and one told again as it was writes nothing
    assert store.count == 24 and kept.counts["revisions"] == 1 and kept.counts["agreed"] == 1
    ask(apart, 2, 1)
    np.testing.assert_array_equal(store.values[int(kept.read_slots[0, 0])], moment(10)[0])
    path = apart.save(tmp_path / "apart.npz")
    assert _format(path) == "cadence-brain/5"
    loaded = same_key(cd.Brain.load(path))
    np.testing.assert_array_equal(loaded.longterm.stores[0].signs[:24], store.signs[:24])
    for agent, verb, _ in FACTS[:6]:
        ask(apart, agent, verb)
        ask(loaded, agent, verb)
        np.testing.assert_array_equal(loaded.longterm.read_slots, kept.read_slots)


def test_the_settings_of_records_at_any_size_are_checked_and_off_by_default(tmp_path):
    plain = brain(longterm=16).longterm
    assert (plain.grow, plain.path, plain.sure, plain.apart, plain.wide) == (
        False, None, 0.0, False, False)
    for settings in ({"longterm_grow": True}, {"longterm_sure": 0.9},
                     {"longterm_grow": 1, **PAIRS}, {"longterm_sure": 1.5, **PAIRS},
                     {"longterm_apart": 1}, {"longterm_path": 3}):
        with pytest.raises(ValueError):
            brain(longterm=600, **settings)
    with pytest.raises(ValueError, match="longterm_turn"):
        cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, backend="cpu", message=70,
                         longterm=16, longterm_turn=TURN, longterm_apart=True)
    with pytest.raises(ValueError, match="needs long-term records"):
        cd.Brain.compose(INPUTS, 4, seed=0, backend="cpu", longterm_path=tmp_path)
    with pytest.raises(ValueError, match="longterm_turn"):  # turns' entries live in memory
        cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, backend="cpu", message=70,
                         longterm=16, longterm_turn=TURN, longterm_path=tmp_path / "turns")
    sure = brain(longterm=600, **PAIRS, longterm_sure=0.999)  # a read that ends early alone
    assert sure.longterm.wide
    tell(sure, 1, 2, 3)
    assert _format(sure.save(tmp_path / "sure.npz")) == "cadence-brain/5"


def test_streams_a_batch_drops_take_their_files_with_them(tmp_path):
    folder = tmp_path / "records"
    kept = brain(longterm=600, **PAIRS, longterm_grow=True, longterm_path=folder)
    kept.reset()
    for on in ((0, 1, 2), (6, 7, 8), (TURN, TURN, TURN), (10, 11, 12)):
        x = np.zeros((3, INPUTS))
        x[[0, 1, 2], list(on)] = 1.0
        kept.act(x, greedy=True)  # three streams, a fact each
    records = kept.longterm
    assert [s.count for s in records.stores] == [1, 1, 1]
    records.keep(np.array([0, 2]))  # stream 1 ends, and its files with it
    assert [s.count for s in records.stores] == [1, 1]
    assert not (folder / "stream-1").exists() and records._folders == ["stream-0", "stream-2"]
    records._streams(3)  # a new stream begins empty
    assert [s.count for s in records.stores] == [1, 1, 0]
