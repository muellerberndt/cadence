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
