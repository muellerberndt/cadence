"""``Brain.compose(longterm_attention=rate)`` (cadence#325): each head of the brain's long-term
records reads a record of its own in one answer, where each head looks is a map its lessons
step, a head with no record holds nothing, and a chain lies on its regions from its end.
Without the option a brain is the brain without it."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.record_heads import RecordHeads

INPUTS = 32  # subjects 0-11, relations 12 and 13, a link 14, values 16-19, "and", "of", the turn
R, R2, LINK, AND, OF, TURN = 12, 13, 14, 20, 21, 31
HEADS = 3


def moment(on: int | None) -> np.ndarray:
    x = np.zeros((1, INPUTS))
    if on is not None:
        x[0, on] = 1.0
    return x


def brain(heads: int = HEADS, **settings):
    made = {"longterm_attention": 1.0, **settings}
    return cd.Brain.compose(
        INPUTS, heads * 5, slots=heads, modules=(32,), seed=0, backend="cpu", message=70,
        longterm=64, longterm_turn=TURN, longterm_distinct=0.1, longterm_match=0.5,
        longterm_heads=heads, **{k: v for k, v in made.items() if v is not None},
    )


def tell(b, relation: int, subject: int, value: int) -> None:
    """One told fact as its own episode: "relation of subject <turn> value"."""
    b.reset()
    for on in (relation, OF, subject, TURN, value):
        b.act(moment(on), greedy=True)


def ask(b, *words: int) -> np.ndarray:
    """A question as its own episode, up to the read of its answer's moment: the drive."""
    b.reset()
    for on in (*words, TURN):
        b.act(moment(on), greedy=True)
    return np.asarray(b.stimulus(moment(None)))


def told(b) -> None:
    """Twenty facts: relation ``R`` of every subject (record ``s`` holds subject ``s``'s),
    ``R2`` of subjects 0 to 5 (records 12 to 17), and two links (records 18 and 19)."""
    for subject in range(12):
        tell(b, R, subject, 16 + subject % 4)
    for subject in range(6):
        tell(b, R2, subject, 16 + (subject + 1) % 4)
    tell(b, LINK, 1, 2)
    tell(b, LINK, 4, 5)


def test_the_option_is_named_checked_and_off_by_default():
    plain = brain(longterm_attention=None)
    assert plain.longterm.attention is None and plain.longterm.attended is None
    assert "attention" not in plain.longterm.to_dict()
    assert "longterm_attention" not in plain.describe()["genes"]
    b = brain()
    assert isinstance(b.longterm.attended, RecordHeads) and b.longterm.attention == 1.0
    assert b.longterm.to_dict()["attention"] == 1.0
    assert b.describe()["genes"]["longterm_attention"] == 1.0
    assert all(bias[-1] == RecordHeads.BAR and not bias[:-1].any() and not m.any()
               for m, bias in b.longterm.attended.maps)  # a newborn head: every gate one half
    assert b.parameters() - plain.parameters() == HEADS * (70 + 1) * (2 * RecordHeads.POSITIONS + 2)
    with pytest.raises(ValueError, match="longterm_distinct"):
        cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, message=70, longterm=16,
                         longterm_turn=TURN, longterm_attention=1.0)
    with pytest.raises(ValueError, match="longterm_attention"):
        brain(longterm_attention=-1.0)


def test_without_the_option_the_brain_reads_and_answers_as_it_did():
    plain, named = brain(longterm_attention=None), brain(longterm_attention=None)
    for b in (plain, named):
        told(b)
    a, c = ask(plain, R, OF, 0), ask(named, R, OF, 0)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(plain.longterm.read_slots, named.longterm.read_slots)
    assert plain.longterm.work is None and not plain.longterm._wlists


def test_each_head_reads_a_record_of_its_own_and_a_record_is_read_once():
    b = brain()
    told(b)
    drive = ask(b, R, OF, 0, AND, 1, AND, 2)  # relation R of subjects 0, 1 and 2
    kept = b.longterm
    held = sorted(int(s) for s in kept.read_slots[0])
    assert held == [0, 1, 2]  # the three records the question names, each on one head
    for head, name in enumerate(kept.ports(HEADS)):
        slot = int(kept.read_slots[0, head])
        region = b.connectome.populations[name]
        np.testing.assert_allclose(drive[0, region], 3.0 * kept.stores[0].values[slot])
    assert kept.work is not None and kept.work > 0


def test_a_head_with_no_record_holds_nothing():
    b = brain()
    told(b)
    drive = ask(b, R, OF, 9)  # one part: subject 9 holds one record
    kept = b.longterm
    assert sorted(int(s) for s in kept.read_slots[0]) == [-1, -1, 9]
    silent = [h for h in range(HEADS) if kept.read_slots[0, h] < 0]
    for head in silent:
        assert not drive[0, b.connectome.populations[kept.ports(HEADS)[head]]].any()
    ask(b, R, OF, 24)  # a word no fact holds: every head holds nothing
    assert (kept.read_slots[0] == -1).all()
    ask(b, 25, 26)  # words no record holds
    assert (kept.read_slots[0] == -1).all()


def test_a_record_is_found_by_its_key_and_not_by_its_value():
    b = brain()
    told(b)
    ask(b, R, OF, 2)  # subject 2 is the subject of two records and the value of a link's
    kept = b.longterm
    assert sorted(int(s) for s in kept.read_slots[0]) == [-1, -1, 2]  # a subject is read once
    assert 18 not in kept.read_slots[0]  # the link's record names subject 2 only as its value
    ask(b, R, AND, R2, OF, 2)  # two relations of one subject: a record for each
    assert sorted(int(s) for s in kept.read_slots[0]) == [-1, 2, 14]
    assert kept.said(0, 18) == kept.entry(b.message.code(moment(2))[0])


def test_a_chain_lies_on_its_regions_from_its_end():
    b = brain(heads=2)
    told(b)
    heads = b.longterm.attended
    # the first head attends the link and its subject, the second the relation and the chain
    for head, places in ((0, (2, 4)), (1, (0, heads.chain))):
        m, bias = heads.maps[head]
        bias[: heads.chain + 1] = -8.0
        bias[list(places)] = 16.0  # over the term by the place from the question's end
    drive = ask(b, R, OF, LINK, OF, 1)  # relation R of the link of subject 1: of subject 2
    kept = b.longterm
    assert [int(s) for s in kept.read_slots[0]] == [2, 18]  # the answering record on head 0
    np.testing.assert_allclose(drive[0, b.connectome.populations["longterm"]],
                               3.0 * moment(18)[0])
    assert heads.last[0].picks == [18, 2]  # read in the chain's order


def test_a_lesson_steps_the_heads_maps_and_a_frozen_rate_holds_them():
    b = brain()
    told(b)
    before = [m.copy() for m, _ in b.longterm.attended.maps]
    moved = 0.0
    for _ in range(6):
        b.reset()
        for on in (R, OF, 0, AND, 1, TURN):
            b.act(moment(on), greedy=True)
        report = b.lesson(moment(None), np.array([[2, 3, 4]]))
        moved += report["heads_moved"]
        assert report["heads_refused"] == 0.0
        b.act(moment(None), greedy=True)
    assert moved > 0
    assert any(not np.array_equal(m, old)
               for (m, _), old in zip(b.longterm.attended.maps, before, strict=True))
    frozen = brain(longterm_attention=0.0)
    told(frozen)
    frozen.reset()
    for on in (R, OF, 0, AND, 1, TURN):
        frozen.act(moment(on), greedy=True)
    assert frozen.lesson(moment(None), np.array([[2, 3, 4]]))["heads_moved"] == 0.0
    assert not any(m.any() for m, _ in frozen.longterm.attended.maps)


def test_the_regions_reach_the_motor_region_where_named():
    plain, direct = brain(), brain(longterm_motor=1.2)
    assert plain.longterm.motor == 0.0 and direct.longterm.motor == 1.2
    assert "motor" not in plain.longterm.to_dict() and direct.longterm.to_dict()["motor"] == 1.2
    assert direct.describe()["genes"]["longterm_motor"] == 1.2
    assert "longterm_motor" not in plain.describe()["genes"]
    motor = direct.connectome.populations["motor"]
    for b, reaches in ((plain, False), (direct, True)):
        pre, post = np.asarray(b.connectome.pre), np.asarray(b.connectome.post)
        for name in b.longterm.ports(HEADS):
            region = b.connectome.populations[name]
            assert bool(np.isin(post[np.isin(pre, region)], motor).any()) is reaches
    # the brain without the pathway is the brain with it less its synapses
    assert direct.connectome.n == plain.connectome.n
    assert len(direct.connectome.pre) == len(plain.connectome.pre) + HEADS * INPUTS * len(motor)
    with pytest.raises(ValueError, match="longterm_motor"):
        brain(longterm_motor=-1.0)


def test_a_saved_brain_with_learned_heads_continues_as_the_one_saved(tmp_path):
    first = brain()
    told(first)
    for _ in range(3):
        first.reset()
        for on in (R, OF, 0, AND, 1, TURN):
            first.act(moment(on), greedy=True)
        first.lesson(moment(None), np.array([[2, 3, 4]]))
        first.act(moment(None), greedy=True)
    first.reset()
    for on in (R, OF, 2, AND):
        first.act(moment(on), greedy=True)
    second = cd.Brain.load(first.save(tmp_path / "heads.npz"))
    for (m, bias), (n, other) in zip(first.longterm.attended.maps,
                                     second.longterm.attended.maps, strict=True):
        np.testing.assert_array_equal(m, n)
        np.testing.assert_array_equal(bias, other)
    assert second.longterm._wlists == first.longterm._wlists
    for on in (1, TURN, None):
        x = moment(on)
        np.testing.assert_array_equal(first.act(x, greedy=True), second.act(x, greedy=True))
        np.testing.assert_array_equal(first.longterm.read_slots, second.longterm.read_slots)
    assert {1, 2} <= {int(s) for s in second.longterm.read_slots[0]}
