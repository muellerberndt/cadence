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
        longterm=64, longterm_turn=TURN, longterm_distinct=0.1, longterm_match=0.6,
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
    """Twenty-four facts: relation ``R`` of every subject (record ``s`` holds subject ``s``'s),
    ``R2`` of subjects 0 to 5 (records 12 to 17), and a link from each of subjects 6 to 11
    (records 18 to 23; subject 6's link is subject 2)."""
    for subject in range(12):
        tell(b, R, subject, 16 + subject % 4)
    for subject in range(6):
        tell(b, R2, subject, 16 + (subject + 1) % 4)
    for subject in range(6, 12):
        tell(b, LINK, subject, subject - 4)
    # one question heard: "and", a word no record holds, is a word of questions from here on
    b.reset()
    for on in (R, OF, 10, AND, 11, TURN, None):
        b.act(moment(on), greedy=True)


def test_the_option_is_named_checked_and_off_by_default():
    plain = brain(longterm_attention=None)
    assert plain.longterm.attention is None and plain.longterm.attended is None
    assert "attention" not in plain.longterm.to_dict()
    assert "longterm_attention" not in plain.describe()["genes"]
    b = brain()
    assert isinstance(b.longterm.attended, RecordHeads) and b.longterm.attention == 1.0
    assert b.longterm.to_dict()["attention"] == 1.0
    assert b.describe()["genes"]["longterm_attention"] == 1.0
    assert not any(m.any() or bias.any() for m, bias in b.longterm.attended.maps)  # one half
    assert b.parameters() - plain.parameters() == HEADS * (70 + 1) * (2 * RecordHeads.POSITIONS + 1)
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
    held = [int(s) for s in kept.read_slots[0]]
    assert held == [0, 1, 2]  # the records the question names, in its order, each on one head
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
    assert [int(s) for s in kept.read_slots[0]] == [9, -1, -1]
    for head in (1, 2):
        assert not drive[0, b.connectome.populations[kept.ports(HEADS)[head]]].any()
    ask(b, R, OF, 24)  # a word no fact holds: every head holds nothing
    assert (kept.read_slots[0] == -1).all()
    ask(b, R, OF, 0, AND, 24, AND, 2)  # the head of the unknown word holds nothing
    assert [int(s) for s in kept.read_slots[0]] == [0, -1, 2]
    assert kept.questions == 1  # a question is counted once its answer's moment is heard
    assert kept.asked_in[kept.entry(b.message.code(moment(AND))[0])] == 1


def test_a_record_is_found_by_its_key_and_not_by_its_value():
    b = brain()
    told(b)
    ask(b, R, OF, 2)  # subject 2 is the subject of two records and the value of a link's
    kept = b.longterm
    assert [int(s) for s in kept.read_slots[0]] == [2, -1, -1]  # a subject is read once
    assert 18 not in kept.read_slots[0]  # the link's record names subject 2 only as its value
    assert kept.said(0, 18) == kept.entry(b.message.code(moment(2))[0])


def test_a_chain_lies_on_its_regions_from_its_end():
    b = brain(heads=2)
    told(b)
    heads = b.longterm.attended
    # the first head attends the link and its subject, the second the relation and the chain
    # the first head holds to the link and its subject, the second to the relation
    for head, places in ((0, (2, 4)), (1, (0,))):
        m, bias = heads.maps[head]
        bias[:] = -8.0
        bias[list(places)] = 16.0  # over the term by the place from the question's end
    drive = ask(b, R, OF, LINK, OF, 6)  # relation R of the link of subject 6: of subject 2
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
    # a question of two context words (a relation and a link): which one a head holds to is
    # what its gates choose (with one context word every gate reads alike)
    asked = (R, OF, LINK, OF, 6, TURN)
    for _ in range(6):
        b.reset()
        for on in asked:
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
    for on in asked:
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
    assert [int(s) for s in second.longterm.read_slots[0]] == [2, 1, -1]


def test_an_imagined_branch_reads_with_the_heads_and_counts_nothing():
    b = brain()
    told(b)
    b.reset()
    for on in (R, OF, 0, AND, 1):
        b.act(moment(on), greedy=True)
    kept = b.longterm
    branch = kept.branch()
    assert branch.attended is not kept.attended and branch.attended.records is branch
    assert branch.attended.maps is kept.attended.maps
    branch.hear(moment(TURN))
    drive = np.zeros((1, b.connectome.n))
    branch.stimulus(drive)
    assert [int(s) for s in branch.read_slots[0]] == [0, 1, -1]
    branch.hear(moment(None))
    assert kept.questions == 1 and branch.questions == 1 and not kept.attended.last
    assert kept.since[0] == -1 and len(kept._asked[0]) == 0


def test_a_cue_from_the_brains_own_state_fetches_records_and_their_context():
    b = brain()
    told(b)
    kept = b.longterm
    heads = kept.attended
    deep = np.array(kept.stores[0].deep, copy=True)
    counts, read_slots, last = dict(kept.counts), kept.read_slots.copy(), dict(heads.last)
    entry = kept.entry(b.message.code(moment(R))[0])
    # what the brain attends (subject 3) with a relation as context: its record of that relation
    found = heads.fetch(moment(3)[0], context=[moment(R)[0]])
    assert [f.slot for f in found] == [3, -1, -1]
    np.testing.assert_array_equal(found[0].content, moment(16 + 3 % 4)[0])
    assert entry in found[0].context and found[0].level == 1.0
    assert found[0].content is not kept.stores[0].values[3]
    # the other relation of the same subject, the cue given as codebook entries
    other = kept.entry(b.message.code(moment(R2))[0])
    focus = kept.entry(b.message.code(moment(2)[0][None])[0])
    assert heads.fetch(focus, context=[other])[0].slot == 14
    # a chain: subject 6's link, then that subject's record of the relation
    chain = heads.fetch(moment(6)[0], context=[moment(R)[0], moment(LINK)[0]])
    assert [f.slot for f in chain[:2]] == [18, 2]
    np.testing.assert_array_equal(chain[1].content, moment(16 + 2 % 4)[0])
    # a moment no record holds fetches nothing, and nothing changes
    assert all(f.slot == -1 and not f.content.any() for f in heads.fetch(moment(24)[0]))
    np.testing.assert_array_equal(kept.stores[0].deep, deep)
    assert kept.counts == counts and np.array_equal(kept.read_slots, read_slots)
    assert heads.last == last
    with pytest.raises(ValueError, match="cue"):
        heads.fetch(np.zeros(3))
