"""Grouped motor slots on the composed brain (issue 142): one softmax per slot, lateral
inhibition within a slot only, and the grouping carried through save and load."""

import numpy as np
import pytest

import cadence as cd
from cadence.regions import motor_cortex, slot_sizes


def motor_pairs(brain):
    """The (pre, post) motor-to-motor synapses of a brain."""
    c = brain.connectome
    motor = np.zeros(c.n, dtype=bool)
    motor[brain.motor_index] = True
    edges = motor[c.pre] & motor[c.post]
    return list(zip(c.pre[edges].tolist(), c.post[edges].tolist(), strict=True))


def test_slot_sizes_cover_the_actions():
    assert slot_sizes(25, 1) == [25]
    assert slot_sizes(25, 5) == [5, 5, 5, 5, 5]
    assert slot_sizes(25, (3, 13, 9)) == [3, 13, 9]
    for bad in (4, (3, 13), (3, 13, 9, 1), (0, 25), True, "3"):
        with pytest.raises(ValueError, match="slot"):
            slot_sizes(25, bad)


def test_slots_group_the_motor_readout():
    brain = cd.Brain.compose(inputs=6, actions=25, modules=(8,), seed=0, slots=(3, 13, 9), recall=False)
    assert brain.learner.slot_sizes.tolist() == [3, 13, 9]
    action = brain.act(np.full((2, 6), 0.3))
    assert action.shape == (2, 3)
    assert (action >= 0).all() and (action < np.array([3, 13, 9])).all()
    p = brain.basal_ganglia.probabilities(brain.basal_ganglia.state)
    assert p.shape == (2, 3, 13)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0)


def test_lateral_inhibition_stays_within_a_slot():
    brain = cd.Brain.compose(
        inputs=6, actions=25, modules=(8,), seed=0, slots=(3, 13, 9), lateral=-0.5, recall=False
    )
    motor = brain.motor_index.tolist()
    group = {}
    start = 0
    for g, size in enumerate((3, 13, 9)):
        for i in range(start, start + size):
            group[motor[i]] = g
        start += size
    pairs = motor_pairs(brain)
    assert len(pairs) == 3 * 2 + 13 * 12 + 9 * 8
    assert all(group[p] == group[q] for p, q in pairs)


def test_an_unset_lateral_wires_no_competition_in_any_slot():
    wide = cd.Brain.compose(inputs=6, actions=25, modules=(8,), seed=0, slots=(3, 13, 9), recall=False)
    assert motor_pairs(wide) == []
    narrow = cd.Brain.compose(inputs=6, actions=25, modules=(8,), seed=0, slots=5, recall=False)
    assert motor_pairs(narrow) == []
    competing = cd.Brain.compose(
        inputs=6, actions=25, modules=(8,), seed=0, slots=5, lateral=-0.5, recall=False
    )
    assert len(motor_pairs(competing)) == 5 * 5 * 4  # an explicit weight stays within each slot
    region = motor_cortex(10, lateral=-0.5, slots=2)
    assert region.circuit is not None and region.circuit.synapses == 2 * 5 * 4


def test_one_slot_is_the_unchanged_default():
    a = cd.Brain.compose(inputs=6, actions=4, modules=(8,), seed=0, recall=False)
    b = cd.Brain.compose(inputs=6, actions=4, modules=(8,), seed=0, slots=1, recall=False)
    for name in ("pre", "post", "sign"):
        np.testing.assert_array_equal(getattr(a.connectome, name), getattr(b.connectome, name))
    np.testing.assert_array_equal(a.brain.efficacy, b.brain.efficacy)
    assert a.learner.slot_sizes.tolist() == [4]
    assert a.act(np.ones((1, 6))).shape == (1,)


def test_slots_are_validated_at_composition():
    for bad in (4, (3, 13), True):
        with pytest.raises(ValueError, match="slot"):
            cd.Brain.compose(inputs=6, actions=25, modules=(8,), slots=bad, recall=False)


def test_compose_and_the_composed_layout_take_slots():
    brain = cd.Brain.compose(6, 25, modules=(8,), slots=(3, 13, 9), seed=0, recall=False)
    assert brain.learner.slot_sizes.tolist() == [3, 13, 9]
    assert motor_pairs(brain) == []
    genome = cd.Genome.composed(6, 10, modules=(8,), slots=2, lateral=-0.5)
    grouped = cd.Brain(cd.develop(genome, seed=0), slots=2)
    assert grouped.learner.slot_sizes.tolist() == [5, 5]
    assert len(motor_pairs(grouped)) == 2 * 5 * 4


def test_save_and_load_carry_the_grouping(tmp_path):
    brain = cd.Brain.compose(inputs=6, actions=25, modules=(8,), seed=0, slots=(3, 13, 9), recall=False)
    rng = np.random.default_rng(0)
    x = rng.random((2, 6))
    brain.step(x)
    brain.step(x, reward=np.array([1.0, -1.0]), done=np.array([False, False]))
    assert brain.last_learning  # the slotted action learned and wrote its memory
    path = brain.save(tmp_path / "slotted.npz")
    loaded = cd.Brain.load(path)
    assert loaded.learner.slot_sizes.tolist() == [3, 13, 9]
    assert loaded.basal_ganglia.learner is loaded.learner
    probe = rng.random((2, 6))
    expected = brain.act(probe, greedy=True)
    np.testing.assert_array_equal(loaded.act(probe, greedy=True), expected)
    assert expected.shape == (2, 3)
