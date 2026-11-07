"""The executed command in the recall memory (``recall_action``): every written slot carries the
command the stream executed at that moment in its key and its value, read on command regions
whose synapses start at zero, and the maps read the command of the moment before. A newborn
settles as one without the option. The founder has none; the option is described and saved
where it is on.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd

INPUTS, ACTIONS = 5, 3


def composed(**options):
    options.setdefault("modules", (16,))
    return cd.Brain.compose(INPUTS, ACTIONS, seed=2, episodic=False, recall=True, recall_founder=True, **options)


def event(i):
    return np.eye(INPUTS)[[i]]


def test_the_founder_keeps_no_command_and_the_option_is_described_where_on():
    founder = composed()
    assert founder.recall.action is False and founder.recall.command_width == 0
    assert "recall_action" not in founder.describe()["genes"]
    assert not any(name.endswith("_action") for name in founder.connectome.populations)
    kept = composed(recall_action=True)
    assert kept.describe()["genes"]["recall_action"] is True
    assert {"recall_action", "next_action"} <= set(kept.connectome.populations)
    with pytest.raises(ValueError, match="action must be boolean"):
        composed(recall_action=1)
    with pytest.raises(ValueError, match="construction setting"):
        kept.retune(recall_action=False)


def test_a_newborn_settles_as_one_without_the_command():
    founder, kept = composed(), composed(recall_action=True)
    action = kept.connectome.populations["recall_action"]
    pre = np.asarray(kept.connectome.pre)
    assert not kept.brain.efficacy[np.isin(pre, action)].any()  # zero synapses, every one
    rng = np.random.default_rng(0)
    for _ in range(12):
        x = event(int(rng.integers(INPUTS)))
        assert (founder.act(x, greedy=True) == kept.act(x, greedy=True)).all()
        np.testing.assert_allclose(
            kept.basal_ganglia.state.activation[0, kept.motor_index],
            founder.basal_ganglia.state.activation[0, founder.motor_index],
        )
        if founder.recall.attended_heads is not None:  # the first moment finds no slot
            np.testing.assert_allclose(kept.recall.attended_heads[0], founder.recall.attended_heads[0])


def test_every_slot_carries_the_command_executed_at_its_moment():
    brain = composed(recall_action=True, arousal=True)
    a = [int(brain.act(event(i), greedy=True)[0]) for i in (0, 1, 2)]
    recall, width = brain.recall, INPUTS
    commands = np.eye(ACTIONS)[a]
    np.testing.assert_array_equal(recall.values[0, :, width:], commands)
    np.testing.assert_array_equal(recall.keys[0, :, width:], commands)
    np.testing.assert_array_equal(recall.values[0, :, :width], np.eye(INPUTS)[[0, 1, 2]])
    np.testing.assert_array_equal(recall.nexts[0, :2], recall.values[0, 1:])
    np.testing.assert_array_equal(recall.acted, commands[-1:])
    # the address holds zero on the command until the map gives it weight
    brain.stimulus(event(3))
    assert not recall.last.address[0, width:].any()
    np.testing.assert_array_equal(recall.last.inputs[0, width:], commands[-1])
    # a moment wait senses executes nothing
    brain.live(event(3))
    brain.wait(event(4))
    assert not recall.values[0, -1, width:].any() and not recall.acted.any()
    brain.reset()
    assert recall.acted.shape == (0, ACTIONS) or not recall.acted.any()


def test_a_slotted_motor_carries_one_choice_per_slot():
    brain = cd.Brain.compose(INPUTS, 5, slots=(3, 2), seed=1, episodic=False, recall=True, recall_founder=True,
                             recall_action=True)
    chosen = brain.act(event(1), greedy=True)
    command = np.zeros(5)
    command[chosen[0, 0]] = command[3 + chosen[0, 1]] = 1.0
    np.testing.assert_array_equal(brain.recall.values[0, 0, INPUTS:], command)


def test_the_maps_read_the_command_and_a_lesson_teaches_them_to():
    brain = composed(recall_action=True, recall_heads=2, recall_window=2)
    recall = brain.recall
    assert recall.address_width == 2 * INPUTS + ACTIONS
    assert recall.head_map(0)[0].shape == (INPUTS + ACTIONS, recall.address_width)
    assert recall.position_maps[1][0].shape[0] == INPUTS + ACTIONS
    assert {"recall_1_action", "next_1_action"} <= set(brain.connectome.populations)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for t in range(6):
            brain.lesson(event(t % 2), [t % 3])
            brain.act(event(t % 2), greedy=True)
    assert np.abs(recall.head_map(0)[0][INPUTS:]).sum() > 0  # the command rows moved
    action = np.isin(np.asarray(brain.connectome.pre), brain.connectome.populations["recall_action"])
    assert np.abs(brain.brain.efficacy[action]).sum() > 0  # the zero synapses learned


def test_streams_keep_and_drop_their_commands():
    brain = composed(recall_action=True)
    x = np.eye(INPUTS)[[0, 1, 2]]
    chosen = brain.act(x, greedy=True)
    recall = brain.recall
    np.testing.assert_array_equal(recall.acted, np.eye(ACTIONS)[chosen])
    recall.reset(3, rows=np.array([1]))
    assert not recall.acted[1].any() and recall.acted[0].any()
    recall.keep(np.array([0, 2]))
    assert recall.acted.shape == (2, ACTIONS)
    recall.end(2)
    assert not recall.acted.any()  # the moment before begins afresh


def test_imagination_moves_a_private_command_and_writes_no_slot():
    brain = composed(recall_action=True)
    brain.act(event(0), greedy=True)
    before = {name: getattr(brain.recall, name).copy() for name in ("keys", "values", "acted")}
    brain.imagine([event(1), event(2)])
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(brain.recall, name), value)


def test_the_commands_survive_a_checkpoint(tmp_path):
    brain = composed(recall_action=True, recall_heads=2, arousal=True)
    for i in range(4):
        brain.live(event(i), reward=None if i == 0 else [float(i % 2)])
    twin = cd.Brain.load(brain.save(tmp_path / "action.npz"))
    assert twin.recall.action
    for name in ("keys", "values", "nexts", "acted"):
        np.testing.assert_array_equal(getattr(twin.recall, name), getattr(brain.recall, name))
    for h in range(2):
        for kept, saved in zip(twin.recall.head_map(h), brain.recall.head_map(h), strict=True):
            np.testing.assert_array_equal(kept, saved)
    for b in (brain, twin):
        b.live(event(4), reward=[1.0])
    np.testing.assert_array_equal(twin.recall.values, brain.recall.values)
    assert twin.describe() == brain.describe()
