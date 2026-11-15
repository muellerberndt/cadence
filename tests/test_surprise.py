"""``surprise``: one signal read from the settle's own residual, and the focus it draws (#337).

Each neuron's surprise is the size of its potential equation's residual at the start of the
moment's settle (the carried state under the moment's drive), against the size the neuron is
used to. ``Brain.compose(surprise=True)`` holds it as ``brain.surprise``; unset, a brain is the
plain brain to the bit, and set, it reads the settle and changes nothing else.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence.space.map import Map
from cadence.space.physics import Physics
from cadence.surprise import Focus, Orienting, Surprise

PLANE = (-1.0, 1.0, -0.2, 1.4)
# a sensory neuron's first change of one unit of drive, with no usual yet: log(1 / floor), less
# what the settle's tolerance leaves
FIRST = 2.9


def one_hot(index: int, width: int = 6) -> np.ndarray:
    return np.eye(width)[[index]]


def life(brain, moments: int = 24, seed: int = 1) -> list[int]:
    """A short continuing life under ``live``: six binary inputs and three actions, the action
    paid when it is the count of active inputs modulo three. Gives the actions."""
    rng = np.random.default_rng(seed)
    reward, actions = None, []
    for _ in range(moments):
        x = (rng.random((1, 6)) < 0.4).astype(float)
        action = brain.live(x, reward=None if reward is None else np.array([reward]))
        reward = float(action[0] == int(x[0].sum()) % 3)
        actions.append(int(action[0]))
    return actions


def stepped(brain, moments: int = 12, seed: int = 3) -> list[int]:
    """The same world under ``step`` with a teacher at every third moment, on two fresh
    streams."""
    brain.reset()
    rng = np.random.default_rng(seed)
    reward, actions = None, []
    for moment in range(moments):
        x = (rng.random((2, 6)) < 0.4).astype(float)
        teacher = x.sum(axis=1).astype(int) % 3 if moment % 3 == 2 else None
        action = brain.step(x, reward=reward, teacher=teacher)
        reward = (action == x.sum(axis=1).astype(int) % 3).astype(float)
        actions.extend(int(a) for a in action)
    return actions


def same_focus(left: Focus | None, right: Focus | None) -> bool:
    """Whether two readings of the focus agree in every field."""
    if left is None or right is None:
        return left is right
    return (left.sense, left.index, left.salience, left.held, left.turned) == (
        right.sense, right.index, right.salience, right.held, right.turned
    ) and np.array_equal(left.place, right.place)


def saved(brain, folder: Path, name: str = "brain.npz") -> dict[str, np.ndarray]:
    path = brain.save(folder / name)
    with np.load(path) as held:
        return {key: held[key] for key in held.files}


# -- the law


def test_a_size_within_the_usual_is_no_surprise_and_a_larger_one_reads_its_log_ratio():
    held = Surprise(3)
    first = held.observe([3.0, 0.04, 0.0])
    # no usual yet: a size above the floor reads log(size / floor), one within it nothing
    assert first == pytest.approx([np.log(3.0 / 0.05), 0.0, 0.0])
    assert held.usual == pytest.approx([1.5, 0.02, 0.0])  # half of each excess at once
    again = held.observe([3.0, 0.04, 0.0])
    assert again == pytest.approx([0.0, 0.0, 0.0])  # 3 lies within 2 * 1.5 + 0.05
    assert held.observe([9.0, 0.0, 0.0])[0] == pytest.approx(np.log(9.0 / (2 * 2.25 + 0.05)))
    assert held.moments == 3


def test_the_usual_rises_fast_and_recovers_slowly_so_a_rare_size_surprises_again():
    held = Surprise(1, adapt=0.5, recover=0.01)
    held.observe([2.0])
    assert held.usual[0] == pytest.approx(1.0)
    assert held.observe([2.0])[0] == 0.0  # habituated at once
    for _ in range(100):
        held.observe([0.0])
    assert held.usual[0] == pytest.approx(1.5 * 0.99**100)  # 1.5 after the second size of 2
    assert held.observe([2.0])[0] > 0.3  # the same size surprises again after a long rest
    quick = Surprise(1, adapt=1.0, recover=1.0, tolerance=1.0)
    quick.observe([2.0])
    assert quick.usual[0] == 2.0
    quick.observe([0.0])
    assert quick.usual[0] == 0.0


def test_a_stream_that_carried_nothing_reads_no_surprise_and_teaches_nothing():
    held = Surprise(2)
    out = held.observe([[3.0, 0.0], [5.0, 5.0]], carried=np.array([True, False]))
    assert out[0, 0] > 0 and not out[1].any()
    assert held.usual == pytest.approx([1.5, 0.0])  # the second stream taught nothing
    held.observe([[9.0, 9.0], [9.0, 9.0]], carried=np.array([False, False]))
    assert held.usual == pytest.approx([1.5, 0.0]) and not held.last.any()
    quiet = Surprise(2)
    quiet.observe([4.0, 4.0], learn=False)
    assert quiet.last.any() and not quiet.usual.any()


def test_moment_is_one_finite_number_a_stream_of_zero_or_more():
    held = Surprise(4)
    assert held.moment.shape == (1,) and held.moment[0] == 0.0
    held.observe([[0.0, 0.0, 0.0, 0.0], [1e300, 0.0, 2.0, 0.0], [0.0, 0.01, 0.0, 0.0]])
    assert held.moment.shape == (3,)
    assert np.isfinite(held.moment).all() and (held.moment >= 0).all()
    assert held.moment[0] == 0.0 and held.moment[2] == 0.0
    assert held.moment[1] == pytest.approx(np.log(1e300 / 0.05))
    held.reset(2)
    assert held.last.shape == (2, 4) and not held.last.any() and held.usual.any()


def test_a_surprise_refuses_what_it_cannot_read():
    for units in (0, True, 1.5):
        with pytest.raises(ValueError, match="one unit or more"):
            Surprise(units)
    with pytest.raises(ValueError, match="floor above zero"):
        Surprise(2, floor=0.0)
    for gene in ("tolerance", "adapt", "recover"):
        with pytest.raises(ValueError, match=gene):
            Surprise(2, **{gene: -0.1})
        with pytest.raises(ValueError, match=gene):
            Surprise(2, **{gene: True})
    with pytest.raises(ValueError, match="adapt"):
        Surprise(2, adapt=1.5)
    with pytest.raises(ValueError, match="usual"):
        Surprise(2, usual=[1.0, -1.0])
    with pytest.raises(ValueError, match="usual"):
        Surprise(2, usual=[1.0, 2.0, 3.0])
    held = Surprise(2)
    for sizes in ([1.0, 2.0, 3.0], [[1.0, np.nan]], np.zeros((0, 2)), np.zeros((1, 1, 2))):
        with pytest.raises(ValueError, match="sizes"):
            held.observe(sizes)
    with pytest.raises(ValueError, match="carried"):
        held.observe([[1.0, 2.0]], carried=[True, False])
    with pytest.raises(ValueError, match="carried"):
        held.observe([[1.0, 2.0]], carried=np.array([1]))
    assert held.moments == 0 and not held.usual.any()  # a refused moment changed nothing


def test_a_surprise_is_read_back_from_its_record_and_arrays():
    held = Surprise(3, tolerance=1.5, floor=0.1, adapt=0.25, recover=0.02)
    held.observe([[1.0, 0.0, 4.0], [2.0, 0.0, 0.0]])
    back = Surprise.from_dict(json.loads(json.dumps(held.to_dict())), held.arrays())
    assert back.settings() == held.settings() and back.moments == 1
    assert np.array_equal(back.usual, held.usual) and np.array_equal(back.last, held.last)
    assert np.array_equal(back.observe([[5.0, 5.0, 5.0]]), held.observe([[5.0, 5.0, 5.0]]))
    fresh = Surprise.from_dict(held.to_dict(), {"usual": held.usual})
    assert fresh.last.shape == (1, 3) and not fresh.last.any()
    with pytest.raises(ValueError, match="invalid saved surprise"):
        Surprise.from_dict({"tolerance": 2.0}, held.arrays())
    with pytest.raises(ValueError, match="invalid saved surprise"):
        Surprise.from_dict(held.to_dict(), {})
    with pytest.raises(ValueError, match="moments"):
        Surprise.from_dict({**held.to_dict(), "moments": -1}, held.arrays())
    with pytest.raises(ValueError, match="last"):
        Surprise.from_dict(held.to_dict(), {"usual": held.usual, "last": np.zeros((1, 2))})


# -- the focus


def test_the_place_of_the_largest_surprise_wins_and_nothing_below_the_least_orients():
    places = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    eye = Orienting()
    assert eye.attend([0.1, 0.4, 0.2], places) is None  # none reaches the least salience
    focus = eye.attend([0.1, 2.0, 0.7], places)
    assert isinstance(focus, Focus) and focus.index == 1 and focus.turned and focus.held == 1
    assert np.array_equal(focus.place, [1.0, 0.0]) and focus.salience == 2.0
    assert eye.focus is focus and eye.moments == 2


def test_the_focus_holds_a_share_follows_its_place_and_leaves_for_a_larger_surprise():
    places = np.array([[0.0, 0.0], [0.05, 0.0], [1.0, 0.0]])
    eye = Orienting(hold=0.8, reach=0.1)
    eye.attend([3.0, 0.0, 0.0], places)
    held = eye.attend([0.0, 0.0, 2.0], places)  # 2 does not exceed the 2.4 it holds
    assert held.index == 0 and not held.turned and held.held == 2
    assert held.salience == pytest.approx(2.4)
    near = eye.attend([0.0, 2.5, 0.0], places)  # renewed within reach: the focus follows it
    assert near.index == 1 and not near.turned and near.held == 3 and near.salience == 2.5
    left = eye.attend([0.0, 0.0, 2.6], places)
    assert left.index == 2 and left.turned and left.held == 1


def test_a_place_the_focus_leaves_is_inhibited_before_it_can_win_again():
    places = np.array([[0.0, 0.0], [1.0, 0.0]])
    eye = Orienting(inhibit=1.0, release=0.5)
    eye.attend([2.0, 0.0], places)
    eye.attend([0.0, 5.0], places)  # the focus leaves place 0
    assert len(eye.inhibited) == 1
    back = eye.attend([9.0, 0.0], places)  # all of its salience is inhibited at once
    assert back.index == 1 and not back.turned
    for _ in range(8):  # the inhibition halves a moment and is released
        eye.attend([0.0, 0.0], places)
    assert eye.focus is None or eye.focus.salience < 5.0
    assert eye.attend([9.0, 0.0], places).index == 0
    free = Orienting(inhibit=0.0)
    free.attend([2.0, 0.0], places)
    free.attend([0.0, 5.0], places)
    assert free.attend([9.0, 0.0], places).index == 0  # no inhibition of return


def test_every_sense_competes_in_one_competition_with_a_gain_and_a_bearing():
    eye = Orienting(gain=1.0, reach=0.2)
    assert eye.bearing([0.0, 0.0]) is None
    assert np.array_equal(eye.gains(np.zeros((3, 2))), np.ones(3))
    focus = eye.attend({
        "sight": ([0.2, 1.0], [[0.0, 1.0], [0.5, 1.0]]),
        "sound": ([3.0], [[-1.0, 0.0]]),
    })
    assert focus.sense == "sound" and focus.index == 0
    assert eye.bearing([0.0, 0.0]) == pytest.approx(np.pi)
    gains = eye.gains([[-1.0, 0.0], [-0.8, 0.0], [1.0, 1.0]])
    assert gains[0] == pytest.approx(2.0) and 1.0 < gains[1] < 2.0
    assert gains[2] == pytest.approx(1.0)


def test_an_orienting_refuses_what_it_cannot_read_and_is_read_back_from_its_parts():
    eye = Orienting()
    with pytest.raises(ValueError, match="finite"):
        eye.attend([np.inf], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="one place a unit"):
        eye.attend([1.0, 2.0], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="one frame"):
        eye.attend({"a": ([1.0], [[0.0, 0.0]]), "b": ([1.0], [[0.0]])})
    assert eye.moments == 0 and eye.focus is None  # a refused moment changed nothing
    with pytest.raises(ValueError, match="hold"):
        Orienting(hold=1.5)
    with pytest.raises(ValueError, match="least"):
        Orienting(least=-1.0)
    places = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    eye.attend([2.0, 0.0, 0.0], places)
    eye.attend([0.0, 5.0, 0.0], places)
    record, arrays = eye.parts()
    back = Orienting.from_parts(json.loads(json.dumps(record)), arrays)
    assert back.settings() == eye.settings() and back.moments == 2
    assert eye.focus is not None and same_focus(back.focus, eye.focus)
    for moment in ([9.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]):
        assert same_focus(eye.attend(moment, places), back.attend(moment, places))
    with pytest.raises(ValueError, match="invalid saved orienting"):
        Orienting.from_parts({"hold": 0.8}, {})
    with pytest.raises(ValueError, match="invalid saved orienting"):
        Orienting.from_parts(record, {})  # a focus without its place


# -- the residual, a neuron


def test_the_residual_a_neuron_is_what_the_settle_starts_from():
    brain = cd.Brain.compose(6, 3, seed=0, backend="cpu")
    graph = brain.brain
    brain.act(one_hot(0), greedy=True)
    carried, before = brain.basal_ganglia.state, np.array(brain.basal_ganglia._drive)
    brain.act(one_hot(1), greedy=True)
    drive = np.array(brain.basal_ganglia._drive)
    error = graph.residual(drive, carried, per_unit=True)
    assert error.shape == (1, brain.connectome.n)
    # its largest size is the row's residual, the one number a settle's check reads
    assert np.abs(error).max(axis=1) == pytest.approx(
        graph.residual(drive, carried, on_device=False), rel=1e-12)
    # for a state that settled within the tolerance it is the change of the drive, to within it
    tolerance = brain.learner.config.tolerance
    assert np.abs(error - (drive - before)).max() <= tolerance
    # the two inputs that changed, by the amplitude of a sensory drive
    assert np.abs(error[0, brain.sensory_index[:2]]) == pytest.approx([1.0, 1.0], abs=tolerance)
    # under its own drive a settled state leaves no more than the tolerance
    assert np.abs(graph.residual(before, carried, per_unit=True)).max() <= tolerance


# -- the brain's surprise


def test_unset_a_brain_holds_no_surprise_and_is_the_plain_brain_to_the_bit(tmp_path):
    plain = cd.Brain.compose(6, 3, seed=2, arousal=True)
    off = cd.Brain.compose(6, 3, seed=2, arousal=True, surprise=False)
    assert plain.surprise is None and plain.orienting is None and plain.focus is None
    assert plain.places is None and off.surprise is None
    assert life(plain) == life(off)
    assert stepped(plain) == stepped(off)
    left, right = saved(plain, tmp_path, "plain.npz"), saved(off, tmp_path, "off.npz")
    assert sorted(left) == sorted(right)
    for name in left:
        assert np.array_equal(left[name], right[name]), name
    assert not any(name.startswith("surprise/") for name in left)
    assert "surprise" not in json.loads(str(left["brain"]))
    assert "surprise" not in plain.describe()["genes"]


def test_set_it_reads_the_settle_and_changes_nothing_else_to_the_bit(tmp_path):
    """A brain with surprise chooses, learns and saves as the plain brain does: the same
    actions over a life, the same weights, and the same saved arrays beside its own."""
    plain = cd.Brain.compose(6, 3, seed=2, arousal=True)
    held = cd.Brain.compose(6, 3, seed=2, arousal=True, surprise=True)
    assert np.array_equal(plain.brain.efficacy, held.brain.efficacy)
    assert life(plain) == life(held)
    assert stepped(plain) == stepped(held)
    assert held.surprise.moments > 0 and held.surprise.usual.any()
    assert np.array_equal(plain.brain.efficacy, held.brain.efficacy)
    assert np.array_equal(plain.brain.bias, held.brain.bias)
    left, right = saved(plain, tmp_path, "plain.npz"), saved(held, tmp_path, "held.npz")
    own = sorted(name for name in right if name.startswith("surprise/"))
    assert own == ["surprise/last", "surprise/usual"]
    assert sorted(left) == sorted(name for name in right if name not in own)
    for name in left:
        if name != "brain":
            assert np.array_equal(left[name], right[name]), name
    record = json.loads(str(right["brain"]))
    told = record.pop("surprise")
    assert record == json.loads(str(left["brain"]))
    assert told["moments"] == held.surprise.moments and told["orienting"]["focus"] is None


@pytest.mark.parametrize("inputs, actions, options", [
    (6, 3, {"learned_code": 16}),
    ((6, 6), 4, {}),
    (6, 3, {"modules": (16, 16), "apical": 0.25}),
    (6, 3, {"memory_choice": True}),
    (6, 4, {"slots": 2, "observers": (8,)}),
    (6, 3, {"recall": False, "efference_amplitude": 1.0}),
], ids=["learned code", "image", "apical", "episodic vote", "slots and observers", "no recall"])
def test_other_compositions_read_it_and_keep_the_plain_brain_s_bits(inputs, actions, options):
    plain = cd.Brain.compose(inputs, actions, seed=3, backend="cpu", arousal=True, **options)
    held = cd.Brain.compose(inputs, actions, seed=3, backend="cpu", arousal=True,
                            surprise=True, **options)
    width = inputs if isinstance(inputs, int) else int(np.prod(inputs))
    rng = np.random.default_rng(0)
    reward, read = None, []
    for _ in range(10):
        x = (rng.random((1, width)) < 0.3).astype(float)
        a, b = plain.live(x, reward=reward), held.live(x, reward=reward)
        assert np.array_equal(a, b)
        reward = np.array([float(np.ravel(a)[0] == 0)])
        read.append(float(held.surprise.moment[0]))
    assert np.array_equal(plain.brain.efficacy, held.brain.efficacy)
    assert held.surprise.size == held.connectome.n and held.surprise.moments == 10
    assert read[0] == 0.0 and max(read) > 0.0 and np.isfinite(read).all()


def test_a_repeated_input_reads_nothing_and_a_change_reads_at_the_neurons_it_drives():
    brain = cd.Brain.compose(6, 3, seed=0, backend="cpu", surprise=True)
    held = brain.surprise
    brain.act(one_hot(0), greedy=True)
    # a stream that begins from rest carries no equilibrium: no surprise, nothing taught
    assert held.last.shape == (1, brain.connectome.n) and not held.last.any()
    assert not held.usual.any() and held.moments == 1
    brain.act(one_hot(0), greedy=True)
    sensory = brain.sensory_index
    assert not held.last[0, sensory].any()  # the same input again: no sensory residual
    carried, before = brain.basal_ganglia.state, np.array(brain.basal_ganglia._drive)
    usual = held.usual.copy()
    brain.act(one_hot(1), greedy=True)
    drive = np.array(brain.basal_ganglia._drive)
    sizes = np.abs(brain.brain.residual(drive, carried, per_unit=True))
    # the surprise is the law's reading of the residual of the carried state under the drive
    expected = Surprise(brain.connectome.n, usual=usual).observe(sizes[0])
    assert np.array_equal(held.last[0], expected)
    changed = sensory[:2]
    assert (held.last[0, changed] > FIRST).all()  # the input that went and the one that came
    assert not held.last[0, sensory[2:]].any()
    assert held.moment.shape == (1,) and held.moment[0] == held.last.max()
    # the sizes are the change of the drive: the two inputs' amplitude
    assert sizes[0, changed] == pytest.approx(np.abs(drive - before)[0, changed], abs=1e-2)
    for _ in range(3):  # the same change again and again becomes usual
        brain.act(one_hot(0), greedy=True)
        brain.act(one_hot(1), greedy=True)
    assert not held.last[0, sensory].any()
    brain.act(one_hot(4), greedy=True)  # an input the brain has not had: it surprises
    assert held.last[0, sensory[4]] > FIRST


def test_every_stream_reads_its_own_surprise():
    brain = cd.Brain.compose(6, 3, seed=1, backend="cpu", surprise=True)
    x = np.concatenate([one_hot(0), one_hot(1), one_hot(2)])
    brain.act(x, greedy=True)
    assert brain.surprise.last.shape == (3, brain.connectome.n) and not brain.surprise.last.any()
    again = x.copy()
    again[1] = one_hot(5)[0]
    brain.act(again, greedy=True)
    moment = brain.surprise.moment
    assert moment.shape == (3,) and np.isfinite(moment).all() and (moment >= 0).all()
    sensory = brain.sensory_index
    assert brain.surprise.last[1, sensory].any()
    assert not brain.surprise.last[0, sensory].any() and not brain.surprise.last[2, sensory].any()
    assert moment[1] > FIRST


def test_a_step_reads_the_moment_at_its_first_settle_and_an_ended_stream_reads_none():
    brain = cd.Brain.compose(6, 3, seed=4, backend="cpu", surprise=True)
    sensory = brain.sensory_index
    x = np.concatenate([one_hot(0), one_hot(0)])
    brain.step(x)
    brain.step(x, reward=np.array([0.0, 1.0]))
    assert not brain.surprise.last[:, sensory].any()
    y = np.concatenate([one_hot(1), one_hot(2)])
    brain.step(y, reward=np.array([1.0, 0.0]), teacher=np.array([1, 2]))
    # read where the outcome's settle starts from the carried state, before the lesson and
    # the act that refreshes it
    assert (brain.surprise.last[0, sensory[:2]] > FIRST).all()
    assert (brain.surprise.last[1, sensory[[0, 2]]] > FIRST).all()
    usual = brain.surprise.usual.copy()
    z = np.concatenate([one_hot(3), one_hot(4)])
    brain.step(z, reward=np.array([0.0, 0.0]), done=np.array([True, False]))
    assert not brain.surprise.last[0].any()  # its episode ended: it begins from rest
    assert brain.surprise.last[1, sensory[4]] > FIRST
    # the usual learned from the stream that carried a state alone
    assert brain.surprise.usual[sensory[3]] == usual[sensory[3]] == 0.0
    assert brain.surprise.usual[sensory[4]] > 0.0


def test_live_and_wait_read_the_moment_in_routine_and_in_repair():
    brain = cd.Brain.compose(6, 3, seed=5, arousal=True, surprise=True)
    sensory = brain.sensory_index
    brain.live(one_hot(0))
    assert not brain.surprise.last.any()
    modes = set()
    for moment in range(1, 9):  # each input comes for the first time: it surprises
        brain.live(one_hot(moment % 6), reward=np.array([float(moment % 2)]))
        modes.add(brain.last_arousal["mode"])
        if moment < 6:
            assert brain.surprise.last[0, sensory[moment % 6]] > FIRST, moment
    assert brain.surprise.moments == 9
    brain.wait(one_hot(3))
    assert brain.surprise.moments == 10 and brain.surprise.last.shape == (1, brain.connectome.n)
    fresh = cd.Brain.compose(6, 3, seed=5, arousal=True, surprise=True)
    fresh.live(one_hot(0))
    fresh.wait(one_hot(1))  # a sensed moment is a moment of the stream
    assert (fresh.surprise.last[0, sensory[:2]] > FIRST).all()
    fresh.live(one_hot(1), reward=np.array([1.0]))
    assert not fresh.surprise.last[0, sensory].any()


def test_an_ended_episode_under_live_begins_from_rest():
    brain = cd.Brain.compose(6, 3, seed=6, arousal=True, surprise=True)
    brain.live(one_hot(0))
    brain.live(one_hot(1), reward=np.array([0.0]))
    assert brain.surprise.last.any()
    usual, moments = brain.surprise.usual.copy(), brain.surprise.moments
    brain.live(one_hot(2), reward=np.array([0.0]), done=np.array([True]))
    assert not brain.surprise.last.any() and brain.surprise.moments == moments + 1
    assert np.array_equal(brain.surprise.usual, usual)


def test_imagination_lessons_and_reads_leave_the_surprise_as_it_was():
    brain = cd.Brain.compose(6, 3, seed=7, backend="cpu", surprise=True,
                             learning_rehearsal=1.0)
    brain.places = np.stack([np.arange(6.0), np.zeros(6)], axis=1)
    for index in (0, 1, 0, 2):
        brain.act(one_hot(index), greedy=True)
    held = brain.surprise

    def state():
        focus = brain.focus
        return (held.usual.copy(), held.last.copy(), held.moments, brain.orienting.moments,
                None if focus is None else (focus.index, focus.salience, focus.held))

    before = state()
    assert before[4] is not None

    def same():
        after = state()
        return (np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
                and before[2:] == after[2:])

    phases = brain.imagine([one_hot(4), one_hot(5), one_hot(3)])
    assert len(phases) == 3 and same()
    brain.predict(one_hot(5))
    brain.value(one_hot(5))
    assert same()
    for label in (1, 2, 0):  # lessons, and the replays of earlier lessons on fresh streams
        brain.lesson(one_hot(4), [label])
    assert brain.rehearsal_draws > 0 and same()
    twin = copy.deepcopy(brain)
    twin.act(one_hot(5), greedy=True)
    assert same() and twin.surprise.moments == held.moments + 1


def test_a_refused_settle_leaves_the_surprise_as_it_was():
    brain = cd.Brain.compose(6, 3, seed=8, backend="cpu", surprise=True)
    brain.act(one_hot(0), greedy=True)
    brain.act(one_hot(1), greedy=True)
    usual, last, moments = brain.surprise.usual.copy(), brain.surprise.last.copy(), 2
    brain.retune(learning_free_steps=0)
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.act(one_hot(2), greedy=True)
    assert brain.surprise.moments == moments and brain._surprise_read is None
    assert np.array_equal(brain.surprise.usual, usual)
    assert np.array_equal(brain.surprise.last, last)
    brain.retune(learning_free_steps=1024)
    brain.act(one_hot(2), greedy=True)
    assert brain.surprise.last[0, brain.sensory_index[2]] > FIRST


def test_reset_begins_the_streams_afresh_and_keeps_what_each_neuron_is_used_to():
    brain = cd.Brain.compose(6, 3, seed=9, backend="cpu", surprise=True)
    brain.places = np.stack([np.arange(6.0), np.zeros(6)], axis=1)
    for index in (0, 1, 2):
        brain.act(np.concatenate([one_hot(index), one_hot(index)]), greedy=True)
    usual = brain.surprise.usual.copy()
    assert brain.focus is not None and brain.surprise.last.shape[0] == 2
    brain.reset()
    assert brain.focus is None and brain.orienting.moments == 0
    assert brain.surprise.last.shape == (1, brain.connectome.n) and not brain.surprise.last.any()
    assert np.array_equal(brain.surprise.usual, usual) and brain.places is not None
    brain.act(one_hot(5), greedy=True)  # the first moment of a new stream reads none
    assert not brain.surprise.last.any() and np.array_equal(brain.surprise.usual, usual)


def test_a_saved_brain_continues_with_the_same_surprise_and_focus(tmp_path):
    brain = cd.Brain.compose(6, 3, seed=10, arousal=True, surprise={"least": 0.2})
    brain.places = np.stack([np.cos(np.arange(6.0)), np.sin(np.arange(6.0))], axis=1)
    rng = np.random.default_rng(0)
    reward = None
    for _ in range(10):
        x = (rng.random((1, 6)) < 0.4).astype(float)
        action = brain.live(x, reward=reward)
        reward = np.array([float(action[0] == 1)])
    assert brain.focus is not None and brain.orienting.inhibited
    back = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert back.describe()["genes"]["surprise"] == brain.describe()["genes"]["surprise"]
    assert np.array_equal(back.surprise.usual, brain.surprise.usual)
    assert np.array_equal(back.surprise.last, brain.surprise.last)
    assert np.array_equal(back.surprise.moment, brain.surprise.moment)
    assert back.surprise.moments == brain.surprise.moments
    assert np.array_equal(back.places, brain.places)
    assert same_focus(back.focus, brain.focus)
    assert len(back.orienting.inhibited) == len(brain.orienting.inhibited)
    for _ in range(8):  # the same next moments, the settled states within rounding
        x = (rng.random((1, 6)) < 0.4).astype(float)
        a, b = brain.live(x, reward=reward), back.live(x, reward=reward)
        assert np.array_equal(a, b)
        np.testing.assert_allclose(back.surprise.last, brain.surprise.last, rtol=0, atol=1e-9)
        np.testing.assert_allclose(back.surprise.usual, brain.surprise.usual, rtol=0, atol=1e-9)
        left, right = brain.focus, back.focus
        assert (left is None) == (right is None)
        if left is not None and right is not None:
            assert (left.sense, left.index, left.held, left.turned) == (
                right.sense, right.index, right.held, right.turned)
            assert left.salience == pytest.approx(right.salience, abs=1e-9)
        reward = np.array([float(a[0] == 1)])


def test_a_brain_saved_between_an_outcome_and_its_act_keeps_the_moment_s_residual(tmp_path):
    brain = cd.Brain.compose(6, 3, seed=11, backend="cpu", surprise=True)
    brain.act(one_hot(0))
    brain.learn(np.array([1.0]), np.array([False]), one_hot(3))
    # the outcome is learned and the stream is at the new moment: its surprise is observed
    assert brain.surprise.moments == 2 and brain._surprise_read is None
    assert brain.surprise.last[0, brain.sensory_index[3]] > FIRST
    last, usual = brain.surprise.last.copy(), brain.surprise.usual.copy()
    back = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert np.array_equal(back._surprise_seen, one_hot(3))
    a, b = brain.act(one_hot(3)), back.act(one_hot(3))
    assert np.array_equal(a, b)
    for held in (brain, back):  # the act that follows observes nothing a second time
        assert held.surprise.moments == 2 and held._surprise_seen is None
        assert np.array_equal(held.surprise.last, last)
        assert np.array_equal(held.surprise.usual, usual)
    brain.act(one_hot(3))  # the next moment, the same observations: read anew
    assert brain.surprise.moments == 3 and not brain.surprise.last[0, brain.sensory_index].any()


def test_the_moment_s_surprise_is_observed_before_its_lesson_and_its_memories_write():
    brain = cd.Brain.compose(6, 3, seed=16, arousal=True, surprise=True)
    seen: list[tuple[int, np.ndarray]] = []
    witness, lesson = brain.recall.witness, brain._teach

    def witnessed(*args, **kwargs):
        seen.append((brain.surprise.moments, brain.surprise.last.copy()))
        return witness(*args, **kwargs)

    def taught(*args, **kwargs):
        seen.append((brain.surprise.moments, brain.surprise.last.copy()))
        return lesson(*args, **kwargs)

    brain.recall.witness, brain._teach = witnessed, taught
    brain.act(one_hot(0), greedy=True)
    brain.act(one_hot(1), greedy=True)
    assert [moments for moments, _ in seen] == [1, 2]
    assert np.array_equal(seen[1][1], brain.surprise.last) and seen[1][1].any()
    del seen[:]
    brain.step(one_hot(2))
    brain.step(one_hot(3), reward=np.array([1.0]), teacher=np.array([1]))
    # the lesson and the memory's write of the fourth moment read that moment's surprise
    assert [moments for moments, _ in seen] == [3, 4, 4]
    assert all(np.array_equal(last, brain.surprise.last) for _, last in seen[1:])
    assert brain.surprise.last[0, brain.sensory_index[3]] > FIRST
    del seen[:]
    brain.reset()
    brain.live(one_hot(0))
    lived = brain.surprise.moments  # the count of every moment observed, across a reset
    for moment in range(1, 5):
        brain.live(one_hot(moment), reward=np.array([float(moment % 2)]))
        assert seen[-1][0] == brain.surprise.moments == lived + moment
        assert np.array_equal(seen[-1][1], brain.surprise.last)


def test_a_saved_surprise_that_does_not_fit_is_refused(tmp_path):
    brain = cd.Brain.compose(6, 3, seed=12, backend="cpu", surprise=True)
    brain.act(one_hot(0), greedy=True)
    held = saved(brain, tmp_path)

    def refused(change, match):
        data = dict(held)
        change(data)
        path = tmp_path / "changed.npz"
        np.savez(path, **data)
        with pytest.raises(ValueError, match=match):
            cd.Brain.load(path)

    def fewer(data):
        data["surprise/usual"] = data["surprise/usual"][:-1]
        del data["surprise/last"]

    refused(fewer, "a neuron")
    refused(lambda d: d.update({"surprise/usual": d["surprise/usual"][:-1]}), "last does not fit")
    refused(lambda d: d.pop("surprise/usual"), "usual residuals")
    refused(lambda d: d.update({"surprise/usual": -d["surprise/usual"] - 1.0}), "zero or more")
    refused(lambda d: d.update({"surprise/last": np.zeros((2, brain.connectome.n))}),
            "one row a stream")

    def without_record(data):
        record = json.loads(str(data["brain"]))
        del record["surprise"]
        data["brain"] = np.array(json.dumps(record))

    refused(without_record, "surprise's record")


def test_compose_names_the_genes_and_refuses_what_it_does_not_know():
    brain = cd.Brain.compose(6, 3, seed=0, surprise={"tolerance": 3.0, "hold": 0.5})
    genes = brain.describe()["genes"]["surprise"]
    assert genes == {"tolerance": 3.0, "floor": 0.05, "adapt": 0.5, "recover": 0.01,
                     "hold": 0.5, "least": 0.5, "reach": 0.1, "inhibit": 1.0, "release": 0.95,
                     "gain": 1.0}
    assert brain.surprise.tolerance == 3.0 and brain.orienting.hold == 0.5
    assert brain.surprise.size == brain.connectome.n
    again = cd.Brain.compose(6, 3, seed=0, surprise=genes)
    assert again.describe()["genes"]["surprise"] == genes
    with pytest.raises(ValueError, match="unknown surprise gene: speed"):
        cd.Brain.compose(6, 3, surprise={"speed": 1.0})
    with pytest.raises(ValueError, match="surprise is True or a mapping"):
        cd.Brain.compose(6, 3, surprise=3)
    with pytest.raises(ValueError, match="floor above zero"):
        cd.Brain.compose(6, 3, surprise={"floor": 0.0})
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(surprise=True)


def test_the_paths_that_do_not_read_the_residual_refuse_surprise_by_name(tmp_path):
    with pytest.raises(ValueError, match="no surprise"):
        cd.Lives(6, 3, lives=2, surprise=True)
    # with no backend named, a device the rule offers is refused by name and the CPU takes it
    unnamed = cd.Brain.compose(6, 3, seed=0, surprise=True)
    offered = unnamed._backends_at

    def with_a_device(rows):
        cpu = [found for found in offered(rows) if found.backend == "cpu"][0]
        return [cpu._replace(backend="torch", why="offered by the test"), cpu]

    unnamed._backends_at, unnamed._unplaced = with_a_device, True
    unnamed.act(one_hot(0), greedy=True)
    unnamed.act(one_hot(1), greedy=True)
    placed = unnamed.placement
    assert placed["backend"] == "cpu" and not placed["pending"]
    assert len(placed["refused"]) == 1 and "backend='cpu'" in placed["refused"][0]
    assert unnamed.surprise.moment[0] > FIRST
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="backend='cpu'"):
        cd.Brain.compose(6, 3, backend="torch", device="cpu", surprise=True)
    brain = cd.Brain.compose(6, 3, seed=0, backend="cpu", surprise=True)
    path = brain.save(tmp_path / "brain.npz")
    with pytest.raises(ValueError, match="backend='cpu'"):
        cd.Brain.load(path, backend="torch", device="cpu")


def test_the_compiled_moment_reads_the_same_surprise_and_acts_as_the_plain_brain():
    stream = [one_hot(index) for index in (0, 1, 1, 3, 0, 5, 2, 2, 4)]
    plain = cd.Brain.compose(6, 3, seed=3, backend="cpu", learning_compiled_moment=True)
    fast = cd.Brain.compose(6, 3, seed=3, backend="cpu", learning_compiled_moment=True,
                            surprise=True)
    slow = cd.Brain.compose(6, 3, seed=3, backend="cpu", surprise=True)
    for x in stream:
        a, b, c = (brain.act(x, greedy=True) for brain in (plain, fast, slow))
        assert np.array_equal(a, b)
        assert np.array_equal(
            plain.basal_ganglia.state.activation, fast.basal_ganglia.state.activation)
        np.testing.assert_allclose(fast.surprise.last, slow.surprise.last, rtol=0, atol=1e-6)
    assert fast.surprise.moments == len(stream) and fast.surprise.usual.any()


def test_the_focus_is_drawn_to_the_place_of_the_neuron_that_surprised():
    brain = cd.Brain.compose(6, 3, seed=13, backend="cpu", surprise=True)
    assert brain.focus is None
    brain.act(one_hot(0), greedy=True)
    brain.act(one_hot(1), greedy=True)
    assert brain.focus is None and brain.orienting.moments == 0  # no places, no competition
    places = np.stack([np.arange(6.0), np.zeros(6)], axis=1)
    brain.places = places
    assert np.array_equal(brain.places, places)
    brain.act(one_hot(4), greedy=True)
    focus = brain.focus
    assert focus is not None and focus.sense == "senses" and focus.turned
    assert focus.index in (1, 4) and focus.salience > FIRST  # the input that went or that came
    assert brain.orienting.bearing([focus.place[0] - 1.0, 0.0]) == pytest.approx(0.0)
    for _ in range(4):
        brain.act(one_hot(4), greedy=True)  # nothing changes: the focus fades and lets go
    assert brain.focus is None or brain.focus.salience < focus.salience
    brain.places = None
    assert brain.places is None
    for value in (np.zeros((5, 2)), np.zeros(6), np.full((6, 2), np.nan), np.zeros((6, 0)), "x"):
        with pytest.raises(ValueError, match="one finite place a sensory neuron"):
            brain.places = value
    plain = cd.Brain.compose(6, 3, seed=13)
    with pytest.raises(ValueError, match="surprise=True"):
        plain.places = places


def test_a_held_map_s_cells_join_the_competition(tmp_path):
    brain = cd.Brain.compose(6, 3, seed=14, backend="cpu", surprise=True)
    brain.space = Map(PLANE, (8, 8), units=4, surprise=True)
    brain.act(one_hot(0), greedy=True)
    assert brain.focus is None  # the map has had no moment
    brain.space.moment(seen=[(0, [0.5, 0.9])])
    brain.act(one_hot(0), greedy=True)
    focus = brain.focus
    assert focus is not None and focus.sense == "map"
    assert np.linalg.norm(focus.place - [0.5, 0.9]) < 0.3
    assert np.array_equal(focus.place, brain.space.centers[focus.index])
    back = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert np.array_equal(back.space.surprise.usual, brain.space.surprise.usual)
    assert np.array_equal(back.space.surprise.last, brain.space.surprise.last)
    assert same_focus(back.focus, focus)
    for held in (brain, back):
        held.space.moment(seen=[(1, [-0.5, 0.2])])
        held.act(one_hot(1), greedy=True)
    assert same_focus(back.focus, brain.focus)
    assert np.array_equal(back.surprise.last, brain.surprise.last)


def test_a_grown_brain_keeps_what_each_neuron_is_used_to():
    brain = cd.Brain.compose(6, 3, seed=15, backend="cpu", surprise={"tolerance": 2.5})
    brain.places = np.stack([np.arange(6.0), np.zeros(6)], axis=1)
    for index in (0, 1, 2, 0):
        brain.act(one_hot(index), greedy=True)
    usual = brain.surprise.usual.copy()
    assert usual.any()
    grown = brain.grow(efference_amplitude=1.0)
    assert grown.connectome.n > brain.connectome.n
    assert grown.surprise.size == grown.connectome.n and grown.surprise.tolerance == 2.5
    for name in ("sensory", "association", "motor"):
        old = np.asarray(brain.connectome.populations[name])
        new = np.asarray(grown.connectome.populations[name])
        assert np.array_equal(grown.surprise.usual[new], usual[old]), name
    assert grown.surprise.usual.sum() == pytest.approx(usual.sum())  # a new neuron has none
    assert np.array_equal(grown.places, brain.places)
    assert grown.focus is None and not grown.surprise.last.any()  # its streams start afresh
    assert np.array_equal(brain.surprise.usual, usual) and brain.surprise.last.any()
    grown.act(one_hot(3), greedy=True)
    grown.act(one_hot(4), greedy=True)
    assert grown.surprise.last[0, grown.sensory_index[4]] > FIRST


def test_a_brain_around_a_trained_module_and_a_grafted_brain_hold_surprise(tmp_path):
    from cadence.space import Interface, Space
    from cadence.space.place import INPUTS, OUTPUTS

    small = dict(place=8, moved=5, motion=9, counted=3, unseen=4, heard=(6, 4), felt=4,
                 hidden=16, shift=12, spread=6, found=8, keep=4)
    space = Space(Interface(plane=PLANE, **small), seed=0, backend="cpu")
    path = space.brain.save_module(
        tmp_path / "place.npz", name="place", inputs=INPUTS, published=("place", *OUTPUTS),
        regions=(*INPUTS, "hidden", *OUTPUTS), interface=space.interface.to_dict())
    module = cd.load_module(path)
    brain = cd.Brain.compose(4, 2, seed=0, backend="cpu", trained=[module],
                             surprise={"tolerance": 2.5})
    assert brain.surprise.size == brain.connectome.n and brain.surprise.tolerance == 2.5
    assert brain.describe()["genes"]["surprise"]["tolerance"] == 2.5
    x = np.zeros((1, 4 + module.width))
    x[0, 0] = 1.0
    brain.act(x, greedy=True)
    x[0, :2] = [0.0, 1.0]
    brain.act(x, greedy=True)
    assert (brain.surprise.last[0, brain.sensory_index[:2]] > FIRST).all()
    # a graft keeps the host's neurons, and each its usual residual; the streams start afresh
    host = cd.Brain.compose(6, 3, seed=1, backend="cpu", recall=False, surprise=True)
    donor = cd.Brain.compose(6, 3, seed=1, backend="cpu", recall=False)
    donor.lesson(one_hot(2), [1])
    for index in (0, 1, 2):
        host.act(one_hot(index), greedy=True)
    usual = host.surprise.usual.copy()
    grafted = host.graft(donor, ["association", "motor"])
    assert usual.any() and np.array_equal(grafted.surprise.usual, usual)
    assert grafted.focus is None and not grafted.surprise.last.any()
    assert np.array_equal(host.surprise.usual, usual) and host.surprise.last.any()


# -- the map's term


def test_a_map_reads_each_thing_s_move_from_where_it_carried_it():
    held = Map(PLANE, (16, 16), units=4, surprise=True)
    plain = Map(PLANE, (16, 16), units=4)
    assert plain.surprise is None and held.surprise.size == held.count

    def moment(**reports):
        held.moment(**reports)
        plain.moment(**reports)
        assert np.array_equal(held.potential, plain.potential)  # the map itself is the same
        return held.surprise.last[0]

    first = moment(seen=[(0, [0.0, 0.6])])
    here = held.weights([0.0, 0.6]) > 0
    assert (first[here] == pytest.approx(np.log(1.0 / 0.05))) and not first[~here].any()
    assert not moment(seen=[(0, [0.0, 0.6])]).any()  # at rest where the map carried it
    assert not moment().any()  # out of sight, held: nothing moved
    jump = moment(seen=[(0, [0.3, 0.6])])  # two steps from where the map held it
    # on the cells it left, against their usual move of half a step; on the cells it came to
    assert jump[here] == pytest.approx(np.log(2.0 / (2 * 0.5 + 0.05)), abs=0.05)
    assert jump[held.weights([0.3, 0.6]) > 0] == pytest.approx(np.log(2.0 / 0.05), abs=0.05)
    other = moment(seen=[(0, [0.3, 0.6]), (1, [-0.6, 0.2])])  # a new thing appears
    there = held.weights([-0.6, 0.2]) > 0
    assert (other[there] > 2.9).all() and not other[held.weights([0.3, 0.6]) > 0].any()
    eye = Orienting()
    assert np.linalg.norm(eye.attend(other, held.centers).place - [-0.6, 0.2]) < 0.2
    held.reset()
    assert not held.surprise.last.any() and held.surprise.usual.any()
    for value in (3, "yes", [1.0]):
        with pytest.raises(ValueError, match="surprise is True or the genes"):
            Map(PLANE, surprise=value)
    with pytest.raises(ValueError, match="floor above zero"):
        Map(PLANE, surprise={"floor": 0.0})


def test_a_map_file_keeps_what_each_cell_is_used_to(tmp_path):
    held = Map(PLANE, (10, 10), units=4, surprise={"tolerance": 1.5})
    for place in ([0.0, 0.6], [0.4, 0.6], [0.4, 0.2]):
        held.moment(seen=[(2, place)])
    back = Map.load(held.save(tmp_path / "map.npz"))
    assert back.surprise.settings() == held.surprise.settings()
    assert back.surprise.moments == held.surprise.moments == 3
    assert np.array_equal(back.surprise.usual, held.surprise.usual)
    assert np.array_equal(back.surprise.last, held.surprise.last)
    for a in (held, back):
        a.moment(seen=[(2, [-0.4, 0.2])])
    assert np.array_equal(back.surprise.last, held.surprise.last)
    without = Map.load(tmp_path / "map.npz", surprise=None)
    assert without.surprise is None and np.array_equal(without.potential, back.potential) is False
    plain = Map(PLANE, (10, 10), units=4)
    assert "surprise" not in plain.parts()[0]
    assert not any(name.startswith("surprise") for name in plain.parts()[1])


def test_with_learned_physics_the_carried_place_is_the_physics_forecast():
    """A thing that goes on along its line is where the physics carries it. Going out of view
    it moves nothing, and the report that finds it again moves it less than the same report
    moves a thing that a map without physics held near where it was last seen."""
    physics = Physics()
    exposure = 1e6  # as if it had watched a world of these shares (its statistics set so)
    physics.stats[:] = [0.005 * exposure, 0.005 * exposure, exposure, 0.5, 1.0, 0.25, 1.0, 1e5,
                        1e5 * 0.015**2, 1e5]
    physics.fit()

    def left_half(points):
        return np.asarray(points)[:, 0] < 0.0

    def everywhere(points):
        return np.ones(len(points), dtype=bool)

    moments = 70
    truth = np.array([-0.8, 0.6]) + np.arange(moments + 1)[:, None] * np.array([0.015, 0.0])
    reports = truth + np.random.default_rng(0).normal(0.0, 0.01, truth.shape)
    leaves = int(np.argmax(truth[:, 0] >= 0.0))  # the first moment out of view
    found, hidden = {}, {}
    for name, held in (("physics", Map(PLANE, physics=physics, surprise=True)),
                       ("plain", Map(PLANE, surprise=True))):
        moved = []
        for t in range(moments):
            held.moment(seen=[(0, reports[t])] if t < leaves else [], lit=left_half)
            moved.append(float(held.surprise.last.max()))
        hidden[name] = moved[leaves:]
        expected = held._expected()
        held.moment(seen=[(0, truth[moments])], lit=everywhere)
        found[name] = float(held.moves(expected).max())
    assert physics.taught
    assert not np.any(hidden["physics"])  # carried along its line: nothing surprises
    assert hidden["plain"][0] > 2.0  # seen empty where it was held: it is moved aside
    assert found["physics"] < 0.5 < found["plain"]
    # a forecast is no experience: it surprises and teaches no cell
    held = Map(PLANE, physics=physics, surprise=True)
    for t in range(20):
        held.moment(seen=[(0, reports[t])], lit=everywhere)
    usual, last = held.surprise.usual.copy(), held.surprise.last.copy()
    ahead = held.forecast(5)
    assert ahead.surprise is None and held.surprise.moments == 20
    assert np.array_equal(held.surprise.usual, usual)
    assert np.array_equal(held.surprise.last, last)
