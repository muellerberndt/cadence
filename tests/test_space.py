"""cadence.space: the regions of a place brain and the space that holds places."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from cadence.space import Interface, Space, Tuned, anatomy, compose
from cadence.space.place import AREA, INPUTS, OUTPUTS

PLANE = (-1.0, 1.0, -0.2, 1.4)
SMALL = dict(place=8, moved=5, motion=9, counted=3, unseen=4, heard=(6, 4), felt=4, hidden=16,
             shift=12, spread=6, found=8, keep=4)


def small(**settings) -> Space:
    return Space(Interface(plane=PLANE, **SMALL), seed=0, backend="cpu", **settings)


def test_importing_cadence_loads_no_space():
    code = "import sys, cadence; print(any(m.startswith('cadence.space') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_a_tuned_slot_teaches_a_value_and_reads_it_back():
    slot = Tuned(-0.06, 0.06, 32)
    values = np.array([-0.06, -0.0123, 0.0, 0.0301, 0.06, 0.2])
    targets = slot.targets(values)
    assert targets.shape == (6, 32) and np.allclose(targets.sum(axis=1), 1.0)
    assert (targets >= 0).all() and ((targets > 0).sum(axis=1) <= 2).all()
    assert np.allclose(targets @ slot.centers, np.clip(values, -0.06, 0.06))
    # a settled slot whose softmax is the target reads the value
    activity = 0.05 * np.log(np.maximum(targets, 1e-12))
    assert np.allclose(slot.mean(activity, 0.05), np.clip(values, -0.06, 0.06), atol=1e-6)
    wide = Tuned(0.002, 2.0, 16, log=True)
    flat = np.zeros((1, 16))
    low, middle, high = (wide.quantile(flat, 0.05, rate)[0] for rate in (0.1, 0.5, 0.9))
    assert 0.002 < low < middle < high < 2.0
    assert np.isclose(wide.mean(0.05 * np.log(np.maximum(wide.targets(np.array([0.05])), 1e-12)),
                                0.05)[0], 0.05, rtol=1e-6)
    with pytest.raises(ValueError, match="2 or more units"):
        Tuned(0.0, 1.0, 1)
    # two values believed: the mean reads between them, the peak reads the likelier one
    two = 0.05 * np.log(np.maximum(0.6 * slot.targets(np.array([0.02]))
                                   + 0.4 * slot.targets(np.array([-0.03])), 1e-12))
    assert np.isclose(slot.mean(two, 0.05)[0], 0.0, atol=1e-6)
    assert np.isclose(slot.peak(two, 0.05)[0], 0.02, atol=1e-3)
    broad = slot.targets(np.array([0.0113]), width=1.5)
    assert np.isclose(broad.sum(), 1.0) and (broad > 0.01).sum() >= 5
    assert np.isclose((broad @ slot.centers)[0], 0.0113, atol=1e-4)


def test_an_interface_names_its_regions_and_reads_back():
    interface = Interface(plane=PLANE, joints=3)
    regions = interface.regions
    assert list(regions) == [*INPUTS, "hidden", *OUTPUTS]
    assert regions == {"place": 32, "moved": 27, "motion": 56, "unseen": 8, "heard": 320,
                       "felt": 512, "hidden": 128, "shift": 64, "spread": 16, "found": 48,
                       "keep": 8}
    assert interface.inputs == 955 and interface.outputs == 136
    assert Interface(plane=PLANE).regions["felt"] == 256  # two joints: 16 readings each
    assert interface.limits == ((-np.pi, np.pi),) * 3
    assert interface.slots == (32, 32, 16, 24, 24, 8)
    assert Interface.from_dict(interface.to_dict()) == interface
    with pytest.raises(ValueError, match="low below high"):
        Interface(plane=(1.0, -1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="under memory"):
        Interface(plane=PLANE, check=80)
    with pytest.raises(ValueError, match="4,096 postures"):
        Interface(plane=PLANE, joints=4, felt=9)
    with pytest.raises(ValueError, match="one \\(low, high\\) range a joint"):
        Interface(plane=PLANE, limits=((0.0, 1.0),))
    ranged = Interface(plane=PLANE, limits=((0.3, 2.8), (-2.6, -0.3)))
    assert Interface.from_dict(ranged.to_dict()) == ranged


def test_the_anatomy_holds_the_interfaces_regions_by_name():
    interface = Interface(plane=PLANE, **SMALL)
    grown = anatomy(interface, seed=0)
    held = grown.populations
    assert {name: len(held[name]) for name in interface.regions} == interface.regions
    assert len(held["sensory"]) == interface.inputs and len(held["motor"]) == interface.outputs
    assert list(held["association"]) == list(held["hidden"])
    brain = compose(interface, seed=0, backend="cpu")
    genes = brain.describe()["genes"]
    assert genes["teaching"] == "founder" and genes["temperature"] == 0.05
    # every synapse runs from the inputs toward the outputs
    inputs, outputs = set(held["sensory"]), set(held["motor"])
    assert not inputs & set(grown.post.tolist()) and not outputs & set(grown.pre.tolist())
    assert list(brain.learner.slot_sizes) == list(interface.slots)
    assert brain.describe().get("memories", {}) == {}


def test_a_place_with_no_information_stays_and_the_brain_is_not_asked():
    space = small()
    space.begin(np.array([[0.3, 0.4], [-0.5, 1.0]]), own=[True, False])
    nothing = np.zeros(2, dtype=bool)
    for _ in range(1000):
        space.advance(nothing, np.zeros((2, 2)), np.zeros(2))
    place, radius = space.where()
    assert np.array_equal(place, [[0.3, 0.4], [-0.5, 1.0]])
    assert space.work["moments"] == 0 and space.since.tolist() == [1000, 1000]
    assert np.allclose(radius, AREA * np.sqrt(space.noise))


def test_a_report_writes_the_place_and_a_thing_at_rest_takes_the_mean_of_its_reports():
    space = small()
    space.begin(np.array([[0.0, 0.5]]))
    rng = np.random.default_rng(0)
    for _ in range(60):
        space.advance([True], np.array([[0.0, 0.5]]) + rng.normal(0.0, 0.01, (1, 2)))
    place, radius = space.where()
    assert np.linalg.norm(place[0] - [0.0, 0.5]) < 0.006  # under one report's error
    assert radius[0] < 0.5 * AREA * np.sqrt(space.noise)
    space.advance([True], np.array([[0.4, 0.9]]))  # seen elsewhere: the report itself
    assert np.array_equal(space.where()[0], [[0.4, 0.9]])
    assert np.isclose(space.where()[1][0], AREA * np.sqrt(space.noise))


def test_a_command_moves_the_arms_own_end_while_unseen():
    space = small()
    space.begin(np.array([[0.3, 0.4], [0.3, 0.4]]), own=[True, False])
    space.advance(np.zeros(2, dtype=bool), np.zeros((2, 2)), np.array([0.05, -0.02]))
    state = space.state()
    assert state["live"].tolist() == [True, False] and space.work["moments"] == 1
    assert state["shift"].shape == (1, 2, 12) and np.allclose(state["shift"].sum(axis=-1), 1.0)
    assert state["moved"][0].any() and not state["moved"][1].any()
    assert np.array_equal(space.where()[0][1], [0.3, 0.4])  # the other thing kept its place
    assert space.since.tolist() == [1, 1]


def test_the_state_holds_every_regions_activity_as_plain_arrays():
    space = small()
    space.begin(np.array([[0.3, 0.4]]), own=True)
    assert set(space.state()) == {"regions", "at", "radius", "since", "noise"}
    space.advance([False], np.zeros((1, 2)), np.array([0.05, 0.0]),
                  cues=np.array([[-2.0, 1.0]]))
    state = space.state()
    interface = space.interface
    assert state["regions"] == interface.regions
    for name in INPUTS:
        assert state[name].shape == (1, interface.regions[name])
    assert state["at"].shape == (1, 2) and state["radius"].shape == (1,)
    assert state["hidden"].shape == (1, 16) and state["spread"].shape == (1, 6)
    assert state["found"].shape == (1, 2, 8) and state["heard"].any()
    assert state["keep"].shape == (1, 4) and np.isclose(state["keep"].sum(), 1.0)
    # the place region held a peak on each axis around the position before the step
    peak = state["place"].reshape(2, interface.place)
    centers = np.linspace(PLANE[0], PLANE[1], interface.place)
    assert 2 <= (peak[0] > 0).sum() <= 5 and 2 <= (peak[1] > 0).sum() <= 5
    assert abs(centers[int(peak[0].argmax())] - 0.3) <= (PLANE[1] - PLANE[0]) / 7


def test_reports_without_names_go_to_the_nearer_rows_of_their_look():
    space = small()
    starts = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]])
    space.begin(starts, looks=[0, 0, 7])
    space.predict()
    seen, z = space.assign(np.array([[0.52, 0.01, 0], [0.01, 0.02, 0], [0.9, 0.9, 3]]))
    assert seen.tolist() == [True, True, False]
    assert np.array_equal(z[:2], [[0.01, 0.02], [0.52, 0.01]])
    space.observe(seen, z)
    cues = space.hear(np.array([[0.1, 0.5, 7], [0.3, 0.2, 9]]))
    assert np.isnan(cues[:2]).all() and cues[2].tolist() == [0.1, 0.5]
    assert np.isnan(space.hear(np.zeros((0, 3)))).all()


def test_a_thing_heard_while_seen_teaches_the_place_a_cue_names():
    space = small()
    rng = np.random.default_rng(4)
    start = np.array([[0.1, 0.6], [-0.4, 0.9]])
    ear = np.array(space.interface.ear)
    space.begin(start, looks=[1, 2])
    for _ in range(12):
        away = start - ear
        cues = np.column_stack([np.arctan2(away[:, 1], away[:, 0]), np.linalg.norm(away, axis=1)])
        space.advance([True, True], start + rng.normal(0.0, 0.01, (2, 2)), cues=cues, learn=True)
    assert space.work["lesson_row_sweeps"] > 0 and space.state()["heard"].any()


def test_a_thing_that_does_not_come_out_where_it_would_be_seen_is_held_back():
    space = small(read="seen")  # the seen movement itself moves the place
    start = np.array([[-0.5, 0.5]])
    space.begin(start)
    for t in range(1, 11):  # seen moving right at 2 cm a step
        space.advance([True], start + [[0.02 * t, 0.0]])
    last = space.where()[0].copy()
    hidden = np.zeros(1, dtype=bool)
    for _ in range(10):  # behind a pillar: carried on along its seen movement
        space.predict()
        space.observe(hidden, np.zeros((1, 2)), in_sight=[False])
    carried = space.where()[0].copy()
    assert carried[0, 0] > last[0, 0] + 0.15
    for _ in range(3):  # it would be seen by now if it had kept going, and no report comes
        space.predict()
        space.observe(hidden, np.zeros((1, 2)), in_sight=[True])
    place, radius = space.where()
    assert np.allclose(place, (last + carried) / 2.0, atol=0.03)  # halfway along the hidden path
    assert radius[0] >= np.linalg.norm(carried - last) / 2.0  # the circle reaches both ends
    space.predict()
    space.observe(hidden, np.zeros((1, 2)), in_sight=[True])
    assert np.array_equal(space.where()[0], place) and space.state()["live"].tolist() == [False]


def test_a_thing_on_a_hidden_stretch_waits_at_its_middle():
    space = small(read="seen")
    start = np.array([[-0.5, 0.5]])
    space.begin(start)
    for t in range(1, 11):  # seen moving right at 2 cm a step, to x = -0.30
        space.advance([True], start + [[0.02 * t, 0.0]])

    def looks(points):  # a pillar's shadow hides x from -0.29 to 0.11
        x = np.asarray(points)[:, 0]
        return (x < -0.29) | (x > 0.11)

    hidden = np.zeros(1, dtype=bool)
    errors = []
    for t in range(1, 41):  # the thing stops 8 cm into the shadow and rests there
        truth = np.array([-0.30 + min(0.02 * t, 0.08), 0.5])
        space.predict()
        place = space.observe(hidden, np.zeros((1, 2)), in_sight=looks)
        errors.append(np.linalg.norm(place[0] - truth))
    place, radius = space.where()
    assert abs(place[0, 0] - (-0.30 + 0.21)) < 0.03  # the middle of the hidden stretch
    assert max(errors) < 0.16 and errors[-1] < 0.16 and radius[0] >= errors[-1]
    assert space.state()["live"].tolist() == [False]  # it waits, and the brain is not asked
    space.advance([True], np.array([[-0.22, 0.5]]))  # seen again: the report writes the place
    assert np.allclose(space.where()[0], [[-0.22, 0.5]])


def test_a_kind_is_answered_with_its_places_and_a_place_with_its_kind():
    space = small()
    starts = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]])
    space.begin(starts, looks=[3, 3, 7])  # two apples and a loaf
    rows, places, radius = space.where_is(3)
    assert rows.tolist() == [0, 1] and np.array_equal(places, starts[:2]) and len(radius) == 2
    assert space.where_is(9)[0].size == 0
    assert space.what_is_at([0.49, 0.02]) == (1, 3) and space.what_is_at([0.01, 0.99]) == (2, 7)
    assert space.what_is_at([0.25, 0.5]) is None  # nothing is held there
    assert space.what_is_at([0.25, 0.5], reach=0.6) is not None


def test_a_thing_at_rest_is_filed_and_found_again_by_what_it_looks_like():
    space = small(records=16)
    rng = np.random.default_rng(1)
    space.begin(np.array([[0.2, 0.3], [-0.4, 0.8]]), looks=[4, 9])
    assert space.find(4) is None
    for _ in range(30):  # at rest from its sixth report on, when its line says so
        reports = np.array([[0.2, 0.3], [-0.4, 0.8]]) + rng.normal(0.0, 0.01, (2, 2))
        space.advance([True, True], reports)
    place, count = space.find(4)
    assert np.linalg.norm(place - [0.2, 0.3]) < 0.01 and count == 16
    assert not space._moving.any()
    assert space.records.count == 2 and space.find(5) is None
    space.advance([True, False], np.array([[0.6, 0.6], [0.0, 0.0]]))  # seen elsewhere
    place, count = space.find(4)
    assert np.array_equal(place, [0.6, 0.6]) and count == 1 and space.records.count == 2
    space.begin(np.array([[np.nan, np.nan], [0.0, 0.0]]), looks=[9, 1])  # the records stay
    assert np.linalg.norm(space.where()[0][0] - [-0.4, 0.8]) < 0.01
    twins = small(records=16)
    twins.begin(np.array([[0.2, 0.3], [-0.4, 0.8]]), looks=[4, 4])
    for _ in range(30):
        twins.advance([True, True], np.array([[0.2, 0.3], [-0.4, 0.8]]))
    assert twins.records.count == 0  # two things that look alike share no record


def test_lessons_hold_only_reports_and_practice_leaves_some_unread():
    space = small(practice=0.2, span=5)
    rng = np.random.default_rng(2)
    velocity = np.array([[0.01, 0.004], [-0.008, 0.006], [0.0, 0.0], [0.006, -0.01]])
    start = np.array([[-0.5, 0.2], [0.5, 0.3], [0.0, 0.8], [-0.2, 1.0]])
    space.begin(start)
    unread = 0
    for t in range(1, 61):
        truth = start + t * velocity
        space.advance(np.ones(4, dtype=bool), truth + rng.normal(0.0, 0.01, (4, 2)), learn=True)
        unread += int((space.since > 0).sum())
    assert space.work["lesson_row_sweeps"] > 0 and unread > 0
    assert 0.005 < np.sqrt(space.noise) < 0.02
    assert space.seen_map.sum() == 4 * 60 and space.visit_map.sum() == 4 * 60
