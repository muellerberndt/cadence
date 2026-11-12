"""cadence.space.map: what is where, held in a sheet's own settled activity."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.space.map import Map, Sense, sparse_codes

PLANE = (-1.0, 1.0, -0.2, 1.4)
APPLE, BREAD, CUP, HAND = 0, 1, 2, 3


def places(count: int, seed: int = 0, apart: float = 0.25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for _ in range(200_000):
        if len(out) == count:
            break
        point = np.array([rng.uniform(-0.85, 0.85), rng.uniform(-0.05, 1.25)])
        if all(np.linalg.norm(point - other) > apart for other in out):
            out.append(point)
    assert len(out) == count
    return np.array(out)


def test_several_things_are_held_for_a_thousand_moments_and_answered_both_ways():
    held = Map(PLANE)
    kinds = [APPLE, APPLE, BREAD, CUP, HAND, 7, 9, 12]           # two things of one kind
    at = places(len(kinds))
    for kind, place in zip(kinds, at, strict=True):              # each enters when it is seen
        held.moment(seen=[(kind, place)])
    for _ in range(1000):                                        # no sensory information
        held.moment()
    for kind in set(kinds):
        want = at[[k == kind for k in kinds]]
        got = held.where_is(kind)
        assert len(got) == len(want)
        for place in want:
            assert np.linalg.norm(got - place, axis=1).min() < 0.001   # a millimetre
    assert [held.what_is_at(place) for place in at] == kinds
    assert held.what_is_at([0.99, 1.39]) is None and len(held.where_is(5)) == 0
    things = held.things()
    assert sorted(t["kind"] for t in things) == sorted(kinds)
    assert all(t["state"] == "held" and t["since"] >= 1000 for t in things)
    assert max(len(t["cells"]) for t in things) <= 4


def test_an_emptied_map_reports_no_thing_and_a_file_keeps_the_map(tmp_path):
    held = Map(PLANE, cells=(16, 13), units=8)
    held.moment(seen=[(APPLE, [0.2, 0.5]), (CUP, [-0.6, 1.0])])
    held.moment()
    again = Map.load(held.save(tmp_path / "map.npz"))
    assert np.array_equal(again.potential, held.potential)
    assert np.allclose(again.where_is(APPLE), held.where_is(APPLE), atol=1e-9)
    assert again.what_is_at([-0.6, 1.0]) == CUP and again.things()[0]["since"] == 1
    held.reset()                                                 # the carried activity emptied
    held.moment()
    assert held.things() == [] and len(held.where_is(APPLE)) == 0
    assert held.what_is_at([0.2, 0.5]) is None and not held.potential.any()


def test_a_thing_seen_moving_is_one_thing_at_its_latest_report():
    held = Map(PLANE)
    rng = np.random.default_rng(1)
    for t in range(60):
        truth = np.array([-0.7 + 0.02 * t, 0.3 + 0.01 * t])
        report = truth + rng.normal(0.0, 0.01, 2)
        held.moment(seen=[(HAND, report)], lit=np.ones(held.count, dtype=bool))
        got = held.where_is(HAND)
        assert len(got) == 1 and np.linalg.norm(got[0] - report) < 0.0005
    (thing,) = held.things()
    assert thing["kind"] == HAND and thing["state"] == "seen" and thing["since"] == 0


def test_things_arrive_unannounced_and_two_of_a_kind_stay_two():
    held = Map(PLANE)
    assert held.things() == []
    held.moment(seen=[(APPLE, [0.0, 0.0])])
    held.moment(seen=[(BREAD, [0.5, 0.9])])                       # a kind the map did not hold
    held.moment(seen=[(APPLE, [-0.8, 1.2])])                      # an apple far from the first
    assert len(held.where_is(APPLE)) == 2 and len(held.where_is(BREAD)) == 1
    held.moment(seen=[(APPLE, [0.3, 0.2])])                       # within a stride of the first
    got = held.where_is(APPLE)
    assert len(got) == 2 and np.linalg.norm(got - [0.3, 0.2], axis=1).min() < 0.0005
    assert np.linalg.norm(got - [0.0, 0.0], axis=1).min() > 0.2   # the old place is cleared
    held.moment(heard=[(BREAD, [0.2, 1.1])])                      # a sound names its new place
    (bread,) = [t for t in held.things() if t["kind"] == BREAD]
    assert bread["state"] == "heard" and np.linalg.norm(bread["place"] - [0.2, 1.1]) < 0.0005


@pytest.mark.parametrize("empty", ["near", "clear", "keep"])
def test_a_place_seen_empty(empty):
    held = Map(PLANE, empty=empty)

    def lit(points):
        return points[:, 0] > -0.1                                # a shadow begins 10 cm away

    held.moment(seen=[(HAND, [0.0, 0.6]), (CUP, [-0.6, 0.3])], lit=lit)
    held.moment(lit=lit)                                          # the hand's place is seen empty
    got, things = held.where_is(HAND), {t["kind"]: t for t in held.things()}
    assert np.allclose(held.where_is(CUP), [[-0.6, 0.3]], atol=0.0005)         # the cup, out of sight, stays
    if empty == "keep":
        assert np.allclose(got, [[0.0, 0.6]], atol=0.0005) and things[HAND]["state"] == "held"
    elif empty == "clear":
        assert len(got) == 0 and things[HAND]["state"] == "not known where"
        assert things[HAND]["place"] is None
        held.moment(seen=[(HAND, [0.4, 0.4])], lit=lit)           # found again when seen
        assert [t["state"] for t in held.things() if t["kind"] == HAND] == ["seen"]
    else:
        assert len(got) == 1 and -0.7 < got[0][0] < -0.1 and abs(got[0][1] - 0.6) < 0.05
        assert things[HAND]["state"] == "near" and things[HAND]["since"] == 1
        assert not lit(held.centers[things[HAND]["cells"]]).any()
        held.moment(seen=[(HAND, [0.05, 0.6])], lit=lit)          # it comes out: one hand again
        assert np.allclose(held.where_is(HAND), [[0.05, 0.6]], atol=0.0005)
        held.moment(lit=np.ones(held.count, dtype=bool))          # no cell is out of sight
        assert [t["state"] for t in held.things()] == ["not known where"] * 2


def test_many_kinds_on_few_units():
    codes = sparse_codes(128, 32)
    assert codes.shape == (128, 32) and (codes.sum(axis=1) == 3).all()
    assert ((codes @ codes.T) - 3 * np.eye(128)).max() <= 1       # two kinds share one unit at most
    held = Map(PLANE, units=32, kinds=128)
    at = places(128, seed=3, apart=0.10)
    for kind, place in enumerate(at):
        held.moment(seen=[(kind, place)])
    for _ in range(50):
        held.moment()
    right = sum(held.what_is_at(place) == kind for kind, place in enumerate(at))
    near = sum(len(got := held.where_is(kind)) == 1 and np.linalg.norm(got[0] - place) < 0.005
               for kind, place in enumerate(at))
    assert right >= 122 and near >= 122                           # .95 of 128
    assert len(held.things()) >= 122


def test_a_thing_in_two_pieces_of_doubt_is_one_entry_and_a_settle_is_kept_by_sweep():
    held = Map(PLANE, trace=True)

    def lit(points):                                              # two shadows, left and right
        return np.abs(points[:, 0]) < 0.15

    held.moment(seen=[(HAND, [0.0, 0.6])], lit=lit)
    held.moment(lit=lit)                                          # seen empty between them
    (hand,) = held.things()
    assert hand["state"] == "near" and len(hand["pieces"]) == 2
    assert sum(len(p) for p in hand["pieces"]) == len(hand["cells"]) and abs(hand["place"][0]) < 0.05
    seen = held.trajectory
    assert seen["asked"] == "moment" and seen["sweeps"] == 2
    assert seen["map"].shape == (3, 32, 26, 16) and seen["where"].shape == (3, 32, 26)
    assert seen["map"][0].sum() == 0 and seen["map"][1].sum() > 0     # the sheet settles first
    assert seen["where"][1].sum() == 0 and seen["where"][2].sum() > 0  # then where and what
    assert np.array_equal(seen["map"][1], seen["map"][2])
    held.where_is(HAND)
    assert held.trajectory["asked"] == "where_is" and held.trajectory["what"][2, HAND] > 0
    held.moment(seen=[(HAND, [0.3, 0.6])], lit=lit)               # a report ends the doubt
    assert [t["state"] for t in held.things()] == ["seen"]


def test_a_sense_is_taught_the_place_a_reading_names():
    held = Map(PLANE, cells=(16, 13))
    arm = Sense(held, ranges=[(0.3, 2.8), (-1.2, 1.2)], per=(10, 10), rate=0.5)

    def hand(joints):                                             # an arm of two links
        a, b = joints[..., 0], joints[..., 0] + joints[..., 1]
        return np.stack([0.45 * np.cos(a) + 0.35 * np.cos(b),
                         0.45 * np.sin(a) + 0.35 * np.sin(b)], axis=-1)

    assert arm.named([1.0, 0.2]) is None                          # before any lesson
    rng = np.random.default_rng(0)
    probes = np.stack([rng.uniform(0.5, 2.6, 20), rng.uniform(-1.0, 1.0, 20)], axis=1)
    for _ in range(150):                                          # felt while it is seen
        joints = np.stack([rng.uniform(0.3, 2.8, 16), rng.uniform(-1.2, 1.2, 16)], axis=1)
        arm.teach(joints, hand(joints))
    off = [np.linalg.norm(arm.named(q) - hand(q)) for q in probes]
    assert arm.lessons == 150 * 16 and np.mean(off) < 0.08       # 5.7 cm on cells of 12 cm
    held.moment(felt=[(HAND, arm.named(probes[0]))])              # the reading writes the map
    (thing,) = held.things()
    assert thing["state"] == "felt" and np.linalg.norm(thing["place"] - hand(probes[0])) < 0.15
