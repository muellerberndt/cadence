"""The parrot (#332): its body, eyes, early vision, reach, focus and brain."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadence.parrot import Body, Scene, Thing, render
from cadence.parrot.brain import GENES, Parrot, triangulate
from cadence.parrot.distance import Distance
from cadence.parrot.focus import Focus
from cadence.parrot.vision import Vision, components
from cadence.parrot.world import HIGH, LOW, NAMES, stand_ins


def test_body_servo_and_eyes():
    body = Body()
    eyes = body.eyes()
    assert np.allclose(eyes.places[:, 1], 0.13 + 0.06 - 0.025 * 0.3 + 0.07 + 0.05)
    assert np.isclose(np.linalg.norm(eyes.places[0] - eyes.places[1]), 0.036)
    assert eyes.places[0, 0] > 0 > eyes.places[1, 0]  # the left eye at +x
    k = NAMES.index("head_yaw")
    targets = np.full(len(NAMES), np.nan)
    targets[k] = 1.0  # as far left as the neck turns
    for _ in range(5):
        angles = body.move(targets)
    assert angles[k] == HIGH[k] == 75.0
    assert np.isclose(body.velocity[k], 0.0) or body.velocity[k] <= 360.0
    turned = body.eyes()
    assert turned.places[0, 2] < eyes.places[0, 2]  # the eyes moved with the turn
    assert np.allclose(Body.unscaled(Body.scaled(body.angles)), body.angles)
    assert LOW[k] == -75.0


def test_render_both_eyes_and_cover():
    scene = Scene([Thing("a", "apple", 0.0, 0.35, stand=0.15),
                   Thing("b", "ball", 0.0, -0.4, stand=0.1)])
    sight = render(scene, Body().eyes())
    assert sight.images.shape == (2, 100, 175, 3) and sight.images.dtype == np.uint8
    # the apple in front: seen by both eyes in the columns toward the beak; the ball behind:
    # in the blind spot
    assert sight.coverage[0, 0] > 20 and sight.coverage[1, 0] > 20
    assert sight.coverage[:, 1].sum() == 0
    assert (np.nonzero(sight.ids[0] == 0)[1] >= 150).all()
    covered = render(scene, Body().eyes(), cover="left")
    assert covered.images[0].max() == 0 and np.array_equal(covered.images[1], sight.images[1])


def test_components_match_a_flood_fill():
    rng = np.random.default_rng(0)
    for _ in range(30):
        mask = rng.random((9, 13)) < 0.5
        labels = components(mask)
        assert ((labels >= 0) == mask).all()
        for (a, b), (c, d) in (((0, 0), (0, 1)), ((3, 4), (4, 4))):
            if mask[a, b] and mask[c, d]:
                assert labels[a, b] == labels[c, d]


def test_disparity_of_a_thing_in_front():
    table = stand_ins()
    vision, body = Vision(), Body()
    thing = Thing("a", "ball", 0.02, 0.32, stand=0.2)
    scene = Scene([thing])
    for _ in range(2):
        figures = vision.see(render(scene, body.eyes()).images)
    eyes = body.eyes()
    centre = thing.center(table)
    sides = [eyes.rotation.T @ (centre - eyes.places[e]) for e in (0, 1)]
    true = math.degrees(math.atan2(sides[1][0], sides[1][2]) - math.atan2(sides[0][0],
                                                                            sides[0][2]))
    left = [f for f in figures if f.eye == 0 and f.disparity is not None]
    assert len(left) == 1 and left[0].pair is not None
    found = float((left[0].disparity * vision.channels).sum() / left[0].disparity.sum())
    assert abs(found - true) < 0.4


def test_reach_starts_silent_and_learns():
    reach = Distance(5, units=12, sizes=4, heights=4, shifts=4, baselines=3)
    rng = np.random.default_rng(1)
    preferred = np.linspace(0.0, 4.0, 5)

    def code(d: float) -> np.ndarray:
        return reach.code(np.exp(-0.5 * (preferred - 1.0 / d) ** 2), 5.0, 0.0)

    test = np.array([code(d) for d in (0.3, 0.5)])
    far, _ = reach.read(reach.settle(test))
    assert np.isnan(far).all()  # a newborn reach has no sense of distance
    for _ in range(150):
        ds = rng.uniform(0.25, 0.6, 6)
        reach.teach(np.array([code(d) for d in ds]), ds)
    far, _ = reach.read(reach.settle(test))
    assert np.all(np.abs(np.log(far / np.array([0.3, 0.5]))) < 0.2)
    assert reach.parameters() == len(reach.pairs)


def test_triangulate_two_rays():
    p = np.array([0.1, 0.2, 0.5])
    o1, o2 = np.array([0.0, 0.2, 0.0]), np.array([0.03, 0.2, 0.0])
    u1, u2 = (p - o1) / np.linalg.norm(p - o1), (p - o2) / np.linalg.norm(p - o2)
    at, gap, s, t, angle = triangulate(o1, u1, o2, u2)
    assert np.allclose(at, p) and gap < 1e-9 and angle > 0
    assert np.isclose(s, np.linalg.norm(p - o1)) and np.isclose(t, np.linalg.norm(p - o2))


def test_focus_holds_then_moves():
    focus = Focus(slots=4)
    things = [{"kind": 0, "place": [0.2, 0.3]}, {"kind": 1, "place": [-0.3, 0.1]}]
    first = focus.choose(things, 0)
    assert first is not None and first.why in ("appeared", "new to the eyes")
    held = [focus.choose(things, m) for m in range(1, 4)]
    assert all(c is not None and c.kind == first.kind for c in held)
    later = [focus.choose(things, m) for m in range(4, 30)]
    assert any(c is not None and c.kind != first.kind for c in later)  # habituation
    moved = [{"kind": 0, "place": [0.2, 0.3]}, {"kind": 1, "place": [-0.3, 0.17]}]
    choice = focus.choose(moved, 31)
    assert choice is not None and choice.kind == 1


def test_parrot_lives_a_few_moments_and_silences():
    scene = Scene([Thing("a", "apple", 0.05, 0.35, stand=0.1),
                   Thing("b", "cup", 0.35, 0.05, stand=0.05)])
    parrot = Parrot({"unknown_gene": 3}, seed=0)
    assert parrot.genes["unknown_gene"] == 3  # carried unchanged
    for _ in range(8):
        parrot.moment(render(scene, parrot.body.eyes()).images)
    assert parrot.moments == 8 and parrot.map.things()
    cost = parrot.cost()
    assert cost["moments"] == 8 and cost["macs"]["map"] > 0
    reach = parrot.regions["distance"]
    parrot.silence(np.arange(reach.start, reach.stop))
    out = parrot.moment(render(scene, parrot.body.eyes()).images, learn=False)
    assert all(np.isnan(s.distance) for s in out["sightings"])
    with pytest.raises(ValueError):
        parrot.silence([parrot.regions["map"].start])
    parrot.silence(None)


def test_random_wiring_has_as_many_synapses():
    hand = Parrot(seed=1)
    loose = Parrot({"wiring": "random"}, seed=1)
    assert loose.distance.parameters() == hand.distance.parameters()
    assert not np.array_equal(loose.distance.pairs, hand.distance.pairs)
    assert set(GENES) <= set(hand.genes)


def test_save_load_and_the_page_host(tmp_path):
    import json

    from cadence.parrot.host import Host, scene_of

    scene = Scene([Thing("a", "apple", 0.05, 0.35, stand=0.1),
                   Thing("b", "cup", 0.35, 0.05, stand=0.05)])
    parrot = Parrot(seed=2)
    for _ in range(12):
        parrot.moment(render(scene, parrot.body.eyes()).images)
    path = parrot.save(tmp_path / "brain.npz", card={"trained on": "a test"})
    again = Parrot.load(path)
    assert np.array_equal(again.distance.graph.efficacy, parrot.distance.graph.efficacy)
    assert again.kinds.used == parrot.kinds.used and again.genes == parrot.genes
    host = Host({"files": str(tmp_path)})
    assert not host.newborn
    described = host.describe()
    json.dumps(described, allow_nan=False)
    assert described["groups"][0]["name"] == "distance"
    host.handle({"command": "scene", "scene": scene_of(scene.things)})
    for _ in range(3):
        reply = host.step({"show_eyes": True}, {})
        json.dumps(reply, allow_nan=False)
    assert set(reply["buffers"]) == {"activity", "angles", "targets", "left", "right"}
    assert host.buffers["activity"].shape == (described["neurons"],)
    assert host.buffers["left"].size == 100 * 175 * 3
    assert host.silence("distance", True)["silenced"] == ["distance"]
    reply = host.step({}, {})
    assert reply["readout"]["distance"] is None


def test_whole_brain_exports(tmp_path):
    import json

    from cadence.parrot.whole import WholeParrot

    whole = WholeParrot(seed=0, association=16, words=8)
    scene = Scene([Thing("a", "apple", 0.03, 0.33, stand=0.12)])
    for k in range(3):
        whole.moment(render(scene, whole.parrot.body.eyes()).images,
                     record="sees an apple" if k == 2 else None)
    whole.moment(render(scene, whole.parrot.body.eyes()).images, reward=1.0,
                 record="gets a pistachio")
    whole.export(tmp_path)
    index = json.loads((tmp_path / "brain.json").read_text())
    n = index["neurons"]
    assert (tmp_path / "neurons.bin").stat().st_size == n * 14
    first = index["moments"][0]
    assert (tmp_path / first["trace"]).stat().st_size == (first["sweeps"] + 1) * n * 2
    assert index["moments"][1]["reward"] == 1.0
    assert {"retina", "wulst", "map", "ncl", "vta"} <= {r["id"] for r in index["regions"]}
    assert sum(t["synapses"] for t in index["tracts"]) == index["synapses"]


def test_heard_records_grow_deeper_with_each_hearing():
    from cadence.parrot.records import HeardRecords

    rng = np.random.default_rng(0)
    records = HeardRecords(width=16, value_width=8, capacity=32)
    apple, cup = np.eye(16)[2], np.eye(16)[9]
    word = rng.normal(size=8)
    for _ in range(4):  # "apple" heard four times with the apple in focus
        records.write(apple, word + rng.normal(scale=0.05, size=8), surprise=0.5)
    records.write(cup, rng.normal(size=8))  # "cup" heard once
    often, once = records.fetch(apple), records.fetch(cup)
    assert often is not None and once is not None
    assert often[1] > once[1]  # the word heard often is the stronger record
    assert np.corrcoef(often[0], word)[0, 1] > 0.9


def test_the_focus_picks_among_many_things_in_view_pointed_and_moving():
    focus = Focus()
    things = [{"kind": k, "place": [0.4 * math.cos(k), 0.4 * math.sin(k)],
               "state": "held"} for k in range(6)]
    things[3]["state"] = "seen"
    first = focus.choose(things, 0)
    assert first is not None and first.kind == 3  # six things: one wins, the one in view
    for m in range(1, 12):  # all at rest for a while: their usual move is near zero
        focus.choose(things, m)
    held = focus.current
    assert held is not None
    ray = np.asarray(things[5]["place"], dtype=float)
    pointed = focus.choose(things, 12, pointed=(np.zeros(2), ray))
    assert pointed is not None and pointed.kind == 5 and pointed.why == "pointed at"
    moving = [dict(t) for t in things]
    moving[1]["place"] = [moving[1]["place"][0] + 0.06, moving[1]["place"][1]]
    drawn = focus.choose(moving, 13)
    assert drawn is not None and drawn.kind == 1 and drawn.why == "moved"


def test_the_host_viz_block():
    import json

    from cadence.parrot.host import Host, scene_of

    host = Host({"whole": "page", "learn": True})
    host.handle({"command": "scene", "scene": scene_of([
        Thing("a", "apple", 0.05, 0.33, stand=0.1), Thing("b", "cup", 0.35, 0.1, stand=0.05)])})
    for _ in range(4):
        reply = host.step({"viz": True}, {"audio": np.zeros(1600, np.float32)})
    viz = reply["viz"]
    json.dumps(viz, allow_nan=False)
    assert viz["v"] == 1 and viz["settle"]["frames"] == 16
    assert {"tectum", "ncl"} <= set(viz["settle"]["regions"])
    assert all(len(v) == 16 for v in viz["settle"]["regions"].values())
    assert host.step({}, {})["viz"] is None


def test_a_still_room_draws_no_surprise_a_push_and_a_new_thing_do():
    """The owner's rule: a surprise is a change the brain did not cause and could not predict."""
    from cadence.parrot.world import NAMES

    things = [Thing("apple", "apple", 0.05, 0.33, stand=0.1),
              Thing("cup", "cup", 0.35, 0.1, stand=0.05),
              Thing("ball", "ball", -0.3, 0.25, stand=0.15),
              Thing("pear", "pear", -0.35, -0.2, stand=0.0),
              Thing("duck", "duck", 0.2, -0.35, stand=0.1)]
    scene = Scene(things)
    parrot = Parrot(seed=0)
    events, turns = [], 0
    yaw = NAMES.index("head_yaw")
    for m in range(300):  # a still room; the head makes its own saccades
        parrot.moment(render(scene, parrot.body.eyes()).images)
        turns += abs(parrot.body.velocity[yaw]) > 10.0
        if m >= 10:
            events += [(m, "moved") for c in parrot.focus.candidates if c.moved == parrot.moments - 1]
            events += [(m, "appeared") for c in parrot.focus.candidates
                       if c.entered == parrot.moments - 1]
    assert turns > 20 and events == []
    # a thing pushed from rest: the visitor points at the apple, the head settles on it, the
    # apple is pushed aside
    apple = things[0]
    for k in range(40):  # the head turns to it, then holds still past the efference window
        e = parrot.body.eyes()
        parrot.point(math.degrees(math.atan2(apple.x - e.pivot[0], apple.z - e.pivot[2]))
                     - parrot.body.joint("head_yaw"))
        parrot.moment(render(scene, parrot.body.eyes()).images, act=k < 8)
        if k >= 8 and parrot.moments - 1 - parrot.focus.turned_at > parrot.focus.after_turn:
            break
    assert any(np.linalg.norm(c.place - [apple.x, apple.z]) < 0.15 for c in parrot.focus.candidates)
    before = parrot.moments
    apple.x += 0.06
    flagged = []
    for _ in range(2):
        parrot.moment(render(scene, parrot.body.eyes()).images)
        flagged += [c for c in parrot.focus.candidates if c.moved >= before]
    assert flagged
    # a thing added where none of its kind is
    scene.things.append(Thing("block", "block", 0.45, 0.35, stand=0.05))  # in view, apart
    added = []
    for _ in range(40):
        parrot.moment(render(scene, parrot.body.eyes()).images)
        added += [c for c in parrot.focus.candidates if c.entered >= before]
    assert added
