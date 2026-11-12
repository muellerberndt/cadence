"""The tracking chamber's contracts: the world's step, the scene files, the controls and the
scores."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(name: str):
    key = f"tracking_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


tr = _load("tracking")
w = tr.w
ARRAYS = ("report", "in_view", "commands", "joints", "truth", "start", "cue", "what")
EXAMPLE_SEED = 8  # scenes/example.json: make --case other --kind stop --seed 8 --steps 60


def same(a, b) -> bool:
    return all(np.allclose(getattr(a, n), getattr(b, n), equal_nan=True, atol=1e-12)
               for n in ARRAYS)


def test_the_step_loop_and_the_recorder_write_one_scene():
    for build in (w.own_scene, w.other_scene, w.room_scene):
        whole = build(np.random.default_rng(7))
        keep = w.record

        def one_step_at_a_time(*a, keep=keep, **k):
            return keep(*a, **{**k, "stepwise": True})

        w.record = one_step_at_a_time
        try:
            stepped = build(np.random.default_rng(7))
        finally:
            w.record = keep
        assert same(whole, stepped) and whole.kind == stepped.kind


def test_one_step_moves_the_arm_and_reports_what_is_in_view():
    geometry = w.Geometry()
    pose = np.array([1.3, -1.1])
    camera, disc = np.array(geometry.camera), np.array(geometry.occluders[0][:2])
    behind = disc + 0.5 * (disc - camera) / np.linalg.norm(disc - camera)
    world = w.World(geometry, pose.copy(), np.array([[-0.5, 1.0], behind]), motor_noise=0.0)
    zero = {"motor": np.zeros(2), "encoder": np.zeros(2), "camera": np.zeros((2, 2)),
            "cue": np.zeros((2, 2))}
    row = w.step(world, np.array([0.1, 0.0]), np.array([[0.02, 0.0], [0.0, 0.0]]),
                 np.array([False, True]), zero)
    assert np.allclose(row["joints"], pose + [0.1, 0.0])
    assert np.allclose(row["truth"], [[-0.48, 1.0], behind])
    assert row["in_view"].tolist() == [True, False]  # the second lies behind the disc
    assert np.isnan(row["report"][1]).all() and np.allclose(row["report"][0], [-0.48, 1.0])
    assert np.isnan(row["cue"][0]).all()
    assert np.allclose(w.cartesian(geometry, row["cue"][1]), behind)


def test_the_blind_spot_is_the_discs_shadow_the_arms_shadow_and_outside_the_field():
    geometry = w.Geometry()
    pose = np.array([1.3, -1.1])
    camera, disc = np.array(geometry.camera), np.array(geometry.occluders[0][:2])
    behind = disc + 0.5 * (disc - camera) / np.linalg.norm(disc - camera)
    beside = behind + np.array([0.0, 0.5])
    chain = w.forward(geometry, pose)
    middle = (chain[0] + chain[1]) / 2
    shaded = middle + 0.15 * (middle - camera) / np.linalg.norm(middle - camera)
    seen = w.visible(geometry, np.array([behind, beside, shaded, [1.5, 0.5]]), pose)
    assert seen.tolist() == [False, True, False, False]
    assert w.visible(geometry, shaded[None], np.array([2.6, -0.4]))[0]  # the arm elsewhere
    narrow = w.Geometry(fov=0.3)
    assert not w.visible(narrow, beside[None], pose)[0]
    # the arm's own hand tops its own link
    assert w.visible(geometry, w.hand(geometry, pose)[None], pose, 0)[0]


def test_an_arm_at_rest_stays_and_a_command_moves_it_by_about_its_size():
    scene = w.own_scene(np.random.default_rng(3))
    moved = np.linalg.norm(np.diff(scene.truth[:, 0], axis=0), axis=1)
    asked = np.abs(scene.commands[1:]).sum(axis=1)
    assert (moved[asked == 0] == 0).all() and (moved[asked > 0] > 0).all()
    assert scene.commands[0].tolist() == [0.0, 0.0]
    made = np.diff(scene.joints, axis=0)
    assert np.abs(made - scene.commands[1:]).max() < 0.03


def test_a_persons_hand_goes_straight_slows_stops_or_turns():
    for kind in ("straight", "slow", "stop", "turn"):
        scene = w.other_scene(np.random.default_rng(5), kind=kind)
        step = np.linalg.norm(np.diff(scene.truth[:, 0], axis=0), axis=1)
        moving = np.flatnonzero(step > 0)
        e = scene.event
        assert (e == -1) == (kind == "straight")
        first = step[0]
        if kind == "straight":
            assert np.allclose(step[moving], first)
        elif kind == "stop":
            assert np.allclose(step[:e], first) and (step[e + 8 :] == 0).all()
        elif kind == "slow":
            late = step[e + 8 :][step[e + 8 :] > 0]
            assert np.allclose(step[:e], first) and (late < 0.51 * first).all()
        else:
            heading = np.arctan2(*np.diff(scene.truth[:, 0], axis=0)[moving].T[::-1])
            turned = abs((heading[-1] - heading[0] + np.pi) % (2 * np.pi) - np.pi)
            assert np.allclose(step[moving], first) and np.deg2rad(39) < turned < np.deg2rad(111)
        assert (scene.commands == 0).all() and np.ptp(scene.joints, axis=0).max() < 0.01


def test_a_hand_left_out_of_view_rests_where_it_was_left():
    for seed in range(3):
        scene = w.other_scene(np.random.default_rng(seed), kind="left")
        e = scene.event
        assert not scene.in_view[e, 0] and scene.in_view[:10, 0].all()
        assert (np.diff(scene.truth[e:, 0], axis=0) == 0).all()
        first = int(np.flatnonzero(~scene.in_view[10:, 0])[0]) + 10
        assert 2 <= e - first <= 20
    mixed = tr.curriculum(0, 60, left=0.3)
    assert sum(s.kind == "left" for s in mixed) == 18 and len(mixed) == 60


def test_a_thing_at_rest_is_shaded_for_as_long_as_asked():
    for scene in (w.other_scene(np.random.default_rng(2), kind="stays", hold=120),
                  w.own_scene(np.random.default_rng(2), kind="stays", hold=120)):
        assert scene.steps == 180 and scene.in_view[:10, 0].all()
        assert not scene.in_view[60:, 0].any()
        assert (np.diff(scene.truth[60:, 0], axis=0) == 0).all()
        assert scene.unseen()[-1, 0] >= 120


def test_a_room_holds_a_thing_heard_moving_and_two_hands_that_cross_unseen():
    heard = w.room_scene(np.random.default_rng(4), kind="heard")
    sounding = np.flatnonzero(np.isfinite(heard.cue[:, 0, 0]))
    assert not heard.in_view[:, 0].any() and len(sounding) >= 18
    assert np.linalg.norm(heard.truth[-1, 0] - heard.truth[0, 0]) >= 0.2
    assert (heard.truth[sounding[-1], 0] == heard.truth[-1, 0]).all()
    assert heard.in_view[:, 1:].all() and len(set(heard.what.tolist())) == 3
    crossing = w.room_scene(np.random.default_rng(4), kind="crossing")
    at = crossing.event
    assert crossing.things == 8 and crossing.what[:2].tolist() == [w.HAND, w.HAND]
    assert not crossing.in_view[at - 4 : at + 5, :2].any() and crossing.in_view[-1, :2].all()
    gap = np.linalg.norm(crossing.truth[:, 0] - crossing.truth[:, 1], axis=1)
    assert gap[at - 4 : at + 5].min() < 0.1
    demo = w.room_scene(np.random.default_rng(4), kind="demo")
    assert demo.things == 4 and demo.what[0] == w.HAND


def test_varied_scenes_have_discs_of_their_own_and_spans_with_the_camera_off():
    plain = w.scenes("other", 21, 12)
    many = w.scenes("other", 21, 12, vary=True)
    assert len({s.geometry.occluders for s in plain}) == 1
    assert len({s.geometry.occluders for s in many}) == 12
    for scene in many:
        radii = [r for _, _, r in scene.geometry.occluders]
        assert 1 <= len(radii) <= 3 and 0.05 <= min(radii) and max(radii) <= 0.15
        assert scene.in_view[0, 0] and np.isnan(scene.report[~scene.in_view]).all()
    scene = w.other_scene(np.random.default_rng(3), kind="straight")
    dark = w.darkened(scene, np.random.default_rng(3))
    off = scene.in_view[:, 0] & ~dark.in_view[:, 0]
    assert 1 <= off.sum() <= 100 and not off[:10].any() and scene.in_view[:, 0].sum() > 0
    assert np.array_equal(dark.truth, scene.truth) and np.isnan(dark.report[off]).all()


def test_a_scene_file_reads_back_as_the_scene(tmp_path):
    one = w.other_scene(np.random.default_rng(1), kind="stop", steps=40)
    many = w.room_scene(np.random.default_rng(1), kind="heard", steps=60)
    for scene in (one, many, w.own_scene(np.random.default_rng(1), w.THREE_JOINTS, steps=30)):
        for suffix in (".json", ".npz"):
            w.save(scene, tmp_path / f"scene{suffix}")
            back = w.load(tmp_path / f"scene{suffix}")
            assert same(scene, back) and back.geometry == scene.geometry
            assert (back.case, back.kind, back.event, back.dt) == (
                scene.case, scene.kind, scene.event, scene.dt)
    held = json.loads((tmp_path / "scene.json").read_text())
    assert np.shape(held["truth"]) == (30, 2) and np.shape(held["in_view"]) == (30,)


def test_the_example_file_is_scored_as_the_generated_scene_is():
    example = w.load(HERE / "scenes" / "example.json")
    made = w.scenes("other", EXAMPLE_SEED, 1, kind="stop", steps=60)[0]
    assert same(example, made) and example.geometry == made.geometry
    assert example.steps == 60 and 0 < (~example.in_view).sum() < 60
    for build in (tr.Still, tr.ConstantVelocity):
        settings = {"q": 1e-6, "r": 0.01}
        a, ra = tr.follow(example, build, settings)
        b, rb = tr.follow(made, build, settings)
        assert tr.table([example], [a], [ra]) == tr.table([made], [b], [rb])


def test_a_scene_is_refused_with_its_reason():
    scene = w.other_scene(np.random.default_rng(1), steps=20)
    data = w.to_dict(scene)
    with pytest.raises(ValueError, match="format"):
        w.from_dict({**data, "format": "something/0"})
    with pytest.raises(ValueError, match="in "):
        w.from_dict({**data, "units": {"length": "mm", "time": "s", "angle": "rad"}})
    with pytest.raises(ValueError, match="commands and joints"):
        w.from_dict({**data, "commands": data["commands"][:-1]})
    seen = int(np.flatnonzero(scene.in_view[:, 0])[0])
    report = list(data["report"])
    report[seen] = None
    with pytest.raises(ValueError, match="finite wherever"):
        w.from_dict({**data, "report": report})
    with pytest.raises(ValueError, match="case"):
        w.from_dict({**data, "case": "elsewhere"})
    with pytest.raises(ValueError, match="'truth'"):
        w.from_dict({k: v for k, v in data.items() if k != "truth"})
    least = w.from_dict({k: data[k] for k in ("format", "case", "report", "in_view", "truth")})
    assert (least.commands == 0).all() and least.geometry == w.Geometry()
    assert np.allclose(least.start, scene.truth[0]) and least.dt == scene.dt


def test_a_learner_is_given_reports_without_names():
    scene = w.room_scene(np.random.default_rng(6), kind="crossing")
    first = tr.detections(scene, 0)
    assert first.shape == (8, 3) and (np.diff(first[:, 2]) >= 0).all()
    assert (np.diff(first[first[:, 2] == w.HAND, 0]) >= 0).all()
    at = scene.event
    rows = tr.detections(scene, at)
    assert len(rows) == int(scene.in_view[at].sum()) and not (rows[:, 2] == w.HAND).any()
    heard = w.room_scene(np.random.default_rng(6), kind="heard")
    step = int(np.flatnonzero(np.isfinite(heard.cue[:, 0, 0]))[0])
    assert (tr.sounds(heard, step)[:, 2] == heard.what[0]).any()
    one = w.other_scene(np.random.default_rng(6), kind="stop")
    feed = tr.given(one)
    out = ~one.in_view[:, 0]
    assert feed["seen"][0] and (feed["report"][out] == 0).all() and not feed["seen"][out].any()


def test_reports_of_things_that_look_alike_go_to_the_nearer_places():
    places = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert tr.assign(places, np.array([[0.9, 0.1], [0.1, 0.0]])) == [(0, 1), (1, 0)]
    assert tr.assign(places, np.array([[0.8, 0.0]])) == [(1, 0)]
    assert tr.assign(places[:1], np.array([[0.8, 0.0], [0.1, 0.0]])) == [(0, 1)]
    assert tr.assign(places, np.zeros((0, 2))) == []


def test_the_controls_answer_as_they_say():
    straight = next(s for s in w.scenes("other", 11, 40, kind="straight")
                    if w.panel(s) == "straight")
    still, _ = tr.follow(straight, tr.Still)
    moving, radius = tr.follow(straight, tr.ConstantVelocity, {"q": 1e-7, "r": 0.01})
    out = ~straight.in_view[:, 0]
    seen = straight.in_view[:, 0] & (np.arange(straight.steps) > 0)  # step 0 is given
    assert np.allclose(still[seen, 0], straight.report[seen, 0])
    assert np.allclose(still[0], straight.start)
    assert tr.errors(straight, moving)[out, 0].mean() < 0.5 * tr.errors(straight, still)[
        out, 0].mean()
    assert (np.diff(radius[:, 0])[out[1:]] > 0).all()  # the area grows while out of view
    own = w.own_scene(np.random.default_rng(8))
    reckoned = tr.reckoning(own)
    assert tr.errors(own, reckoned).mean() < 0.03
    assert tr.errors(own, tr.reckoning(own, encoders=True)).max() < 0.005
    settings = tr.fit_kalman(w.scenes("other", 12, 6))
    assert set(settings) == {"q", "r"}
    crossing = w.room_scene(np.random.default_rng(6), kind="crossing")
    answers, _ = tr.follow(crossing, tr.ConstantVelocity, {"q": 1e-7, "r": 0.01})
    assert tr.crossing([crossing], [answers])["kept"] in (0.0, 1.0)
    heard = w.room_scene(np.random.default_rng(6), kind="heard")
    deaf, _ = tr.follow(heard, tr.Still)
    hearing, _ = tr.follow(heard, tr.Still, {"cue": 0.08}, hear=True)
    assert tr.heard([heard], [hearing])["after"] < 0.5 * tr.heard([heard], [deaf])["after"]


def test_the_scores_read_the_true_answers_as_exact():
    scenes = w.scenes("other", 9, 12) + w.scenes("other", 9, quota={"stop": 2, "straight": 2})
    exact = [s.truth.copy() for s in scenes]
    got = tr.table(scenes, exact, [np.full((s.steps, 1), 0.01) for s in scenes])
    assert got["in_view"] == 0.0 and got["out_of_view"] == 0.0
    assert got["held"]["in_view"] == 1.0 and got["held"]["out_of_view"] == 1.0
    assert np.isclose(got["held"]["radius_in_view"], 0.01)
    assert 0.011 < got["camera"] < 0.014  # the report's own error: a mean distance
    assert sum(b["steps"] for b in got["by_unseen"].values()) == round(
        got["steps"] * got["share_out_of_view"])
    assert {"stop", "straight"} <= set(got["panels"]) and got["panels"]["stop"]["scenes"] >= 2
    off = [s.truth + [0.03, 0.04] for s in scenes]
    assert np.isclose(tr.table(scenes, off)["out_of_view"], 0.05)
    rest = w.scenes("other", 9, 2, kind="stays")
    held = tr.stays(rest, [s.truth for s in rest])
    assert held["unseen_end"] == 0.0 and held["moved"] == 0.0 and held["unseen_steps"] >= 1000
    rooms = w.scenes("room", 9, 2, kind="crossing")
    kept = tr.crossing(rooms, [s.truth for s in rooms])
    swapped = tr.crossing(rooms, [s.truth[:, [1, 0, 2, 3, 4, 5, 6, 7]] for s in rooms])
    assert kept["kept"] == 1.0 and swapped["kept"] == 0.0 and kept["rest_error"] == 0.0


def test_a_floor_brain_is_taught_and_answers_every_step():
    scenes = w.scenes("own", 13, 4, steps=12)
    for arm, inputs in (("raw", 5), ("code", 51)):
        floor = tr.Floor(scenes[0].geometry, arm, founder=0)
        assert floor.senses.inputs == inputs
        floor.train(scenes, rows=2)
        answers = floor.answer(scenes)
        assert len(answers) == 4 and answers[0].shape == (12, 1, 2)
        assert np.isfinite(answers[0]).all() and floor.macs() > 0
        assert floor.work["moments"] == 48
    seen = tr.Floor(scenes[0].geometry, "code", founder=0, lessons="camera", joints=True)
    assert seen.senses.inputs == 75
    seen.train(scenes[:2], rows=2)


def test_the_matched_baselines_train_and_answer():
    pytest.importorskip("torch")
    controls = _load("controls")
    scenes = w.scenes("other", 14, 12, steps=30)
    assert abs(controls.count(controls.build("lstm", controls.matched_width(
        "lstm", 5000, 5), 5, 0)) - 5000) < 300
    hidden = controls.spans(np.random.default_rng(0), np.ones(400, dtype=bool))
    assert not hidden[0] and 0.2 < hidden.mean() < 0.7
    for arm, lessons in (("lstm-online", "camera"), ("transformer-offline", "truth")):
        answers, extra = controls.life(arm, scenes[:8], scenes[8:], 0, lessons=lessons,
                                       joints=False, parameters=5000, rows=4)
        assert len(answers) == 4 and answers[0].shape == (30, 1, 2)
        assert extra["macs"] > 0 and extra["milliseconds"] > 0
