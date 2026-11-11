"""The pixel chamber (cadence#284): the rendered task, its panels and controls, the
convolutional baseline and a short life of each kind."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str):
    key = f"pixels_{name}"  # a name of this chamber's own: other benchmarks have a chamber.py
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


ch = _load("chamber")
px = _load("pixels")


def test_the_rule_approaches_food_avoids_the_hazard_and_aims_at_the_target():
    assert ch.label_of("food", "left") == ch.LEFT and ch.label_of("food", "right") == ch.RIGHT
    assert ch.label_of("hazard", "left") == ch.RIGHT and ch.label_of("hazard", "right") == ch.LEFT
    assert ch.label_of("target", "left") == ch.LEFT and ch.label_of("target", "centre") == ch.FIRE
    with pytest.raises(ValueError):
        ch.label_of("food", "centre")
    labels = [ch.label_of(*s) for s in ch.SITUATIONS]
    assert labels.count(ch.LEFT) == 3 and labels.count(ch.RIGHT) == 3 and labels.count(ch.FIRE) == 2


@pytest.mark.parametrize("name", list(ch.SHAPES))
def test_frames_have_the_shape_and_range_and_show_one_object(name):
    shape = ch.SHAPES[name]
    frames, labels = ch.panel(0, "held_out", shape, 16)
    assert frames.shape == (16, *shape) and labels.shape == (16,)
    assert frames.min() >= 0.0 and frames.max() <= 1.0
    scene = ch.draw_scene(np.random.default_rng(3), ch.TRAINED, shape[1] / shape[0])
    clean = ch.render(scene, shape, None)
    lit = np.abs(clean - scene.background).max(axis=2) > 1e-9
    assert lit.any()
    blank = ch.render(replace(scene, blank=True), shape, None)
    assert np.allclose(blank, scene.background)


def test_the_panels_draw_each_factor_from_its_held_out_range_alone():
    rng = np.random.default_rng(0)
    for name, conditions in ch.PANEL_CONDITIONS.items():
        scenes = [ch.draw_scene(rng, conditions) for _ in range(200)]
        vs = np.array([s.item.v for s in scenes])
        scales = np.array([s.item.scale for s in scenes])
        held_v = (vs >= 0.45) & (vs <= 0.55)
        held_scale = (scales >= 0.95) & (scales <= 1.05)
        assert held_v.all() if name == "positions" else not held_v.any()
        assert held_scale.all() if name == "scales" else not held_scale.any()
        kinds = {d.kind for s in scenes for d in s.distractors}
        if name == "distractor":
            assert kinds == {"triangle"} and all(len(s.distractors) == 1 for s in scenes)
        else:
            assert kinds <= {"bar", "square"}
        backgrounds = np.array([s.background for s in scenes])
        assert (backgrounds >= 0.35).all() if name == "lighting" else (backgrounds <= 0.25).all()


def test_distractors_never_overlap_the_object():
    rng = np.random.default_rng(1)
    for _ in range(200):
        s = ch.draw_scene(rng, ch.PANEL_CONDITIONS["distractor"])
        for d in s.distractors:
            gap = np.hypot(d.u - s.item.u, d.v - s.item.v)
            assert gap > ch.RADIUS * (s.item.scale + ch.DISTRACTOR_RADIUS * d.scale)


def test_panels_and_the_schedule_repeat_per_founder_and_differ_between_founders():
    a, la = ch.panel(2, "scales", ch.SHAPES["grey84"], 8)
    b, lb = ch.panel(2, "scales", ch.SHAPES["grey84"], 8)
    c, _ = ch.panel(3, "scales", ch.SHAPES["grey84"], 8)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(la, lb)
    assert not np.array_equal(a, c)
    first = next(ch.Schedule(4, ch.SHAPES["grey84"]).batches(1))
    again = next(ch.Schedule(4, ch.SHAPES["grey84"]).batches(1))
    np.testing.assert_array_equal(first[0], again[0])


def test_the_controls_blank_the_screen_or_shuffle_the_labels():
    shape = ch.SHAPES["grey84"]
    frames, labels = next(ch.Schedule(5, shape, rows=16, control="blank").batches(1))
    assert np.ptp(frames.reshape(16, -1).mean(axis=1)) < 0.25  # the background alone
    assert np.all(np.std(frames.reshape(16, -1), axis=1) < 0.05)
    plain = next(ch.Schedule(5, shape, rows=16).batches(1))
    shuffled = next(ch.Schedule(5, shape, rows=16, control="shuffled").batches(1))
    np.testing.assert_array_equal(plain[0], shuffled[0])
    many = ch.Schedule(5, shape, rows=16, control="shuffled")
    plain_many = ch.Schedule(5, shape, rows=16)
    pairs = zip(many.batches(40), plain_many.batches(40), strict=True)
    agree = [np.mean(a[1] == b[1]) for a, b in pairs]
    assert abs(np.mean(agree) - (2 * 0.375**2 + 0.25**2)) < 0.05  # independent of the frames
    with pytest.raises(ValueError):
        ch.Schedule(0, shape, control="dark")


def test_the_coarse_code_and_unit_rows():
    batch = np.random.default_rng(0).random((3, 84, 84, 1))
    assert ch.pooled(batch, 4).shape == (3, 441)
    assert np.allclose(np.linalg.norm(ch.unit(batch), axis=1), 1.0)
    assert ch.summary({"held_out": 0.9, "positions": 0.86, "scales": 0.9, "distractor": 0.84})[
        "invariant"] is False


def test_the_probe_reads_a_separable_representation():
    rng = np.random.default_rng(0)
    labels = rng.integers(3, size=200)
    feats = np.eye(3)[labels] + 0.05 * rng.normal(size=(200, 3))
    out = px.ridge_probe(feats, labels, {"held_out": feats[:50]}, {"held_out": labels[:50]})
    assert out["panels"]["held_out"] == 1.0


def test_the_baseline_counts_its_parameters_and_learns_a_lesson():
    pytest.importorskip("torch")
    cn = _load("convnet")
    shape = ch.SHAPES["grey84"]
    net = cn.ConvActorCritic(shape, ch.ACTIONS, 32, seed=0)
    assert net.parameters() == cn.parameters(shape, ch.ACTIONS, 32)
    assert cn.macs(shape, ch.ACTIONS, 256) > cn.macs(shape, ch.ACTIONS, 32)
    assert abs(cn.parameters(shape, 3, cn.hidden_for(shape, 3, 400_000)) - 400_000) < 3000
    frames, labels = ch.panel(0, "held_out", shape, 16)
    before = net.logits(frames).detach().numpy().copy()
    net.lesson(frames, labels)
    assert not np.allclose(before, net.logits(frames).detach().numpy())
    actions = net.act(frames)
    net.reward(frames, actions, (actions == labels).astype(float))
    assert net.frames == 32


def test_the_tied_arm_shares_the_pool_rows_and_every_life_counts_its_work():
    pytest.importorskip("torch")
    cn = _load("convnet")
    shape = ch.SHAPES["grey84"]
    plain = px.compose("fields", shape, 0, "lesson")
    tied = px.compose("fields_tied", shape, 0, "lesson")
    rows = 3 * 64  # one pool row: three pool columns of 64 fields
    assert plain.connectome.synapses == tied.connectome.synapses
    assert plain.parameters() - tied.parameters() == 2 * rows * 64
    assert tied.describe()["layout"]["tied"][0]["blocks"] == [3, 1]
    assert cn.hidden_for(shape, ch.ACTIONS, tied.parameters(), 1) == 5
    spec = {"founder": 0, "condition": "lesson", "budgets": [16], "rows": 8, "panel": 8}
    out = px.life({"arm": "fields_tied", **spec})
    assert out["parameters"] == tied.parameters()
    assert set(out["readings"][-1]["panels"]) == set(ch.PANELS)
    fields = tied.fields.macs()
    assert out["macs"]["lesson"] > fields["code"] + fields["learn"]
    assert fields["code"] < out["macs"]["read"] < out["macs"]["lesson"]
    assert out["macs"]["lessons"] == 16 * out["macs"]["lesson"]
    control = px.life({"arm": "conv_matched_tied", **spec})
    forward = cn.macs(shape, ch.ACTIONS, 5)
    assert control["macs"] == {"lesson": 3.0 * forward, "read": float(forward),
                               "lessons": 48.0 * forward}
    assert abs(control["parameters"] - out["parameters"]) < 1300
    assert px.life({"arm": "pooled", **spec, "condition": "reward"})["macs"] is None


@pytest.mark.parametrize("arm,condition", [("pooled", "lesson"), ("pooled", "reward")])
def test_a_short_life_reads_every_panel(arm, condition):
    out = px.life({"arm": arm, "founder": 0, "condition": condition, "budgets": [16],
                   "rows": 8, "panel": 8})
    reading = out["readings"][-1]
    assert reading["frames"] == 16 and set(reading["panels"]) == set(ch.PANELS)
    assert out["parameters"] > 0
