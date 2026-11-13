"""The retina chamber's contracts: what a thing looks like, the eyes' images, rooms clear from
the perch, a short life of both arms and the controls' images (cadence#330)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import retina  # noqa: E402
import world  # noqa: E402

SHORT = {"naming_rooms": 10, "namings": 15, "practice_rooms": 6, "asked": 10, "test_rooms": 2,
         "unseen_rooms": 2, "new_rooms": 2}


def test_every_kind_is_one_outline_in_one_size() -> None:
    assert sorted(retina.LOOKS) == sorted((o, s) for o in range(4) for s in range(3))
    for kind in range(world.KINDS):
        shape = retina.outline(kind, np.array([1.0, 0.0]), 0.0)
        across = shape[:, 1].max() - shape[:, 1].min()
        assert across == pytest.approx(retina.SIZES[retina.LOOKS[kind][1]], rel=0.02)
        if retina.OUTLINES[retina.LOOKS[kind][0]] == "dish":
            assert shape[:, 0].min() >= 1.0 - 1e-9  # open toward the room's middle
        else:
            assert shape[:, 0].min() < 1.0  # its face toward the room's middle


def test_the_eyes_see_the_things_nearer_than_the_walls() -> None:
    from cadence.words.retina import parrot_eyes

    eyes = parrot_eyes()
    r = world.Room([world.Thing(2, 0, np.array([1.0, 0.0]))], heading=math.radians(-70.0),
                   extra={"turns": {0: 0.0}})
    images = retina.render(0, r, eyes, np.zeros(2), r.heading)
    assert images.shape == (2, eyes[0].receptors, 4)
    depth = images[0, :, 3]  # the left eye looks along the x axis
    middle = eyes[0].receptors // 2
    face = 1.0 - 0.015 - eyes[0].origin(np.zeros(2), r.heading)[0]  # the board's face
    assert depth[middle] == pytest.approx(face, abs=0.005)
    assert depth.max() > 1.99 and (images[..., :3] >= 0).all() and (images[..., :3] <= 1).all()
    assert images[0, middle, 0] > 0.8 > images[0, middle, 1]  # red, lit squarely


def test_rooms_are_clear_from_the_perch_and_hold_out_a_band() -> None:
    events = retina.script(5, **SHORT)
    rooms = [e[1] for e in events if e[0] == "room"]
    assert rooms and all(retina.clear(5, r) for r in rooms)
    assert not any(retina._held_out(t.place) for r in rooms for t in r.things)
    assert world.VIEW == math.radians(60.0)  # the words chamber's own camera is left as it was
    asked = [e for e in events if e[0] == "ask"]
    assert {e[1] for e in asked} >= {"W1_practised", "W2_where", "W3_colour", "W5_by_place"}


def test_a_short_life() -> None:
    record: list[dict] = []
    out = retina.life(3, record=record, views=6, **SHORT)
    assert out["R1"]["found"] >= 0.95 and out["R1"]["spurious"] <= 0.05
    assert out["R2"]["in_view"] >= 0.9
    assert out["camera"]["W1"] >= 0.9 and out["eyes"]["W4_never_taught"] == 1.0
    assert out["eyes"]["W6_ablated"] == 0.0 and out["eyes"]["parameters"] > 256 * 26
    assert out["eyes"]["macs"] > 0 and out["R3"]["n"] == 6
    assert {r["arm"] for r in record} == {"eyes", "camera"}


def test_the_controls_see_the_same_images() -> None:
    pytest.importorskip("torch")
    import retina_controls

    controls = retina_controls.controls
    images, items = retina_controls.episodes(5, **SHORT)
    events = retina.script(5, **SHORT)
    assert sum(k == "ask" for k, _, _ in items) == sum(e[0] == "ask" for e in events)
    moments = sum(9 if e[0] == "room" else 1 for e in events if e[0] in ("room", "turn"))
    assert images.shape[0] == moments and images.shape[1:] == (2, 321, 4)
    tok = retina_controls.Tokens(world.lexicon(5))
    train = [first for kind, first, _ in items if kind == "train"]
    for model in ("cnn", "vit"):
        width = retina_controls.matched_width(model, 15000, len(tok), images.shape[2])
        control = retina_controls.Control(model, width, len(tok), images, 0, 0.003)
        assert abs(controls.count(control.net) - 15000) < 1500
        assert np.isfinite(control.update(train[:4]))
