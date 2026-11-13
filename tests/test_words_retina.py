"""Two retinas: the eyes' geometry, things found in an image, their places, the categories
learned from what is seen, and the map's reports (``cadence.words.retina``, #330)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadence.words.retina import Categories, Eye, Retina, parrot_eyes


def _image(eye: Eye, heading: float, things: list[tuple[float, float, float, tuple]],
           wall: float = 3.0) -> np.ndarray:
    """One eye's image of round things ``(x, y, radius, rgb)`` before a round wall ``wall``
    metres from the eye at the plane's middle, each receptor the mean of its rays."""
    origin = eye.origin(np.zeros(2), heading)
    angles = eye.facing(heading) + eye.rays
    u = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    depth = np.full(angles.shape, wall)
    rgb = np.broadcast_to(np.array([0.5, 0.5, 0.5]), angles.shape + (3,)).copy()
    for x, y, radius, colour in things:
        w = np.array([x, y]) - origin
        along = u @ w
        miss = np.sum(w * w) - along ** 2
        inside = (miss <= radius ** 2) & (along > 0)
        t = along - np.sqrt(np.maximum(radius ** 2 - miss, 0.0))
        hit = inside & (t < depth)
        depth = np.where(hit, t, depth)
        rgb = np.where(hit[..., None], np.array(colour), rgb)
    return np.concatenate([rgb.mean(axis=1), depth.mean(axis=1)[:, None]], axis=1)


def test_an_eye_is_sharp_at_its_middle_and_coarse_at_its_edges() -> None:
    eye = Eye(receptors=101, field=math.radians(80.0), fovea=0.4)
    assert eye.edges[0] == pytest.approx(-math.radians(80.0))
    assert eye.edges[-1] == pytest.approx(math.radians(80.0))
    assert np.all(np.diff(eye.edges) > 0) and len(eye.centers) == 101
    assert eye.widths[50] < eye.widths[0] and eye.widths[0] / eye.widths[50] == pytest.approx(
        (2 - 0.4) / 0.4, rel=0.05)
    with pytest.raises(ValueError):
        Eye(receptors=100)


def test_a_parrots_eyes_look_to_its_sides() -> None:
    left, right = parrot_eyes()
    assert left.facing(0.0) == pytest.approx(math.radians(70.0))
    assert right.facing(0.0) == pytest.approx(-math.radians(70.0))
    # 20 degrees seen by both eyes in front of the beak, 60 behind seen by neither
    assert math.degrees(left.axis - left.field) == pytest.approx(-10.0)
    assert math.degrees(left.axis + left.field) == pytest.approx(150.0)
    assert left.origin(np.zeros(2), 0.0)[1] > 0 > right.origin(np.zeros(2), 0.0)[1]


def test_a_thing_is_found_with_its_place_and_width() -> None:
    retina = Retina(eyes=(Eye(receptors=201, field=math.radians(80.0)),))
    image = _image(retina.eyes[0], 0.0, [(1.2, 0.1, 0.1, (0.9, 0.1, 0.1))])
    figures = retina.find(image[None], np.zeros(2), 0.0)
    assert len(figures) == 1
    f = figures[0]
    assert f.whole and f.full >= retina.least
    assert f.width == pytest.approx(0.2, abs=0.02)
    assert np.linalg.norm(f.place - np.array([1.2, 0.1])) < 0.03  # a disc's middle
    assert f.bend == pytest.approx(2.0, abs=0.5)  # it bulges toward the eye
    assert f.rgb[0] > 0.8 and f.rgb[1] < 0.2


def test_a_thing_partly_behind_another_is_found_in_part() -> None:
    retina = Retina(eyes=(Eye(receptors=201, field=math.radians(80.0)),))
    image = _image(retina.eyes[0], 0.0, [(0.8, 0.0, 0.08, (0.1, 0.1, 0.9)),
                                         (1.6, 0.14, 0.1, (0.9, 0.9, 0.1))])
    figures = retina.find(image[None], np.zeros(2), 0.0)
    near = min(figures, key=lambda f: float(np.linalg.norm(f.place - [0.8, 0.0])))
    far = min(figures, key=lambda f: float(np.linalg.norm(f.place - [1.6, 0.14])))
    assert near.whole and not far.whole and far.form is None


def test_the_eyes_see_up_to_where_their_rays_reach() -> None:
    retina = Retina(eyes=(Eye(receptors=201, field=math.radians(80.0)),))
    image = _image(retina.eyes[0], 0.0, [(1.0, 0.0, 0.1, (0.9, 0.1, 0.1))])
    lit = retina.lit(image[None], np.zeros(2), 0.0)
    seen = lit(np.array([[0.5, 0.0], [1.2, 0.0], [1.0, 0.8], [-1.0, 0.0]]))
    # in front of the thing, behind it, beside it in the field, outside the field
    assert seen.tolist() == [True, False, True, False]


def test_categories_enter_for_new_sorts_and_keep_what_they_were_seen_as() -> None:
    a, b = np.zeros(12), np.zeros(12)
    a[:3], b[6:9] = 1 / math.sqrt(3), 1 / math.sqrt(3)
    near_a = a + 0.05 * np.eye(12)[3]
    cats = Categories(12, 4, vigilance=0.3)
    assert cats.see(a) == 0 and cats.used == 1
    assert cats.see(b) == 1 and cats.used == 2
    assert cats.see(near_a / np.linalg.norm(near_a)) == 0
    active = cats.settle(np.stack([a, b]))
    assert active.shape == (2, 4) and active[0, 0] > active[0, 1] and active[1, 1] > active[1, 0]
    # a code no category knows: no category without entering, a new one with
    c = np.zeros(12)
    c[10:] = 1 / math.sqrt(2)
    assert cats.see(c, learn=False) is None and cats.used == 2
    assert cats.see(c) == 2 and cats.counts[2] == 1
    assert cats.parameters() == 12 * 4


def test_the_retina_reports_what_it_recognised() -> None:
    retina = Retina(eyes=(Eye(receptors=201, field=math.radians(80.0)),), kinds=4, colours=4)
    red_disc = _image(retina.eyes[0], 0.0, [(1.2, 0.1, 0.1, (0.9, 0.1, 0.1))])
    reports, lit = retina.see(red_disc[None], np.zeros(2), 0.0)
    assert len(reports) == 1 and reports[0][0] == 0 and reports[0][2] == {"colour": 0}
    blue_small = _image(retina.eyes[0], 0.0, [(0.9, -0.05, 0.05, (0.1, 0.2, 0.9))])
    reports, _ = retina.see(blue_small[None], np.zeros(2), 0.0)
    assert len(reports) == 1 and reports[0][0] == 1 and reports[0][2] == {"colour": 1}
    # the same red disc elsewhere is the first kind and colour again
    again = _image(retina.eyes[0], 0.0, [(1.0, -0.3, 0.1, (0.9, 0.1, 0.1))])
    reports, _ = retina.see(again[None], np.zeros(2), 0.0)
    assert [(k, v["colour"]) for k, _, v in reports] == [(0, 0)]
    assert retina.kinds.used == 2 and retina.colours.used == 2
    assert callable(lit) and retina.parameters() == (
        retina.kinds.parameters() + retina.colours.parameters())


def test_the_head_turns_to_bring_a_place_to_an_eyes_middle() -> None:
    retina = Retina()
    place = np.array([0.0, 1.5])
    heading = retina.look_at(place, np.zeros(2), 0.0)
    eye = retina.eyes[0]  # the left eye sees it nearer its middle
    assert math.remainder(eye.facing(heading) - math.pi / 2, 2 * math.pi) == pytest.approx(0.0)
