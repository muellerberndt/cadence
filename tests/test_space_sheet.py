"""cadence.space.sheet: a belief over places held in a sheet's own settled activity."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.space import Interface, Space
from cadence.space.sheet import OFFSETS, PASS, Sheet

PLANE = (-1.0, 1.0, -0.2, 1.4)
SMALL = dict(place=8, moved=5, motion=9, counted=3, unseen=4, heard=(6, 4), felt=4, hidden=16,
             shift=12, spread=6, found=8, keep=4)


def test_a_peak_at_rest_is_held_for_a_thousand_moments():
    sheet = Sheet(PLANE)
    point = np.array([[0.313, 0.777]])
    sheet.begin(1)
    sheet.write([True], point, [False])
    assert np.linalg.norm(sheet.where() - point) < 1e-6      # weights on four cells: the place
    peak = float(np.tanh(sheet.potential / 2.0).max())
    for _ in range(1000):
        sheet.step([True], np.zeros((1, 2)))
    assert np.linalg.norm(sheet.where() - point) < 0.0005    # half a millimetre in 1,000 moments
    assert abs(float(np.tanh(sheet.potential / 2.0).max()) - peak) < 0.01 * peak
    assert np.linalg.norm(sheet.potential) == pytest.approx(sheet.level, rel=1e-6)
    assert sheet.area()[0].sum() <= 4 and sheet.holds(point)[0]


def test_the_settle_is_at_its_fixed_point_after_one_sweep():
    sheet = Sheet(PLANE, cells=(12, 10))
    sheet.begin(1)
    sheet.write([True], [[0.1, 0.5]], [True])
    drive = np.zeros((1, sheet.neurons))
    drive[0, sheet._went[5]] = sheet.potential[0, 0]
    one = sheet._settle(drive, None, sweeps=1)
    assert np.array_equal(one, sheet._settle(drive, None, sweeps=4))
    assert np.linalg.norm(one) == pytest.approx(sheet.level, rel=1e-6)   # the pool's gain


def test_the_kernels_move_mass_along_the_seen_movement():
    sheet = Sheet(PLANE)
    start, movement = np.array([[-0.5, 0.2]]), np.array([[0.012, -0.004]])
    sheet.begin(1)
    sheet.write([True], start, [True])
    for _ in range(40):
        sheet.step([True], movement)
    assert np.linalg.norm(sheet.where() - (start + 40 * movement)) < 1e-9
    assert sheet.mass()[0, 0].sum() == pytest.approx(1.0)        # all of it on the moving sheet
    assert np.abs(sheet.remainder).max() <= 0.5                  # under half a cell is owed


def test_cells_in_sight_with_no_report_are_emptied():
    sheet = Sheet(PLANE)
    sheet.kernel[81:90] = 0.1 * PASS      # a tenth of the moving mass comes to rest each moment
    sheet.kernel[:81] *= 0.9
    sheet._build()
    sheet.begin(1)
    sheet.write([True], [[-0.5, 0.6]], [True])
    for _ in range(12):
        sheet.step([True], [[0.03, 0.0]])
    mass = sheet.mass()[0]
    assert 0.2 < mass[1].sum() < 0.8                             # a trail of mass at rest
    lit = sheet.centers[:, 0] > -0.2                             # the camera sees the far end
    assert (mass.sum(axis=0)[lit]).sum() > 0.2
    sheet.step([True], [[0.03, 0.0]], lit)
    after = sheet.mass()[0].sum(axis=0)
    assert after[lit].sum() == 0.0 and after.sum() == pytest.approx(1.0)
    assert sheet.where()[0, 0] < -0.2 and not sheet.area()[0].ravel()[lit].any()


def test_a_thing_seen_empty_where_it_was_held_is_looked_for_out_of_sight():
    sheet = Sheet(PLANE)
    sheet.begin(1)
    sheet.write([True], [[0.0, 0.6]], [False])                   # at rest, in the light
    lit = sheet.centers[:, 0] > -0.1                             # a shadow begins 10 cm away
    sheet.step([True], np.zeros((1, 2)), lit)
    place = sheet.where()[0]
    assert not sheet.gone[0] and -0.5 < place[0] < -0.1 and abs(place[1] - 0.6) < 0.05
    sheet.step([True], np.zeros((1, 2)), np.ones(sheet.count, dtype=bool))
    assert sheet.gone[0] and np.allclose(sheet.where()[0], place)   # no cell is out of sight


def test_lessons_teach_the_share_that_goes_on_and_the_share_that_stops():
    sheet = Sheet(PLANE, cells=(12, 10), born="even", rate=0.2)
    rng = np.random.default_rng(0)
    rows = 48
    every = np.ones(rows, dtype=bool)
    first = None
    for _ in range(150):
        sheet.begin(rows)
        place = np.stack([rng.uniform(-0.6, 0.6, rows), rng.uniform(0.2, 1.0, rows)], axis=1)
        sheet.write(every, place, every)
        movement = np.tile([0.17, 0.0], (rows, 1))               # one cell a moment along x
        sheet.step(every, movement)
        kept = rng.random(rows) > 0.2
        error = sheet.teach(sheet.taken(), every, np.where(kept[:, None], place + movement, place),
                            kept)
        first = error if first is None else first
    shares = sheet.shares()
    shift = OFFSETS.index((1, 0))
    assert error < 0.6 * first
    assert shares["keep"][shift] == pytest.approx(0.8, abs=0.1)
    assert shares["stop"][shift] == pytest.approx(0.2, abs=0.1)
    assert sheet.kernel[shift * 9 : shift * 9 + 9].argmax() == shift   # one cell along the shift
    assert sheet.kernel[90] == PASS and (sheet.kernel >= 0).all()


def test_a_space_beside_a_sheet_holds_a_thing_through_a_shadow():
    space = Space(Interface(plane=PLANE, **SMALL), seed=0, backend="cpu", sheet={"cells": (24, 20)})
    rng = np.random.default_rng(1)
    start, velocity = np.array([[-0.6, 0.2], [0.5, 0.9]]), np.array([[0.012, 0.004], [0.0, 0.0]])
    space.begin(start)
    for t in range(1, 25):
        space.advance(np.ones(2, dtype=bool), start + t * velocity + rng.normal(0, 0.005, (2, 2)))

    def in_sight(points):
        return ~((points[:, 0] > -0.4) & (points[:, 0] < 0.7))   # a band the camera does not see

    nothing = np.zeros(2, dtype=bool)
    for _t in range(25, 60):
        space.advance(nothing, np.zeros((2, 2)), in_sight=in_sight)
    where, radius = space.where()
    assert np.linalg.norm(where[0] - (start[0] + 59 * velocity[0])) < 0.05
    assert np.linalg.norm(where[1] - start[1]) < 0.01            # the thing at rest stayed
    assert space.work["moments"] == 0                            # the place brain was not asked
    area, state = space.area(), space.state()
    assert area.shape == (2, 24, 20) and area[0].sum() >= 1 and radius[0] > 0
    assert state["moving"].shape == (2, 24, 20) and state["moving"][0].sum() > 0.99
    assert state["rest"][1].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="flags on the sheet"):
        space.advance(nothing, np.zeros((2, 2)), in_sight=np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match="read from a sheet"):
        Space(Interface(plane=PLANE, **SMALL), seed=0, backend="cpu").area()


def test_a_space_without_a_sheet_answers_as_before_and_a_module_file_keeps_the_kernels(tmp_path):
    plain = Space(Interface(plane=PLANE, **SMALL), seed=3, backend="cpu")
    beside = Space(Interface(plane=PLANE, **SMALL), seed=3, backend="cpu", sheet={"cells": (12, 10)})
    assert plain.sheet is None
    assert np.array_equal(plain.brain.brain.efficacy, beside.brain.brain.efficacy)
    rng = np.random.default_rng(2)
    start = np.array([[-0.5, 0.3]])
    beside.begin(start)
    for t in range(1, 40):                                        # truth lessons in a nursery
        truth = start + min(t, 25) * np.array([[0.02, 0.0]])
        beside.advance([t < 12], truth + rng.normal(0, 0.005, (1, 2)), learn=True, truth=truth)
    assert beside.sheet.lessons > 10
    path = beside.save(tmp_path / "place.npz", card={"lessons": "truth"})
    module = cd.load_module(path)
    assert module.record["beside"] == ["sheet/kernel"]
    again = Space.load(module)
    assert np.allclose(again.sheet.kernel, beside.sheet.kernel, atol=1e-6)
    assert again.sheet.cells == (12, 10) and again.sheet.lessons == beside.sheet.lessons
    assert np.array_equal(again.brain.brain.efficacy.astype(np.float32),
                          beside.brain.brain.efficacy.astype(np.float32))
    assert Space.load(module, sheet=None).sheet is None          # the regions alone
