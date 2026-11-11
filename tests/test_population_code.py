"""The population code: tuned inputs that tile a reading's range and stay quiet away from it."""

import numpy as np
import pytest

from cadence import PopulationCode


def test_a_reading_lights_the_inputs_nearest_its_value():
    code = PopulationCode([(0.0, 1.0)], per=11)
    row = code([0.3])[0]
    assert code.inputs == 11 and row.shape == (11,)
    assert np.argmax(row) == 3 and row[3] == pytest.approx(1.0)
    assert row[2] == pytest.approx(np.exp(-0.5)) and row[4] == pytest.approx(np.exp(-0.5))
    assert row[1] == pytest.approx(np.exp(-2.0)) and row[5] == pytest.approx(np.exp(-2.0))
    assert (row[[0, 6, 7, 8, 9, 10]] == 0.0).all()  # three spacings away: under the floor, silent


def test_readings_are_coded_side_by_side_and_rows_are_kept():
    code = PopulationCode([(-1.0, 1.0), (0.0, 10.0)], per=8)
    rows = code(np.array([[0.0, 5.0], [-1.0, 10.0]]))
    assert rows.shape == (2, 16)
    assert np.argmax(rows[1, :8]) == 0 and np.argmax(rows[1, 8:]) == 7
    assert np.array_equal(code([0.0, 5.0]), rows[:1])  # one row in, one row out


def test_values_outside_the_range_are_read_at_the_nearer_end():
    code = PopulationCode([(0.0, 1.0)], per=5)
    assert np.array_equal(code([3.0]), code([1.0])) and np.array_equal(code([-7.0]), code([0.0]))
    assert code([3.0])[0, 4] == pytest.approx(1.0)  # far past the range, never dark


def test_unclamped_values_outside_the_range_fade_at_the_nearer_end():
    code = PopulationCode([(0.0, 1.0)], per=5, clamp=False)
    near, far = code([1.1])[0], code([3.0])[0]
    assert np.argmax(near) == 4 and 0.0 < near[4] < 1.0
    assert not far.any()
    dark = np.sqrt(-2 * np.log(0.3))  # the spacings past the end where a floor of 0.3 goes dark
    floor = PopulationCode([(0.0, 1.0)], per=5, floor=0.3, clamp=False)
    assert floor([1.0 + 0.25 * (dark - 0.01)]).any() and not floor([1.0 + 0.25 * (dark + 0.01)]).any()


def test_the_floor_and_the_tuning_width():
    dense = PopulationCode([(0.0, 1.0)], per=5, floor=0.0)
    assert (dense([0.5]) > 0).all()  # every input answers a little without a floor


@pytest.mark.parametrize("kwargs", [dict(ranges=[(1.0, 0.0)]), dict(ranges=[]), dict(ranges=[(0.0, np.inf)]),
                                    dict(ranges=[(0.0, 1.0)], per=1), dict(ranges=[(0.0, 1.0)], per=True),
                                    dict(ranges=[(0.0, 1.0)], floor=1.0),
                                    dict(ranges=[(0.0, 1.0)], clamp=1)])
def test_invalid_codes_are_refused(kwargs):
    with pytest.raises(ValueError):
        PopulationCode(**kwargs)


def test_invalid_readings_are_refused():
    code = PopulationCode([(0.0, 1.0), (0.0, 1.0)])
    with pytest.raises(ValueError):
        code([0.5])
    with pytest.raises(ValueError):
        code([0.5, np.nan])
