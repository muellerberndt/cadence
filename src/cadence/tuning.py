"""A population code for continuous readings: tuned inputs, quiet away from their value."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class PopulationCode:
    """Spread each continuous reading over ``per`` inputs with Gaussian tuning curves.

    The preferred values of a reading's inputs tile its range ``(low, high)`` evenly; each input
    reads ``exp(-0.5 * ((value - preferred) / spacing) ** 2)``, where ``spacing`` is the distance
    between neighbouring preferred values, and a response under ``floor`` reads zero. With the
    default floor of 0.05 a reading lights the inputs within two spacings of its value, five at
    most (two spacings away reads exp(-2), about 0.14), and leaves the rest at zero, so a brain's
    drive comes from where the reading is rather than from its size.

    A value outside the range is read at the nearer end (``clamp=True``, the default), so a
    reading never goes dark: the rare states past a range are often the ones that matter, such as
    a robot righting itself when the ranges came from percentiles of its upright gait. With
    ``clamp=False`` a value outside the range lights the input at the nearer end less brightly,
    and nothing once it lies ``sqrt(-2 ln floor)`` spacings past the end (2.45 at the default
    floor, 1.55 at a floor of 0.3).

    The code is fixed and stateless: it learns nothing and a brain does not save it. Build it
    once, with the ranges the readings actually cover, and give the brain ``code.inputs`` inputs.
    """

    def __init__(
        self,
        ranges: Sequence[tuple[float, float]],
        per: int = 16,
        floor: float = 0.05,
        clamp: bool = True,
    ):
        if isinstance(per, bool) or not isinstance(per, int) or per < 2:
            raise ValueError("per must be an integer of at least 2 tuned inputs a reading")
        if not 0.0 <= float(floor) < 1.0:
            raise ValueError("floor must be in [0, 1)")
        if not isinstance(clamp, bool):
            raise ValueError("clamp must be True or False")
        bounds = np.asarray(ranges, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) == 0:
            raise ValueError("ranges must be a nonempty sequence of (low, high) pairs")
        if not np.isfinite(bounds).all() or (bounds[:, 1] <= bounds[:, 0]).any():
            raise ValueError("every range must be finite with low below high")
        self.ranges = [(float(lo), float(hi)) for lo, hi in bounds]
        self.per, self.floor, self.clamp = per, float(floor), clamp
        self.low, self.high = bounds[:, 0].copy(), bounds[:, 1].copy()
        self.preferred = np.linspace(bounds[:, 0], bounds[:, 1], per, axis=1)  # (readings, per)
        self.spacing = (bounds[:, 1] - bounds[:, 0]) / (per - 1)  # (readings,)
        self.inputs = per * len(bounds)

    def __call__(self, readings: object) -> np.ndarray:
        """Code ``(readings,)`` or ``(rows, readings)`` values as ``(rows, inputs)`` rows."""
        values = np.asarray(readings, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.ranges):
            raise ValueError(
                f"expected {len(self.ranges)} readings per row, got shape {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("readings must be finite")
        if self.clamp:
            values = np.clip(values, self.low, self.high)
        distance = (values[:, :, None] - self.preferred[None]) / self.spacing[None, :, None]
        response = np.exp(-0.5 * distance**2)
        response[response < self.floor] = 0.0
        return response.reshape(len(values), self.inputs)
