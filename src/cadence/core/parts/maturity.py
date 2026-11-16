"""``cadence.plasticity.Maturity`` in the core: the scale and the recovery of a brain's maturity and
the use of its plasticity; ``factor`` reads the step factor, ``relax`` lets one moment pass."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...plasticity import Maturity

CLASS = Maturity


def state_of(held: Maturity) -> dict[str, Any]:
    return {"scale": np.float64(held.scale), "recovery": np.float64(held.recovery),
            "use": np.float64(held.use)}


def restore(held: Maturity, state: dict[str, Any]) -> None:
    held.scale, held.recovery = float(state["scale"]), float(state["recovery"])
    held.use = float(state["use"])


def _none() -> dict[str, Any]:
    return {}


METHODS: dict[str, Any] = {
    "factor": (_none, lambda value: {"factor": np.float64(value)}),
    "relax": (_none, lambda value: {}),
}
