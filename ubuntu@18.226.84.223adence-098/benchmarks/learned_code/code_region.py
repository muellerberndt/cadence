"""A learned code region: a sparse code settled by local competition against two neighbours.

The region holds ``units`` code units. At each moment they settle by the locally competitive
algorithm (Rozell, Johnson, Baraniuk and Olshausen, Neural Computation 2008): each unit
integrates its drive with a leak, publishes a thresholded output, and inhibits the others
through the Gram matrix of the dictionary. The settled state ``a >= 0`` minimizes

    E(a) = 1/2 |x - Phi a|^2  +  top_down/2 |a - p|^2  +  sum_i threshold_i a_i

over the two boundaries of the region:

* bottom-up, the observation ``x``: ``Phi a`` is the code's reconstruction of it;
* top-down, the context: ``p = W [c, 1]`` is a prediction of the code from ``c``, the settled
  activity of the context region of the moment before (unit length), through a predictor
  ``W`` learned by a delta rule.

The settle is the leaky integration

    u <- u + step * (Phi^T x + top_down * p - u - (Gram - I) a - top_down * a),
    a  = max(u - threshold, 0),

whose fixed points are the minimizers of ``E``: an active unit's threshold equals its share of
the bottom-up residual, ``Phi_i . (x - Phi a)``, plus ``top_down * (p_i - a_i)``, and a silent
unit's is at least that. It stops when the largest drive imbalance falls below ``tolerance``
or after ``budget`` sweeps; every settle and every sweep is counted.

Learning is local to each boundary, from its own residual:

* the dictionary, Hebbian on the bottom-up residual, ``Phi += rate * (x - Phi a) a^T`` (mean
  over the rows), then each unit's column scaled back to unit length (a unit's own synaptic
  scaling);
* the predictor, the delta rule on the top-down residual, ``W += r * (a - p) z^T`` (mean over
  the rows) with ``z = [c, 1]`` and ``r = min(predictor_rate, 1 / |z|^2)``, the row's rate
  bounded by its own input energy;
* the thresholds, homeostatic: each moves by ``threshold_rate`` times the share of rows in
  which its unit was active less the target share, floored at zero, so a unit used too often
  becomes harder to drive and an unused one easier.

While the code fits, the residuals are small and nothing moves. With ``top_down`` zero the
region is plain sparse coding of the observation, and its predictor still learns (it is read
by nothing in the settle). NumPy only; nothing here changes the library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# the code's own random stream of a founder's seed
CODE_STREAM = 21_101


@dataclass(frozen=True)
class CodeSettings:
    """The region's construction and its rates; ``top_down`` 0 is the region without its
    top-down boundary."""

    units: int = 64
    top_down: float = 1.0
    threshold: float = 0.1
    step: float = 0.1
    budget: int = 400
    tolerance: float = 1e-4
    dictionary_rate: float = 0.02
    predictor_rate: float = 0.05
    threshold_rate: float = 0.01
    target: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.units, int) or self.units < 1:
            raise ValueError("units must be a positive integer")
        if not isinstance(self.budget, int) or self.budget < 1:
            raise ValueError("budget must be a positive integer")
        for name in ("top_down", "threshold", "dictionary_rate", "predictor_rate",
                     "threshold_rate"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative real")
        if not 0 < self.step <= 1:
            raise ValueError("step lies in (0, 1]")
        if not 0 < self.target < 1:
            raise ValueError("target lies in (0, 1)")
        if not self.tolerance > 0:
            raise ValueError("tolerance must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unit_rows(values: np.ndarray) -> np.ndarray:
    """Each row scaled to unit length; a zero row stays zero."""
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return np.asarray(values / np.where(norm > 0.0, norm, 1.0))


class CodeRegion:
    """``units`` code units reading ``inputs`` observation values and predicted from a context
    of ``context`` values; see the module note for the law."""

    def __init__(self, inputs: int, context: int, settings: CodeSettings, *, seed: int) -> None:
        if inputs < 1 or context < 1:
            raise ValueError("inputs and context must be positive")
        self.inputs, self.context_width, self.settings = int(inputs), int(context), settings
        rng = np.random.default_rng(np.random.SeedSequence([CODE_STREAM, int(seed)]))
        phi = rng.standard_normal((self.inputs, settings.units))
        self.phi = phi / np.linalg.norm(phi, axis=0, keepdims=True)
        self.w = np.zeros((settings.units, self.context_width + 1))
        self.thresholds = np.full(settings.units, float(settings.threshold))
        self.learning = True
        self.work = {
            "settles": 0, "settle_rows": 0, "sweeps": 0, "capped": 0,
            "updates": 0, "update_rows": 0,
        }

    # -- the two boundaries

    def predict(self, context: np.ndarray | None, batch: int) -> np.ndarray:
        """The top-down prediction of the code, ``(batch, units)``, from the context region's
        settled activity of the moment before; None (a stream's first moment) reads zeros."""
        return self._z(context, batch) @ self.w.T

    def _z(self, context: np.ndarray | None, batch: int) -> np.ndarray:
        z = np.zeros((batch, self.context_width + 1))
        if context is not None:
            z[:, : self.context_width] = unit_rows(np.asarray(context, dtype=float))
        z[:, -1] = 1.0
        return z

    # -- the settle

    def settle(
        self, x: np.ndarray, p: np.ndarray, *, top_down: float | None = None
    ) -> tuple[np.ndarray, int]:
        """The settled code of observations ``x`` under the prediction ``p``: ``(batch,
        units)`` and the sweeps taken. ``top_down`` overrides the setting for this settle
        alone (0 settles on the observation alone)."""
        s = self.settings
        gamma = s.top_down if top_down is None else float(top_down)
        x = np.asarray(x, dtype=float)
        drive = x @ self.phi + gamma * p
        gram = self.phi.T @ self.phi
        inhibit = gram - np.eye(s.units) * (1.0 - gamma)
        u = np.zeros((len(x), s.units))
        sweeps, settled = 0, False
        while sweeps < s.budget:
            sweeps += 1
            a = np.maximum(u - self.thresholds, 0.0)
            du = drive - u - a @ inhibit
            if float(np.max(np.abs(du), initial=0.0)) < s.tolerance:
                settled = True
                break
            u += s.step * du
        if not settled:
            self.work["capped"] += 1
        self.work["settles"] += 1
        self.work["settle_rows"] += len(x)
        self.work["sweeps"] += sweeps
        return np.maximum(u - self.thresholds, 0.0), sweeps

    def residuals(self, x: np.ndarray, a: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
        """Per row: the bottom-up residual's squared norm over the observation's, and the
        top-down residual's squared norm."""
        x = np.asarray(x, dtype=float)
        rx = x - a @ self.phi.T
        energy = np.maximum(np.sum(x * x, axis=1), 1e-12)
        return {"bottom_up": np.sum(rx * rx, axis=1) / energy,
                "top_down": np.sum((a - p) ** 2, axis=1)}

    # -- learning

    def learn(self, x: np.ndarray, a: np.ndarray, p: np.ndarray,
              context: np.ndarray | None) -> None:
        """One update of the dictionary, the predictor and the thresholds from this moment's
        settled code; nothing when learning is off."""
        if not self.learning:
            return
        s = self.settings
        x = np.asarray(x, dtype=float)
        n = len(x)
        rx = x - a @ self.phi.T
        self.phi += s.dictionary_rate * (rx.T @ a) / n
        norm = np.linalg.norm(self.phi, axis=0, keepdims=True)
        self.phi /= np.where(norm > 0.0, norm, 1.0)
        z = self._z(context, n)
        rate = np.minimum(s.predictor_rate, 1.0 / np.sum(z * z, axis=1))
        self.w += ((a - p) * rate[:, None]).T @ z / n
        active = np.mean(a > 0.0, axis=0)
        self.thresholds = np.maximum(
            self.thresholds + s.threshold_rate * (active - s.target), 0.0
        )
        self.work["updates"] += 1
        self.work["update_rows"] += n

    # -- reports

    def nbytes(self) -> int:
        """The region's learned tables: dictionary, predictor and thresholds."""
        return int(self.phi.nbytes + self.w.nbytes + self.thresholds.nbytes)

    def describe(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "context": self.context_width,
                "settings": self.settings.to_dict(), "bytes": self.nbytes()}
