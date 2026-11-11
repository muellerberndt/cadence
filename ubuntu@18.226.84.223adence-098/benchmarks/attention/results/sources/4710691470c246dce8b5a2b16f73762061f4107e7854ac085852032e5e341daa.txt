"""The settling certificate: a contraction rate from the row mass and the neuron's slope.

One settling step of a brain without adaptation is ``v <- (1 - dt) v + dt (W act(v) + c)``.
With ``L`` a Lipschitz constant of the activation and ``rho`` the largest absolute incoming
effective weight sum of any neuron, the step contracts the sup norm by

    q = 1 - dt (1 - L rho)

whenever ``L rho < 1``. Then the equilibrium is unique, ``k`` steps from any start leave at
most ``q^k / (1 - q)`` times the first movement, the last observed movement bounds the
remaining error by ``movement / (1 - q)``, and a warm start after a stimulus change of size
``delta`` needs about ``log(dt delta / ((1 - q) tol)) / log(1 / q)`` steps. These follow from
the Banach fixed-point theorem.

The certificate applies to the free phase without adaptation. A nudge adds a term and
adaptation adds a slow variable; neither is covered. For them, ``Brain.residual`` and
``Brain.equilibrate`` check the equations at the measured state, without a uniqueness or
convergence guarantee. A brain that is not certified may settle in practice; the certificate
then says nothing either way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .brain import Brain
    from .neuron import NeuronModel


def lipschitz_constant(model: NeuronModel) -> float:
    """Supremum of the activation's slope: ``(lambda / 4) max(1 / (1 - rho), leak / rho)``."""
    rest = model.rest_emission
    return float(model.slope / 4.0 * max(1.0 / (1.0 - rest), model.leak / rest))


def row_mass(brain: Brain) -> float:
    """The largest absolute incoming effective weight sum over all neurons."""
    incoming = np.bincount(
        brain.connectome.post, weights=np.abs(brain.weights), minlength=brain.connectome.n
    )
    return float(incoming.max()) if len(incoming) else 0.0


@dataclass(frozen=True)
class Certificate:
    """What the contraction argument certifies for a brain's free settling."""

    row_mass: float
    lipschitz: float
    dt: float
    adaptation: bool

    def __post_init__(self) -> None:
        if not np.isfinite([self.row_mass, self.lipschitz, self.dt]).all():
            raise ValueError("certificate parameters must be finite")
        if self.row_mass < 0 or self.lipschitz <= 0 or not 0 < self.dt <= 1:
            raise ValueError("row mass must be nonnegative, Lipschitz positive, dt in (0, 1]")

    @property
    def rate(self) -> float:
        """The contraction constant ``1 - dt (1 - L rho)``; below one when certified."""
        return 1.0 - self.dt * (1.0 - self.lipschitz * self.row_mass)

    @property
    def certified(self) -> bool:
        return self.rate < 1.0 and not self.adaptation

    @property
    def mass_limit(self) -> float:
        """The row mass below which this neuron model is certified: ``1 / L``."""
        return 1.0 / self.lipschitz

    def error_bound(self, movement: np.ndarray | float) -> np.ndarray:
        """Remaining distance to the equilibrium from the last step's movement (sup norm).

        ``movement`` is the largest change of a potential in the last step, per row;
        the bound is ``movement / (1 - rate)``. Uncertified brains get ``inf``.
        """
        out = np.asarray(movement, dtype=float)
        if not np.isfinite(out).all() or (out < 0).any():
            raise ValueError("movement must be finite and nonnegative")
        if not self.certified:
            return np.full_like(out, np.inf)
        return np.asarray(out / (1.0 - self.rate))

    def apriori_bound(self, first_movement: np.ndarray | float, steps: int) -> np.ndarray:
        """Distance after ``steps`` settling steps, from the first step's movement."""
        out = np.asarray(first_movement, dtype=float)
        if not np.isfinite(out).all() or (out < 0).any():
            raise ValueError("first movement must be finite and nonnegative")
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if not self.certified:
            return np.full_like(out, np.inf)
        return np.asarray(out * self.rate**steps / (1.0 - self.rate))

    def steps_for(self, change: float, tolerance: float) -> int:
        """Warm-start steps after a stimulus change of size ``change`` (sup norm) to be
        within ``tolerance`` of the new equilibrium. Zero for no change."""
        if not self.certified:
            raise ValueError("an uncertified brain has no step bound")
        if not np.isfinite(change) or change < 0:
            raise ValueError("change must be finite and nonnegative")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if change == 0.0:
            return 0
        log_ratio = (
            math.log(self.dt) + math.log(change) - math.log1p(-self.rate) - math.log(tolerance)
        )
        if log_ratio <= 0.0:
            return 0
        if self.rate == 0.0:  # dt=1 and no coupling reaches the fixed point in one step
            return 1
        return int(math.ceil(log_ratio / -math.log(self.rate)))

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "row_mass": self.row_mass,
            "lipschitz": self.lipschitz,
            "dt": self.dt,
            "rate": self.rate,
            "certified": self.certified,
            "mass_limit": self.mass_limit,
            "adaptation": self.adaptation,
        }


def certificate(brain: Brain) -> Certificate:
    """The settling certificate of ``brain``'s free phase.

    Reads the effective weights, the neuron model's slope and leak, and whether adaptation
    is present. Ablation masks only remove synapses, so a certified brain stays certified
    under any mask.
    """
    model = brain.neuron_model
    return Certificate(
        row_mass=row_mass(brain),
        lipschitz=lipschitz_constant(model),
        dt=float(model.dt),
        adaptation=model.adaptation is not None,
    )


@dataclass(frozen=True)
class EPStructure:
    """Structural checks only; not a certificate of convergence or a loss gradient."""

    fixed_inputs: tuple[int, ...]
    free_neurons: int
    free_asymmetry: float
    fixed_incoming_mass: float
    adaptation: bool
    tolerance: float

    @property
    def compatible(self) -> bool:
        """Whether the inspected structure admits the usual symmetric free-state argument."""
        return (
            self.free_asymmetry <= self.tolerance
            and self.fixed_incoming_mass <= self.tolerance
            and not self.adaptation
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "compatible": self.compatible}


def ep_structure(
    brain: Brain, *, fixed_inputs: Sequence[int] = (), tolerance: float = 1e-12
) -> EPStructure:
    """Inspect effective free/free symmetry, autonomous inputs, and adaptation in O(E) space.

    Source neurons with no incoming effective weight have a drive-determined equilibrium.
    After convergence, their outgoing projections are external fields; no reverse edge is
    needed for the EP argument on the remaining neurons. The caller must keep these inputs
    unchanged and unnudged between phases and hold their own parameters fixed when taking
    derivatives. This function checks that they really are sources, rather than allowing
    an arbitrary asymmetric subgraph to be hidden by naming it an input.

    This does NOT check phase residuals, stability, activation smoothness, finite-beta
    bias, parameter tying, or the contact/gain and loss-temperature conversion of a raw
    contrast. Inspect those separately before interpreting a contrast as a gradient.
    Parallel effective edges are summed before comparing the free/free matrix.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    n, graph = brain.connectome.n, brain.connectome
    indices = np.asarray(list(fixed_inputs))
    if indices.size and (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or (indices < 0).any()
        or (indices >= n).any()
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("fixed_inputs must contain distinct valid neuron indices")
    indices = indices.astype(np.int64)
    fixed = np.zeros(n, dtype=bool)
    fixed[indices] = True
    key, inverse = np.unique(graph.pre * n + graph.post, return_inverse=True)
    weights = np.bincount(inverse, weights=brain.weights, minlength=len(key))
    pre, post = key // n, key % n
    incoming = np.bincount(post[fixed[post]], weights=np.abs(weights[fixed[post]]), minlength=n)
    selected = ~fixed[pre] & ~fixed[post]
    free_key, free_weight = key[selected], weights[selected]
    reverse_key = post[selected] * n + pre[selected]
    hits = np.searchsorted(free_key, reverse_key)
    other = np.zeros_like(free_weight)
    if len(free_key):
        valid = (hits < len(free_key)) & (
            free_key[np.minimum(hits, len(free_key) - 1)] == reverse_key
        )
        other[valid] = free_weight[hits[valid]]
    return EPStructure(
        fixed_inputs=tuple(int(i) for i in indices),
        free_neurons=n - len(indices),
        free_asymmetry=float(np.max(np.abs(free_weight - other), initial=0.0)),
        fixed_incoming_mass=float(np.max(incoming, initial=0.0)),
        adaptation=brain.neuron_model.adaptation is not None,
        tolerance=tolerance,
    )
