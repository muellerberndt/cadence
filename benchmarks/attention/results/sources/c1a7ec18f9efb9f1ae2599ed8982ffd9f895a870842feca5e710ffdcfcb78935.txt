"""A learned code of meaning: a region of the brain's own neurons settling to a sparse code.

``Brain.compose(..., learned_code=units)`` places a ``code`` region of that many neurons
between the sensory region and the first module. Its synapses and biases form the region's
own sub-network, qualified before its context-free code is held as a source through the
whole brain's settle. The region uses the brain's neuron model:

* the dictionary ``Phi`` (``inputs`` by ``units``, unit-length columns) is the one-way
  synapses from the sensory neurons at the gain ``read = 1 / f(A)`` (``f`` the neuron's
  activation, ``A`` the model's stimulus amplitude): the region reads the sensory
  activation in units of one full input, ``x' = read * s_sensory``, so that one input at
  amplitude 1 drives the unit whose column matches it by one, and a code unit ``j`` is
  driven by ``Phi_j . x'``;
* the dictionary's Gram matrix is the lateral inhibition among the code units: every ordered
  pair ``j != k`` carries ``-(Phi_j . Phi_k)``, so units whose columns overlap compete;
* the predictor ``W`` (``units`` by the prefrontal width) is the one-way synapses from the
  prefrontal region, the working trace of the association activity of the moment before,
  scaled by ``top_down``: the context's prediction of the code, ``p = W s_prefrontal``;
* each unit's threshold is its bias, ``-t_j``;
* a standing boundary on the code units, the solver's quadratic anchor at gain ``top_down``
  toward zero, ``-top_down * s_j``, held through the code's own phase: with the predictor's
  drive it is the pull ``top_down * (p_j - s_j)`` toward the prediction, the gradient of
  ``top_down / 2 * |s - p|^2``.

A code unit therefore settles to ``v_j = Phi_j . (x' - Phi s) + s_j + top_down (p_j -
s_j) - t_j`` with ``s_j = f(v_j)`` the neuron's rectified sigmoid: the sparse state that
reconstructs the sensory activation, as read, through the dictionary as far as the
thresholds allow,
drawn toward what the context predicts. No synapse leads back into the region, so its
fixed point depends on the observation and the trace alone; the brain settles it first,
with the same solver and tolerance, addresses its memories on it, and settles the whole
graph warm from it.

Learning is local to each boundary and happens at each accepted explicit ``act`` or ``step``
moment, and in ``live`` while aroused; ``wait`` follows the issued action's mode. Routine
``live`` moments keep the tables still. Each row of the batch is one event, from the
moment's free settle alone: the lesson's nudged phases, the
outcome's eligibility phases, the imagined branches and the readings taken on copies enter
no table, as the contrast and the dopamine enter no synapse of the region
(``a = max(s, 0)``, the rectified code;
``x'`` the sensory activation as read; ``z`` the prefrontal activation; ``p = W z``):

* the dictionary, Hebbian on the bottom-up residual, ``Phi += rate * sum_rows (x' - Phi a)
  a^T``, each column then scaled back to unit length (a unit's own synaptic scaling);
* the lateral weights, recomputed from the dictionary;
* the predictor, the delta rule on the top-down residual normalized by the context's
  energy, ``W += sum_rows r_row (a - p) z^T / max(1, rows * predictor)`` with
  ``r_row = predictor / max(|z|^2,
  predictor^2)``: before the batch bound, a row closes the share ``predictor`` of its own gap
  whatever the
  magnitude of its context (the prefrontal activation is small in the brain's units), a
  context fainter than ``|z| = predictor`` counted at that energy, so that no row commands a
  synaptic step above its gap. The batch divisor bounds the sum of simultaneous context
  updates, so repeated rows cannot amplify the prediction error; the founder rate leaves
  batches of at most 16 rows unchanged;
* the thresholds, homeostatic, ``t += threshold_rate * sum_rows ((a_free > active) -
  target)`` with ``a_free`` the moment's context-free code (the region settled on the
  observation alone, the code ``Brain.observation_code`` gives) and ``active`` the active
  floor 0.05 (``LearnedCode.ACTIVE``): a unit's share is the share of the observations that
  drive it on their own above the floor, so what the context predicts may recruit a unit
  without raising its threshold, a unit that fires below the floor counts as under-active,
  and the context-free code keeps the target share by construction; floored at zero, with a
  repair tied to the region's own disagreement: a row whose code leaves at least the share
  ``repair`` of its input's energy unexplained (``|x' - Phi a|^2 >= repair * |x'|^2``; at
  ``repair`` 1, ``LearnedCode.REPAIR``, a row whose code is silent or worse than nothing),
  in context or context-free (the memories' key), lowers the threshold of every unit silent
  in that row by ``threshold_rate * silence``, a silent unit in such a row counting as that
  many under-active rows, so no observation is left without a unit; a unit that fires, and
  a row the code reconstructs, are left to the share rule alone. There is no ceiling: a
  unit the world drives often has its threshold raised until its share is the target, and
  the rows it then leaves unexplained lower it again.

While the code fits, the residuals are small and nothing moves. The rates are constants
of the region (``LearnedCode.RATE``, ``PREDICTOR`` and ``THRESHOLD_RATE``); ``learning``
False holds every table still, for a reading on a copy. The memories key on the rectified
settled code at unit length joined to the observation's own identity, ``unit([sqrt(1 - w)
code, sqrt(w) unit(x)])`` with ``w`` the identity weight 0.9 (``LearnedCode.IDENTITY``;
``Brain.observation_code``): the identity part keeps a binding readable while the code part
moves as the region learns. An observation wider than the code is keyed on the code
alone (``LearnedCode.joins``), so that the key keeps the code's width.

With the founder standing gain 2, the lateral coupling less that gain is ``-I - G``,
``G`` the dictionary's Gram matrix, so the slope-weighted coupling is nonpositive. The
brain's synchronous sweep can still oscillate when overlapping columns are active together:
its negative modes need ``dt * (1 - mu_min) < 2``. ``stability`` reports the halvings for
that bound. Local contraction also requires the largest slope-weighted coupling eigenvalue
below one; custom controls, including ``top_down=0`` with a large negative-branch leak,
need that separate check. Damping cannot stabilize an intrinsically growing mode.
The code's phase integrates at its selected step with its budget scaled to preserve the
model integration horizon
(``restep``); the code is held as a source through the whole settle, so the rest of the
brain keeps the model's step. The restoring term subtracts ``top_down`` times the slope from
the coupling's diagonal. ``rest_stability`` bounds the same negative modes over every state
the region can reach, using the largest slope on either activation branch: a code
phase that does not settle at its step is settled once more at that bound before the moment
is refused. This bound is available from the first moment even when the dictionary's
columns all overlap (one input: every column is plus or minus the same vector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .brain import Brain as NeuralGraph
from .brain import BrainState, Equilibrium, Nudge
from .connectome import Connectome
from .neuron import NeuronModel

__all__ = ["LearnedCode"]

# The dictionary's own stream of the brain's seed, apart from the streams the brain develops,
# acts and draws its fixed code from (``SeedSequence([seed, 7103])``).
_CODE_STREAM = 7109


def _unit_columns(values: np.ndarray) -> np.ndarray:
    """Each column scaled to unit length; a zero column stays zero."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        norm = np.linalg.norm(values, axis=0, keepdims=True)
    out = values / np.where(norm > 0.0, norm, 1.0)
    # Keep the ordinary arithmetic unchanged; rescale only columns whose squared norm
    # overflowed or underflowed although their finite coordinates are nonzero.
    exceptional = (~np.isfinite(norm) | ((norm == 0.0) & (values != 0.0).any(axis=0)))[0]
    if exceptional.any():
        part = values[:, exceptional]
        scale = np.max(np.abs(part), axis=0, keepdims=True)
        scaled = part / np.where(scale > 0.0, scale, 1.0)
        out[:, exceptional] = scaled / np.linalg.norm(scaled, axis=0, keepdims=True)
    return np.asarray(out)


def _allocation(widths: tuple[int, ...], units: int) -> tuple[int, ...]:
    """Units per field: in proportion to the fields' widths, at least two each, the
    remainders given to the largest fractional shares first."""
    total = float(sum(widths))
    raw = [units * width / total for width in widths]
    counts = [max(2, int(np.floor(share))) for share in raw]
    while sum(counts) < units:
        gaps = [share - count for share, count in zip(raw, counts, strict=True)]
        counts[int(np.argmax(gaps))] += 1
    while sum(counts) > units:
        spare = [i for i, count in enumerate(counts) if count > 2]
        counts[min(spare, key=lambda i: raw[i] - counts[i])] -= 1
    return tuple(counts)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Each row scaled to unit length; a row without a positive entry stays zero."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        norm = np.linalg.norm(values, axis=1, keepdims=True)
    out = values / np.where(norm > 0.0, norm, 1.0)
    nonzero = (values != 0.0).any(axis=1)[:, None]
    exceptional = (~np.isfinite(norm) | ((norm == 0.0) & nonzero))[:, 0]
    if exceptional.any():
        out[exceptional] = _unit_columns(values[exceptional].T).T
    return np.asarray(out)


@dataclass
class LearnedCode:
    """The learned code region's tables, sub-network and rules; see the module note.

    ``inputs`` is the sensory width, ``units`` the region's, ``context`` the prefrontal
    width; ``model`` the brain's neuron model, under which the region settles, and from
    which the read gain follows (``read``, one over the activation one input at amplitude
    1 gives a sensory neuron at rest). ``fields`` (the widths of the observation's fields,
    each unit reading one) is the region's one setting, fixed at construction. Its rules
    run at constants the readings selected (``TOP_DOWN``, the prediction's weight on the
    units; ``RATE``, the dictionary's Hebbian rate per row; ``PREDICTOR``, the share of a
    row's prediction gap closed per event; ``THRESHOLD``, the initial threshold of every
    unit; ``THRESHOLD_RATE`` and ``TARGET``, the share of rows a unit should be active in;
    ``REPAIR``, the share of a row's input energy its code may leave unexplained before the
    row lowers its silent units' thresholds; ``SILENCE``, how many under-active rows a
    silent unit in such a row counts as; ``ACTIVE``, the activation a unit must reach to
    count as active in the thresholds' share; ``IDENTITY``, the weight of
    the observation's own identity in the memories' key), held as the instance's
    attributes of the same names in lower case. Two of them are for readings and controls:
    ``learning`` (True; False holds every table still, for a reading on a copy) and
    ``top_down`` (0 is the control without the top-down boundary).
    """

    inputs: int
    units: int
    context: int
    model: NeuronModel
    seed: int = 0
    fields: tuple[int, ...] | None = None
    dictionary: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    thresholds: np.ndarray = field(init=False, repr=False)
    read: float = field(init=False)

    # the region's constants, the values the readings selected
    TOP_DOWN = 2.0
    RATE = 0.125
    PREDICTOR = 0.0625
    THRESHOLD = 0.1
    THRESHOLD_RATE = 0.000625
    TARGET = 0.1
    REPAIR = 1.0
    SILENCE = 10.0
    ACTIVE = 0.05
    IDENTITY = 0.9
    GENES = ("fields",)
    RUNTIME: tuple[str, ...] = ()
    CONSTRUCTION = ("fields",)
    # Keep the lower end of the linearized sweep's spectrum at least -0.5.
    # Contraction also requires its upper end below one.
    STEP_MARGIN = 1.5
    # settings the readings retired, refused by name with what the region does instead
    REMOVED = {
        "cortex": "the first module reads the code alone",
        "output": "the first module reads the rectified code at unit length",
        "floor": "every code that is not silent is read at unit length",
        "scale": "the memories' key takes the norm the observation itself has as a key",
        "place": "the associative store and the episodic place records key on the "
        "context-free code",
        "reads": "the cortex and the memories read the context-free code; the region learns "
        "from the code in context",
        "held": "the code units are held at the code phase's fixed point through the whole "
        "settle, and the rest of the brain keeps the model's step",
        "top_down": "the prediction's weight is a constant of the region (2.0); a control sets "
        "brain.learned_code.top_down = 0",
        "rate": "the region's rates are constants; a reading on a copy sets "
        "brain.learned_code.learning = False",
        "predictor": "the predictor's share per event is a constant of the region (0.0625)",
        "threshold": "the initial threshold is a constant of the region (0.1)",
        "threshold_rate": "the thresholds' rate is a constant of the region (0.000625)",
        "target": "the thresholds' target share is a constant of the region (0.1)",
        "repair": "the repair's level is a constant of the region (1.0)",
        "silence": "the repair's weight is a constant of the region (10)",
        "identity": "the identity's weight in the memories' key is a constant of the region "
        "(0.9)",
        "active": "the thresholds' active floor is a constant of the region (0.05)",
    }

    @staticmethod
    def joins(inputs: int, units: int) -> bool:
        """Whether the memories' key joins the observation's own identity to the code: for an
        observation no wider than the code. A wider one (a camera's frame) is keyed on the
        code alone, so that the key keeps the code's width."""
        return int(inputs) <= int(units)

    @classmethod
    def founders(cls) -> dict[str, Any]:
        """The founder value of every setting."""
        return {"fields": None}

    @staticmethod
    def setting(name: str, value: Any) -> Any:
        """One setting, validated and converted; a removed setting is refused."""
        if name in LearnedCode.REMOVED:
            raise ValueError(f"learned_code_{name} was removed: {LearnedCode.REMOVED[name]}")
        if name == "fields":
            if value is None:
                return None
            if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
                raise ValueError("fields is None or a sequence of field widths")
            widths = tuple(value)
            if not widths or any(
                isinstance(w, (bool, np.bool_)) or not isinstance(w, (int, np.integer)) or w < 1
                for w in widths
            ):
                raise ValueError("fields is None or a sequence of positive field widths")
            return tuple(int(w) for w in widths)
        raise TypeError(f"unknown learned code setting: {name!r}")

    def __post_init__(self) -> None:
        for name in ("inputs", "units", "context"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if self.inputs < 1 or self.context < 1:
            raise ValueError("inputs and context must be positive")
        if self.units < 2:
            raise ValueError("a learned code holds at least 2 units")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be a nonnegative integer")
        self.seed = int(self.seed)
        for name in self.GENES:
            setattr(self, name, self.setting(name, getattr(self, name)))
        # the constants as the instance's attributes; ``learning`` False holds every table
        # still (a reading on a copy), ``top_down`` 0 is the control without the boundary
        self.learning = True
        self.top_down = self.TOP_DOWN
        self.rate, self.predictor = self.RATE, self.PREDICTOR
        self.threshold, self.threshold_rate = self.THRESHOLD, self.THRESHOLD_RATE
        self.target, self.repair, self.silence = self.TARGET, self.REPAIR, self.SILENCE
        self.active = self.ACTIVE
        self.identity = self.IDENTITY if self.joins(self.inputs, self.units) else 0.0
        # the fields: each unit reads one field of the observation's coordinates
        self.mask: np.ndarray | None = None
        self.field_units: tuple[int, ...] = (self.units,)
        if self.fields is not None:
            if sum(self.fields) != self.inputs:
                raise ValueError(
                    f"the fields' widths sum to {sum(self.fields)}, not to the {self.inputs} inputs"
                )
            if self.units < 2 * len(self.fields):
                raise ValueError("a learned code holds at least two units per field")
            self.field_units = _allocation(self.fields, self.units)
            field_of_input = np.repeat(np.arange(len(self.fields)), self.fields)
            field_of_unit = np.repeat(np.arange(len(self.fields)), self.field_units)
            self.mask = field_of_input[:, None] == field_of_unit[None, :]
        amplitude = np.array([self.model.stimulus_amplitude])
        full = float(np.asarray(self.model.activation(amplitude)).item())
        if not full > 0.0:
            raise ValueError(
                "a learned code needs a stimulus amplitude that drives a sensory neuron"
            )
        self.read = 1.0 / full
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, _CODE_STREAM]))
        draw = rng.standard_normal((self.inputs, self.units))
        if self.mask is not None:
            draw = draw * self.mask
        self.dictionary = _unit_columns(draw)
        self.weights = np.zeros((self.units, self.context))
        self.thresholds = np.full(self.units, float(self.threshold))
        # the sub-network: the units with every ordered pair of lateral synapses
        pre, post = np.meshgrid(np.arange(self.units), np.arange(self.units), indexing="ij")
        apart = pre.ravel() != post.ravel()
        self._pre, self._post = pre.ravel()[apart], post.ravel()[apart]
        self.connectome = Connectome.from_synapses(
            self.units, pre=self._pre, post=self._post, sign=self.lateral()[self._pre, self._post],
            label=f"learned-code:{self.units}",
        )
        # the connectome sorts its synapses by (post, pre); the lateral weights are read in it
        self._pre, self._post = self.connectome.pre, self.connectome.post
        self.halvings = 0  # the step's halvings the settled code needs (``stability``)
        self.graph = NeuralGraph(
            self.connectome, self.model, efficacy=self.lateral_efficacy(), bias=-self.thresholds
        )
        self.work = {
            "settles": 0, "settle_rows": 0, "sweeps": 0, "capped": 0,
            "updates": 0, "update_rows": 0, "repairs": 0,
        }
        self.last: dict[str, Any] | None = None

    # -- the step

    def stability(self, v: np.ndarray) -> int:
        """The step's halvings that damp the negative modes at potentials ``v``
        (``(batch, units)``): the least ``h``
        with ``dt / 2^h * (1 - mu_min) <= STEP_MARGIN``, ``mu_min`` the smallest eigenvalue
        over the rows of ``D^1/2 (gain L - top_down I) D^1/2`` (``D`` the activation's slopes at
        ``v``, ``L`` the lateral weights, ``gain`` the neuron model's synaptic gain,
        ``top_down`` the standing boundary's gain); 0 when
        the bound holds at the model's own step. Local contraction also requires the largest
        coupling eigenvalue below one, as the founder standing gain ensures."""
        v = np.atleast_2d(np.asarray(v, dtype=float))
        slopes = np.maximum(self.model.slope_at(v), 0.0)
        root = np.sqrt(slopes)
        lateral = self.model.gain * self.lateral()
        coupling = root[:, :, None] * lateral[None, :, :] * root[:, None, :]
        # the standing boundary's restoring term, -top_down on every unit's own diagonal
        index = np.arange(self.units)
        coupling[:, index, index] -= self.top_down * slopes
        mu = float(np.linalg.eigvalsh(coupling)[:, 0].min())
        need = self.model.dt * (1.0 - mu) / self.STEP_MARGIN
        return 0 if need <= 1.0 else int(np.ceil(np.log2(need)))

    def rest_stability(self) -> int:
        """The negative-mode step bound of ``stability`` over every state, from the
        dictionary alone. With every slope at most the activation's largest,
        ``s_max``, the coupling's smallest eigenvalue is at least ``s_max (gain *
        (1 - lambda_max(G)) - top_down)``, ``G`` the dictionary's Gram matrix. This does not
        stabilize a custom control whose largest coupling eigenvalue is at least one."""
        rest = float(self.model.rest_emission)
        steepest = max(rest, 0.5)  # the active branch's steepest point of the sigmoid
        positive = self.model.slope * steepest * (1.0 - steepest) / (1.0 - rest)
        negative_peak = min(rest, 0.5)
        negative = (
            self.model.leak * self.model.slope * negative_peak * (1.0 - negative_peak) / rest
        )
        s_max = max(positive, negative)
        largest = float(np.linalg.eigvalsh(self.gram())[-1])
        mu = s_max * (self.model.gain * (1.0 - largest) - self.top_down)
        need = self.model.dt * (1.0 - mu) / self.STEP_MARGIN
        return 0 if need <= 1.0 else int(np.ceil(np.log2(need)))

    def standing(self) -> Nudge | None:
        """The region's standing boundary over its own units: the quadratic anchor toward
        zero at gain ``top_down`` on every unit, ``-top_down * s``; None at ``top_down`` 0."""
        if not self.top_down:
            return None
        zeros = np.zeros(self.units)
        return Nudge(
            zeros, zeros, 0.0, anchor=zeros, anchor_gain=np.full(self.units, self.top_down)
        )

    def restep(self, halvings: int) -> None:
        """The sub-network at the model's step halved ``halvings`` times."""
        if (
            isinstance(halvings, (bool, np.bool_))
            or not isinstance(halvings, (int, np.integer))
            or halvings < 0
        ):
            raise ValueError("halvings must be a nonnegative integer")
        halvings = int(halvings)
        if halvings == self.halvings:
            return
        if halvings > 1074 or np.ldexp(self.model.dt, -halvings) <= 0.0:
            raise ValueError("the learned code's step must remain positive")
        graph = NeuralGraph(
            self.connectome,
            self.model.replace(dt=float(np.ldexp(self.model.dt, -halvings))),
            efficacy=self.lateral_efficacy(), bias=-self.thresholds,
        )
        self.halvings, self.graph = halvings, graph

    # -- the tables as synapses

    def gram(self) -> np.ndarray:
        return np.asarray(self.dictionary.T @ self.dictionary)

    def lateral(self) -> np.ndarray:
        """The lateral weight from unit ``j`` to unit ``k``, ``-(Phi_j . Phi_k)``, as a
        ``(units, units)`` matrix whose diagonal is zero (a neuron has no synapse onto
        itself)."""
        out = -self.gram()
        np.fill_diagonal(out, 0.0)
        return np.asarray(out)

    def lateral_efficacy(self) -> np.ndarray:
        """The lateral weights in the sub-network's synapse order."""
        return np.asarray(self.lateral()[self._pre, self._post])

    def install(self) -> None:
        """The sub-network takes the tables as they are: the lateral weights and the
        thresholds as biases."""
        self.graph = self.graph.with_parameters(
            efficacy=self.lateral_efficacy(), bias=-self.thresholds
        )

    def take(self, dictionary: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> None:
        """Replace the tables (a checkpoint's), validated, and install them."""
        dictionary = np.asarray(dictionary, dtype=float)
        weights = np.asarray(weights, dtype=float)
        thresholds = np.asarray(thresholds, dtype=float)
        if (
            dictionary.shape != (self.inputs, self.units)
            or weights.shape != (self.units, self.context)
            or thresholds.shape != (self.units,)
            or not np.isfinite(dictionary).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(thresholds).all()
            or (thresholds < 0).any()
        ):
            raise ValueError("invalid learned code tables")
        norms = np.linalg.norm(dictionary, axis=0)
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-9):
            raise ValueError("a learned code's dictionary has unit-length columns")
        if self.mask is not None and dictionary[~self.mask].any():
            raise ValueError("a learned code's dictionary reads its own fields only")
        self.dictionary, self.weights, self.thresholds = dictionary, weights, thresholds
        self.install()

    # -- the settle

    def stimulus(self, sensory: np.ndarray, context: np.ndarray | None) -> np.ndarray:
        """The drive of the code units, ``(batch, units)``: the dictionary's read of the
        sensory activation at the region's gain plus ``top_down`` times the predictor's read
        of the prefrontal activation; without a context the prediction is zero."""
        drive = (self.read * np.asarray(sensory, dtype=float)) @ self.dictionary
        if context is not None and self.top_down:
            drive = drive + self.top_down * self.prediction(context)
        return np.asarray(drive)

    def prediction(self, context: np.ndarray) -> np.ndarray:
        """``p = W z`` for the ``(batch, context)`` prefrontal activation."""
        return np.asarray(np.asarray(context, dtype=float) @ self.weights.T)

    def settle(
        self,
        stimulus: np.ndarray,
        warm: BrainState | None,
        *,
        budget: int,
        tolerance: float,
        damping: int = 3,
    ) -> Equilibrium:
        """The region settled under ``stimulus`` by the brain's solver policy: the fast
        repair at the model's step, then halved steps within the same ``budget``, to the
        residual ``tolerance``; counted as work."""
        phase = self.graph.equilibrate(
            stimulus, state=warm, budget=budget, tolerance=tolerance, damping=damping,
            nudge=self.standing(),
        )
        self.work["settles"] += 1
        self.work["settle_rows"] += len(stimulus)
        self.work["sweeps"] += int(phase.steps)
        if not np.all(phase.qualified):
            self.work["capped"] += 1
        return phase

    def code_of(self, activation: np.ndarray) -> np.ndarray:
        """The code the memories key on and the first module reads: the rectified
        activation at unit length; a silent code reads nothing."""
        return _unit_rows(np.maximum(np.atleast_2d(np.asarray(activation, dtype=float)), 0.0))

    def residuals(
        self, sensory: np.ndarray, activation: np.ndarray, context: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Per row: the bottom-up residual's squared norm over the sensory activation's
        (both as read), and the top-down residual's squared norm (zero without a
        context)."""
        x = self.read * np.atleast_2d(np.asarray(sensory, dtype=float))
        a = np.maximum(np.atleast_2d(np.asarray(activation, dtype=float)), 0.0)
        rx = x - a @ self.dictionary.T
        energy = np.maximum(np.sum(x * x, axis=1), 1e-12)
        top = np.zeros(len(x)) if context is None else np.sum((a - self.prediction(context)) ** 2,
                                                              axis=1)
        return {"bottom_up": np.sum(rx * rx, axis=1) / energy, "top_down": top}

    # -- learning

    def learn(
        self, sensory: np.ndarray, activation: np.ndarray, context: np.ndarray | None,
        free: np.ndarray | None = None,
    ) -> bool:
        """One step of the dictionary, the predictor and the thresholds from a settled
        moment, each row an event; returns whether any table moved (every rate zero, or an
        empty batch, moves nothing). ``free`` is the moment's context-free code (the
        rectified activation of the region settled on the observation alone) where one was
        settled: the thresholds' share reads it, and the repair reads what it leaves
        unexplained beside the code in context; without it the code itself serves both.
        The sub-network takes the new tables. Invalid inputs or a nonfinite proposal refuse
        the complete step, preserving the tables, graph and work counts. If a column's
        proposed vector cancels exactly, its existing direction stays."""
        def rows(value: np.ndarray, width: int, name: str) -> np.ndarray:
            out = np.atleast_2d(np.asarray(value, dtype=float))
            if out.ndim != 2 or out.shape[1] != width or not np.isfinite(out).all():
                raise ValueError(f"{name} must have finite rows of width {width}")
            return out

        x = rows(sensory, self.inputs, "sensory")
        a = rows(activation, self.units, "activation")
        z = None if context is None else rows(context, self.context, "context")
        own = None if free is None else rows(free, self.units, "free code")
        if any(len(value) != len(x) for value in (a, z, own) if value is not None):
            raise ValueError("the learned code's inputs need one row per event")
        with np.errstate(over="ignore", invalid="ignore"):
            x = self.read * x
        if not np.isfinite(x).all():
            raise ValueError("the learned code's sensory read must remain finite")
        a = np.maximum(a, 0.0)
        own = None if own is None else np.maximum(own, 0.0)
        n = len(x)
        if not n or not self.learning or not (self.rate or self.predictor or self.threshold_rate):
            return False
        saved = self.dictionary, self.weights, self.thresholds, self.graph, self.work.copy()
        try:
            return self._learn(x, a, z, own)
        except Exception:
            self.dictionary, self.weights, self.thresholds, self.graph, work = saved
            self.work.clear()
            self.work.update(work)
            raise

    def _learn(
        self, x: np.ndarray, a: np.ndarray, context: np.ndarray | None, free: np.ndarray | None,
    ) -> bool:
        """The original local rules on validated rows, rolled back by ``learn`` on refusal."""
        n = len(x)
        moved = False
        if self.rate:
            # the columns of the units that fired move and are scaled back to unit length; a
            # silent unit's column stays as it is, bit for bit
            fired = (a > 0.0).any(axis=0)
            if fired.any():
                residual = x - a @ self.dictionary.T
                dictionary = self.dictionary.copy()
                step = dictionary[:, fired] + self.rate * (residual.T @ a[:, fired])
                if self.mask is not None:  # a unit's column stays within its field
                    step = step * self.mask[:, fired]
                if not np.isfinite(step).all():
                    raise ValueError("the learned code's dictionary must remain finite")
                cancelled = ~(step != 0.0).any(axis=0)
                if cancelled.any():
                    # A zero vector has no direction to scale back to unit length.
                    # Keep that unit's existing direction instead of erasing its dictionary.
                    step[:, cancelled] = dictionary[:, fired][:, cancelled]
                dictionary[:, fired] = _unit_columns(step)
                if not np.isfinite(dictionary).all():
                    raise ValueError("the learned code's dictionary must remain finite")
                self.dictionary = dictionary
                moved = True
        if self.predictor and context is not None:
            z = np.atleast_2d(np.asarray(context, dtype=float))
            # the share of the row's gap, normalized by its context's energy, floored at
            # predictor^2; a silent context (z = 0) moves nothing
            rate = self.predictor / np.maximum(np.sum(z * z, axis=1), self.predictor**2)
            error = a - self.prediction(z)
            # Each normalized context outer product has trace at most ``predictor``.
            # Their positive-semidefinite sum therefore has largest eigenvalue at most
            # n * predictor: this divisor keeps every mode of the simultaneous update
            # non-expansive, including duplicate contexts in a large batch. At the founder
            # rate it preserves the original law through 16 rows, the chamber batch size.
            divisor = max(1.0, n * self.predictor)
            weights = self.weights + ((error * rate[:, None]).T @ z) / divisor
            if not np.isfinite(weights).all():
                raise ValueError("the learned code's predictor must remain finite")
            self.weights = weights
            moved = True
        if self.threshold_rate:
            own = a
            if free is not None:
                own = np.maximum(np.atleast_2d(np.asarray(free, dtype=float)), 0.0)
            # the share: the observations a unit fires for on their own, context-free, at an
            # activation above ``active`` (a unit firing below it counts as under-active)
            active = (own > self.active).astype(float) - self.target
            step = self.threshold_rate * active.sum(axis=0)
            if self.silence:
                # the repair: a row whose code leaves at least the share ``repair`` of its
                # input's energy unexplained, in context or context-free, lowers the
                # thresholds of the units silent in it; counted per unit and row
                repairs = self.unexplained(x, a)
                if free is not None:
                    repairs = repairs + self.unexplained(x, own)
                step = step - self.threshold_rate * self.silence * repairs
                self.work["repairs"] += int(repairs.sum())
            self.thresholds = np.maximum(self.thresholds + step, 0.0)
            if not np.isfinite(self.thresholds).all():
                raise ValueError("the learned code's thresholds must remain finite")
            moved = True
        self.work["updates"] += 1
        self.work["update_rows"] += n
        if moved:
            self.install()
        return moved

    def unexplained(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Per unit, how many rows of ``a`` (rectified codes of the read inputs ``x``)
        leave at least the share ``repair`` of their input's energy unexplained and have the
        unit silent; a row without input energy counts for no unit."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        a = np.atleast_2d(np.asarray(a, dtype=float))
        residual = x - a @ self.dictionary.T
        energy = np.sum(x * x, axis=1)
        left = np.sum(residual * residual, axis=1)
        repairing = (energy > 0.0) & (left >= self.repair * energy)
        return np.asarray((repairing[:, None] & ~(a > 0.0)).sum(axis=0), dtype=float)

    # -- reports

    def arrays(self) -> dict[str, np.ndarray]:
        """The tables by name, as a checkpoint holds them."""
        return {
            "dictionary": self.dictionary, "weights": self.weights, "thresholds": self.thresholds,
        }

    def parameters(self) -> int:
        return int(self.dictionary.size + self.weights.size + self.thresholds.size)

    def nbytes(self) -> int:
        return int(self.dictionary.nbytes + self.weights.nbytes + self.thresholds.nbytes)

    def genes(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.GENES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs, "units": self.units, "context": self.context,
            "seed": self.seed, **self.genes(), "stepping": int(self.halvings),
            "learning": bool(self.learning), "top_down": float(self.top_down),
            "work": dict(self.work),
        }
