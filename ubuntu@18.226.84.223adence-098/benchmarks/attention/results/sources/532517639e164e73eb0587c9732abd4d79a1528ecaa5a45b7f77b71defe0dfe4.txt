"""Carried state as a stimulus: a brain that carries its last equilibrium into the next one.

If settling has a unique attracting fixed point, fully settling the next input
erases the starting state. Multiple attractors and interrupted settling can retain
history, but are not a reliable addressed store. Here a range of *context* neurons,
one per hidden neuron, is stimulated to a leaky trace of the hidden neurons' own activation at
the previous inputs:

    c <- decay * c + (1 - decay) * h

so the state reverberates and fades over about ``1 / (1 - decay)`` inputs. The context
neurons hear nothing (they are a source range of the block transport, so their product is
computed once per settling run) and their synapses into the hidden neurons learn under the
same free/nudged rule as every other synapse. Credit does not flow back through time: the
rule sees the trace as a stimulus and does not differentiate through it.
``Trace`` keeps the trace for a batch of streams.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np

from ._configuration import real
from .brain import Brain as NeuralGraph
from .brain import BrainState
from .connectome import Connectome
from .learning import LearnedState

__all__ = ["Trace", "FastSynapses", "History", "Recall", "columns"]


def _stream_rows(rows: np.ndarray, batch: int) -> np.ndarray:
    """Validate stream indices or a boolean mask before changing any stream state."""
    selected = np.asarray(rows)
    if (
        selected.ndim != 1
        or (selected.size and selected.dtype.kind not in "biu")
        or (selected.dtype.kind == "b" and selected.shape != (batch,))
        or (selected.dtype.kind == "u" and (selected >= batch).any())
    ):
        raise ValueError("rows must be stream indices or a boolean mask matching the batch")
    if not selected.size and selected.dtype.kind != "b":
        return np.zeros(0, dtype=np.int64)
    try:
        return np.asarray(np.arange(batch, dtype=np.int64)[selected], dtype=np.int64)
    except IndexError as exc:
        raise ValueError(
            "rows must be stream indices or a boolean mask matching the batch"
        ) from exc


def columns(index: np.ndarray) -> slice | np.ndarray:
    """A slice when the neurons are one contiguous range, else the index array.

    Reading or writing the columns of a wide batch through a slice is a strided pass;
    through an index array it is a gather that comes back Fortran-ordered (and a scatter
    on write), which costs tens of times more on a batch of thousands of rows.
    """
    if len(index) and np.array_equal(index, np.arange(index[0], index[0] + len(index))):
        return slice(int(index[0]), int(index[0]) + len(index))
    return np.asarray(index, dtype=np.int64)


@dataclass
class Trace:
    """A quantity decaying across moments, kept per stream, entering the next settling run
    as a stimulus: the memory of the moment before.

    Over a range's activation it is the trace of that range, the echo of the interpretation;
    of the input neurons it is an afterimage of the picture itself. ``last`` holds the
    source read of the latest moment and ``cold`` the streams that have not been read yet.
    """

    connectome: Connectome
    decay: float = 0.5
    amplitude: float = 1.0  # the stimulus amplitude of the neuron model
    source: str = "hidden"  # the range traced
    # the range the trace enters as a stimulus, one neuron per source neuron
    target: str = "context"
    trace: np.ndarray = field(init=False)
    last: np.ndarray = field(init=False)
    cold: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.decay < 1:
            raise ValueError("decay lies in [0, 1)")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        self.glow = np.asarray(self.connectome.populations[self.target], dtype=np.int64)
        self.hidden = np.asarray(self.connectome.populations[self.source], dtype=np.int64)
        if len(self.glow) != len(self.hidden):
            raise ValueError(f"one {self.target} neuron per {self.source} neuron")
        self._glow_columns = columns(self.glow)
        self._hidden_columns = columns(self.hidden)
        self.reset(0)

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Fresh for every stream, or for ``rows`` only (streams that begin again)."""
        if rows is None or len(self.trace) != batch:
            self.trace = np.zeros((batch, len(self.hidden)))
            self.last = np.zeros((batch, len(self.hidden)))
            self.cold = np.ones(batch, dtype=bool)
        else:
            self.trace[rows] = 0.0
            self.last[rows] = 0.0
            self.cold[rows] = True

    def keep(self, rows: np.ndarray) -> None:
        """Keep the traces of ``rows`` only (streams that ended are dropped)."""
        self.trace, self.last, self.cold = self.trace[rows], self.last[rows], self.cold[rows]

    def stimulate(self, drive: np.ndarray) -> np.ndarray:
        """Read the trace into a drive copy; another batch size reads zero context."""
        out = np.array(drive, dtype=float)
        out[:, self._glow_columns] = (
            self.amplitude * self.trace if len(self.trace) == len(out) else 0.0
        )
        return out

    def update(self, state: BrainState) -> None:
        """After the free phase: the trace decays toward the source neurons' activation."""
        h = self._source_activation(state)
        if len(self.trace) != len(h):
            self.reset(len(h))
        self.trace = self.decay * self.trace + (1.0 - self.decay) * h
        self.last = h.copy()  # own the previous moment even if the caller reuses state storage
        self.cold[:] = False

    def _source_activation(self, state: BrainState) -> np.ndarray:
        """The source range's activation as a host array. A state that rests on a torch device
        is sliced there and only the slice comes to the host: for an afterimage of a retina
        of thousands of neurons that is the trace's whole cost."""
        device = getattr(state, "device", None)
        s = device.get("s") if isinstance(device, dict) else None
        if s is not None and hasattr(s, "device") and hasattr(s, "cpu"):
            cols = self._hidden_columns
            if isinstance(cols, slice):
                part = s[:, cols]
            else:
                import torch

                part = s[:, torch.as_tensor(np.asarray(cols), device=s.device)]
            return np.ascontiguousarray(part.detach().to("cpu").numpy().astype(float))
        return np.ascontiguousarray(np.atleast_2d(state.activation)[:, self._hidden_columns])

    def ringing(self, floor: float = 0.1) -> np.ndarray:
        """Each source neuron's trace magnitude over the row's mean magnitude, plus ``floor``;
        one for every other neuron: a salience for the eligibility of the synapses
        out of them, so that what is still ringing is what a signal writes through
        (``ActorCritic.salience``)."""
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("floor must be finite and nonnegative")
        out = np.ones((len(self.trace), self.connectome.n))
        magnitude = np.abs(self.trace)
        mean = magnitude.mean(axis=1, keepdims=True) + 1e-9
        out[:, self._hidden_columns] = floor + magnitude / mean
        return out

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "decay": self.decay,
            "amplitude": self.amplitude,
            "source": self.source,
            "target": self.target,
            "neurons": int(len(self.glow)),
        }

    # -- the memory contract (``cadence.memories``)

    def stimulus(self, drive: np.ndarray, moment: Any = None) -> np.ndarray:
        """The trace read into a drive copy (``stimulate``)."""
        return self.stimulate(drive)

    def branch(self) -> Trace:
        """A private copy for an imagined branch, its arrays its own."""
        branch = copy(self)
        for name in ("trace", "last", "cold"):
            setattr(branch, name, getattr(self, name).copy())
        return branch

    def begin(self, keep: bool = False) -> None:
        """The streams begin afresh."""
        self.reset(0)

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps."""
        return {name: getattr(self, name) for name in ("trace", "last", "cold")}


@dataclass
class Efference(Trace):
    """The corollary discharge: a Trace written from the command the brain issued.

    One ``efference`` neuron per motor neuron carries the fading one-hot of the actions the
    stream executed, and the next moment reads it through the plastic projection from the
    ``efference`` range into the association region. The working trace copies the settled
    state before the decision, so under identical observations it carries the executed
    action only through whatever margin the motor competition left, and nothing of a
    sampled choice; this trace carries the command itself. The write is fixed, as the
    working trace's is; what the brain does with its own last command is learned.
    """

    source: str = "motor"
    target: str = "efference"

    def update(self, state: BrainState) -> None:
        raise TypeError("the efference copy is written by the issued command; use issue")

    def issue(self, command: np.ndarray) -> None:
        """After the decision: the trace decays toward the ``(batch, motor)`` command, one
        row per stream with ones at the motor neurons of the executed action."""
        command = np.asarray(command, dtype=float)
        if command.ndim != 2 or command.shape[1] != len(self.hidden):
            raise ValueError("a command is one row of motor values per stream")
        if not np.isfinite(command).all():
            raise ValueError("a command must be finite")
        if len(self.trace) != len(command):
            self.reset(len(command))
        self.trace = self.decay * self.trace + (1.0 - self.decay) * command
        self.last = command.copy()
        self.cold[:] = False


@dataclass
class History:
    """The observations as heard, carried at several timescales in a region of their own.

    One block of ``target`` neurons per decay, each block one neuron per ``source`` neuron.
    After each act the block of decay ``d`` moves toward the observation the brain acted on,
    ``h <- d * h + (1 - d) * x``: the values the ``source`` neurons were stimulated with,
    before any settling. The next moment reads ``amplitude * h`` as the stimulus of the
    block's own neurons, so at a decision the history holds the past and the sensory drive
    holds the present. A fast block keeps the last moments apart, a slow one keeps an
    average of many. The write is fixed, as the working trace's is; the association region
    reads the blocks through a plastic projection, so what the past means is learned. This
    is the pattern of the efference copy applied to the senses: in a closed loop the brain
    hears its own output as its next observation.
    """

    connectome: Connectome
    decays: tuple[float, ...]
    amplitude: float = 3.0
    source: str = "sensory"  # the range heard
    target: str = "history"  # one block of source width per decay
    trace: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        decays = tuple(self.decays)
        if not decays or any(
            isinstance(d, (bool, np.bool_))
            or not isinstance(d, (int, float, np.integer, np.floating))
            or not 0 <= d < 1
            for d in decays
        ):
            raise ValueError("decays are one or more real values in [0, 1)")
        self.decays = tuple(float(d) for d in decays)
        if (
            isinstance(self.amplitude, (bool, np.bool_))
            or not isinstance(self.amplitude, (int, float, np.integer, np.floating))
            or not np.isfinite(self.amplitude)
            or self.amplitude < 0
        ):
            raise ValueError("amplitude must be a finite nonnegative real scalar")
        self.amplitude = float(self.amplitude)
        self.glow = np.asarray(self.connectome.populations[self.target], dtype=np.int64)
        self.width = len(self.connectome.populations[self.source])
        if len(self.glow) != len(self.decays) * self.width:
            raise ValueError(f"one block of {self.source} width per decay in {self.target}")
        self._glow_columns = columns(self.glow)
        # the decay of every history neuron: block by block
        self._decay = np.repeat(np.asarray(self.decays), self.width)
        self.reset(0)

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Silence for every stream, or for ``rows`` only (streams that begin again)."""
        if rows is None or len(self.trace) != batch:
            self.trace = np.zeros((batch, len(self.glow)))
        else:
            self.trace[rows] = 0.0

    def keep(self, rows: np.ndarray) -> None:
        """Keep the histories of ``rows`` only (streams that ended are dropped)."""
        self.trace = self.trace[rows]

    def stimulate(self, drive: np.ndarray) -> np.ndarray:
        """Read the history into a drive copy; another batch size reads silence."""
        out = np.array(drive, dtype=float)
        out[:, self._glow_columns] = (
            self.amplitude * self.trace if len(self.trace) == len(out) else 0.0
        )
        return out

    def hear(self, observations: np.ndarray) -> None:
        """After the act: every block decays toward the ``(batch, source)`` observations the
        streams acted on; another batch size begins the streams in silence first."""
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.width:
            raise ValueError(f"observations are one row of {self.width} values per stream")
        if not np.isfinite(x).all():
            raise ValueError("observations must be finite")
        if len(self.trace) != len(x):
            self.reset(len(x))
        heard = np.tile(x, (1, len(self.decays)))
        self.trace = self._decay * self.trace + (1.0 - self._decay) * heard

    def to_dict(self) -> dict[str, Any]:
        return {
            "decays": list(self.decays),
            "amplitude": self.amplitude,
            "source": self.source,
            "target": self.target,
            "neurons": int(len(self.glow)),
        }

    # -- the memory contract (``cadence.memories``)

    def stimulus(self, drive: np.ndarray, moment: Any = None) -> np.ndarray:
        """The history read into a drive copy (``stimulate``)."""
        return self.stimulate(drive)

    def branch(self) -> History:
        """A private copy for an imagined branch, its trace its own."""
        branch = copy(self)
        branch.trace = self.trace.copy()
        return branch

    def begin(self, keep: bool = False) -> None:
        """The streams begin afresh."""
        self.reset(0)

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps."""
        return {"trace": self.trace}


class _Moment(NamedTuple):
    """The present moment of every stream as the recall memory addresses it."""

    event: np.ndarray  # the observations as given, to recognise the moment at its write
    cortex: np.ndarray  # the settled association activity of the moment before
    address: np.ndarray  # the address q, computed before the settle
    key: np.ndarray  # the fixed part of the address, the slot's key once it is written


@dataclass
class Recall:
    """The index memory, a held-input recall: each stream's recent moments as slots, the
    address of a moment computed before its settle and the recalled drive held through it.

    Each moment the brain accepts (an act, a moment ``wait`` senses) is written after its
    settle as one slot per stream (``witness``). The slot's key is ``heard * x``: ``x`` is
    the observation as given (a brain gives its observation code). The slot's value is the
    event ``x``, and the slot before it learns its next: the value of the moment that
    followed it, zero until that moment comes. Before each settle the present moment's
    address

        q = heard * x + x W_x + c W_c + b

    (``c`` the settled association activity of the moment before) weighs the stream's slots
    once, ``p_i = softmax_i(q . K_i / temperature - recency * age_i)`` (kept as ``attended``;
    with a ``null`` slot of logit ``null / temperature`` in the softmax, which reads nothing, the
    slots' shares sum to one less the null slot's), and
    ``stimulate`` sets the drive of the ``target`` region to ``amplitude * sum_i p_i V_i``
    and, with ``next``, of the ``next`` region to ``amplitude * sum_i p_i N_i``. Both
    regions are sources: their drive is held through the settle as a sensory drive is, so
    the settle answers on the mixture the address chose and does not revise the choice, and
    the address of a moment cannot depend on what that moment recalls (a second recall takes
    a second moment). Their one-way plastic synapses into the association region learn by
    the settle's own contrast, as the sensory ones do. With ``W_x``, ``W_c`` and ``b`` at
    zero the address is the event heard: a moment that shares an event with a stored one
    finds it before any lesson.

    ``learn`` is the address lesson, a rule of its own beside the graph's contrast, which the
    brain runs after teacher lessons (at the founder ``address_reward=0`` rewards leave the
    map as it is). After a lesson it reads the association's movement between the nudged
    phases back through the recall and next synapses, ``d``, scores each slot's value and
    next against it,
    ``(d_r . V_i + d_n . N_i) / |d|`` (the stored values themselves, not their effect
    through the source neurons), weighs the scores with the attention the address gave the
    slots, ``p* = softmax(score / address_temperature + log p)``, and takes the
    least-mean-squares step of the map toward ``K* = sum_i p*_i K_i`` on ``z = [x, c, 1]``
    at the taught rows, each row at the rate ``min(address, 1 / |z|^2)``; a step whose
    result is not finite is refused. The founder step (``address_step="lms"``) takes the
    error ``K* - q``. With ``address_step="toward"`` the error is ``K* - sum_i p_i K_i``, the
    named keys less the mixture the address attended to (``p`` over the slots alone, a null
    slot left out, as in the prior): to first order the step raises the logits of the slots
    the scores name against those the address attended to, whatever the keys' norms.
    ``capacity`` bounds the slots per stream, the oldest dropped first. ``reset`` empties
    streams, and an empty stream reads silence. ``end`` ends their episodes and keeps their
    slots: the latest slot of each learns no successor from the next episode's first moment,
    and the moment before begins afresh; the brain ends an episode so under
    ``Brain.reset(keep=("slots",))`` and empties the stream when an episode ends in a life.

    With ``window`` k above 1 (a construction setting, with ``window_decay`` d) the key's
    fixed part is an ordered window of the stream's last k events, ``[heard * x_t, d * heard
    * x_{t-1}, ..., d^(k-1) * heard * x_{t-k+1}]``, each position on coordinates of its own:
    ``x_{t-j}`` the events of the stream's previous moments, acts and moments ``wait`` senses,
    written or not, zeros before its first moment and after an episode ends. The address
    takes the key's width, ``k * inputs``; the map reads ``[x_t, c, 1]`` as before.

    With ``address_reward`` above zero reward trains the map as well: at each sampled act
    the step the address lesson would take at each row, were the act's action the lesson's
    label, is that row's eligibility (``contrast_rows``), which the actor keeps in a trace
    of its own beside the synapses' and steps by ``address_reward`` times the dopamine
    (``propose`` and ``accept``). The eligibility is a lesson's step at the lesson's bounded
    rate ``min(address, 1 / |z|^2)``: at ``address_reward=1`` a unit of dopamine moves the
    map by one lesson's step, and ``address=0`` leaves the map as it is under reward too.

    The settings of 0.84.3 that this version removed (``RETIRED``, and ``recency=None``) are
    refused with a message naming that release; a checkpoint of 0.84.3 carries the keys its
    format always wrote for three of them, which load at their one value (``FORMAT``).
    """

    connectome: Connectome
    amplitude: float = 3.0
    temperature: float = 0.25
    heard: float = 1.0
    recency: float = 0.25
    capacity: int = 64
    next: bool = True
    address: float = 0.01
    address_temperature: float = 0.1
    # the options beside the founder memory (``OPTIONS``)
    address_step: str = "lms"
    null: float | None = None
    address_reward: float = 0.0
    window: int = 1
    window_decay: float = 1.0
    source: str = "sensory"  # the observations' range
    target: str = "recall"  # the recall region; the next region is ``next``
    # the width of the observations the memory is given, when it is not the source range's: a
    # brain whose memories key on a code of the observation passes the code
    inputs: int = 0
    writes: int = 0  # slots written over the memory's life, one per stream and moment
    keys: np.ndarray = field(init=False, repr=False)
    values: np.ndarray = field(init=False, repr=False)
    nexts: np.ndarray = field(init=False, repr=False)
    held: np.ndarray = field(init=False, repr=False)
    previous: np.ndarray = field(init=False, repr=False)
    w_x: np.ndarray = field(init=False, repr=False)
    w_c: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)

    STEPS = ("lms", "toward")
    # the settings ``Brain.retune`` may change in a life, and those that shape the memory
    RUNTIME = ("amplitude", "temperature", "heard", "recency", "address", "address_temperature")
    CONSTRUCTION = ("capacity", "next")
    # options beside the founder memory: a memory describes and saves an option only where it
    # differs from its founder, and a saved memory without one takes its founder; each is
    # runtime except the window, which shapes the memory
    OPTIONS = ("address_step", "null", "address_reward", "window", "window_decay")
    CONSTRUCTION_OPTIONS = ("window", "window_decay")
    # the settings of 0.84.3 this version removed, refused by name with a message naming it
    RETIRED = ("write", "key", "episodes", "value", "context", "context_decay")
    # the keys the checkpoint format of 0.84.3 (``cadence-generic/8`` and ``/9``) always wrote
    # for removed settings, at the one value at which such a file loads
    FORMAT = MappingProxyType({"value": "event", "context": 0.0, "context_decay": 0.5})

    @classmethod
    def founders(cls) -> dict[str, Any]:
        """The founder value of every setting the memory always describes: the genes of
        ``Brain.compose(..., recall=True)``; ``options()`` gives the options beside them."""
        defaults = {item.name: item.default for item in fields(cls)}
        return {name: defaults[name] for name in (*cls.RUNTIME, *cls.CONSTRUCTION)}

    @classmethod
    def options(cls) -> dict[str, Any]:
        """The founder value of every option (``OPTIONS``), a gene ``recall_<option>`` of
        ``Brain.compose(..., recall=True)`` as well, which a memory describes and saves only
        where it differs from its founder."""
        defaults = {item.name: item.default for item in fields(cls)}
        return {name: defaults[name] for name in cls.OPTIONS}

    @staticmethod
    def removed(name: str) -> str:
        """The refusal of a setting this version removed: ``recall_<name>`` of ``RETIRED``, or
        ``recall_recency=None``."""
        setting = (
            "recall_recency=None, the power-law age charge,"
            if name == "recency"
            else f"recall_{name}"
        )
        return f"{setting} was removed in this version; cadence-net 0.84.3 takes it"

    @staticmethod
    def setting(name: str, value: Any) -> Any:
        """One setting, validated and converted; ``name`` is a field of the memory. A
        setting this version removed is refused by name."""
        if name in Recall.RETIRED or (name == "recency" and value is None):
            raise ValueError(Recall.removed(name))
        if name in ("amplitude", "recency", "address"):
            return real(name, value, low=0)
        if name in ("temperature", "address_temperature"):
            number = real(name, value, low=0)
            if number <= 0:
                raise ValueError(f"{name} must be positive")
            return number
        if name == "heard":
            return real(name, value)
        if name == "capacity":
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError("capacity must be an integer of at least 1")
            return int(value)
        if name == "next":
            if not isinstance(value, bool):
                raise ValueError("next must be boolean")
            return value
        if name == "address_step":
            if not isinstance(value, str) or value not in Recall.STEPS:
                raise ValueError(f"address_step is one of {Recall.STEPS}")
            return value
        if name == "null":
            return None if value is None else real(name, value)
        if name == "address_reward":
            return real(name, value, low=0)
        if name == "window":
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError("window must be an integer of at least 1")
            return int(value)
        if name == "window_decay":
            number = real(name, value, low=0)
            if not 0 < number <= 1:
                raise ValueError("window_decay must lie in (0, 1]")
            return number
        raise TypeError(f"unknown recall setting: {name!r}")

    @staticmethod
    def null_logit(null: float | None, temperature: float) -> float | None:
        """The null slot's logit ``null / temperature``, or None without a null slot; a pair
        of settings whose logit is not finite is refused."""
        if null is None:
            return None
        logit = float(null) / float(temperature)
        if not np.isfinite(logit):
            raise ValueError("null / temperature must be finite")
        return logit

    def __post_init__(self) -> None:
        for name in (*self.RUNTIME, *self.CONSTRUCTION, *self.OPTIONS):
            setattr(self, name, self.setting(name, getattr(self, name)))
        self.null_logit(self.null, self.temperature)
        populations = self.connectome.populations
        regions = (self.source, "association", self.target, *(("next",) if self.next else ()))
        for name in regions:
            if name not in populations:
                raise ValueError(f"a recall memory needs a population named {name!r}")
        self.observed: np.ndarray = np.asarray(populations[self.source], dtype=np.int64)
        self.association: np.ndarray = np.asarray(populations["association"], dtype=np.int64)
        self.recall_neurons: np.ndarray = np.asarray(populations[self.target], dtype=np.int64)
        self.next_neurons: np.ndarray = (
            np.asarray(populations["next"], dtype=np.int64) if self.next else np.zeros(0, np.int64)
        )
        if (
            isinstance(self.inputs, (bool, np.bool_))
            or not isinstance(self.inputs, (int, np.integer))
            or self.inputs < 0
        ):
            raise ValueError("inputs must be a nonnegative integer; 0 reads the source range")
        self.inputs = int(self.inputs)
        self.key_width = self.inputs or len(self.observed)
        self.value_width = self.key_width  # the values are the events
        if len(self.recall_neurons) != self.value_width or (
            self.next and len(self.next_neurons) != self.value_width
        ):
            raise ValueError(
                f"the {self.target} and next regions need one neuron per value coordinate "
                f"({self.value_width})"
            )
        width = self.key_width
        self.address_width = self.window * width  # the window's k events
        self.w_x = np.zeros((width, self.address_width))
        self.w_c = np.zeros((len(self.association), self.address_width))
        self.b = np.zeros(self.address_width)
        self._recall_columns = columns(self.recall_neurons)
        self._next_columns = columns(self.next_neurons)
        self._association_columns = columns(self.association)
        self._recall_synapses = self._synapses(self.recall_neurons)
        self._next_synapses = self._synapses(self.next_neurons)
        self.last: _Moment | None = None
        self._attended: np.ndarray | None = None  # the attention of the latest read
        self._private = False  # an imagined branch reads the slots and writes none
        self.reset(0)

    # -- layout and per-stream state

    @property
    def neurons(self) -> int:
        """The neurons the memory drives: the recall region, and the next region with ``next``."""
        return int(len(self.recall_neurons) + len(self.next_neurons))

    @property
    def streams(self) -> int:
        return int(len(self.keys))

    @property
    def slots(self) -> int:
        """The slots every stream holds (a stream that began since holds fewer, ``held``)."""
        return int(self.keys.shape[1])

    def ages(self) -> np.ndarray:
        """Moments since each slot was written: ``slots - 1`` for the oldest, 0 the latest."""
        return np.arange(self.slots - 1, -1, -1, dtype=float)

    @property
    def attended(self) -> np.ndarray | None:
        """The attention ``p`` the latest read gave each stream's slots, ``(streams, slots)``,
        as that read used it (recency included) and before the moment was written, read-only;
        None before a read, when the read found no slots, and once the streams changed."""
        return self._attended

    def _empty(self, batch: int) -> None:
        self.keys = np.zeros((batch, 0, self.address_width))
        self.values = np.zeros((batch, 0, self.value_width))
        self.nexts = np.zeros((batch, 0, self.value_width))
        self.held = np.zeros((batch, 0), dtype=bool)
        self.previous = np.zeros((batch, len(self.association)))
        # per stream: its latest slot closed an episode (``end``) and learns no successor
        self.ended = np.zeros(batch, dtype=bool)
        # the window's previous events per stream, the latest first; none: zeros
        self.past = np.zeros((0, self.window - 1, self.key_width))
        self.last = None
        self._attended = None

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Empty every stream, or the streams of ``rows`` only (streams that begin again): their
        slots, the cortex of the moment before and the window's events. The address map is
        learned and stays."""
        if rows is not None:
            rows = _stream_rows(rows, batch)
        if rows is None or self.streams != batch:
            self._empty(batch)
            return
        chosen = np.zeros(batch, dtype=bool)
        chosen[rows] = True
        if chosen.all():
            self._empty(batch)
            return
        if not chosen.any():
            return
        keep = ~chosen
        # new arrays, never writes into the old ones: a caller may hold them to restore
        held = self.held & keep[:, None]
        keys = np.where(keep[:, None, None], self.keys, 0.0)
        values = np.where(keep[:, None, None], self.values, 0.0)
        nexts = np.where(keep[:, None, None], self.nexts, 0.0)
        # slots no stream holds any more are dropped from the oldest end
        lead = held.any(axis=0)
        first = int(np.argmax(lead)) if lead.any() else held.shape[1]
        self.keys, self.values, self.nexts, self.held = (
            a[:, first:].copy() for a in (keys, values, nexts, held)
        )
        self._fresh(keep)
        self.ended = self.ended & keep
        self.last = None
        self._attended = None

    def _fresh(self, keep: np.ndarray) -> None:
        """The moment before begins afresh for the streams ``keep`` does not mark: the cortex
        of the moment before and the window's events."""
        batch = len(keep)
        self.previous = np.where(keep[:, None], self.previous, 0.0)
        if len(self.past) == batch:
            self.past = np.where(keep[:, None, None], self.past, 0.0)

    def end(self, batch: int, rows: np.ndarray | None = None) -> None:
        """The streams of ``rows`` (every stream without) end an episode and keep their
        slots: the latest slot of each learns no successor from the stream's next written
        moment, and the moment before (the cortex of the moment before and the window's
        events) begins afresh. The
        address map is learned and stays. Another batch size empties the streams, as
        ``reset`` does."""
        if rows is not None:
            rows = _stream_rows(rows, batch)
        if self.streams != batch:
            self._empty(batch)
            return
        chosen = np.zeros(batch, dtype=bool)
        chosen[slice(None) if rows is None else rows] = True
        if not chosen.any():
            return
        # new arrays, never writes into the old ones: a caller may hold them to restore
        self.ended = self.ended | chosen
        self._fresh(~chosen)
        self.last = None
        self._attended = None

    def keep(self, rows: np.ndarray) -> None:
        """Keep the memories of ``rows`` only (streams that ended are dropped)."""
        rows = _stream_rows(rows, self.streams)
        self.keys, self.values, self.nexts, self.held = (
            a[rows] for a in (self.keys, self.values, self.nexts, self.held)
        )
        self.previous = self.previous[rows]
        if len(self.past):
            self.past = self.past[rows]
        self.ended = self.ended[rows]
        self.last = None
        self._attended = None

    def snapshot(self) -> dict[str, Any]:
        """The per-stream state, to restore after a refused operation; every write replaces
        these arrays rather than changing them."""
        return {
            name: getattr(self, name)
            for name in ("keys", "values", "nexts", "held", "previous", "ended", "past")
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Return to a ``snapshot``; the next moment is addressed afresh."""
        for name, value in snapshot.items():
            setattr(self, name, value)
        self.last = None
        self._attended = None

    def reading(self) -> tuple[_Moment | None, np.ndarray | None]:
        """The latest read as ``witness``, ``learn`` and ``contrast_rows`` take it: the moment
        it addressed (``last``, None once the moment is written) and the attention it gave the
        slots (``attended``). ``restore``, ``end`` and ``reset`` address the next moment
        afresh; a read taken on the state they leave is held through them with ``hold``."""
        return self.last, self._attended

    def hold(self, reading: tuple[_Moment | None, np.ndarray | None]) -> None:
        """Hold a ``reading`` as the latest read: one taken on the state the streams are in.
        A forecast at an episode's end reads the streams as the end leaves them before the
        end is committed; once it is, that read is the moment's, so the act it decides keeps
        the read's eligibility and ``attended`` describes it."""
        self.last, self._attended = reading

    # -- the memory contract (``cadence.memories``)

    def stimulus(self, drive: np.ndarray, moment: Any) -> np.ndarray:
        """The moment's address read into a drive copy (``stimulate`` with its code)."""
        return self.stimulate(drive, moment.code)

    def branch(self) -> Recall:
        """A private copy for an imagined branch (``imagined``)."""
        return self.imagined()

    def begin(self, keep: bool = False) -> None:
        """The streams begin afresh: with ``keep`` each ends an episode and its slots stay
        (``end``), else the slots are emptied; the address map is learned and stays."""
        if keep:
            self.end(self.streams)
        else:
            self.reset(0)

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps: the slots, the address map, the streams whose latest
        slot closed an episode (while one does) and the window's previous events."""
        out = {
            name: getattr(self, name)
            for name in ("keys", "values", "nexts", "held", "previous", "w_x", "w_c", "b")
        }
        if self.ended.any():
            out["ended"] = self.ended
        if self.window > 1:
            out["past"] = self.past
        return out

    def imagined(self) -> Recall:
        """A private copy for an imagined branch: it reads the live slots and writes none; the
        cortex of the moment before and the window's events follow the branch."""
        branch = copy(self)
        branch.previous = self.previous.copy()
        branch.past = self.past.copy()
        branch.last = None
        branch._attended = None
        branch._private = True
        return branch

    # -- the moment's address and read

    def _port(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.key_width:
            raise ValueError(f"observations are one row of {self.key_width} values per stream")
        if not np.isfinite(x).all():
            raise ValueError("observations must be finite")
        return x

    def _moment(self, event: np.ndarray) -> _Moment:
        """The address of the present moment of each stream, from the cortex of the moment
        before."""
        x = np.asarray(event, dtype=float)
        n = len(x)
        c = (
            self.previous
            if len(self.previous) == n
            else np.zeros((n, len(self.association)))
        )
        key = self.heard * x
        if self.window > 1:
            # the stream's previous events in order, each on coordinates of its own
            past = (
                self.past
                if len(self.past) == n
                else np.zeros((n, self.window - 1, self.key_width))
            )
            scale = self.heard * self.window_decay ** np.arange(1, self.window)
            key = np.concatenate([key, (scale[None, :, None] * past).reshape(n, -1)], axis=1)
        q = key + x @ self.w_x + c @ self.w_c + self.b
        return _Moment(x, c, q, key)

    def _attention(
        self, address: np.ndarray, keys: np.ndarray, held: np.ndarray | None
    ) -> np.ndarray:
        """The logits of the attention over slots: ``address . key / temperature`` less
        ``recency`` times each slot's age, ``-inf`` at a slot its stream does not hold."""
        logits = np.einsum("bk,bmk->bm", address, keys) / self.temperature
        if self.recency:
            logits = logits - np.asarray(self.recency * self.ages())[None, :]
        return logits if held is None else np.where(held, logits, -np.inf)

    def weights(self, address: np.ndarray) -> np.ndarray | None:
        """``p`` over each stream's slots for the ``(batch, key)`` addresses, or None when the
        streams hold no slot or are another batch. With a ``null`` slot, whose logit
        ``null / temperature`` joins the softmax and which reads nothing, the slots' shares
        sum to one less the null slot's."""
        if not self.slots or self.streams != len(address):
            return None
        if self.null is not None:
            logits = self._attention(address, self.keys, None if self.held.all() else self.held)
            null = self.null / self.temperature
            top = np.maximum(logits.max(axis=1), null)  # a stream that holds no slot: the null
            p = np.exp(logits - top[:, None])
            total = p.sum(axis=1) + np.exp(null - top)
            return np.asarray(p / total[:, None])
        if self.held.all():
            logits = self._attention(address, self.keys, None)
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            return np.asarray(p / p.sum(axis=1, keepdims=True))
        # a stream that began since holds fewer slots; one that holds none reads silence
        logits = self._attention(address, self.keys, self.held)
        empty = ~self.held.any(axis=1)
        top = np.where(empty, 0.0, logits.max(axis=1))
        p = np.exp(logits - top[:, None])
        total = p.sum(axis=1, keepdims=True)
        return np.asarray(p / np.where(total > 0, total, 1.0))

    def stimulate(self, drive: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """Address the present moment of every stream and read its slots into a drive copy:
        ``amplitude * sum_i p_i V_i`` on the recall region and, with ``next``,
        ``amplitude * sum_i p_i N_i`` on the next region. ``observations`` are ``(batch,
        inputs)``. Streams without slots, or of another batch, read silence. The moment is
        kept for its write and its lesson, and the attention the read used as ``attended``."""
        moment = self._moment(self._port(observations))
        self.last = moment
        out = np.array(drive, dtype=float)
        p = self.weights(moment.address)
        if p is not None:
            p.flags.writeable = False
        self._attended = p  # the attention this read uses, kept past the moment's write
        if p is not None:
            out[:, self._recall_columns] = self.amplitude * np.einsum("bm,bmc->bc", p, self.values)
            if self.next:
                out[:, self._next_columns] = self.amplitude * np.einsum(
                    "bm,bmc->bc", p, self.nexts
                )
        return out

    # -- the write after the settle

    def activity(self, state: BrainState) -> np.ndarray:
        """The association region's settled activation as a host array, ``(batch,
        association)``. A state on a torch device is sliced there, and only the slice comes
        to the host."""
        device = getattr(state, "device", None)
        s = device.get("s") if isinstance(device, dict) else None
        cols = self._association_columns
        if s is not None and hasattr(s, "device") and hasattr(s, "cpu"):
            if isinstance(cols, slice):
                part = s[:, cols]
            else:
                import torch

                part = s[:, torch.as_tensor(np.asarray(cols), device=s.device)]
            return np.ascontiguousarray(part.detach().to("cpu").numpy().astype(float))
        settled = np.atleast_2d(np.asarray(state.activation, dtype=float))
        return np.asarray(settled[:, cols], dtype=float).copy()

    def witness(self, observations: np.ndarray, state: BrainState) -> None:
        """After the settle: the ``(batch, inputs)`` observations the streams acted on join
        their slots under the moment's key, the slot before learns them as its next, and the
        settled cortex of ``state`` and the window move on to this moment. Another batch
        size begins the streams with empty memories first. An imagined branch moves the
        cortex and the window only."""
        x = self._port(observations)
        moment = self.last
        if moment is None or moment.event.shape != x.shape or not np.array_equal(moment.event, x):
            moment = self._moment(x)
        cortex = self.activity(state)
        if len(cortex) != len(x):
            raise ValueError("the settled state needs one row per stream")
        if not self._private:
            self._observe(moment.key, moment.event)
            self.writes += len(x)
        self.previous = cortex
        if self.window > 1:  # this moment enters the window, the oldest leaves it
            past = (
                self.past
                if len(self.past) == len(x)
                else np.zeros((len(x), self.window - 1, self.key_width))
            )
            self.past = np.concatenate([moment.event[:, None, :], past[:, :-1]], axis=1)
        self.last = None

    def _observe(self, key: np.ndarray, value: np.ndarray) -> None:
        """Write one moment per stream: the latest slot learns what followed it, a new slot
        holds this moment; beyond ``capacity`` the oldest slot is dropped. A stream that began
        since the others (``reset`` of its rows) holds fewer slots: its latest held slot,
        wherever it lies, learns this moment as its next. A stream whose latest slot closed
        an episode (``end``) teaches it nothing, and its new slot is open again."""
        if len(key) != self.streams:
            self._empty(len(key))
        nexts = self.nexts
        if self.slots:
            nexts = nexts.copy()
            if self.held.all() and not self.ended.any():
                nexts[:, -1] = value
            else:
                # each stream's latest held slot learns what followed it, unless it closed an
                # episode
                held = self.held
                rows = held.any(axis=1) & ~self.ended
                latest = held.shape[1] - 1 - np.argmax(held[:, ::-1], axis=1)
                index = np.flatnonzero(rows)
                nexts[index, latest[index]] = value[index]
        fresh = np.ones((len(key), 1), dtype=bool)
        if self.ended.any():  # every stream's new slot is open to its successor
            self.ended = np.zeros(len(key), dtype=bool)
        keys = np.concatenate([self.keys, key[:, None, :]], axis=1)
        values = np.concatenate([self.values, value[:, None, :]], axis=1)
        nexts = np.concatenate([nexts, np.zeros_like(value)[:, None, :]], axis=1)
        held = np.concatenate([self.held, fresh], axis=1)
        if keys.shape[1] > self.capacity:
            drop = keys.shape[1] - self.capacity
            keys, values, nexts, held = (a[:, drop:].copy() for a in (keys, values, nexts, held))
        self.keys, self.values, self.nexts, self.held = keys, values, nexts, held

    # -- the address lesson

    def _synapses(self, region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The synapses from ``region`` into the association region: their indices, and the
        source and association coordinate of each."""
        connectome = self.connectome
        pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
        local_source = -np.ones(connectome.n, dtype=np.int64)
        local_source[region] = np.arange(len(region))
        local_cortex = -np.ones(connectome.n, dtype=np.int64)
        local_cortex[self.association] = np.arange(len(self.association))
        synapses = np.flatnonzero((local_source[pre] >= 0) & (local_cortex[post] >= 0))
        return synapses, local_source[pre[synapses]], local_cortex[post[synapses]]

    def readback(self, graph: NeuralGraph, region: str, movement: np.ndarray) -> np.ndarray:
        """``movement @ W.T`` for the one-way synapses of the ``"recall"`` or ``"next"``
        region into the association region: per source neuron, the sum over its synapses of
        effective weight times the movement of its association neuron."""
        synapses, rows, cols = self._recall_synapses if region == "recall" else self._next_synapses
        width = len(self.recall_neurons) if region == "recall" else len(self.next_neurons)
        w = np.zeros((width, len(self.association)))
        np.add.at(w, (rows, cols), graph._gain_pre[synapses] * graph.efficacy[synapses])
        return np.asarray(movement @ w.T)

    def learn(self, graph: NeuralGraph, learned: LearnedState) -> dict[str, float]:
        """The address lesson after a lesson whose phases are ``learned``, at its taught rows,
        through the synapses of ``graph`` (the learner's brain after the lesson's update).
        Reads the moment the lesson's drive addressed (``stimulate``); returns how many rows
        moved, the mean absolute error the step corrects before it (``address_error``: the
        addresses' error ``K* - q`` under ``address_step="lms"``, the attended mixture's
        ``K* - sum_i p_i K_i`` under ``"toward"``), and how many rows a refused step left as
        they were (``address_refused``)."""
        quiet = {"address_rows": 0.0, "address_error": 0.0, "address_refused": 0.0}
        moment = self.last
        if (
            not self.address
            or moment is None
            or learned.free is None
            or learned.nudged is None
            or learned.nudged is learned.free
        ):
            return quiet
        taught = learned.taught
        rows = (
            np.arange(len(moment.address))
            if taught is None
            else np.flatnonzero(np.asarray(taught, dtype=bool))
        )
        if not self.slots or not len(rows) or self.streams != len(moment.address):
            return quiet
        plus = np.atleast_2d(learned.nudged.activation)
        minus = np.atleast_2d(
            learned.free.activation if learned.opposite is None else learned.opposite.activation
        )
        movement = (plus - minus)[:, self.association]
        d_value = self.readback(graph, "recall", movement)
        d_next = self.readback(graph, "next", movement) if self.next else None
        return self._step(moment, rows, d_value, d_next)

    def _step(
        self, moment: _Moment, rows: np.ndarray, d_value: np.ndarray, d_next: np.ndarray | None
    ) -> dict[str, float]:
        """The least-mean-squares step of the address map toward the keys of the slots the
        scores name, on each row's inputs ``z = [x, c, 1]``. A row's rate is ``address``
        bounded by the row's own energy, ``r = min(address, 1 / |z|^2)``: a row taught alone
        moves its own address toward its target, at most onto it. The rows' steps are
        averaged, and each row's step moves the addresses of the rows whose inputs overlap
        its own, so in a batch a row can pass its target. The batch error does not grow: the
        rows' squared distances to their targets, each weighted by the row's rate ``r``, sum
        to no more after the step than before (with equal weights where no row reaches the
        bound). The candidate map and the candidate addresses of the rows are checked before
        the step is kept; one that is not finite refuses the step, the map stays as it was
        and the report counts the refused rows. With ``address_step="toward"`` each row's
        error, the one ``address_error`` reports, is its target less the keys' mixture its
        address attended to."""
        quiet = {"address_rows": 0.0, "address_error": 0.0, "address_refused": 0.0}
        found = self._row_steps(moment, rows, d_value, d_next)
        if found is None:
            return quiet
        _, x, c, key, error, step = found
        n = len(x)
        rate = self.address
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            w_x = rate * x.T @ step
            w_x /= n
            w_x += self.w_x
            w_c = rate * c.T @ step
            w_c /= n
            w_c += self.w_c
            b = self.b + rate * step.mean(axis=0)
            address = key + x @ w_x + c @ w_c + b
        if not all(np.isfinite(a).all() for a in (w_x, w_c, b, address)):
            return {**quiet, "address_refused": float(n)}
        self.w_x, self.w_c, self.b = w_x, w_c, b
        return {
            "address_rows": float(n),
            "address_error": float(np.abs(error).mean()),
            "address_refused": 0.0,
        }

    def _row_steps(
        self, moment: _Moment, rows: np.ndarray, d_value: np.ndarray, d_next: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Each row's step of the address lesson before the average over rows: the rows of
        ``rows`` that move (a movement and a held slot), and at those rows the event ``x``,
        the cortex ``c``, the key, the error toward the named keys and the error bounded by
        the row's own energy, ``error * min(1, 1 / (address |z|^2))``. None when no row
        moves. The arithmetic is unchecked: a caller checks what it keeps."""
        values, nexts, keys = self.values[rows], self.nexts[rows], self.keys[rows]
        held = self.held[rows]
        masked = not held.all()
        score = np.einsum("bc,bmc->bm", d_value, values)
        norm = np.sum(d_value * d_value, axis=1)
        if d_next is not None:
            score += np.einsum("bc,bmc->bm", d_next, nexts)
            norm += np.sum(d_next * d_next, axis=1)
        norm = np.sqrt(norm)
        moving = norm > 1e-12
        if masked:
            moving &= held.any(axis=1)  # a stream that began since and holds no slot
        if not moving.any():
            return None
        x, c, q, key = (
            a[rows][moving] for a in (moment.event, moment.cortex, moment.address, moment.key)
        )
        rate = self.address
        # finite inputs can still overflow the arithmetic below: the check before the step is
        # kept refuses what is not finite
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            logits = score[moving] / norm[moving, None] / self.address_temperature
            if masked:
                logits = np.where(held[moving], logits, -np.inf)
            # among the moments the address attended to: the attention is the prior
            attention = self._attention(q, keys[moving], held[moving] if masked else None)
            attention -= attention.max(axis=1, keepdims=True)
            attention -= np.log(np.exp(attention).sum(axis=1, keepdims=True))
            logits = logits + attention
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            target = np.einsum("bm,bmk->bk", p, keys[moving])
            if self.address_step == "toward":
                # from the mixture the address attended to: moving q by s (K* - K_bar) raises
                # the named slots' mean logit by s |K* - K_bar|^2 / T over the attended ones'
                error = target - np.einsum("bm,bmk->bk", np.exp(attention), keys[moving])
            else:
                error = target - q
            # each row's gain on its own address, rate * |z|^2, is at most one: a row taught
            # alone lands on its target at most, and a row within the bound keeps the rate as
            # it is. Over a batch the step maps the errors E to (I - M) E, M = rate / n * G D
            # with G the rows' Gram matrix of z and D their bounds; D^(1/2) M D^(-1/2) is
            # symmetric with its eigenvalues in [0, 1] (its trace is the mean of the rows'
            # gains), so sum_i D_i |e_i|^2 does not grow, though one row can pass its target.
            energy = np.einsum("bi,bi->b", x, x) + np.einsum("bi,bi->b", c, c) + 1.0
            step = error * np.minimum(1.0, 1.0 / (rate * energy))[:, None]
        return moving, x, c, key, error, step

    # -- reward trains the address map

    def contrast_rows(
        self, graph: NeuralGraph, plus: BrainState, minus: BrainState
    ) -> np.ndarray | None:
        """The address map's eligibility from a sampled act, one ``(inputs + association + 1,
        inputs)`` array per row (the rows of ``w_x``, ``w_c`` and ``b`` in that order), or
        None at ``address_reward=0``: the step the address lesson would take at each row,
        were the act's action a lesson's label, before the average over rows. The
        association's movement between the act's two nudged phases, ``plus`` less ``minus``,
        is read back through the recall and next synapses of ``graph`` and scored against the
        slots of the moment the act read (``last``, before its write); a row contributes
        ``min(address, 1 / |z|^2)`` times the outer product of ``z = [x, c, 1]`` and its
        error, of the form ``address_step`` names. A row that does not move, or whose
        contribution is not finite, keeps none."""
        if not self.address_reward:
            return None
        movement = self.activity(plus) - self.activity(minus)
        batch, width = len(movement), self.key_width
        out = np.zeros((batch, width + len(self.association) + 1, self.address_width))
        moment = self.last
        if moment is None or not self.slots or self.streams != batch or len(moment.key) != batch:
            return out
        d_value = self.readback(graph, "recall", movement)
        d_next = self.readback(graph, "next", movement) if self.next else None
        found = self._row_steps(moment, np.arange(batch), d_value, d_next)
        if found is None:
            return out
        moving, x, c, _, _, step = found
        z = np.concatenate([x, c, np.ones((len(x), 1))], axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            rows = self.address * np.einsum("bi,bk->bik", z, step)
        kept = np.isfinite(rows).all(axis=(1, 2))
        out[np.flatnonzero(moving)[kept]] = rows[kept]
        return out

    def propose(
        self, step: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        """The address map after a reward's step, not installed: ``step`` is the rows'
        eligibility weighted by the dopamine and averaged (``contrast_rows``' shape), taken at
        ``address_reward``. Returns ``w_x``, ``w_c``, ``b`` and the norm of the step, or None
        at ``address_reward=0``, which takes none; a step whose map is not finite is refused
        (``ValueError``)."""
        if not self.address_reward:
            return None
        step = np.asarray(step, dtype=float)
        width = self.key_width
        shape = (width + len(self.association) + 1, self.address_width)
        if step.shape != shape:
            raise ValueError(f"an address map's step is a {shape} array")
        with np.errstate(over="ignore", invalid="ignore"):
            moved = self.address_reward * step
            w_x, w_c, b = self.w_x + moved[:width], self.w_c + moved[width:-1], self.b + moved[-1]
        if not all(np.isfinite(a).all() for a in (moved, w_x, w_c, b)):
            raise ValueError("the address map's reward step must be finite")
        return w_x, w_c, b, float(np.linalg.norm(moved))

    def accept(self, proposal: tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> None:
        """Install a step ``propose`` made."""
        self.w_x, self.w_c, self.b = proposal[0], proposal[1], proposal[2]

    def to_dict(self) -> dict[str, Any]:
        """The memory's settings and counts: the founders, the ports, and each option where it
        differs from its founder."""
        out = {
            **{name: getattr(self, name) for name in (*self.RUNTIME, *self.CONSTRUCTION)},
            "source": self.source,
            "target": self.target,
            "neurons": self.neurons,
            "writes": int(self.writes),
        }
        if self.inputs:  # only a memory keyed on a code of the observation records its width
            out["inputs"] = self.inputs
        # an option at its founder is left out of the description
        founders = self.options()
        out.update(
            {name: getattr(self, name) for name in self.OPTIONS
             if getattr(self, name) != founders[name]}
        )
        return out


@dataclass
class PatternSeparator:
    """Pattern separation: expand a key into a wider random code and keep the strongest winners.

    ``inputs`` key neurons project through a fixed random matrix onto ``expansion`` code
    neurons; the ``winners`` largest positive entries stay and the rest are zero. Expansion
    can reduce overlap between correlated keys; it does not guarantee separate codes.
    Codes with disjoint winners do not interfere: a delta write changes another key's read
    in proportion to their dot product. ``center`` > 0 keeps a running mean of the keys
    seen by ``observe`` (forgetting factor ``center``) and subtracts it first. Updating that
    mean can move the code of a previously stored key.

    ``fan_in`` above zero makes the projection sparse: each code neuron reads ``fan_in``
    distinct inputs chosen at random from the seed, with random weights of variance
    ``1 / fan_in``. ``index`` and ``weight``, ``(expansion, fan_in)`` each, hold them in place
    of ``projection``, so the separator's storage and work follow the expansion and the
    fan-in, whatever the number of inputs. The founder ``fan_in=0`` keeps the dense
    ``projection``; ``index`` and ``weight`` are then None, and ``projection`` is None for a
    sparse separator.

    ``cover`` above zero (with a fan-in) reads every input in both signs, through two
    rectified channels: ON ``max(x, 0)``, channel ``i`` of input ``i``, and OFF
    ``max(-x, 0)``, channel ``inputs + i`` (``channels`` is ``2 * inputs``, and ``index``
    numbers channels). Each channel is given ``cover`` distinct code neurons first, each
    code neuron's remaining fan-in is then filled with distinct channels at random, and the
    weights are positive, uniform in ``(0, 1]`` over ``sqrt(fan_in)``. Without ``center``, a
    single input of either sign, the others zero, drives each reader of its channel above
    zero, so it never codes to zero, and a nonnegative key leaves every OFF channel silent.
    It needs ``fan_in`` within the channels, ``cover * 2 * inputs`` within
    ``expansion * fan_in`` and ``cover`` within ``expansion``. The founder ``cover=0`` draws
    the signed sparse weights over the inputs independently, as above.

    The projection is drawn once from the seed and never learns. A brain's ``code=units``
    observation code and its episodic ``memory_expansion`` separators (sparse when
    ``memory_fan_in`` is set) are such fixed random codes; a brain's learned code is the
    region ``learned_code=units`` adds (``cadence.code.LearnedCode``).
    """

    inputs: int
    expansion: int
    winners: int
    seed: int = 0
    center: float = 0.0
    fan_in: int = 0
    cover: int = 0
    projection: np.ndarray | None = field(init=False, repr=False)
    index: np.ndarray | None = field(init=False, repr=False)
    weight: np.ndarray | None = field(init=False, repr=False)
    mean: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("inputs", "expansion", "winners"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        fan_in = self.fan_in
        for name in ("fan_in", "cover"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if fan_in > self.channels:
            raise ValueError(
                "fan_in must not exceed the channels, two per input"
                if self.cover
                else "fan_in must not exceed the inputs"
            )
        if self.winners > self.expansion:
            raise ValueError("winners must not exceed the expansion")
        if not 0 <= self.center < 1:
            raise ValueError("center lies in [0, 1)")
        if self.cover and (
            not fan_in
            or self.cover > self.expansion
            or self.cover * self.channels > self.expansion * fan_in
        ):
            raise ValueError(
                "cover needs a fan-in, at most the expansion, and cover * 2 * inputs (the "
                "channels) within expansion * fan_in"
            )
        rng = np.random.default_rng(self.seed)
        self.projection = self.index = self.weight = None
        if fan_in and self.cover:
            self.index = self._covering(rng)
            self.weight = (1.0 - rng.random((self.expansion, fan_in))) / np.sqrt(fan_in)
        elif fan_in:
            # each code neuron's inputs, distinct and in order, then their weights
            self.index = np.sort(
                np.stack([rng.choice(self.inputs, fan_in, replace=False)
                          for _ in range(self.expansion)]),
                axis=1,
            ).astype(np.int64)
            self.weight = rng.standard_normal((self.expansion, fan_in)) / np.sqrt(fan_in)
        else:
            self.projection = (
                rng.standard_normal((self.inputs, self.expansion)) / np.sqrt(self.inputs)
            )
        self.mean = np.zeros(self.inputs)

    @property
    def channels(self) -> int:
        """The values a code neuron's fan-in is drawn from: the inputs, or with ``cover`` the
        ON and the OFF channel of each input, ``2 * inputs``."""
        return 2 * int(self.inputs) if self.cover else int(self.inputs)

    def _covering(self, rng: np.random.Generator) -> np.ndarray:
        """Each code neuron's channels, distinct and in order, every channel among those of
        ``cover`` distinct code neurons: the readers of the channels are dealt first, from the
        places of every neuron's fan-in in a random order, and each neuron's places left over
        take distinct other channels at random."""
        channels, units, fan_in, cover = self.channels, self.expansion, self.fan_in, self.cover
        places = rng.permutation(np.repeat(np.arange(units), fan_in))
        readers = places[: cover * channels].reshape(channels, cover).copy()
        spare = places[cover * channels:].copy()
        ordered = np.sort(readers, axis=1)
        for row in np.flatnonzero((ordered[:, 1:] == ordered[:, :-1]).any(axis=1)):
            # a neuron dealt twice to one channel trades the place for another neuron's
            given = readers[row]
            for k in range(1, cover):
                for _ in range(64 * (channels + len(spare))):
                    if given[k] not in given[:k]:
                        break
                    open_ = np.flatnonzero(~np.isin(spare, given))
                    if len(open_):
                        s = open_[rng.integers(len(open_))]
                        given[k], spare[s] = spare[s], given[k]
                        continue
                    other, place = rng.integers(channels), rng.integers(cover)
                    if readers[other, place] not in given and given[k] not in readers[other]:
                        given[k], readers[other, place] = readers[other, place], given[k]
                else:
                    raise ValueError("the channels could not be covered")
        owner = readers.ravel()
        order = np.argsort(owner, kind="stable")  # each neuron's given channels, in order
        read = np.repeat(np.arange(channels), cover)[order]
        bounds = np.searchsorted(owner[order], np.arange(units + 1))
        index = np.empty((units, fan_in), dtype=np.int64)
        for unit in range(units):
            given = read[bounds[unit] : bounds[unit + 1]]
            # the remaining places: distinct channels the neuron does not read yet, by rank
            ranks = rng.choice(channels - len(given), fan_in - len(given), replace=False)
            extra = ranks + np.searchsorted(given - np.arange(len(given)), ranks, side="right")
            index[unit] = np.sort(np.concatenate([given, extra]))
        return index

    def habituate(self, keys: np.ndarray) -> None:
        """Initialize the running mean from ``keys`` before storing records.

        With positive ``center``, later learning calls continue updating this mean;
        initialization does not freeze the codes used to address stored records.
        """
        x = np.asarray(keys, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all() or not len(x):
            raise ValueError(f"keys must be a finite nonempty (batch, {self.inputs}) array")
        self.mean = (x / len(x)).sum(axis=0)

    def code(self, key: np.ndarray, learn: bool = False) -> np.ndarray:
        """The sparse code of ``(batch, inputs)`` keys: ``(batch, expansion)``, ``winners``
        nonzero entries per row at most. ``learn`` updates the running mean (writes only).
        With ``cover`` each key is read through its ON and its OFF channels."""
        x = np.asarray(key, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all():
            raise ValueError(f"key must be a finite (batch, {self.inputs}) array")
        mean = self.mean
        if self.center > 0.0:
            if learn and len(x):
                mean = self.center * mean + (1.0 - self.center) * (x / len(x)).sum(axis=0)
            with np.errstate(over="ignore", invalid="ignore"):
                x = x - mean
        with np.errstate(over="ignore", invalid="ignore"):
            if self.fan_in:
                assert self.index is not None and self.weight is not None
                if self.cover:  # the ON channels, then the OFF channels
                    x = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=1)
                drive = np.maximum(np.einsum("bek,ek->be", x[:, self.index], self.weight), 0.0)
            else:
                assert self.projection is not None
                drive = np.maximum(x @ self.projection, 0.0)
        if not np.isfinite(mean).all() or not np.isfinite(drive).all():
            raise ValueError("pattern separation overflowed; scale the input keys")
        if learn:
            self.mean = mean
        if self.winners >= self.expansion:
            return np.asarray(drive)
        keep = np.argpartition(-drive, self.winners - 1, axis=1)[:, : self.winners]
        out: np.ndarray = np.zeros_like(drive)
        np.put_along_axis(out, keep, np.take_along_axis(drive, keep, axis=1), axis=1)
        return np.asarray(out)

    def arrays(self) -> dict[str, np.ndarray]:
        """The separator's state by name, as a checkpoint holds it: ``projection`` and ``mean``,
        or with a fan-in ``index``, ``weight`` and ``mean``."""
        if self.fan_in:
            assert self.index is not None and self.weight is not None
            return {"index": self.index, "weight": self.weight, "mean": self.mean}
        assert self.projection is not None
        return {"projection": self.projection, "mean": self.mean}

    def to_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "inputs": self.inputs,
            "expansion": self.expansion,
            "winners": self.winners,
            "seed": self.seed,
            "center": self.center,
        }
        if self.fan_in:  # a dense separator leaves fan_in out of its description
            out["fan_in"] = int(self.fan_in)
        if self.cover:  # and a separator without cover leaves cover out
            out["cover"] = int(self.cover)
        return out


@dataclass
class FastSynapses:
    """A bounded associative memory between two ranges, one matrix per stream.

    ``rule="hebb"`` (the default) adds an outer product.
    ``rule="delta"`` reads before writing: for a unit key ``k``,
    ``M += rate * outer(k, value - k @ M)``. Repeated evidence then stops changing
    a correct prediction. A rate-one write corrects this key's unscaled prediction; other
    nonorthogonal keys can interfere. This is the delta/LMS rule. Its capacity is bounded,
    and its addressing is fixed: the key itself, or the separator's code of it.

    ``observe``/``recall`` operate directly on key/value ports. ``update``/``read``
    adapt the same operations to settled states and full drives. Key normalisation
    reads the whole key vector; a synapse uses its key coordinate and the post neuron's
    prediction error. Strengths decay once per observation/update, never on reads.
    """

    pre: np.ndarray
    post: np.ndarray
    decay: float = 1.0
    rate: float = 1.0
    amplitude: float = 1.0
    normalize: bool = False  # unit keys and cue, the read divided by the decayed count of writes
    replace: bool = False  # a write clears what its active pre neurons held (a slot; one-hot keys)
    rule: str = "hebb"  # "delta": unit keys, residual writes, no count-averaged read
    separator: PatternSeparator | None = None  # pattern separation of the keys before use
    strength: np.ndarray = field(init=False)
    mass: np.ndarray = field(init=False)  # (batch,) the decayed count of writes, for ``normalize``
    writes: int = 0

    def __post_init__(self) -> None:
        for name in ("pre", "post"):
            raw = np.asarray(getattr(self, name))
            if (
                raw.dtype.kind not in "iuf"
                or not np.isfinite(raw).all()
                or (raw != np.floor(raw)).any()
            ):
                raise ValueError(f"{name} must contain integer neuron indices")
        self.pre = np.asarray(self.pre, dtype=np.int64)
        self.post = np.asarray(self.post, dtype=np.int64)
        for name, neurons in (("pre", self.pre), ("post", self.post)):
            if neurons.ndim != 1 or not len(neurons) or np.any(neurons < 0):
                raise ValueError(f"{name} must be a nonempty vector of neuron indices")
            if len(np.unique(neurons)) != len(neurons):
                raise ValueError(f"{name} neuron indices must be unique")
        if not 0 <= self.decay <= 1:
            raise ValueError("decay lies in [0, 1]")
        if not np.isfinite(self.rate) or self.rate < 0 or not np.isfinite(self.amplitude):
            raise ValueError("rate must be finite and nonnegative; amplitude must be finite")
        if self.rule not in ("hebb", "delta"):
            raise ValueError("rule must be 'hebb' or 'delta'")
        if self.rule == "delta":
            if self.rate > 1:
                raise ValueError("delta rate lies in [0, 1]")
            if self.normalize or self.replace:
                raise ValueError(
                    "delta already normalises keys; normalize/replace are Hebbian modes"
                )
        if self.separator is not None and self.separator.inputs != len(self.pre):
            raise ValueError("the separator's inputs must equal the number of pre neurons")
        self._pre_columns = columns(self.pre)
        self._post_columns = columns(self.post)
        self.strength = np.zeros((0, self.key_width, len(self.post)))
        self.mass = np.zeros(0)

    @property
    def key_width(self) -> int:
        """Rows of the strength matrix: the code width with a separator, else the pre count."""
        return self.separator.expansion if self.separator is not None else int(len(self.pre))

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Clear all streams, or only the selected streams at episode boundaries."""
        if rows is None or len(self.strength) != batch:
            self.strength = np.zeros((batch, self.key_width, len(self.post)))
            self.mass = np.zeros(batch)
        else:
            self.strength[rows] = 0.0
            self.mass[rows] = 0.0

    def keep(self, rows: np.ndarray) -> None:
        self.strength = self.strength[rows]
        self.mass = self.mass[rows]

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        # The Hebbian normalisation, with a small-norm floor.
        return np.asarray(x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))

    @staticmethod
    def _delta_unit(x: np.ndarray) -> np.ndarray:
        # Scale first so even very small/large finite keys have a well-defined direction.
        scale = np.max(np.abs(x), axis=-1, keepdims=True)
        scaled = x / np.where(scale > 0, scale, 1.0)
        norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
        return np.asarray(scaled / np.where(norm > 0, norm, 1.0))

    @staticmethod
    def _port(values: np.ndarray, width: int, name: str) -> np.ndarray:
        out = np.asarray(values, dtype=float)
        if out.ndim != 2 or out.shape[1] != width or not np.isfinite(out).all():
            raise ValueError(f"{name} must be a finite (batch, {width}) array")
        return out

    def recall(self, key: np.ndarray) -> np.ndarray:
        """Read ``(batch, pre)`` keys as ``(batch, post)`` values without changing memory.

        A different batch size reads the baseline without erasing live streams. Delta uses
        unit keys and does not divide by the number of writes. ``amplitude``
        scales the returned drive, not the residual used when learning a value.
        """
        cue = self._port(key, len(self.pre), "key")
        same_batch = len(self.strength) == len(cue)
        strength = self.strength if same_batch else self._baseline()[None, :, :]
        if self.separator is not None:
            cue = self.separator.code(cue)
        if self.rule == "delta":
            cue = self._delta_unit(cue)
        elif self.normalize:
            cue = self._unit(cue)
        out = (cue[:, None, :] @ strength)[:, 0, :]
        if self.normalize:
            mass = self.mass if same_batch else np.zeros(len(cue))
            out = out / np.maximum(mass, 1e-12)[:, None]
        return np.asarray(self.amplitude * out)

    def _baseline(self) -> np.ndarray:
        """The unobserved stream's readback, without changing any stored stream."""
        return np.zeros((self.key_width, len(self.post)))

    def observe(self, key: np.ndarray, value: np.ndarray, write: np.ndarray | None = None) -> None:
        """Decay once, then write selected rows from the key and observed-value ports.

        Inputs are ``(batch, pre)`` and ``(batch, post)``. ``write=None`` writes
        every row; a boolean mask gates individual streams. A zero key cannot
        write a delta association. Validation happens before any state change.
        For prediction tasks call ``recall`` before revealing the new value.
        """
        key = self._port(key, len(self.pre), "key")
        value = self._port(value, len(self.post), "value")
        batch = len(key)
        if len(value) != batch:
            raise ValueError("key and value batches must match")
        gate = np.ones(batch, dtype=bool) if write is None else np.asarray(write)
        if gate.shape != (batch,) or gate.dtype != np.bool_:
            raise ValueError("write must be a boolean vector with one entry per stream")
        same_batch = len(self.strength) == batch
        strength = (
            self.strength * self.decay
            if same_batch
            else np.zeros((batch, self.key_width, len(self.post)))
        )
        mass = self.mass * self.decay if same_batch else np.zeros(batch)
        if self.rule == "delta":
            gate = gate & np.any(key != 0.0, axis=1)
        rows = np.flatnonzero(gate)
        if not len(rows):
            self.strength, self.mass = strength, mass
            return
        a, b = key[rows], value[rows]
        previous_mean = None if self.separator is None else self.separator.mean
        if self.separator is not None:
            a = self.separator.code(a, learn=True)
        if self.rule == "delta":
            a = self._delta_unit(a)
        elif self.normalize:
            a = self._unit(a)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.rule == "delta":
                b = b - (a[:, None, :] @ strength[rows])[:, 0, :]
            elif self.replace:
                strength[rows] *= (a <= 0.0)[:, :, None]
            strength[rows] += self.rate * a[:, :, None] * b[:, None, :]
            mass[rows] += self.rate
        if not np.isfinite(strength).all() or not np.isfinite(mass).all():
            if self.separator is not None and previous_mean is not None:
                self.separator.mean = previous_mean
            raise ValueError("synaptic update overflowed; scale the observed keys and values")
        self.strength, self.mass = strength, mass
        self.writes += len(rows)

    def read(self, drive: np.ndarray) -> np.ndarray:
        """The post neurons' drive from the pre range's stimulus in ``drive``: ``(batch, post)``.

        Normalized Hebbian mode uses unit keys and cue, then divides the read by the
        decayed count of writes. Cosine weights can be signed, so this need not be a
        convex average. ``rate`` and ``amplitude`` also scale the returned values.
        """
        return self.recall(drive[:, self._pre_columns])

    def stimulate(
        self, drive: np.ndarray, inplace: bool = False, key: np.ndarray | None = None
    ) -> np.ndarray:
        """Add the read to the post columns of ``drive`` (a copy, unless ``inplace``): the read
        of the pre columns of ``drive``, or with ``key`` the read of that ``(batch, pre)``
        key, for a memory keyed on a code of the observation rather than on neurons."""
        out = drive if inplace else np.array(drive, dtype=float)
        out[:, self._post_columns] += self.read(out) if key is None else self.recall(key)
        return out

    # -- the memory contract (``cadence.memories``)

    def stimulus(self, drive: np.ndarray, moment: Any) -> np.ndarray:
        """The read added to a drive copy: under the moment's ``place`` key, or of the drive
        itself without one; nothing where the stores do not read into the drive."""
        if not moment.stores:
            return drive
        if moment.place is None:  # an instrument may wrap the drive's own read
            return self.stimulate(drive)
        return self.stimulate(drive, key=moment.place)

    def branch(self) -> FastSynapses:
        """An imagined branch reads the records and writes none: the memory itself."""
        return self

    def begin(self, keep: bool = False) -> None:
        """The streams begin afresh; the records stay."""

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps: the ports, strengths, masses and the separator's
        tables."""
        out = {name: getattr(self, name) for name in ("pre", "post", "strength", "mass")}
        if self.separator is not None:
            out.update({"separator/" + k: v for k, v in self.separator.arrays().items()})
        return out

    def update(
        self,
        state: BrainState,
        write: np.ndarray | None = None,
        post: np.ndarray | None = None,
    ) -> None:
        """After settling: strengths fade once; the rows in ``write`` store the settled pre
        activity as the key and the post activity as the value (``observe``). With
        ``write=None`` no row writes.

        ``post`` replaces the post neurons' activations for the write, ``(batch, post)``: what
        actually followed (the next symbols read) rather than what the net settled on.
        """
        s = np.atleast_2d(state.activation)
        self.observe(
            s[:, self._pre_columns],
            s[:, self._post_columns] if post is None else post,
            np.zeros(len(s), dtype=bool) if write is None else write,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre": int(len(self.pre)),
            "post": int(len(self.post)),
            "decay": self.decay,
            "rate": self.rate,
            "amplitude": self.amplitude,
            "normalize": self.normalize,
            "replace": self.replace,
            "rule": self.rule,
            "separator": None if self.separator is None else self.separator.to_dict(),
            "writes": self.writes,
        }
