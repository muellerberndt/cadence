"""The index memory: the brain addresses its own recent moments; what it finds drives the settle.

The fixed read of ``token_binding`` recalls the stored event that shares a token with the
present one and generalizes to bindings never trained; it cannot find a moment the present
shares nothing with. A read inside the settle, a learned query and a recall competing for the
stored moments through one energy, starts its query and recall at rest and completes on a
stored moment nearly at random. Here the address of each moment is computed before
its settle, from what the brain has at that point, and the recall enters the settle as a
drive held through it, as the fixed read's does:

    q_t = heard * x_t + x_t W_x + c_{t-1} W_c + b        the moment's address
    p_t = softmax_i(q_t . K_i / T - recency * age_i)       over the stream's slots
    recall drive = alpha sum_i p_i V_i,   next drive = alpha sum_i p_i N_i

``x_t`` is the observation, ``c_{t-1}`` the settled association activity of the moment
before. After the settle the moment is written: key ``K_t`` the fixed part of its address
(``heard * x_t``, plus ``context`` times the fading history of the events before it), value
``V_t`` the event (``value="event"``) or the settled association activity
(``value="cortex"``), and the previous slot's next ``N_{t-1} = V_t``. At ``W = 0`` the
address is the event heard: a question that shares an event with a stored moment finds it
from the first lesson. The recall and next neurons are sources with one-way synapses into the
association region: their drive is held through the settle, which answers on the mixture the
address chose and does not revise the choice, and the settle's contrast teaches those
synapses like the sensory ones. A moment's address cannot depend on what that moment
recalls: a second step takes a second moment (a think event). With ``recency=None`` a slot's
age costs ``r * log(1 + age_i)``, ``r = heard^2 / T / log(1 + capacity)``, and nothing with an
unbounded capacity. With ``write="aroused"`` a moment is written for the streams whose act was
chosen aroused alone (``IndexBrain.aroused_rows``, or a sampled act); with a number ``null`` a
null slot of logit ``null / T`` joins the softmax and reads nothing.

The address map ``W_x, W_c, b`` learns by the address lesson after every lesson, a rule of
its own beside the graph's contrast. The association's movement between the nudged phases is
read back through the recall's and the next's own synapses,
``d = W_{recall->association} (c_+ - c_-)``, and each slot scores ``(d_r . V_i + d_n . N_i) /
|d|``, the stored values themselves against that movement. The scores are weighed with the
moment's own attention ``p``, ``p* = softmax(score / address_temperature + address_prior *
log p)``; ``K* = sum_i p*_i K_i`` is the target, and the map takes the least-mean-squares
step toward ``K*`` on its inputs ``z = [x_t, c_{t-1}, 1]`` at the taught rows, each row at
the rate ``min(address, 1 / |z|^2)``, so that a row taught alone moves its own address at
most onto its target. The rows' steps are averaged: in a batch a row can pass its target,
and the batch error, the rows' squared errors weighted by their rates, does not grow. A
step whose result is not finite is refused. With ``address_step="toward"`` the step's error
is ``K* - sum_i p_i K_i``, the named keys less the mixture the moment attended to, in place
of ``K* - q``: to first order the named slots' logits rise against the attended ones. With the
attention as the prior, a moment whose own content happens to be the answer does not displace
the moment the address found by its relation (a fresh successor is the value of the moment
after the cued one). Keys hold the fixed part of the address: a key written with the learned
map carries the map's own output, and the lesson then moves the map toward targets that
contain it. Nothing outside the brain names the
moment; the world supplies only the answer it already supplies.

Instrument level; nothing here changes the library.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, motor_cortex, prefrontal_cortex

HERE = Path(__file__).resolve().parent


def _sibling(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tb = _sibling("token_binding")

GENES: dict[str, Any] = {
    "alpha": 3.0,  # the recall's drive of a slot holding the whole weight (the fixed read's)
    "temperature": 0.25,  # the softmax temperature over the slots
    "heard": 1.0,  # the present event's weight in its own address
    "recall_scale": 1.2,  # initial magnitude of the recall and next projections (the fixed read's)
    "recency": 0.0,  # logit cost per moment of a slot's age; None: r log(1 + age), see charges
    "next": True,  # the next neurons and each slot's successor value
    "value": "event",  # "event": the event as witnessed; "cortex": the association activity
    "capacity": 0,  # most slots per stream, the oldest dropped first; 0: no bound
    "address": 0.0,  # the address lesson's rate; 0: the address stays the event heard
    "address_temperature": 0.1,  # the softmax temperature of the answer's address
    "address_prior": 1.0,  # weight of the moment's own attention in naming the answer's slot
    "address_step": "lms",  # "toward": the error is K* less the attended keys' mixture
    "write": "always",  # "aroused": a moment is written for the streams that acted aroused
    "null": None,  # a number: a null slot of logit null / T that reads nothing
    "context": 0.0,  # weight of the fading history of earlier events in a moment's key
    "context_decay": 0.5,  # the history's retention per moment
    "code": 0,  # code neurons of the observation's sparse code; 0: the observation itself
    "code_fan_in": 16,  # inputs each code neuron reads
    "code_winners": 24,  # code neurons a code keeps
}
VALUES = ("event", "cortex")
STEPS = ("lms", "toward")
WRITES = ("always", "aroused")


class SlotMemory:
    """Each stream's recent moments as slots: keys ``(streams, slots, key_width)``, values and
    nexts ``(streams, slots, value_width)``; slot ``-1`` is the latest, its next still zero.
    ``held`` ``(streams, slots)`` marks the slots each stream wrote: a moment written for some
    streams alone is a column every stream has, held by the written ones."""

    def __init__(self, key_width: int, value_width: int, capacity: int = 0) -> None:
        if capacity < 0:
            raise ValueError("capacity is a nonnegative slot count")
        self.key_width, self.value_width = int(key_width), int(value_width)
        self.capacity = int(capacity)
        self.reset(0)

    def reset(self, batch: int) -> None:
        """``batch`` streams without slots."""
        b = int(batch)
        self.keys = np.zeros((b, 0, self.key_width))
        self.values = np.zeros((b, 0, self.value_width))
        self.nexts = np.zeros((b, 0, self.value_width))
        self.held = np.zeros((b, 0), dtype=bool)

    @property
    def streams(self) -> int:
        return int(len(self.keys))

    @property
    def slots(self) -> int:
        return int(self.keys.shape[1])

    def ages(self) -> np.ndarray:
        """Moments since each slot was written: ``slots - 1`` for the oldest, 0 the latest."""
        return np.arange(self.slots - 1, -1, -1, dtype=float)

    def observe(self, key: np.ndarray, value: np.ndarray,
                written: np.ndarray | None = None) -> None:
        """Write one moment per stream: the latest slot learns what followed it, a new slot
        holds this moment. A new batch size starts fresh streams. With ``written``, one flag
        per stream, the column is held by the written streams alone (zeros elsewhere), and
        each written stream's latest held slot learns this moment as its next."""
        key, value = np.asarray(key, dtype=float), np.asarray(value, dtype=float)
        if key.ndim != 2 or key.shape[1] != self.key_width or not np.isfinite(key).all():
            raise ValueError(f"keys must be a finite (batch, {self.key_width}) array")
        if value.shape != (len(key), self.value_width) or not np.isfinite(value).all():
            raise ValueError(f"values must be a finite (batch, {self.value_width}) array")
        if written is not None and (written.dtype != np.bool_ or written.shape != (len(key),)):
            raise ValueError("written is one flag per stream")
        if len(key) != self.streams:
            self.reset(len(key))
        if self.slots:
            if written is None and self.held.all():
                self.nexts[:, -1] = value
            else:
                held = self.held
                rows = held.any(axis=1) if written is None else held.any(axis=1) & written
                latest = held.shape[1] - 1 - np.argmax(held[:, ::-1], axis=1)
                index = np.flatnonzero(rows)
                self.nexts[index, latest[index]] = value[index]
        fresh = np.ones((len(key), 1), dtype=bool)
        if written is not None:
            fresh = written[:, None].copy()
            key, value = np.where(fresh, key, 0.0), np.where(fresh, value, 0.0)
        self.keys = np.concatenate([self.keys, key[:, None, :]], axis=1)
        self.values = np.concatenate([self.values, value[:, None, :]], axis=1)
        self.nexts = np.concatenate([self.nexts, np.zeros_like(value)[:, None, :]], axis=1)
        self.held = np.concatenate([self.held, fresh], axis=1)
        if self.capacity and self.slots > self.capacity:
            drop = self.slots - self.capacity
            self.keys, self.values, self.nexts, self.held = (
                a[:, drop:].copy() for a in (self.keys, self.values, self.nexts, self.held)
            )

    def take(self, other: SlotMemory, rows: np.ndarray | None = None) -> None:
        """This memory becomes a copy of ``other``'s streams (``rows`` of them, in order)."""
        arrays = (other.keys, other.values, other.nexts, other.held)
        if rows is not None:
            arrays = tuple(a[rows] for a in arrays)
        self.keys, self.values, self.nexts, self.held = (a.copy() for a in arrays)


class SparseCode:
    """The library's sparse code of an observation for wide inputs: ``units`` code neurons
    with positive weights reading each input through two rectified channels, ON ``max(x, 0)``
    and OFF ``max(-x, 0)``, every channel read by two of them at least, each reading
    ``fan_in`` channels (more where ``fan_in`` cannot read every channel twice); the
    ``winners`` largest sums stay, the code scaled to unit length. Storage and work follow
    the units and the fan-in, whatever the number of inputs. The construction is
    ``PatternSeparator`` with ``cover=2``, as ``Brain.compose(code=...)`` builds it, and its
    seed is drawn from the founder's on a stream of its own, ``SeedSequence([seed, 7103])``,
    as the library draws the brain's: the founder's seed itself is the stream ``develop``
    builds the brain from and the brain acts with."""

    COVER = 2
    STREAM = 7103

    def __init__(self, inputs: int, units: int, fan_in: int, winners: int, seed: int) -> None:
        if not 0 < fan_in <= inputs or not 0 < winners <= units:
            raise ValueError("0 < fan_in <= inputs and 0 < winners <= units")
        from cadence.stream import PatternSeparator

        channels = 2 * inputs  # an ON and an OFF channel per input
        fan_in = min(channels, max(int(fan_in), -(-self.COVER * channels // units)))
        own = int(np.random.SeedSequence([seed, self.STREAM]).generate_state(1)[0])
        self.separator = PatternSeparator(inputs, units, winners, seed=own, fan_in=fan_in,
                                          cover=self.COVER)
        self.units, self.winners = int(units), int(winners)

    @property
    def index(self) -> np.ndarray:
        """The channels each code neuron reads, ``(units, fan_in)``: ``i`` the ON channel of
        input ``i``, ``inputs + i`` its OFF channel."""
        index = self.separator.index
        assert index is not None
        return index

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.separator.code(np.atleast_2d(np.asarray(x, dtype=float)))
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return np.asarray(out / np.where(norm > 0.0, norm, 1.0))


class IndexMemory:
    """The slots of each stream and the learned address map of a moment."""

    def __init__(self, code_width: int, cortex_width: int, value_width: int,
                 genes: dict[str, Any]) -> None:
        self.genes = genes
        self.heard = float(genes["heard"])
        self.temperature = float(genes["temperature"])
        self.recency = None if genes["recency"] is None else float(genes["recency"])
        charge = [] if self.recency is None else [self.recency]
        if not np.isfinite([self.heard, self.temperature, *charge]).all():
            raise ValueError("heard, temperature and recency must be finite")
        if self.temperature <= 0:
            raise ValueError("the temperature must be positive")
        self.slots = SlotMemory(code_width, value_width, int(genes["capacity"]))
        self.w_x = np.zeros((code_width, code_width))
        self.w_c = np.zeros((cortex_width, code_width))
        self.b = np.zeros(code_width)
        self.cortex_width = int(cortex_width)
        self.previous = np.zeros((0, cortex_width))  # each stream's settled association, t - 1
        self.history = np.zeros((0, code_width))  # each stream's fading history of earlier events
        self.context = float(genes["context"])
        self.context_decay = float(genes["context_decay"])
        # the options beside the declared genes; genes recorded before them take their founders
        self.address_step = str(genes.get("address_step", "lms"))
        if self.address_step not in STEPS:
            raise ValueError(f"address_step is one of {STEPS}")
        self.writing = str(genes.get("write", "always"))
        if self.writing not in WRITES:
            raise ValueError(f"write is one of {WRITES}")
        null = genes.get("null")
        self.null = None if null is None else float(null)
        logit = [] if self.null is None else [self.null, self.null / self.temperature]
        if not np.isfinite(logit).all():
            raise ValueError("null and its logit null / temperature must be finite")
        # (x, c, q, k) of the present moment: event, cortex before it, address, key
        self.last: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._attended: np.ndarray | None = None  # the attention of the latest read

    @property
    def attended(self) -> np.ndarray | None:
        """The attention the latest read used over each stream's slots, recency included,
        recorded before the moment's write; read-only, None before a read."""
        return self._attended

    def charges(self) -> np.ndarray:
        """The logit each slot's age costs it, oldest first: ``recency * age``, or with
        ``recency=None`` the power-law charge ``r * log(1 + age)``, ``r = heard^2 /
        temperature / log(1 + capacity)``, nothing with an unbounded capacity."""
        ages = self.slots.ages()
        if self.recency is None:
            capacity = self.slots.capacity
            exponent = self.heard**2 / self.temperature / np.log1p(capacity) if capacity else 0.0
            return np.asarray(exponent * np.log1p(ages))
        return np.asarray(self.recency * ages)

    def reset(self) -> None:
        self.slots.reset(0)
        self.previous = np.zeros((0, self.cortex_width))
        self.history = np.zeros((0, self.slots.key_width))
        self.last = None
        self._attended = None

    def address(self, x: np.ndarray) -> np.ndarray:
        """The address of the present moment of each stream; kept for its write and lesson.

        The key is the fixed part: the event heard and, with ``context``, the fading history of
        the events before it. The address adds the learned map of the event and the cortex of
        the moment before."""
        x = np.asarray(x, dtype=float)
        n = len(x)
        c = self.previous if len(self.previous) == n else np.zeros((n, self.cortex_width))
        key = self.heard * x
        if self.context:
            h = self.history if len(self.history) == n else np.zeros_like(x)
            key = key + self.context * h
        q = key + x @ self.w_x + c @ self.w_c + self.b
        self.last = (x, c, q, key)
        return q

    def weights(self, q: np.ndarray) -> np.ndarray | None:
        """``p`` over each stream's slots for the addresses ``q``, or None without slots. A
        slot its stream does not hold weighs nothing; with a ``null`` slot the slots' shares
        sum to one less the null slot's."""
        memory = self.slots
        if not memory.slots or memory.streams != len(q):
            return None
        logits = np.einsum("bk,bmk->bm", q, memory.keys) / self.temperature
        if self.recency is None or self.recency:
            logits = logits - self.charges()[None, :]
        held = memory.held
        if self.null is not None:
            logits = logits if held.all() else np.where(held, logits, -np.inf)
            null = self.null / self.temperature
            top = np.maximum(logits.max(axis=1), null)
            p = np.exp(logits - top[:, None])
            total = p.sum(axis=1) + np.exp(null - top)
            return np.asarray(p / total[:, None])
        if not held.all():
            logits = np.where(held, logits, -np.inf)
            top = np.where(~held.any(axis=1), 0.0, logits.max(axis=1))
            p = np.exp(logits - top[:, None])
            total = p.sum(axis=1, keepdims=True)
            return np.asarray(p / np.where(total > 0, total, 1.0))
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        return np.asarray(p / p.sum(axis=1, keepdims=True))

    def read(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """The recalled value and next of each stream: ``sum_i p_i V_i``, ``sum_i p_i N_i``;
        the attention ``p`` stays as ``attended``."""
        p = self.weights(q)
        if p is not None:
            p.flags.writeable = False
        self._attended = p
        if p is None:
            return None
        memory = self.slots
        return (np.einsum("bm,bmc->bc", p, memory.values),
                np.einsum("bm,bmc->bc", p, memory.nexts))

    def write(self, value: np.ndarray, cortex: np.ndarray, rows: np.ndarray | None = None) -> None:
        """The present moment joins the slots under its key, for the streams of ``rows`` alone
        when given (one flag per stream; no stream written leaves the slots as they were); the
        cortex and the history move on to it."""
        assert self.last is not None, "a moment is written after its address"
        x, _, _, key = self.last
        rows = None if rows is None else np.asarray(rows, dtype=bool)
        if rows is None or rows.all():
            self.slots.observe(key, value)
        elif rows.any():
            self.slots.observe(key, value, rows)
        elif len(x) != self.slots.streams:
            self.slots.reset(len(x))
        self.previous = np.asarray(cortex, dtype=float).copy()
        if self.context:
            h = self.history if len(self.history) == len(x) else np.zeros_like(x)
            self.history = self.context_decay * h + x

    def learn(self, rows: np.ndarray, d_value: np.ndarray, d_next: np.ndarray | None,
              rate: float, temperature: float) -> dict[str, float]:
        """The address lesson at ``rows``: toward the keys of the slots the scores name, each
        row's rate bounded by its own energy, ``min(rate, 1 / |z|^2)`` on ``z = [x, c, 1]``,
        the rows' steps averaged. A candidate map or address that is not finite refuses the
        step: the map stays and ``address_refused`` counts the rows. With
        ``address_step="toward"`` a row's error, the one ``address_error`` reports, is its
        target less the attended keys' mixture."""
        assert self.last is not None
        quiet = {"address_rows": 0.0, "address_error": 0.0, "address_refused": 0.0}
        memory = self.slots
        if not memory.slots or not len(rows):
            return quiet
        values, nexts, keys = memory.values[rows], memory.nexts[rows], memory.keys[rows]
        held = memory.held[rows]
        masked = not held.all()
        score = np.einsum("bc,bmc->bm", d_value, values)
        norm = np.sum(d_value * d_value, axis=1)
        if d_next is not None:
            score += np.einsum("bc,bmc->bm", d_next, nexts)
            norm += np.sum(d_next * d_next, axis=1)
        norm = np.sqrt(norm)
        moving = norm > 1e-12
        if masked:
            moving &= held.any(axis=1)  # a stream that wrote nothing yet holds no slot
        if not moving.any():
            return quiet
        x, c, q, key = (a[rows][moving] for a in self.last)
        n = len(q)
        toward = self.address_step == "toward"
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            logits = score[moving] / norm[moving, None] / temperature
            if masked:
                logits = np.where(held[moving], logits, -np.inf)
            prior = float(self.genes["address_prior"])
            if prior or toward:
                # among the moments the address attended to: the attention is the prior
                attention = np.einsum("bk,bmk->bm", q, keys[moving]) / self.temperature
                if self.recency is None or self.recency:
                    attention = attention - self.charges()[None, :]
                if masked:
                    attention = np.where(held[moving], attention, -np.inf)
                attention -= attention.max(axis=1, keepdims=True)
                attention -= np.log(np.exp(attention).sum(axis=1, keepdims=True))
                if prior:
                    logits = logits + prior * attention
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            target = np.einsum("bm,bmk->bk", p, keys[moving])
            if toward:
                # from the mixture the moment attended to, toward the keys the scores name
                error = target - np.einsum("bm,bmk->bk", np.exp(attention), keys[moving])
            else:
                error = target - q
            # each row's gain on its own address, rate * |z|^2, at most one
            energy = np.einsum("bi,bi->b", x, x) + np.einsum("bi,bi->b", c, c) + 1.0
            step = error * np.minimum(1.0, 1.0 / (rate * energy))[:, None]
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
        return {"address_rows": float(n), "address_error": float(np.abs(error).mean()),
                "address_refused": 0.0}


class IndexBrain(tb.ContentBrain):
    """The composed brain plus recall and next regions driven by its index memory.
    ``build(seed)`` without ``read`` is the composed brain; ``inputs``, ``actions``,
    ``lateral`` and ``slots`` put it on another task's ports."""

    index: IndexMemory | None = None
    # with ``write="aroused"``: the streams a caller marks aroused for one act, cleared by the
    # caller after it; unset, every stream of a sampled act and none of a greedy one
    aroused_rows: np.ndarray | None = None
    index_recall: np.ndarray | None = None
    index_next: np.ndarray | None = None
    association_index: np.ndarray | None = None
    genes: dict[str, Any] | None = None
    code: SparseCode | None = None

    @classmethod
    def build(  # type: ignore[override]
        cls,
        seed: int,
        *,
        width: int = tb.WIDTH,
        read: dict[str, Any] | None = None,
        inputs: int = tb.INPUTS,
        actions: int = tb.VALUES,
        lateral: float = tb.LATERAL,
        slots: int | tuple[int, ...] = 1,
        **options: Any,
    ) -> IndexBrain:
        genes = None if read is None else dict(GENES, **read)
        if genes is not None and genes["value"] not in VALUES:
            raise ValueError(f"value is one of {VALUES}")
        regions = [
            Region("sensory", inputs),
            Region("association", width),
            prefrontal_cortex(width),
            motor_cortex(actions, lateral=lateral, slots=slots),
        ]
        projections = [
            Projection("sensory", "association", scale=1.0, reciprocal=False),
            Projection("association", "motor"),
            Projection("prefrontal", "association", scale=1.2, reciprocal=False),
        ]
        value_width = code_width = inputs
        if genes is not None:
            code_width = int(genes["code"]) or inputs
            value_width = code_width if genes["value"] == "event" else width
            regions.append(Region("recall", value_width))
            projections.append(Projection("recall", "association", scale=genes["recall_scale"],
                                          reciprocal=False))
            if genes["next"]:
                regions.append(Region("next", value_width))
                projections.append(Projection("next", "association",
                                              scale=genes["recall_scale"], reciprocal=False))
        genome = Genome(tuple(regions), tuple(projections), label="composed-brain")
        brain = cls(develop(genome, seed=seed), seed=seed, slots=slots, **options)
        if genes is not None:
            populations = brain.connectome.populations
            brain.index_recall = np.asarray(populations["recall"], dtype=np.int64)
            if genes["next"]:
                brain.index_next = np.asarray(populations["next"], dtype=np.int64)
            brain.association_index = np.asarray(populations["association"], dtype=np.int64)
            brain.index = IndexMemory(code_width, width, value_width, genes)
            brain.genes = genes
            if genes["code"]:
                brain.code = SparseCode(inputs, int(genes["code"]), int(genes["code_fan_in"]),
                                        int(genes["code_winners"]), seed)
        return brain

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        drive = super().stimulus(observations, memory=memory)
        if memory and self.index is not None:
            x = np.asarray(self._observations(observations), dtype=float)
            found = self.index.read(self.index.address(self.coded(x)))
            if found is not None:
                alpha = float(self.index.genes["alpha"])
                drive[:, self.index_recall] = alpha * found[0]
                if self.index_next is not None:
                    drive[:, self.index_next] = alpha * found[1]
        return drive

    def _choose(self, x: np.ndarray, drive: np.ndarray, free: Any, **kwargs: Any) -> np.ndarray:
        action = super()._choose(x, drive, free, **kwargs)
        if self.index is not None:
            s = np.atleast_2d(np.asarray(free.activation, dtype=float))
            cortex = s[:, self.association_index]
            event = self.index.genes["value"] == "event"
            value = self.coded(np.asarray(x, dtype=float)) if event else cortex
            rows = None
            if self.index.writing == "aroused":
                marked = self.aroused_rows
                rows = (np.full(len(cortex), not kwargs.get("greedy", False)) if marked is None
                        else np.asarray(marked, dtype=bool))
            self.index.write(value, cortex, rows)
        return action

    def coded(self, x: np.ndarray) -> np.ndarray:
        """The observation as the memory addresses it: itself, or its sparse code."""
        return x if self.code is None else self.code(x)

    def reset(self) -> None:
        super().reset()
        if self.index is not None:
            self.index.reset()

    def readback(self, region: np.ndarray, movement: np.ndarray) -> np.ndarray:
        """``movement @ W.T`` for a source region's one-way synapses into the association:
        per source neuron, the sum over its synapses of weight times the post's movement."""
        assert self.association_index is not None
        graph = self.learner.brain
        connectome = graph.connectome
        pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
        local_source = -np.ones(connectome.n, dtype=np.int64)
        local_source[region] = np.arange(len(region))
        local_assoc = -np.ones(connectome.n, dtype=np.int64)
        local_assoc[self.association_index] = np.arange(len(self.association_index))
        joins = (local_source[pre] >= 0) & (local_assoc[post] >= 0)
        w = np.zeros((len(region), len(self.association_index)))
        np.add.at(w, (local_source[pre[joins]], local_assoc[post[joins]]),
                  graph._gain_pre[joins] * graph.efficacy[joins])
        return np.asarray(movement @ w.T)


def address_lesson(brain: IndexBrain, learned: Any, rows: np.ndarray) -> dict[str, float]:
    """The address lesson from a lesson's phases at the taught ``rows`` of the batch."""
    genes, index = brain.genes, brain.index
    assert genes is not None and index is not None and brain.association_index is not None
    if learned is None or learned.nudged is None or learned.nudged is learned.free:
        return {"address_rows": 0.0, "address_error": 0.0, "address_refused": 0.0}
    plus = np.atleast_2d(learned.nudged.activation)
    minus = np.atleast_2d(
        learned.free.activation if learned.opposite is None else learned.opposite.activation
    )
    movement = (plus - minus)[:, brain.association_index]
    assert brain.index_recall is not None
    d_value = brain.readback(brain.index_recall, movement)
    d_next = None if brain.index_next is None else brain.readback(brain.index_next, movement)
    return index.learn(rows, d_value, d_next, float(genes["address"]),
                       float(genes["address_temperature"]))


def lesson(brain: Any, x: np.ndarray, target: np.ndarray, work: dict[str, Any]) -> np.ndarray:
    """``token_binding.lesson`` plus the address lesson; returns the answer judged."""
    from cadence.learning import LearningPhaseError

    drive = brain.stimulus(x)
    answer = brain._choices(brain._qualified(drive, brain._activity(), operation="act"))
    judged = answer if brain.learner.config.surprise else None
    work["lessons"] += 1
    work["lesson_rows"] += len(target)
    try:
        learned, report = brain.learner.step(drive, target, answer=judged)
    except LearningPhaseError:
        work["lesson_refusals"] += 1
        return np.asarray(answer)
    work["rows_taught"] += int(report["accepted_presentations"])
    work["lesson_sweeps"] += int(report["total_steps"])
    work["lessons_exhausted"] = work.get("lessons_exhausted", 0) + int(
        report.get("free_budget_exhausted", 0.0)
    )
    genes = getattr(brain, "genes", None)
    if genes and genes["address"] and report.get("accepted_presentations", 0):
        taught = getattr(learned, "taught", None)
        rows = (np.arange(len(target)) if taught is None
                else np.flatnonzero(np.asarray(taught, dtype=bool)))
        step = address_lesson(brain, learned, rows)
        work["address_rows"] = work.get("address_rows", 0) + int(step["address_rows"])
        work["address_refused"] = work.get("address_refused", 0) + int(step["address_refused"])
    return np.asarray(answer)
