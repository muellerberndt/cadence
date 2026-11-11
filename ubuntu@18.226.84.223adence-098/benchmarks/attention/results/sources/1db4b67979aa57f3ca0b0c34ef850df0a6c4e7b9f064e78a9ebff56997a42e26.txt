"""Experimental reciprocal memory equations, outside the supported Brain interface.

Witnessed slots are immutable boundary data. Query and read are live neuronal
activations. The agreement candidate uses E = a/2 |r - R(q)|², where R is
the softmax mixture of witnessed values (and successors). Both drives are the
negative derivatives of that SAME energy. In particular the query drive uses
the actual softmax Jacobian, including its subtractive mean and null slot.
The historical default masks this energy off for an empty stream. Optional
``empty_anchor=True`` retains the same agreement with the zero-content null:
E = a/2 |r|², read drive -a r, and zero query drive when no record is held.

The linear control uses E = -a q M r with M = K.T V / held_count.
The competitive negative control uses -a T logsumexp((q K + r V)/T).
Its positive feedback can invent a confident recollection without evidence.

No backpropagation update is installed: existing tied graph synapses receive
their local free/nudged contrast. This CPU float64 instrument deliberately
bypasses compiled kernels; it is not a new library option.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from cadence import Brain, NeuralGraph
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region


def frozen(array: Any, dtype: Any = float) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def selected_rows(rows: Any, batch: int) -> np.ndarray:
    """Unique stream indices, preserving an explicit order or a boolean mask's order."""
    selected = np.asarray(rows)
    if selected.ndim != 1:
        raise ValueError("rows must be a one-dimensional mask or integer indices")
    if selected.dtype == np.bool_:
        if selected.shape != (batch,):
            raise ValueError("a row mask needs one flag per stream")
        return np.flatnonzero(selected)
    if not selected.size:
        return np.zeros(0, dtype=np.int64)
    if selected.dtype.kind not in "iu":
        raise ValueError("row indices must be integers")
    if ((selected < 0).any() or (selected > np.iinfo(np.int64).max).any()
            or (selected >= batch).any()):
        raise ValueError("row index is outside the memory streams")
    selected = selected.astype(np.int64)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("row indices must be unique")
    return selected


@dataclass(frozen=True)
class Memory:
    """One immutable witnessed memory boundary, held across all phases of a moment.

    Values concatenate event and successor coordinates; both use one selector.
    ``null`` is in unscaled match units and always denotes zero key and content.
    Slots absent from a stream have exactly zero probability and contribution.
    ``empty_anchor`` applies only to agreement: an empty stream's live read is
    drawn toward zero instead of omitting its agreement energy. Finite coupling
    is a restoring force, not a clamp or a guarantee against spurious recall.
    """

    query: np.ndarray
    read: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    held: np.ndarray
    ages: np.ndarray
    amplitude: float = 3.0
    temperature: float = 0.25
    recency: float = 0.0
    null: float = 0.0
    mode: str = "agreement"
    empty_anchor: bool = False

    def __post_init__(self) -> None:
        for name in ("amplitude", "temperature", "recency", "null"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.amplitude < 0 or self.temperature <= 0 or self.recency < 0:
            raise ValueError("nonnegative amplitude/recency and positive temperature required")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled = np.divide([self.amplitude, self.null], self.temperature)
        if not np.isfinite(scaled).all():
            raise ValueError("scaled memory parameters must be finite")
        if self.mode not in ("agreement", "linear", "competitive"):
            raise ValueError("unknown memory equation")
        object.__setattr__(self, "empty_anchor", anchor_setting(self.empty_anchor, self.mode))
        for name in ("query", "read"):
            index = np.asarray(getattr(self, name))
            if (index.ndim != 1 or index.dtype.kind not in "iu" or (index < 0).any()
                    or (index > np.iinfo(np.int64).max).any()):
                raise ValueError("ports must be one-dimensional nonnegative integer indices")
            object.__setattr__(self, name, frozen(index, np.int64))
        ports = np.concatenate((self.query, self.read))
        if len(np.unique(ports)) != len(ports) or not len(self.query) or not len(self.read):
            raise ValueError("nonempty, disjoint ports with unique coordinates required")
        for name in ("keys", "values", "ages"):
            value = np.asarray(getattr(self, name), dtype=float)
            if not np.isfinite(value).all():
                raise ValueError("memory data must be finite")
            object.__setattr__(self, name, frozen(value))
        held = np.asarray(self.held)
        if held.dtype != np.bool_:
            raise ValueError("held must contain boolean slot flags")
        object.__setattr__(self, "held", frozen(held, bool))
        if self.keys.ndim != 3 or self.values.ndim != 3:
            raise ValueError("keys and values need stream, slot and coordinate axes")
        batch, slots, width = self.keys.shape
        if (width != len(self.query) or self.values.shape != (batch, slots, len(self.read))
                or self.held.shape != (batch, slots) or self.ages.shape != (slots,)):
            raise ValueError("memory shapes must match stream, slot and port widths")
        if (self.ages < 0).any():
            raise ValueError("ages must be nonnegative")

    @property
    def streams(self) -> int:
        return int(len(self.keys))

    def take(self, rows: Any) -> Memory:
        """An immutable snapshot of selected streams, in the supplied order.

        A boolean mask has one flag per stream. Integer indices may reorder the streams
        but must be unique. Ports, ages and settings keep the same meanings in every phase.
        """
        index = selected_rows(rows, self.streams)
        return replace(self, keys=self.keys[index], values=self.values[index],
                       held=self.held[index])

    def _ports(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)
        if (state.ndim != 2 or len(state) != len(self.keys)
                or state.shape[1] <= max(int(self.query.max()), int(self.read.max()))):
            raise ValueError("activity must match memory streams and include both ports")
        return state[:, self.query], state[:, self.read]

    def weights(self, state: np.ndarray) -> np.ndarray:
        if self.mode == "linear":
            raise ValueError("the linear transport has no probability selector")
        q, r = self._ports(state)
        match = np.einsum("bk,bmk->bm", q, self.keys)
        if self.mode == "competitive":
            match += np.einsum("br,bmr->bm", r, self.values)
        logits = match / self.temperature - self.recency * self.ages
        logits = np.where(self.held, logits, -np.inf)
        null = self.null / self.temperature
        # Include the null before finding the maximum, including empty streams.
        top = np.maximum(np.max(logits, axis=1, initial=-np.inf), null)
        p = np.exp(logits - top[:, None])
        return np.asarray(p / (p.sum(axis=1) + np.exp(null - top))[:, None])

    def energy(self, state: np.ndarray) -> np.ndarray:
        q, r = self._ports(state)
        if self.mode == "linear":
            projected = np.einsum("bk,bmk->bm", q, self.keys)
            score = np.einsum("br,bmr->bm", r, self.values)
            count = np.maximum(1, self.held.sum(axis=1))
            return -self.amplitude * (projected * score * self.held).sum(axis=1) / count
        if self.mode == "competitive":
            logits = (np.einsum("bk,bmk->bm", q, self.keys)
                      + np.einsum("br,bmr->bm", r, self.values)) / self.temperature
            logits = np.where(self.held, logits - self.recency * self.ages, -np.inf)
            null = self.null / self.temperature
            top = np.maximum(np.max(logits, axis=1, initial=-np.inf), null)
            return -self.amplitude * self.temperature * (
                top + np.log(np.exp(logits - top[:, None]).sum(axis=1) + np.exp(null - top)))
        mixture = np.einsum("bm,bmr->br", self.weights(state), self.values)
        occupied = np.ones(self.streams, dtype=bool) if self.empty_anchor else self.held.any(axis=1)
        return (0.5 * self.amplitude * ((r - mixture) ** 2).sum(axis=1)
                * occupied)

    def drive(self, state: np.ndarray) -> np.ndarray:
        q, r = self._ports(state)
        out = np.zeros_like(state, dtype=float)
        if self.mode == "linear":
            count = np.maximum(1, self.held.sum(axis=1))[:, None]
            projected = np.einsum("bk,bmk->bm", q, self.keys) * self.held
            score = np.einsum("br,bmr->bm", r, self.values) * self.held
            out[:, self.query] = self.amplitude * np.einsum("bm,bmk->bk", score, self.keys) / count
            out[:, self.read] = (
                self.amplitude * np.einsum("bm,bmr->br", projected, self.values) / count)
            return out
        p = self.weights(state)
        mixture = np.einsum("bm,bmr->br", p, self.values)
        if self.mode == "competitive":
            out[:, self.query] = self.amplitude * np.einsum("bm,bmk->bk", p, self.keys)
            out[:, self.read] = self.amplitude * mixture
            return out
        occupied = np.ones(self.streams, dtype=bool) if self.empty_anchor else self.held.any(axis=1)
        error = (r - mixture) * occupied[:, None]
        score = np.einsum("br,bmr->bm", error, self.values)
        # Null contributes zero to E[score] and E[K], but retains its probability.
        centered = score - (p * score).sum(axis=1, keepdims=True)
        out[:, self.query] = (self.amplitude / self.temperature
                             * np.einsum("bm,bmk->bk", p * centered, self.keys))
        out[:, self.read] = -self.amplitude * error
        return out


def anchor_setting(value: Any, mode: str) -> bool:
    """The empty-read boundary is explicit and confined to agreement equations."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("empty_anchor must be boolean")
    if value and mode != "agreement":
        raise ValueError("empty_anchor is supported only for agreement memory")
    return bool(value)


class Graph(NeuralGraph):
    """CPU reference that includes memory in every transport AND residual check.

    Parameter copies retain the boundary. Saved-drive teaching explicitly takes
    matching stream rows. Damping remains unsupported because its fallback would
    lose the boundary. Production integration is a later gate.
    """

    memory: Memory | None = None

    def __init__(self, *args: Any, backend: str = "cpu", **kwargs: Any) -> None:
        if backend != "cpu":
            raise ValueError("attention instrument supports CPU float64 only")
        super().__init__(*args, backend="cpu", **kwargs)

    def with_parameters(self, *, efficacy: np.ndarray | None = None,
                        log_gain: np.ndarray | None = None,
                        bias: np.ndarray | None = None) -> Graph:
        if log_gain is None:
            return super().with_parameters(efficacy=efficacy, bias=bias)
        new = Graph(self.connectome, self.neuron_model,
                    efficacy=self.efficacy if efficacy is None else efficacy,
                    log_gain=log_gain, bias=self.bias if bias is None else bias,
                    dense_limit=self.dense_limit, layout=self.layout, precision=self.precision)
        new.memory = self.memory
        return new

    @property
    def _blocks(self) -> None:
        return None  # All phases and residuals use _synaptic_input, even with Numba installed.

    def _synaptic_input(self, s: np.ndarray, blocks: Any = None) -> np.ndarray:
        result = super()._synaptic_input(s, blocks)
        return result if self.memory is None else result + self.memory.drive(s)

    def equilibrate(self, *args: Any, damping: int = 0, **kwargs: Any) -> Any:
        if damping:
            raise ValueError("attention instrument requires damping=0")
        return super().equilibrate(*args, damping=0, **kwargs)


class AttentionBrain(Brain):
    """Teacher-only experimental composition; raw/fixed/learned codes are controls.

    This is not a supported continuing-life API. Reward credit, private
    imagination and saved pending continuation must be integrated and tested
    before promotion. No inherited convenience method establishes those claims.
    """

    mode: str = "agreement"
    empty_anchor: bool = False

    @classmethod
    def compose(cls, inputs: int, actions: int, *, mode: str = "agreement",
                dt: float = 0.1, seed: int = 0, empty_anchor: bool = False,
                **options: Any) -> AttentionBrain:
        if options.get("backend", "cpu") != "cpu":
            raise ValueError("attention instrument supports CPU float64 only")
        if options.get("learning_surprise", False):
            raise ValueError("partial-row teaching is not implemented in this instrument")
        if options.get("recall_window", 1) != 1:
            raise ValueError("attention instrument requires a one-event key")
        if mode not in ("agreement", "linear", "competitive"):
            raise ValueError("unknown memory equation")
        empty_anchor = anchor_setting(empty_anchor, mode)
        options.update(recall=True, recall_address=0.0, recall_null=0.0,
                       learning_qualified=True, learning_damping=0)
        options.setdefault("learning_nudged_steps", options.get("learning_free_steps", 1024))
        genome = Genome.composed(inputs, actions, seed=seed, **options)
        width = Genome.observation_width(inputs, options.get("code"), options.get("learned_code"))
        ports = {"recall", "next"}
        projections = tuple(
            Projection(p.pre, p.post, p.density, p.sign, p.scale, p.count,
                       True if p.pre in ports and p.post == "association" else p.reciprocal)
            for p in genome.projections
        ) + (Projection("query", "association", scale=0.0),)
        genome = Genome((*genome.regions, Region("query", width)), projections,
                        label="experimental-coupled-attention")
        settings = {k: v for k, v in options.items()
                    if k not in ("modules", "observers", "lateral", "scales")}
        brain = super().compose(inputs, actions, connectome=develop(genome, seed=seed),
                                seed=seed, **settings)
        brain.mode, brain.empty_anchor = mode, empty_anchor
        original = brain.brain
        brain.learner.brain = Graph(original.connectome, original.neuron_model.replace(dt=dt),
                                    efficacy=original.efficacy, bias=original.bias,
                                    log_gain=original.log_gain)
        return brain

    def act(self, observations: Any, *, greedy: bool = True, **options: Any) -> np.ndarray:
        if not greedy:
            raise NotImplementedError("reward eligibility is not integrated in this instrument")
        return super().act(observations, greedy=True, **options)

    def reset(self, *, keep: Any = ("arousal",), **renamed: Any) -> None:
        """Begin a stream without retaining the previous solve's boundary cache."""
        super().reset(keep=keep, **renamed)
        self.brain.memory = None
        if hasattr(self, "head_attended"):
            self.head_attended = None

    def teach_saved(
        self, drive: np.ndarray, target: np.ndarray, wrong: Any, memory_snapshot: Memory,
    ) -> Any:
        """Teach selected rows of one captured pre-query drive and memory boundary.

        The caller captures both before the query act writes a new moment. Its actual wrong
        rows choose the lesson externally; every free/positive/negative phase reads the same
        immutable subset. No stimulus is rebuilt, no moment is witnessed and no separate
        address lesson runs. Returns ``Learner.step``'s pair, or None without selected rows.
        """
        cfg = self.learner.config
        if not cfg.qualified or cfg.damping or cfg.surprise:
            raise ValueError("saved attention lessons need qualified phases and damping=surprise=0")
        graph = self.brain
        if not isinstance(graph, Graph):
            raise ValueError("saved attention lessons need the experimental graph")
        drive = np.asarray(drive, dtype=float)
        if (drive.ndim != 2 or drive.shape[1] != graph.connectome.n or not len(drive)
                or not np.isfinite(drive).all()):
            raise ValueError("saved drive must be a nonempty finite stream-by-neuron array")
        target = self.learner._labels(np.asarray(target))
        if len(target) != len(drive) or memory_snapshot.streams != len(drive):
            raise ValueError("saved drive, labels and memory need the same streams")
        rows = selected_rows(wrong, len(drive))
        if not len(rows):
            return None
        subset = memory_snapshot.take(rows)
        graph.memory = subset
        try:
            return self.learner.step(drive[rows], target[rows])
        finally:
            # A successful local update replaces the graph. Restore both versions so no
            # live or retained graph keeps a smaller batch's boundary after this lesson.
            graph.memory = memory_snapshot
            self.brain.memory = memory_snapshot

    def imagine(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("private imagination is not integrated in this instrument")

    def learn(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("actual-outcome credit is not integrated in this instrument")

    def step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("use lesson and greedy act in this component instrument")

    def live(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("routine and reward credit are not integrated in this instrument")

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("waiting is not integrated in this instrument")

    def save(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("coupled continuation is not checkpointed by this instrument")

    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("coupled continuation is not checkpointed by this instrument")

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        cfg = self.learner.config
        if not cfg.qualified or cfg.damping or cfg.surprise:
            raise ValueError("attention instrument needs qualified full-row teaching and damping=0")
        if self.recall is None or self.recall.address != 0 or self.recall.window != 1:
            raise ValueError("attention instrument requires one-event keys and no address lesson")
        drive = super().stimulus(observations, memory=memory)
        graph = self.brain
        assert isinstance(graph, Graph)
        graph.memory = None
        if not memory:
            return drive
        recall = self.recall
        assert recall is not None and recall.last is not None
        query = np.asarray(self.connectome.populations["query"], dtype=np.int64)
        read = np.concatenate((recall.recall_neurons, recall.next_neurons))
        drive[:, read] = 0.0  # replace the held read, not an extra drive beside it
        drive[:, query] = recall.last.key
        recall._attended = None  # the held read's weights do not describe this live solve
        batch = len(drive)
        matching = recall.streams == batch
        content = (np.concatenate((recall.values, recall.nexts), axis=2)
                   if recall.next else recall.values)
        graph.memory = Memory(
            query, read,
            recall.keys if matching else np.zeros((batch, 0, len(query))),
            content if matching else np.zeros((batch, 0, len(read))),
            recall.held if matching else np.zeros((batch, 0), dtype=bool),
            recall.ages() if matching else np.zeros(0),
            amplitude=recall.amplitude, temperature=recall.temperature,
            recency=recall.recency, mode=self.mode, empty_anchor=self.empty_anchor,
        )
        return drive

    def _settled(self, x: np.ndarray) -> Any:
        graph = self.brain
        assert isinstance(graph, Graph)
        previous = graph.memory
        recall = self.recall
        assert recall is not None
        reading = recall.reading()
        try:
            return super()._settled(x)
        except BaseException:
            graph.memory = previous
            recall.hold(reading)
            raise

    def _choose(self, x: np.ndarray, drive: np.ndarray, free: Any, **kwargs: Any) -> np.ndarray:
        graph = self.brain
        assert isinstance(graph, Graph)
        if graph.memory is not None and self.recall is not None:
            self.recall._attended = (
                None if graph.memory.mode == "linear"
                else frozen(graph.memory.weights(np.atleast_2d(free.activation))))
        return super()._choose(x, drive, free, **kwargs)
