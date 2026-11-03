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

import importlib.util
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from cadence import Brain, NeuralGraph
from cadence.brain import BrainState, Equilibrium, Nudge
from cadence.genome import Genome, Projection, develop
from cadence.memories import Memories
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


def settlement_dt(dt: float | None, options: dict[str, Any]) -> float:
    """Keep the raw prototype step and the foundation's held-code model distinct."""
    learned = bool(options.get("learned_code"))
    if dt is None:
        return 1.0 if learned else 0.1
    if learned and dt != 1.0:
        raise ValueError("the held learned code requires the composed outer dt=1.0")
    return dt


def build_memories(recall: Any, queries: tuple[np.ndarray, ...],
                   reads: tuple[np.ndarray, ...], *, mode: str, empty_anchor: bool,
                   port_copies: int = 1) -> tuple[Memory, ...]:
    """One immutable witnessed bank for live reads and private branch reads alike.

    The caller has read this Recall's moment before building the boundary. Heads
    share its witnessed records; none receives a role, label or separate write.
    """
    if recall.last is None or not queries or len(queries) != len(reads):
        raise ValueError("a memory boundary needs the addressed moment and all head ports")
    if (isinstance(port_copies, (bool, np.bool_))
            or not isinstance(port_copies, (int, np.integer)) or port_copies < 1):
        raise ValueError("port_copies must be a positive integer")
    batch = len(recall.last.key)
    copies = port_copies
    if recall.streams == batch:
        keys = np.tile(recall.keys, (1, 1, copies)) / np.sqrt(copies)
        content = (np.concatenate((recall.values, recall.nexts), axis=2)
                   if recall.next else recall.values)
        values = np.tile(content, (1, 1, copies)) / np.sqrt(copies)
        held, ages = recall.held, recall.ages()
    else:
        keys = np.zeros((batch, 0, len(queries[0])))
        values = np.zeros((batch, 0, len(reads[0])))
        held, ages = np.zeros((batch, 0), dtype=bool), np.zeros(0)
    return tuple(Memory(query, read, keys, values, held, ages,
                        amplitude=recall.amplitude / len(queries),
                        temperature=recall.temperature, recency=recall.recency,
                        mode=mode, empty_anchor=empty_anchor)
                 for query, read in zip(queries, reads, strict=True))


_checkpoint = None


def checkpoint_module():
    global _checkpoint
    if _checkpoint is None:
        spec = importlib.util.spec_from_file_location(
            "attention_lifecycle_checkpoint", Path(__file__).with_name("attention_checkpoint.py"))
        assert spec is not None and spec.loader is not None
        _checkpoint = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _checkpoint
        spec.loader.exec_module(_checkpoint)
    return _checkpoint


class Graph(NeuralGraph):
    """CPU reference that includes memory in every transport AND residual check.

    Parameter copies and bounded numerical damping retain the exact boundary.
    Saved-drive teaching explicitly takes matching stream rows. Production
    integration is a later gate.
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

    def _with_dt(self, dt: float) -> Graph:
        """A numerical candidate on the same parameters and immutable boundary."""
        candidate = Graph(
            self.connectome, self.neuron_model.replace(dt=dt), backend=self.backend,
            efficacy=self.efficacy, log_gain=self.log_gain, bias=self.bias,
            dense_limit=self.dense_limit, layout=self.layout, precision=self.precision,
        )
        candidate.memory = self.memory
        return candidate

    def equilibrate(
        self, drive: np.ndarray, *, budget: int = 512, chunk: int = 32,
        tolerance: float = 1e-5, state: BrainState | None = None,
        mask: np.ndarray | None = None, nudge: Nudge | None = None, damping: int = 0,
    ) -> Equilibrium:
        """The library's bounded halving schedule with memory in every candidate.

        Halvings share one sweep budget, preserve every live coordinate and check
        the original complete equations. Numerical damping changes no memory,
        parameters or teaching nudge.
        """
        return super().equilibrate(
            drive, budget=budget, chunk=chunk, tolerance=tolerance,
            state=state, mask=mask, nudge=nudge, damping=damping,
        )


class AttentionBrain(Brain):
    """Experimental lifecycle composition; raw/fixed/learned codes are controls.

    This benchmark prototype is not a supported library API or a behavioral
    promotion. Reward, private branches and pending continuation reuse the
    foundation's laws, with explicit coupled-boundary custody. Their mechanical
    checks do not establish usefulness of an experimental memory equation.
    """

    mode: str = "agreement"
    empty_anchor: bool = False
    _graph_type = Graph
    _memory_type = Memory
    _pending_boundary: Any = None
    _action_boundary: Any = None
    _forecast_boundary: Any = None

    @classmethod
    def compose(cls, inputs: int, actions: int, *, mode: str = "agreement",
                dt: float | None = None, seed: int = 0, empty_anchor: bool = False,
                **options: Any) -> AttentionBrain:
        dt = settlement_dt(dt, options)
        if options.get("backend", "cpu") != "cpu":
            raise ValueError("attention instrument supports CPU float64 only")
        if options.get("learning_surprise", False):
            raise ValueError("partial-row teaching is not implemented in this instrument")
        if options.get("recall_window", 1) != 1:
            raise ValueError("attention instrument requires a one-event key")
        if options.get("recall_address_reward", 0.0):
            raise ValueError("coupled queries have no separate reward address rule")
        if mode not in ("agreement", "linear", "competitive"):
            raise ValueError("unknown memory equation")
        empty_anchor = anchor_setting(empty_anchor, mode)
        options.update(recall=True, recall_address=0.0, recall_null=0.0, learning_qualified=True)
        options.setdefault("learning_damping", 3 if options.get("learned_code") else 0)
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

    def _boundary_state(self) -> tuple[Any, ...]:
        return (self.brain, self.brain.memory,
                None if self.recall is None else self.recall.reading(),
                getattr(self, "head_attended", None), self._pending_boundary,
                self._action_boundary, self._forecast_boundary)

    def _restore_boundary(self, saved: tuple[Any, ...]) -> None:
        graph, memory, reading, attended, pending, action, forecast = saved
        graph.memory = self.brain.memory = memory
        if self.recall is not None and reading is not None:
            self.recall.hold(reading)
        if hasattr(self, "head_attended"):
            self.head_attended = attended
        self._pending_boundary, self._action_boundary, self._forecast_boundary = (
            pending, action, forecast)

    def act(self, observations: Any, *, greedy: bool = False, **options: Any) -> np.ndarray:
        """The ordinary issued-action contract, on the coupled memory equations."""
        saved = self._boundary_state()
        try:
            return Brain.act(self, observations, greedy=greedy, **options)
        except BaseException:
            self._restore_boundary(saved)
            raise

    def reset(self, *, keep: Any = ("arousal",), **renamed: Any) -> None:
        """Begin a stream without retaining the previous solve's boundary cache."""
        super().reset(keep=keep, **renamed)
        self.brain.memory = None
        self._pending_boundary = self._action_boundary = self._forecast_boundary = None
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
        if self.basal_ganglia._pending is not None:
            raise RuntimeError("report the issued action's outcome before teaching")
        if not cfg.qualified or cfg.surprise:
            raise ValueError("saved attention lessons need qualified phases and surprise=0")
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

    def learn(self, *args: Any, **kwargs: Any) -> Any:
        saved = self._boundary_state()
        if self.basal_ganglia._pending is not None and self._pending_boundary is None:
            raise RuntimeError("feedback needs the boundary of the issued coupled action")
        try:
            report = Brain.learn(self, *args, **kwargs)
        except BaseException:
            self._restore_boundary(saved)
            raise
        self._pending_boundary = None
        self._action_boundary = None
        return report

    def step(self, *args: Any, **kwargs: Any) -> Any:
        # Accepted feedback remains accepted if the subsequent action refuses.
        return Brain.step(self, *args, **kwargs)

    def live(self, *args: Any, **kwargs: Any) -> Any:
        saved = self._boundary_state()
        pending, lived = self.basal_ganglia._pending, self._lived
        try:
            return Brain.live(self, *args, **kwargs)
        except BaseException:
            accepted = ((pending is not None and self.basal_ganglia._pending is not pending)
                        or (lived is not None and self._lived is not lived))
            if not accepted:
                self._restore_boundary(saved)
            elif self.basal_ganglia._pending is None and self._lived is None:
                self._pending_boundary = self._action_boundary = None
            raise

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        saved = self._boundary_state()
        try:
            return Brain.wait(self, *args, **kwargs)
        except BaseException:
            self._restore_boundary(saved)
            raise

    def _forecast(self, x: np.ndarray, ended: bool) -> Any:
        saved = self._boundary_state()
        try:
            answer = Brain._forecast(self, x, ended)
        except BaseException:
            self._restore_boundary(saved)
            raise
        self._forecast_boundary = self.brain.memory
        return answer

    def lesson(self, *args: Any, **kwargs: Any) -> Any:
        saved = self._boundary_state()
        try:
            return Brain.lesson(self, *args, **kwargs)
        except BaseException:
            self._restore_boundary(saved)
            raise

    def predict(self, observations: Any) -> np.ndarray:
        saved, code = self._boundary_state(), self._private_code_state()
        try:
            return Brain.predict(self, observations)
        finally:
            self._restore_boundary(saved)
            self._restore_private_code(code)

    def _private_code_state(self) -> tuple[Any, ...]:
        region = self.learned_code
        held = None if region is None else (
            region.halvings, region.graph, region.last, dict(region.work))
        return self._coded_moment, self._code_settlement, held

    def _restore_private_code(self, saved: tuple[Any, ...]) -> None:
        self._coded_moment, self._code_settlement, held = saved
        if self.learned_code is not None and held is not None:
            region = self.learned_code
            region.halvings, region.graph, region.last, work = held
            region.work.clear()
            region.work.update(work)

    def imagine(self, observations: Any, *, budget: int = 1024,
                tolerance: float = 1e-6) -> tuple[Equilibrium, ...]:
        """The existing private branch, using the same coupled bank builder.

        Traces/context follow the branch; witnessed slots are read without new
        records. The branch graph owns its boundary, never the live actor's.
        """
        from cadence.brain import _copy_state

        if (isinstance(budget, (bool, np.bool_))
                or not isinstance(budget, (int, np.integer)) or budget < 0):
            raise ValueError("budget must be a nonnegative integer")
        if (isinstance(tolerance, (bool, np.bool_))
                or not isinstance(tolerance, (int, float, np.integer, np.floating))
                or not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("tolerance must be finite and nonnegative")
        sequence = tuple(self._observations(value) for value in observations)
        if not sequence:
            return ()
        batch = len(sequence[0])
        if any(len(value) != batch for value in sequence):
            raise ValueError("imagined observations must preserve stream identities")
        current = _copy_state(self._activity())
        if current is not None and len(np.atleast_2d(current.v)) != batch:
            raise ValueError("imagined batch must match the live streams")
        branch = Memories({name: memory.branch() for name, memory in self.memories.items()})
        trace, echo = branch.get("working_memory"), branch.get("efference")
        heard, recall = branch.get("history"), branch.get("recall")
        episodic = branch.get("episodic")
        original, code = self.brain, self._private_code_state()
        boundary = self._boundary_state()
        self.learner.brain = original._with_dt(original.neuron_model.dt)
        phases = []
        try:
            for value in sequence:
                self._prepare(value, trace=trace)
                drive = self.stimulus(value, memory=False)
                coded, place = self._coded(value), self._placed(value)
                drive = branch.stimulus(drive, self._read_moment(value, coded, place))
                assert recall is not None
                drive = self._install_memory(drive, recall)
                phase = self._equilibrate(drive, current, budget=budget, tolerance=tolerance)
                phases.append(phase)
                if not np.all(phase.qualified):
                    break
                current = _copy_state(phase.state)
                if trace is not None:
                    trace.update(phase.state)
                if echo is not None:
                    key = None if episodic is None else episodic.key(coded)
                    if episodic is None:
                        choice = self._choices(phase.state)
                    else:
                        probability = self.basal_ganglia.probabilities(
                            phase.state, bias=episodic.vote(place, key))
                        choice = np.asarray(np.argmax(probability, axis=-1), dtype=np.int64)
                    if episodic is not None and key is not None:
                        episodic.move(key, choice)
                    echo.issue(self._command(choice))
                if heard is not None:
                    heard.hear(self._heard(value, coded))
                recall.witness(coded, phase.state)
        finally:
            self.learner.brain = original
            self._restore_boundary(boundary)
            self._restore_private_code(code)
        return tuple(phases)

    def save(self, path: Any) -> Any:
        return checkpoint_module().save(self, path)

    @classmethod
    def load(cls, path: Any, **kwargs: Any) -> Any:
        return checkpoint_module().load(path, cls=cls, **kwargs)

    def _memory_ports(self) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        assert self.recall is not None
        return ((np.asarray(self.connectome.populations["query"]),),
                (np.concatenate((self.recall.recall_neurons, self.recall.next_neurons)),))

    def _memory_from_recall(self, recall: Any) -> Any:
        queries, reads = self._memory_ports()
        return build_memories(recall, queries, reads, mode=self.mode,
                              empty_anchor=self.empty_anchor)[0]

    def _install_memory(self, drive: np.ndarray, recall: Any) -> np.ndarray:
        queries, reads = self._memory_ports()
        copies = getattr(self, "port_copies", 1)
        cue = np.tile(recall.last.key, (1, copies)) / np.sqrt(copies)
        for query, read in zip(queries, reads, strict=True):
            drive[:, query], drive[:, read] = cue, 0.0
        self.brain.memory = self._memory_from_recall(recall)
        recall._attended = None
        return drive

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        cfg = self.learner.config
        if not cfg.qualified or cfg.surprise:
            raise ValueError(
                "attention instrument needs qualified full-row teaching and surprise=0")
        if (self.recall is None or self.recall.address != 0 or self.recall.window != 1
                or self.recall.address_reward != 0):
            raise ValueError("attention instrument requires one-event keys and no address lesson")
        drive = super().stimulus(observations, memory=memory)
        graph = self.brain
        assert isinstance(graph, Graph)
        graph.memory = None
        if not memory:
            return drive
        recall = self.recall
        assert recall is not None and recall.last is not None
        return self._install_memory(drive, recall)

    def _settled(self, x: np.ndarray) -> Any:
        saved = self._boundary_state()
        try:
            return super()._settled(x)
        except BaseException:
            self._restore_boundary(saved)
            raise

    def _attend(self, free: Any) -> None:
        graph = self.brain
        assert isinstance(graph, Graph)
        if graph.memory is not None and self.recall is not None:
            self.recall._attended = (
                None if graph.memory.mode == "linear"
                else frozen(graph.memory.weights(np.atleast_2d(free.activation))))

    def _choose(self, x: np.ndarray, drive: np.ndarray, free: Any, **kwargs: Any) -> np.ndarray:
        boundary, saved = self.brain.memory, self._boundary_state()
        agent = self.basal_ganglia
        fields = ("_free", "_free_brain", "_drive", "_pending", "_pending_memory")
        cached = {name: getattr(agent, name) for name in fields}
        rng = deepcopy(agent.rng.bit_generator.state)
        prepared, awaited = self._prepared, self._awaiting
        try:
            self._attend(free)
            action = Brain._choose(self, x, drive, free, **kwargs)
        except BaseException:
            agent.__dict__.update(cached)
            agent.rng.bit_generator.state = rng
            self._prepared, self._awaiting = prepared, awaited
            self._restore_boundary(saved)
            raise
        self._action_boundary = boundary
        self._pending_boundary = boundary if agent._pending is not None else None
        return action
