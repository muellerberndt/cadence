"""A learned read: the query is the brain's own activity, coupled to a content memory inside the
same equilibrium.

``token_binding.ContentBrain`` reads its content memory with the present observation, fixed
before the settle. Here two regions join the composition: a ``query`` region, one neuron per
key coordinate, joined to the association region by a plastic reciprocal projection, and a
``recall`` region, one neuron per content coordinate, joined to the association region the
same way. The memory ``M`` (one matrix per stream, ``(key, content)``, written as each event is
heard: key and content are the event) couples them while every phase settles: the recall
neurons receive ``alpha * q @ M`` and the query neurons ``alpha * M @ r``, the paired
transports of the energy term ``-alpha * q . M r``. ``M`` is held fixed through the free and
the nudged phases, and the residual that qualifies a state includes the coupling. The query is
not supervised: the answer's nudge reaches the association-to-query synapses only through the
recall and the coupling, as a local contrast like every other synapse's.

The coupling enters every solve of the brain as a nudge term (``CoupledNudge``), installed in
the brain's graph by class (``CoupledGraph``), so it survives the learner's parameter updates
and the instrument's forks. Instrument level; nothing here changes the library.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from cadence import brain as cadence_brain
from cadence.brain import Brain as NeuralGraph
from cadence.brain import Equilibrium, Nudge, _copy_state
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, motor_cortex, prefrontal_cortex
from cadence.stream import FastSynapses

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

ALPHA = 3.0  # the coupling's amplitude, both directions
HEARD = 0.0  # the present observation's drive on the query neurons; 0: the query is the cortex's
QUERY_SCALE = 1.0  # initial magnitude of the association-query projection
RECALL_SCALE = 1.0  # initial magnitude of the recall-association projection


class Coupling:
    """The bilinear exchange between query and recall neurons through a per-stream memory."""

    def __init__(self, query: np.ndarray, recall: np.ndarray, memory: FastSynapses,
                 alpha: float = ALPHA) -> None:
        self.query = np.asarray(query, dtype=np.int64)
        self.recall = np.asarray(recall, dtype=np.int64)
        self.memory = memory
        self.alpha = float(alpha)
        self.rows: np.ndarray | None = None  # streams of a lesson restricted to some rows

    def matrix(self, batch: int) -> np.ndarray | None:
        strength = self.memory.strength
        if self.rows is not None and batch == len(self.rows) and len(strength):
            return np.asarray(strength[self.rows])
        if len(strength) == batch:
            return np.asarray(strength)
        return None  # a batch of other streams reads an empty memory

    def drive(self, s: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(np.asarray(s, dtype=float))
        out = np.zeros_like(s)
        m = self.matrix(len(s))
        if m is None or not m.any():
            return out
        q, r = s[:, self.query], s[:, self.recall]
        out[:, self.recall] = self.alpha * np.einsum("bk,bkc->bc", q, m)
        out[:, self.query] = self.alpha * np.einsum("bc,bkc->bk", r, m)
        return out


@dataclass(frozen=True)
class CoupledNudge(Nudge):
    """A nudge plus the memory coupling; with a zero mask it is the coupling alone."""

    coupling: Coupling | None = None

    def drive(self, s: np.ndarray) -> np.ndarray:
        base = super().drive(s)
        if self.coupling is None:
            return base
        extra = self.coupling.drive(s)
        return base + (extra[0] if np.ndim(s) == 1 else extra)

    @classmethod
    def around(cls, nudge: Nudge | None, coupling: Coupling, n: int) -> CoupledNudge:
        if nudge is None:
            return cls(np.zeros(n), np.zeros(n), 1.0, coupling=coupling)
        values = {f.name: getattr(nudge, f.name) for f in fields(Nudge)}
        return cls(**values, coupling=coupling)


class CoupledGraph(NeuralGraph):
    """The brain's graph with its memory coupling in every settle and residual."""

    coupling: Coupling

    def _coupled(self, nudge: Nudge | None) -> Nudge:
        if isinstance(nudge, CoupledNudge):
            return nudge
        return CoupledNudge.around(nudge, self.coupling, self.connectome.n)

    def settle_batch(self, drive: np.ndarray, **kwargs: Any) -> Any:
        kwargs["nudge"] = self._coupled(kwargs.get("nudge"))
        return super().settle_batch(drive, **kwargs)

    def residual(self, drive: np.ndarray, state: Any, **kwargs: Any) -> np.ndarray:
        kwargs["nudge"] = self._coupled(kwargs.get("nudge"))
        return super().residual(drive, state, **kwargs)

    def equilibrate(self, drive: np.ndarray, **kwargs: Any) -> Equilibrium:
        # the coupled nudge also reaches the damped candidates the solve builds on its own
        kwargs["nudge"] = self._coupled(kwargs.get("nudge"))
        return super().equilibrate(drive, **kwargs)


def coupled(graph: NeuralGraph, coupling: Coupling) -> NeuralGraph:
    """Install the coupling in a graph by class; parameter updates copy both. The coupling
    enters the settle through the nudge's drive, which the NumPy solver calls at every step
    and the compiled dense kernel never does, so under the kernel it is refused."""
    if cadence_brain._FUSED:
        raise RuntimeError(
            "the learned read enters the settle through the nudge's drive; "
            "run with CADENCE_FUSED=0"
        )
    graph.__class__ = CoupledGraph
    graph.coupling = coupling  # type: ignore[attr-defined]
    return graph


class ReadBrain(tb.ContentBrain):
    """The composed brain plus query and recall regions coupled through a content memory that
    writes each event as it is heard. ``build(seed, read=None)`` is ``ContentBrain.build(seed)``."""

    @classmethod
    def build(  # type: ignore[override]
        cls,
        seed: int,
        *,
        width: int = tb.WIDTH,
        read: dict[str, Any] | None = None,
        inputs: int = tb.INPUTS,
        actions: int = tb.VALUES,
        **options: Any,
    ) -> ReadBrain:
        regions = [
            Region("sensory", inputs),
            Region("association", width),
            prefrontal_cortex(width),
            motor_cortex(actions, lateral=tb.LATERAL, slots=1),
        ]
        projections = [
            Projection("sensory", "association", scale=1.0, reciprocal=False),
            Projection("association", "motor"),
            Projection("prefrontal", "association", scale=1.2, reciprocal=False),
        ]
        if read is not None:
            regions += [Region("query", inputs), Region("recall", inputs)]
            projections += [
                Projection("association", "query", scale=read.get("query_scale", QUERY_SCALE)),
                Projection("recall", "association", scale=read.get("recall_scale", RECALL_SCALE)),
            ]
        genome = Genome(tuple(regions), tuple(projections), label="composed-brain")
        brain = cls(develop(genome, seed=seed), seed=seed, slots=1, **options)
        if read is not None:
            populations = brain.connectome.populations
            query = np.asarray(populations["query"], dtype=np.int64)
            recall = np.asarray(populations["recall"], dtype=np.int64)
            memory = FastSynapses(brain.sensory_index, recall, decay=1.0, rate=1.0,
                                  amplitude=1.0, rule=read.get("rule", "hebb"))
            coupled(brain.learner.brain, Coupling(query, recall, memory, read.get("alpha", ALPHA)))
            brain.heard = float(read.get("heard", HEARD))
        return brain

    heard: float = 0.0

    @property
    def coupling(self) -> Coupling | None:
        return getattr(self.learner.brain, "coupling", None)

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        """With ``heard``, the present observation also drives the query neurons: the query
        starts as the event heard now, and the cortex learns to move it."""
        drive = super().stimulus(observations, memory=memory)
        coupling = self.coupling
        if self.heard and coupling is not None:
            x = np.asarray(self._observations(observations), dtype=float)
            drive[:, coupling.query] += self.heard * x
        return drive

    def _choose(self, x: np.ndarray, drive: np.ndarray, free: Any, **kwargs: Any) -> np.ndarray:
        action = super()._choose(x, drive, free, **kwargs)
        coupling = self.coupling
        if coupling is not None:
            x = np.asarray(x, dtype=float)
            if len(coupling.memory.strength) != len(x):
                coupling.memory.reset(len(x))
            coupling.memory.observe(x, x)  # the event as heard, keyed by itself
        return action

    def reset(self) -> None:
        super().reset()
        coupling = self.coupling
        if coupling is not None:
            coupling.memory.reset(0)

    def _equilibrate(
        self, drive: np.ndarray, state: Any, *, budget: int, tolerance: float
    ) -> Equilibrium:
        """``Brain._equilibrate`` with the coupling in its half-step fallback as well: the
        library's finite fallback builds a plain graph at half the step; here that graph is
        coupled too. Without a coupling, the library's method."""
        coupling = self.coupling
        cfg = self.learner.config
        if coupling is None or cfg.qualified:
            return super()._equilibrate(drive, state, budget=budget, tolerance=tolerance)
        brain = self.learner.brain
        first_budget = budget if budget < 2 else (budget + 1) // 2
        phase = brain.equilibrate(
            drive, state=_copy_state(state), budget=first_budget, tolerance=tolerance
        )
        if np.all(phase.qualified) or phase.steps >= budget:
            return phase
        if not all(
            np.isfinite(value).all()
            for value in (phase.state.v, phase.state.activation, phase.state.adaptation)
        ):
            return phase
        damped = coupled(
            NeuralGraph(
                brain.connectome,
                brain.neuron_model.replace(dt=brain.neuron_model.dt / 2),
                backend=brain.backend,
                efficacy=brain.efficacy,
                log_gain=brain.log_gain,
                bias=brain.bias,
                dense_limit=brain.dense_limit,
                layout=brain.layout,
                precision=brain.precision,
            ),
            coupling,
        )
        following = damped.equilibrate(
            drive, state=_copy_state(phase.state), budget=budget - phase.steps, tolerance=tolerance
        )
        residual = brain.residual(drive, following.state)
        return Equilibrium(
            following.state,
            residual,
            phase.steps + following.steps,
            tolerance,
            residual_checks=phase.residual_checks + following.residual_checks + 1,
            damping_halvings=1,
        )


def lesson(brain: ReadBrain, x: np.ndarray, target: np.ndarray, work: dict[str, Any]) -> None:
    """``token_binding.lesson``, the coupling restricted to the rows the lesson teaches."""
    from cadence.learning import LearningPhaseError

    drive = brain.stimulus(x)
    answer = None
    coupling = getattr(brain, "coupling", None)
    if brain.learner.config.surprise:
        answer = brain._choices(brain._qualified(drive, brain._activity(), operation="act"))
        if coupling is not None:
            coupling.rows = np.flatnonzero(np.asarray(answer) != np.asarray(target))
    work["lessons"] += 1
    work["lesson_rows"] += len(target)
    try:
        _, report = brain.learner.step(drive, target, answer=answer)
    except LearningPhaseError:
        work["lesson_refusals"] += 1
        return
    finally:
        if coupling is not None:
            coupling.rows = None
    work["rows_taught"] += int(report["accepted_presentations"])
    work["lesson_sweeps"] += int(report["total_steps"])


def train_round(brain: ReadBrain, trials: dict[str, np.ndarray], work: dict[str, Any]) -> bool:
    brain.reset()
    seq = tb.events(trials)
    for x in seq[:-1]:
        if tb.act(brain, x, work) is None:
            return False
    lesson(brain, seq[-1], tb.answers(trials), work)
    return tb.act(brain, seq[-1], work) is not None


def read(
    brain: ReadBrain, trials: dict[str, np.ndarray], condition: str, neutral: int,
    donor: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Answers of a copy on a panel: the prefix, the lesion, then the greedy query act."""
    twin = tb.fork(brain)
    twin.reset()
    seq = tb.events(trials, neutral)
    work = tb.new_work()
    for x in seq[:-1]:
        tb.act(twin, x, work)
    coupling = twin.coupling
    if condition == "memory_erased" and coupling is not None:
        coupling.memory.reset(0)
    elif condition == "trace_erased" and twin.working_memory is not None:
        twin.working_memory.reset(0)
    elif condition == "memory_transplant" and coupling is not None and donor is not None:
        other = tb.fork(brain)
        other.reset()
        for x in tb.events(donor, neutral)[:-1]:
            tb.act(other, x, work)
        assert other.coupling is not None
        coupling.memory.strength = other.coupling.memory.strength.copy()
        coupling.memory.mass = other.coupling.memory.mass.copy()
    out = tb.act(twin, seq[-1], work)
    return np.full(len(trials["query"]), -1) if out is None else out


def read_life(brain: ReadBrain, seed: int, held: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(np.random.SeedSequence([tb.PANEL_SEED, seed, 1]))
    for name, (trials, neutral) in tb.panels(seed, held).items():
        truth = tb.answers(trials)
        donor = tb.revalue(trials, rng)
        row: dict[str, float] = {}
        for condition in ("intact", "memory_erased", "trace_erased", "memory_transplant"):
            given = read(brain, trials, condition, neutral, donor)
            row[condition] = tb.accuracy(given, truth)
            if condition == "memory_transplant":
                row["follows_donor"] = tb.accuracy(given, tb.answers(donor))
        out[name] = row
    return out
