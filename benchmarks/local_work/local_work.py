"""Work follows affected knowledge: a dependency-aware local repair against full settlement.

One composed brain holds S independent skills: skill k reads its own block of ``d`` cue
neurons and answers in its own motor slot of ``a`` actions. The skills are acquired by the
ordinary teacher lessons of ``Brain.step(teacher=...)``. Then, in a continuing greedy life,
one bounded disturbance at a time is applied to skill 0 (a new cue, or one lesson that changes
one relation) while every other skill's input stays, and two lives answer each moment from the
same previous state:

* the **full** life: the library's ``Brain.act`` as it is, which qualifies the whole graph
  (all neurons advance in every sweep, residual transports at chunk boundaries);
* the **local** life: the ``LocalRepair`` prototype of this instrument, which keeps the last
  qualified state, its synaptic inputs and its equation errors, puts the neurons whose
  equation is violated by the disturbance on a worklist, updates them with the library's
  own neuron rule, pushes their activation changes along their outgoing contacts, re-checks
  the receivers, and stops when no error exceeds the tolerance; then one full residual
  transport verifies every live coordinate, as ``act`` does.

A third, uncharged reference (``sync_reference``) runs the library's synchronous sweep over
every neuron and stops at the first sweep whose complete residual is within tolerance: it
separates the chunk quantization of ``equilibrate`` from the saving of a bounded worklist.

Two anatomies carry the same skills. ``dense`` is ``Brain.compose`` as released: every
projection is dense, so one association cortex of ``sum(widths)`` neurons is shared by all
skills. ``sparse`` is the block-diagonal anatomy: S single-skill compositions developed
separately and joined into one connectome, region-major, so that each skill is a reciprocal
sub-graph of its own (sensory block, association block, working-trace block and motor slot)
and the whole is one composed brain with one learner, one working trace and S motor slots. At
S = 1 the two anatomies are the same connectome. ``Brain.compose`` offers no sparse
development; ``Projection(density=...)`` thins a projection by a random mask drawn over the
full source-by-destination array, which does not produce block structure and still allocates
that array, so the block anatomy is built here from developed single-skill compositions.

Everything here is an instrument on top of the library's public settle and residual
primitives and the stream bookkeeping of ``Brain.act``; no library code changes.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import cadence as cd
from cadence.brain import BrainState
from cadence.connectome import Connectome

REGIONS = ("sensory", "association", "prefrontal", "motor")


# -- the world: S independent cue-to-action skills


def skill_maps(S: int, founder: int, d: int, a: int) -> list[np.ndarray]:
    """Skill k's map from its ``d`` cues to its ``a`` actions, fixed per founder and k.

    Skill 0, the disturbed skill, has the same map at every S; every map has every action.
    """
    maps = []
    for k in range(S):
        rng = np.random.default_rng([founder, 11, k])
        while True:
            table = rng.integers(0, a, size=d)
            if len(set(table.tolist())) == a:
                break
        maps.append(table)
    return maps


def observations(cues: np.ndarray, d: int) -> np.ndarray:
    """One-hot cue per block: ``cues`` is ``(batch, S)``, the result ``(batch, S * d)``."""
    cues = np.asarray(cues, dtype=np.int64)
    batch, S = cues.shape
    x = np.zeros((batch, S * d))
    for k in range(S):
        x[np.arange(batch), k * d + cues[:, k]] = 1.0
    return x


def labels(cues: np.ndarray, maps: Sequence[np.ndarray]) -> np.ndarray:
    """The teacher's labels: ``(batch, S)`` for several slots, ``(batch,)`` for one."""
    cues = np.asarray(cues, dtype=np.int64)
    y = np.stack([maps[k][cues[:, k]] for k in range(cues.shape[1])], axis=1)
    return y[:, 0] if cues.shape[1] == 1 else y


# -- the two anatomies


def block_sparse_connectome(parts: Sequence[Connectome], label: str) -> Connectome:
    """Join single-skill connectomes region-major into one block-diagonal connectome.

    The joined populations ``sensory``, ``association``, ``prefrontal`` and ``motor`` are the
    concatenations of the parts' populations in part order, so the i-th association neuron
    and the i-th prefrontal neuron belong to the same part (the working trace copies one onto
    the other) and the motor slots follow the parts. Each part's populations are kept as
    ``<region>_<k>`` so that the library's block layout cuts the ranges at the block borders.
    """
    offsets: dict[tuple[int, str], int] = {}
    populations: dict[str, list[int]] = {name: [] for name in REGIONS}
    n = 0
    for name in REGIONS:
        for k, part in enumerate(parts):
            members = part.populations[name]
            offsets[(k, name)] = n - members[0]
            populations[name].extend(range(n, n + len(members)))
            populations[f"{name}_{k}"] = list(range(n, n + len(members)))
            n += len(members)
    pre, post, count, sign = [], [], [], []
    for k, part in enumerate(parts):
        region_of = np.empty(part.n, dtype=object)
        for name in REGIONS:
            for i in part.populations[name]:
                region_of[i] = name
        if any(region is None for region in region_of):
            raise ValueError("every neuron of a part must belong to one of the four regions")
        g = np.array([offsets[(k, region_of[i])] + i for i in range(part.n)], dtype=np.int64)
        pre.append(g[part.pre])
        post.append(g[part.post])
        count.append(part.count)
        sign.append(part.sign)
    populations["motor/actions"] = list(populations["motor"])
    return Connectome.from_synapses(
        n,
        pre=np.concatenate(pre),
        post=np.concatenate(post),
        count=np.concatenate(count),
        sign=np.concatenate(sign),
        populations=populations,
        label=label,
    )


def genes(S: int, actor_rates: bool = False) -> dict[str, Any]:
    """The declared construction settings beside the composed founder.

    The learner nudges every motor slot at ``beta / slots`` and divides the contrast by the
    full ``2 beta`` (the mean loss over the slots), so each skill's lesson step is ``1 / S``
    of the single-skill step; ``learning_eta`` and ``learning_eta_bias`` are scaled by S so
    that skill 0 is taught at the same rate at every S. The reward learner's rates are zero
    unless ``actor_rates`` is set, so the zero-reward feedback ``step`` gives its previous
    sampled act changes no parameter and every parameter change comes from a lesson.
    """
    out: dict[str, Any] = {"learning_eta": 0.5 * S, "learning_eta_bias": 0.05 * S}
    if not actor_rates:
        out.update({"actor_eta": 0.0, "actor_eta_bias": 0.0, "actor_eta_critic": 0.0})
    return out


def part_seed(founder: int, k: int) -> int:
    """Block 0 is developed from the founder seed itself; block k from ``founder * 100 + k``."""
    return founder if k == 0 else founder * 100 + k


def build_brain(
    arm: str, widths: Sequence[int], founder: int, *, d: int = 4, a: int = 2
) -> cd.Brain:
    """The composed brain of one arm: ``dense`` (``Brain.compose`` as released, one shared
    association cortex of ``sum(widths)``) or ``sparse`` (the block-diagonal join of
    single-skill compositions of the given widths)."""
    S = len(widths)
    if arm == "dense":
        return cd.Brain.compose(
            inputs=S * d, actions=S * a, modules=(int(sum(widths)),), slots=S, seed=founder,
            episodic=False, **genes(S),
        )
    if arm == "sparse":
        parts = [
            cd.Brain.compose(
                inputs=d, actions=a, modules=(int(width),), slots=1,
                seed=part_seed(founder, k), episodic=False,
            ).connectome
            for k, width in enumerate(widths)
        ]
        connectome = block_sparse_connectome(parts, label=f"block-sparse-{S}")
        return cd.Brain(connectome, seed=founder, slots=S, episodic=False, **genes(S))
    raise ValueError(f"unknown arm {arm!r}")


def block_of(brain: cd.Brain, widths: Sequence[int], d: int, a: int) -> np.ndarray:
    """The skill block of every neuron, by region order: sensory, association, prefrontal
    (one block per skill in that order), motor (one slot per skill)."""
    n = brain.connectome.n
    out = np.full(n, -1, dtype=np.int64)
    S = len(widths)
    bounds = np.concatenate([[0], np.cumsum(widths)])
    for k in range(S):
        out[brain.sensory_index[k * d : (k + 1) * d]] = k
        out[brain.association_index[bounds[k] : bounds[k + 1]]] = k
        prefrontal = np.asarray(brain.connectome.populations["prefrontal"], dtype=np.int64)
        out[prefrontal[bounds[k] : bounds[k + 1]]] = k
        out[brain.motor_index[k * a : (k + 1) * a]] = k
    if (out < 0).any():
        raise ValueError("every neuron must belong to a skill block")
    return out


# -- acquisition by the ordinary teacher lessons


def acquire(
    brain: cd.Brain,
    maps: Sequence[np.ndarray],
    founder: int,
    *,
    d: int,
    lessons: int = 400,
    batch: int = 16,
    every: int = 20,
    gate: float = 0.95,
    probe_rows: int = 256,
) -> dict[str, Any]:
    """Teach all skills together with ``step(teacher=...)`` on ``batch`` parallel streams
    until every skill's probe accuracy reaches ``gate`` or ``lessons`` are spent.

    The probe is ``predict`` (independent samples, no trace) on ``probe_rows`` observations
    with uniform cues per block, fixed by the founder, read every ``every`` lessons.
    """
    S = len(maps)
    rng = np.random.default_rng([founder, 5, S])
    probe = np.random.default_rng([founder, 7]).integers(0, d, size=(probe_rows, S))
    x_probe = observations(probe, d)
    y_probe = np.asarray(labels(probe, maps)).reshape(probe_rows, S)
    history: list[dict[str, Any]] = []
    settle_sweeps = 0
    settle_checks = 0
    started = time.perf_counter()
    taken = 0
    accuracy = np.zeros(S)
    for lesson in range(lessons):
        cues = rng.integers(0, d, size=(batch, S))
        brain.step(observations(cues, d), teacher=labels(cues, maps))
        taken = lesson + 1
        report = brain.last_settlement
        if report is not None:
            settle_sweeps += int(report["steps"])
            settle_checks += int(report["residual_checks"])
        if taken % every == 0 or taken == lessons:
            predicted = brain.predict(x_probe).reshape(probe_rows, S)
            accuracy = (predicted == y_probe).mean(axis=0)
            history.append({"lessons": taken, "accuracy": [float(v) for v in accuracy]})
            if bool((accuracy >= gate).all()):
                break
    certificate = cd.certificate(brain.brain)
    return {
        "lessons": taken,
        "batch": batch,
        "accuracy": [float(v) for v in accuracy],
        "at_gate": bool((accuracy >= gate).all()),
        "gate": gate,
        "history": history,
        "lesson_act_sweeps": settle_sweeps,
        "lesson_act_residual_checks": settle_checks,
        "seconds": time.perf_counter() - started,
        "certificate": certificate.to_dict(),
    }


# -- the graph as the instrument reads it


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """The concatenation of ``range(starts[i], starts[i] + lengths[i])`` without a loop."""
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.asarray(np.repeat(starts, lengths) + np.arange(total, dtype=np.int64) - offsets)


@dataclass
class Graph:
    """The connectome, the current effective weights and both adjacency orders.

    The library keeps its synapses sorted by receiver (``post``); the sender order is a
    fixed permutation of it. ``transport`` is one complete pass over every contact.
    """

    graph: Any  # the NeuralGraph whose weights this reads
    n: int = field(init=False)
    E: int = field(init=False)
    pre: np.ndarray = field(init=False)
    post: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    bias: np.ndarray = field(init=False)
    in_ptr: np.ndarray = field(init=False)
    out_order: np.ndarray = field(init=False)
    out_ptr: np.ndarray = field(init=False)
    out_degree: np.ndarray = field(init=False)
    in_degree: np.ndarray = field(init=False)
    source_contacts: int = field(init=False)  # contacts whose sender hears nothing

    def __post_init__(self) -> None:
        c = self.graph.connectome
        self.n, self.E = int(c.n), int(c.synapses)
        self.pre = np.asarray(c.pre, dtype=np.int64)
        self.post = np.asarray(c.post, dtype=np.int64)
        self.weights = np.array(self.graph.weights, dtype=float)
        self.bias = np.array(self.graph.bias, dtype=float)
        self.in_degree = np.bincount(self.post, minlength=self.n)
        self.out_degree = np.bincount(self.pre, minlength=self.n)
        self.in_ptr = np.concatenate([[0], np.cumsum(self.in_degree)]).astype(np.int64)
        self.out_order = np.argsort(self.pre, kind="stable")
        self.out_ptr = np.concatenate([[0], np.cumsum(self.out_degree)]).astype(np.int64)
        self.source_contacts = int((self.in_degree[self.pre] == 0).sum())

    @property
    def model(self) -> Any:
        return self.graph.neuron_model

    def refresh(self, graph: Any) -> tuple[np.ndarray, np.ndarray]:
        """Read the weights and biases of ``graph`` (the same topology) and return the
        changes against what was read before."""
        weights = np.array(graph.weights, dtype=float)
        bias = np.array(graph.bias, dtype=float)
        if weights.shape != self.weights.shape or bias.shape != self.bias.shape:
            raise ValueError("a relation change must keep the topology")
        delta_w, delta_b = weights - self.weights, bias - self.bias
        self.graph, self.weights, self.bias = graph, weights, bias
        return delta_w, delta_b

    def transport(self, s: np.ndarray) -> np.ndarray:
        """Every neuron's synaptic input from activations ``s``: one pass over all contacts."""
        return np.asarray(
            np.bincount(self.post, weights=self.weights * s[self.pre], minlength=self.n)
        )

    def out_contacts(self, neurons: np.ndarray) -> np.ndarray:
        """The synapse indices leaving ``neurons``."""
        return np.asarray(self.out_order[_ranges(self.out_ptr[neurons], self.out_degree[neurons])])


# -- the prototype scheduler


@dataclass
class RepairCounts:
    """What one local repair did; every count is of the instrument's own operations."""

    rounds: int = 0
    updates: int = 0  # neuron equation evaluations with a write of the potential
    messages: int = 0  # contacts along which an activation change was pushed
    residual_evaluations: int = 0  # equation errors read from cached synaptic inputs
    global_checks: int = 0  # complete residual transports (E contacts, n equations each)
    seed_changed_drive: int = 0  # drive entries that changed at this moment
    seed_active: int = 0  # of those, equations violated beyond tolerance
    relation_changed_synapses: int = 0  # synapses whose weight changed (relation change)
    relation_changed_biases: int = 0
    largest_active: int = 0  # the largest worklist of any round
    cone: int = 0  # neurons updated at least once this moment
    qualified: bool = True
    residual: float = 0.0  # the instrument's own final complete residual
    exhausted: bool = False
    updated: np.ndarray | None = None  # the neurons updated at least once, not serialized

    def contacts(self, E: int) -> int:
        return self.messages + self.global_checks * E

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds, "updates": self.updates, "messages": self.messages,
            "residual_evaluations": self.residual_evaluations,
            "global_checks": self.global_checks,
            "seed_changed_drive": self.seed_changed_drive, "seed_active": self.seed_active,
            "relation_changed_synapses": self.relation_changed_synapses,
            "relation_changed_biases": self.relation_changed_biases,
            "largest_active": self.largest_active, "cone": self.cone,
            "qualified": self.qualified, "residual": self.residual,
            "exhausted": self.exhausted,
        }


class LocalRepair:
    """Dependency-aware repair of one stream's qualified state after a bounded disturbance.

    The rule, exactly:

    1. Keep, from the last complete check, every neuron's potential ``v``, activation
       ``s = act(v)``, synaptic input ``u`` and signed equation error
       ``e = u + drive + bias - v``.
    2. A changed drive entry changes ``e`` of that neuron alone by the drive difference; a
       changed weight on a contact changes ``u`` and ``e`` of its receiver by the weight
       difference times the sender's activation; a changed bias changes ``e`` of its neuron.
       The worklist is the set of neurons whose ``|e|`` exceeds the tolerance.
    3. A round takes the whole worklist at once: every member moves by the library's own
       rule, ``v <- v + dt * e``, publishes ``s = act(v)``, and its activation change times
       each outgoing weight is added to the receiver's ``u`` (one message per contact). The
       errors of the members and of every receiver are re-read from the cached inputs; the
       next worklist is those whose ``|e|`` exceeds the tolerance. Nothing else is read.
    4. When the worklist is empty, one complete residual transport re-reads every synaptic
       input and every equation; the state is qualified when the largest error is within
       the tolerance, as the library's check requires. The fresh inputs replace the cache.
       If the complete check finds an error beyond tolerance (an exhausted round budget,
       or drift of the cached sums), its errors seed another repair, up to ``max_checks``
       complete checks per moment; a moment that ends unqualified is a failure.

    Round budget and tolerance are the learner's ``free_steps`` and ``tolerance`` unless
    given. Counts are of these operations alone; the host seconds are the caller's.
    """

    def __init__(
        self,
        graph: Any,
        state: BrainState,
        drive: np.ndarray,
        *,
        tolerance: float,
        max_rounds: int = 1024,
        max_checks: int = 3,
        margin: float = 1.0,
    ) -> None:
        if not np.isfinite(margin) or margin < 1.0:
            raise ValueError("margin must be a finite factor of at least one")
        self.g = Graph(graph)
        self.tolerance = float(tolerance)
        # the worklist threshold: the tolerance over the margin (amendment 1); the complete
        # check stays at the tolerance, the library's qualification
        self.threshold = float(tolerance) / float(margin)
        self.max_rounds = int(max_rounds)
        self.max_checks = int(max_checks)
        self.v = np.array(np.atleast_2d(state.v)[0], dtype=float)
        self.s = self.g.model.activation(self.v)
        self.drive = np.array(np.atleast_2d(drive)[0], dtype=float)
        self.u = self.g.transport(self.s)  # the first cache: one complete transport
        self.e = self.u + self.drive + self.g.bias - self.v
        self.initial_checks = 1

    @property
    def state(self) -> BrainState:
        return BrainState(
            v=self.v[None, :].copy(), activation=self.s[None, :].copy(),
            adaptation=np.zeros((1, self.g.n)), steps=0,
        )

    def relation_changed(self, graph: Any, counts: RepairCounts) -> np.ndarray:
        """Account for changed weights and biases; returns the receivers whose error moved."""
        delta_w, delta_b = self.g.refresh(graph)
        changed = np.flatnonzero(delta_w)
        counts.relation_changed_synapses = int(len(changed))
        receivers = np.zeros(0, dtype=np.int64)
        if len(changed):
            du = np.zeros(self.g.n)
            np.add.at(du, self.g.post[changed], delta_w[changed] * self.s[self.g.pre[changed]])
            receivers = np.unique(self.g.post[changed])
            self.u[receivers] += du[receivers]
            self.e[receivers] += du[receivers]
        biased = np.flatnonzero(delta_b)
        counts.relation_changed_biases = int(len(biased))
        if len(biased):
            self.e[biased] += delta_b[biased]
            receivers = np.union1d(receivers, biased)
        return receivers

    def repair(
        self, drive: np.ndarray, *, graph: Any = None, counts: RepairCounts | None = None
    ) -> tuple[BrainState, RepairCounts]:
        """Repair the kept state under a new ``drive`` (and ``graph`` after a relation change)."""
        counts = RepairCounts() if counts is None else counts
        g = self.g
        touched = np.zeros(0, dtype=np.int64)
        if graph is not None and graph is not g.graph:
            touched = self.relation_changed(graph, counts)
        new = np.array(np.atleast_2d(drive)[0], dtype=float)
        delta = new - self.drive
        changed = np.flatnonzero(delta)
        counts.seed_changed_drive = int(len(changed))
        self.e[changed] += delta[changed]
        self.drive = new
        candidates = np.union1d(changed, touched)
        active = candidates[np.abs(self.e[candidates]) > self.threshold]
        counts.seed_active = int(len(active))
        ever = np.zeros(g.n, dtype=bool)
        dt = float(g.model.dt)
        for check in range(self.max_checks):
            while len(active) and counts.rounds < self.max_rounds:
                counts.rounds += 1
                counts.updates += int(len(active))
                counts.largest_active = max(counts.largest_active, int(len(active)))
                ever[active] = True
                # the library's neuron rule on the worklist, from the cached inputs
                self.v[active] += dt * self.e[active]
                published = g.model.activation(self.v[active])
                ds = published - self.s[active]
                self.s[active] = published
                # one message per outgoing contact of every member
                contacts = g.out_contacts(active)  # in worklist order, sender by sender
                counts.messages += int(len(contacts))
                # one message per contact, summed at the receivers; the scatter uses an
                # n-length accumulator, an implementation shortcut that a compiled kernel
                # would not need and that is not counted as work
                self.u += np.bincount(
                    g.post[contacts],
                    weights=g.weights[contacts] * np.repeat(ds, g.out_degree[active]),
                    minlength=g.n,
                )
                receivers = np.union1d(active, g.post[contacts])
                counts.residual_evaluations += int(len(receivers))
                self.e[receivers] = (
                    self.u[receivers] + self.drive[receivers] + g.bias[receivers]
                    - self.v[receivers]
                )
                active = receivers[np.abs(self.e[receivers]) > self.threshold]
            if not len(active) and counts.rounds == 0 and check == 0 and not len(touched):
                # nothing was violated and nothing moved: the kept certificate stands
                break
            # the complete check, as the library's qualification requires
            counts.global_checks += 1
            self.u = g.transport(self.s)
            self.e = self.u + self.drive + g.bias - self.v
            counts.residual = float(np.max(np.abs(self.e)))
            if counts.residual <= self.tolerance:
                break
            active = np.flatnonzero(np.abs(self.e) > self.threshold)
            if counts.rounds >= self.max_rounds:
                counts.exhausted = True
                break
        counts.qualified = bool(counts.residual <= self.tolerance) if (
            counts.global_checks
        ) else True
        counts.cone = int(ever.sum())
        counts.updated = ever
        return self.state, counts


# -- the full arm and the synchronous reference


def effective_contacts(g: Graph, sweeps: int, checks: int) -> int:
    """Contacts the library's block transport reads: products from ranges that hear nothing
    are reused once their activation stops changing, which takes at most two sweeps."""
    recurrent = g.E - g.source_contacts
    return recurrent * sweeps + g.source_contacts * min(sweeps, 2) + g.E * checks


def sync_reference(
    g: Graph, state: BrainState, drive: np.ndarray, *, tolerance: float, max_sweeps: int = 1024
) -> dict[str, Any]:
    """The library's synchronous rule over every neuron from the same state, checked after
    every sweep: the sweeps a full settlement would take without chunk quantization."""
    v = np.array(np.atleast_2d(state.v)[0], dtype=float)
    d = np.array(np.atleast_2d(drive)[0], dtype=float)
    s = g.model.activation(v)
    dt = float(g.model.dt)
    e = g.transport(s) + d + g.bias - v
    over_tolerance = int((np.abs(e) > tolerance).sum())
    sweeps = 0
    residual = float(np.max(np.abs(e)))
    while residual > tolerance and sweeps < max_sweeps:
        v += dt * e
        s = g.model.activation(v)
        e = g.transport(s) + d + g.bias - v
        sweeps += 1
        residual = float(np.max(np.abs(e)))
    return {
        "sweeps": sweeps, "residual": residual, "qualified": residual <= tolerance,
        "updates": g.n * sweeps, "contacts": g.E * (2 * sweeps + 1), "activation": s,
        "over_tolerance_at_start": over_tolerance,
    }


def full_moment(brain: cd.Brain, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """One greedy moment of the library's own life, with its settlement report."""
    started = time.perf_counter()
    action = brain.act(x, greedy=True)
    seconds = time.perf_counter() - started
    report = brain.last_settlement
    assert report is not None
    return action, {
        "sweeps": int(report["steps"]), "residual_checks": int(report["residual_checks"]),
        "damping_halvings": int(report["damping_halvings"]),
        "stagnation_checks": int(report["stagnation_checks"]),
        "qualified": bool(report["qualified"]), "residual": float(report["max_residual"]),
        "seconds": seconds,
    }


def local_moment(
    brain: cd.Brain, repair: LocalRepair, x: np.ndarray, *, relation: bool = False
) -> tuple[np.ndarray, dict[str, Any], BrainState, np.ndarray, np.ndarray]:
    """One greedy moment of the local life: the drive the library composes (senses plus the
    working trace), the repair from the stream's kept state, then the library's own
    bookkeeping of a chosen action (the trace advances, the stream keeps the state)."""
    started = time.perf_counter()
    validated = brain._observations(x)
    drive = brain.stimulus(validated)
    state, counts = repair.repair(drive, graph=brain.brain if relation else None)
    action = brain._choose(validated, drive, state, greedy=True)
    seconds = time.perf_counter() - started
    record = counts.to_dict()
    record["seconds"] = seconds
    updated = counts.updated if counts.updated is not None else np.zeros(repair.g.n, bool)
    return action, record, state, drive, updated


def reference_state(
    graph: Any, drive: np.ndarray, state: BrainState, *, tolerance: float = 1e-7,
    budget: int = 512,
) -> np.ndarray | None:
    """A tighter solution of the same equations from ``state``, or None if the budget ends
    first: the yardstick both arms' states are measured against."""
    from cadence.patch import _copy_state

    phase = graph.equilibrate(
        drive, state=_copy_state(state), budget=budget, chunk=32, tolerance=tolerance
    )
    if not bool(np.all(phase.qualified)):
        return None
    return np.asarray(np.asarray(phase.state.activation)[0], dtype=float)


def clone(brain: cd.Brain) -> cd.Brain:
    """The same brain as a second, independent life: a deep copy that shares the immutable
    connectome (a frozen value the library never writes), or, where a component resists
    copying, the library's own checkpoint written and read back."""
    try:
        memo: dict[int, Any] = {id(brain.connectome): brain.connectome}
        return copy.deepcopy(brain, memo)
    except TypeError:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "life.npz"
            brain.save(path)
            return cd.Brain.load(path)


# -- one case: acquisition, then a continuing life under disturbances in two arms


@dataclass(frozen=True)
class CaseSpec:
    arm: str
    widths: tuple[int, ...]
    founder: int
    d: int = 4
    a: int = 2
    lessons: int = 400
    probes: int = 6
    relation_probes: int = 3
    hold: int = 64
    state_gate: float = 10.0  # times the tolerance, on activations
    reference_moments: int = 3  # change moment and the ripple moments measured against it
    local_margin: float = 1.0  # amendment 1: the worklist stops at tolerance / margin

    @property
    def S(self) -> int:
        return len(self.widths)

    @property
    def name(self) -> str:
        margin = "" if self.local_margin == 1.0 else f"-m{self.local_margin:g}"
        return (
            f"{self.arm}-S{self.S}-w{'x'.join(str(w) for w in self.widths)}-f{self.founder}"
            f"{margin}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "arm": self.arm, "widths": list(self.widths), "S": self.S,
            "founder": self.founder, "d": self.d, "a": self.a, "lessons": self.lessons,
            "probes": self.probes, "relation_probes": self.relation_probes, "hold": self.hold,
            "state_gate": self.state_gate, "reference_moments": self.reference_moments,
            "local_margin": self.local_margin,
        }


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def run_case(spec: CaseSpec, *, verbose: bool = False) -> dict[str, Any]:
    """Acquire the skills, then live the probe schedule in both arms and compare."""
    started = time.perf_counter()
    maps = skill_maps(spec.S, spec.founder, spec.d, spec.a)
    brain = build_brain(spec.arm, spec.widths, spec.founder, d=spec.d, a=spec.a)
    blocks = block_of(brain, spec.widths, spec.d, spec.a)
    acquisition = acquire(brain, maps, spec.founder, d=spec.d, lessons=spec.lessons)
    assert brain.learner.config.tolerance is not None
    tolerance = float(brain.learner.config.tolerance)
    free_steps = int(brain.learner.config.free_steps)
    brain.reset()
    full = brain
    local = clone(brain)
    g = Graph(full.brain)
    cone_contacts = int((blocks[g.pre] == 0).sum())
    anatomy = {
        "n": g.n, "E": g.E, "source_contacts": g.source_contacts,
        "block0_neurons": int((blocks == 0).sum()), "block0_contacts": cone_contacts,
        "layout_ranges": int(full.brain.layout.ranges),
        "layout_pairs": int(full.brain.layout.pairs),
        "transport": full.brain.to_dict()["transport"],
        "tolerance": tolerance, "free_steps": free_steps, "dt": float(g.model.dt),
    }
    cues = np.zeros((1, spec.S), dtype=np.int64)
    x = observations(cues, spec.d)
    moments: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []

    # the first moment: both lives answer by one full settlement; the local life then
    # keeps its state, its synaptic inputs and its equation errors
    action_full, report_full = full_moment(full, x)
    action_local, report_local = full_moment(local, x)
    state_local = local.basal_ganglia.state
    drive_first = local.basal_ganglia._drive
    assert state_local is not None and drive_first is not None
    repair = LocalRepair(
        local.brain, state_local, drive_first, tolerance=tolerance,
        max_rounds=free_steps, margin=spec.local_margin,
    )
    moments.append({
        "index": 0, "kind": "first", "episode": -1, "cue0": 0,
        "full": report_full, "local": {"initial_checks": repair.initial_checks,
                                       "seconds": report_local["seconds"]},
        "same_action": bool(np.array_equal(action_full, action_local)),
    })

    def live(kind: str, episode: int, relation: bool, want_reference: bool) -> dict[str, Any]:
        """One moment in both lives, from each stream's own kept state."""
        full_state = full.basal_ganglia.state
        assert full_state is not None
        before = BrainState(
            v=np.array(full_state.v, copy=True), activation=np.array(full_state.activation,
                                                                   copy=True),
            adaptation=np.array(full_state.adaptation, copy=True), steps=0,
        )
        drive_full = full.stimulus(x)
        action_full, report_full = full_moment(full, x)
        action_local, report_local, state_local, drive_local, updated = local_moment(
            local, repair, x, relation=relation
        )
        settled_full = full.basal_ganglia.state
        assert settled_full is not None
        # the library's own check of the local state, uncharged: the verification of the
        # instrument's complete residual
        library_residual = float(local.brain.residual(drive_local, state_local)[0])
        report_full["updates"] = g.n * report_full["sweeps"]
        report_full["contacts_nominal"] = g.E * (
            report_full["sweeps"] + report_full["residual_checks"]
        )
        report_full["contacts_effective"] = effective_contacts(
            g, report_full["sweeps"], report_full["residual_checks"]
        )
        report_full["residual_evaluations"] = g.n * report_full["residual_checks"]
        report_local["library_residual"] = library_residual
        report_local["library_qualified"] = library_residual <= tolerance
        report_local["contacts"] = report_local["messages"] + g.E * report_local[
            "global_checks"
        ]
        reference = sync_reference(g, before, drive_full, tolerance=tolerance,
                                   max_sweeps=free_steps)
        record: dict[str, Any] = {
            "index": len(moments), "kind": kind, "episode": episode, "cue0": int(cues[0, 0]),
            "full": report_full, "local": report_local,
            "sync": {k: v for k, v in reference.items() if k != "activation"},
            "same_action": bool(np.array_equal(action_full, action_local)),
            "action_full": np.asarray(action_full).ravel().tolist(),
            "action_local": np.asarray(action_local).ravel().tolist(),
            "correct_full": bool(np.array_equal(
                np.asarray(action_full).ravel(), np.asarray(labels(cues, maps)).ravel()
            )),
            "correct_local": bool(np.array_equal(
                np.asarray(action_local).ravel(), np.asarray(labels(cues, maps)).ravel()
            )),
            "distance_activation": _distance(settled_full.activation, state_local.activation),
            "distance_potential": _distance(settled_full.v, state_local.v),
            "distance_sync_full": _distance(reference["activation"], settled_full.activation),
            "distance_sync_local": _distance(reference["activation"], state_local.activation),
            "drive_distance": _distance(drive_full, drive_local),
            "cone_outside_block0": int((blocks[updated] != 0).sum()),
            "cone_by_block": np.bincount(blocks[updated], minlength=spec.S).tolist(),
            "state_gate": _distance(settled_full.activation, state_local.activation)
            <= spec.state_gate * tolerance,
        }
        if want_reference:
            tight = reference_state(full.brain, drive_full, settled_full)
            record["reference"] = None if tight is None else {
                "distance_full": _distance(tight, settled_full.activation),
                "distance_local": _distance(tight, state_local.activation),
            }
        moments.append(record)
        if verbose:
            print(
                f"  {record['index']:4d} {kind:9s} full {report_full['sweeps']:4d}/"
                f"{report_full['residual_checks']} local r{report_local['rounds']:3d} "
                f"u{report_local['updates']:5d} m{report_local['messages']:7d} "
                f"c{report_local['global_checks']} same {record['same_action']} "
                f"dist {record['distance_activation']:.2e}"
            )
        return record

    def hold(kind_prefix: str, episode: int) -> dict[str, Any]:
        """Hold the present observation until both lives are stationary, or ``hold`` moments."""
        held = 0
        stationary_full = stationary_local = None
        while held < spec.hold:
            record = live(f"{kind_prefix}-hold", episode, False, held < spec.reference_moments - 1)
            held += 1
            if stationary_full is None and record["full"]["sweeps"] == 0:
                stationary_full = held
            if stationary_local is None and record["local"]["updates"] == 0:
                stationary_local = held
            if stationary_full is not None and stationary_local is not None:
                break
        return {"held": held, "stationary_full": stationary_full,
                "stationary_local": stationary_local}

    # the routine: hold the first observation until stationary
    episodes.append({"kind": "settle-in", "episode": -1, **hold("settle", -1)})
    # input disturbances: a new cue for skill 0, the other skills' cues unchanged
    for p in range(spec.probes):
        cues[0, 0] = (cues[0, 0] + 1 + (p % (spec.d - 1))) % spec.d
        x = observations(cues, spec.d)
        change = live("input", p, False, True)
        episodes.append({"kind": "input", "episode": p, "change_index": change["index"],
                         **hold("input", p)})
    # relation disturbances: one lesson on skill 0 at the present observation, the label the
    # other action than the map's; both lives take the same lesson from their own states
    for r in range(spec.relation_probes):
        episode = spec.probes + r
        y = np.asarray(labels(cues, maps)).reshape(1, spec.S).copy()
        y[0, 0] = (y[0, 0] + 1 + r) % spec.a
        shown = y[:, 0] if spec.S == 1 else y
        weights_before = np.array(local.brain.weights, copy=True)
        full.lesson(x, shown)
        local.lesson(x, shown)
        g.refresh(full.brain)  # the reference and the effective-contact count follow the full life
        changed = int(np.count_nonzero(local.brain.weights - weights_before))
        change = live("relation", episode, True, True)
        change["lesson_changed_synapses"] = changed
        change["lesson_changed_synapses_above_1e-9"] = int(
            (np.abs(local.brain.weights - weights_before) > 1e-9).sum()
        )
        episodes.append({"kind": "relation", "episode": episode,
                         "change_index": change["index"], **hold("relation", episode)})
    comparisons = [m for m in moments if m["kind"] != "first"]
    gates = {
        "same_action": all(m["same_action"] for m in moments),
        "local_qualified": all(m["local"]["library_qualified"] for m in comparisons),
        "state_within_gate": all(m["state_gate"] for m in comparisons),
        "max_distance_activation": max(m["distance_activation"] for m in comparisons),
        "moments": len(moments),
        "disagreements": [m["index"] for m in moments if not m["same_action"]],
    }
    return {
        "spec": spec.to_dict(), "anatomy": anatomy, "acquisition": acquisition,
        "maps": [m.tolist() for m in maps], "episodes": episodes, "moments": moments,
        "gates": gates, "seconds": time.perf_counter() - started,
    }


def summarize(case: dict[str, Any]) -> dict[str, Any]:
    """Per-case totals: the change moments, whole episodes and stationary moments."""
    moments = case["moments"]
    E, n = case["anatomy"]["E"], case["anatomy"]["n"]

    def totals(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {}
        full = [m["full"] for m in selected]
        local = [m["local"] for m in selected]
        sync = [m["sync"] for m in selected]
        return {
            "moments": len(selected),
            "full_sweeps": sum(f["sweeps"] for f in full),
            "full_checks": sum(f["residual_checks"] for f in full),
            "full_updates": sum(f["updates"] for f in full),
            "full_contacts_nominal": sum(f["contacts_nominal"] for f in full),
            "full_contacts_effective": sum(f["contacts_effective"] for f in full),
            "full_seconds": sum(f["seconds"] for f in full),
            "local_rounds": sum(lo["rounds"] for lo in local),
            "local_updates": sum(lo["updates"] for lo in local),
            "local_messages": sum(lo["messages"] for lo in local),
            "local_checks": sum(lo["global_checks"] for lo in local),
            "local_contacts": sum(lo["contacts"] for lo in local),
            "local_check_contacts": E * sum(lo["global_checks"] for lo in local),
            "local_residual_evaluations": sum(lo["residual_evaluations"] for lo in local),
            "local_seconds": sum(lo["seconds"] for lo in local),
            "local_largest_cone": max(lo["cone"] for lo in local),
            "sync_sweeps": sum(s["sweeps"] for s in sync),
            "sync_updates": sum(s["updates"] for s in sync),
        }

    by_kind: dict[str, Any] = {}
    for kind in ("input", "relation"):
        changes = [m for m in moments if m["kind"] == kind]
        holds = [m for m in moments if m["kind"] == f"{kind}-hold"]
        by_kind[kind] = {"change": totals(changes), "hold": totals(holds)}
    stationary = [
        m for m in moments if m["kind"].endswith("-hold") and m["full"]["sweeps"] == 0
    ]
    return {
        "name": case["spec"]["name"], "arm": case["spec"]["arm"], "S": case["spec"]["S"],
        "widths": case["spec"]["widths"], "founder": case["spec"]["founder"],
        "n": n, "E": E, "block0_neurons": case["anatomy"]["block0_neurons"],
        "block0_contacts": case["anatomy"]["block0_contacts"],
        "acquired": case["acquisition"]["at_gate"], "lessons": case["acquisition"]["lessons"],
        "accuracy": case["acquisition"]["accuracy"],
        "row_mass": case["acquisition"]["certificate"]["row_mass"],
        "gates": case["gates"], "episodes": case["episodes"], "by_kind": by_kind,
        "stationary": totals(stationary), "seconds": case["seconds"],
    }


def environment() -> dict[str, Any]:
    import platform
    import sys

    out: dict[str, Any] = {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "cadence": cd.__version__, "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name in ("scipy", "numba", "torch"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "present")
        except ImportError:
            out[name] = None
    from cadence.brain import _FUSED

    out["fused_kernel"] = bool(_FUSED)
    return out


def case_specs(
    campaign: str, founders: Sequence[int], arms: Sequence[str] = ("dense", "sparse"),
    local_margin: float = 1.0,
) -> list[CaseSpec]:
    """The declared case grids: ``skills`` scales S at width 16; ``widths`` keeps two
    skills and widens the undisturbed one. A ``local_margin`` above one is amendment 1."""
    specs = []
    for founder in founders:
        for arm in arms:
            if campaign == "skills":
                for S in (1, 2, 4, 8):
                    specs.append(CaseSpec(arm, tuple([16] * S), founder,
                                          local_margin=local_margin))
            elif campaign == "widths":
                for width in (16, 64, 256, 1024):
                    specs.append(CaseSpec(arm, (16, width), founder, local_margin=local_margin))
            else:
                raise ValueError(f"unknown campaign {campaign!r}")
    return specs

