"""Operating-point controls and motor-only credit measurements for teaching."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .blocks import Layout
from .connectome import Connectome

if TYPE_CHECKING:
    from .brain import Brain, BrainState
    from .learning import Learner


@dataclass(frozen=True)
class CreditPlan:
    hidden: np.ndarray
    regions: tuple[tuple[str, np.ndarray], ...]

    @classmethod
    def build(cls, graph: Connectome, outputs: np.ndarray) -> CreditPlan:
        # A hidden unit has an input and can affect the answer. Include a
        # one-way branch here so a missing return route is visible in the report.
        reaches = np.zeros(graph.n, dtype=bool)
        reaches[outputs] = True
        front = reaches.copy()
        while front.any():
            previous = np.zeros_like(front)
            previous[graph.pre[front[graph.post]]] = True
            front = previous & ~reaches
            reaches |= front
        reaches[outputs] = False
        reaches &= np.bincount(graph.post, minlength=graph.n) > 0
        covered = np.zeros_like(reaches)
        named = []
        for name, members in graph.populations.items():
            if "/" in name:
                continue
            index = np.asarray(members, dtype=np.intp)
            index = index[reaches[index] & ~covered[index]]
            if len(index):
                named.append((name, index))
                covered[index] = True
        if (reaches & ~covered).any():
            named.append(("unnamed", np.flatnonzero(reaches & ~covered)))
        return cls(reaches, tuple(named))


def projection_groups(
    layout: Layout, paired: np.ndarray, reverse: np.ndarray,
    members: np.ndarray, member_groups: np.ndarray, group_count: int,
) -> np.ndarray:
    """Join blocks which must be scaled together to keep efficacy ties."""
    block = layout.block_of_edge
    parent = np.arange(layout.pairs, dtype=np.intp)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    if len(members):
        group_block = np.full(group_count, layout.pairs, dtype=np.intp)
        np.minimum.at(group_block, member_groups, block[members])
        linked = np.unique(np.stack((block[members], group_block[member_groups]), axis=1), axis=0)
        for first, second in linked:
            parent[root(int(first))] = root(int(second))
    if len(paired):
        keys = np.unique(block[paired] * layout.pairs + block[reverse])
        for key in keys:
            first, second = divmod(int(key), layout.pairs)
            parent[root(first)] = root(second)
    return np.fromiter((root(i) for i in range(layout.pairs)), dtype=np.intp)


def limit_projections(
    layout: Layout, scale: np.ndarray, gain: np.ndarray, activity: np.ndarray,
    adjustable: np.ndarray, measured: np.ndarray, groups: np.ndarray, voltage: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Let each incoming projection spend an equal share of the region's budget.

    Read the candidate effective weights on the lesson's free activities. Limit
    their adjustable drive on this batch, keeping each block's direction and
    sharing the tighter factor over ties. This is not a bound on arbitrary
    observations, held current, or the spectral gain of a recurrent loop.
    """
    if not measured.any():
        return scale, 0
    counts = np.bincount(layout.pair_post[measured], minlength=layout.ranges)
    effective = layout.flat(np.where(adjustable, scale * gain, 0.0))
    factors = np.ones(layout.pairs)
    for block in np.flatnonzero(measured):
        k = int(block)
        a0, a1, b0, b1 = layout.bounds(k)
        table = effective[layout.offset[k]:layout.offset[k + 1]].reshape(a1 - a0, b1 - b0)
        peak = float(np.max(np.abs(activity[:, a0:a1] @ table)))
        dest = layout.pair_post[k]
        share = voltage[dest] / counts[dest]
        if peak > share:
            factors[k] = share / peak
    limited = int(np.count_nonzero(factors < 1.0))
    if not limited:
        return scale, 0
    by_group = np.ones(layout.pairs)
    np.minimum.at(by_group, groups, factors)
    return np.where(adjustable, scale * by_group[groups[layout.block_of_edge]], scale), limited


def _reachable_from_motor(graph: Connectome, outputs: np.ndarray) -> np.ndarray:
    reach = np.zeros(graph.n, dtype=bool)
    reach[outputs] = True
    front = reach.copy()
    while front.any():
        next_hop = np.zeros_like(front)
        next_hop[graph.post[front[graph.pre]]] = True
        front = next_hop & ~reach
        reach |= front
    return reach


@dataclass
class _Probe:
    free: np.ndarray
    delta: np.ndarray
    held: np.ndarray
    responses: np.ndarray
    sweeps: int
    residual_checks: int


def _signed_probe(
    brain: Brain, drive: np.ndarray, pull: np.ndarray, pools: list[np.ndarray],
    *, budget: int, tolerance: float, warm: BrainState | None = None,
) -> tuple[_Probe | None, int, int]:
    from .brain import BrainState, Nudge, _on_host

    settled = brain.equilibrate(drive, state=warm, budget=budget, tolerance=tolerance, chunk=16)
    steps, checks = settled.steps, settled.residual_checks
    if not settled.qualified.all():
        return None, steps, checks
    held = brain.apical_gain(drive, settled.state)
    assert held is not None
    # Pure additive force on the motor, with the free state's gain held exactly
    # as in teaching. No desired class, hidden target, or optimizer is involved.
    def both(array: np.ndarray) -> np.ndarray:
        return np.concatenate((array, array), axis=0)

    zeros = np.zeros(drive.shape[1])
    hold = Nudge(zeros, zeros, 0.0, apical_gain=both(held))
    # Pair the signs in one solve, so early stopping at a different sweep
    # cannot make unrelated free drift masquerade as deep contrast.
    start = BrainState(both(_on_host(settled.state, "v")),
                       both(_on_host(settled.state, "activation")),
                       both(_on_host(settled.state, "adaptation")), steps=0)
    result = brain.equilibrate(
        np.concatenate((drive + pull, drive - pull)), state=start, nudge=hold,
        budget=budget, tolerance=tolerance, chunk=16,
    )
    steps += result.steps
    checks += result.residual_checks
    if not result.qualified.all():
        return None, steps, checks
    plus, minus = np.split(_on_host(result.state, "activation"), 2, axis=0)
    delta = plus - minus
    return _Probe(
        _on_host(settled.state, "activation"), delta, held,
        np.asarray([np.abs(delta[:, p]).mean() for p in pools]), steps, checks,
    ), steps, checks


def calibrate_feedback(
    learner: Learner, drive: np.ndarray, budget: int,
) -> tuple[Brain, dict[str, Any]]:
    """Bounded, label-free search over existing receiving pools, using actual settled response."""
    brain = learner.brain
    apical = brain.apical
    report: dict[str, Any] = dict(
        status="no_apical_feedback", probes=0, sweeps=0, residual_checks=0,
    )
    plan = learner._credit_plan()
    reach = _reachable_from_motor(brain.connectome, learner.output_index)
    assert learner.plastic_synapses is not None
    edges = learner.plastic_synapses
    if learner.synapse_rate is not None:
        edges = edges & (learner.synapse_rate > 0)
    plastic = np.bincount(brain.connectome.post[edges], minlength=brain.connectome.n) > 0
    active = plan.hidden & plastic
    report["unreachable"] = [
        name for name, index in plan.regions if (active[index] & ~reach[index]).any()
    ]
    if apical is None or not apical.members:
        return brain, report
    measured = [(name, index[active[index] & reach[index]]) for name, index in plan.regions]
    measured = [(name, index) for name, index in measured if len(index)]
    each = np.arange(brain.connectome.n)
    pools = [each[index] for index in apical.members]
    eligible = np.asarray([bool((active[index] & reach[index]).any()) for index in pools])
    if not measured or not eligible.any():
        report["status"] = "no_trainable_feedback_pool"
        return brain, report
    init = apical.fractions
    report["baseline_fractions"] = init.tolist()
    report["pools"] = [int(apical.groups[index[0]]) for index in pools]
    region_name = np.full(brain.connectome.n, "", dtype=object)
    for name, ids in brain.connectome.populations.items():
        if "/" not in name:
            region_name[np.asarray(ids, dtype=np.intp)] = name
    report["pool_names"] = [str(region_name[index[0]]) for index in pools]
    # Repeatable independent directions; center within a motor slot (do not
    # mistake common-mode gain for the ability to tell choices apart).
    direction = np.random.default_rng(819).normal(size=(len(drive), len(learner.output_index)))
    for start, size in zip(learner.slot_offsets, learner.slot_sizes, strict=True):
        group = direction[:, start:start + size]
        if size > 1:
            group -= group.mean(axis=1, keepdims=True)
    pull = np.zeros_like(drive)
    pull[:, learner.output_index] = direction * (
        0.004 / brain.neuron_model.slope / np.max(np.abs(direction), axis=1, keepdims=True))
    tolerance = min(learner.config.tolerance or 3e-3, float(np.abs(pull).max()) / 256)
    report["tolerance"] = tolerance
    units = pools + [index for _, index in measured] + [learner.output_index]

    def run(graph: Brain, fractions: np.ndarray) -> _Probe | None:
        probe, steps, checks = _signed_probe(graph, drive, pull, units, budget=budget,
                                             tolerance=tolerance)
        report["probes"] += 1
        report["sweeps"] += steps
        report["residual_checks"] += checks
        report.setdefault("candidates", []).append(dict(
            fractions=fractions.tolist(), qualified=probe is not None, sweeps=steps,
        ))
        return probe

    base = run(brain, init)
    if base is None:
        report["status"] = "baseline_unsettled"
        return brain, report
    floor = 1e-14
    def depth_score(probe: _Probe) -> float:
        # Compare absolute response to the same applied force, not to a motor
        # response the candidate could shrink just to improve a ratio.
        observed = np.sort(probe.responses[len(pools):-1])[:2]
        return float(np.log(np.maximum(observed, floor)).mean())

    def describe(probe: _Probe) -> dict[str, Any]:
        basal, feedback = brain._split_input(probe.delta, *brain._apical_transports())
        _, _, flat, _, pairs, _, _ = brain._apical_parts()
        hops = []
        if flat is not None:
            by_neuron = np.zeros_like(probe.delta)
            for k, ids in enumerate(pools):
                by_neuron[:, ids] = probe.held[:, k:k + 1]
            layout = brain.layout
            for k in pairs:
                a0, a1, b0, b1 = layout.bounds(k)
                table = flat[layout.offset[k]:layout.offset[k + 1]].reshape(a1 - a0, b1 - b0)
                incoming = probe.delta[:, a0:a1] @ table * by_neuron[:, b0:b1]
                hops.append({"pre": str(region_name[a0]), "post": str(region_name[b0]),
                             "drive": float(np.abs(incoming).mean())})
        return {"regions": {name: float(probe.responses[len(pools) + k])
                            for k, (name, _) in enumerate(measured)},
                "motor": float(probe.responses[-1]),
                "apical_projections": hops,
                "pool_feedback_drive": [
                    float(np.abs(feedback[:, ids] * probe.held[:, k:k + 1]).mean())
                    for k, ids in enumerate(pools)
                ],
                "pool_basal_drive": [float(np.abs(basal[:, ids]).mean()) for ids in pools]}

    report["baseline"] = describe(base)
    score = depth_score(base)
    selected, best, chosen = brain, base, init
    # Try common scaling and the weakest two receiving pools. Use the same
    # base inputs in every trial. Cap change per recheck and in total.
    weakest = sorted(np.flatnonzero(eligible), key=lambda i: base.responses[i])[:2]
    candidates = [np.flatnonzero(eligible), *(np.array([i]) for i in weakest)]
    used = {tuple(init)}
    for indexes in candidates:
        for factor in (1.6, 1.0 / 1.6):
            proposal = init.copy()
            proposal[indexes] = np.clip(proposal[indexes] * factor,
                                        apical.fraction / 3.0, apical.fraction * 3.0)
            ids = tuple(proposal)
            if ids in used:
                continue
            used.add(ids)
            graph = brain._with_apical(replace(apical, pool_fractions=tuple(map(float, proposal))))
            probe = run(graph, proposal)
            item = report["candidates"][-1]
            if probe is None:
                item["decision"] = "unsettled"
                continue
            drift = float(np.max(np.abs(probe.free - base.free)))
            value = depth_score(probe)
            item.update(score=value, max_free_drift=drift)
            makes_settle_harder = probe.sweeps > base.sweeps + 16
            reduces_motor = probe.responses[-1] < 0.5 * base.responses[-1]
            if drift > 0.15:
                item["decision"] = "free_state_shift"
            elif makes_settle_harder:
                item["decision"] = "extra_sweeps"
            elif reduces_motor:
                item["decision"] = "motor_response_loss"
            elif value > score + np.log(1.1):
                selected, best, chosen, score = graph, probe, proposal, value
                item["decision"] = "better"
            else:
                item["decision"] = "no_deep_gain"
    report["selected"] = describe(best)
    report["fractions"] = chosen.tolist()
    report["status"] = "adjusted" if selected is not brain else "no_safe_improvement"
    return selected, report
