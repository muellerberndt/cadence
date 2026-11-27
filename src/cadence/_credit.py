"""Operating-point controls and motor-only credit measurements for teaching."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .blocks import Layout
from .connectome import Connectome

if TYPE_CHECKING:
    from .brain import Brain, BrainState, Equilibrium, Nudge
    from .learning import Learner


@dataclass(frozen=True)
class CreditPlan:
    hidden: np.ndarray
    regions: tuple[tuple[str, np.ndarray], ...]
    motor_distance: np.ndarray

    @classmethod
    def build(cls, graph: Connectome, outputs: np.ndarray) -> CreditPlan:
        # A hidden unit has an input and can affect the answer. Include a
        # one-way branch here so a missing return route is visible in the report.
        live = graph.count > 0
        pre, post = graph.pre[live], graph.post[live]
        reaches = _distances(graph.n, outputs, post, pre) >= 0
        distance = _distances(graph.n, outputs, pre, post)
        reaches[outputs] = False
        reaches &= np.bincount(post, minlength=graph.n) > 0
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
        return cls(reaches, tuple(named), distance)


def _distances(n: int, start: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """The shortest directed distances along physical contacts, or -1."""
    distance = np.full(n, -1, dtype=np.intp)
    distance[start] = 0
    front = distance == 0
    depth = 0
    while front.any():
        depth += 1
        next_hop = np.zeros(n, dtype=bool)
        next_hop[post[front[pre]]] = True
        front = next_hop & (distance < 0)
        distance[front] = depth
    return distance


def center_bias(voltage: np.ndarray, slope: float, threshold: float, rate: float) -> np.ndarray:
    """A bounded local step against sustained excursions outside a responsive range.

    Compute in logits relative to rest and the sigmoid midpoint. Keep normal
    positive responses unchanged and avoid resting all participating cells
    exactly on the sigmoid's re-basing seam, where the two slopes disagree.
    """
    low = 0.1 / slope
    high = max(low, threshold + 1.5 / slope)
    error = voltage - np.clip(voltage, low, high)
    # Clip each row before averaging: one mistaken operating point in a
    # batch must not erase the evidence from all the other rows.
    return np.asarray(-rate * np.clip(error, -2.0 / slope, 2.0 / slope).mean(axis=0))


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


@dataclass
class _Probe:
    free: np.ndarray
    potential: np.ndarray
    delta: np.ndarray
    held: np.ndarray | None
    responses: np.ndarray
    sweeps: int
    residual_checks: int


def transport_contrast(
    brain: Brain, drive: np.ndarray, free: BrainState, pull: Nudge, limit: float,
    budget: int,
) -> tuple[tuple[Equilibrium, Equilibrium] | None, dict[str, float]]:
    """Transport a small motor contrast without amplifying the standing return.

    With the free apical gain held, add (factor-1) times the existing return
    of (s-s_free). At zero motor force the modified phase still has the same
    free equilibrium. Both signs use identical feedback and are solved together.
    """
    from .brain import BrainState, _host_copy, _on_host, _resident

    work = {"credit_transport_factor": 0., "credit_transport_trials": 0.,
            "credit_transport_row_sweeps": 0., "credit_transport_checks": 0.}
    apical = brain.apical
    if apical is None or pull.apical_gain is None or pull.anchor_gain is not None or budget < 1:
        return None, work
    held = _host_copy(pull.apical_gain) if _resident(pull.apical_gain) else pull.apical_gain
    voltage, activity, adaptation = (
        _on_host(free, key) for key in ("v", "activation", "adaptation")
    )
    force = np.max(np.abs(pull.drive(activity)), axis=1)
    if not (force > 0).any():
        return None, work
    tolerance = min(1e-6, float(force[force > 0].min()) / 128.)
    inputs = _host_copy(drive) if _resident(drive) else drive
    _, raw = brain._split_input(activity, *brain._apical_transports())
    expected = apical.drive(raw, raw, held)

    def both(value: np.ndarray) -> np.ndarray:
        return np.concatenate((value, value), axis=0)

    warm = BrainState(both(voltage), both(activity), both(adaptation), steps=0)
    rows = len(activity)
    weights = np.ones(rows) if pull.weight is None else pull.weight
    # Keep halving above the unamplified control. With the maximum supported limit of six,
    # this tries at most four candidates and no longer skips 1.5 when both six and three fail.
    factor = float(limit)
    factors = []
    while factor > 1.:
        factors.append(factor)
        factor *= .5
    for factor in (*factors, 1.):
        pair = replace(
            pull, target=both(np.broadcast_to(pull.target, activity.shape)),
            weight=np.concatenate((weights, -weights)), apical_gain=factor * both(held),
        )
        result = brain.equilibrate(
            both(inputs - (factor - 1.) * expected), state=warm,
            nudge=pair, budget=budget, tolerance=tolerance, chunk=16,
        )
        work["credit_transport_trials"] += 1.
        sweeps = result.row_sweeps if result.row_sweeps is not None else 2 * rows * result.steps
        work["credit_transport_row_sweeps"] += float(sweeps)
        work["credit_transport_checks"] += float(result.residual_checks)
        if not result.qualified.all():
            continue
        sv, sa, sd = (_on_host(result.state, name) for name in ("v", "activation", "adaptation"))
        if np.max(np.abs(sa - warm.activation)) >= .01:
            continue
        motor = pull.mask != 0
        moved = np.any(sv[:rows, motor] != sv[rows:, motor], axis=1)
        if ((force > 0) & ~moved).any():
            continue
        slices = (slice(0, rows), slice(rows, None))
        phases = []
        for window in slices:
            state = BrainState(sv[window], sa[window], sd[window], steps=result.steps)
            phases.append(replace(result, state=state, residual=result.residual[window],
                                  row_sweeps=rows * result.steps))
        work["credit_transport_factor"] = factor
        return (phases[0], phases[1]), work
    return None, work


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
    # Pure additive force on the motor, with the free state's gain held exactly
    # as in teaching. No desired class, hidden target, or optimizer is involved.
    def both(array: np.ndarray) -> np.ndarray:
        return np.concatenate((array, array), axis=0)

    zeros = np.zeros(drive.shape[1])
    hold = None if held is None else Nudge(zeros, zeros, 0.0, apical_gain=both(held))
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
        _on_host(settled.state, "activation"), _on_host(settled.state, "v"), delta, held,
        np.asarray([np.abs(delta[:, p]).mean() for p in pools]), steps, checks,
    ), steps, checks


def calibrate_feedback(
    learner: Learner, drive: np.ndarray, budget: int,
) -> tuple[Brain, dict[str, Any]]:
    """Bounded, label-free search using the free and motor-perturbed equilibria."""
    brain = learner.brain
    apical = brain.apical
    report: dict[str, Any] = dict(
        status="no_trainable_feedback_path", probes=0, sweeps=0, residual_checks=0,
    )
    plan = learner._credit_plan()
    reach = plan.motor_distance >= 0
    assert learner.plastic_synapses is not None
    edges = learner.plastic_synapses
    if learner.synapse_rate is not None:
        edges = edges & (learner.synapse_rate > 0)
    stepping = learner._stepping_neurons()
    cn = brain.connectome
    plastic = (np.bincount(cn.post[edges], minlength=cn.n) > 0) | stepping
    active = plan.hidden & plastic
    report["unreachable"] = [
        name for name, index in plan.regions if (active[index] & ~reach[index]).any()
    ]
    measured = [(name, index[active[index] & reach[index]]) for name, index in plan.regions]
    measured = [(name, index) for name, index in measured if len(index)]
    if not measured:
        return brain, report
    each = np.arange(brain.connectome.n)
    pools = [] if apical is None else [each[index] for index in apical.members]
    eligible = np.asarray([bool((active[index] & reach[index]).any()) for index in pools])
    init = np.zeros(0) if apical is None else apical.fractions
    report["baseline_fractions"] = init.tolist()
    report["pools"] = [] if apical is None else [int(apical.groups[index[0]]) for index in pools]
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
        group -= group.mean(axis=1, keepdims=True)
    peak = np.max(np.abs(direction), axis=1, keepdims=True)
    if not np.all(peak > 0):
        report["status"] = "no_motor_choice_to_probe"
        return brain, report
    pull = np.zeros_like(drive)
    pull[:, learner.output_index] = direction * (
        0.004 / brain.neuron_model.slope / peak)
    tolerance = min(learner.config.tolerance or 3e-3, float(np.abs(pull).max()) / 256)
    report["tolerance"] = tolerance
    units = pools + [index for _, index in measured] + [learner.output_index]

    def run(graph: Brain, fractions: np.ndarray, edit: str, centered: int = 0) -> _Probe | None:
        probe, steps, checks = _signed_probe(graph, drive, pull, units, budget=budget,
                                             tolerance=tolerance)
        report["probes"] += 1
        report["sweeps"] += steps
        report["residual_checks"] += checks
        report.setdefault("candidates", []).append(dict(
            edit=edit, fractions=fractions.tolist(), centered_neurons=centered,
            qualified=probe is not None, sweeps=steps,
        ))
        return probe

    base = run(brain, init, "baseline")
    if base is None:
        report["status"] = "baseline_unsettled"
        return brain, report
    floor = 1e-14
    if base.responses[-1] <= floor:
        report["status"] = "no_motor_probe_response"
        return brain, report
    def depth_score(probe: _Probe) -> float:
        # Compare absolute response to the same applied force, not to a motor
        # response the candidate could shrink just to improve a ratio.
        observed = np.sort(probe.responses[len(pools):-1])[:2]
        return float(np.log(np.maximum(observed, floor)).mean())

    def motor_answer(probe: _Probe) -> np.ndarray:
        motor = probe.free[:, learner.output_index]
        return np.stack([
            motor[:, start:start + size].argmax(axis=1)
            for start, size in zip(learner.slot_offsets, learner.slot_sizes, strict=True)
        ], axis=1)

    def describe(probe: _Probe) -> dict[str, Any]:
        result: dict[str, Any] = {
            "regions": {name: float(probe.responses[len(pools) + k])
                        for k, (name, _) in enumerate(measured)},
            "motor": float(probe.responses[-1]),
            "slopes": {name: float(brain.neuron_model.slope_at(probe.potential[:, ids]).mean())
                       for name, ids in measured},
        }
        held = probe.held
        if held is None:
            return result
        basal, feedback = brain._split_input(probe.delta, *brain._apical_transports())
        _, _, flat, _, pairs, _, _ = brain._apical_parts()
        hops = []
        if flat is not None:
            by_neuron = np.zeros_like(probe.delta)
            for k, ids in enumerate(pools):
                by_neuron[:, ids] = held[:, k:k + 1]
            layout = brain.layout
            for k in pairs:
                a0, a1, b0, b1 = layout.bounds(k)
                table = flat[layout.offset[k]:layout.offset[k + 1]].reshape(a1 - a0, b1 - b0)
                incoming = probe.delta[:, a0:a1] @ table * by_neuron[:, b0:b1]
                hops.append({"pre": str(region_name[a0]), "post": str(region_name[b0]),
                             "drive": float(np.abs(incoming).mean())})
        result.update(
            apical_projections=hops,
            pool_feedback_drive=[
                float(np.abs(feedback[:, ids] * held[:, k:k + 1]).mean())
                for k, ids in enumerate(pools)
            ],
            pool_basal_drive=[float(np.abs(basal[:, ids]).mean()) for ids in pools],
        )
        return result

    report["baseline"] = describe(base)
    score = depth_score(base)
    selected, best, chosen = brain, base, init
    base_answer = motor_answer(base)
    chosen_edit = "none"

    def consider(
        graph: Brain, fractions: np.ndarray, edit: str, centered: int = 0,
    ) -> None:
        nonlocal selected, best, chosen, score, chosen_edit
        probe = run(graph, fractions, edit, centered)
        item = report["candidates"][-1]
        if probe is None:
            item["decision"] = "unsettled"
            return
        drift = float(np.max(np.abs(probe.free - base.free)))
        value = depth_score(probe)
        flips = float(np.mean(motor_answer(probe) != base_answer))
        item.update(score=value, max_free_drift=drift, motor_slot_flip=flips)
        makes_settle_harder = probe.sweeps > base.sweeps + 16
        reduces_motor = probe.responses[-1] < 0.5 * base.responses[-1]
        if drift > 0.15:
            item["decision"] = "free_state_shift"
        elif flips > 0.0:
            item["decision"] = "free_motor_choice_changed"
        elif makes_settle_harder:
            item["decision"] = "extra_sweeps"
        elif reduces_motor:
            item["decision"] = "motor_response_loss"
        elif value > score + np.log(1.1):
            selected, best, chosen, score = graph, probe, fractions, value
            chosen_edit = edit
            item["decision"] = "better"
        else:
            item["decision"] = "no_deep_gain"

    # Try common scaling and the weakest two receiving pools. Use the same
    # base inputs in every trial. Cap change per recheck and in total.
    if apical is not None and eligible.any():
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
                graph = brain._with_apical(
                    replace(apical, pool_fractions=tuple(map(float, proposal))))
                consider(graph, proposal, "apical_gain")

    # After the gain search, try a local rest-point change, including on
    # graphs with no apical compartment. The free and both signed solves
    # see the same candidate bias. Neither targets nor optimizer state enter.
    adjustable = reach & plan.hidden & stepping
    adjust_regions = sorted(
        (i for i, (_, region) in enumerate(measured) if adjustable[region].any()),
        key=lambda i: base.responses[len(pools) + i],
    )[:2]
    if adjust_regions:
        bias_ids = np.concatenate([
            measured[i][1][adjustable[measured[i][1]]] for i in adjust_regions
        ])
        rule = brain.neuron_model
        start, start_probe, fractions = selected, best, chosen
        for rate in (0.05, 0.25):
            step = center_bias(start_probe.potential[:, bias_ids], rule.slope, rule.threshold, rate)
            if step.any():
                bias = start.bias.copy()
                bias[bias_ids] += step
                consider(start.with_parameters(bias=bias), fractions, "center_bias",
                         int(np.count_nonzero(step)))
    report["selected"] = describe(best)
    report["fractions"] = chosen.tolist()
    report["selected_edit"] = chosen_edit
    report["status"] = "adjusted" if selected is not brain else "no_safe_improvement"
    return selected, report
