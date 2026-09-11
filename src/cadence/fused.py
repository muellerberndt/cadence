"""A fused settlement kernel for dense wirings on the CPU.

The reference loop in ``settle`` does one NumPy operation per term per step
and pays interpreter and memory traffic for each. This kernel does the same
arithmetic, in the same order and the same float64, in one compiled loop:
transport as the block products of ``cadence.blocks`` (a block whose source
range did not move since the previous step is reused), then every owner's
repair, activation, adaptation, and nudge in place. It is used by ``Settlement`` on the CPU
backend when the wiring is dense and no trajectory is requested; the
conformance check and ``settle_owner_by_owner`` remain the reference it is
measured against. Nothing an owner could not see enters: the kernel reads
the same inbox, clamp, bias, and nudge the reference reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import numba as _numba

    njit: Any = _numba.njit
except ImportError:  # pragma: no cover
    njit = None

if TYPE_CHECKING:
    from .blocks import Layout
    from .rules import GradedRule
    from .settle import Nudge


__all__ = ["available", "fused_settle"]


def available() -> bool:
    return njit is not None


if njit is not None:

    @njit(cache=True)
    def _activation(v, slope, threshold, rest, leak, out):
        n = v.shape[0]
        scale_up = 1.0 / (1.0 - rest)
        scale_down = leak / rest if rest > 0.0 else 0.0
        for i in range(n):
            r = 1.0 / (1.0 + np.exp(-slope * (v[i] - threshold))) - rest
            if r > 0.0:
                out[i] = r * scale_up
            elif leak == 0.0:
                out[i] = 0.0
            else:
                out[i] = r * scale_down

    @njit(cache=True)
    def _act_scalar(x, slope, threshold, rest, leak):
        r = 1.0 / (1.0 + np.exp(-slope * (x - threshold))) - rest
        if r > 0.0:
            return r * (1.0 / (1.0 - rest))
        if leak == 0.0:
            return 0.0
        return r * (leak / rest)

    @njit(cache=True)
    def _kernel(
        v,
        a,
        s,
        standing,
        flat,
        starts,
        pair_pre,
        pair_post,
        offset,
        has_inbox,
        freezable,
        keep,
        masked,
        dt,
        slope,
        threshold,
        rest,
        leak,
        has_adapt,
        adapt_strength,
        adapt_tau,
        has_nudge,
        target,
        nmask,
        gid,
        ngroups,
        beta,
        softmax_t,
        weight,
        steps,
        tolerance,
        use_tolerance,
    ):
        batch, n = v.shape
        group = np.flatnonzero(nmask > 0.0)
        position = np.full(n, -1, dtype=np.int64)  # an output owner's place in the softmax group
        for k in range(group.shape[0]):
            position[group[k]] = k
        taken = 0
        p = np.empty(group.shape[0])
        zmax = np.empty(max(ngroups, 1))
        total = np.empty(max(ngroups, 1))
        # the block transport: one cached product per block, recomputed when its source range moved
        ranges = starts.shape[0] - 1
        pairs = pair_pre.shape[0]
        cache_offset = np.zeros(pairs + 1, dtype=np.int64)
        for k in range(pairs):
            b = pair_post[k]
            cache_offset[k + 1] = cache_offset[k] + batch * (starts[b + 1] - starts[b])
        cache = np.zeros(cache_offset[pairs])
        moved_range = np.ones(ranges, dtype=np.bool_)  # activation changed in the last repair pass
        frozen = np.zeros(ranges, dtype=np.bool_)  # a range that can never change again
        inbox = np.zeros((batch, n))
        for t in range(steps):
            for r in range(ranges):
                if has_inbox[r]:
                    for b in range(batch):
                        for i in range(starts[r], starts[r + 1]):
                            inbox[b, i] = 0.0
            for k in range(pairs):
                a0, a1 = starts[pair_pre[k]], starts[pair_pre[k] + 1]
                b0, b1 = starts[pair_post[k]], starts[pair_post[k] + 1]
                nb = b1 - b0
                c0 = cache_offset[k]
                if t == 0 or moved_range[pair_pre[k]]:
                    block = flat[offset[k] : offset[k + 1]].reshape(a1 - a0, nb)
                    product = np.dot(np.ascontiguousarray(s[:, a0:a1]), block)
                    for b in range(batch):
                        for j in range(nb):
                            cache[c0 + b * nb + j] = product[b, j]
                for b in range(batch):
                    for j in range(nb):
                        inbox[b, b0 + j] += cache[c0 + b * nb + j]
            for r in range(ranges):
                moved_range[r] = False
            moved = 0.0
            for b in range(batch):
                # the nudge on this row, from the activations before the step: one softmax per group
                if has_nudge and softmax_t > 0.0:
                    for g in range(ngroups):
                        zmax[g] = -1e300
                        total[g] = 0.0
                    for k in range(group.shape[0]):
                        g = gid[group[k]]
                        z = s[b, group[k]] / softmax_t
                        p[k] = z
                        if z > zmax[g]:
                            zmax[g] = z
                    for k in range(group.shape[0]):
                        g = gid[group[k]]
                        p[k] = np.exp(p[k] - zmax[g])
                        total[g] += p[k]
                    for k in range(group.shape[0]):
                        p[k] /= total[gid[group[k]]]
                for r in range(ranges):
                    if frozen[r]:
                        continue
                    for i in range(starts[r], starts[r + 1]):
                        tot = inbox[b, i] + standing[b, i]
                        if has_adapt:
                            tot -= adapt_strength * a[b, i]
                        if has_nudge and nmask[i] > 0.0 and softmax_t <= 0.0:
                            tot += beta * weight[b] * (target[b, i] - s[b, i])
                        tot -= v[b, i]
                        vn = v[b, i] + dt * tot
                        if has_nudge and softmax_t > 0.0 and nmask[i] > 0.0:
                            vn += dt * beta * weight[b] * (target[b, i] - p[position[i]])
                        if masked:
                            vn *= keep[i]
                        if (
                            vn != v[b, i]
                        ):  # an owner whose potential did not move publishes what it did
                            v[b, i] = vn
                            sn = _act_scalar(vn, slope, threshold, rest, leak)
                            if masked:
                                sn *= keep[i]
                            d = abs(sn - s[b, i])
                            if d > 0.0:
                                moved_range[r] = True
                                if d > moved:
                                    moved = d
                            s[b, i] = sn
                        if has_adapt:
                            a[b, i] = a[b, i] + (s[b, i] - a[b, i]) / adapt_tau
            for r in range(
                ranges
            ):  # a still range with nothing arriving stays still: skip it from now on
                if freezable[r] and not moved_range[r] and t > 0:
                    frozen[r] = True
            taken = t + 1
            if use_tolerance and moved < tolerance:
                break
        return taken


def fused_settle(
    v: np.ndarray,
    a: np.ndarray,
    drive: np.ndarray,
    bias: np.ndarray,
    layout: Layout,
    flat: np.ndarray,
    keep: np.ndarray,
    rule: GradedRule,
    nudge: Nudge | None,
    steps: int,
    tolerance: float | None,
    activation: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Run the fused kernel in place on ``v`` and ``a``; returns ``(s, taken)``.

    ``activation`` is the published activation that goes with ``v`` when the
    settlement continues from a state; it saves recomputing it."""
    assert njit is not None
    batch, n = v.shape
    standing = np.ascontiguousarray(drive + bias)
    masked = bool((keep != 1.0).any())
    if (
        activation is not None
    ):  # the state's own published activation: what the kernel computed last
        s = np.array(activation, dtype=float)
    elif not v.any():  # from rest every owner publishes the same thing
        row = np.empty(n)
        _activation(np.zeros(n), rule.slope, rule.threshold, rule.rest_emission, rule.leak, row)
        s = np.empty((batch, n))
        s[:] = row
    else:
        s = np.empty_like(v)
        for b in range(batch):
            _activation(v[b], rule.slope, rule.threshold, rule.rest_emission, rule.leak, s[b])
    if masked:
        s *= keep
    adapt = rule.adaptation
    has_adapt = adapt is not None
    if nudge is not None:
        target = np.ascontiguousarray(np.broadcast_to(nudge.target, (batch, n)).astype(float))
        nmask = np.asarray(nudge.mask, float)
        softmax_t = (
            float(nudge.softmax_temperature) if nudge.softmax_temperature is not None else 0.0
        )
        weight = np.ones(batch) if nudge.weight is None else np.asarray(nudge.weight, float)
        beta = float(nudge.beta)
        if nudge.groups is None:
            gid = np.where(nmask > 0, 0, -1).astype(np.int64)
            ngroups = 1
        else:
            raw = np.asarray(nudge.groups, dtype=np.int64)
            ids = np.unique(raw[raw >= 0])
            gid = np.full(n, -1, dtype=np.int64)
            for k, g in enumerate(ids):
                gid[raw == g] = k
            ngroups = len(ids)
    else:
        target = np.zeros((1, 1))
        nmask = np.zeros(n)
        gid = np.full(n, -1, dtype=np.int64)
        ngroups = 0
        softmax_t, weight, beta = 0.0, np.ones(batch), 0.0
    lay = layout
    has_inbox = np.zeros(lay.ranges, dtype=np.bool_)
    has_inbox[lay.pair_post] = True
    freezable = ~has_inbox  # nothing arrives: once still, still for good
    if has_adapt:
        freezable = np.zeros(lay.ranges, dtype=np.bool_)
    elif nudge is not None:
        nudged = np.searchsorted(lay.starts, np.flatnonzero(nmask > 0), side="right") - 1
        freezable[nudged] = False
    taken = _kernel(
        v,
        a,
        s,
        standing,
        flat,
        lay.starts,
        lay.pair_pre,
        lay.pair_post,
        lay.offset,
        has_inbox,
        freezable,
        np.asarray(keep, float),
        masked,
        rule.dt,
        rule.slope,
        rule.threshold,
        rule.rest_emission,
        rule.leak,
        has_adapt,
        0.0 if adapt is None else adapt.strength,
        1.0 if adapt is None else adapt.tau_steps,
        nudge is not None,
        target,
        nmask,
        gid,
        ngroups,
        beta,
        softmax_t,
        weight,
        int(steps),
        float(tolerance) if tolerance is not None else 0.0,
        tolerance is not None,
    )
    return s, taken


if njit is not None:

    @njit(cache=True)
    def _trace_step(
        trace, trace_bias, decay, s_plus, s_minus, pre, post, span, delta, step_scale, step_bias
    ):
        """One pass per row: contrast from the two phases, trace decay and accumulation, the
        dopamine-weighted sum into the step. ``trace`` is (batch, edges) and ``trace_bias``
        (batch, n)."""
        batch, edges = trace.shape
        n = trace_bias.shape[1]
        for e in range(edges):
            step_scale[e] = 0.0
        for i in range(n):
            step_bias[i] = 0.0
        for b in range(batch):
            d = delta[b] / batch
            for e in range(edges):
                c = (
                    s_plus[b, pre[e]] * s_plus[b, post[e]]
                    - s_minus[b, pre[e]] * s_minus[b, post[e]]
                ) / span
                t = decay * trace[b, e] + c
                trace[b, e] = t
                step_scale[e] += d * t
            for i in range(n):
                c = (s_plus[b, i] - s_minus[b, i]) / span
                t = decay * trace_bias[b, i] + c
                trace_bias[b, i] = t
                step_bias[i] += d * t


def trace_step(
    trace: np.ndarray,
    trace_bias: np.ndarray,
    decay: float,
    s_plus: np.ndarray,
    s_minus: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    span: float,
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused three-factor step: returns ``(step_scale, step_bias)``; traces are updated in place."""
    assert njit is not None
    step_scale = np.empty(trace.shape[1])
    step_bias = np.empty(trace_bias.shape[1])
    _trace_step(
        trace,
        trace_bias,
        float(decay),
        s_plus,
        s_minus,
        pre,
        post,
        float(span),
        np.ascontiguousarray(delta, dtype=np.float64),
        step_scale,
        step_bias,
    )
    return step_scale, step_bias


if njit is not None:

    @njit(cache=True)
    def _contrast_mean(s_plus, s_minus, pre, post, span, out_edges, out_owners):
        """Batch-mean contrast per overlap and per owner, one pass, no (batch, edges) temporary."""
        batch, n = s_plus.shape
        edges = pre.shape[0]
        for e in range(edges):
            out_edges[e] = 0.0
        for i in range(n):
            out_owners[i] = 0.0
        for b in range(batch):
            for e in range(edges):
                out_edges[e] += (
                    s_plus[b, pre[e]] * s_plus[b, post[e]]
                    - s_minus[b, pre[e]] * s_minus[b, post[e]]
                )
            for i in range(n):
                out_owners[i] += s_plus[b, i] - s_minus[b, i]
        scale = 1.0 / (batch * span)
        for e in range(edges):
            out_edges[e] *= scale
        for i in range(n):
            out_owners[i] *= scale


def contrast_mean(
    s_plus: np.ndarray, s_minus: np.ndarray, pre: np.ndarray, post: np.ndarray, span: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fused ``Learner.contrast``: ``(per_overlap, per_owner)`` batch means divided by ``span``."""
    assert njit is not None
    out_edges = np.empty(pre.shape[0])
    out_owners = np.empty(s_plus.shape[1])
    _contrast_mean(
        np.ascontiguousarray(s_plus),
        np.ascontiguousarray(s_minus),
        pre,
        post,
        float(span),
        out_edges,
        out_owners,
    )
    return out_edges, out_owners
