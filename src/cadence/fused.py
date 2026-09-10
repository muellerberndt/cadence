"""A fused settlement kernel for dense wirings on the CPU.

The reference loop in ``settle`` does one NumPy operation per term per step
and pays interpreter and memory traffic for each. This kernel does the same
arithmetic, in the same order and the same float64, in one compiled loop:
transport as one matrix product, then every owner's repair, activation,
adaptation, and nudge in place. It is used by ``Settlement`` on the CPU
backend when the wiring is dense and no trajectory is requested; the
conformance check and ``settle_owner_by_owner`` remain the reference it is
measured against. Nothing an owner could not see enters: the kernel reads
the same inbox, clamp, bias, and nudge the reference reads.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

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
    def _kernel(v, a, s, standing, dense, keep, masked, dt, slope, threshold, rest, leak, has_adapt, adapt_strength, adapt_tau, has_nudge, target, nmask, gid, ngroups, beta, softmax_t, weight, steps, tolerance, use_tolerance):
        batch, n = v.shape
        group = np.flatnonzero(nmask > 0.0)
        taken = 0
        previous = np.empty(n)
        p = np.empty(group.shape[0])
        zmax = np.empty(max(ngroups, 1))
        total = np.empty(max(ngroups, 1))
        for t in range(steps):
            inbox = np.dot(s, dense)
            moved = 0.0
            for b in range(batch):
                # the nudge on this row, from the activations before the step: one softmax per group
                if has_nudge:
                    if softmax_t > 0.0:
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
                for i in range(n):
                    tot = inbox[b, i] + standing[b, i]
                    if has_adapt:
                        tot -= adapt_strength * a[b, i]
                    if has_nudge and nmask[i] > 0.0:
                        if softmax_t > 0.0:
                            pass
                        else:
                            tot += beta * weight[b] * (target[b, i] - s[b, i])
                    tot -= v[b, i]
                    v[b, i] = v[b, i] + dt * tot
                if has_nudge and softmax_t > 0.0:
                    for k in range(group.shape[0]):
                        i = group[k]
                        v[b, i] += dt * beta * weight[b] * (target[b, i] - p[k])
                if masked:
                    for i in range(n):
                        v[b, i] *= keep[i]
                for i in range(n):
                    previous[i] = s[b, i]
                _activation(v[b], slope, threshold, rest, leak, s[b])
                if masked:
                    for i in range(n):
                        s[b, i] *= keep[i]
                if has_adapt:
                    for i in range(n):
                        a[b, i] = a[b, i] + (s[b, i] - a[b, i]) / adapt_tau
                for i in range(n):
                    d = abs(s[b, i] - previous[i])
                    if d > moved:
                        moved = d
            taken = t + 1
            if use_tolerance and moved < tolerance:
                break
        return taken


def fused_settle(v, a, drive, bias, dense, keep, rule, nudge, steps, tolerance):
    """Run the fused kernel in place on ``v`` and ``a``; returns ``(s, taken)``."""
    assert njit is not None
    batch, n = v.shape
    standing = np.ascontiguousarray(drive + bias)
    masked = bool((keep != 1.0).any())
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
        softmax_t = float(nudge.softmax_temperature) if nudge.softmax_temperature is not None else 0.0
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
    taken = _kernel(
        v, a, s, standing, dense, np.asarray(keep, float), masked, rule.dt, rule.slope, rule.threshold, rule.rest_emission, rule.leak,
        has_adapt, adapt.strength if has_adapt else 0.0, adapt.tau_steps if has_adapt else 1.0,
        nudge is not None, target, nmask, gid, ngroups, beta, softmax_t, weight, int(steps), float(tolerance) if tolerance is not None else 0.0, tolerance is not None,
    )
    return s, taken


if njit is not None:

    @njit(cache=True)
    def _trace_step(trace, trace_bias, decay, s_plus, s_minus, pre, post, span, delta, step_scale, step_bias):
        """One pass per row: contrast from the two phases, trace decay and accumulation, the
        dopamine-weighted sum into the step. ``trace`` is (batch, edges), ``trace_bias`` (batch, n)."""
        batch, edges = trace.shape
        n = trace_bias.shape[1]
        for e in range(edges):
            step_scale[e] = 0.0
        for i in range(n):
            step_bias[i] = 0.0
        for b in range(batch):
            d = delta[b] / batch
            for e in range(edges):
                c = (s_plus[b, pre[e]] * s_plus[b, post[e]] - s_minus[b, pre[e]] * s_minus[b, post[e]]) / span
                t = decay * trace[b, e] + c
                trace[b, e] = t
                step_scale[e] += d * t
            for i in range(n):
                c = (s_plus[b, i] - s_minus[b, i]) / span
                t = decay * trace_bias[b, i] + c
                trace_bias[b, i] = t
                step_bias[i] += d * t


def trace_step(trace, trace_bias, decay, s_plus, s_minus, pre, post, span, delta):
    """Fused three-factor step: returns ``(step_scale, step_bias)``; traces are updated in place."""
    assert njit is not None
    step_scale = np.empty(trace.shape[1])
    step_bias = np.empty(trace_bias.shape[1])
    _trace_step(trace, trace_bias, float(decay), s_plus, s_minus, pre, post, float(span), np.ascontiguousarray(delta, dtype=np.float64), step_scale, step_bias)
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
                out_edges[e] += s_plus[b, pre[e]] * s_plus[b, post[e]] - s_minus[b, pre[e]] * s_minus[b, post[e]]
            for i in range(n):
                out_owners[i] += s_plus[b, i] - s_minus[b, i]
        scale = 1.0 / (batch * span)
        for e in range(edges):
            out_edges[e] *= scale
        for i in range(n):
            out_owners[i] *= scale


def contrast_mean(s_plus, s_minus, pre, post, span):
    """Fused ``Learner.contrast``: returns ``(per_overlap, per_owner)`` batch means divided by ``span``."""
    assert njit is not None
    out_edges = np.empty(pre.shape[0])
    out_owners = np.empty(s_plus.shape[1])
    _contrast_mean(np.ascontiguousarray(s_plus), np.ascontiguousarray(s_minus), pre, post, float(span), out_edges, out_owners)
    return out_edges, out_owners
