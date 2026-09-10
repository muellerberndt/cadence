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
    def _kernel(v, a, s, standing, dense, keep, masked, dt, slope, threshold, rest, leak, has_adapt, adapt_strength, adapt_tau, has_nudge, target, nmask, beta, softmax_t, weight, steps, tolerance, use_tolerance):
        batch, n = v.shape
        group = np.flatnonzero(nmask > 0.0)
        taken = 0
        previous = np.empty(n)
        p = np.empty(group.shape[0])
        for t in range(steps):
            inbox = np.dot(s, dense)
            moved = 0.0
            for b in range(batch):
                # the nudge on this row, from the activations before the step
                if has_nudge:
                    if softmax_t > 0.0:
                        zmax = -1e300
                        for k in range(group.shape[0]):
                            z = s[b, group[k]] / softmax_t
                            p[k] = z
                            if z > zmax:
                                zmax = z
                        total = 0.0
                        for k in range(group.shape[0]):
                            p[k] = np.exp(p[k] - zmax)
                            total += p[k]
                        for k in range(group.shape[0]):
                            p[k] /= total
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
    else:
        target = np.zeros((1, 1))
        nmask = np.zeros(n)
        softmax_t, weight, beta = 0.0, np.ones(batch), 0.0
    taken = _kernel(
        v, a, s, standing, dense, np.asarray(keep, float), masked, rule.dt, rule.slope, rule.threshold, rule.rest_emission, rule.leak,
        has_adapt, adapt.strength if has_adapt else 0.0, adapt.tau_steps if has_adapt else 1.0,
        nudge is not None, target, nmask, beta, softmax_t, weight, int(steps), float(tolerance) if tolerance is not None else 0.0, tolerance is not None,
    )
    return s, taken
