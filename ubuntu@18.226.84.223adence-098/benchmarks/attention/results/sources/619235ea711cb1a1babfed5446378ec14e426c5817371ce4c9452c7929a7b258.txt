"""A fused settling kernel for dense connectomes on the CPU.

The reference loop in ``settle`` does one NumPy operation per term per step
and pays interpreter and memory traffic for each. This kernel does the same
arithmetic, in the same order and the same float64, in one compiled loop:
transport as the block products of ``cadence.blocks`` (a block whose source
range did not move since the previous step is reused), then every neuron's
update, activation, adaptation, and nudge in place. ``Brain`` uses it on the CPU backend
when Numba is installed, the connectome's blocks fit ``dense_limit``, and no trajectory or
settlement recording is requested (``CADENCE_FUSED=0`` forces the NumPy loop); the
conformance check and ``settle_neuron_by_neuron`` are the reference it is
measured against. Nothing a neuron could not see enters: the kernel reads
the same synaptic input, stimulus, bias, and nudge the reference reads.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

try:
    import numba as _numba

    njit: Any = _numba.njit
except ImportError:  # pragma: no cover
    njit = None

if TYPE_CHECKING:
    from .blocks import Layout
    from .brain import Nudge
    from .neuron import NeuronModel


__all__ = ["available", "fused_residual", "fused_settle"]

_Kernel = TypeVar("_Kernel", bound=Callable[..., Any])


def _compiled(kernel: _Kernel) -> _Kernel:
    """``numba.njit(cache=True)``, keeping the kernel's annotated signature for the checker."""
    assert njit is not None
    return cast(_Kernel, njit(cache=True)(kernel))


def available() -> bool:
    return njit is not None


if njit is not None:

    @_compiled
    def _activation(
        v: np.ndarray,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        out: np.ndarray,
    ) -> None:
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

    @_compiled
    def _act_scalar(
        x: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
    ) -> float:
        r: float = 1.0 / (1.0 + np.exp(-slope * (x - threshold))) - rest
        if r > 0.0:
            return r * (1.0 / (1.0 - rest))
        if leak == 0.0:
            return 0.0
        return r * (leak / rest)

    @_compiled
    def _softmax_groups(
        s: np.ndarray, group: np.ndarray, gid: np.ndarray, ngroups: int,
        temperatures: np.ndarray, p: np.ndarray, zmax: np.ndarray, total: np.ndarray,
    ) -> None:
        for g in range(ngroups):
            zmax[g] = -np.inf
            total[g] = 0.0
        finite = True
        for k in range(group.shape[0]):
            i = group[k]
            z = s[i] / temperatures[i]
            p[k] = z
            finite = finite and np.isfinite(z)
            if z > zmax[gid[i]]:
                zmax[gid[i]] = z
        if not finite:
            # A subnormal temperature may overflow raw logits. Center first only
            # in that case, preserving the ordinary scalar path's arithmetic.
            for g in range(ngroups):
                zmax[g] = -np.inf
            for k in range(group.shape[0]):
                i = group[k]
                if s[i] > zmax[gid[i]]:
                    zmax[gid[i]] = s[i]
            for k in range(group.shape[0]):
                i = group[k]
                p[k] = (s[i] - zmax[gid[i]]) / temperatures[i]
            for g in range(ngroups):
                zmax[g] = 0.0
        for k in range(group.shape[0]):
            g = gid[group[k]]
            p[k] = np.exp(p[k] - zmax[g])
            total[g] += p[k]
        for k in range(group.shape[0]):
            p[k] /= total[gid[group[k]]]

    @_compiled
    def _kernel(
        v: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        flat: np.ndarray,
        starts: np.ndarray,
        pair_pre: np.ndarray,
        pair_post: np.ndarray,
        offset: np.ndarray,
        has_synaptic_input: np.ndarray,
        freezable: np.ndarray,
        keep: np.ndarray,
        masked: bool,
        dt: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        has_adapt: bool,
        adapt_strength: float,
        adapt_tau: float,
        has_nudge: bool,
        target: np.ndarray,
        nmask: np.ndarray,
        gid: np.ndarray,
        ngroups: int,
        beta: float,
        softmax_t: np.ndarray,
        weight: np.ndarray,
        anchor: np.ndarray,
        anchor_gain: np.ndarray,
        steps: int,
        tolerance: float,
        use_tolerance: bool,
    ) -> tuple[int, np.ndarray]:
        batch, n = v.shape
        group = np.flatnonzero(nmask > 0.0)
        position = np.full(n, -1, dtype=np.int64)  # an output neuron's place in the softmax group
        for k in range(group.shape[0]):
            position[group[k]] = k
        taken = 0
        activity_change = np.zeros(batch)
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
        moved_range = np.ones(ranges, dtype=np.bool_)  # activation changed in the last step
        moved_potential = np.ones(ranges, dtype=np.bool_)
        frozen = np.zeros(ranges, dtype=np.bool_)  # a range that can never change again
        synaptic_input = np.zeros((batch, n))
        for t in range(steps):
            for r in range(ranges):
                if has_synaptic_input[r]:
                    for b in range(batch):
                        for i in range(starts[r], starts[r + 1]):
                            synaptic_input[b, i] = 0.0
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
                        synaptic_input[b, b0 + j] += cache[c0 + b * nb + j]
            for r in range(ranges):
                moved_range[r] = False
                moved_potential[r] = False
            moved = 0.0
            for b in range(batch):
                # the nudge on this row, from the activations before the step: one softmax per group
                if has_nudge and softmax_t[0] > 0.0:
                    _softmax_groups(s[b], group, gid, ngroups, softmax_t, p, zmax, total)
                for r in range(ranges):
                    if frozen[r]:
                        continue
                    for i in range(starts[r], starts[r + 1]):
                        tot = synaptic_input[b, i] + standing[b, i]
                        if has_adapt:
                            tot -= adapt_strength * a[b, i]
                        if has_nudge and nmask[i] > 0.0 and softmax_t[0] <= 0.0:
                            tot += beta * weight[b] * nmask[i] * (target[b, i] - s[b, i])
                        if anchor_gain[i] > 0.0:
                            tot += anchor_gain[i] * (anchor[b, i] - s[b, i])
                        tot -= v[b, i]
                        vn = v[b, i] + dt * tot
                        if has_nudge and softmax_t[0] > 0.0 and nmask[i] > 0.0:
                            vn += dt * beta * weight[b] * (target[b, i] - p[position[i]]) * nmask[i]
                        if masked:
                            vn *= keep[b, i]
                        if (
                            vn != v[b, i]
                        ):  # a neuron whose potential did not move publishes what it did
                            moved_potential[r] = True
                            v[b, i] = vn
                            sn = _act_scalar(vn, slope, threshold, rest, leak)
                            if masked:
                                sn *= keep[b, i]
                            d = abs(sn - s[b, i])
                            activity_change[b] += d
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
                if freezable[r] and not moved_potential[r] and t > 0:
                    frozen[r] = True
            taken = t + 1
            if use_tolerance and moved < tolerance:
                break
        return taken, activity_change


def _nudge_arrays(
    nudge: Nudge | None, batch: int, n: int
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """The nudge as the dense arrays the kernels read: target, mask, group id per neuron
    (0..ngroups-1, -1 for none), the group count, beta, the per-neuron softmax temperatures (0 for a
    plain nudge) and the per-row weight. A softmax nudge drives only grouped neurons."""
    if nudge is None:
        return (
            np.zeros((1, 1)),
            np.zeros(n),
            np.full(n, -1, dtype=np.int64),
            0,
            0.0,
            np.zeros(n),
            np.ones(batch),
            np.zeros((1, 1)),
            np.zeros(n),
        )
    target = np.ascontiguousarray(np.broadcast_to(nudge.target, (batch, n)).astype(float))
    nmask = np.asarray(nudge.mask, float)
    softmax_t = (
        np.zeros(n) if nudge.softmax_temperature is None
        else np.full(n, nudge.softmax_temperature)
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
    if softmax_t[0] > 0:
        nmask = np.where(gid >= 0, nmask, 0.0)
    anchor, anchor_gain = np.zeros((1, 1)), np.zeros(n)
    if nudge.anchor is not None:
        anchor = np.ascontiguousarray(np.broadcast_to(nudge.anchor, (batch, n)), dtype=float)
        anchor_gain = np.ascontiguousarray(nudge.anchor_gain, dtype=float)
    return target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain


def fused_settle(
    v: np.ndarray,
    a: np.ndarray,
    drive: np.ndarray,
    bias: np.ndarray,
    layout: Layout,
    flat: np.ndarray,
    keep: np.ndarray,
    neuron_model: NeuronModel,
    nudge: Nudge | None,
    steps: int,
    tolerance: float | None,
    activation: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Run the fused kernel in place on ``v`` and ``a``; returns ``(s, taken, activity_change)``.

    ``activation`` is accepted and ignored. Publication is recomputed
    from ``v`` under the current mask, since cached readback may use an old mask.
    """
    assert njit is not None
    batch, n = v.shape
    standing = np.ascontiguousarray(drive + bias)
    masked = bool((keep != 1.0).any())
    # A broadcast view preserves shared masks without allocating a batch-sized copy.
    keep = np.broadcast_to(np.asarray(keep, float), (batch, n))
    if not v.any():  # from rest every neuron publishes the same thing
        row = np.empty(n)
        m = neuron_model
        _activation(np.zeros(n), m.slope, m.threshold, m.rest_emission, m.leak, row)
        s = np.empty((batch, n))
        s[:] = row
    else:
        s = np.empty_like(v)
        for b in range(batch):
            m = neuron_model
            _activation(v[b], m.slope, m.threshold, m.rest_emission, m.leak, s[b])
    if masked:
        s *= keep
    adapt = neuron_model.adaptation
    has_adapt = adapt is not None
    target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain = _nudge_arrays(
        nudge, batch, n
    )
    lay = layout
    has_synaptic_input = np.zeros(lay.ranges, dtype=np.bool_)
    has_synaptic_input[lay.pair_post] = True
    freezable = ~has_synaptic_input  # nothing arrives: once still, still for good
    if has_adapt:
        freezable = np.zeros(lay.ranges, dtype=np.bool_)
    elif nudge is not None:
        nudged = (
            np.searchsorted(
                lay.starts, np.flatnonzero((nmask > 0) | (anchor_gain > 0)), side="right"
            )
            - 1
        )
        freezable[nudged] = False
    taken, activity_change = _kernel(
        v,
        a,
        s,
        standing,
        flat,
        lay.starts,
        lay.pair_pre,
        lay.pair_post,
        lay.offset,
        has_synaptic_input,
        freezable,
        np.asarray(keep, float),
        masked,
        neuron_model.dt,
        neuron_model.slope,
        neuron_model.threshold,
        neuron_model.rest_emission,
        neuron_model.leak,
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
        anchor,
        anchor_gain,
        int(steps),
        float(tolerance) if tolerance is not None else 0.0,
        tolerance is not None,
    )
    return s, taken, activity_change


if njit is not None:

    @_compiled
    def _residual_kernel(
        v: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        flat: np.ndarray,
        starts: np.ndarray,
        pair_pre: np.ndarray,
        pair_post: np.ndarray,
        offset: np.ndarray,
        keep: np.ndarray,
        masked: bool,
        dt: float,
        has_adapt: bool,
        adapt_strength: float,
        has_nudge: bool,
        target: np.ndarray,
        nmask: np.ndarray,
        gid: np.ndarray,
        ngroups: int,
        beta: float,
        softmax_t: np.ndarray,
        weight: np.ndarray,
        anchor: np.ndarray,
        anchor_gain: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """The fixed-point equation error of every row at the published activations ``s``:
        one block transport, then the same terms the settle kernel adds, without a step."""
        batch, n = v.shape
        pairs = pair_pre.shape[0]
        synaptic_input = np.zeros((batch, n))
        for k in range(pairs):
            a0, a1 = starts[pair_pre[k]], starts[pair_pre[k] + 1]
            b0, b1 = starts[pair_post[k]], starts[pair_post[k] + 1]
            nb = b1 - b0
            block = flat[offset[k] : offset[k + 1]].reshape(a1 - a0, nb)
            product = np.dot(np.ascontiguousarray(s[:, a0:a1]), block)
            for b in range(batch):
                for j in range(nb):
                    synaptic_input[b, b0 + j] += product[b, j]
        group = np.flatnonzero(nmask > 0.0)
        position = np.full(n, -1, dtype=np.int64)
        for k in range(group.shape[0]):
            position[group[k]] = k
        p = np.empty(group.shape[0])
        zmax = np.empty(max(ngroups, 1))
        total = np.empty(max(ngroups, 1))
        for b in range(batch):
            if has_nudge and softmax_t[0] > 0.0:
                _softmax_groups(s[b], group, gid, ngroups, softmax_t, p, zmax, total)
            worst = 0.0
            finite = True
            for i in range(n):
                tot = synaptic_input[b, i] + standing[b, i] - v[b, i]
                if has_adapt:
                    tot -= adapt_strength * a[b, i]
                if anchor_gain[i] > 0.0:
                    tot += anchor_gain[i] * (anchor[b, i] - s[b, i])
                if has_nudge and nmask[i] > 0.0:
                    if softmax_t[0] > 0.0:
                        tot += beta * weight[b] * (target[b, i] - p[position[i]]) * nmask[i]
                    else:
                        tot += beta * weight[b] * nmask[i] * (target[b, i] - s[b, i])
                k_ = keep[b, i] if masked else 1.0
                err = abs(k_ * tot + (k_ - 1.0) * v[b, i] / dt)
                if has_adapt:
                    gap = abs(s[b, i] - a[b, i])
                    if gap > err:
                        err = gap
                if not (err <= 1e300):
                    finite = False
                elif err > worst:
                    worst = err
            out[b] = worst if finite else np.inf


def fused_residual(
    v: np.ndarray,
    a: np.ndarray,
    drive: np.ndarray,
    bias: np.ndarray,
    layout: Layout,
    flat: np.ndarray,
    keep: np.ndarray,
    neuron_model: NeuronModel,
    nudge: Nudge | None,
) -> np.ndarray:
    """``Brain.residual`` for a host state of a blocked connectome, in one compiled pass.

    The same equations as the NumPy path: the activation of ``v`` under the mask, one block
    transport, the potential equation with adaptation and nudge, the mask's projection, and
    the adaptation equation; the maximum absolute error per row, infinity where it is not
    finite. Nothing is changed.
    """
    assert njit is not None
    batch, n = v.shape
    m = neuron_model
    s = np.empty((batch, n))
    for b in range(batch):
        _activation(np.ascontiguousarray(v[b]), m.slope, m.threshold, m.rest_emission, m.leak, s[b])
    keep = np.broadcast_to(np.asarray(keep, float), (batch, n))
    masked = bool((keep != 1.0).any())
    if masked:
        s *= keep
    adapt = m.adaptation
    target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain = _nudge_arrays(
        nudge, batch, n
    )
    out = np.empty(batch)
    _residual_kernel(
        np.ascontiguousarray(v, dtype=float),
        np.ascontiguousarray(a, dtype=float),
        s,
        np.ascontiguousarray(drive + bias),
        flat,
        layout.starts,
        layout.pair_pre,
        layout.pair_post,
        layout.offset,
        np.ascontiguousarray(keep, dtype=float),
        masked,
        m.dt,
        adapt is not None,
        0.0 if adapt is None else adapt.strength,
        nudge is not None,
        target,
        nmask,
        gid,
        ngroups,
        beta,
        softmax_t,
        weight,
        anchor,
        anchor_gain,
        out,
    )
    return out


if njit is not None:

    @_compiled
    def _trace_step(
        trace: np.ndarray,
        trace_bias: np.ndarray,
        decay: float,
        s_plus: np.ndarray,
        s_minus: np.ndarray,
        pre: np.ndarray,
        post: np.ndarray,
        span: float,
        delta: np.ndarray,
        step_scale: np.ndarray,
        step_bias: np.ndarray,
    ) -> None:
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
                a_plus, a_minus = s_plus[b, pre[e]], s_minus[b, pre[e]]
                b_plus, b_minus = s_plus[b, post[e]], s_minus[b, post[e]]
                c = (a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus) / span
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

    @_compiled
    def _contrast_mean(
        s_plus: np.ndarray,
        s_minus: np.ndarray,
        pre: np.ndarray,
        post: np.ndarray,
        span: float,
        out_edges: np.ndarray,
        out_neurons: np.ndarray,
    ) -> None:
        """Batch-mean contrast per synapse and per neuron, one pass, no (batch, edges) temporary."""
        batch, n = s_plus.shape
        edges = pre.shape[0]
        for e in range(edges):
            out_edges[e] = 0.0
        for i in range(n):
            out_neurons[i] = 0.0
        for b in range(batch):
            for e in range(edges):
                a_plus, a_minus = s_plus[b, pre[e]], s_minus[b, pre[e]]
                b_plus, b_minus = s_plus[b, post[e]], s_minus[b, post[e]]
                out_edges[e] += a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus
            for i in range(n):
                out_neurons[i] += s_plus[b, i] - s_minus[b, i]
        scale = 1.0 / (batch * span)
        for e in range(edges):
            out_edges[e] *= scale
        for i in range(n):
            out_neurons[i] *= scale


def contrast_mean(
    s_plus: np.ndarray, s_minus: np.ndarray, pre: np.ndarray, post: np.ndarray, span: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fused ``Learner.contrast``: ``(per_synapse, per_neuron)`` batch means divided by ``span``."""
    assert njit is not None
    out_edges = np.empty(pre.shape[0])
    out_neurons = np.empty(s_plus.shape[1])
    _contrast_mean(
        np.ascontiguousarray(s_plus),
        np.ascontiguousarray(s_minus),
        pre,
        post,
        float(span),
        out_edges,
        out_neurons,
    )
    return out_edges, out_neurons
