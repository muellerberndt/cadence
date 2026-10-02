"""Private exact activity cache for one frozen graph and parameter vector.

This is arithmetic reuse, not a learning rule, attention policy or public solver.
Every current state-gradient component remains present in the certificate. A
caller must serialize access to one cache; independent worker proposals use forks.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from dataclasses import dataclass

from . import _repair as R
from ._validation import integer, number


@dataclass(frozen=True)
class _Stamp:
    owner: object
    reads: tuple


class ActivityCache:
    """Exact forward/adjoint reuse, with atomic validation and failed-update rollback.

    Parameters/configuration are immutable. Replace the cache after a model
    update or restore. ``update`` receives complete input/state vectors; detecting
    their changes costs O(inputs + states). Energy reduction and certification
    cost O(states), even when no relation needs recomputation. Work counters count
    these scans and result/fork copies separately from relation edge visits.
    ``copied_values`` tracks cache/result arrays, not every reference-kernel
    allocation or Python container operation; wall time remains necessary.
    """

    def __init__(
        self,
        graph,
        inputs,
        state,
        weights,
        biases,
        *,
        state_prior=0.01,
        state_bound=1.0,
        parameter_bound=4.0,
    ):
        inputs, state, weights, biases, _, _ = R._arguments(
            graph,
            inputs,
            state,
            weights,
            biases,
            None,
            None,
        )
        self._alpha = number(state_prior, "state_prior", positive=True)
        self._state_bound = number(state_bound, "state_bound", positive=True)
        self._parameter_bound = number(
            parameter_bound, "parameter_bound", positive=True
        )
        if any(abs(v) > self._state_bound for v in state):
            raise ValueError("Initial state exceeds state_bound")
        if any(abs(v) > self._parameter_bound for v in (*weights, *biases)):
            raise ValueError("Initial parameters exceed parameter_bound")
        self._graph, self._inputs, self._state = graph, inputs, state
        self._weights, self._biases = weights, biases
        self._identity = object()
        n = graph.n_patches
        self._u_versions, self._x_versions = [0] * graph.n_inputs, [0] * n
        self._work = dict.fromkeys(
            (
                "setups",
                "evaluations",
                "patch_visits",
                "edge_visits",
                "dependency_visits",
                "coefficient_checks",
                "input_checks",
                "state_checks",
                "finite_checks",
                "energy_terms",
                "certificate_checks",
                "copied_values",
                "forks",
                "scope_checks",
            ),
            0,
        )
        self._work.update(
            setups=1,
            coefficient_checks=len(weights) + n,
            state_checks=n,
            finite_checks=len(inputs) + len(weights) + 2 * n + 3,
        )
        self._input_users = [[] for _ in inputs]
        self._state_users, self._error_users = ([[] for _ in state] for _ in range(2))
        self._reverse_error, self._gradient_terms = (
            [[] for _ in state] for _ in range(2)
        )
        for kind, source, target in graph.edges:
            self._work["dependency_visits"] += 1
            users = (
                self._input_users
                if kind == "input"
                else self._state_users
                if kind == "state"
                else self._error_users
            )
            users[source].append(target)
        for target in reversed(graph.residual_order):
            self._gradient_terms[target].append((None, target))
            for edge in graph.incoming[target]:
                self._work["dependency_visits"] += 1
                kind, source, _ = graph.edges[edge]
                if kind == "state":
                    self._gradient_terms[source].append((edge, target))
                elif kind == "residual":
                    self._reverse_error[source].append((edge, target))
        self._rank = {target: i for i, target in enumerate(graph.residual_order)}
        visits = {"edge_visits": 0, "patch_visits": 0}
        initial = R._evaluate(
            graph,
            inputs,
            state,
            weights,
            biases,
            self._alpha,
            None,
            None,
            0.1,
            visits,
            parameter_gradients=False,
        )
        self._work["evaluations"] += 1
        for key, value in visits.items():
            self._work[key] += value
        self._p, self._e, self._g = (
            list(initial[k]) for k in ("predictions", "errors", "gradient_state")
        )
        self._energy = initial["energy"]
        self._adj, self._ap = list(self._e), [0.0] * n
        for target in reversed(graph.residual_order):
            self._work["patch_visits"] += 1
            for edge, consumer in self._reverse_error[target]:
                self._work["edge_visits"] += 1
                self._adj[target] += self._ap[consumer] * weights[edge]
            # Incoming residual contributions must precede this node's derivative.
            self._ap[target] = -self._adj[target] * (1.0 - self._p[target] ** 2)
        self._work["energy_terms"] += 2 * n
        self._work["finite_checks"] += 1 + 3 * n
        self._work["copied_values"] += len(inputs) + len(weights) + 9 * n

    @property
    def graph(self):
        return self._graph

    @property
    def inputs(self):
        return self._inputs

    @property
    def state(self):
        return self._state

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def work(self):
        return dict(self._work)

    def evaluation(self):
        """Return current arithmetic; omit the unused E-sized parameter signal array."""
        self._work["copied_values"] += 3 * len(self._state)
        return dict(
            energy=self._energy,
            predictions=tuple(self._p),
            errors=tuple(self._e),
            gradient_state=tuple(self._g),
            gradient_weights=(),
            gradient_biases=(),
        )

    def fork(self):
        """Independent proposal copy; parent pays copy cost, child counters start at zero."""
        child = object.__new__(type(self))
        child.__dict__ = dict(self.__dict__)
        for key in ("_p", "_e", "_g", "_adj", "_ap", "_u_versions", "_x_versions"):
            child.__dict__[key] = list(self.__dict__[key])
            self._work["copied_values"] += len(child.__dict__[key])
        self._work["forks"] += 1
        child._work = dict.fromkeys(self._work, 0)
        child._identity = object()
        return child

    def update(self, *, inputs=None, state=None):
        """Recompute only invalidated relations/adjoints; exceptions change no values."""
        try:
            return self._update(inputs=inputs, state=state)
        except ArithmeticError as error:
            raise ValueError(
                "Energy or gradient exceeds the finite numeric range"
            ) from error

    def _update(self, *, inputs, state):
        self._work["evaluations"] += 1
        u = (
            self._inputs
            if inputs is None
            else self._vector(inputs, self.graph.n_inputs, "inputs")
        )
        x = (
            self._state
            if state is None
            else self._vector(state, self.graph.n_patches, "state")
        )
        for value in x:
            self._work["state_checks"] += 1
            if abs(value) > self._state_bound:
                raise ValueError("State exceeds state_bound")
        self._work["input_checks"] += len(u)
        self._work["state_checks"] += len(x)
        changed_u = {
            i for i, (a, b) in enumerate(zip(u, self._inputs, strict=True)) if a != b
        }
        changed_x = {
            i for i, (a, b) in enumerate(zip(x, self._state, strict=True)) if a != b
        }
        p, e, adj, ap, grad = {}, {}, {}, {}, {}
        dirty_p = set()
        for changed, users in (
            (changed_u, self._input_users),
            (changed_x, self._state_users),
        ):
            for source in changed:
                self._work["dependency_visits"] += len(users[source])
                dirty_p.update(users[source])
        queue = [(self._rank[i], i) for i in dirty_p | changed_x]
        heapq.heapify(queue)
        queued = dirty_p | changed_x
        p_changed, e_changed = set(), set()
        while queue:
            _, target = heapq.heappop(queue)
            self._work["patch_visits"] += 1
            if target in dirty_p:
                terms = [self._biases[target]]
                for edge in self.graph.incoming[target]:
                    self._work["edge_visits"] += 1
                    kind, source, _ = self.graph.edges[edge]
                    signal = (
                        u[source]
                        if kind == "input"
                        else x[source]
                        if kind == "state"
                        else e.get(source, self._e[source])
                    )
                    terms.append(self._weights[edge] * signal)
                activation = math.fsum(terms)
                self._finite(activation)
                p[target] = math.tanh(activation)
                if p[target] != self._p[target]:
                    p_changed.add(target)
            e[target] = x[target] - p.get(target, self._p[target])
            self._finite(e[target])
            if e[target] != self._e[target]:
                e_changed.add(target)
                for consumer in self._error_users[target]:
                    self._work["dependency_visits"] += 1
                    dirty_p.add(consumer)
                    if consumer not in queued:
                        heapq.heappush(queue, (self._rank[consumer], consumer))
                        queued.add(consumer)
        dirty_adj = set(e_changed)
        queue = [(-self._rank[i], i) for i in dirty_adj | p_changed]
        heapq.heapify(queue)
        queued = dirty_adj | p_changed
        dirty_g = set(changed_x)
        while queue:
            _, target = heapq.heappop(queue)
            self._work["patch_visits"] += 1
            if target in dirty_adj:
                value = e.get(target, self._e[target])
                for edge, consumer in self._reverse_error[target]:
                    self._work["edge_visits"] += 1
                    value += ap.get(consumer, self._ap[consumer]) * self._weights[edge]
                self._finite(value)
                adj[target] = value
                if value != self._adj[target]:
                    dirty_g.add(target)
            ap[target] = -adj.get(target, self._adj[target]) * (
                1.0 - p.get(target, self._p[target]) ** 2
            )
            self._finite(ap[target])
            if ap[target] != self._ap[target]:
                for edge in self.graph.incoming[target]:
                    self._work["dependency_visits"] += 1
                    kind, source, _ = self.graph.edges[edge]
                    if kind == "state":
                        dirty_g.add(source)
                    elif kind == "residual":
                        dirty_adj.add(source)
                        if source not in queued:
                            heapq.heappush(queue, (-self._rank[source], source))
                            queued.add(source)
        for source in dirty_g:
            self._work["patch_visits"] += 1
            value = self._alpha * x[source]
            for edge, target in self._gradient_terms[source]:
                self._work["dependency_visits"] += 1
                if edge is None:
                    value += adj.get(target, self._adj[target])
                else:
                    self._work["edge_visits"] += 1
                    value += ap.get(target, self._ap[target]) * self._weights[edge]
            self._finite(value)
            grad[source] = value
        errors = (e.get(i, v) for i, v in enumerate(self._e))
        energy = 0.5 * math.fsum(v * v for v in errors)
        energy += 0.5 * self._alpha * math.fsum(v * v for v in x)
        self._work["energy_terms"] += 2 * len(x)
        self._finite(energy)
        # No mutation until every derived quantity has passed finite validation.
        for target, changes in (
            (self._p, p),
            (self._e, e),
            (self._adj, adj),
            (self._ap, ap),
            (self._g, grad),
        ):
            for i, value in changes.items():
                target[i] = value
        for changed, versions in (
            (changed_u, self._u_versions),
            (changed_x, self._x_versions),
        ):
            for i in changed:
                versions[i] += 1
        self._inputs, self._state, self._energy = u, x, energy
        return self.evaluation()

    def _finite(self, value):
        self._work["finite_checks"] += 1
        if not math.isfinite(value):
            raise ValueError("Energy or gradient exceeds the finite numeric range")

    def _vector(self, values, length, name):
        # Same boundary as R._vector, with attempted validation charged on failure.
        if isinstance(values, (str, bytes, Mapping)):
            raise ValueError(f"{name} must contain {length} finite numbers")

        def checked():
            for value in values:
                self._work["finite_checks"] += 1
                result = number(value, name)
                self._work["copied_values"] += 1
                yield result

        try:
            result = tuple(checked())
        except TypeError as error:
            raise ValueError(f"{name} must contain {length} finite numbers") from error
        if len(result) != length:
            raise ValueError(f"{name} must contain {length} finite numbers")
        return result

    def certificate(self, *, clamps=None, tolerance=1e-6):
        """Scan every eligible coordinate of the complete current exact gradient."""
        tolerance = number(tolerance, "tolerance", positive=True)
        clamps = {} if clamps is None else clamps
        if not isinstance(clamps, Mapping):
            raise ValueError("clamps must be a mapping")
        fixed = {}
        for index, value in clamps.items():
            index, value = integer(index, "clamp index"), number(value, "clamp value")
            if (
                index >= len(self._state)
                or abs(value) > self._state_bound
                or self._state[index] != value
            ):
                raise ValueError("Clamp must match the current bounded state")
            fixed[index] = value
        residual = 0.0
        for i, (x, g) in enumerate(zip(self._state, self._g, strict=True)):
            self._work["certificate_checks"] += 1
            if i not in fixed:
                residual = max(
                    residual, abs(R._projected_component(x, g, self._state_bound))
                )
        return {"stationarity": residual, "qualified": residual <= tolerance}

    def stamp(self, indices):
        """Conservative gradient-read versions, including currently zero contacts.

        Clamp eligibility is an owner concern: a stamp binds state/input reads,
        not permission to publish under a later mask. Any different cache/model
        invalidates the stamp. Direct states are leaves, not derived predictions.
        """
        selected = {integer(i, "state index") for i in indices}
        if not selected or any(i >= len(self._state) for i in selected):
            raise ValueError("Require valid state indices")
        factors = set(selected)
        for i in selected:
            self._work["dependency_visits"] += len(self._state_users[i])
            factors.update(self._state_users[i])
        pending = list(factors)
        while pending:
            for target in self._error_users[pending.pop()]:
                self._work["dependency_visits"] += 1
                if target not in factors:
                    factors.add(target)
                    pending.append(target)
        reads, seen, pending = {("x", i) for i in selected}, set(), list(factors)
        while pending:
            target = pending.pop()
            if target in seen:
                continue
            seen.add(target)
            reads.add(("x", target))
            for edge in self.graph.incoming[target]:
                self._work["dependency_visits"] += 1
                kind, source, _ = self.graph.edges[edge]
                if kind == "residual":
                    pending.append(source)
                else:
                    reads.add(("u" if kind == "input" else "x", source))
        result = tuple(
            (kind, i, (self._u_versions if kind == "u" else self._x_versions)[i])
            for kind, i in sorted(reads)
        )
        self._work["scope_checks"] += len(result)
        return _Stamp(self._identity, result)

    def matches(self, stamp):
        if not isinstance(stamp, _Stamp) or stamp.owner is not self._identity:
            return False
        self._work["scope_checks"] += len(stamp.reads)
        return all(
            (self._u_versions if kind == "u" else self._x_versions)[i] == version
            for kind, i, version in stamp.reads
        )
