"""Private automatic asynchronous activity repair with a complete certificate.

The common energy, gradient, boxes and Armijo rule are unchanged. This module
owns no learning, body, reward, action decoder or task-success criterion. Block
work is recruited from current projected gradients without application flags.
No unresolved coupled coordinate is omitted from publication qualification.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

from . import _repair as R
from ._incremental import ActivityCache
from ._validation import canonical, integer, number


def _dependencies(graph, blocks):
    """Read sets of complete block gradients, including transitive error paths."""
    terms = [set() for _ in range(graph.n_patches)]
    visits = 0
    for target in graph.residual_order:
        reads = {("x", target), ("b", target)}
        for edge in graph.incoming[target]:
            visits += 1
            kind, source, _ = graph.edges[edge]
            reads.add(("w", edge))
            if kind == "residual":
                visits += len(terms[source])
                reads.update(terms[source])
            else:
                reads.add(("u" if kind == "input" else "x", source))
        terms[target] = reads
    result = []
    for block in blocks:
        variables = {("x", i) for i in block}
        reads = variables | {("clamp", i) for i in block} | {("model", 0)}
        for term in terms:
            visits += len(term)
            if term & variables:
                reads.update(term)
        result.append(tuple(sorted(reads)))
    return tuple(result), visits


@dataclass(frozen=True)
class _Context:
    token: int
    cycle: int
    block: int
    stamps: tuple
    clamps: tuple
    cache: ActivityCache


@dataclass(frozen=True)
class _Proposal:
    context: _Context
    status: str
    values: tuple = ()
    slope: float = 0.0


class ActivitySession:
    """One canonical activity state; independently progressing block proposals.

    ``begin`` installs a validated context, ``advance`` automatically collects
    and recruits work, and ``result`` exposes state only when the entire graph
    currently qualifies. There is at most one worker per block. Parameters are
    immutable during each cycle; replacing them starts a new model generation,
    not a learning admission. The original public Brain is never mutated.

    Results are revision-bound snapshots, not actuator leases. A caller must
    check ``is_current`` at its own serialized publication boundary. Pending
    work need not block a qualified result if its block is already valid in the
    current context, but all pending work remains charged and marked unfinished.
    Threads provide independent progress, not promised CPU throughput/deadlines.
    """

    _FROZEN = frozenset(
        (
            "graph",
            "blocks",
            "reads",
            "state_prior",
            "state_bound",
            "parameter_bound",
            "tolerance",
            "step",
            "backtracks",
            "workers",
        )
    )

    def __setattr__(self, name, value):
        if name in self._FROZEN and hasattr(self, name):
            raise AttributeError("configuration is immutable")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._FROZEN:
            raise AttributeError("configuration is immutable")
        super().__delattr__(name)

    def __init__(
        self,
        graph,
        blocks,
        inputs,
        state,
        weights,
        biases,
        *,
        clamps=None,
        state_prior=0.01,
        state_bound=1.0,
        parameter_bound=4.0,
        tolerance=1e-6,
        step=1.0,
        backtracks=32,
        workers=None,
    ):
        if not isinstance(graph, R.Graph):
            raise TypeError("graph must be a repair Graph")
        self.graph = graph
        self.blocks = tuple(tuple(integer(i, "state index") for i in b) for b in blocks)
        flat = [i for block in self.blocks for i in block]
        if (
            not self.blocks
            or any(not b for b in self.blocks)
            or sorted(flat) != list(range(graph.n_patches))
        ):
            raise ValueError("blocks must partition every state exactly once")
        self.reads, dependency_visits = _dependencies(graph, self.blocks)
        for name, value in (
            ("state_prior", state_prior),
            ("state_bound", state_bound),
            ("parameter_bound", parameter_bound),
            ("tolerance", tolerance),
            ("step", step),
        ):
            setattr(self, name, number(value, name, positive=True))
        self.backtracks = integer(backtracks, "backtracks", 1)
        workers = (
            min(32, len(self.blocks))
            if workers is None
            else integer(workers, "workers", 1)
        )
        self.workers = workers
        self._lock = threading.RLock()
        self._versions, self._pending, self._blocked = {}, {}, {}
        self._work = {
            "owner_dependency_visits": dependency_visits,
            "owner_coordinate_checks": 0,
            "owner_context_bytes": 0,
        }
        self._counts = dict.fromkeys(
            (
                "submissions",
                "submit_failures",
                "returns",
                "commits",
                "stale",
                "refused",
                "errors",
                "proposals",
                "backtracks",
                "certificates",
            ),
            0,
        )
        self._token = self._revision = self._cycle = self._started = 0
        self._budget = 0
        self._closed = self._unknown = self._broken = False
        self._issued_result = None
        self._born = time.monotonic()
        self._clamps = self._fixed({} if clamps is None else clamps)
        initial = R._vector(state, graph.n_patches, "state")
        if any(abs(v) > self.state_bound for v in initial):
            raise ValueError("state exceeds bound")
        initial = tuple(self._clamps.get(i, x) for i, x in enumerate(initial))
        self._cache = self._new_cache(inputs, initial, weights, biases)
        self._model_digest = self._model_identity(self._cache)
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="cadence-repair"
        )

    def fork(self):
        """Independent qualified continuation, without re-evaluating the graph.

        Only a drained, qualified owner can fork. The parent pays cache and
        owner copy work; the child starts new counters and its own worker pool.
        This is private transactional activity, not a public Brain checkpoint.
        """
        with self._lock:
            if self._closed or self._broken or self._pending:
                raise ValueError("fork requires an open drained owner")
            if not self.result()["qualified"]:
                raise ValueError("fork requires qualified activity")
            child = object.__new__(type(self))
            for name in self._FROZEN:
                setattr(child, name, getattr(self, name))
            child._lock = threading.RLock()
            child._cache = self._call(self._cache, "fork")
            child._versions = dict(self._versions)
            self._work["owner_copied_slots"] = (
                self._work.get("owner_copied_slots", 0)
                + len(self._FROZEN)
                + 2 * len(self._versions)
            )
            child._pending, child._blocked = {}, {}
            child._work = dict.fromkeys(self._work, 0)
            child._counts = dict.fromkeys(self._counts, 0)
            child._model_digest = self._model_digest
            child._token = child._revision = child._cycle = child._started = 0
            child._budget = 0
            child._closed = child._unknown = child._broken = False
            child._issued_result = None
            child._born = time.monotonic()
            child._clamps = dict(self._clamps)
            self._work["owner_copied_slots"] += 2 * len(self._clamps)
            child._pool = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="cadence-repair"
            )
            return child

    @classmethod
    def from_brain(
        cls, brain, inputs, *, targets=None, interventions=None, workers=None
    ):
        """Copy a native graph/continuation; no jobs start until ``begin``."""
        flat, fixed = brain._arguments(inputs, targets, interventions)
        return cls(
            brain.graph,
            tuple(brain._population_ranges.values()),
            flat,
            brain.state,
            brain.weights,
            brain.biases,
            clamps=fixed,
            workers=workers,
            **{
                k: brain.config[k]
                for k in (
                    "state_prior",
                    "state_bound",
                    "parameter_bound",
                    "tolerance",
                    "step",
                    "backtracks",
                )
            },
        )

    def _fixed(self, clamps):
        if not isinstance(clamps, Mapping):
            raise TypeError("clamps must be a mapping")
        result = {}
        for i, x in clamps.items():
            self._work["owner_coordinate_checks"] += 1
            i, x = integer(i, "clamp"), number(x, "clamp")
            if i >= self.graph.n_patches or abs(x) > self.state_bound:
                raise ValueError("clamp exceeds bounds")
            result[i] = x
        return result

    def _add_work(self, before, after):
        with self._lock:
            for name, total in after.items():
                delta = total - before.get(name, 0)
                if delta < 0:
                    raise RuntimeError("cache work went backwards")
                self._work[name] = self._work.get(name, 0) + delta

    def _call(self, cache, method, **kwargs):
        before = cache.work
        try:
            return getattr(cache, method)(**kwargs)
        finally:
            self._add_work(before, cache.work)

    def _new_cache(self, inputs, state, weights, biases):
        try:
            cache = ActivityCache(
                self.graph,
                inputs,
                state,
                weights,
                biases,
                state_prior=self.state_prior,
                state_bound=self.state_bound,
                parameter_bound=self.parameter_bound,
            )
        except Exception:  # noqa: BLE001 - failed setup may contain unreturned work.
            self._unknown = True
            raise
        self._add_work({}, cache.work)
        return cache

    def _stamp(self, block):
        self._work["owner_dependency_visits"] += len(self.reads[block])
        return tuple((key, self._versions.get(key, 0)) for key in self.reads[block])

    def _bump(self, kind, old, new):
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            self._work["owner_coordinate_checks"] += 1
            if a != b:
                key = (kind, i)
                self._versions[key] = self._versions.get(key, 0) + 1

    def begin(self, inputs, *, clamps=None, weights=None, biases=None, budget=2048):
        """Start one frozen-parameter context; old jobs are version-checked later.

        Clamps default to empty for the new cycle. Validation is atomic. Budget
        counts newly submitted block jobs, not solver sweeps or elapsed time.
        There is no hard cancellation of already executing numerical work.
        """
        budget = integer(budget, "budget")
        with self._lock:
            if self._closed:
                raise ValueError("owner is closed")
            if self._broken:
                raise ValueError("worker pool failed; reconstruct the owner")
            fixed = self._fixed({} if clamps is None else clamps)
            inputs = R._vector(inputs, self.graph.n_inputs, "inputs")
            w = (
                self._cache.weights
                if weights is None
                else R._vector(weights, len(self.graph.edges), "weights")
            )
            b = (
                self._cache.biases
                if biases is None
                else R._vector(biases, self.graph.n_patches, "biases")
            )
            state = tuple(fixed.get(i, x) for i, x in enumerate(self._cache.state))
            replaced = w != self._cache.weights or b != self._cache.biases
            if replaced:
                candidate = self._new_cache(inputs, state, w, b)
            else:
                # Fork before mutation so a rejected context leaves all owner state intact.
                candidate = self._call(self._cache, "fork")
                self._call(candidate, "update", inputs=inputs, state=state)
            self._bump("u", self._cache.inputs, inputs)
            self._bump("x", self._cache.state, state)
            if replaced:
                self._versions[("model", 0)] = self._versions.get(("model", 0), 0) + 1
                self._model_digest = self._model_identity(candidate)
            for i in self._clamps.keys() | fixed.keys():
                if self._clamps.get(i) != fixed.get(i):
                    key = ("clamp", i)
                    self._versions[key] = self._versions.get(key, 0) + 1
            self._cache, self._clamps = candidate, fixed
            self._issued_result = None
            self._cycle += 1
            self._revision += 1
            self._budget, self._started, self._blocked = budget, 0, {}
        return self.advance()

    def _residual(self, state, evaluation, fixed, block=None):
        with self._lock:
            self._work["owner_coordinate_checks"] += (
                self.graph.n_patches if block is None else len(block)
            )
        return max(
            (
                abs(
                    R._projected_component(
                        state[i], evaluation["gradient_state"][i], self.state_bound
                    )
                )
                for i in (range(self.graph.n_patches) if block is None else block)
                if i not in fixed
            ),
            default=0.0,
        )

    def _accept(self, before, after, slope, state, fixed):
        return (
            math.isfinite(slope)
            and slope < 0
            and (
                after["energy"] <= before["energy"] + 1e-4 * slope
                or (
                    abs(after["energy"] - before["energy"])
                    <= 8 * math.ulp(before["energy"])
                    and self._residual(state, after, fixed) <= self.tolerance
                )
            )
        )

    def _launch(self, context, released, accepted):
        # submit() can enqueue/start this wrapper and then raise while creating
        # another thread. Only an owned Future may authorize numerical work.
        released.wait()
        if not accepted.is_set():
            return _Proposal(context, "cancelled_before_compute")
        return self._compute(context)

    def _compute(self, context):
        cache, fixed = context.cache, dict(context.clamps)
        initial = self._call(cache, "evaluation")
        state, block = cache.state, self.blocks[context.block]
        if self._residual(state, initial, fixed, block) <= self.tolerance:
            return _Proposal(context, "stationary")
        scale = self.step
        for _ in range(self.backtracks):
            trial = list(state)
            for i in block:
                if i not in fixed:
                    trial[i] = R._clip(
                        state[i] - scale * initial["gradient_state"][i],
                        self.state_bound,
                    )
            trial = tuple(trial)
            slope = math.fsum(
                initial["gradient_state"][i] * (trial[i] - state[i]) for i in block
            )
            with self._lock:
                self._counts["proposals"] += 1
            try:
                proposed = self._call(cache, "update", state=trial)
                if self._accept(initial, proposed, slope, trial, fixed):
                    return _Proposal(
                        context, "ready", tuple(trial[i] for i in block), slope
                    )
            except ValueError:
                pass
            with self._lock:
                self._counts["backtracks"] += 1
            scale *= 0.5
        return _Proposal(context, "refused")

    def _apply(self, context, proposal):
        if proposal.context is not context:
            raise ValueError("foreign worker proposal")
        if context.stamps != self._stamp(context.block):
            self._counts["stale"] += 1
            return "stale"
        if proposal.status != "ready":
            if proposal.status != "stationary":
                self._blocked[context.block] = self._stamp(context.block)
                self._counts["refused"] += 1
            return proposal.status
        block = self.blocks[context.block]
        if len(proposal.values) != len(block) or any(
            not math.isfinite(x) or abs(x) > self.state_bound for x in proposal.values
        ):
            raise ValueError("invalid worker coordinates")
        trial = list(self._cache.state)
        for i, value in zip(block, proposal.values, strict=True):
            if i in self._clamps and value != self._clamps[i]:
                raise ValueError("worker changed factual clamp")
            trial[i] = value
        trial = tuple(trial)
        before = self._call(self._cache, "evaluation")
        slope = math.fsum(
            before["gradient_state"][i] * (trial[i] - self._cache.state[i])
            for i in block
        )
        if slope != proposal.slope:
            raise ValueError(
                "worker slope does not match current qualified dependencies"
            )
        candidate = self._call(self._cache, "fork")
        after = self._call(candidate, "update", state=trial)
        if not self._accept(before, after, proposal.slope, trial, self._clamps):
            self._blocked[context.block] = self._stamp(context.block)
            self._counts["refused"] += 1
            return "refused"
        self._bump("x", self._cache.state, trial)
        self._cache = candidate
        self._revision += 1
        self._counts["commits"] += 1
        return "committed"

    def advance(self):
        """Collect available workers and recruit all current nonstationary blocks."""
        with self._lock:
            if self._closed:
                raise ValueError("owner is closed")
            returned, started = [], []
            for block, (context, future) in tuple(self._pending.items()):
                if not future.done():
                    continue
                del self._pending[block]
                self._counts["returns"] += 1
                try:
                    proposal = future.result()
                    status = (
                        "discarded_after_submit_failure"
                        if self._broken
                        else self._apply(context, proposal)
                    )
                    row = {"status": status}
                except Exception as error:  # noqa: BLE001 - retain worker failures.
                    self._unknown = True
                    self._counts["errors"] += 1
                    self._blocked[block] = self._stamp(block)
                    row = {
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                returned.append(
                    {
                        "token": context.token,
                        "cycle": context.cycle,
                        "block": block,
                        **row,
                    }
                )
            if self._broken:
                return {
                    "cycle": self._cycle,
                    "revision": self._revision,
                    "returned": returned,
                    "started": [],
                    "pending": len(self._pending),
                }
            evaluated = self._call(self._cache, "evaluation")
            residuals = [
                (
                    self._residual(self._cache.state, evaluated, self._clamps, indices),
                    block,
                )
                for block, indices in enumerate(self.blocks)
            ]
            for residual, block in sorted(
                residuals, key=lambda item: (-item[0], item[1])
            ):
                if self._started >= self._budget:
                    break
                if block in self._pending or self._blocked.get(block) == self._stamp(
                    block
                ):
                    continue
                if residual <= self.tolerance:
                    continue
                # Current jobs with overlapping write/gradient-read sets cannot
                # both commit unchanged. Avoid starvation by not repeatedly
                # invalidating a slower coupled proposal; obsolete jobs do not
                # lock the new context. This is internal dependency scheduling.
                conflict = False
                writes = {("x", i) for i in self.blocks[block] if i not in self._clamps}
                for other, (pending, _) in self._pending.items():
                    if pending.stamps != self._stamp(other):
                        continue
                    other_writes = {
                        ("x", i) for i in self.blocks[other] if i not in self._clamps
                    }
                    self._work["owner_dependency_visits"] += len(writes) + len(
                        other_writes
                    )
                    if writes.intersection(
                        self.reads[other]
                    ) or other_writes.intersection(self.reads[block]):
                        conflict = True
                        break
                if conflict:
                    continue
                self._token += 1
                context = _Context(
                    self._token,
                    self._cycle,
                    block,
                    self._stamp(block),
                    tuple(sorted(self._clamps.items())),
                    self._call(self._cache, "fork"),
                )
                released, accepted = threading.Event(), threading.Event()
                self._started += 1
                self._counts["submissions"] += 1
                try:
                    future = self._pool.submit(
                        self._launch, context, released, accepted
                    )
                except Exception as error:  # noqa: BLE001 - failed submit may enqueue.
                    self._broken = self._unknown = True
                    self._issued_result = None
                    released.set()  # denied: any orphan wrapper performs no solve
                    self._pool.shutdown(wait=False, cancel_futures=True)
                    self._counts["errors"] += 1
                    self._counts["submit_failures"] += 1
                    returned.append(
                        {
                            "token": context.token,
                            "cycle": context.cycle,
                            "block": block,
                            "status": "submit_error",
                            "error": str(error),
                        }
                    )
                    break
                self._pending[block] = context, future
                accepted.set()
                released.set()
                started.append(
                    {"token": context.token, "cycle": context.cycle, "block": block}
                )
            return {
                "cycle": self._cycle,
                "revision": self._revision,
                "returned": returned,
                "started": started,
                "pending": len(self._pending),
            }

    def result(self):
        """A complete current certificate; no publishable state on refusal."""
        with self._lock:
            certificate = self._call(
                self._cache,
                "certificate",
                clamps=self._clamps,
                tolerance=self.tolerance,
            )
            self._counts["certificates"] += 1
            evaluated = self._call(self._cache, "evaluation")
            qualified = (
                not self._closed and not self._broken and certificate["qualified"]
            )
            result = {
                "qualified": qualified,
                "state": self._cache.state if qualified else None,
                "stationarity": certificate["stationarity"],
                "energy": evaluated["energy"],
                "predictions": evaluated["predictions"] if qualified else None,
                "errors": evaluated["errors"] if qualified else None,
                "cycle": self._cycle,
                "revision": self._revision,
                "context_sha256": self._digest(),
                "reason": "closed"
                if self._closed
                else "submit_error"
                if self._broken
                else "qualified"
                if qualified
                else "working"
                if self._pending
                else "budget"
                if self._started >= self._budget
                else "refused",
                "work": dict(self._work),
                "counts": dict(self._counts),
                "pending": len(self._pending),
                "work_complete": not self._pending,
                "unknown_work": self._unknown,
                "clamps": tuple(sorted(self._clamps.items())),
                "elapsed_seconds": time.monotonic() - self._born,
                "work_scope": "tracked cache arithmetic and owner dependency/coordinate/hash work; Python/OS overhead appears only in wall time",
            }
            self._issued_result = canonical(result) if qualified else None
            return result

    def _digest(self):
        payload = canonical(
            [
                self._model_digest,
                self._cache.inputs,
                self._cache.state,
                sorted(self._clamps.items()),
                self._cycle,
                self._revision,
            ]
        ).encode()
        self._work["owner_context_bytes"] += len(payload)
        return hashlib.sha256(payload).hexdigest()

    def _model_identity(self, cache):
        payload = canonical(
            [
                self.graph.n_inputs,
                self.graph.n_patches,
                self.graph.edges,
                self.blocks,
                cache.weights,
                cache.biases,
                self.state_prior,
                self.state_bound,
                self.parameter_bound,
                self.tolerance,
                self.step,
                self.backtracks,
            ]
        ).encode()
        self._work["owner_context_bytes"] += len(payload)
        return hashlib.sha256(payload).hexdigest()

    def is_current(self, result):
        with self._lock:
            return (
                not self._closed
                and not self._broken
                and self._issued_result is not None
                and canonical(result) == self._issued_result
                and result.get("context_sha256") == self._digest()
            )

    def run_until_complete(self, *, seconds=60.0):
        """Continue the begun cycle until qualified, terminal refusal or soft cap.

        Pending obsolete workers do not delay a currently complete certificate.
        A deadline does not kill a worker or publish an unqualified state.
        Returned work may therefore remain a lower bound until ``close(wait=True)``.
        """
        seconds = number(seconds, "seconds", positive=True)
        deadline = time.monotonic() + seconds
        with self._lock:
            cycle = self._cycle
        while True:
            self.advance()
            result = self.result()
            if result["cycle"] != cycle:
                return {
                    **result,
                    "qualified": False,
                    "state": None,
                    "predictions": None,
                    "errors": None,
                    "reason": "context_changed",
                }
            if time.monotonic() >= deadline:
                with self._lock:
                    self._issued_result = None
                return {
                    **result,
                    "qualified": False,
                    "state": None,
                    "predictions": None,
                    "errors": None,
                    "reason": "deadline",
                }
            if result["qualified"] or not result["pending"]:
                return result
            with self._lock:
                futures = [future for _, future in self._pending.values()]
            wait(
                futures,
                timeout=max(0.0, deadline - time.monotonic()),
                return_when=FIRST_COMPLETED,
            )

    def close(self, *, wait=True):
        """Stop new publication; drain workers if requested, never hide their cost."""
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=False)
        with self._lock:
            for block, (_context, future) in tuple(self._pending.items()):
                if future.done():
                    del self._pending[block]
                    self._counts["returns"] += 1
                    try:
                        future.result()
                    except Exception:  # noqa: BLE001 - close retains unknown work.
                        self._unknown = True
                        self._counts["errors"] += 1
            return self.result()
