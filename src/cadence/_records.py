"""Private, fixed-parameter temporal record boundary; NOT a public Cortex mode.

Cross-block contacts become fixed historical inputs. This changes the joint
objective, despite retaining the same local patch rule. Both fast AND slow
states remain free in every complete certificate. Slow proposals are private
future revisions; none of their unresolved coordinates belong to active state.
There is no learner, action policy, snapshot API or recursive-benefit claim.
"""

from __future__ import annotations

import copy
import hashlib
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from . import _repair as R
from ._attention import ActivitySession
from ._validation import canonical, integer, number


def _hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class _Record:
    version: int
    event: int
    inputs: tuple
    state: tuple
    errors: tuple
    fast_source: str
    slow_source: str


@dataclass(frozen=True)
class Ticket:
    """Owned pre-outcome prediction; identity, not merely field equality, matters."""

    event: int
    record: int
    covered_events: tuple
    goal: int
    required: float | None
    predicted_value: float | None
    forecast: tuple
    inputs: tuple
    state: tuple
    errors: tuple
    certificate: str


@dataclass(frozen=True)
class _Request:
    event: int
    base: int
    goal_epoch: int
    record: _Record


class _Tracked(ActivitySession):
    def __init__(self, *args, **kwargs):
        self.jobs = [0, 0]
        super().__init__(*args, **kwargs)

    def _compute(self, context):
        with self._lock:
            self.jobs[context.block] += 1
        return super()._compute(context)

    def fork(self):
        child = super().fork()
        child.jobs = [0, 0]
        return child


class RecordSession:
    """One active graph/state/model plus a bounded evidence ledger and proposal.

    ``outcome_inputs`` are owner-written ACK channels. ``step`` supplies only
    the other input coordinates, in original order. The static ``fast`` set is
    a genome boundary, not a per-call attention flag. Startup qualifies both
    blocks; afterward only surprise or terminal goal deficit requests a new
    slow record. Quiet ACKs do not starve a pending proposal.

    All calls are serialized by this owner. A background thread computes only
    a private candidate. Commit requalifies at the latest fast inputs, then
    atomically replaces the complete active state/record. A refused candidate
    never commits; step may first adopt an independently completed background
    proposal. Refusal does not clear persistent demand. Parameters never
    change in this prototype. Historical errors are from their named LOWERED
    solve, not current residuals of the original graph.
    """

    _FROZEN = frozenset(
        (
            "original",
            "fast",
            "slow",
            "forecast",
            "outcome_inputs",
            "live_inputs",
            "value_index",
            "threshold",
            "goal_tolerance",
            "capacity",
            "budget",
            "seconds",
            "config",
            "weights",
            "biases",
            "graph",
            "ports",
            "graph_sha256",
            "lowered_sha256",
            "model_sha256",
        )
    )

    def __setattr__(self, name, value):
        if name in self._FROZEN and hasattr(self, name):
            raise AttributeError("record model and configuration are immutable")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._FROZEN:
            raise AttributeError("record model and configuration are immutable")
        super().__delattr__(name)

    def __init__(
        self,
        graph,
        fast,
        inputs,
        state,
        weights,
        biases,
        *,
        forecast_indices,
        outcome_inputs=(),
        value_index=None,
        surprise_threshold=0.05,
        goal_tolerance=0.0,
        capacity=256,
        budget=2048,
        seconds=5.0,
        **config,
    ):
        self._lock = threading.RLock()
        self._closed = self._unknown = False
        self.original = graph
        self.fast = tuple(integer(i, "fast coordinate") for i in fast)
        if (
            not self.fast
            or len(set(self.fast)) != len(self.fast)
            or any(i >= graph.n_patches for i in self.fast)
        ):
            raise ValueError("fast must contain distinct graph coordinates")
        self.slow = tuple(i for i in range(graph.n_patches) if i not in self.fast)
        if not self.slow:
            raise ValueError("a temporal boundary needs both blocks")
        self.forecast = self._indices(forecast_indices, graph.n_patches, "forecast")
        self.outcome_inputs = self._indices(outcome_inputs, graph.n_inputs, "outcome")
        if len(self.outcome_inputs) != len(self.forecast) or not self.forecast:
            raise ValueError("one outcome input is required per forecast coordinate")
        self.live_inputs = tuple(
            i for i in range(graph.n_inputs) if i not in self.outcome_inputs
        )
        self.value_index = (
            None if value_index is None else integer(value_index, "value_index")
        )
        if self.value_index is not None and self.value_index >= graph.n_patches:
            raise ValueError("value index outside graph")
        self.threshold = number(surprise_threshold, "surprise_threshold", positive=True)
        self.goal_tolerance = number(goal_tolerance, "goal_tolerance")
        if self.goal_tolerance < 0:
            raise ValueError("goal tolerance must be nonnegative")
        self.capacity = integer(capacity, "capacity", 1)
        self.budget = integer(budget, "budget")
        self.seconds = number(seconds, "seconds", positive=True)
        self.config = MappingProxyType(dict(config))
        self.weights = R._vector(weights, len(graph.edges), "weights")
        self.biases = R._vector(biases, graph.n_patches, "biases")
        self._inputs = R._vector(inputs, graph.n_inputs, "inputs")
        state = R._vector(state, graph.n_patches, "state")
        self.graph, self.ports = self._lower(graph)
        self.graph_sha256 = _hash((graph.n_inputs, graph.n_patches, graph.edges))
        self.lowered_sha256 = _hash(
            (self.graph.n_inputs, self.graph.n_patches, self.graph.edges)
        )
        self.model_sha256 = _hash((self.weights, self.biases, dict(self.config)))
        self._work, self._jobs = (
            {},
            {
                "startup": [0, 0],
                "foreground": [0, 0],
                "proposal": [0, 0],
                "commit": [0, 0],
            },
        )
        self._counts = dict.fromkeys(
            (
                "steps",
                "acks",
                "requests",
                "commits",
                "stale",
                "refused",
                "submit_failures",
                "record_values",
                "worker_errors",
                "proposal_returns",
                "discarded_on_close",
            ),
            0,
        )
        self._born = time.monotonic()
        self._ticket = self._pending = None
        self._goal = self._goal_epoch = 0
        self._covered = set()
        self._required = None
        self._revision = 0
        self._issued_result = None
        self._threads = []
        self._attempted = set()
        self._ledger, self._demand, self._events = [], {}, {}
        self._needs = {}
        self._record = _Record(
            0,
            0,
            self._inputs,
            state,
            (0.0,) * graph.n_patches,
            "genesis estimate",
            "genesis estimate",
        )
        # Genesis errors are explicitly estimates. Obtain a qualified source
        # once, then condition on its exact recorded state/error and requalify.
        initial, result = self._solve(self._record, self._inputs, state, "startup")
        self._retire(initial, "startup")
        if not result["qualified"]:
            raise ValueError("initial graph refused qualification")
        self._record = replace(
            self._record,
            state=result["state"],
            errors=result["errors"],
            fast_source=result["context_sha256"],
            slow_source=result["context_sha256"],
        )
        self._active, result = self._solve(
            self._record, self._inputs, result["state"], "startup"
        )
        if not result["qualified"]:
            self._retire(self._active, "startup")
            raise ValueError("initial record-conditioned graph refused qualification")
        self._active_kind = "startup"

    @staticmethod
    def _indices(values, limit, name):
        result = tuple(integer(i, name) for i in values)
        if len(set(result)) != len(result) or any(i >= limit for i in result):
            raise ValueError(f"{name} coordinates must be distinct and in bounds")
        return result

    def _lower(self, graph):
        ports, edges = [], []
        for kind, source, target in graph.edges:
            held = (
                kind == "input"
                and (target in self.slow or source in self.outcome_inputs)
            ) or (kind != "input" and ((source in self.fast) != (target in self.fast)))
            if held:
                key = (kind, source)
                if key not in ports:
                    ports.append(key)
                edges.append(("input", graph.n_inputs + ports.index(key), target))
            else:
                edges.append((kind, source, target))
        return R.Graph(
            graph.n_inputs + len(ports), graph.n_patches, tuple(edges)
        ), tuple(ports)

    def _expanded(self, record, inputs):
        values = tuple(
            record.inputs[i]
            if kind == "input"
            else record.state[i]
            if kind == "state"
            else record.errors[i]
            for kind, i in self.ports
        )
        with self._lock:
            self._counts["record_values"] += len(values)
        return tuple(inputs) + values

    def _solve(self, record, inputs, state, role, *, base=None):
        session = None
        try:
            if base is None:
                session = _Tracked(
                    self.graph,
                    (self.fast, self.slow),
                    self._expanded(record, inputs),
                    state,
                    self.weights,
                    self.biases,
                    **self.config,
                )
            else:
                session = base.fork()
                if tuple(state) != session._cache.state:
                    session._bump("x", session._cache.state, state)
                    session._call(session._cache, "update", state=state)
            session.begin(self._expanded(record, inputs), budget=self.budget)
            return session, session.run_until_complete(seconds=self.seconds)
        except Exception:
            if session is not None:
                self._retire(session, role)
            else:
                self._unknown = True  # constructor/fork may have unreturned work
            raise

    def _retire(self, session, role):
        result = session.close()
        with self._lock:
            for name, count in result["work"].items():
                self._work[name] = self._work.get(name, 0) + count
            for i, count in enumerate(session.jobs):
                self._jobs[role][i] += count
            self._unknown |= result["unknown_work"]

    def _open(self):
        if self._closed:
            raise ValueError("record owner is closed")

    def step(self, inputs, *, goal_version=0, required_value=None):
        """Qualified forecast from the complete active record-conditioned graph."""
        with self._lock:
            self._open()
            if self._ticket is not None:
                raise ValueError("acknowledge the outstanding forecast first")
            if len(self._ledger) >= self.capacity:
                raise ValueError("bounded evidence ledger is full")
            goal = integer(goal_version, "goal_version")
            required = (
                None
                if required_value is None
                else number(required_value, "required_value")
            )
            live = R._vector(inputs, len(self.live_inputs), "live inputs")
            if (goal, required) != (self._goal, self._required):
                self._goal_epoch += 1
                self._goal, self._required = goal, required
                self._revision += 1
                self._issued_result = None
            self.poll()
            full = list(self._inputs)
            for i, value in zip(self.live_inputs, live, strict=True):
                full[i] = value
            full = tuple(full)
            previous = self._active.result()
            candidate, result = self._solve(
                self._record, full, previous["state"], "foreground", base=self._active
            )
            if not result["qualified"]:
                self._retire(candidate, "foreground")
                self._counts["refused"] += 1
                return {"qualified": False, "reason": result["reason"], "ticket": None}
            self._retire(self._active, self._active_kind)
            self._active, self._active_kind, self._inputs = (
                candidate,
                "foreground",
                full,
            )
            self._counts["steps"] += 1
            self._revision += 1
            self._ticket = Ticket(
                self._counts["steps"],
                self._record.version,
                tuple(sorted(self._covered)),
                goal,
                required,
                None if self.value_index is None else result["state"][self.value_index],
                tuple(result["state"][i] for i in self.forecast),
                full,
                result["state"],
                result["errors"],
                result["context_sha256"],
            )
            self._ticket_record = self._record
            return {
                "qualified": True,
                "forecast": self._ticket.forecast,
                "ticket": self._ticket,
                "certificate": self.result(),
            }

    def acknowledge(self, ticket, actual, *, task_value=None, terminal=False):
        """Bind one real outcome; no caller-supplied prediction or wake flag."""
        with self._lock:
            self._open()
            if ticket is not self._ticket or ticket is None:
                raise ValueError("foreign, stale or duplicate forecast")
            actual = R._vector(actual, len(self.forecast), "actual outcome")
            value = None if task_value is None else number(task_value, "task_value")
            if type(terminal) is not bool:
                raise ValueError("terminal must be boolean")
            if terminal and ticket.required is not None and value is None:
                raise ValueError("terminal need requires actual task value")
            miss = max(abs(a - b) for a, b in zip(actual, ticket.forecast, strict=True))
            deficit = (
                terminal
                and ticket.required is not None
                and value < ticket.required - self.goal_tolerance
            )
            wake = miss > self.threshold or deficit
            full = list(ticket.inputs)
            for i, observed in zip(self.outcome_inputs, actual, strict=True):
                full[i] = observed
            record = _Record(
                self._record.version + 1,
                ticket.event,
                tuple(full),
                ticket.state,
                ticket.errors,
                ticket.certificate,
                ticket.certificate,
            )
            row = {
                "event": ticket.event,
                "goal": ticket.goal,
                "forecast": ticket.forecast,
                "actual": actual,
                "task_value": value,
                "predicted_value": ticket.predicted_value,
                "required": ticket.required,
                "terminal": terminal,
                "surprise": miss,
                "goal_deficit": deficit,
                "wake": wake,
                "record_at_issue": ticket.record,
                "certificate": ticket.certificate,
                "issued": asdict(ticket),
                "record_at_issue_details": asdict(self._ticket_record),
            }
            self._ledger.append(row)
            self._counts["acks"] += 1
            self._revision += 1
            self._issued_result = None
            self._ticket = None
            key = (
                ticket.goal,
                ticket.required,
                tuple(ticket.inputs[i] for i in self.live_inputs),
            )
            if wake:
                old = self._demand.get(key)
                terminal_required = bool(deficit) or (
                    old is not None and self._needs[old]
                )
                self._demand[key] = ticket.event
                self._events[ticket.event] = record
                self._needs[ticket.event] = terminal_required
            elif (
                key in self._demand
                and self._demand[key] in ticket.covered_events
                and ticket.event > self._demand[key]
                and (not self._needs[self._demand[key]] or terminal)
            ):
                # Commit/fit success is insufficient: require a later factual
                # quiet outcome issued using a revision covering that demand.
                del self._demand[key]
            self._start()
            return copy.deepcopy(row)

    def _start(self):
        available = sorted(
            event
            for key, event in self._demand.items()
            if key[:2] == (self._goal, self._required)
            and (self._goal_epoch, event) not in self._attempted
        )
        event = available[0] if available else None
        identity = (self._goal_epoch, event)
        if self._pending is not None or event is None or identity in self._attempted:
            return
        self._attempted.add(identity)
        request = _Request(
            event,
            self._record.version,
            self._goal_epoch,
            replace(self._events[event], version=self._record.version + 1),
        )
        future, release, allowed = Future(), threading.Event(), threading.Event()

        def worker():
            release.wait()
            if not allowed.is_set():
                return
            try:
                future.set_result(self._propose(request))
            except (
                BaseException
            ) as error:  # every terminated worker must resolve its Future
                future.set_exception(
                    error
                    if isinstance(error, Exception)
                    else RuntimeError(
                        f"worker terminated: {type(error).__name__}: {error}"
                    )
                )

        thread = threading.Thread(
            target=worker, name="cadence-record-proposal", daemon=True
        )
        self._threads.append(thread)
        self._counts["requests"] += 1
        try:
            thread.start()
        except Exception:  # noqa: BLE001 - thread startup can fail after work.
            self._unknown = True
            self._counts["submit_failures"] += 1
            release.set()
            return
        self._pending = request, future, thread
        allowed.set()
        release.set()

    def _propose(self, request):
        # This private full-graph solve is a candidate, not active state. Both
        # blocks remain free; every job and validation scan is charged.
        session, result = self._solve(
            request.record, request.record.inputs, request.record.state, "proposal"
        )
        self._retire(session, "proposal")
        return result

    def poll(self):
        """Adopt only a qualified current proposal, through full requalification."""
        with self._lock:
            self._open()
            if self._pending is None:
                self._start()
                return None
            if not self._pending[1].done():
                return None
            request, future, _thread = self._pending
            self._pending = None
            self._counts["proposal_returns"] += 1
            try:
                proposal = future.result()
            except Exception as error:  # noqa: BLE001 - refused work remains demand.
                self._unknown = True
                self._counts["refused"] += 1
                self._counts["worker_errors"] += 1
                self._start()
                return {"status": "error", "error": str(error)}
            if (
                request.base != self._record.version
                or request.goal_epoch != self._goal_epoch
            ):
                self._counts["stale"] += 1
                self._start()
                return {"status": "stale"}
            if not proposal["qualified"]:
                self._counts["refused"] += 1
                self._start()
                return {"status": "refused"}
            state, errors = list(request.record.state), list(request.record.errors)
            current = self._active.result()
            initial = list(current["state"])
            for i in self.slow:
                state[i], errors[i], initial[i] = (
                    proposal["state"][i],
                    proposal["errors"][i],
                    proposal["state"][i],
                )
            record = replace(
                request.record,
                state=tuple(state),
                errors=tuple(errors),
                slow_source=proposal["context_sha256"],
            )
            # Latest fast activity is the continuation. No proposed intermediate
            # state is ever installed; owner lock excludes concurrent publication.
            try:
                session, result = self._solve(
                    record, self._inputs, tuple(initial), "commit", base=self._active
                )
            except Exception as error:  # noqa: BLE001 - invalid combined context is refused atomically.
                self._counts["refused"] += 1
                self._counts["worker_errors"] += 1
                self._start()
                return {
                    "status": "commit_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            if not result["qualified"]:
                self._retire(session, "commit")
                self._counts["refused"] += 1
                self._start()
                return {"status": "refused_commit"}
            self._retire(self._active, self._active_kind)
            self._active, self._active_kind, self._record = session, "commit", record
            self._covered.add(request.event)
            self._counts["commits"] += 1
            self._revision += 1
            self._issued_result = None
            self._start()  # newer unresolved evidence is not discarded/starved
            return {"status": "committed", "record_version": record.version}

    def result(self):
        with self._lock:
            result = self._active.result()
            work, jobs = dict(self._work), {k: list(v) for k, v in self._jobs.items()}
            for name, count in result["work"].items():
                work[name] = work.get(name, 0) + count
            for i, count in enumerate(self._active.jobs):
                jobs[self._active_kind][i] += count
            receipt = {
                "qualified": not self._closed and result["qualified"],
                "state": result["state"] if not self._closed else None,
                "stationarity": result["stationarity"],
                "energy": result["energy"],
                "original_graph_sha256": self.graph_sha256,
                "lowered_graph_sha256": self.lowered_sha256,
                "model_sha256": self.model_sha256,
                "context_sha256": self._context(),
                "record_version": self._record.version,
                "record_event": self._record.event,
                "record_age_events": self._counts["acks"] - self._record.event,
                "record_sources": {
                    "fast": self._record.fast_source,
                    "slow": self._record.slow_source,
                },
                "historical_ports": self.ports,
                "demand": tuple(
                    {
                        "goal": key[0],
                        "required": key[1],
                        "inputs": key[2],
                        "event": event,
                        "terminal_required": self._needs[event],
                    }
                    for key, event in self._demand.items()
                ),
                "pending": self._pending is not None,
                "counts": dict(self._counts),
                "work": work,
                "jobs": jobs,
                "unknown_work": self._unknown or result["unknown_work"],
                "work_complete": self._pending is None and result["work_complete"],
                "seconds": time.monotonic() - self._born,
                "scope": "ALL current free states of the lowered graph, conditional on explicitly historical committed record; pending evidence is not incorporated",
                "recovery_scope": "later actual quiet ACK on the same exact live input and goal/requirement; terminal deficits additionally require a satisfactory terminal value; finite context control, not learned generalization",
                "work_scope": "cache and activity-owner counters plus record_values; Python bookkeeping/thread/hash overhead is included only in wall time; pending numerical work is a lower bound until drained",
            }
            self._issued_result = canonical(receipt) if receipt["qualified"] else None
            return receipt

    def _context(self):
        return _hash(
            (
                self.model_sha256,
                self.lowered_sha256,
                self._revision,
                self._record.version,
                self._goal_epoch,
                self._inputs,
                self._active._cache.state,
            )
        )

    def is_current(self, receipt):
        with self._lock:
            return (
                not self._closed
                and self._issued_result is not None
                and canonical(receipt) == self._issued_result
                and receipt.get("context_sha256") == self._context()
            )

    def ledger(self):
        with self._lock:
            return copy.deepcopy(tuple(self._ledger))

    def close(self):
        with self._lock:
            if self._closed:
                return self.result()
            self._closed = True
            self._issued_result = None
            pending = self._pending
        for thread in self._threads:
            if thread.ident is not None:
                thread.join()
        with self._lock:
            if pending is not None:
                self._counts["proposal_returns"] += 1
                self._counts["discarded_on_close"] += 1
                try:
                    discarded = pending[1].result()
                    if not discarded["qualified"]:
                        self._counts["refused"] += 1
                except Exception:  # noqa: BLE001 - drain cannot erase failed work.
                    self._unknown = True
                    self._counts["worker_errors"] += 1
            self._pending = None
            self._active.close()
            return self.result()
