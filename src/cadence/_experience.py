"""Private integration of qualified activity and factual background learning.

This is a prediction experiment owner, not an action policy or public API.
One serial caller supplies present measurements and later actual outcomes.
Internal prediction mismatch recruits replay without per-population flags.
No checkpoint API is supplied: a Brain snapshot alone omits this owner's
pending forecast, evidence, demand and worker custody.
"""

from __future__ import annotations

import hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from ._attention import ActivitySession
from ._validation import canonical, integer, number
from .brain import Brain
from .ports import _values


class ForecastSession:
    """One owned brain with serial activity admissions and private batch work.

    ``step`` issues a fully qualified, unclamped forecast. ``feedback`` supplies
    the later facts for precisely that forecast, even if learning subsequently
    refuses. Learning runs on an immutable parameter generation. Only ``poll``
    may install its accepted parameters, preserving newer live activity.

    Demand persists until a later observation of the same input/measurement
    context confirms recovery. An unrelated quiet context cannot clear it.
    Unresolved-context overflow remains explicit and cannot claim recovery.
    This exact-context policy is a finite-fixture candidate, not learned
    generalization of attention. An accepted
    fit is not evidence of recovery. Each evidence revision funds at most one
    bounded batch. Terminal reward below a configured goal also leaves demand,
    but factual predictor replay is not a reward-to-action credit algorithm.
    Users of this private experiment must not claim that it solves planning.

    Methods require one serial owner. Worker threads provide independent
    progress; they promise neither parallel Python throughput nor hard timing.
    """

    _FROZEN = frozenset(
        ("forecasts", "threshold", "capacity", "batch_size", "learning_budget", "goal")
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
        brain,
        *,
        forecasts,
        threshold=0.05,
        capacity=32,
        batch_size=8,
        learning_budget=None,
        goal=None,
        workers=None,
    ):
        self._brain = Brain.from_snapshot(brain.snapshot())
        threshold = number(threshold, "threshold")
        if threshold < 0:
            raise ValueError("threshold must be nonnegative")
        capacity = integer(capacity, "capacity", 1)
        batch_size = integer(batch_size, "batch_size", 1)
        if capacity > 4096 or batch_size > capacity:
            raise ValueError("batch_size <= capacity <= 4096 is required")
        if isinstance(forecasts, (str, bytes)):
            raise ValueError("forecasts must be a nonempty sequence of output names")
        forecasts = tuple(forecasts)
        names = {o.name for o in self._brain._outputs}
        if (
            not forecasts
            or any(type(x) is not str for x in forecasts)
            or len(set(forecasts)) != len(forecasts)
            or not set(forecasts) <= names
        ):
            raise ValueError("forecasts must contain unique declared output names")
        self.forecasts, self.threshold = forecasts, threshold
        self.capacity, self.batch_size = capacity, batch_size
        self.learning_budget = (
            None
            if learning_budget is None
            else integer(learning_budget, "learning_budget")
        )
        self.goal = None if goal is None else number(goal, "goal")
        self._workers = None if workers is None else integer(workers, "workers", 1)
        self._forecast_indices = frozenset(
            self._brain._population_ranges[o.reads.name][i]
            for o in self._brain._outputs
            if o.name in forecasts
            for i in o.indices
        )
        self._activity = None
        self._retired = []
        self._retired_work = {}
        self._retired_counts = {}
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cadence-witness"
        )
        self._rows = deque(maxlen=capacity)
        self._receipts = deque(maxlen=capacity)
        self._pending = self._active = self._last_feedback = None
        self._demand = set()
        self._deficits = {}
        self._deficit_overflow = False
        self._generation = self._evidence = self._decision = self._last_requested = 0
        self._total_reward = 0.0
        self._terminal = self._closed = self._unknown = False
        self._learning_disabled = self._pool_uncertain = False
        self._learning_work = {}
        self._counts = dict.fromkeys(
            (
                "issued",
                "refused",
                "feedback",
                "surprises",
                "terminal_failures",
                "scheduled",
                "committed",
                "fit_refused",
                "fit_errors",
                "submit_errors",
                "discarded_on_close",
                "stale",
            ),
            0,
        )

    def _open(self):
        if self._closed:
            raise ValueError("session is closed")

    def _samples(self, supplied, nodes, *, partial=False):
        values = self._brain._mapping(supplied, nodes, partial=partial)
        result = {}
        for node in nodes:
            if node.name in values:
                flat = _values(values[node.name], node.shape, node.name)
                result[node.name] = flat if node.shape else flat[0]
        return result

    def step(self, inputs, *, targets=None, budget=256, seconds=60.0):
        """Issue one forecast; acknowledge it before another can be issued.

        Optional targets must be present factual measurements, never future
        goals. Forecast coordinates must remain free, including output aliases.
        A refusal leaves the accepted brain activity and forecast ID unchanged.
        """
        self._open()
        if self._pending is not None:
            raise ValueError("supply actual feedback before issuing another forecast")
        if self._terminal:
            raise ValueError("start_episode before issuing a new forecast")
        budget = integer(budget, "budget")
        seconds = number(seconds, "seconds", positive=True)
        inputs = self._samples(inputs, self._brain._inputs)
        targets = self._samples(
            {} if targets is None else targets, self._brain._outputs, partial=True
        )
        flat, fixed = self._brain._arguments(inputs, targets)
        if self._forecast_indices & fixed.keys():
            raise ValueError("forecast coordinates must remain unclamped")
        self.poll()
        self._reap()
        if self._retired:
            self._counts["refused"] += 1
            return {
                "accepted": False,
                "decision_id": None,
                "outputs": None,
                "activity": {"qualified": False, "reason": "previous_work_pending"},
            }
        if self._activity is None:
            self._activity = ActivitySession.from_brain(
                self._brain,
                inputs,
                targets=targets,
                workers=self._workers,
            )
        self._activity.begin(
            flat,
            clamps=fixed,
            weights=self._brain.weights,
            biases=self._brain.biases,
            budget=budget,
        )
        result = self._activity.run_until_complete(seconds=seconds)
        if not result["qualified"] or not self._activity.is_current(result):
            self._counts["refused"] += 1
            self._activity.close(wait=False)
            self._retired.append(self._activity)
            self._activity = None
            return {
                "accepted": False,
                "decision_id": None,
                "outputs": None,
                "activity": result,
            }
        self._brain._state = tuple(result["state"])
        outputs = self._brain._outputs_from(self._brain.state)
        self._decision += 1
        self._counts["issued"] += 1
        self._pending = dict(
            id=self._decision,
            generation=self._generation,
            inputs=inputs,
            targets=targets,
            outputs=outputs,
        )
        return {
            "accepted": True,
            "decision_id": self._decision,
            "outputs": deepcopy(outputs),
            "activity": result,
        }

    def feedback(self, decision_id, outcomes, *, reward=0.0, terminal=False):
        """Bind actual later measurements to an immutable issued prediction.

        Exact latest retries are idempotent. Conflicting, old, malformed and
        unissued outcomes cannot alter evidence, demand or parameter history.
        Every forecast output must be supplied, with no additional outcomes.
        """
        self._open()
        decision_id = integer(decision_id, "decision_id", 1)
        reward = number(reward, "reward")
        if type(terminal) is not bool:
            raise ValueError("terminal must be boolean")
        outcomes = self._samples(outcomes, self._brain._outputs, partial=True)
        if outcomes.keys() != set(self.forecasts):
            raise ValueError("outcomes must supply exactly the forecast outputs")
        payload = canonical([decision_id, outcomes, reward, terminal])
        if self._last_feedback is not None and decision_id == self._last_feedback[0]:
            if payload != self._last_feedback[1]:
                raise ValueError("conflicting latest feedback")
            return {**deepcopy(self._last_feedback[2]), "duplicate": True}
        if self._pending is None or decision_id != self._pending["id"]:
            raise ValueError("feedback must match the current issued decision")
        total = number(self._total_reward + reward, "total reward")
        pending = self._pending
        labels = {**pending["targets"], **outcomes}
        # Validates bounds and overlapping output aliases before accepting facts.
        self._brain._arguments(pending["inputs"], labels)
        values = {
            o.name: _values(outcomes[o.name], o.shape, o.name)
            for o in self._brain._outputs
            if o.name in self.forecasts
        }
        surprise = max(
            abs(actual - predicted)
            for name, row in values.items()
            for actual, predicted in zip(row, pending["outputs"][name], strict=True)
        )
        self._evidence += 1
        witness = dict(
            decision_id=decision_id,
            generation=pending["generation"],
            forecast=deepcopy(pending["outputs"]),
            reward=reward,
            terminal=terminal,
            surprise=surprise,
        )
        self._rows.append(
            (self._evidence, deepcopy(pending["inputs"]), deepcopy(labels), witness)
        )
        self._counts["feedback"] += 1
        self._total_reward, self._terminal = total, terminal
        context = hashlib.sha256(
            canonical([pending["inputs"], pending["targets"]]).encode()
        ).hexdigest()
        if surprise > self.threshold:
            if context not in self._deficits and len(self._deficits) >= self.capacity:
                del self._deficits[next(iter(self._deficits))]
                self._deficit_overflow = True
            self._deficits[context] = self._evidence
            self._demand.add("surprise")
            self._counts["surprises"] += 1
        elif pending["generation"] == self._generation:
            self._deficits.pop(context, None)
            if not self._deficits and not self._deficit_overflow:
                self._demand.discard("surprise")
        if terminal and self.goal is not None:
            if total < self.goal:
                self._demand.add("terminal_failure")
                self._counts["terminal_failures"] += 1
            else:
                self._demand.discard("terminal_failure")
        self._pending = None
        result = dict(
            duplicate=False,
            evidence_id=self._evidence,
            surprise=surprise,
            forecast_generation=pending["generation"],
            demand=sorted(self._demand),
        )
        self._last_feedback = (decision_id, payload, deepcopy(result))
        self._schedule()
        return result

    @staticmethod
    def _fit(snapshot, rows, budget):
        candidate = Brain.from_snapshot(snapshot)
        result = candidate.observe_batch(rows, budget=budget, source="witness")
        return candidate, result

    def _schedule(self):
        if (
            self._closed
            or self._learning_disabled
            or self._active is not None
            or not self._demand
            or not self._rows
            or self._evidence <= self._last_requested
        ):
            return
        rows = list(self._rows)
        # Most recent half + evenly spaced older witnesses preserves a bounded
        # mix of retained routine and new facts, without task-specific labels.
        recent = min(len(rows), max(1, self.batch_size // 2))
        older = rows[:-recent]
        count = min(len(older), self.batch_size - recent)
        selected = [older[i * len(older) // count] for i in range(count)] + rows[
            -recent:
        ]
        snapshot = self._brain.snapshot()
        job = dict(
            generation=self._generation,
            evidence=self._evidence,
            reasons=sorted(self._demand),
            evidence_ids=[r[0] for r in selected],
            witnesses=deepcopy(selected),
            parameter_sha256=self._parameter_digest(),
        )
        self._last_requested = self._evidence
        self._counts["scheduled"] += 1
        try:
            future = self._pool.submit(
                self._fit,
                snapshot,
                [(r[1], r[2]) for r in selected],
                self.learning_budget,
            )
        except RuntimeError as error:
            self._counts["submit_errors"] += 1
            # submit() may have queued work before thread creation failed.
            # Keep its possible work unknown, prevent future orphan execution,
            # and retire this pool without blocking foreground observations.
            self._unknown = self._learning_disabled = self._pool_uncertain = True
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._receipts.append(
                {"status": "submit_errors", "job": job, "error": str(error)}
            )
            return
        self._active = (job, future)

    def _reap(self, *, wait=False):
        pending = []
        for session in self._retired:
            result = session.close(wait=wait)
            if not result["work_complete"]:
                pending.append(session)
                continue
            self._unknown |= result["unknown_work"]
            for key, value in result["work"].items():
                self._retired_work[key] = self._retired_work.get(key, 0) + value
            for key, value in result["counts"].items():
                self._retired_counts[key] = self._retired_counts.get(key, 0) + value
        self._retired = pending

    def _parameter_digest(self):
        return hashlib.sha256(
            canonical(
                [
                    self._brain.weights,
                    self._brain.biases,
                    self._brain._event_id,
                    self._brain._event_digest,
                ]
            ).encode()
        ).hexdigest()

    def poll(self):
        """Collect at most one background proposal; preserve current live state."""
        self._open()
        receipt = {"status": "idle"}
        if self._active is not None:
            job, future = self._active
            if not future.done():
                return {"status": "running"}
            self._active = None
            try:
                candidate, result = future.result()
                for name, value in result.get("work", {}).items():
                    if isinstance(value, (int, float)):
                        self._learning_work[name] = (
                            self._learning_work.get(name, 0) + value
                        )
                if (
                    job["generation"] != self._generation
                    or job["parameter_sha256"] != self._parameter_digest()
                ):
                    status = "stale"
                elif result["accepted"]:
                    # Candidate is the privately owned native Brain that actually
                    # admitted this batch. Copy only its admitted model/event.
                    self._brain._admit(
                        result, candidate._event_id, candidate._event_digest
                    )
                    self._generation += 1
                    status = "committed"
                else:
                    status = "fit_refused"
                receipt = {"status": status, "job": job, "result": result}
            except Exception as error:  # noqa: BLE001 - keep failed work visible.
                self._unknown = True
                status = "fit_errors"
                receipt = {
                    "status": status,
                    "job": job,
                    "error": f"{type(error).__name__}: {error}",
                }
            self._counts[status] += 1
            self._receipts.append(deepcopy(receipt))
        self._schedule()
        return receipt

    def start_episode(self):
        """Reset reward accumulation, retaining actual facts and unresolved demand."""
        self._open()
        if self._pending is not None:
            raise ValueError("acknowledge the pending forecast first")
        self._total_reward, self._terminal = 0.0, False

    def inspect(self):
        return dict(
            generation=self._generation,
            evidence=self._evidence,
            pending=None if self._pending is None else self._pending["id"],
            demand=sorted(self._demand),
            unresolved_contexts=len(self._deficits),
            deficit_overflow=self._deficit_overflow,
            counts=dict(self._counts),
            learning_work=dict(self._learning_work),
            retired_activity_work=dict(self._retired_work),
            retired_activity_counts=dict(self._retired_counts),
            retired_pending=len(self._retired),
            unknown_work=self._unknown,
            active=self._active is not None,
            learning_disabled=self._learning_disabled,
            terminal=self._terminal,
            total_reward=self._total_reward,
            receipts=deepcopy(list(self._receipts)),
        )

    def close(self, *, wait=True):
        """Drain and account for work; never silently discard a factual outcome."""
        self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=False)
        if wait:
            self._pool_uncertain = False
        # Closed owners do not install background parameter proposals. The
        # result remains inspectable, but is marked discarded instead of learnt.
        if self._active is not None and self._active[1].done():
            job, future = self._active
            self._active = None
            try:
                _, result = future.result()
                for name, value in result.get("work", {}).items():
                    if isinstance(value, (int, float)):
                        self._learning_work[name] = (
                            self._learning_work.get(name, 0) + value
                        )
                self._receipts.append(
                    {"status": "discarded_on_close", "job": job, "result": result}
                )
                self._counts["discarded_on_close"] += 1
            except Exception as error:  # noqa: BLE001
                self._unknown = True
                self._counts["fit_errors"] += 1
                self._receipts.append(
                    {"status": "fit_errors", "job": job, "error": str(error)}
                )
        if self._activity is not None:
            self._retired.append(self._activity)
            self._activity = None
        self._reap(wait=wait)
        sessions = self._retired
        results = [s.close(wait=wait) for s in sessions]
        return {
            **self.inspect(),
            "activity": results,
            "work_complete": self._active is None
            and not self._pool_uncertain
            and all(r["work_complete"] for r in results),
        }
