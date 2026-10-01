"""Bounded replay and temporally linked reward credit through the patch rule.

Action values are settled patch outputs. The orchestration below computes
explicit Q-learning teaching estimates; it is not an additional optimizer,
biological dopamine model, or evidence of planning emerging inside a solve.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from importlib.resources import files
from types import MappingProxyType

from ._validation import canonical, integer, number, strict_json
from .brain import MAX_CHECKPOINT_BYTES, Brain
from .ports import _values

_IMPLEMENTATION = hashlib.sha256(
    files(__package__).joinpath("reinforcement.py").read_bytes()
).hexdigest()


def _feedback_digest(action, reward, following, terminal):
    return hashlib.sha256(
        canonical((action, reward, following, terminal)).encode()
    ).hexdigest()


class Reinforcement:
    """Learn discrete choices from bounded rewards and observed transitions.

    ``brain`` declares an ``action_input`` vector of length ``actions`` and
    one scalar ``value_output``. Other inputs describe the situation, including
    any explicit history and drives. ``act`` supplies a one-hot action to each
    candidate query and retains the selected qualified activity. ``feedback``
    acknowledges the issued decision identity and the action actually executed,
    pairing it with its reward and subsequent observation. An actuator may
    execute a different action; credit belongs to the acknowledged action.

    Alternatively, ``action_input=None`` and a tuple of ``actions`` scalar
    ``value_output`` names expose one distinct patch value per action in a
    single joint query. The same reward rule teaches only the selected output.

    Targets are ``(1-discount)*value_scale*reward/reward_scale +
    discount*clip(max_next_value, -value_scale, value_scale)``; terminal
    transitions omit the second term. ``discount`` is in [0,1), exploration
    in [0,1], and rewards in [-reward_scale,reward_scale]. Target projection
    and reward scaling are explicit parts of this algorithm. Updates use
    ``observe_batch(source='estimate')`` with no separate parameter updater.

    Replay is a bounded FIFO store sampled uniformly, always including the
    latest transition. It mitigates forgetting but does not guarantee retention.
    All calls, including direct access to ``brain``, require one serial owner.

    ``credit_horizon`` bounds the number of consecutive observed transitions
    in a target (one by default). Longer returns stop before a recorded action
    that is not greedy under the frozen pre-update values, or at a context gap,
    reset, terminal record or unavailable future. Reset marks an episode/credit
    boundary even when no action is pending. This is an experimental 0.60
    candidate, not a claim that long-horizon learning is already qualified.
    """

    def __init__(
        self,
        brain,
        *,
        actions,
        action_input="action",
        value_output="value",
        discount=0.95,
        exploration=0.1,
        reward_scale=1.0,
        value_scale=0.9,
        capacity=1024,
        batch_size=16,
        credit_horizon=1,
        seed=0,
    ):
        if not isinstance(brain, Brain):
            raise ValueError("brain must be a compiled Brain")
        actions = integer(actions, "actions", 2)
        capacity = integer(capacity, "capacity", 1)
        batch_size = integer(batch_size, "batch_size", 1)
        credit_horizon = integer(credit_horizon, "credit_horizon", 1)
        seed = integer(seed, "seed")
        if batch_size > capacity:
            raise ValueError("batch_size must not exceed capacity")
        if credit_horizon > capacity:
            raise ValueError("credit_horizon must not exceed replay capacity")
        discount = number(discount, "discount")
        exploration = number(exploration, "exploration")
        reward_scale = number(reward_scale, "reward_scale", positive=True)
        value_scale = number(value_scale, "value_scale", positive=True)
        if not 0 <= discount < 1 or not 0 <= exploration <= 1:
            raise ValueError("discount must be in [0,1); exploration in [0,1]")
        if value_scale >= min(
            1 / (1 + brain.config["state_prior"]), brain.config["state_bound"]
        ):
            raise ValueError("value_scale must be below the attainable output bound")
        inputs = {p.name: p for p in brain._inputs}
        outputs = {p.name: p for p in brain._outputs}
        if action_input is None:
            if (
                not isinstance(value_output, (tuple, list))
                or len(value_output) != actions
            ):
                raise ValueError(
                    "Vector values require one scalar output name per action"
                )
            value_output = tuple(value_output)
            names = value_output
        else:
            if (
                not isinstance(action_input, str)
                or action_input not in inputs
                or inputs[action_input].size != actions
            ):
                raise ValueError(
                    "action_input must name a sensor with one coordinate per action"
                )
            names = (value_output,)
        if any(
            not isinstance(name, str)
            or name not in outputs
            or len(outputs[name].indices) != 1
            for name in names
        ):
            raise ValueError("value_output must name scalar-valued outputs")
        selected = tuple(outputs[name] for name in names)
        if len({(value.reads.name, value.indices[0]) for value in selected}) != len(
            names
        ):
            raise ValueError("Action values must expose distinct patches, not aliases")
        self.brain = brain
        self.config = MappingProxyType(
            dict(
                actions=actions,
                action_input=action_input,
                value_output=value_output,
                discount=discount,
                exploration=exploration,
                reward_scale=reward_scale,
                value_scale=value_scale,
                capacity=capacity,
                batch_size=batch_size,
                credit_horizon=credit_horizon,
                seed=seed,
            )
        )
        self._sensors = tuple(p for p in brain._inputs if p.name != action_input)
        self._value_outputs = selected
        self._rng = random.Random(seed)
        try:
            self._records = deque(maxlen=capacity)
        except OverflowError as error:
            raise ValueError("capacity exceeds this platform's index range") from error
        self._pending = None
        self._transitions = 0
        self._updates = 0
        self._episode = 0
        self._issued_decisions = 0
        self._last_feedback = None

    def _context(self, inputs):
        supplied = self.brain._mapping(inputs, self._sensors)
        return {
            p.name: (
                _values(supplied[p.name], p.shape, p.name)[0]
                if not p.shape
                else tuple(_values(supplied[p.name], p.shape, p.name))
            )
            for p in self._sensors
        }

    def _inputs(self, context, action):
        if self.config["action_input"] is None:
            return dict(context)
        return {
            **context,
            self.config["action_input"]: tuple(
                float(i == action) for i in range(self.config["actions"])
            ),
        }

    def _values(self, context, budget):
        if self.config["action_input"] is None:
            result = self.brain.settle(context, budget=budget)
            return [result], tuple(
                result["outputs"][value.name][0] for value in self._value_outputs
            )
        results = [
            self.brain.settle(self._inputs(context, a), budget=budget)
            for a in range(self.config["actions"])
        ]
        return results, tuple(
            r["outputs"][self.config["value_output"]][0] for r in results
        )

    def act(self, inputs, *, explore=True, budget=None):
        """Select and retain a qualified action; call feedback before acting again.

        All candidate queries must qualify, including exploration. Refusal
        changes neither pending experience, RNG, nor brain state. Ties are
        broken uniformly. Returns action index, issued ``decision_id``, values
        and exploration status. A refusal has ``action=None`` and
        ``decision_id=None`` and consumes no identity. An accepted proposal is
        not execution evidence; feedback must acknowledge actual execution.
        ``explore=False`` disables epsilon moves.
        ``work`` includes every candidate query and the selected state update;
        ``budget`` bounds each underlying solve separately.
        """
        if type(explore) is not bool:
            raise ValueError("explore must be boolean")
        if self._pending is not None:
            raise ValueError("Supply feedback or reset the pending action first")
        context = self._context(inputs)
        results, values = self._values(context, budget)
        work = Counter()
        for result in results:
            work.update(result["work"])
        if not all(r["qualified"] for r in results):
            return dict(
                accepted=False,
                action=None,
                decision_id=None,
                values=values,
                reason="query_refused",
                work=dict(work),
            )
        before = self._rng.getstate()
        exploratory = explore and self._rng.random() < self.config["exploration"]
        best = max(values)
        choices = (
            list(range(len(values)))
            if exploratory
            else [i for i, v in enumerate(values) if v == best]
        )
        action = self._rng.choice(choices)
        try:
            result = self.brain.step(self._inputs(context, action), budget=budget)
        except Exception:
            self._rng.setstate(before)
            raise
        work.update(result["work"])
        if not result["accepted"]:
            self._rng.setstate(before)
            return dict(
                accepted=False,
                action=None,
                decision_id=None,
                values=values,
                reason="step_refused",
                work=dict(work),
            )
        self._issued_decisions += 1
        self._pending = (context, action, self._issued_decisions)
        return dict(
            accepted=True,
            action=action,
            decision_id=self._issued_decisions,
            values=values,
            exploratory=exploratory,
            settlement=result,
            work=dict(work),
        )

    def feedback(
        self,
        reward,
        next_inputs=None,
        *,
        decision_id,
        executed_action,
        terminal=False,
        learn=True,
        budget=None,
    ):
        """Record an actual outcome and optionally attempt one replay update.

        ``decision_id`` identifies the accepted proposal; ``executed_action``
        is the action the body actually performed, even if it differs from the
        proposal. Both acknowledgments are required. An unexecuted or expired
        proposal must be discarded with reset, not given fabricated feedback.

        Nonterminal outcomes require next_inputs. Terminal outcomes require
        None; terminal means there is no future reward continuation, not just
        that a collector stopped. Invalid arguments change nothing. A valid
        record is retained and consumes the pending action even if the later
        numerical update refuses or raises. An identical retry of the latest
        stored feedback returns ``stored=False, duplicate=True`` without any
        learning or pending-action change, including after a newer act. A
        changed payload or any other nonpending identity is rejected. Identity
        covers actual action, reward, next context and terminal status;
        ``learn`` and ``budget`` are execution options, not evidence. Retry
        learning with replay(), not feedback(). Feedback identities are separate
        from the brain's learning-admission event identities.
        ``learn=False`` records experience without changing learned parameters.
        """
        decision_id = integer(decision_id, "decision_id", 1)
        executed_action = integer(executed_action, "executed_action")
        if executed_action >= self.config["actions"]:
            raise ValueError("executed_action is outside the declared actions")
        if type(terminal) is not bool or type(learn) is not bool:
            raise ValueError("terminal and learn must be boolean")
        reward = number(reward, "reward")
        if abs(reward) > self.config["reward_scale"]:
            raise ValueError("reward exceeds reward_scale; scale rewards explicitly")
        if budget is not None:
            integer(budget, "budget")
        if terminal:
            if next_inputs is not None:
                raise ValueError("Terminal feedback requires next_inputs=None")
            following = None
        else:
            following = self._context(next_inputs)
        digest = _feedback_digest(executed_action, reward, following, terminal)
        if self._last_feedback is not None and decision_id == self._last_feedback[0]:
            if digest != self._last_feedback[1]:
                raise ValueError("Feedback conflicts with the latest stored outcome")
            return dict(
                accepted=False,
                stored=False,
                duplicate=True,
                decision_id=decision_id,
                reason="duplicate_feedback",
                transitions=self._transitions,
                work={},
            )
        if self._pending is None or decision_id != self._pending[2]:
            raise ValueError("decision_id must identify the pending accepted act")
        context, _, _ = self._pending
        self._records.append(
            (context, executed_action, reward, following, self._episode, decision_id)
        )
        self._last_feedback = decision_id, digest
        if terminal:
            self._episode += 1
        self._transitions += 1
        self._pending = None
        result = (
            self.replay(budget=budget)
            if learn
            else dict(accepted=False, reason="learning_disabled")
        )
        return {
            **result,
            "stored": True,
            "duplicate": False,
            "decision_id": decision_id,
            "transitions": self._transitions,
        }

    def _target(self, index, budget, work):
        """Finite greedy-cut return, always using the frozen pre-update brain.

        Continue along actual adjacent records only while the next recorded
        action is greedy under these parameters. Otherwise bootstrap at that
        observation. No rewards beyond a cut, terminal, reset, missing record
        or horizon enter this target. This is target construction, not another
        parameter update rule or a nonlinear-learning convergence guarantee.
        """
        rewards = []
        scale, discount = self.config["value_scale"], self.config["discount"]
        cursor = index
        while True:
            _, _, reward, following, episode, _ = self._records[cursor]
            rewards.append(
                (1 - discount) * scale * (reward / self.config["reward_scale"])
            )
            if following is None:
                value, reason = 0.0, "terminal"
                break
            if discount == 0:
                # Future values cannot affect an immediate-reward target. An
                # irrelevant bootstrap query must not block its admission.
                value, reason = 0.0, "zero_discount"
                break
            results, values = self._values(following, budget)
            for result in results:
                work.update(result["work"])
            if not all(result["qualified"] for result in results):
                return None, len(rewards), "bootstrap_refused"
            best = max(values)
            value = max(-scale, min(scale, best))
            if len(rewards) == self.config["credit_horizon"]:
                reason = "horizon"
                break
            if cursor + 1 == len(self._records):
                reason = "pending_future"
                break
            next_context, next_action, _, _, next_episode, _ = self._records[cursor + 1]
            if next_episode != episode or following != next_context:
                reason = "discontinuity"
                break
            if values[next_action] != best:
                reason = "off_policy"
                break
            cursor += 1
        for reward in reversed(rewards):
            value = reward + discount * value
        return value, len(rewards), reason

    def replay(self, *, budget=None):
        """Fit one sampled batch of Q estimates, preserving current live activity.

        Compute all targets with the unchanged pre-update brain, using up to
        credit_horizon observed transitions with greedy-action cuts. Terminal
        samples never bootstrap. Qualification failures commit no parameters;
        replay contents remain available. This is an approximate Q-learning
        algorithm with a nonlinear function approximator, not a convergence
        guarantee. Each call charges ordinary brain work for every next-action
        query as well as the batch fit. The returned ``work`` aggregates all
        those solves; other solve diagnostics describe the fit. ``budget``
        bounds each underlying solve separately.
        """
        if budget is not None:
            integer(budget, "budget")
        if not self._records:
            return dict(accepted=False, reason="empty_replay")
        rng_before = self._rng.getstate()
        count = min(len(self._records), self.config["batch_size"])
        indices = self._rng.sample(range(len(self._records) - 1), count - 1) + [
            len(self._records) - 1
        ]
        examples, targets, horizons, stops = [], [], [], []
        work = Counter()
        try:
            for index in indices:
                context, action, _, _, _, _ = self._records[index]
                target, horizon, stop = self._target(index, budget, work)
                if target is None:
                    self._rng.setstate(rng_before)
                    return dict(
                        accepted=False, reason="bootstrap_refused", work=dict(work)
                    )
                targets.append(target)
                horizons.append(horizon)
                stops.append(stop)
                output = self._value_outputs[
                    action if self.config["action_input"] is None else 0
                ]
                examples.append(
                    (
                        self._inputs(context, action),
                        {
                            output.name: target if not output.shape else (target,),
                        },
                    )
                )
            result = self.brain.observe_batch(
                examples, budget=budget, source="estimate"
            )
        except Exception:
            self._rng.setstate(rng_before)
            raise
        if result["accepted"]:
            self._updates += 1
        else:
            self._rng.setstate(rng_before)
        work.update(result["work"])
        return {
            **result,
            "indices": tuple(indices),
            "targets": tuple(targets),
            "credit_horizons": tuple(horizons),
            "credit_stops": tuple(stops),
            "updates": self._updates,
            "work": dict(work),
        }

    def reset(self):
        """End a credit segment without reward; issued identities are never reused."""
        self._pending = None
        self._episode += 1

    def inspect(self):
        """Return configuration and experience counts without exposing owned data."""
        return dict(
            config=dict(self.config),
            records=len(self._records),
            transitions=self._transitions,
            updates=self._updates,
            pending=self._pending is not None,
            episode=self._episode,
            issued_decisions=self._issued_decisions,
            pending_decision_id=self._pending[2] if self._pending else None,
            pending_action=self._pending[1] if self._pending else None,
            last_feedback_id=self._last_feedback[0] if self._last_feedback else None,
        )

    def snapshot(self):
        """Save brain, replay, pending action and RNG for exact continuation."""
        text = canonical(
            dict(
                schema="reinforcement/3",
                implementation=_IMPLEMENTATION,
                brain=self.brain.snapshot(),
                config=dict(self.config),
                records=list(self._records),
                pending=self._pending,
                rng=self._rng.getstate(),
                transitions=self._transitions,
                updates=self._updates,
                episode=self._episode,
                issued_decisions=self._issued_decisions,
                last_feedback=self._last_feedback,
            )
        )
        if len(text.encode()) > MAX_CHECKPOINT_BYTES:
            raise ValueError("Checkpoint exceeds text-size budget")
        return text

    @classmethod
    def from_snapshot(cls, text):
        """Validate a complete checkpoint before constructing a continuation."""
        data = strict_json(text, MAX_CHECKPOINT_BYTES)
        fields = {
            "schema",
            "implementation",
            "brain",
            "config",
            "records",
            "pending",
            "rng",
            "transitions",
            "updates",
            "episode",
            "issued_decisions",
            "last_feedback",
        }
        if (
            not isinstance(data, dict)
            or set(data) != fields
            or data["schema"] != "reinforcement/3"
            or data["implementation"] != _IMPLEMENTATION
        ):
            raise ValueError("Unsupported reinforcement checkpoint")
        try:
            obj = cls(Brain.from_snapshot(data["brain"]), **data["config"])
            if set(data["config"]) != set(obj.config):
                raise ValueError("Checkpoint needs complete configuration")
            obj._transitions = integer(data["transitions"], "transitions")
            obj._updates = integer(data["updates"], "updates")
            obj._episode = integer(data["episode"], "episode")
            obj._issued_decisions = integer(
                data["issued_decisions"], "issued_decisions"
            )
            if obj._transitions > obj._issued_decisions:
                raise ValueError("Transitions exceed issued decisions")
            if obj._updates and not obj._transitions:
                raise ValueError("Updates require recorded transitions")
            if not isinstance(data["records"], list) or len(data["records"]) != min(
                obj._transitions, obj.config["capacity"]
            ):
                raise ValueError("Invalid replay length")

            def action_index(value):
                action = integer(value, "action")
                if action >= obj.config["actions"]:
                    raise ValueError("Invalid action index")
                return action

            previous = None
            for row in data["records"]:
                if not isinstance(row, list) or len(row) != 6:
                    raise ValueError("Invalid transition")
                context, action = obj._context(row[0]), action_index(row[1])
                reward = number(row[2], "reward")
                if abs(reward) > obj.config["reward_scale"]:
                    raise ValueError("Invalid reward bound")
                following = None if row[3] is None else obj._context(row[3])
                episode = integer(row[4], "record episode")
                decision_id = integer(row[5], "record decision_id", 1)
                if decision_id > obj._issued_decisions or decision_id < (
                    obj._transitions - len(data["records"]) + 1
                ):
                    raise ValueError("Invalid recorded decision identity")
                if episode > obj._episode or (
                    following is None and episode == obj._episode
                ):
                    raise ValueError("Invalid episode continuation")
                if previous is not None and (
                    episode < previous[4]
                    or (previous[3] is None and episode == previous[4])
                    or decision_id <= previous[5]
                    or (episode == previous[4] and decision_id != previous[5] + 1)
                ):
                    raise ValueError("Invalid replay episode ordering")
                # Each missing issued decision was abandoned by reset, which
                # advances the episode. A terminal record advances it as well.
                if previous is None:
                    canceled = decision_id - (
                        obj._transitions - len(data["records"]) + 1
                    )
                    available_episodes = episode
                else:
                    canceled = decision_id - previous[5] - 1
                    available_episodes = (
                        episode - previous[4] - int(previous[3] is None)
                    )
                if canceled > available_episodes:
                    raise ValueError("Recorded decision gaps require reset episodes")
                previous = (context, action, reward, following, episode, decision_id)
                obj._records.append(previous)
            receipt = data["last_feedback"]
            if previous is None:
                if receipt is not None:
                    raise ValueError("Feedback receipt requires a recorded transition")
            else:
                if (
                    not isinstance(receipt, list)
                    or len(receipt) != 2
                    or integer(receipt[0], "last feedback decision_id", 1)
                    != previous[5]
                    or receipt[1]
                    != _feedback_digest(
                        previous[1], previous[2], previous[3], previous[3] is None
                    )
                ):
                    raise ValueError("Feedback receipt must match the latest record")
                obj._last_feedback = tuple(receipt)
            pending = data["pending"]
            if pending is not None:
                if not isinstance(pending, list) or len(pending) != 3:
                    raise ValueError("Invalid pending action")
                identity = integer(pending[2], "pending decision_id", 1)
                if identity != obj._issued_decisions or (
                    previous is not None and identity <= previous[5]
                ):
                    raise ValueError("Invalid pending decision identity")
                obj._pending = (
                    obj._context(pending[0]),
                    action_index(pending[1]),
                    identity,
                )
            # With no reset after the latest nonterminal record, a pending
            # decision must be its immediate successor. The same accounting
            # covers empty histories and gaps across genuine reset episodes.
            last_identity, last_episode, terminal = (
                (0, 0, False)
                if previous is None
                else (previous[5], previous[4], previous[3] is None)
            )
            canceled = (
                obj._issued_decisions - last_identity - int(obj._pending is not None)
            )
            if canceled < 0 or canceled > obj._episode - last_episode - int(terminal):
                raise ValueError("Issued decision gaps require reset episodes")
            rng = data["rng"]
            if (
                not isinstance(rng, list)
                or len(rng) != 3
                or rng[0] != 3
                or type(rng[0]) is not int
                or rng[2] is not None
                or not isinstance(rng[1], list)
                or len(rng[1]) != 625
            ):
                raise ValueError("Invalid RNG state")
            if any(
                type(v) is not int or not 0 <= v <= (624 if i == 624 else 2**32 - 1)
                for i, v in enumerate(rng[1])
            ):
                raise ValueError("Invalid RNG word")
            obj._rng.setstate((3, tuple(rng[1]), None))
            if obj._updates > obj.brain.inspect()["admissions"]:
                raise ValueError("Update count exceeds brain admissions")
            return obj
        except (TypeError, KeyError, OverflowError, AttributeError) as error:
            raise ValueError("Malformed reinforcement checkpoint") from error
