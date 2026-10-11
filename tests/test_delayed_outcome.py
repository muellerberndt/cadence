"""An outcome that arrives after the stream has sensed more: ``Brain.wait`` and ``decision_id``.

Contract
--------
A ``live`` action owns the next actual outcome of its stream. ``Brain.wait`` settles the
observations that arrive before that outcome: the stream's activity and its working trace
advance, while the awaited action keeps the forecasts made before it, its eligibility, the
situation it was chosen in and its identity. Parameters, the critic, eligibility traces,
associative memory, random state, the copy of the issued command and the arousal state stay as
they were; each settle is reported by ``last_settlement``, outside arousal's counts. When
``live`` later receives the outcome, it is credited as an immediate outcome of that action would
be: the same actor and critic eligibility, the same forecast and the same associative record.
The next state settles from the state the stream sensed last, as do the next answer, the next
routine forecast and imagination; a finished episode still starts from rest.
``Brain.decision_id`` names the awaited ``live`` action; an outcome reported under any other
identity is refused before anything changes. A checkpoint taken while waiting resumes the wait.

The checks below exercise those statements on small composed brains: against an immediate twin
restored from a checkpoint taken before the wait, against transcriptions of the settles that
must start from the sensed state (the same solver called with that state), and with probes that
record the state each settle starts from. A mutation harness replaces runtime methods by
deliberately wrong variants and records which checks kill each one.

What neither side establishes: a behavioral or cognitive gain from waiting, the timing of a
real body, more than one awaited action per stream, batched streams, or that the body executed
the issued action. Comparisons use exact equality where both sides perform the same operations
in the same order.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import cadence as cd

EYE = np.eye(4)
TORCH = importlib.util.find_spec("torch") is not None


def _life(*, youth: int = 0, seed: int = 0, efference: float = 0.3, **options: Any) -> cd.Brain:
    return cd.Brain.compose(
        4,
        2,
        modules=(8,),
        seed=seed,
        working_memory_amplitude=0.3,
        efference_amplitude=efference,
        arousal=cd.ArousalConfig(youth=youth),
        **options,
    )


def _twin(brain: cd.Brain, path, **options: Any) -> cd.Brain:
    return cd.Brain.load(brain.save(path), **options)


def _durable(brain: cd.Brain) -> dict[str, Any]:
    """Everything an awaited outcome must not change before it arrives."""
    agent, memory = brain.basal_ganglia, brain.hippocampus
    values: dict[str, Any] = {
        "efficacy": np.asarray(brain.brain.efficacy).copy(),
        "bias": np.asarray(brain.brain.bias).copy(),
        "critic": np.append(agent.w_critic, agent.b_critic),
        "moments": np.concatenate(
            [agent.velocity, agent.velocity_bias, agent.second_moment, agent.second_moment_bias]
        ),
        "consolidated": memory.consolidated.copy(),
        "strength": memory.strength.copy(),
        "mass": memory.mass.copy(),
        "counts": np.array([brain.learner.updates, agent.updates, memory.writes]),
        "rng": json.dumps(brain.rng.bit_generator.state, sort_keys=True, default=str),
        "actor_rng": json.dumps(agent.rng.bit_generator.state, sort_keys=True, default=str),
        "arousal": json.dumps(brain.arousal.to_dict(), sort_keys=True),
    }
    for name in ("trace", "trace_bias", "trace_critic"):
        value = getattr(agent, name)
        values[name] = None if value is None else value.copy()
    if agent._trace_device is not None:
        for name, tensor in zip(("trace", "trace_bias"), agent._trace_device, strict=True):
            values["device_" + name] = tensor.detach().cpu().double().numpy().copy()
    if brain.efference is not None:
        for name in ("trace", "last", "cold"):
            values["efference/" + name] = getattr(brain.efference, name).copy()
    return values


def _stream(brain: cd.Brain) -> dict[str, Any]:
    """The short-term state an outcome after a finished episode must leave as it would be."""
    values = _durable(brain)
    for name in ("trace", "last", "cold"):
        values["working/" + name] = getattr(brain.working_memory, name).copy()
    state = brain.basal_ganglia.state
    for name in ("v", "activation", "adaptation"):
        values["free/" + name] = np.asarray(getattr(state, name)).copy()
    return values


def _same(first: dict[str, Any], second: dict[str, Any]) -> None:
    assert first.keys() == second.keys()
    for name, value in first.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(second[name], value, err_msg=name)
        else:
            assert second[name] == value, name


def _owing(
    *, youth: int, seed: int = 0, efference: float = 0.3, signed: bool = True, **options: Any
) -> cd.Brain:
    """A life whose last action awaits its outcome. A young life has the eligibility of
    earlier sampled outcomes; a calm life paid nothing (``signed=False``) acts in routine."""
    brain = _life(youth=youth, seed=seed, efference=efference, **options)
    action = brain.live(EYE[[0]])
    for moment in range(4):
        reward = (1.0 if int(action[0]) == moment % 2 else -1.0) if signed else 0.0
        action = brain.live(EYE[[(moment + 1) % 4]], reward=[reward])
    assert brain.pending_feedback
    return brain


def _spy(owner: Any, name: str, record: list[Any], argument: str) -> None:
    """Record the starting state each call of ``owner.name`` receives."""
    original = getattr(owner, name)

    def spy(*args: Any, **kwargs: Any) -> Any:
        record.append(kwargs.get(argument, args[1] if len(args) > 1 else None))
        return original(*args, **kwargs)

    setattr(owner, name, spy)


# ---------------------------------------------------------------------------
# The checks


def check_waiting_keeps_the_awaited_action() -> None:
    """Custody: waiting takes no outcome, issues nothing and changes no durable state."""
    for sampled in (True, False):
        brain = _owing(youth=40) if sampled else _owing(youth=0, signed=False)
        agent = brain.basal_ganglia
        assert (agent._pending is not None) == sampled and brain._lived[3] == sampled
        if sampled:
            assert agent.trace is not None and np.abs(agent.trace).max() > 0
        pending, lived, state = agent._pending, brain._lived, agent.state
        moment = None if brain._moment is None else tuple(a.copy() for a in brain._moment)
        before, decision_id, reading = _durable(brain), brain.decision_id, brain.last_arousal
        assert decision_id == brain.arousal.age
        for frame in (EYE[[1]], EYE[[2]], EYE[[3]]):
            assert brain.wait(frame) is None
        assert agent._pending is pending and brain._lived is lived and agent.state is state
        if moment is None:
            assert brain._moment is None
        else:
            for kept, value in zip(moment, brain._moment, strict=True):
                np.testing.assert_array_equal(value, kept)
        _same(before, _durable(brain))
        assert brain.pending_feedback and brain.decision_id == decision_id
        assert brain.last_arousal is reading


def check_waiting_advances_activity_and_working_trace(tmp_path) -> None:
    """The second wait settles from the first one's state; the trace follows each state."""
    brain = _owing(youth=40)
    brain.wait(EYE[[1]])
    first = brain._awaiting[1]
    assert not np.array_equal(first.activation, brain.basal_ganglia.state.activation)
    twin = _twin(brain, tmp_path / "after-first-wait.npz")
    trace = brain.working_memory
    old = trace.trace.copy()
    brain.wait(EYE[[2]])
    sensed = brain._awaiting[1]
    cfg = twin.learner.config
    expected = twin._equilibrate(
        twin.stimulus(EYE[[2]]), twin._awaiting[1], budget=cfg.free_steps, tolerance=cfg.tolerance
    ).state
    np.testing.assert_array_equal(sensed.activation, expected.activation)
    np.testing.assert_array_equal(sensed.v, expected.v)
    source = np.asarray(sensed.activation)[:, brain.association_index]
    np.testing.assert_array_equal(trace.trace, trace.decay * old + (1.0 - trace.decay) * source)
    np.testing.assert_array_equal(trace.last, source)


def check_waiting_work_is_reported_beside_the_arousal_counts() -> None:
    """Each wait reports its settle; arousal counts live moments, and its totals stay the sum
    of the moments' readings in a life that waits."""
    brain = _life(youth=3)
    brain.live(EYE[[0]])
    readings = [brain.last_arousal["sweeps"]]
    for moment in range(8):
        for frame in range(moment % 3):
            before = brain.arousal.to_dict()
            brain.wait(EYE[[(moment + frame + 1) % 4]])
            report = brain.last_settlement
            assert report["operation"] == "wait" and report["qualified"] and report["steps"] > 0
            assert brain.arousal.to_dict() == before
        brain.live(EYE[[moment % 4]], reward=[float(moment % 2)], decision_id=brain.decision_id)
        readings.append(brain.last_arousal["sweeps"])
    assert sum(brain.arousal.sweeps.values()) == sum(readings)
    assert sum(brain.arousal.moments.values()) == brain.arousal.age == 9


def check_a_delayed_outcome_is_credited_as_an_immediate_one(tmp_path) -> None:
    """The waited brain and an immediate twin credit the same action, forecast and record."""
    brain = _owing(youth=40)
    assert brain.basal_ganglia._pending is not None
    twin = _twin(brain, tmp_path / "before-the-wait.npz")
    forecast = float(brain.basal_ganglia._pending[3][0])
    decision_id = brain.decision_id
    for frame in (EYE[[1]], EYE[[2]]):
        brain.wait(frame)
    brain.live(EYE[[3]], reward=[1.0], decision_id=decision_id)
    twin.live(EYE[[3]], reward=[1.0], decision_id=decision_id)
    for one in (brain, twin):
        assert one.last_learning["value"] == forecast
        assert one.last_arousal["learned"]
    agent, other = brain.basal_ganglia, twin.basal_ganglia
    for name in ("trace", "trace_bias", "trace_critic"):
        np.testing.assert_array_equal(getattr(agent, name), getattr(other, name), err_msg=name)
    for name in ("consolidated", "strength", "mass"):
        np.testing.assert_array_equal(
            getattr(brain.hippocampus, name), getattr(twin.hippocampus, name), err_msg=name
        )
    assert brain.hippocampus.writes == twin.hippocampus.writes


def check_a_finished_episode_after_a_wait_starts_from_rest(tmp_path) -> None:
    """With ``done``, the next life starts from rest: the waited brain and an immediate twin
    are the same afterwards, for a sampled and for a routine awaited action."""
    for sampled in (True, False):
        brain = _owing(youth=40) if sampled else _owing(youth=0, signed=False)
        twin = _twin(brain, tmp_path / f"finished-{sampled}.npz")
        for frame in (EYE[[1]], EYE[[2]]):
            brain.wait(frame)
        decision_id = brain.decision_id
        a = brain.live(EYE[[3]], reward=[1.0], done=[True], decision_id=decision_id)
        b = twin.live(EYE[[3]], reward=[1.0], done=[True], decision_id=decision_id)
        np.testing.assert_array_equal(a, b)
        _same(_stream(brain), _stream(twin))
        for moment in range(6):
            reward = [1.0 if int(a[0]) == moment % 2 else -1.0]
            a = brain.live(EYE[[moment % 4]], reward=reward)
            b = twin.live(EYE[[moment % 4]], reward=reward)
            np.testing.assert_array_equal(a, b)
        _same(_stream(brain), _stream(twin))


def check_the_next_state_settles_from_the_sensed_state(tmp_path) -> None:
    """``learn`` after a wait: the transcription of its next-state settle from the sensed state."""
    brain = _owing(youth=40)
    for frame in (EYE[[1]], EYE[[2]]):
        brain.wait(frame)
    twin = _twin(brain, tmp_path / "waiting.npz")
    reward, done, following = np.array([1.0]), np.array([False]), EYE[[3]]
    keys, action = twin._moment
    twin._record(keys, action, reward, np.abs(reward))
    expected = twin.learner.free(twin.stimulus(following), warm=twin._awaiting[1])
    brain.learn(reward, done, following)
    state = brain.basal_ganglia.state
    np.testing.assert_array_equal(state.activation, expected.activation)
    np.testing.assert_array_equal(state.v, expected.v)
    assert brain._awaited() is None and not brain.pending_feedback


def check_every_settle_after_a_wait_starts_from_the_sensed_state() -> None:
    """Probes on the solver: the next wait, imagination, an answer, the outcome's next state and
    a routine forecast start from the sensed state; a finished episode's forecast from rest."""
    brain = _owing(youth=40)
    action_state = brain.basal_ganglia.state
    brain.wait(EYE[[1]])
    sensed = brain._awaiting[1]
    starts: list[Any] = []
    _spy(brain, "_equilibrate", starts, "state")
    brain.wait(EYE[[2]])
    assert starts[-1] is sensed
    sensed = brain._awaiting[1]
    brain.imagine([EYE[[3]]])
    np.testing.assert_array_equal(starts[-1].activation, sensed.activation)
    assert not np.array_equal(starts[-1].activation, action_state.activation)
    warmed: list[Any] = []
    _spy(brain.learner, "free", warmed, "warm")
    brain.live(EYE[[3]], reward=[1.0], decision_id=brain.decision_id)
    assert warmed[0] is sensed  # the outcome's next state
    routine = _owing(youth=0, signed=False)
    routine.wait(EYE[[1]])
    sensed = routine._awaiting[1]
    starts = []
    _spy(routine, "_equilibrate", starts, "state")
    routine.live(EYE[[2]], reward=[0.0], decision_id=routine.decision_id)
    assert starts[0] is sensed  # the routine forecast
    finished = _owing(youth=0, signed=False)
    finished.wait(EYE[[1]])
    starts = []
    _spy(finished, "_equilibrate", starts, "state")
    finished.live(EYE[[2]], reward=[0.0], done=[True], decision_id=finished.decision_id)
    assert starts[0] is None  # a finished episode's forecast starts from rest
    answer = _owing(youth=40)
    answer.wait(EYE[[1]])
    sensed = answer._awaiting[1]
    starts = []
    _spy(answer, "_equilibrate", starts, "state")
    answer.act(EYE[[2]], greedy=True)
    assert starts[0] is sensed  # an answer that replaces the awaited action


def check_the_device_next_state_settles_from_the_sensed_state() -> None:
    """On the torch backend the outcome's next state also starts from the sensed state."""
    if not TORCH:
        return
    brain = _owing(youth=40, backend="torch")
    brain.wait(EYE[[1]])
    sensed = brain._awaiting[1]
    warmed: list[Any] = []
    _spy(brain.learner, "free", warmed, "warm")
    brain.live(EYE[[2]], reward=[1.0], decision_id=brain.decision_id)
    assert warmed[0] is sensed


def check_decision_ids_name_the_awaited_live_action() -> None:
    """None before an action; ``arousal.age`` once ``live`` issues one; unchanged by waiting,
    never reused after a reset."""
    brain = _life(youth=40)
    assert brain.decision_id is None
    brain.live(EYE[[0]])
    assert brain.decision_id == 1 == brain.arousal.age
    brain.wait(EYE[[1]])
    assert brain.decision_id == 1
    brain.live(EYE[[2]], reward=[0.0], decision_id=np.int64(1))
    assert brain.decision_id == 2 == brain.arousal.age
    brain.act(EYE[[3]])  # an action that act sampled owns the outcome, without a number
    assert brain.pending_feedback and brain.decision_id is None
    brain.live(EYE[[0]], reward=[0.0])  # adopted as before
    assert brain.decision_id == 3
    brain.reset()
    assert brain.decision_id is None
    brain.live(EYE[[1]])
    assert brain.decision_id == 4


def check_an_outcome_of_another_decision_is_refused() -> None:
    """Stale, unissued and malformed identities change nothing; the awaited one is taken once."""
    brain = _owing(youth=40)
    brain.wait(EYE[[1]])
    owner = brain.decision_id
    sensed, trace = brain._awaiting, brain.working_memory.trace.copy()
    before = _durable(brain)
    for bad in (owner - 1, owner + 1, 0, -1, True, np.bool_(True), float(owner), str(owner)):
        with pytest.raises(ValueError, match="decision_id"):
            brain.live(EYE[[2]], reward=[1.0], decision_id=bad)
    _same(before, _durable(brain))
    assert brain._awaiting is sensed and brain.decision_id == owner
    np.testing.assert_array_equal(brain.working_memory.trace, trace)
    updates = brain.basal_ganglia.updates
    brain.live(EYE[[2]], reward=[1.0], decision_id=owner)
    assert brain.basal_ganglia.updates == updates + 1 and brain.decision_id == owner + 1
    before = _durable(brain)
    with pytest.raises(ValueError, match="does not own the next outcome"):
        brain.live(EYE[[3]], reward=[1.0], decision_id=owner)  # the same outcome, again
    _same(before, _durable(brain))
    assert brain.decision_id == owner + 1


def check_an_accepted_outcome_is_not_taken_again_after_the_answer_refuses() -> None:
    """Once taken, an outcome is gone even if the following answer refuses."""
    brain = _owing(youth=40)
    brain.wait(EYE[[1]])
    owner = brain.decision_id
    updates = brain.basal_ganglia.updates

    def refuse(x):
        raise RuntimeError("brain did not settle within 0 steps")

    brain._settled = refuse  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.live(EYE[[2]], reward=[1.0], decision_id=owner)
    del brain._settled
    assert brain.basal_ganglia.updates == updates + 1
    assert not brain.pending_feedback and brain.decision_id is None
    with pytest.raises(ValueError, match="live\\(observations\\) alone"):
        brain.live(EYE[[2]], reward=[1.0], decision_id=owner)
    with pytest.raises(RuntimeError, match="awaiting its outcome"):
        brain.wait(EYE[[2]])
    brain.live(EYE[[2]])
    assert brain.basal_ganglia.updates == updates + 1 and brain.decision_id == owner + 1


def check_a_refused_wait_changes_nothing() -> None:
    """A wait that cannot qualify, or with invalid observations, leaves the stream as it was."""
    brain = _owing(youth=40)
    brain.wait(EYE[[1]])
    sensed, trace = brain._awaiting, brain.working_memory.trace.copy()
    before = _durable(brain)
    for bad in (EYE[:2], np.ones((1, 3)), np.array([[np.nan, 0.0, 0.0, 0.0]])):
        with pytest.raises(ValueError):
            brain.wait(bad)
    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=0)
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.wait(EYE[[2]])
    assert brain.last_settlement["operation"] == "wait" and not brain.last_settlement["qualified"]
    _same(before, _durable(brain))
    assert brain._awaiting is sensed
    np.testing.assert_array_equal(brain.working_memory.trace, trace)
    brain.learner.config = config
    brain.wait(EYE[[2]])
    assert brain._awaiting is not sensed


def check_waiting_needs_a_live_action_awaiting_its_outcome() -> None:
    plain = cd.Brain.compose(4, 2, modules=(8,), seed=0)
    plain.step(EYE[[0]])
    with pytest.raises(ValueError, match="arousal"):
        plain.wait(EYE[[1]])
    brain = _life(youth=40)
    with pytest.raises(RuntimeError, match="awaiting its outcome"):
        brain.wait(EYE[[0]])
    brain.live(EYE[[0]])
    brain.act(EYE[[1]], greedy=True)  # a greedy act takes the stream; no outcome is owed
    before = _durable(brain)
    with pytest.raises(RuntimeError, match="awaiting its outcome"):
        brain.wait(EYE[[2]])
    _same(before, _durable(brain))
    assert brain._awaiting is None


def check_a_routine_outcome_after_a_wait_is_measured_against_its_own_forecasts() -> None:
    """A calm action waits; its outcome is measured against the record held when it acted."""
    brain = _life(youth=0)
    for _ in range(6):
        brain.live(EYE[[0]], reward=None if brain.decision_id is None else [0.0])
    lived = brain._lived
    assert not lived[3] and brain.last_arousal["mode"] == "routine"
    action, record = int(lived[1][0]), lived[6]
    recalled = {cue: brain.hippocampus.recall(EYE[[cue]])[0].copy() for cue in (1, 2)}
    for frame in (EYE[[1]], EYE[[2]]):
        brain.wait(frame)
    brain.live(EYE[[3]], reward=[-1.0], decision_id=brain.decision_id)
    reading = brain.last_arousal
    assert reading["record_error"] == pytest.approx(abs(-1.0 - record))
    assert reading["mode"] == "aroused" and reading["recorded"] and not reading["learned"]
    assert brain.hippocampus.recall(EYE[[0]])[0][action] < -0.5  # where the action was chosen
    for cue, value in recalled.items():  # and not where the stream waited
        np.testing.assert_array_equal(brain.hippocampus.recall(EYE[[cue]])[0], value)


def check_a_life_saved_while_waiting_continues_identically(tmp_path) -> None:
    """A sampled, a routine and an ``act`` action awaited across a save: the same life after."""
    for case in ("sampled", "routine", "acted"):
        if case == "routine":
            brain = _owing(youth=0, efference=0.0, signed=False, seed=4)
        else:
            brain = _owing(youth=40, seed=2)
        if case == "acted":
            brain.act(EYE[[2]])
            assert brain.decision_id is None and brain.pending_feedback
        for frame in (EYE[[1]], EYE[[2]]):
            brain.wait(frame)
        path = brain.save(tmp_path / f"waiting-{case}.npz")
        with np.load(path) as saved:
            meta = json.loads(str(saved["generic"]))
            assert meta["format"] == "cadence-generic/5" and "awaiting" in meta
            assert {"awaiting/v", "awaiting/activation", "awaiting/adaptation"} <= set(saved.files)
        twin = cd.Brain.load(path)
        assert twin.decision_id == brain.decision_id and twin.pending_feedback
        sensed, restored = brain._awaited(), twin._awaited()
        assert restored is not None
        for name in ("v", "activation", "adaptation"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(sensed, name))
        a = None
        for moment in range(24):
            if moment == 0:
                for one in (brain, twin):
                    one.wait(EYE[[3]])
                    assert one.last_settlement["steps"] == brain.last_settlement["steps"]
                reward = [1.0]
            else:
                reward = [1.0 if int(a[0]) == moment % 2 else -1.0]
            decision_id = brain.decision_id
            a = brain.live(EYE[[moment % 4]], reward=reward, decision_id=decision_id)
            b = twin.live(EYE[[moment % 4]], reward=reward, decision_id=decision_id)
            np.testing.assert_array_equal(a, b)
            assert brain.last_arousal == twin.last_arousal
        _same(_durable(brain), _durable(twin))


CHECKS: dict[str, Callable[..., None]] = {
    name.removeprefix("check_"): value
    for name, value in dict(globals()).items()
    if name.startswith("check_")
}


def _run(check: Callable[..., None], tmp_path) -> None:
    if check.__code__.co_argcount:
        check(tmp_path)
    else:
        check()


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_runtime_satisfies_the_contract_check(name: str, tmp_path) -> None:
    _run(CHECKS[name], tmp_path)


# ---------------------------------------------------------------------------
# Beside the contract: the simpler controls and the calls that end a wait


def test_an_omitted_reward_is_an_outcome_where_a_wait_takes_none():
    """The released loop reports frames between an action and its outcome as zero outcomes:
    the late reward is recorded for the action chosen at the last frame. ``wait`` keeps it for
    the action that earned it."""
    zero, waited = _life(youth=40), _life(youth=40)
    first = zero.live(EYE[[0]])
    np.testing.assert_array_equal(first, waited.live(EYE[[0]]))
    late = zero.live(EYE[[1]])
    late = zero.live(EYE[[2]])
    zero.live(EYE[[3]], reward=[1.0])
    owner = waited.decision_id
    for frame in (EYE[[1]], EYE[[2]]):
        waited.wait(frame)
    waited.live(EYE[[3]], reward=[1.0], decision_id=owner)
    assert zero.basal_ganglia.updates == 3 and zero.hippocampus.writes == 3
    assert waited.basal_ganglia.updates == 1 and waited.hippocampus.writes == 1
    assert zero.hippocampus.recall(EYE[[2]])[0][int(late[0])] > 0.5
    assert zero.hippocampus.recall(EYE[[0]])[0][int(first[0])] == pytest.approx(0.0)
    assert waited.hippocampus.recall(EYE[[0]])[0][int(first[0])] > 0.5
    for cue in (1, 2):
        assert not waited.hippocampus.recall(EYE[[cue]]).any()


def test_act_replaces_an_awaited_action_and_step_takes_its_omitted_reward():
    """An outcome that never comes: ``act`` replaces the awaited action without learning,
    while ``step`` keeps its documented meaning and takes a zero outcome first."""
    for call in ("act", "step"):
        brain = _owing(youth=40)
        brain.wait(EYE[[1]])
        owner, updates = brain.decision_id, brain.basal_ganglia.updates
        writes = brain.hippocampus.writes
        getattr(brain, call)(EYE[[2]])
        learned = int(call == "step")
        assert brain.basal_ganglia.updates == updates + learned
        assert brain.hippocampus.writes == writes + learned
        assert brain._awaited() is None and brain.decision_id is None and brain.pending_feedback
        with pytest.raises(ValueError, match="decision_id"):
            brain.live(EYE[[3]], reward=[1.0], decision_id=owner)  # the late outcome
    routine = _owing(youth=0, signed=False)
    routine.wait(EYE[[1]])
    before = _durable(routine)
    routine.step(EYE[[2]])  # a routine action has no sampled eligibility to take a reward
    assert routine.basal_ganglia.updates == before["counts"][1]


def test_a_routine_action_cannot_take_its_outcome_through_learn():
    brain = _owing(youth=0, signed=False)
    brain.wait(EYE[[1]])
    sensed, before = brain._awaiting, _durable(brain)
    with pytest.raises(RuntimeError, match="non-greedy"):
        brain.learn(np.array([1.0]), np.array([False]), EYE[[2]])
    _same(before, _durable(brain))
    assert brain._awaiting is sensed and brain.pending_feedback


def test_naming_the_decision_changes_nothing_in_a_life():
    plain, named = _life(youth=20, seed=3), _life(youth=20, seed=3)
    a = plain.live(EYE[[0]])
    b = named.live(EYE[[0]])
    for moment in range(60):
        reward = [1.0 if int(a[0]) == moment % 2 else -1.0]
        x = EYE[[(moment * 3 + 1) % 4]]
        a = plain.live(x, reward=reward)
        b = named.live(x, reward=reward, decision_id=named.decision_id)
        np.testing.assert_array_equal(a, b)
        assert plain.last_arousal == named.last_arousal
    _same(_durable(plain), _durable(named))


def test_a_life_that_never_waits_keeps_its_checkpoint_format(tmp_path):
    for efference, expected in ((0.0, "cadence-generic/3"), (0.3, "cadence-generic/4")):
        brain = _owing(youth=40, efference=efference)
        with np.load(brain.save(tmp_path / f"format-{efference}.npz")) as saved:
            meta = json.loads(str(saved["generic"]))
            assert meta["format"] == expected and "awaiting" not in meta
            assert not any(name.startswith("awaiting/") for name in saved.files)
        brain.wait(EYE[[1]])
        brain.live(EYE[[2]], reward=[0.0])  # the wait ended with its outcome
        with np.load(brain.save(tmp_path / f"after-{efference}.npz")) as saved:
            assert json.loads(str(saved["generic"]))["format"] == expected


def _rewrite(path, change) -> None:
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    metadata = json.loads(str(arrays["generic"]))
    change(metadata, arrays)
    arrays["generic"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)


@pytest.mark.parametrize(
    "defect",
    [
        "older_format",
        "missing_metadata",
        "missing_state",
        "shape",
        "nonfinite",
        "steps",
        "without_arousal",
        "without_an_awaited_action",
        "stray_arrays",
    ],
)
def test_a_corrupt_waiting_checkpoint_is_refused(tmp_path, defect):
    brain = _owing(youth=0, efference=0.0, signed=False)  # a routine action awaits
    assert brain.basal_ganglia._pending is None and not brain._lived[3]
    brain.wait(EYE[[1]])
    path = brain.save(tmp_path / "waiting.npz")
    if defect == "stray_arrays":
        brain.live(EYE[[2]], reward=[0.0])
        path = brain.save(tmp_path / "not-waiting.npz")
        sensed = {name: np.zeros((1, brain.connectome.n)) for name in ("v", "activation")}

    def corrupt(metadata, arrays):
        if defect == "older_format":
            metadata["format"] = "cadence-generic/3"
        elif defect == "missing_metadata":
            del metadata["awaiting"]
        elif defect == "missing_state":
            del arrays["awaiting/adaptation"]
        elif defect == "shape":
            arrays["awaiting/v"] = arrays["awaiting/v"][:, :-1]
        elif defect == "nonfinite":
            arrays["awaiting/activation"][0, 0] = np.inf
        elif defect == "steps":
            metadata["awaiting"]["steps"] = True
        elif defect == "without_arousal":
            del metadata["arousal"], metadata["lived"]
            del arrays["lived/observations"], arrays["lived/action"]
        elif defect == "without_an_awaited_action":
            del metadata["lived"]
            del arrays["lived/observations"], arrays["lived/action"]
        else:
            for name, value in sensed.items():
                arrays["awaiting/" + name] = value

    _rewrite(path, corrupt)
    with pytest.raises(ValueError):
        cd.Brain.load(path)


def test_a_slotted_action_waits_and_is_credited_as_an_immediate_one(tmp_path):
    brain = cd.Brain.compose(
        4, 4, modules=(8,), slots=2, seed=1, working_memory_amplitude=0.3,
        arousal=cd.ArousalConfig(youth=40),
    )
    action = brain.live(EYE[[0]])
    action = brain.live(EYE[[1]], reward=[float(action[0, 0] == 0)])
    twin = _twin(brain, tmp_path / "slotted.npz")
    owner = brain.decision_id
    brain.wait(EYE[[2]])
    for one in (brain, twin):
        one.live(EYE[[3]], reward=[1.0], decision_id=owner)
    for name in ("trace", "trace_bias", "trace_critic"):
        np.testing.assert_array_equal(
            getattr(brain.basal_ganglia, name), getattr(twin.basal_ganglia, name)
        )
    np.testing.assert_array_equal(brain.hippocampus.consolidated, twin.hippocampus.consolidated)


def test_a_wait_on_the_torch_backend_credits_and_resumes_like_the_host(tmp_path):
    torch = pytest.importorskip("torch")
    brain = cd.Brain.compose(
        4, 2, modules=(8,), seed=0, working_memory_amplitude=0.3, backend="torch",
        arousal=cd.ArousalConfig(youth=40),
    )
    action = brain.live(EYE[[0]])
    for moment in range(3):
        action = brain.live(EYE[[moment + 1]], reward=[float(action[0] == moment % 2)])
    twin = _twin(brain, tmp_path / "torch-before.npz")
    owner = brain.decision_id
    brain.wait(EYE[[2]])
    resumed = cd.Brain.load(brain.save(tmp_path / "torch-waiting.npz"), backend="torch")
    for one in (brain, twin, resumed):
        one.live(EYE[[3]], reward=[1.0], decision_id=owner)
    for agent in (twin.basal_ganglia, resumed.basal_ganglia):
        np.testing.assert_array_equal(agent.trace_critic, brain.basal_ganglia.trace_critic)
    for moment in range(12):
        reward = [float(moment % 3 == 0) - 0.5]
        x = EYE[[moment % 4]]
        np.testing.assert_array_equal(
            brain.live(x, reward=reward), resumed.live(x, reward=reward)
        )
    # The resumed brain settles from host copies of states the running brain holds on its
    # device, so its running statistics can agree only to the kernel's rounding, as after
    # any torch save and load. Counts, settings and actions agree exactly: a resume that
    # dropped the sensed state shows in the learning sweeps.
    rel = 1e-4 if brain.brain._torch.dtype == torch.float32 else 1e-7
    kept, continued = brain.arousal.to_dict(), resumed.arousal.to_dict()
    assert kept.keys() == continued.keys()
    for name, value in kept.items():
        if isinstance(value, float):
            assert continued[name] == pytest.approx(value, rel=rel, abs=rel), name
        else:
            assert continued[name] == value, name


# ---------------------------------------------------------------------------
# The mutation harness


def _mutant_wait(variant: str) -> Callable[[cd.Brain, Any], None]:
    """``Brain.wait`` with one deliberate defect; ``control`` is the runtime statement."""

    def wait(self: cd.Brain, observations: Any) -> None:
        if variant == "takes_a_zero_outcome":
            self.live(observations)
            return
        if self.arousal is None:
            raise ValueError(
                "wait needs arousal genes; construct the brain with arousal=True"
            )
        x = self._observations(observations)
        current = self.basal_ganglia.state
        if len(x) != 1 or (current is not None and len(np.atleast_2d(current.v)) != 1):
            raise ValueError("wait follows one continuing stream; reset before changing streams")
        if variant != "waits_on_a_replaced_action" and (
            current is None or not self.pending_feedback
        ):
            raise RuntimeError(
                "wait needs an issued action awaiting its outcome; start with live(observations)"
            )
        drive = self.stimulus(x)
        try:
            state = self._qualified(drive, self._activity(), operation="wait")
        except RuntimeError:
            if variant != "keeps_an_unqualified_state":
                raise
            cfg = self.learner.config
            state = self._equilibrate(
                drive, self._activity(), budget=cfg.free_steps, tolerance=cfg.tolerance
            ).state
        if self.working_memory is not None and variant != "skips_the_working_trace":
            self.working_memory.update(state)
        if variant == "rewrites_the_command_copy" and self.efference is not None:
            self.efference.issue(self._command(self._choices(state)))
        if variant == "fades_eligibility":
            self.basal_ganglia.fade()
        if variant == "installs_the_sensed_state":
            # the earlier design: the sensed state becomes the action's own state
            agent = self.basal_ganglia
            agent._free, agent._drive = state, drive.copy()
            if self._lived is not None:
                self._lived = (*self._lived[:4], state, *self._lived[5:])
            current = state
        self._awaiting = (current, state)
        assert self._last_settlement is not None
        steps = int(self._last_settlement["steps"])
        if variant == "counts_a_lived_moment":
            self.arousal.lived(steps)
        elif variant == "charges_arousal_sweeps":
            self.arousal.sweeps[self.arousal.mode] += steps

    return wait


def _next_state(variant: str) -> Callable[..., cd.BrainState]:
    """``ActorCritic._next_state`` on the host, with one deliberate defect."""

    def next_state(
        self: cd.ActorCritic, drive: np.ndarray, done: np.ndarray, warm: Any = None
    ) -> cd.BrainState:
        if warm is None or variant == "from_the_action_state":
            warm = self._free
        elif variant == "keeps_finished_rows_warm":
            return self.learner.free(drive, warm=warm)
        assert warm is not None
        if done.any():
            v, a = warm.v.copy(), warm.adaptation.copy()
            v[done], a[done] = 0.0, 0.0
            warm = cd.BrainState(v, warm.activation, a, warm.steps)
        return self.learner.free(drive, warm=warm)

    return next_state


_SAVE, _LOAD = cd.Brain.save, cd.Brain.load
_LEARN, _FORECAST, _RESET = cd.Brain.learn, cd.Brain._forecast, cd.Brain.reset
_LEARN_DEVICE = cd.ActorCritic._learn_device


def _strip_awaiting(path) -> None:
    def strip(metadata, arrays):
        if metadata.pop("awaiting", None) is not None:
            metadata["format"] = "cadence-generic/4" if metadata.get("efference") else (
                "cadence-generic/3"
            )
            for name in [name for name in arrays if name.startswith("awaiting/")]:
                del arrays[name]

    _rewrite(path, strip)


def _save_without_the_sensed_state(self: cd.Brain, path):
    written = _SAVE(self, path)
    _strip_awaiting(written)
    return written


def _save_a_wait_only_with_an_identity(self: cd.Brain, path):
    written = _SAVE(self, path)
    if self.decision_id is None:
        _strip_awaiting(written)
    return written


def _load_without_the_sensed_state(cls, path, **options):
    brain = _LOAD(path, **options)
    brain._awaiting = None
    return brain


def _learn_after_ending_the_wait(self: cd.Brain, *args: Any, **kwargs: Any):
    self._awaiting = None
    return _LEARN(self, *args, **kwargs)


def _forecast_from_the_action_state(self: cd.Brain, x: np.ndarray, ended: bool):
    awaiting, self._awaiting = self._awaiting, None
    try:
        return _FORECAST(self, x, ended)
    finally:
        self._awaiting = awaiting


def _reset_with_a_new_age(self: cd.Brain) -> None:
    _RESET(self)
    if self.arousal is not None:
        self.arousal.age = 0


def _sensed_state_without_ownership(self: cd.Brain):
    return None if self._awaiting is None else self._awaiting[1]


def _device_learning_without_the_sensed_state(self, *args: Any):
    return _LEARN_DEVICE(self, *args[:-1], None)


MUTANTS: dict[str, tuple[type, str, Any]] = {
    **{
        "wait_" + variant: (cd.Brain, "wait", _mutant_wait(variant))
        for variant in (
            "takes_a_zero_outcome",
            "installs_the_sensed_state",
            "skips_the_working_trace",
            "fades_eligibility",
            "counts_a_lived_moment",
            "charges_arousal_sweeps",
            "rewrites_the_command_copy",
            "keeps_an_unqualified_state",
            "waits_on_a_replaced_action",
        )
    },
    "decision_not_checked": (cd.Brain, "_owned", lambda self, decision_id: None),
    "settles_from_the_action_state": (
        cd.Brain, "_activity", lambda self: self.basal_ganglia.state
    ),
    "sensed_state_without_ownership": (cd.Brain, "_awaited", _sensed_state_without_ownership),
    "next_state_from_the_action_state": (
        cd.ActorCritic, "_next_state", _next_state("from_the_action_state")
    ),
    "finished_rows_keep_the_sensed_state": (
        cd.ActorCritic, "_next_state", _next_state("keeps_finished_rows_warm")
    ),
    "live_ends_the_wait_before_learning": (cd.Brain, "learn", _learn_after_ending_the_wait),
    "routine_forecast_from_the_action_state": (
        cd.Brain, "_forecast", _forecast_from_the_action_state
    ),
    "reset_reuses_decision_ids": (cd.Brain, "reset", _reset_with_a_new_age),
    "checkpoint_drops_the_sensed_state": (cd.Brain, "save", _save_without_the_sensed_state),
    "checkpoint_drops_a_wait_without_identity": (
        cd.Brain, "save", _save_a_wait_only_with_an_identity
    ),
    "resume_drops_the_sensed_state": (
        cd.Brain, "load", classmethod(_load_without_the_sensed_state)
    ),
}
if TORCH:
    MUTANTS["device_learning_without_the_sensed_state"] = (
        cd.ActorCritic, "_learn_device", _device_learning_without_the_sensed_state
    )


def failing_checks(tmp_path) -> list[str]:
    """The names of the contract checks that fail under the current methods."""
    failed = []
    for index, (name, check) in enumerate(sorted(CHECKS.items())):
        folder = tmp_path / f"{index:02d}"
        folder.mkdir()
        try:
            _run(check, folder)
        except (
            AssertionError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            pytest.fail.Exception,  # an expected refusal that did not happen
        ):
            failed.append(name)
    return failed


def test_the_control_template_passes_every_check(monkeypatch, tmp_path):
    """The mutant templates with no defect are the runtime statements: no check fails."""
    monkeypatch.setattr(cd.Brain, "wait", _mutant_wait("control"))
    monkeypatch.setattr(cd.ActorCritic, "_next_state", _next_state("control"))
    assert failing_checks(tmp_path) == []


def test_every_mutant_is_killed(monkeypatch, tmp_path):
    """Each deliberate defect fails at least one contract check; the table names the killers."""
    rows = []
    for index, (name, (owner, attribute, function)) in enumerate(MUTANTS.items()):
        folder = tmp_path / f"m{index:02d}"
        folder.mkdir()
        with monkeypatch.context() as patched:
            patched.setattr(owner, attribute, function)
            rows.append((name, failing_checks(folder)))
    width = max(len(name) for name, _ in rows)
    print(f"\n{'mutant':<{width}}  killed by")
    for name, killers in rows:
        print(f"{name:<{width}}  {', '.join(killers) or 'SURVIVED'}")
    survivors = [name for name, killers in rows if not killers]
    print(f"mutation score {len(rows) - len(survivors)}/{len(rows)}")
    assert not survivors, survivors
