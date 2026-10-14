"""The beat paid by the world: one continuing ``Brain.live`` life learns to alternate two
actions under identical drive from the world's reward alone.

Every moment the walker sees the same drive and pays nothing. A step that differs from
its last step earns one unit, paid at the next moment as the outcome of the preceding
action; a repeated step earns nothing. No teacher labels anything. Nothing in the
observation tells the walker what it did; the only state that can carry the beat is
what it carries itself. This is the reward half of the continuing-action chamber of
issue 116 (roadmap row 06 of issue 109); the supervised half is ``steady_rhythm.py``.

Arms on the same founder weights and the same world: ``live`` (the declared System 1
with the efference copy, routine while outcomes match, aroused and learning when they
do not), ``nocopy`` (the same brain without the copy), ``defaults`` (the copy on the
composed reward defaults instead of the declared operating point), ``step`` (always
learning, every moment sampled), ``lambda_control`` (the eligibility decay at the
protocol's control value, zero unless ``lam_control`` is declared), ``frozen`` (no
learning: the founder's greedy choice, which shows whether a founder had the beat from
birth), ``tabular`` (Q-learning given the one bit the copy carries, the last action),
``random`` (uniform). Mid-life the live brain is
saved with its last action awaiting its outcome; a loaded twin continues beside it.
Forks of the mid-life checkpoint meet a pause of one, two or four silent events, or one
distractor event, and keep living under the same rule.

Run: ``python benchmarks/rhythm/reward_rhythm.py --out DIR``; verify with ``--verify DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rhythm_inputs as inputs  # noqa: E402

import cadence as cd  # noqa: E402
from cadence import Brain, Receipt  # noqa: E402

SCHEMA = "steady-rhythm-reward/1"
INSTRUMENT_REVISION = 2
PROTOCOL_PATH = Path(__file__).with_name("protocol-reward.json")
ARMS = ("live", "nocopy", "defaults", "step", "lambda_control", "frozen", "tabular", "random")
BRAIN_ARMS = ("live", "nocopy", "defaults", "step", "lambda_control", "frozen")
REQUIRED = ("schema", "world", "life", "operating_point", "arousal", "tabular", "seeds", "gates")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    missing = [key for key in REQUIRED if key not in protocol]
    if missing:
        raise ValueError(f"protocol lacks {missing}")
    if protocol["schema"] not in {f"steady-rhythm-reward/{i}" for i in (1, 2, 3)}:
        raise ValueError("unsupported reward protocol schema")
    if set(protocol["seeds"]["development"]) & set(protocol["seeds"]["confirmation"]):
        raise ValueError("confirmation seeds must be fresh")
    for seeds in protocol["seeds"].values():
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("protocol seeds must be nonempty, unique and nonnegative")
    life = protocol["life"]
    if (
        any(
            life[key] <= 0
            for key in ("moments", "block", "window", "probe_at", "greedy_probe", "continuation")
        )
        or life["probe_at"] + max(life["window"], life["continuation"]) > life["moments"]
    ):
        raise ValueError("life must include the complete scored window and continuation")
    return protocol, hashlib.sha256(raw).hexdigest()


# -- the world


def observation(kind: int) -> np.ndarray:
    """One stream's observation: the frozen input module's event for row 0."""
    return inputs.observation(kind)[:1]


def pay(previous: int | None, action: int, rule: dict) -> float:
    """The world's rule: a step that differs from the last step earns ``alternate``, a
    repeated step ``repeat``; the first step of a life earns ``repeat``."""
    if previous is None:
        return float(rule["repeat"])
    return float(rule["alternate"] if action != previous else rule["repeat"])


# -- brains


def make_brain(seed: int, protocol: dict, arm: str) -> Brain:
    point = protocol["operating_point"]
    options: dict[str, Any] = {
        "episodic": False,
        "working_memory_amplitude": point["trace_amplitude"],
        "working_memory_decay": point["trace_decay"],
        # the composition of releases before 0.82.0 read the trace back at scale 12
        "working_memory_scale": point.get("trace_scale", 12.0),
    }
    if arm != "nocopy":
        options["efference_amplitude"] = point["efference_amplitude"]
        options["efference_decay"] = point["efference_decay"]
    if arm not in ("step", "frozen"):
        genes = {k: v for k, v in protocol["arousal"].items() if k != "note"}
        options["arousal"] = cd.ArousalConfig(**genes)
    brain = Brain.compose(
        inputs.INPUTS, inputs.ACTIONS, modules=tuple(point["modules"]), seed=seed, **options
    )
    if arm != "defaults":
        actor = {k: point[k] for k in ("eta", "lam", "gamma", "eta_critic") if k in point}
        if "eta" in actor:
            actor["eta_bias"] = actor["eta"] / 10.0  # the composed rule of issue 143
        if arm == "lambda_control":
            actor["lam"] = float(point.get("lam_control", 0.0))
        brain.basal_ganglia.config = replace(brain.basal_ganglia.config, **actor)
    return brain


class Work:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "routine": 0,
            "aroused": 0,
            "sweeps_routine": 0,
            "sweeps_aroused": 0,
            "learning_sweeps": 0,
            "refusals": 0,
            "refused_sweeps": 0,
            "checkpoints": 0,
            "probe_acts": 0,
            "probe_sweeps": 0,
        }
        self.seconds = 0.0


def moment(brain: Brain, arm: str, x: np.ndarray, reward: float | None, work: Work) -> dict:
    """One moment of a brain arm: the executed action, its mode, the base policy's belief in
    alternating (the probability of the action opposite to the last executed one) and the
    work. A refused answer is a missed step: the world counts it as a repeat, while
    ``retry_reward`` retains only feedback still owed to a previously issued action."""
    start = time.perf_counter()
    agent = brain.basal_ganglia
    previous_settlement = brain.last_settlement
    sampled_before = agent._pending is not None
    try:
        if arm == "frozen":
            action = int(brain.act(x, greedy=True)[0])
            settlement = brain.last_settlement
            assert settlement is not None
            work.counts["routine"] += 1
            work.counts["sweeps_routine"] += int(settlement["steps"])
            aroused, temperature = False, None
        elif arm == "step":
            feedback = {} if reward is None else {"reward": [reward], "done": [False]}
            action = int(brain.step(x, **feedback)[0])
            settlement = brain.last_settlement
            assert settlement is not None
            work.counts["aroused"] += 1
            work.counts["sweeps_aroused"] += int(settlement["steps"])
            pending = agent._pending
            if pending is not None:
                work.counts["learning_sweeps"] += int(pending[1].steps) + int(pending[2].steps)
            work.counts["learning_sweeps"] += int(brain.last_learning.get("free_steps", 0))
            aroused, temperature = True, brain.learner.config.temperature
        else:
            feedback = {} if reward is None else {"reward": [reward], "done": [False]}
            action = int(brain.live(x, **feedback)[0])
            reading = brain.last_arousal
            assert reading is not None
            aroused = reading["mode"] == "aroused"
            mode = "aroused" if aroused else "routine"
            work.counts[mode] += 1
            work.counts["sweeps_" + mode] += int(reading["sweeps"])
            work.counts["learning_sweeps"] += int(reading["learning_sweeps"])
            temperature = reading["temperature"]
    except RuntimeError:
        settlement = brain.last_settlement
        if (
            settlement is not None
            and settlement is not previous_settlement
            and not settlement["qualified"]
        ):
            work.counts["refusals"] += 1
            work.counts["refused_sweeps"] += int(settlement["steps"])
            lived = brain._lived
            pending = agent._pending is not None or (
                arm not in ("step", "frozen")
                and lived is not None
                and not lived[3]
                and lived[4] is agent.state
            )
            if sampled_before and not pending:
                # Feedback may have succeeded before the following answer refused.
                work.counts["learning_sweeps"] += int(brain.last_learning.get("free_steps", 0))
            return {
                "action": None,
                "aroused": None,
                "belief": None,
                "retry_reward": reward if pending else None,
            }
        raise
    finally:
        work.seconds += time.perf_counter() - start
    state = agent.state
    assert state is not None
    policy = np.asarray(agent.probabilities(state))[0]
    return {
        "action": action,
        "aroused": aroused,
        "policy": policy.tolist(),
        "temperature": temperature,
    }


class Tabular:
    """Q-learning with epsilon-greedy choice over the one bit the copy carries: the last
    action (or none). It sees the same reward stream under the same rule."""

    def __init__(self, seed: int, config: dict) -> None:
        self.q = np.zeros((3, inputs.ACTIONS))
        self.rng = np.random.default_rng(seed)
        self.alpha, self.epsilon, self.gamma = config["alpha"], config["epsilon"], config["gamma"]
        self.state = 2

    def act(self, reward: float | None) -> int:
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(0, inputs.ACTIONS))
        else:
            action = int(np.argmax(self.q[self.state]))
        self._pending = (self.state, action)
        return action

    def outcome(self, reward: float) -> None:
        s, a = self._pending
        s2 = a
        self.q[s, a] += self.alpha * (reward + self.gamma * self.q[s2].max() - self.q[s, a])
        self.state = s2


# -- scoring


def alternation(actions: list[int | None]) -> float:
    pairs = [
        float(a is not None and b is not None and a != b)
        for a, b in zip(actions, actions[1:], strict=False)
    ]
    return float(np.mean(pairs)) if pairs else 0.0


def blocks(values: list, size: int) -> list[float]:
    out = []
    for start in range(0, len(values), size):
        chunk = [v for v in values[start : start + size] if v is not None]
        out.append(float(np.mean(chunk)) if chunk else 0.0)
    return out


def beliefs(records: list[dict], actions: list[int | None]) -> list[float | None]:
    """The base policy's probability, at each moment, of the action opposite to the last
    executed one: its belief in the beat, read from the living brain."""
    out: list[float | None] = []
    previous: int | None = None
    for record, action in zip(records, actions, strict=True):
        policy = record.get("policy")
        if previous is None or policy is None:
            out.append(None)
        else:
            out.append(float(policy[1 - previous]))
        if action is not None:
            previous = action
    return out


# -- stages


def live_life(
    brain: Brain | None,
    arm: str,
    protocol: dict,
    seed: int,
    directory: Path,
    *,
    moments: int,
    probe_at: int,
) -> dict:
    """One life of ``moments`` drive events. At ``probe_at`` the brain is saved with its
    last action awaiting its outcome; a twin loaded from that probe lives beside the
    original for ``continuation`` moments under the same outcomes, and the probe also
    feeds the greedy probe and the disturbance forks."""
    rule = protocol["world"]["reward"]
    continuation = protocol["life"]["continuation"]
    work = Work()
    twin_work = Work()
    rng = np.random.default_rng(seed)
    table = Tabular(seed, protocol["tabular"]) if arm == "tabular" else None
    x = observation(inputs.KIND_DRIVE)
    actions: list[int | None] = []
    records: list[dict] = []
    rewards: list[float] = []
    feedback: list[float | None] = []
    previous: int | None = None
    reward: float | None = None
    probe: Path | None = None
    probe_boundary: dict | None = None
    twin: Brain | None = None
    twin_actions: list[int | None] = []
    twin_previous: int | None = None
    twin_reward: float | None = None
    custody: dict | None = None
    for t in range(moments):
        if t == probe_at and brain is not None:
            probe = brain.save(directory / "probe.npz")
            probe_boundary = {"previous_action": previous, "pending_reward": reward}
            twin = Brain.load(probe)
            work.counts["checkpoints"] += 2
            twin_previous, twin_reward = previous, reward
        if twin is not None:
            # the twin meets the same world state the original meets at this moment
            record_twin = moment(twin, arm, x, twin_reward, twin_work)
            b = record_twin["action"]
            twin_reward = (
                pay(twin_previous, b, rule) if b is not None else record_twin["retry_reward"]
            )
            twin_actions.append(b)
            if b is not None:
                twin_previous = b
        feedback.append(reward)
        control_started = None
        if arm == "random":
            control_started = time.perf_counter()
            action: int | None = int(rng.integers(0, inputs.ACTIONS))
            record = {"action": action, "aroused": None}
        elif arm == "tabular":
            control_started = time.perf_counter()
            assert table is not None
            action = table.act(reward)
            record = {"action": action, "aroused": None}
        else:
            assert brain is not None
            record = moment(brain, arm, x, reward, work)
            action = record["action"]
        earned = pay(previous, action, rule) if action is not None else float(rule["repeat"])
        # A missed step has no issued action to credit. Its world score cannot replace
        # an earlier action's pending outcome, nor become feedback for no action.
        reward = earned if action is not None else record["retry_reward"]
        if table is not None:
            table.outcome(earned)
        if control_started is not None:
            work.seconds += time.perf_counter() - control_started
        actions.append(action)
        records.append(record)
        rewards.append(earned)
        if action is not None:
            previous = action
        if twin is not None and len(twin_actions) == continuation:
            assert brain is not None
            with tempfile.TemporaryDirectory() as temporary:
                first = brain.save(Path(temporary) / "own.npz")
                second = twin.save(Path(temporary) / "twin.npz")
                work.counts["checkpoints"] += 2
                with (
                    np.load(first, allow_pickle=False) as aa,
                    np.load(second, allow_pickle=False) as bb,
                ):
                    same = set(aa.files) == set(bb.files) and all(
                        np.array_equal(aa[key], bb[key]) for key in aa.files
                    )
            custody = {
                "actions_equal": actions[probe_at:] == twin_actions,
                "saved_arrays_equal": bool(same),
                "alternation": alternation(twin_actions),
                "twin_work": {**twin_work.counts, "calls_seconds": twin_work.seconds},
            }
            twin = None
    block = protocol["life"]["block"]
    window = protocol["life"]["window"]
    aroused = [r["aroused"] for r in records]
    belief = beliefs(records, actions)
    result = {
        "arm": arm,
        "seed": seed,
        "actions": actions,
        "rewards": rewards,
        "feedback": feedback,
        "aroused": aroused,
        "belief": belief,
        "alternation": alternation(actions),
        "alternation_blocks": blocks(
            [
                float(a is not None and b is not None and a != b)
                for a, b in zip(actions, actions[1:], strict=False)
            ],
            block,
        ),
        "aroused_blocks": blocks(aroused, block) if arm in BRAIN_ARMS else None,
        "income_blocks": blocks(rewards, block),
        "belief_blocks": blocks(belief, block) if arm in BRAIN_ARMS else None,
        "window_alternation": alternation(actions[-window:]),
        "window_aroused": float(np.mean([a for a in aroused[-window:] if a is not None]))
        if arm in BRAIN_ARMS and any(a is not None for a in aroused[-window:])
        else None,
        "window_income": float(np.mean(rewards[-window:])),
        "refusals": work.counts["refusals"],
        "work": {**work.counts, "calls_seconds": work.seconds},
        "probe": None if probe is None else probe.name,
        "probe_boundary": probe_boundary,
        "final_boundary": {"previous_action": previous, "pending_reward": reward},
        "continuation": custody,
    }
    return result


def greedy_probe(probe: Path, protocol: dict, work: Work) -> dict:
    """The acquired policy without exploration: a loaded copy acts greedily."""
    copy = Brain.load(probe)
    work.counts["checkpoints"] += 1
    x = observation(inputs.KIND_DRIVE)
    actions: list[int | None] = []
    for _ in range(protocol["life"]["greedy_probe"]):
        start = time.perf_counter()
        previous_settlement = copy.last_settlement
        try:
            actions.append(int(copy.act(x, greedy=True)[0]))
        except RuntimeError:
            settlement = copy.last_settlement
            if settlement is None or settlement is previous_settlement or settlement["qualified"]:
                raise
            work.counts["refusals"] += 1
            actions.append(None)
        finally:
            work.seconds += time.perf_counter() - start
        settlement = copy.last_settlement
        assert settlement is not None
        work.counts["probe_acts"] += 1
        work.counts["probe_sweeps"] += int(settlement["steps"])
    return {"actions": actions, "alternation": alternation(actions)}


def disturb(
    probe: Path, arm: str, name: str, protocol: dict, work: Work, *, boundary: dict
) -> dict:
    """A fork of the probe lives through ``pre`` drive events, the disturbance (``k`` silent
    events or one distractor), then ``post`` drive events, paid under the same rule
    throughout; recovery is alternation over the post events."""
    spec = protocol["life"]["disturbances"]
    fork = Brain.load(probe)
    work.counts["checkpoints"] += 1
    rule = protocol["world"]["reward"]
    kinds = [inputs.KIND_DRIVE] * spec["pre"]
    if name.startswith("pause"):
        kinds += [inputs.KIND_PAUSE] * int(name[len("pause") :])
    else:
        kinds += [inputs.KIND_DISTRACTOR]
    kinds += [inputs.KIND_DRIVE] * spec["post"]
    actions: list[int | None] = []
    rewards: list[float] = []
    feedback: list[float | None] = []
    previous: int | None = boundary["previous_action"]
    reward: float | None = boundary["pending_reward"]
    for kind in kinds:
        feedback.append(reward)
        record = moment(fork, arm, observation(kind), reward, work)
        action = record["action"]
        earned = pay(previous, action, rule) if action is not None else float(rule["repeat"])
        reward = earned if action is not None else record["retry_reward"]
        actions.append(action)
        rewards.append(earned)
        if action is not None:
            previous = action
    post = actions[-spec["post"] :]
    return {
        "actions": actions,
        "rewards": rewards,
        "feedback": feedback,
        "pre_alternation": alternation(actions[: spec["pre"]]),
        "post_alternation": alternation(post),
        "recovered": alternation(post) == 1.0,
    }


def run_founder(seed: int, arm: str, protocol: dict, directory: Path, forks: bool) -> dict:
    directory.mkdir(parents=True)
    life = protocol["life"]
    brain = make_brain(seed, protocol, arm) if arm in BRAIN_ARMS else None
    if brain is not None:
        brain.save(directory / "initial.npz")
    result = live_life(
        brain, arm, protocol, seed, directory, moments=life["moments"], probe_at=life["probe_at"]
    )
    if brain is not None:
        probe = directory / "probe.npz"
        result["work"]["checkpoints"] += 1  # the initial save above
        work = Work()
        result["greedy_probe"] = greedy_probe(probe, protocol, work)
        if forks:
            result["disturbances"] = {
                name: disturb(probe, arm, name, protocol, work, boundary=result["probe_boundary"])
                for name in [f"pause{k}" for k in life["disturbances"]["pauses"]] + ["distractor"]
            }
        result["probe_work"] = {**work.counts, "calls_seconds": work.seconds}
        brain.save(directory / "final.npz")
        result["work"]["checkpoints"] += 1
    return result


def _mean(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return float(np.mean(present)) if present else None


def aggregate(runs: list[dict], protocol: dict, *, admissible: bool = True) -> dict:
    out: dict = {}
    gates = protocol["gates"]
    expected = {(seed, arm) for seed in protocol["seeds"]["confirmation"] for arm in ARMS}
    observed = [(run["seed"], run["arm"]) for run in runs]
    admitted = admissible and len(observed) == len(expected) and set(observed) == expected
    learned_gate = protocol["schema"] != "steady-rhythm-reward/1"
    for arm in sorted({run["arm"] for run in runs}):
        selected = [run for run in runs if run["arm"] == arm]
        founders = len(selected)
        entry: dict = {
            "founders": founders,
            "window_alternation": _mean([r["window_alternation"] for r in selected]),
            "window_aroused": _mean([r["window_aroused"] for r in selected]),
            "window_income": _mean([r["window_income"] for r in selected]),
            "life_alternation": _mean([r["alternation"] for r in selected]),
            "alternation_blocks": [
                _mean([r["alternation_blocks"][i] for r in selected])
                for i in range(len(selected[0]["alternation_blocks"]))
            ],
            "aroused_blocks": None
            if selected[0]["aroused_blocks"] is None
            else [
                _mean([r["aroused_blocks"][i] for r in selected])
                for i in range(len(selected[0]["aroused_blocks"]))
            ],
            "refusals": int(sum(r["refusals"] for r in selected)),
        }
        if arm in BRAIN_ARMS:
            entry["greedy_probe_alternation"] = _mean(
                [r["greedy_probe"]["alternation"] for r in selected]
            )
            entry["continuation_equal"] = int(
                sum(
                    r["continuation"]["actions_equal"] and r["continuation"]["saved_arrays_equal"]
                    for r in selected
                )
            )
            if "disturbances" in selected[0]:
                entry["disturbances"] = {
                    name: {
                        "pre_alternation": _mean(
                            [r["disturbances"][name]["pre_alternation"] for r in selected]
                        ),
                        "post_alternation": _mean(
                            [r["disturbances"][name]["post_alternation"] for r in selected]
                        ),
                        "recovered": int(
                            sum(r["disturbances"][name]["recovered"] for r in selected)
                        ),
                    }
                    for name in selected[0]["disturbances"]
                }
            frozen = {
                run["seed"]: run["window_alternation"] for run in runs if run["arm"] == "frozen"
            }
            entry["gates"] = {
                "acquired": int(
                    sum(r["window_alternation"] >= gates["window_alternation"] for r in selected)
                ),
                # learned: the founder alternates after learning and did not from birth
                "learned": int(
                    sum(
                        r["window_alternation"] >= gates["window_alternation"]
                        and r["seed"] in frozen
                        and frozen[r["seed"]] < gates["window_alternation"]
                        for r in selected
                    )
                ),
                "greedy": int(
                    sum(
                        r["greedy_probe"]["alternation"] >= gates["greedy_alternation"]
                        for r in selected
                    )
                ),
                "calm": int(
                    sum(
                        r["window_aroused"] is not None
                        and r["window_aroused"] <= gates["window_aroused"]
                        for r in selected
                    )
                ),
                "continued": entry["continuation_equal"],
                "required": gates["share"],
                "expected_founders": len(protocol["seeds"]["confirmation"]),
                "admissible": admitted,
            }
            # The same founders must satisfy every clause of the frozen statement.
            # Missing controls and unfinished/custom censuses cannot earn a pass.
            entry["gates"]["jointly_qualified"] = sum(
                r["window_alternation"] >= gates["window_alternation"]
                and (
                    not learned_gate
                    or (r["seed"] in frozen and frozen[r["seed"]] < gates["window_alternation"])
                )
                and r["greedy_probe"]["alternation"] >= gates["greedy_alternation"]
                and r["window_aroused"] is not None
                and r["window_aroused"] <= gates["window_aroused"]
                and r["continuation"]["actions_equal"]
                and r["continuation"]["saved_arrays_equal"]
                for r in selected
            )
            entry["gates"]["passed"] = (
                admitted and entry["gates"]["jointly_qualified"] >= gates["share"]
            )
        out[arm] = entry
    return out


# -- custody


def _check_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs from the recorded events or declaration")


def _audit_events(
    run: dict, length: int, rule: dict, *, previous: int | None = None, pending: float | None = None
) -> None:
    actions, rewards, feedback = run["actions"], run["rewards"], run["feedback"]
    if len(actions) != length or len(rewards) != length or len(feedback) != length:
        raise ValueError("event census differs from the protocol")
    for index, action in enumerate(actions):
        if action not in (None, 0, 1):
            raise ValueError("invalid recorded action")
        earned = pay(previous, action, rule) if action is not None else float(rule["repeat"])
        _check_equal(rewards[index], earned, "reward")
        # A refusal may consume feedback or leave it pending. It may never invent
        # another outcome; the live refusal tests establish which branch applies.
        if index == 0:
            _check_equal(feedback[index], pending, "initial feedback")
        elif actions[index - 1] is not None:
            _check_equal(feedback[index], rewards[index - 1], "feedback")
        elif feedback[index] not in (None, feedback[index - 1]):
            raise ValueError("refused event invented feedback")
        if action is not None:
            previous = action


def _audit_run(run: dict, protocol: dict, forks: bool) -> None:
    life, rule = protocol["life"], protocol["world"]["reward"]
    _audit_events(run, life["moments"], rule)
    actions, rewards = run["actions"], run["rewards"]
    aroused, belief = run["aroused"], run["belief"]
    if len(aroused) != len(actions) or len(belief) != len(actions):
        raise ValueError("diagnostic census differs from the events")
    brain = run["arm"] in BRAIN_ARMS
    window, block = life["window"], life["block"]
    expected = {
        "alternation": alternation(actions),
        "alternation_blocks": blocks(
            [
                float(a is not None and b is not None and a != b)
                for a, b in zip(actions, actions[1:], strict=False)
            ],
            block,
        ),
        "aroused_blocks": blocks(aroused, block) if brain else None,
        "income_blocks": blocks(rewards, block),
        "belief_blocks": blocks(belief, block) if brain else None,
        "window_alternation": alternation(actions[-window:]),
        "window_aroused": _mean(aroused[-window:]) if brain else None,
        "window_income": float(np.mean(rewards[-window:])),
        "refusals": actions.count(None),
    }
    for key, value in expected.items():
        _check_equal(run[key], value, key)
    if not brain:
        return
    _check_equal(run["work"]["refusals"], actions.count(None), "refused work")
    _check_equal(run["work"]["routine"], aroused.count(False), "routine work")
    _check_equal(run["work"]["aroused"], aroused.count(True), "aroused work")
    probe_at = life["probe_at"]
    previous = next((a for a in reversed(actions[:probe_at]) if a is not None), None)
    boundary = {"previous_action": previous, "pending_reward": run["feedback"][probe_at]}
    _check_equal(run["probe_boundary"], boundary, "probe boundary")
    final = run["final_boundary"]
    _check_equal(
        final["previous_action"],
        next((a for a in reversed(actions) if a is not None), None),
        "final world boundary",
    )
    if actions[-1] is not None:
        _check_equal(final["pending_reward"], rewards[-1], "final pending reward")
    elif final["pending_reward"] not in (None, run["feedback"][-1]):
        raise ValueError("final refusal invented feedback")
    greedy = run["greedy_probe"]
    if len(greedy["actions"]) != life["greedy_probe"]:
        raise ValueError("greedy probe census differs")
    _check_equal(greedy["alternation"], alternation(greedy["actions"]), "greedy alternation")
    custody = run["continuation"]
    if not all(type(custody[key]) is bool for key in ("actions_equal", "saved_arrays_equal")):
        raise ValueError("continuation claims must be boolean")
    if custody["actions_equal"]:
        _check_equal(
            custody["alternation"],
            alternation(actions[probe_at : probe_at + life["continuation"]]),
            "continuation alternation",
        )
    spec = life["disturbances"]
    names = [f"pause{k}" for k in spec["pauses"]] + ["distractor"] if forks else []
    _check_equal(sorted(run.get("disturbances", {})), sorted(names), "disturbance census")
    for name in names:
        disturbed = run["disturbances"][name]
        length = spec["pre"] + spec["post"] + (int(name[5:]) if name.startswith("pause") else 1)
        _audit_events(
            disturbed, length, rule, previous=previous, pending=boundary["pending_reward"]
        )
        post = alternation(disturbed["actions"][-spec["post"] :])
        _check_equal(
            disturbed["pre_alternation"],
            alternation(disturbed["actions"][: spec["pre"]]),
            "disturbance pre alternation",
        )
        _check_equal(disturbed["post_alternation"], post, "disturbance post alternation")
        _check_equal(disturbed["recovered"], post == 1.0, "disturbance recovery")


def _audit_body(body: dict, frozen: dict, declaration: dict) -> None:
    """Check revision-2 arithmetic and admission without rerunning the brain.

    Continuation equality remains a source-bound runtime observation, not a fact
    derivable from the compressed receipt alone.
    """
    _check_equal(body["declaration"], declaration, "declaration copy")
    _check_equal(declaration["instrument_revision"], INSTRUMENT_REVISION, "instrument revision")
    overrides = declaration["overrides"]
    if set(overrides) - {"moments", "forks", "seeds", "arms"}:
        raise ValueError("unknown protocol override")
    effective = json.loads(json.dumps(frozen))
    if "moments" in overrides:
        effective["life"]["moments"] = overrides["moments"]
    _check_equal(body["protocol"], effective, "effective protocol")
    seeds = overrides.get("seeds", frozen["seeds"]["confirmation"])
    arms = overrides.get("arms", list(ARMS))
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(seed < 0 for seed in seeds)
        or not arms
        or len(set(arms)) != len(arms)
        or any(arm not in ARMS for arm in arms)
    ):
        raise ValueError("invalid declared census")
    _check_equal(declaration["seeds"], seeds, "declared seeds")
    _check_equal(declaration["arms"], arms, "declared arms")
    _check_equal(declaration["frozen_protocol"], not overrides, "frozen declaration")
    _check_equal(body["frozen_protocol"], not overrides, "frozen protocol")
    _check_equal(declaration["protocol_sha256"], body["protocol_sha256"], "declared protocol hash")
    planned = [[seed, arm] for seed in seeds for arm in arms]
    completed = [[run["seed"], run["arm"]] for run in body["runs"]]
    _check_equal(body["planned_founders"], planned, "planned census")
    _check_equal(body["completed_founders"], completed, "completed census")
    _check_equal(completed, planned[: len(completed)], "run census")
    _check_equal(body["capped"], len(completed) < len(planned), "capped census")
    for run in body["runs"]:
        _audit_run(run, effective, overrides.get("forks", True))
    expected = aggregate(body["runs"], effective, admissible=not overrides and not body["capped"])
    _check_equal(body["aggregate"], expected, "aggregate and gates")


def verify(directory: Path) -> tuple[bool, str]:
    try:
        path = directory / "summary.json"
        raw = json.loads(path.read_text())
        sources = [(item["path"], directory / item["path"]) for item in raw["source"]["files"]]

        def check(body: dict) -> str | None:
            for name, expected in body["artifacts"].items():
                if sha256(directory / name) != expected:
                    return f"artifact differs: {name}"
            if body["protocol_sha256"] != sha256(directory / "protocol.json"):
                return "protocol copy differs from the recorded hash"
            revision = body["declaration"].get("instrument_revision")
            if revision is not None:
                if revision != INSTRUMENT_REVISION:
                    return "unsupported instrument revision"
                frozen, _ = load_protocol(directory / "protocol.json")
                _audit_body(body, frozen, json.loads((directory / "declaration.json").read_text()))
            return None

        valid, reason = Receipt.verify(path, sources=sources, check=check)
        if not valid:
            return False, reason
        legacy = raw["body"]["declaration"].get("instrument_revision") is None
        return True, (
            "legacy receipt: canonical form, digest, sources and artifact custody agree; "
            "historical scores and gates retained, not validated by the corrected instrument"
            if legacy
            else "canonical form, digest, sources, artifact custody, event arithmetic, "
            "census and gates agree"
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return False, f"cannot verify artifact: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out", type=Path)
    destination.add_argument("--verify", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--moments", type=int, default=None)
    parser.add_argument("--no-forks", action="store_true")
    parser.add_argument("--time-cap", type=float, default=None)
    args = parser.parse_args(argv)
    if args.verify is not None:
        valid, reason = verify(args.verify)
        print(json.dumps({"valid": valid, "reason": reason}))
        return 0 if valid else 1
    protocol, protocol_sha = load_protocol(args.protocol)
    overrides: dict[str, Any] = {}
    if args.moments is not None:
        if args.moments <= protocol["life"]["probe_at"] + max(
            protocol["life"]["window"], protocol["life"]["continuation"]
        ):
            parser.error("moments must leave the probe before the scored window")
        protocol["life"]["moments"] = args.moments
        overrides["moments"] = args.moments
    if args.no_forks:
        overrides["forks"] = False
    seeds = list(protocol["seeds"]["confirmation"]) if args.seeds is None else list(args.seeds)
    if (
        len(seeds) != len(set(seeds))
        or any(seed < 0 for seed in seeds)
        or any(arm not in ARMS for arm in args.arms)
        or len(args.arms) != len(set(args.arms))
    ):
        parser.error("invalid seeds or arms")
    if args.time_cap is not None and (not np.isfinite(args.time_cap) or args.time_cap <= 0):
        parser.error("time-cap must be finite and positive")
    if seeds != protocol["seeds"]["confirmation"]:
        overrides["seeds"] = seeds
    if list(args.arms) != list(ARMS):
        overrides["arms"] = list(args.arms)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_bytes(Path(args.protocol).read_bytes())
    began = time.perf_counter()
    declaration = {
        "schema": SCHEMA,
        "instrument_revision": INSTRUMENT_REVISION,
        "evidence_scope": "Instrument revision 2; reruns on previously used confirmation "
        "seeds are audits, not fresh confirmation.",
        "protocol_sha256": protocol_sha,
        "frozen_protocol": not overrides,
        "overrides": overrides,
        "seeds": seeds,
        "arms": list(args.arms),
        "python": sys.version,
        "platform": platform.platform(),
        "cadence_version": cd.__version__,
        "cadence_import": str(Path(cd.__file__).resolve()),
        "cpu_count": os.cpu_count(),
        "time_cap": args.time_cap,
    }
    (args.out / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    sources, origins = [], []
    package = Path(cd.__file__).resolve().parent
    own = [Path(__file__).resolve(), Path(inputs.__file__).resolve()]
    for source in [*own, *sorted(package.rglob("*.py"))]:
        relative = (
            "source/" + source.name
            if source in own
            else "source/cadence/" + source.relative_to(package).as_posix()
        )
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources.append((relative, target))
        origins.append((source, target))
    runs: list[dict] = []
    capped = False
    planned = [(seed, arm) for seed in seeds for arm in args.arms]
    for seed, arm in planned:
        if args.time_cap is not None and time.perf_counter() - began > args.time_cap:
            capped = True
            break
        result = run_founder(seed, arm, protocol, args.out / f"seed{seed}-{arm}", not args.no_forks)
        runs.append(result)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "arm": arm,
                    "window_alternation": round(result["window_alternation"], 3),
                    "greedy": None
                    if "greedy_probe" not in result
                    else round(result["greedy_probe"]["alternation"], 3),
                    "window_aroused": None
                    if result["window_aroused"] is None
                    else round(result["window_aroused"], 3),
                    "seconds": round(time.perf_counter() - began, 1),
                }
            ),
            flush=True,
        )
    assert all(sha256(original) == sha256(copy) for original, copy in origins), (
        "source changed during the run; retain this incomplete attempt and rerun"
    )
    artifacts = {
        path.relative_to(args.out).as_posix(): sha256(path)
        for path in sorted(args.out.rglob("*"))
        if path.is_file()
        and "source" not in path.relative_to(args.out).parts
        and path.name != "summary.json"
    }
    summary = {
        "declaration": declaration,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "frozen_protocol": not overrides,
        "runs": runs,
        "planned_founders": [list(item) for item in planned],
        "completed_founders": [[run["seed"], run["arm"]] for run in runs],
        "capped": capped,
        "seconds": time.perf_counter() - began,
        "aggregate": aggregate(runs, protocol, admissible=not overrides and not capped),
        "artifacts": artifacts,
        "work_scope": "Every moment of every arm, every learning phase, probe, twin, "
        "disturbance fork and checkpoint is charged; a refused answer is a missed step. "
        "calls_seconds exclude checkpoint IO; run seconds include IO. Sweeps are not "
        "joules.",
    }
    Receipt.build(SCHEMA, summary, sources).write(args.out / "summary.json")
    valid, reason = verify(args.out)
    assert valid, reason
    print(
        json.dumps({"verified": valid, "capped": capped, "summary": str(args.out / "summary.json")})
    )
    return 3 if capped else 0


if __name__ == "__main__":
    raise SystemExit(main())
