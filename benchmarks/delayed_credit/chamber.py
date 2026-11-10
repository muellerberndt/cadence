"""The delayed-credit chamber (cadence#283): lives of the brain and its controls.

One life is one learner on one world (``tasks.py``) at one delay, every episode's outcome
delivered after its last act and nothing paid before it. The brain lives through ``step``:
the outcome arrives with the next episode's first observation together with ``done``, so the
library clears the carried context between episodes; every outcome is learned from. The
conventional learners (``baselines.py``) live the same episodes in the same order.

Kinds: ``brain`` (``Brain.compose(inputs, 2, seed=founder, arousal=True)`` with the arm's
settings), ``lstm`` and ``tdlambda`` (sized to the brain's parameter count), ``random``
(uniform actions) and ``solver`` (the exact solver). Controls on any learner: ``yoked`` (the
outcome a uniform-random partner earned on the same episode, whatever the learner did),
``shuffled`` (one of the learner's own earlier outcomes, drawn uniformly), ``frozen`` (the
newborn read at every budget, nothing learned).

Readings at every budget, on a copy that never changes the life: each panel episode from a
fresh stream, every act greedy. ``choice``: the share of panel episodes whose decision (the
T-maze's choice, the key-door's reach) is right; ``door``: the share of key-door episodes whose
door act is right for the pouch it shows. The training log per window of ``WINDOW`` episodes:
the share of decisions right as acted, the policy's probability of the right decision when it
was taken, and for the brain the credit diagnostics (``diagnose``).

    python benchmarks/delayed_credit/chamber.py --kinds brain --arms default \\
        --tasks tmaze keydoor --delays 10 50 --founders 0 1 2 3 4 5 --out DIR/lives.jsonl
    python benchmarks/delayed_credit/chamber.py --summary DIR/lives.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np


def _sibling(name: str) -> Any:
    """A module beside this file, loaded by path under a name of its own."""
    import importlib.util

    key = f"delayed_credit_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


tk = _sibling("tasks")
baselines = _sibling("baselines")

KINDS = ("brain", "lstm", "tdlambda", "random", "solver")
CONTROLS = ("none", "yoked", "shuffled", "frozen")
BUDGETS = (256, 1024, 2048)
WINDOW = 128
FOUNDERS = tuple(range(6))
GATE = {"choice": 0.9, "founders": 5, "within": 0.05}
SLOW = {"actor_eta": 0.1, "actor_eta_bias": 0.01}
UNDECAYED = {"actor_gamma": 1.0, "actor_lam": 1.0}
ARMS: dict[str, dict[str, Any]] = {
    "default": {},
    "recall_off": {"recall": False},
    "slow_episodic": {**SLOW, "memory_choice": True},
    "undecayed": dict(UNDECAYED),
    "slow_undecayed": {**SLOW, **UNDECAYED},
    "slow_episodic_undecayed": {**SLOW, "memory_choice": True, **UNDECAYED},
    "critic_raised": {"actor_eta_critic": 5.0},
    # the actor's eligibility held through the episode, the critic's decay unchanged (#283)
    "held": {"actor_hold": 1.0},
    "recall_off_held": {"recall": False, "actor_hold": 1.0},
    "slow_held": {**SLOW, "actor_hold": 1.0},
    "slow_episodic_held": {**SLOW, "memory_choice": True, "actor_hold": 1.0},
}


# ----------------------------------------------------------------------------------- brain


def make_brain(task: str, arm: str, founder: int, settings: dict[str, Any] | None = None) -> Any:
    import cadence as cd

    options = {"backend": "cpu", **ARMS[arm], **(settings or {})}
    return cd.Brain.compose(tk.INPUTS[task], tk.ACTIONS, seed=int(founder), arousal=True,
                            **options)


def brain_parameters(task: str, arm: str = "default",
                     settings: dict[str, Any] | None = None) -> int:
    """The brain's parameter count on a world: the size the conventional learners match."""
    warnings.simplefilter("ignore")
    return int(make_brain(task, arm, 0, settings).parameters())


def _probability(brain: Any, action: int) -> float:
    """The readout's probability of ``action`` in the free phase of the stream's last act."""
    bg = brain.basal_ganglia
    return float(np.asarray(bg.probabilities(bg.state))[0, action])


def _read_brain(brain: Any, task: str, founder: int, delay: int) -> dict[str, Any]:
    fork = copy.deepcopy(brain)
    choice, door, outcome, p_right = [], [], [], []
    for cue, corridor in tk.panel(task, founder, delay):
        fork.reset()
        episode = tk.Episode(task, founder, delay, cue, corridor)
        while not episode.done:
            action = int(fork.act(episode.observation()[None, :], greedy=True)[0])
            if episode.moment == 0:
                p_right.append(_probability(fork, tk.right_action(cue, founder)))
            episode.act(action)
            if task == "tmaze":
                break
        choice.append(episode.right)
        if task == "keydoor":
            door.append(episode.door == (tk.open_action(founder) if episode.key
                                         else 1 - tk.open_action(founder)))
            outcome.append(episode.reward)
    return _reading(choice, door, outcome, p_right)


def _reading(choice: list, door: list, outcome: list, p_right: list) -> dict[str, Any]:
    return {
        "choice": float(np.mean(choice)),
        "door": float(np.mean(door)) if door else None,
        "outcome": float(np.mean(outcome)) if outcome else None,
        "p_right": float(np.mean(p_right)) if p_right else None,
    }


def _brain_work(brain: Any, synapses: int, eligibility: int) -> float:
    """The analytic multiply-accumulates of the last ``step``: every settle sweep multiplies
    each synapse once (the act's free phase, the next state's settle and the two nudged
    phases of the eligibility), and the trace and the step touch each synapse three times."""
    settled = brain.last_settlement or {}
    learned = brain.last_learning or {}
    sweeps = float(settled.get("steps", 0)) + float(learned.get("free_steps", 0.0))
    if learned:
        sweeps += 2 * eligibility
    return synapses * (sweeps + (3 if learned else 0))


# ------------------------------------------------------------------------------- controls


class Delivery:
    """The outcome a control delivers in place of the learner's own."""

    def __init__(self, control: str, task: str, founder: int, delay: int) -> None:
        self.control, self.task, self.founder, self.delay = control, task, founder, delay
        self.own: list[float] = []
        index = tk.TASKS.index(task)
        self.rng = np.random.default_rng(
            np.random.SeedSequence([tk.SEEDS.get(control, 0), founder, delay, index])
        )

    def __call__(self, episode: tk.Episode) -> float:
        self.own.append(episode.reward)
        if self.control == "yoked":
            partner = tk.Episode(self.task, self.founder, self.delay, episode.cue,
                                 episode.corridor)
            while not partner.done:
                partner.act(int(self.rng.integers(tk.ACTIONS)))
            return partner.reward
        if self.control == "shuffled":
            return self.own[int(self.rng.integers(len(self.own)))]
        return episode.reward


# ----------------------------------------------------------------------------------- lives


def _windows(values: list[Any]) -> list[float | None]:
    out: list[float | None] = []
    for start in range(0, len(values), WINDOW):
        chunk = [v for v in values[start:start + WINDOW] if v is not None]
        out.append(float(np.mean(chunk)) if chunk else None)
    return out


def _split(values: list[Any], right: list[bool]) -> dict[str, list[float | None]]:
    """A per-episode diagnostic's windows over right and wrong decisions apart."""
    return {
        "right": _windows([v if r else None for v, r in zip(values, right, strict=False)]),
        "wrong": _windows([v if not r else None for v, r in zip(values, right, strict=False)]),
    }


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``kind``, ``task``, ``delay``, ``founder``; ``arm`` and ``settings`` for the
    brain, ``learner`` settings for a conventional learner, ``control``, ``budgets``."""
    warnings.simplefilter("ignore")
    kind, task = spec["kind"], spec["task"]
    delay, founder = int(spec["delay"]), int(spec["founder"])
    control = spec.get("control", "none")
    budgets = sorted(int(b) for b in spec.get("budgets", BUDGETS))
    if kind not in KINDS or control not in CONTROLS:
        raise ValueError("unknown kind or control")
    arm = spec.get("arm", "default")
    driver = spec.get("driver", "step")
    if driver not in ("step", "live"):
        raise ValueError("the brain lives through step or live")
    started, cpu = time.monotonic(), time.process_time()
    row: dict[str, Any] = {
        "kind": kind, "task": task, "delay": delay, "founder": founder, "control": control,
        "arm": arm if kind == "brain" else spec.get("tag"), "budgets": budgets,
        "driver": driver if kind == "brain" else None,
    }
    inputs = tk.INPUTS[task]
    learner: Any = None
    brain: Any = None
    synapses = eligibility = 0
    if kind == "brain":
        import cadence as cd

        row["cadence"] = cd.__version__
        row["settings"] = {**ARMS[arm], **spec.get("settings", {})}
        brain = make_brain(task, arm, founder, spec.get("settings"))
        row["genes"] = {k: v for k, v in brain.describe()["genes"].items()
                        if k.startswith(("actor_", "memory_", "recall"))}
        row["parameters"] = int(brain.parameters())
        synapses = int(brain.describe()["layout"]["synapses"])
        eligibility = int(brain.basal_ganglia.config.eligibility_steps
                          or brain.learner.config.nudged_steps)
    elif kind in ("lstm", "tdlambda"):
        target = int(spec.get("match", brain_parameters(task)))
        hidden = baselines.matched_hidden(target, inputs, tk.ACTIONS, kind)
        cls = baselines.LSTMActorCritic if kind == "lstm" else baselines.TDLambda
        learner = cls(inputs, tk.ACTIONS, hidden, founder, **spec.get("learner", {}))
        row["learner"] = learner.describe()
        row["parameters"] = learner.parameter_count
        row["matched_to"] = target
    rng = np.random.default_rng(np.random.SeedSequence([founder, delay, 283]))
    deliver = Delivery(control, task, founder, delay)
    work = {"moments": 0, "episodes": 0, "multiply_accumulates": 0.0}

    def read() -> dict[str, Any]:
        if kind == "brain":
            return _read_brain(brain, task, founder, delay)
        return _read_other(kind, learner, task, founder, delay)

    log: dict[str, list[Any]] = {k: [] for k in (
        "right", "reward", "p_right", "value_decision", "dopamine_decision", "value_after",
        "dopamine_outcome", "value_last", "share", "cosine", "trace", "aroused_decision",
        "aroused_corridor")}
    readings: list[dict[str, Any]] = []
    decay = 1.0
    if kind == "brain":
        decay = float(brain.basal_ganglia.config.decay)  # the actor's, with a hold its own
    episodes = [] if control == "frozen" else tk.schedule(task, founder, delay, budgets[-1])
    if control == "frozen":  # the newborn, read at every budget
        newborn = read()
        readings = [{"episodes": b, **newborn} for b in budgets]
    owed: float | None = None
    for index, (cue, corridor) in enumerate(episodes):
        if index in budgets:
            readings.append({"episodes": index, **read()})
        episode = tk.Episode(task, founder, delay, cue, corridor)
        right_action = tk.right_action(cue, founder)
        if kind == "lstm":
            learner.begin()
        first_trace = None
        while not episode.done:
            x = episode.observation()
            moment = episode.moment
            if kind == "brain":
                move = brain.live if driver == "live" else brain.step
                if moment == 0 and owed is not None:
                    action = int(move(x[None, :], reward=[owed], done=[True])[0])
                    log["dopamine_outcome"].append(brain.last_learning.get("dopamine"))
                    log["value_last"].append(brain.last_learning.get("value"))
                elif moment == 0:
                    action = int(move(x[None, :])[0])
                else:
                    action = int(move(x[None, :], reward=[0.0], done=[False])[0])
                if driver == "live":
                    aroused = (brain.last_arousal or {}).get("mode") == "aroused"
                    log["aroused_decision" if moment == 0 else "aroused_corridor"].append(
                        float(aroused))
                work["multiply_accumulates"] += _brain_work(brain, synapses, eligibility)
                if moment == 0:
                    log["p_right"].append(_probability(brain, right_action))
                trace = brain.basal_ganglia.trace
                if moment == 1:
                    log["value_decision"].append(brain.last_learning.get("value"))
                    log["dopamine_decision"].append(brain.last_learning.get("dopamine"))
                    first_trace = None if trace is None else np.array(trace[0], copy=True)
                if moment == 2:  # the value of the moment after the decision
                    log["value_after"].append(brain.last_learning.get("value"))
                if moment == episode.moments - 1 and first_trace is not None and trace is not None:
                    last = np.asarray(trace[0])
                    norm = float(np.linalg.norm(last))
                    part = decay ** (moment - 1) * float(np.linalg.norm(first_trace))
                    log["share"].append(part / norm if norm > 0 else None)
                    denominator = norm * float(np.linalg.norm(first_trace))
                    log["cosine"].append(float(last @ first_trace) / denominator
                                         if denominator > 0 else None)
                    log["trace"].append(norm)
            elif kind == "lstm":
                if moment == 0:
                    log["p_right"].append(learner.probability(x, right_action))
                action = learner.act(x)
            elif kind == "tdlambda":
                action = learner.act(x, first=moment == 0)
                if moment == 0:
                    log["p_right"].append(float(learner.learner.probs[right_action]))
            elif kind == "solver":
                action = tk.solve(task, founder, x)
            else:
                action = int(rng.integers(tk.ACTIONS))
            episode.act(action)
            work["moments"] += 1
        delivered = deliver(episode)
        if kind == "lstm":
            learner.finish(delivered)
        elif kind == "tdlambda":
            learner.finish(delivered)
        owed = delivered
        work["episodes"] += 1
        log["right"].append(bool(episode.right))
        log["reward"].append(float(episode.reward))
    if episodes and len(episodes) in budgets:
        readings.append({"episodes": len(episodes), **read()})
    if learner is not None:
        work["multiply_accumulates"] = float(learner.work["multiply_accumulates"])
    row["readings"] = readings
    row["final"] = readings[-1] if readings else None
    tail = log["right"][-256:]
    row["acted_right_last256"] = float(np.mean(tail)) if tail else None
    row["training"] = {
        "right": _windows(log["right"]),
        "reward": _windows(log["reward"]),
        "p_right": _windows(log["p_right"]),
    }
    if kind == "brain" and log["right"]:
        right = log["right"]
        row["diagnostics"] = {
            "decay_per_moment": decay,
            "decay_at_outcome": decay ** delay,
            "value_decision": _split(log["value_decision"], right),
            "dopamine_decision": _split(log["dopamine_decision"], right),
            "value_after": _split(log["value_after"], right),
            # an outcome is logged with the next episode's first step
            "dopamine_outcome": _split(log["dopamine_outcome"],
                                       right[:len(log["dopamine_outcome"])]),
            "value_last": _split(log["value_last"], right[:len(log["value_last"])]),
            "share": _windows(log["share"]),
            "cosine": _windows(log["cosine"]),
            "trace": _windows(log["trace"]),
            "aroused_decision": _windows(log["aroused_decision"]),
            "aroused_corridor": _windows(log["aroused_corridor"]),
        }
    row["work"] = {**work, "seconds": time.monotonic() - started,
                   "cpu_seconds": time.process_time() - cpu}
    if work["episodes"]:
        row["work"]["per_episode"] = {
            "seconds": row["work"]["seconds"] / work["episodes"],
            "multiply_accumulates": work["multiply_accumulates"] / work["episodes"],
        }
    return row


def _read_other(kind: str, learner: Any, task: str, founder: int,
                delay: int) -> dict[str, Any]:
    choice, door, outcome, p_right = [], [], [], []
    rng = np.random.default_rng(np.random.SeedSequence([founder, delay, 2830]))
    for cue, corridor in tk.panel(task, founder, delay):
        episode = tk.Episode(task, founder, delay, cue, corridor)
        if kind in ("lstm", "tdlambda"):
            def respond(action: int, episode: tk.Episode = episode) -> np.ndarray | None:
                episode.act(action)
                if episode.done or task == "tmaze":
                    return None
                return episode.observation()

            p_right.append(learner.probability(episode.observation(),
                                               tk.right_action(cue, founder)))
            learner.greedy([episode.observation()], respond)
        else:
            while not episode.done:
                x = episode.observation()
                episode.act(tk.solve(task, founder, x) if kind == "solver"
                            else int(rng.integers(tk.ACTIONS)))
        choice.append(episode.right)
        if task == "keydoor":
            door.append(episode.door == (tk.open_action(founder) if episode.key
                                         else 1 - tk.open_action(founder)))
            outcome.append(episode.reward)
    return _reading(choice, door, outcome, p_right)


# ------------------------------------------------------------------------------------- run


def _one(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_life(spec)
    except Exception as error:  # a refused life is recorded as a failure, never dropped
        return {**spec, "error": f"{type(error).__name__}: {error}"}


def specs_of(args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for kind in args.kinds:
        arms = args.arms if kind == "brain" else [None]
        for arm in arms:
            for task in args.tasks:
                for delay in args.delays:
                    for founder in args.founders:
                        spec: dict[str, Any] = {
                            "kind": kind, "task": task, "delay": delay, "founder": founder,
                            "control": args.control, "budgets": args.budgets,
                        }
                        if arm is not None:
                            spec["arm"] = arm
                        if args.settings:
                            spec["settings"] = json.loads(args.settings)
                        if args.learner:
                            spec["learner"] = json.loads(args.learner)
                        if args.tag:
                            spec["tag"] = args.tag
                        if args.driver != "step":
                            spec["driver"] = args.driver
                        out.append(spec)
    return out


def summary(path: str) -> str:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    cells: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        arm = row.get("arm")
        if row.get("driver") == "live":
            arm = f"{arm} (live)"
        key = (row["kind"], arm, row.get("control", "none"), row["task"], row["delay"])
        cells.setdefault(key, []).append(row)
    lines = ["| kind | arm | control | task | delay | parameters | founders at .90 | choice "
             "| p(right) | acted, last 256 | door | seconds per episode | MAC per episode |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key in sorted(cells, key=str):
        group = cells[key]
        ok = [r for r in group if "error" not in r]
        finals = [r["final"] for r in ok if r.get("final")]
        passing = sum(f["choice"] >= GATE["choice"] for f in finals)

        def mean(values: list[Any]) -> str:
            values = [v for v in values if v is not None]
            return f"{np.mean(values):.2f}" if values else "-"

        per = [r["work"].get("per_episode", {}) for r in ok]
        lines.append(
            f"| {key[0]} | {key[1] or '-'} | {key[2]} | {key[3]} | {key[4]} "
            f"| {mean([r.get('parameters') for r in ok])} "
            f"| {passing} of {len(group)} | {mean([f['choice'] for f in finals])} "
            f"| {mean([f['p_right'] for f in finals])} "
            f"| {mean([r.get('acted_right_last256') for r in ok])} "
            f"| {mean([f['door'] for f in finals])} "
            f"| {mean([p.get('seconds') for p in per])} "
            f"| {mean([p.get('multiply_accumulates') for p in per])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kinds", nargs="+", default=["brain"], choices=KINDS)
    parser.add_argument("--arms", nargs="+", default=["default"], choices=sorted(ARMS))
    parser.add_argument("--tasks", nargs="+", default=list(tk.TASKS), choices=tk.TASKS)
    parser.add_argument("--delays", nargs="+", type=int, default=list(tk.DELAYS))
    parser.add_argument("--founders", nargs="+", type=int, default=list(FOUNDERS))
    parser.add_argument("--control", default="none", choices=CONTROLS)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--settings", help="JSON settings over the brain arm's")
    parser.add_argument("--learner", help="JSON settings of a conventional learner")
    parser.add_argument("--tag", help="a conventional learner's name for its settings in the table")
    parser.add_argument("--driver", default="step", choices=("step", "live"),
                        help="the brain's moments through step (every outcome) or live")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out")
    parser.add_argument("--summary")
    args = parser.parse_args(argv)
    if args.summary:
        print(summary(args.summary))
        return 0
    if not args.out:
        parser.error("--out is required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    specs = specs_of(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        if args.workers <= 1:
            results = map(_one, specs)
            for row in results:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
        else:
            with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
                for row in pool.imap_unordered(_one, specs):
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
