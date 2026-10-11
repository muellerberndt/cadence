"""The prerequisite-chain nursery: can one continuing ``Brain.compose`` life learn a chain of
actions whose pay comes at the end, and what makes it collapse onto one action?

A chain of ``steps`` actions must be taken in order (gather wood, gather stone, make fire,
warm up); two further actions do nothing. The chain's state, how many prerequisites the
creature holds, is in the observation (``observed``) or not (``hidden``). Pay comes at the
completion only (``end``), at every correct step and the completion (``shaped``), or at the
completion of episodes that begin at a random stage with the prerequisites granted
(``lessons``: the backward lessons, interleaved). The arms are the arena founder brain at
its gene step, at a tenth of it, at the gene step tapered to a tenth at mid-life, and with
the readout's intrinsic plasticity; the controls are uniform random, the stationary
memoryless ceiling (uniform over the chain's actions), a frozen newborn and a tabular
learner with the same information. Readings come from saved copies every
``probe_every`` moments: completions per thousand moments by the greedy and by the sampled
policy from stage zero, the greedy action and the right action's probability at every
stage, the share of the life's own moments spent on its most chosen action, the aroused
share, the learned moments and the sign of their dopamine. A development instrument of
issues 111, 113 and 84 (``docs/sequential-tasks.md``); it establishes no default and
declares no gate.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

import cadence as cd  # noqa: E402
from cadence.receipts import Receipt, canonical_json  # noqa: E402

SCHEMA = "prerequisite-chain/1"
PROTOCOL = Path(__file__).with_name("protocol.json")
INPUTS = 16
DISTRACTORS = 2
MODES = ("observed", "hidden")
PAYS = ("end", "shaped", "lessons")
ARMS = ("gene", "small", "taper", "homeo")
CONTROLS = ("random", "stationary", "frozen", "tabular")


# ----------------------------------------------------------------------------------- world


def make_world(seed: int, steps: int) -> dict[str, Any]:
    """The chain of one seed: which actions form it, in which order, and the background."""
    if not 2 <= steps <= INPUTS - 4:
        raise ValueError("steps lies in [2, 12]")
    rng = np.random.default_rng([seed, 7])
    order = rng.permutation(steps + DISTRACTORS)
    background = np.sort(rng.choice(np.arange(steps + 1, INPUTS), 3, replace=False))
    return {
        "seed": seed,
        "steps": steps,
        "actions": steps + DISTRACTORS,
        "chain": [int(a) for a in order[:steps]],  # the right action at stage k
        "distractors": [int(a) for a in order[steps:]],
        "background": [int(b) for b in background],
    }


def observe(world: dict[str, Any], stage: int, mode: str) -> np.ndarray:
    """One observation: the cue, the background, and the stage when it is observed."""
    x = np.zeros((1, INPUTS))
    x[0, world["steps"]] = 1.0  # the need's cue, always on
    x[0, world["background"]] = 1.0
    if mode == "observed":
        x[0, stage] = 1.0
    return x


def start_stage(pay: str, rng: np.random.Generator, steps: int) -> int:
    return int(rng.integers(steps)) if pay == "lessons" else 0


def outcome(
    world: dict[str, Any], stage: int, action: int, pay: str, rng: np.random.Generator
) -> tuple[float, int, bool, bool]:
    """What an action at a stage does: (reward, next stage, right, completed)."""
    steps = world["steps"]
    right = action == world["chain"][stage]
    if not right:
        return 0.0, stage, False, False
    if stage + 1 == steps:
        return 1.0, start_stage(pay, rng, steps), True, True
    return (1.0 / steps if pay == "shaped" else 0.0), stage + 1, True, False


def competent_income(steps: int) -> float:
    """What a life that always takes the right action earns per moment under ``end`` pay."""
    return 1.0 / steps


# ----------------------------------------------------------------------------------- brain


def make_brain(point: dict[str, Any], seed: int, world: dict[str, Any], need: float) -> cd.Brain:
    """The arena's founder brain on this world's ports; ``point["learning"]`` names
    ``LearnerConfig`` fields to override, the readout's intrinsic plasticity among them."""
    learning = {
        f"learning_{name}": value for name, value in dict(point.get("learning", {})).items()
    }
    arousal = dict(point["arousal"])
    arousal["need"] = need
    return cd.Brain.compose(
        INPUTS,
        world["actions"],
        modules=tuple(int(w) for w in point["modules"]),
        sensory_scale=float(point["sensory_scale"]),
        temperature=float(point["temperature"]),
        working_memory_amplitude=float(point["trace_amplitude"]),
        working_memory_decay=float(point["trace_decay"]),
        consolidation=float(point["consolidation"]),
        episodic=bool(point.get("episodic", True)),
        memory_rate=float(point.get("memory_rate", 1.0)),
        actor_eta=float(point["eta"]),
        actor_eta_bias=float(point["eta_bias"]),
        actor_lam=float(point["lam"]),
        actor_gamma=float(point["gamma"]),
        actor_eta_critic=float(point["eta_critic"]),
        arousal=arousal,
        seed=seed,
        **learning,
    )


def arm_point(point: dict[str, Any], arm: str, stages: dict[str, Any]) -> dict[str, Any]:
    """The brain point an arm is born with (``taper`` is born at the gene step)."""
    own = json.loads(json.dumps(point))
    genes = dict(stages.get(arm, {}))
    if arm == "small":
        own["eta"], own["eta_bias"] = float(genes["actor_eta"]), float(genes["actor_eta_bias"])
    if arm == "homeo":
        own.setdefault("learning", {}).update(
            {k[len("learning_") :]: v for k, v in genes.items() if k.startswith("learning_")}
        )
    return own


# --------------------------------------------------------------------------------- readings


def saved_copy(brain: cd.Brain, directory: Path) -> cd.Brain:
    """A copy of the brain through its own checkpoint, its stream state reset."""
    path = directory / f"probe-{os.getpid()}.npz"
    brain.save(path)
    copy = cd.Brain.load(path)
    copy.reset()
    return copy


def run_policy(
    world: dict[str, Any],
    mode: str,
    moments: int,
    choose: Any,
    seed: int,
) -> dict[str, Any]:
    """Run a policy from stage zero under ``end`` pay without learning; ``choose(x, stage)``
    returns the action. Returns completions, the right-action share and the per-stage tallies."""
    steps = world["steps"]
    rng = np.random.default_rng([seed, 11])
    stage, completions, right_count = 0, 0, 0
    actions = np.zeros((steps, world["actions"]), dtype=int)
    for _ in range(moments):
        x = observe(world, stage, mode)
        action = int(choose(x, stage))
        actions[stage, action] += 1
        _, stage, right, completed = outcome(world, stage, action, "end", rng)
        completions += int(completed)
        right_count += int(right)
    most = int(actions.sum(axis=0).max())
    return {
        "completions_per_1000": 1000.0 * completions / moments,
        "right_share": right_count / moments,
        "one_action_share": most / moments,
        "stage_actions": actions.tolist(),
    }


def readings(
    brain: cd.Brain,
    world: dict[str, Any],
    mode: str,
    temperature: float,
    moments: int,
    seed: int,
    directory: Path,
) -> dict[str, Any]:
    """From saved copies: the greedy and the sampled policy run from stage zero, and the
    greedy answer and the right action's probability at every stage (observed mode)."""
    greedy = saved_copy(brain, directory)
    out: dict[str, Any] = {
        "greedy": run_policy(world, mode, moments, lambda x, s: greedy.act(x, greedy=True)[0], seed)
    }
    sampler = saved_copy(brain, directory)
    rng = np.random.default_rng([seed, 13])

    def sample(x: np.ndarray, stage: int) -> int:
        sampler.act(x, greedy=True)
        state = sampler.basal_ganglia.state
        probs = np.asarray(sampler.basal_ganglia.probabilities(state, temperature))[0]
        return int(rng.choice(len(probs), p=probs / probs.sum()))

    out["sampled"] = run_policy(world, mode, moments, sample, seed)
    probe = saved_copy(brain, directory)
    per_stage = []
    for stage in range(world["steps"]):
        x = observe(world, stage, mode)
        action = int(probe.act(x, greedy=True)[0])
        state = probe.basal_ganglia.state
        probs = np.asarray(probe.basal_ganglia.probabilities(state, temperature))[0]
        right = world["chain"][stage]
        per_stage.append(
            {
                "greedy": action,
                "right": right,
                "correct": action == right,
                "p_right": float(probs[right]),
                "margin": float(probs[right] - np.delete(probs, right).max()),
            }
        )
        probe.reset()
    out["stages"] = per_stage
    out["greedy_stage_accuracy"] = float(np.mean([s["correct"] for s in per_stage]))
    out["p_right_mean"] = float(np.mean([s["p_right"] for s in per_stage]))
    return out


class Ledger:
    """What a stretch of a life cost and what the brain learned from it."""

    def __init__(self, actions: int) -> None:
        self.moments = 0
        self.completions = 0
        self.right = 0
        self.aroused = 0
        self.learned = 0
        self.reward = 0.0
        self.dopamine = 0.0
        self.positive = 0
        self.actions = np.zeros(actions)

    def moment(self, brain: cd.Brain, action: int, reward: float, right: bool, done: bool) -> None:
        reading = brain.last_arousal
        assert reading is not None
        self.moments += 1
        self.completions += int(done)
        self.right += int(right)
        self.reward += reward
        self.actions[action] += 1
        self.aroused += int(reading["mode"] == "aroused")
        if reading["learned"]:
            report = brain.last_learning
            self.learned += 1
            d = float(report["dopamine"])
            self.dopamine += d
            self.positive += int(d > 0)

    def summary(self) -> dict[str, Any]:
        n = max(self.moments, 1)
        k = max(self.learned, 1)
        return {
            "moments": self.moments,
            "completions_per_1000": 1000.0 * self.completions / n,
            "right_share": self.right / n,
            "income": self.reward / n,
            "aroused_share": self.aroused / n,
            "learned": self.learned,
            "mean_dopamine": self.dopamine / k,
            "positive_share": self.positive / k,
            "one_action_share": float(self.actions.max() / n),
            "actions": self.actions.tolist(),
        }


# ------------------------------------------------------------------------------------- life


def run_life(
    arm: str, mode: str, pay: str, seed: int, protocol: dict[str, Any], checkpoints: Path
) -> dict[str, Any]:
    """One brain arm living ``moments`` in one mode under one pay, probed every
    ``probe_every`` moments from saved copies."""
    started = time.perf_counter()
    steps = int(protocol["steps"])
    world = make_world(seed, steps)
    moments = int(protocol["moments"])
    probe_every = int(protocol["probe_every"])
    probe_moments = int(protocol["probe_moments"])
    temperature = float(protocol["brain"]["temperature"])
    need = float(protocol["need_share"]) * competent_income(steps)
    point = arm_point(protocol["brain"], arm, protocol["stages"])
    brain = make_brain(point, seed, world, need)
    rng = np.random.default_rng([seed, 3])
    stage = start_stage(pay, rng, steps)
    reward: float | None = None
    totals = Ledger(world["actions"])
    probes = [
        {
            "moment": 0,
            "readings": readings(brain, world, mode, temperature, probe_moments, seed, checkpoints),
            "window": Ledger(world["actions"]).summary(),
        }
    ]
    tapered = False
    for start in range(0, moments, probe_every):
        stop = min(start + probe_every, moments)
        if arm == "taper" and not tapered and start >= moments // 2:
            genes = dict(protocol["stages"]["taper"])
            brain.retune(**genes)
            tapered = True
        window = Ledger(world["actions"])
        for _ in range(start, stop):
            x = observe(world, stage, mode)
            fed = None if reward is None else np.array([reward])
            action = int(brain.live(x, reward=fed)[0])
            reward, stage, right, done = outcome(world, stage, action, pay, rng)
            window.moment(brain, action, reward, right, done)
        for name in ("moments", "completions", "right", "aroused", "learned", "positive"):
            setattr(totals, name, getattr(totals, name) + getattr(window, name))
        totals.reward += window.reward
        totals.dopamine += window.dopamine
        totals.actions = totals.actions + window.actions
        probes.append(
            {
                "moment": stop,
                "readings": readings(
                    brain, world, mode, temperature, probe_moments, seed, checkpoints
                ),
                "window": window.summary(),
            }
        )
    return {
        "kind": "brain",
        "arm": arm,
        "mode": mode,
        "pay": pay,
        "seed": seed,
        "world": world,
        "need": need,
        "describe": {k: brain.describe()[k] for k in ("actor", "arousal", "learning")},
        "probes": probes,
        "totals": totals.summary(),
        "final": probes[-1]["readings"],
        "seconds": time.perf_counter() - started,
    }


def run_control(
    control: str, mode: str, pay: str, seed: int, protocol: dict[str, Any], checkpoints: Path
) -> dict[str, Any]:
    """A control life: uniform random, the stationary ceiling, a frozen newborn or a tabular
    learner over the observation it gets."""
    started = time.perf_counter()
    steps = int(protocol["steps"])
    world = make_world(seed, steps)
    moments = int(protocol["moments"])
    probe_moments = int(protocol["probe_moments"])
    rng = np.random.default_rng([seed, 5])
    actions = world["actions"]
    row: dict[str, Any] = {
        "kind": "control",
        "arm": control,
        "mode": mode,
        "pay": pay,
        "seed": seed,
        "world": world,
    }
    if control == "random":
        choose = lambda x, s: rng.integers(actions)  # noqa: E731
        row["final"] = {"greedy": run_policy(world, mode, probe_moments, choose, seed)}
    elif control == "stationary":
        chain = world["chain"]
        choose = lambda x, s: chain[rng.integers(steps)]  # noqa: E731
        row["final"] = {"greedy": run_policy(world, mode, probe_moments, choose, seed)}
    elif control == "frozen":
        need = float(protocol["need_share"]) * competent_income(steps)
        brain = make_brain(protocol["brain"], seed, world, need)
        row["final"] = readings(
            brain,
            world,
            mode,
            float(protocol["brain"]["temperature"]),
            probe_moments,
            seed,
            checkpoints,
        )
    elif control == "tabular":
        t = protocol["tabular"]
        alpha, epsilon, gamma = float(t["alpha"]), float(t["epsilon"]), float(t["gamma"])
        states = steps if mode == "observed" else 1
        q = np.zeros((states, actions))

        def key(stage: int) -> int:
            return stage if mode == "observed" else 0

        stage = start_stage(pay, rng, steps)
        for _ in range(moments):
            s = key(stage)
            a = int(rng.integers(actions)) if rng.random() < epsilon else int(np.argmax(q[s]))
            reward, next_stage, _, done = outcome(world, stage, a, pay, rng)
            target = reward + (0.0 if done else gamma * q[key(next_stage)].max())
            q[s, a] += alpha * (target - q[s, a])
            stage = next_stage
        greedy = lambda x, s: int(np.argmax(q[key(s)]))  # noqa: E731

        def sample(x: np.ndarray, s: int) -> int:
            return (
                int(rng.integers(actions)) if rng.random() < epsilon else int(np.argmax(q[key(s)]))
            )

        row["final"] = {
            "greedy": run_policy(world, mode, probe_moments, greedy, seed),
            "sampled": run_policy(world, mode, probe_moments, sample, seed),
            "q": q.tolist(),
        }
    else:
        raise ValueError(f"unknown control {control}")
    row["seconds"] = time.perf_counter() - started
    return row


# -------------------------------------------------------------------------------------- run


def _life_job(args: tuple[str, str, str, int, dict[str, Any], str]) -> dict[str, Any]:
    arm, mode, pay, seed, protocol, checkpoints = args
    if arm in CONTROLS:
        return run_control(arm, mode, pay, seed, protocol, Path(checkpoints))
    return run_life(arm, mode, pay, seed, protocol, Path(checkpoints))


def run(
    protocol: dict[str, Any],
    arms: list[str],
    modes: list[str],
    pays: list[str],
    seeds: list[int],
    workers: int,
    checkpoints: Path,
    log: Any = None,
) -> dict[str, Any]:
    checkpoints.mkdir(parents=True, exist_ok=True)
    jobs = [
        (arm, mode, pay, seed, protocol, str(checkpoints))
        for seed in seeds
        for mode in modes
        for pay in pays
        for arm in arms
        # a control without learning does not depend on the pay
        if not (arm in ("random", "stationary", "frozen") and pay != pays[0])
    ]
    if workers <= 1 or len(jobs) <= 1:
        rows = []
        for job in jobs:
            rows.append(_life_job(job))
            if log is not None:
                print(".", end="", file=log, flush=True)
    else:
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            rows = []
            for row in pool.imap_unordered(_life_job, jobs):
                rows.append(row)
                if log is not None:
                    print(".", end="", file=log, flush=True)
    if log is not None:
        print(file=log)
    rows.sort(key=lambda r: (r["mode"], r["pay"], r["kind"], r["arm"], r["seed"]))
    return {"rows": rows}


# --------------------------------------------------------------------------------- receipts


def sources() -> list[tuple[str, Path]]:
    package = Path(cd.__file__).resolve().parent
    return [
        ("prerequisite_chain.py", Path(__file__).resolve()),
        *(
            ("cadence/" + path.relative_to(package).as_posix(), path)
            for path in sorted(package.rglob("*.py"))
        ),
    ]


def read_receipt(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    stored: dict[str, Any] = json.loads(data)
    return stored


def write_receipt(path: Path, body: dict[str, Any], manifest: dict[str, Any]) -> None:
    receipt = Receipt.build(SCHEMA, body, sources())
    if receipt.source != manifest:
        raise RuntimeError("a source file changed during the run; discard it and run again")
    data = (canonical_json(receipt.to_dict()) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def verify(path: Path, *, current: bool = False) -> tuple[bool, str]:
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    with tempfile.TemporaryDirectory() as directory:
        plain = Path(directory) / "receipt.json"
        plain.write_bytes(data)
        stored = json.loads(data)
        if stored.get("kind") != SCHEMA:
            return False, f"not a {SCHEMA} receipt"
        return Receipt.verify(plain, sources=sources() if current else None)


# ----------------------------------------------------------------------------------- report


def _mean(values: list[Any], fmt: str = "{:.2f}") -> str:
    present = [float(v) for v in values if v is not None]
    return fmt.format(float(np.mean(present))) if present else "-"


def markdown(body: dict[str, Any]) -> str:
    rows = body["rows"]
    steps = body["protocol"]["steps"]
    lines = [
        f"# {SCHEMA}: a chain of {steps} actions, {len(body['seeds'])} seeds, {len(rows)} lives",
        "",
        "Mean over seeds. Completions per thousand moments of the greedy (and the sampled)"
        " policy run from stage zero under pay at the end; the greedy answer's accuracy over the"
        " stages and the right action's probability (observed mode); the life's share of"
        " moments on its most chosen action, aroused share, learned moments and the share of"
        " them with positive dopamine.",
        "",
        "| mode | pay | arm | completions greedy / sampled | stage accuracy | p(right)"
        " | one action | aroused | learned | dopamine positive |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["mode"], r["pay"], r["arm"]), []).append(r)
    for (mode, pay, arm), group in sorted(groups.items()):
        g = _mean([r["final"]["greedy"]["completions_per_1000"] for r in group], "{:.0f}")
        s = _mean(
            [
                r["final"]["sampled"]["completions_per_1000"]
                for r in group
                if "sampled" in r["final"]
            ],
            "{:.0f}",
        )
        acc = _mean([r["final"].get("greedy_stage_accuracy") for r in group])
        pr = _mean([r["final"].get("p_right_mean") for r in group])
        one = _mean(
            [
                r["totals"]["one_action_share"]
                if "totals" in r
                else r["final"]["greedy"]["one_action_share"]
                for r in group
            ]
        )
        aroused = _mean([r["totals"]["aroused_share"] for r in group if "totals" in r])
        learned = _mean([r["totals"]["learned"] for r in group if "totals" in r], "{:.0f}")
        pos = _mean([r["totals"]["positive_share"] for r in group if "totals" in r])
        lines.append(
            f"| {mode} | {pay} | {arm} | {g} / {s} | {acc} | {pr} | {one} | {aroused}"
            f" | {learned} | {pos} |"
        )
    return "\n".join(lines)


def load_protocol(path: Path) -> tuple[dict[str, Any], bytes]:
    frozen = path.read_bytes()
    protocol: dict[str, Any] = json.loads(frozen)
    for key in (
        "steps",
        "moments",
        "probe_every",
        "probe_moments",
        "need_share",
        "brain",
        "stages",
        "tabular",
        "seeds",
    ):
        if key not in protocol:
            raise ValueError(f"protocol lacks {key}")
    return protocol, frozen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--arms", nargs="*", default=list(ARMS) + list(CONTROLS), choices=ARMS + CONTROLS
    )
    parser.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    parser.add_argument("--pays", nargs="*", default=list(PAYS), choices=PAYS)
    parser.add_argument(
        "--steps", type=int, default=None, help="chain length (the protocol's otherwise)"
    )
    parser.add_argument(
        "--moments", type=int, default=None, help="moments per life (the protocol's otherwise)"
    )
    parser.add_argument(
        "--learning",
        nargs="*",
        default=[],
        metavar="FIELD=VALUE",
        help="override LearnerConfig fields of every brain arm",
    )
    parser.add_argument(
        "--brain",
        nargs="*",
        default=[],
        metavar="FIELD=VALUE",
        help="override fields of the brain point",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="print a receipt's tables")
    parser.add_argument("--verify", type=Path, default=None)
    parser.add_argument(
        "--current", action="store_true", help="with --verify: also require the current sources"
    )
    args = parser.parse_args(argv)
    if args.report is not None:
        stored = read_receipt(args.report)
        print(markdown(stored["body"]))
        return 0
    if args.verify is not None:
        valid, reason = verify(args.verify, current=args.current)
        print(reason)
        return 0 if valid else 1
    protocol, frozen = load_protocol(args.protocol)
    overrides: dict[str, Any] = {}
    if args.steps is not None:
        protocol["steps"] = overrides["steps"] = int(args.steps)
    if args.moments is not None:
        protocol["moments"] = overrides["moments"] = int(args.moments)

    def pairs(items: list[str], name: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in items:
            field, _, raw = item.partition("=")
            if not field or not raw:
                parser.error(f"{name} takes FIELD=VALUE pairs")
            if field == "episodic":
                out[field] = raw.lower() in ("1", "true", "yes")
            else:
                out[field] = float(raw) if "." in raw or "e" in raw else int(raw)
        return out

    if args.learning:
        learning = pairs(args.learning, "--learning")
        protocol["brain"].setdefault("learning", {}).update(learning)
        overrides["learning"] = learning
    if args.brain:
        brain_overrides = pairs(args.brain, "--brain")
        protocol["brain"].update(brain_overrides)
        overrides["brain"] = brain_overrides
    seeds = args.seeds if args.seeds is not None else list(protocol["seeds"]["development"])
    manifest = Receipt.build(SCHEMA, {}, sources()).source
    out = args.out
    if out is None:
        stamp = time.strftime("%Y-%m-%d-%H%M%S")
        out = Path(__file__).with_name("results") / f"development-{stamp}.json.gz"
    checkpoints = args.checkpoints or Path(tempfile.mkdtemp(prefix="chain-probes-"))
    started = time.perf_counter()
    print(
        f"{len(seeds)} seeds x {len(args.modes)} modes x {len(args.pays)} pays"
        f" x {len(args.arms)} arms",
        file=sys.stderr,
    )
    results = run(
        protocol, args.arms, args.modes, args.pays, seeds, args.workers, checkpoints, log=sys.stderr
    )
    body = {
        "schema": SCHEMA,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(frozen).hexdigest(),
        "overrides": overrides,
        "arms": args.arms,
        "modes": args.modes,
        "pays": args.pays,
        "seeds": seeds,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cadence": cd.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "seconds": time.perf_counter() - started,
        **results,
    }
    write_receipt(out, body, manifest)
    print(f"wrote {out}", file=sys.stderr)
    print(markdown(body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
