"""The competing-skill ring: does an acquired policy survive continued learning?

A continuing ``Brain.compose`` life is raised in a nursery until one skill is routine: four
situations, each a sparse pattern over sixteen inputs, each paid for one of four actions.
Then it enters a ring. The ring can bring a competing skill (four new situations that share
half their inputs with the nursery's and pay a different action), unavoidable burn (an
action-independent loss on a quarter of the moments, which makes the dopamine noisy and
keeps the income below what life used to pay), both, or neither (the nursery continued).
The nursery skill keeps paying throughout. This is the failure of
[issue 169](https://github.com/muellerberndt/cadence/issues/169) in a table: in the robot
arena, a policy acquired in a nursery was lost under continued learning at its own actor
step, and the refresher in the nursery itself was enough to lose it.

The brain is the arena's founder: modules (48, 24), sensory scale 4, learner temperature
0.3, actor eta 0.03 with eta_bias 0.003, lam 0.6, gamma 0.95, eta_critic 5, the composed
working trace at amplitude 0.3 and decay 0.1, the associative memory at consolidation
0.05, the founder arousal law with youth 300 and a need of 0.9 of the competent nursery
income (the arena's need is in its own reward units). Every arm starts the ring from
the same saved nursery brain of its seed; the arms differ only in what ``retune`` changes
at the ring's door (the 0.80.0 life-stage interface), so the comparison is on one acquired
brain:

- ``gene``      nothing changes: the founder's actor step throughout;
- ``small``     actor eta 0.003 (the arena's twenty-fight control);
- ``smallest``  actor eta 0.001 (the arena's application setting);
- ``calm``      arousal threshold 0.5 (the arena's threshold control: learning rarely);
- ``stage``     the arena's ring stage: eta 0.001, need 0, heat 0, temperature 0.2 and the
                arousal's reward references forgotten;
- ``frozen``    the nursery brain answers greedily and takes no outcome;
- ``random``    uniform random actions.

Readings every ``probe_every`` moments, each on its own saved and reloaded copy so the
living brain is never read for a measurement: the greedy action for every situation (the
driving test's analogue: accuracy on the nursery skill A and on the ring skill B), the base
policy's probability of the right action and its margin over the best other action. On
the living brain: the share of aroused moments, the outcomes learned from, the mean
dopamine and its raw temporal-difference error, the mean actor step; and the drift of the
efficacies since the ring's door, per projection (sensory to module, module to
association and back, association to motor and back, prefrontal to association) and of
the biases per region. Over the whole ring, the sign consistency of each synapse's steps,
``|sum of steps| / sqrt(n * sum of squared steps)``: one for a signal that always pushes
the same way, near zero for noise that random-walks the synapse.

A run writes a ``cadence.Receipt`` bound to this file and every module of the library;
``--verify`` checks one and ``--report`` prints its tables. Nothing here is a claim about
a body or a default: the world is a table and the arms are the controls the issue names.
Run ``python benchmarks/competing/competing_skills.py --help``.
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

SCHEMA = "competing-skills/1"
ARMS = ("gene", "small", "smallest", "calm", "stage", "frozen", "random")
RINGS = ("full", "competing", "burn", "none")
PROTOCOL = Path(__file__).with_name("protocol.json")
INPUTS, ACTIONS, ACTIVE, SITUATIONS = 16, 4, 4, 8
BURN = -0.5


# ----------------------------------------------------------------------------------- world


def make_world(seed: int) -> dict[str, Any]:
    """The situations of one seed: four nursery patterns of four active inputs among sixteen,
    four ring patterns that keep two inputs of their nursery twin and take two others, the
    nursery's right actions (a permutation of the four) and the ring's (the nursery's shifted
    by a fixed one to three, so every ring twin asks a different action than its nursery
    twin)."""
    rng = np.random.default_rng([seed, 7])
    patterns = np.zeros((SITUATIONS, INPUTS))
    for i in range(SITUATIONS // 2):
        patterns[i, rng.choice(INPUTS, ACTIVE, replace=False)] = 1.0
    for i in range(SITUATIONS // 2):
        own = np.flatnonzero(patterns[i])
        kept = rng.choice(own, ACTIVE // 2, replace=False)
        others = np.setdiff1d(np.arange(INPUTS), own)
        fresh = rng.choice(others, ACTIVE - ACTIVE // 2, replace=False)
        patterns[SITUATIONS // 2 + i, np.concatenate([kept, fresh])] = 1.0
    nursery = rng.permutation(ACTIONS)
    shift = int(rng.integers(1, ACTIONS))
    ring = (nursery + shift) % ACTIONS
    correct = np.concatenate([nursery, ring]).astype(int)
    overlap = patterns[: SITUATIONS // 2] @ patterns[SITUATIONS // 2 :].T
    return {
        "patterns": patterns,
        "correct": correct,
        "shift": shift,
        "overlap_mean": float(overlap.mean()),
        "overlap_twins": float(np.diag(overlap).mean()),
    }


def make_moments(
    seed: int, moments: int, *, competing: bool, burn: bool, burn_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ring's events of one seed: which situation each moment presents (nursery
    situations only, or nursery and ring situations alike), whether it burns, and the
    coin that decides a probabilistic payoff. The coins and the situations are the same
    whether or not the ring burns, so the rings differ only as declared."""
    rng = np.random.default_rng([seed, 11])
    pool = SITUATIONS if competing else SITUATIONS // 2
    situations = rng.integers(pool, size=moments)
    coins = rng.random(moments)
    burns = rng.random(moments) < burn_rate if burn else np.zeros(moments, dtype=bool)
    return situations, burns, coins


def nursery_moments(seed: int, moments: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([seed, 5])
    return rng.integers(SITUATIONS // 2, size=moments), rng.random(moments)


HARD = {"right": 1.0, "wrong": 0.0}  # the right action always pays, a wrong one never


def reward_of(
    situation: int,
    action: int,
    correct: np.ndarray,
    burned: bool,
    coin: float = 0.0,
    contingency: dict[str, float] = HARD,
) -> float:
    """The outcome of an action: +1 with the contingency's probability for the right action
    (``right``) or a wrong one (``wrong``), decided by the moment's coin; plus the burn.
    The arena's progress pay is a weak contingency of this kind, not a certain payoff."""
    right = action == int(correct[situation])
    chance = float(contingency["right" if right else "wrong"])
    return (1.0 if coin < chance else 0.0) + (BURN if burned else 0.0)


def expected_income(right_share: float, contingency: dict[str, float], burn_share: float) -> float:
    right, wrong = float(contingency["right"]), float(contingency["wrong"])
    return right_share * right + (1.0 - right_share) * wrong + BURN * burn_share


# ----------------------------------------------------------------------------------- brain


def make_brain(point: dict[str, Any], seed: int) -> cd.Brain:
    """The arena's founder brain on this world's ports. ``point["learning"]`` names
    ``LearnerConfig`` fields to override (recorded in the receipt), the readout's intrinsic
    plasticity among them."""
    learning = {
        f"learning_{name}": value for name, value in dict(point.get("learning", {})).items()
    }
    return cd.Brain.compose(
        INPUTS,
        ACTIONS,
        modules=tuple(int(w) for w in point["modules"]),
        sensory_scale=float(point["sensory_scale"]),
        temperature=float(point["temperature"]),
        working_memory_amplitude=float(point["trace_amplitude"]),
        working_memory_decay=float(point["trace_decay"]),
        consolidation=float(point["consolidation"]),
        actor_eta=float(point["eta"]),
        actor_eta_bias=float(point["eta_bias"]),
        actor_lam=float(point["lam"]),
        actor_gamma=float(point["gamma"]),
        actor_eta_critic=float(point["eta_critic"]),
        arousal=dict(point["arousal"]),
        seed=seed,
        **learning,
    )


def enter_ring(brain: cd.Brain, arm: str, stages: dict[str, Any]) -> None:
    """What the arm changes at the ring's door, through ``retune`` on the acquired brain."""
    if arm in ("gene", "frozen", "random"):
        return
    genes = dict(stages[arm])
    reset = bool(genes.pop("reset_arousal", False))
    brain.retune(reset_arousal=reset, **genes)


REGIONS = ("sensory", "module_0", "association", "prefrontal", "motor")


def synapse_groups(brain: cd.Brain) -> dict[str, np.ndarray]:
    """Boolean masks of the synapses by projection (pre region to post region) and of the
    neurons by region, from the composed brain's contiguous populations."""
    w = brain.connectome
    region = np.full(w.n, -1)
    for index, name in enumerate(REGIONS):
        region[np.asarray(w.populations[name])] = index
    assert (region >= 0).all(), "a composed brain's neurons all belong to a named region"
    groups: dict[str, np.ndarray] = {}
    pre, post = region[w.pre], region[w.post]
    for a, source in enumerate(REGIONS):
        for b, target in enumerate(REGIONS):
            mask = (pre == a) & (post == b)
            if mask.any():
                groups[f"{source}->{target}"] = mask
    for index, name in enumerate(REGIONS):
        groups[f"bias:{name}"] = region == index
    return groups


def drift(
    brain: cd.Brain,
    reference: tuple[np.ndarray, np.ndarray],
    groups: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Per group: the RMS change of the parameters since ``reference`` relative to the RMS of
    the reference, and the largest absolute change."""
    efficacy, bias = brain.brain.efficacy, brain.brain.bias
    out: dict[str, dict[str, float]] = {}
    for name, mask in groups.items():
        if name.startswith("bias:"):
            now, then = bias[mask], reference[1][mask]
        else:
            now, then = efficacy[mask], reference[0][mask]
        change = now - then
        scale = float(np.sqrt(np.mean(then**2))) + 1e-12
        out[name] = {
            "rms": float(np.sqrt(np.mean(change**2)) / scale),
            "max": float(np.abs(change).max()),
        }
    return out


def frozen_readings(
    brain: cd.Brain, patterns: np.ndarray, correct: np.ndarray, temperature: float
) -> dict[str, Any]:
    """Every situation answered greedily by its own saved and reloaded copy: the choice, the
    policy's probability of the right action at ``temperature`` (the founder's base
    temperature, so arms that retune theirs are read on one scale), its margin over the
    best other action, and the sweeps the copies settled. Each situation is read twice: by
    a copy that carries the living brain's working trace and warm state into the answer
    (``carried``, the arena's frozen driving test), and by a copy whose stream state is
    reset first (``reset``: the working trace, the warm state and the arousal cleared, the
    parameters and the associative memory kept), so a change of the carried readings with
    unchanged reset readings is the trace carrying new context. The top-level readings are
    the carried ones."""
    half = len(patterns) // 2
    out: dict[str, Any] = {"sweeps": 0}
    with tempfile.TemporaryDirectory() as directory:
        path = brain.save(os.path.join(directory, "brain.npz"))
        for probe in ("carried", "reset"):
            choices, right, margins = [], [], []
            for situation in range(len(patterns)):
                copy = cd.Brain.load(path)
                if probe == "reset":
                    copy.reset()
                choice = int(copy.act(patterns[[situation]], greedy=True)[0])
                state = copy.basal_ganglia.state
                assert state is not None and copy.last_settlement is not None
                p = np.asarray(copy.basal_ganglia.probabilities(state, temperature))[0]
                target = int(correct[situation])
                others = np.delete(p, target)
                choices.append(choice)
                right.append(float(p[target]))
                margins.append(float(p[target] - others.max()))
                out["sweeps"] += int(copy.last_settlement["steps"])
            hit = [c == int(t) for c, t in zip(choices, correct, strict=True)]
            readings = {
                "choices": choices,
                "p_correct": right,
                "margin": margins,
                "accuracy_a": float(np.mean(hit[:half])),
                "accuracy_b": float(np.mean(hit[half:])),
                "p_correct_a": float(np.mean(right[:half])),
                "p_correct_b": float(np.mean(right[half:])),
                "margin_a": float(np.mean(margins[:half])),
                "margin_b": float(np.mean(margins[half:])),
            }
            if probe == "carried":
                out.update(readings)
            else:
                out["reset"] = readings
    return out


# ------------------------------------------------------------------------------------ life


class Ledger:
    """What a stretch of moments cost and what the brain learned from them."""

    def __init__(self) -> None:
        self.moments = 0
        self.aroused = 0
        self.learned = 0
        self.burned = 0
        self.reward = 0.0
        self.right = 0
        self.sweeps = 0
        self.learning_sweeps = 0
        self.delta = 0.0
        self.td_error = 0.0
        self.scale_step = 0.0
        self.bias_step = 0.0
        self.dopamine = 0.0

    def moment(self, brain: cd.Brain, reward: float, right: bool, burned: bool) -> None:
        reading = brain.last_arousal
        assert reading is not None
        self.moments += 1
        self.aroused += int(reading["mode"] == "aroused")
        self.sweeps += int(reading["sweeps"])
        self.learning_sweeps += int(reading["learning_sweeps"])
        self.reward += reward
        self.right += int(right)
        self.burned += int(burned)
        if reading["learned"]:
            report = brain.last_learning
            self.learned += 1
            self.delta += float(report["delta"])
            self.dopamine += float(report["dopamine"])
            self.td_error += float(report["td_error"])
            self.scale_step += float(report["scale_step"])
            self.bias_step += float(report["bias_step"])

    def summary(self) -> dict[str, Any]:
        n = max(self.moments, 1)
        k = max(self.learned, 1)
        return {
            "moments": self.moments,
            "aroused_share": self.aroused / n,
            "learned": self.learned,
            "burned": self.burned,
            "income": self.reward / n,
            "right_share": self.right / n,
            "sweeps": self.sweeps,
            "learning_sweeps": self.learning_sweeps,
            "mean_delta": self.delta / k,
            "mean_dopamine": self.dopamine / k,
            "mean_td_error": self.td_error / k,
            "mean_scale_step": self.scale_step / k,
            "mean_bias_step": self.bias_step / k,
        }


class Consistency:
    """The sign consistency of every synapse's actor steps across the learned moments."""

    def __init__(self, synapses: int) -> None:
        self.total = np.zeros(synapses)
        self.square = np.zeros(synapses)
        self.count = 0

    def add(self, step: np.ndarray) -> None:
        self.total += step
        self.square += step * step
        self.count += 1

    def by_group(self, groups: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        if not self.count:
            return out
        rms = np.sqrt(self.square / self.count)
        moved = rms > 0
        consistency = np.zeros(len(rms))
        consistency[moved] = np.abs(self.total[moved]) / np.sqrt(self.count * self.square[moved])
        for name, mask in groups.items():
            if name.startswith("bias:"):
                continue
            part = mask & moved
            out[name] = {
                "consistency": float(consistency[part].mean()) if part.any() else 0.0,
                "rms_step": float(rms[mask].mean()),
                "net": float(np.abs(self.total[mask]).mean()),
                "moved": int(part.sum()),
                "synapses": int(mask.sum()),
                "count": int(self.count),
            }
        return out


def live_stretch(
    brain: cd.Brain,
    world: dict[str, Any],
    situations: np.ndarray,
    burns: np.ndarray,
    coins: np.ndarray,
    ledger: Ledger,
    consistency: Consistency | None,
    *,
    reward: float | None,
) -> float | None:
    """Live the moments in order; returns the outcome owed to the last action."""
    patterns, correct = world["patterns"], world["correct"]
    contingency = world["contingency"]
    events = zip(situations.tolist(), burns.tolist(), coins.tolist(), strict=True)
    for situation, burned, coin in events:
        before = (
            brain.brain.efficacy.copy()
            if consistency is not None and brain.pending_feedback
            else None
        )
        pay = None if reward is None else np.array([reward])
        action = int(brain.live(patterns[[situation]], reward=pay)[0])
        reading = brain.last_arousal
        assert reading is not None
        if before is not None and reading["learned"]:
            assert consistency is not None
            consistency.add(brain.brain.efficacy - before)
        right = action == int(correct[situation])
        reward = reward_of(situation, action, correct, burned, coin, contingency)
        ledger.moment(brain, reward, right, burned)
    return reward


def run_life(
    arm: str, ring: str, seed: int, protocol: dict[str, Any], nursery_path: Path
) -> dict[str, Any]:
    """One arm of one seed through the ring, from the seed's saved nursery brain."""
    started = time.perf_counter()
    world = make_world(seed)
    world["contingency"] = dict(protocol.get("contingency", HARD))
    moments = int(protocol["phases"]["ring"])
    probe_every = int(protocol["probe_every"])
    situations, burns, coins = make_moments(
        seed,
        moments,
        competing=ring in ("full", "competing"),
        burn=ring in ("full", "burn"),
        burn_rate=float(protocol["burn_rate"]),
    )
    probes: list[dict[str, Any]] = []
    totals = Ledger()
    row: dict[str, Any] = {
        "arm": arm,
        "ring": ring,
        "seed": seed,
        "world": {k: v for k, v in world.items() if k != "patterns"},
    }
    if arm == "random":
        rng = np.random.default_rng([seed, 13])
        actions = rng.integers(ACTIONS, size=moments)
        right = actions == world["correct"][situations]
        row["probes"] = []
        row["totals"] = {
            "moments": moments,
            "right_share": float(right.mean()),
            "income": expected_income(float(right.mean()), world["contingency"], burns.mean()),
        }
        row["final"] = {"accuracy_a": 1 / ACTIONS, "accuracy_b": 1 / ACTIONS}
        row["seconds"] = time.perf_counter() - started
        return row
    brain = cd.Brain.load(nursery_path)
    owed = json.loads(nursery_path.with_suffix(".json").read_text())["owed"]
    temperature = float(protocol["brain"]["temperature"])
    groups = synapse_groups(brain)
    enter_ring(brain, arm, protocol["stages"])
    row["door"] = {
        "actor": brain.describe()["actor"],
        "arousal": brain.describe()["arousal"],
        "learning": brain.describe()["learning"],
        "owed": owed,
    }
    reference = (brain.brain.efficacy.copy(), brain.brain.bias.copy())
    entrance = frozen_readings(brain, world["patterns"], world["correct"], temperature)
    probes.append({"moment": 0, "readings": entrance, "window": Ledger().summary()})
    consistency = Consistency(brain.connectome.synapses)
    reward: float | None = None if owed is None else float(owed)
    if arm == "frozen":
        correct = world["correct"]
        right = 0
        for situation in situations.tolist():
            action = int(brain.act(world["patterns"][[situation]], greedy=True)[0])
            right += int(action == int(correct[situation]))
        row["probes"] = probes
        row["totals"] = {
            "moments": moments,
            "right_share": right / moments,
            "income": expected_income(right / moments, world["contingency"], float(burns.mean())),
        }
        row["final"] = frozen_readings(brain, world["patterns"], world["correct"], temperature)
        row["seconds"] = time.perf_counter() - started
        return row
    for start in range(0, moments, probe_every):
        stop = min(start + probe_every, moments)
        window = Ledger()
        reward = live_stretch(
            brain,
            world,
            situations[start:stop],
            burns[start:stop],
            coins[start:stop],
            window,
            consistency,
            reward=reward,
        )
        for name in vars(window):
            setattr(totals, name, getattr(totals, name) + getattr(window, name))
        probes.append(
            {
                "moment": stop,
                "readings": frozen_readings(
                    brain, world["patterns"], world["correct"], temperature
                ),
                "window": window.summary(),
                "drift": drift(brain, reference, groups),
                "level": float(brain.arousal.level) if brain.arousal is not None else None,
                "want": float(brain.arousal.want) if brain.arousal is not None else None,
            }
        )
    row["probes"] = probes
    row["totals"] = totals.summary()
    row["final"] = probes[-1]["readings"]
    row["consistency"] = consistency.by_group(groups)
    row["drift"] = drift(brain, reference, groups)
    row["seconds"] = time.perf_counter() - started
    return row


def raise_nursery(seed: int, protocol: dict[str, Any], path: Path) -> dict[str, Any]:
    """The nursery of one seed: the founder brain lives the nursery moments and is saved at
    the door of the ring. Returns its readings."""
    started = time.perf_counter()
    world = make_world(seed)
    world["contingency"] = dict(protocol.get("contingency", HARD))
    moments = int(protocol["phases"]["nursery"])
    probe_every = int(protocol["probe_every"])
    brain = make_brain(protocol["brain"], seed)
    temperature = float(protocol["brain"]["temperature"])
    situations, coins = nursery_moments(seed, moments)
    burns = np.zeros(moments, dtype=bool)
    probes: list[dict[str, Any]] = []
    totals = Ledger()
    reward: float | None = None
    for start in range(0, moments, probe_every):
        stop = min(start + probe_every, moments)
        window = Ledger()
        reward = live_stretch(
            brain,
            world,
            situations[start:stop],
            burns[start:stop],
            coins[start:stop],
            window,
            None,
            reward=reward,
        )
        for name in vars(window):
            setattr(totals, name, getattr(totals, name) + getattr(window, name))
        probes.append(
            {
                "moment": stop,
                "readings": frozen_readings(
                    brain, world["patterns"], world["correct"], temperature
                ),
                "window": window.summary(),
            }
        )
    # The outcome of the last nursery action is owed; the brain is saved with that pending
    # feedback, as a life saved mid-stream is, and the ring's first moment pays it.
    brain.save(path)
    owed_path = path.with_suffix(".json")
    owed_path.write_text(json.dumps({"owed": reward}))
    return {
        "seed": seed,
        "probes": probes,
        "totals": totals.summary(),
        "final": probes[-1]["readings"],
        "owed": reward,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "describe": brain.describe(),
        "seconds": time.perf_counter() - started,
    }


# ------------------------------------------------------------------------------------- run


def _nursery_job(args: tuple[int, dict[str, Any], str]) -> dict[str, Any]:
    seed, protocol, path = args
    return raise_nursery(seed, protocol, Path(path))


def _ring_job(args: tuple[str, str, int, dict[str, Any], str]) -> dict[str, Any]:
    arm, ring, seed, protocol, path = args
    return run_life(arm, ring, seed, protocol, Path(path))


def run(
    protocol: dict[str, Any],
    arms: list[str],
    rings: list[str],
    seeds: list[int],
    workers: int,
    checkpoints: Path,
    log: Any = None,
) -> dict[str, Any]:
    checkpoints.mkdir(parents=True, exist_ok=True)
    nursery_jobs = [(seed, protocol, str(checkpoints / f"nursery-{seed}.npz")) for seed in seeds]
    ring_jobs = [
        (arm, ring, seed, protocol, str(checkpoints / f"nursery-{seed}.npz"))
        for seed in seeds
        for arm in arms
        for ring in (rings if arm not in ("frozen", "random") else [rings[0]])
    ]

    def execute(function: Any, jobs: list[Any]) -> list[dict[str, Any]]:
        if workers <= 1 or len(jobs) <= 1:
            results = []
            for job in jobs:
                results.append(function(job))
                if log is not None:
                    print(".", end="", file=log, flush=True)
            return results
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            results = []
            for result in pool.imap_unordered(function, jobs):
                results.append(result)
                if log is not None:
                    print(".", end="", file=log, flush=True)
            return results

    nurseries = execute(_nursery_job, nursery_jobs)
    rows = execute(_ring_job, ring_jobs)
    nurseries.sort(key=lambda r: r["seed"])
    rows.sort(key=lambda r: (r["ring"], r["arm"], r["seed"]))
    if log is not None:
        print(file=log)
    return {"nurseries": nurseries, "rows": rows}


# --------------------------------------------------------------------------------- receipts


def sources() -> list[tuple[str, Path]]:
    package = Path(cd.__file__).resolve().parent
    return [
        ("competing_skills.py", Path(__file__).resolve()),
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
    return stored["body"] if "body" in stored else stored


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


def _mean_min(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{np.mean(values):.2f} ({np.min(values):.2f})"


def _mean(values: list[Any], fmt: str = "{:.2f}") -> str:
    present = [float(v) for v in values if v is not None]
    return fmt.format(np.mean(present)) if present else "-"


def markdown(body: dict[str, Any]) -> str:
    rows = body["rows"]
    contingency = body["protocol"].get("contingency", HARD)
    learning = body["protocol"]["brain"].get("learning") or {}
    lines = [
        f"# {SCHEMA}: {len(body['nurseries'])} seeds, {len(rows)} ring lives, "
        f"the right action pays with probability {contingency['right']:g}, "
        f"a wrong one with {contingency['wrong']:g}"
        + (f", learning overrides {learning}" if learning else ""),
        "",
    ]
    lines.append("## Nursery: skill A at the door of the ring")
    lines.append("")
    lines.append(
        "| Seed | A accuracy | A p(correct) | A margin | Aroused share | Learned | Income |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for n in body["nurseries"]:
        f, t = n["final"], n["totals"]
        lines.append(
            f"| {n['seed']} | {f['accuracy_a']:.2f} | {f['p_correct_a']:.2f} | "
            f"{f['margin_a']:.2f} | {t['aroused_share']:.2f} | {t['learned']} | "
            f"{t['income']:.2f} |"
        )
    lines.append("")
    columns = (
        "Arm",
        "A at door",
        "A at end",
        "A at end, trace reset",
        "B at end",
        "B at end, trace reset",
        "A p(correct) end",
        "A margin end",
        "Aroused share",
        "Learned",
        "Mean |dopamine|",
        "Drift assoc->motor",
        "Drift sensory->module",
        "Consistency assoc->motor",
        "Income",
    )
    for ring in RINGS:
        group = [r for r in rows if r["ring"] == ring]
        if not group:
            continue
        lines.append(f"## Ring `{ring}`: skill A kept, skill B acquired, mean (minimum) over seeds")
        lines.append("")
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + " --- |" * len(columns))
        for arm in ARMS:
            lives = [r for r in group if r["arm"] == arm]
            if not lives:
                continue
            door = [r["probes"][0]["readings"]["accuracy_a"] for r in lives if r["probes"]]
            end_a = [r["final"]["accuracy_a"] for r in lives]
            end_b = [r["final"]["accuracy_b"] for r in lives]
            totals = [r["totals"] for r in lives]
            drifts = [r["drift"] for r in lives if "drift" in r]
            cons = [r["consistency"] for r in lives if r.get("consistency")]
            reset_a = [r["final"]["reset"]["accuracy_a"] for r in lives if "reset" in r["final"]]
            reset_b = [r["final"]["reset"]["accuracy_b"] for r in lives if "reset" in r["final"]]
            cells = [
                f"`{arm}`",
                _mean_min(door),
                _mean_min(end_a),
                _mean_min(reset_a),
                _mean_min(end_b),
                _mean_min(reset_b),
                _mean([r["final"].get("p_correct_a") for r in lives]),
                _mean([r["final"].get("margin_a") for r in lives]),
                _mean([t.get("aroused_share") for t in totals]),
                _mean([t.get("learned") for t in totals], "{:.0f}"),
                _mean([t.get("mean_delta") for t in totals], "{:.3f}"),
                _mean([d["association->motor"]["rms"] for d in drifts], "{:.3f}"),
                _mean([d["sensory->module_0"]["rms"] for d in drifts], "{:.3f}"),
                _mean([c["association->motor"]["consistency"] for c in cons]),
                _mean([t["income"] for t in totals]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def curves(body: dict[str, Any], ring: str, arm: str) -> str:
    """Skill A accuracy and the aroused share along the ring, per seed."""
    lines = [
        f"ring {ring}, arm {arm}: moment: A accuracy carried / A accuracy with the trace reset / "
        "aroused share, per seed"
    ]
    lives = [r for r in body["rows"] if r["ring"] == ring and r["arm"] == arm]
    if not lives:
        return "no lives"
    moments = [p["moment"] for p in lives[0]["probes"]]
    for index, moment in enumerate(moments):
        cells = []
        for life in lives:
            probe = life["probes"][index]
            readings = probe["readings"]
            reset = readings.get("reset", {}).get("accuracy_a")
            cells.append(
                f"{readings['accuracy_a']:.2f}/"
                f"{'-' if reset is None else f'{reset:.2f}'}/"
                f"{probe['window']['aroused_share']:.2f}"
            )
        lines.append(f"{moment:6d}: " + "  ".join(cells))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    parser.add_argument("--rings", nargs="*", default=list(RINGS), choices=RINGS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--nursery", type=int, default=None, help="override the nursery moments")
    parser.add_argument("--ring", type=int, default=None, help="override the ring moments")
    parser.add_argument("--probe-every", type=int, default=None)
    parser.add_argument(
        "--learning",
        nargs="*",
        default=[],
        metavar="FIELD=VALUE",
        help="override LearnerConfig fields of the brain, e.g. homeostasis_rate=0.02",
    )
    parser.add_argument(
        "--contingency",
        nargs=2,
        type=float,
        default=None,
        metavar=("RIGHT", "WRONG"),
        help="probability that the right / a wrong action pays +1 (the protocol's is 1 and 0)",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path, default=None, help="where nurseries are saved")
    parser.add_argument("--report", type=Path, default=None, help="print a receipt's tables")
    parser.add_argument("--curves", nargs=2, default=None, metavar=("RING", "ARM"))
    parser.add_argument("--verify", type=Path, default=None, help="check a receipt")
    parser.add_argument(
        "--current", action="store_true", help="with --verify: also require the current sources"
    )
    args = parser.parse_args(argv)
    if args.verify is not None:
        ok, message = verify(args.verify, current=args.current)
        print(message or ("verified" if ok else "failed"))
        return 0 if ok else 1
    if args.report is not None:
        body = read_receipt(args.report)
        if args.curves is not None:
            print(curves(body, *args.curves))
        else:
            print(markdown(body))
        return 0
    frozen = args.protocol.read_bytes()
    protocol = json.loads(frozen)
    overrides: dict[str, Any] = {}
    if args.nursery is not None:
        protocol["phases"]["nursery"] = overrides["nursery"] = args.nursery
    if args.ring is not None:
        protocol["phases"]["ring"] = overrides["ring"] = args.ring
    if args.probe_every is not None:
        protocol["probe_every"] = overrides["probe_every"] = args.probe_every
    if args.learning:
        learning: dict[str, Any] = dict(protocol["brain"].get("learning", {}))
        for item in args.learning:
            field, _, raw = item.partition("=")
            if not field or not raw:
                parser.error("--learning takes FIELD=VALUE pairs")
            learning[field] = float(raw) if "." in raw or "e" in raw else int(raw)
        protocol["brain"]["learning"] = overrides["learning"] = learning
    if args.contingency is not None:
        right, wrong = args.contingency
        if not 0 <= wrong <= right <= 1:
            parser.error("--contingency needs 0 <= WRONG <= RIGHT <= 1")
        protocol["contingency"] = overrides["contingency"] = {"right": right, "wrong": wrong}
    seeds = args.seeds if args.seeds is not None else list(protocol["seeds"]["development"])
    manifest = Receipt.build(SCHEMA, {}, sources()).source
    out = args.out
    if out is None:
        stamp = time.strftime("%Y-%m-%d-%H%M%S")
        out = Path(__file__).with_name("results") / f"development-{stamp}.json.gz"
    checkpoints = args.checkpoints or Path(tempfile.mkdtemp(prefix="competing-nursery-"))
    started = time.perf_counter()
    print(f"{len(seeds)} seeds x {len(args.arms)} arms x {len(args.rings)} rings", file=sys.stderr)
    results = run(protocol, args.arms, args.rings, seeds, args.workers, checkpoints, log=sys.stderr)
    body = {
        "schema": SCHEMA,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(frozen).hexdigest(),
        "overrides": overrides,
        "arms": args.arms,
        "rings": args.rings,
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
