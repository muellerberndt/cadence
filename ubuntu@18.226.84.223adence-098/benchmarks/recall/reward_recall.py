"""Reward-based recall: one continuing ``Brain.live`` life paid at the query, and the write into
the working trace as an action of the basal ganglia (``REWARD_RECALL_PROTOCOL.md``).

The world is the finite recall chamber's (``finite_horizon_inputs``): a cue of one of two
values, neutral or distractor events, an optional replacement, then a query at which the
brain answers and is paid. Every chamber so far taught the answer by lesson; here the brain is
paid for remembering, through ``Brain.live``, with the working trace as the only carrier.

The ``gated`` arm composes a second motor slot of two choices, hold or write. The library
writes the trace after every moment; after a hold the instrument restores the trace, its last
activation and its cold flag to their values before the moment, which equals skipping the
founder's write. The choice is sampled and credited by the same actor and the same reward as
the answer, through its eligibility over the hold span. The ``fixed`` arm is the founder.

    PYTHONPATH=src python benchmarks/recall/reward_recall.py --out OUT --seeds 0 1
    PYTHONPATH=src python benchmarks/recall/reward_recall.py --report OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import finite_horizon_inputs as inputs  # noqa: E402

SCHEMA = "reward-recall/1"
PROTOCOL = HERE / "protocol-reward.json"
HOLD, WRITE = 0, 1
KINDS = ("neutral", "distractor", "start", "write", "query")
CONDITIONS = {c.name: c for c in inputs.TRAIN_CONDITIONS}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema") != SCHEMA:
        raise ValueError(f"protocol schema must be {SCHEMA}")
    unknown = [c for c in protocol["conditions"] if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions {unknown}")
    return protocol, hashlib.sha256(raw).hexdigest()


def kind_of(x: np.ndarray) -> str:
    """The event's marker: one of the first eight coordinates is lit."""
    marker = int(np.argmax(x[:8]))
    return KINDS[min(marker, 4)]


def compose(recipe: dict[str, Any], seed: int, gated: bool) -> Any:
    from cadence import Brain  # noqa: PLC0415

    options = dict(recipe["compose"])
    options["modules"] = tuple(int(w) for w in options["modules"])
    actions, slots = (4, (2, 2)) if gated else (2, 1)
    arousal = (
        dict(recipe["arousal"]) if recipe.get("arousal") else True
    )  # a mapping patches the founder
    return Brain.compose(
        inputs.INPUTS,
        actions, backend="cpu",
        slots=slots,
        seed=int(seed),
        arousal=arousal,
        episodic=False,
        **{"recall": False, **options},
    )


def decode(output: Any, gated: bool) -> tuple[int, int]:
    """The answer (slot 0) and the write choice (slot 1, or the founder's write)."""
    flat = np.asarray(output).reshape(-1)
    if len(flat) != (2 if gated else 1):
        raise ValueError(f"expected {2 if gated else 1} choices, got {flat.shape}")
    answer = int(flat[0])
    if answer not in (0, 1):
        raise ValueError(f"answer {answer} outside the alphabet")
    write = int(flat[1]) if gated else WRITE
    if write not in (HOLD, WRITE):
        raise ValueError(f"write choice {write} outside the alphabet")
    return answer, write


def snapshot(brain: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace = brain.working_memory
    return trace.trace.copy(), trace.last.copy(), trace.cold.copy()


def restore(brain: Any, saved: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    trace = brain.working_memory
    if trace.trace.shape != saved[0].shape:
        trace.reset(len(saved[0]))
    trace.trace[...] = saved[0]
    trace.last[...] = saved[1]
    trace.cold[...] = saved[2]


def episodes_of(
    rng: np.random.Generator, names: list[str], count: int, history: bool = False
) -> list[dict[str, Any]]:
    """``count`` single-stream episodes, conditions drawn uniformly, every stream of each
    frozen eight-stream draw used in turn (the draws keep their paired labels). With
    ``history`` the asked cue's raw payload is appended at the query (the external-history
    input of the finite chamber), a reward-path control and never a recall arm."""
    out: list[dict[str, Any]] = []
    while len(out) < count:
        name = names[int(rng.integers(len(names)))]
        episode = inputs.make_episode(rng, CONDITIONS[name])
        observations = episode.observations
        if history:
            observations = inputs.append_observed_history(observations)
        for row in range(inputs.STREAMS):
            if len(out) >= count:
                break
            out.append(
                {
                    "condition": name,
                    "observations": np.ascontiguousarray(observations[:, row, :]),
                    "label": int(episode.labels[row]),
                }
            )
    return out


@dataclass
class Tally:
    episodes: int = 0
    right: int = 0
    income: float = 0.0
    moments: int = 0
    aroused: int = 0
    learned: int = 0
    positive_dopamine: int = 0
    sweeps: int = 0
    writes: dict[str, list[int]] = field(default_factory=lambda: {k: [0, 0] for k in KINDS})

    def moment(self, brain: Any, kind: str, write: int) -> None:
        self.moments += 1
        self.aroused += int(bool(brain.arousal.aroused))
        learning = brain.last_learning or {}
        if learning:
            self.learned += 1
            self.positive_dopamine += int(float(learning.get("dopamine", 0.0)) > 0)
        settlement = brain.last_settlement or {}
        self.sweeps += int(settlement.get("steps", 0))
        self.writes[kind][0] += int(write == WRITE)
        self.writes[kind][1] += 1

    def summary(self) -> dict[str, Any]:
        n = max(self.moments, 1)
        return {
            "episodes": self.episodes,
            "success_rate": self.right / max(self.episodes, 1),
            "income": self.income,
            "moments": self.moments,
            "aroused_share": self.aroused / n,
            "learned_moments": self.learned,
            "positive_dopamine_share": self.positive_dopamine / max(self.learned, 1),
            "sweeps": self.sweeps,
            "write_share": {k: (v[0] / v[1] if v[1] else None) for k, v in self.writes.items()},
        }


class Life:
    """One continuing ``Brain.live`` stream across episodes; the pay of a query arrives with the
    next episode's first observation."""

    def __init__(
        self, brain: Any, gated: bool, pay: dict[str, float], interface: str = "live"
    ) -> None:
        if interface not in ("live", "step"):
            raise ValueError("interface is live or step")
        self.brain, self.gated, self.pay, self.interface = brain, gated, pay, interface
        self.call = brain.live if interface == "live" else brain.step
        self.pending: float | None = None
        self.started = False

    def moment(
        self, x: np.ndarray, reward: float | None, done: bool, tally: Tally
    ) -> tuple[int, int]:
        saved = snapshot(self.brain) if self.gated else None
        if not self.started:
            out = self.call(x[None, :])
            self.started = True
        else:
            out = self.call(
                x[None, :],
                reward=np.asarray([0.0 if reward is None else reward]),
                done=np.asarray([done]),
            )
        answer, write = decode(out, self.gated)
        if self.gated and write == HOLD:
            assert saved is not None
            restore(self.brain, saved)
        tally.moment(self.brain, kind_of(x), write)
        return answer, write

    def episode(self, episode: dict[str, Any], tally: Tally) -> int:
        observations, label = episode["observations"], episode["label"]
        answer = -1
        for i, x in enumerate(observations):
            if i == 0:
                answer, _ = self.moment(x, self.pending, self.pending is not None, tally)
            else:
                answer, _ = self.moment(x, 0.0, False, tally)
        right = answer == label
        self.pending = self.pay["right"] if right else self.pay["wrong"]
        tally.episodes += 1
        tally.right += int(right)
        tally.income += self.pending
        return answer

    def settle_pay(self, tally: Tally) -> None:
        """Deliver the last query's pay with a START observation; the issued action is the
        world's throwaway (V5's order)."""
        if self.pending is None:
            return
        start = np.zeros(inputs.INPUTS)
        start[inputs.START] = 1.0
        self.moment(start, self.pending, True, tally)
        self.pending = None


def evaluate(
    checkpoint: Path, protocol: dict[str, Any], gated: bool, seed: int, history: bool = False
) -> dict[str, Any]:
    """Frozen copies read every condition under each mode and lesion; the checkpoint stays."""
    from cadence import Brain  # noqa: PLC0415

    before = sha256(checkpoint)
    evaluation = protocol["evaluation"]
    count = int(evaluation["episodes_per_condition"])
    out: dict[str, Any] = {}
    writes: dict[str, list[int]] = {k: [0, 0] for k in KINDS}
    for mode in evaluation["modes"]:
        out[mode] = {}
        for lesion in evaluation["lesions"]:
            out[mode][lesion] = {}
            for name in protocol["conditions"]:
                brain = Brain.load(checkpoint)
                brain.basal_ganglia.rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            seed,
                            7,
                            evaluation["modes"].index(mode),
                            evaluation["lesions"].index(lesion),
                            protocol["conditions"].index(name),
                        ]
                    )
                )
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, 11, protocol["conditions"].index(name)])
                )
                episodes = episodes_of(rng, [name], count, history)
                hits = 0
                ones = 0
                for episode in episodes:
                    brain.reset()
                    observations = episode["observations"]
                    answer = -1
                    for i, x in enumerate(observations):
                        if i == len(observations) - 1:
                            if lesion == "erased":
                                brain.working_memory.reset(1, rows=np.array([0]))
                            elif lesion == "reset":
                                brain.reset()
                        saved = snapshot(brain) if gated else None
                        answer, write = decode(
                            brain.act(x[None, :], greedy=mode == "greedy"), gated
                        )
                        if gated and write == HOLD:
                            assert saved is not None
                            restore(brain, saved)
                        if lesion == "intact" and mode == "greedy":
                            kind = kind_of(x)
                            writes[kind][0] += int(write == WRITE)
                            writes[kind][1] += 1
                    hits += int(answer == episode["label"])
                    ones += int(answer == 1)
                out[mode][lesion][name] = hits / count
                out[mode][lesion][name + "/answer-1-share"] = ones / count
    if sha256(checkpoint) != before:
        raise RuntimeError("evaluation changed the checkpoint")
    out["write_share_greedy_intact"] = {
        k: (v[0] / v[1] if v[1] else None) for k, v in writes.items()
    }
    return out


def history_control(protocol: dict[str, Any], seed: int) -> dict[str, Any]:
    """A tabular learner given the asked value at the query: Q over (value, answer)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 13]))
    q = np.zeros((2, 2))
    alpha, epsilon = 0.1, 0.1
    pay = protocol["pay"]
    for episode in episodes_of(rng, protocol["conditions"], protocol["training_episodes"]):
        value = episode["label"]
        answer = int(rng.integers(2)) if rng.random() < epsilon else int(np.argmax(q[value]))
        reward = pay["right"] if answer == value else pay["wrong"]
        q[value, answer] += alpha * (reward - q[value, answer])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 17]))
    accuracy = {}
    for name in protocol["conditions"]:
        episodes = episodes_of(rng, [name], protocol["evaluation"]["episodes_per_condition"])
        accuracy[name] = float(
            np.mean([int(np.argmax(q[e["label"]])) == e["label"] for e in episodes])
        )
    return {"accuracy": accuracy, "q": q.tolist()}


def random_control(protocol: dict[str, Any], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 19]))
    count = protocol["evaluation"]["episodes_per_condition"]
    return {
        name: float(
            np.mean([int(rng.integers(2)) == e["label"] for e in episodes_of(rng, [name], count)])
        )
        for name in protocol["conditions"]
    }


def gates_for(
    probe: dict[str, Any], protocol: dict[str, Any], history: dict[str, Any]
) -> dict[str, Any]:
    g = protocol["gates"]
    greedy, erased, reset = (
        probe["greedy"]["intact"],
        probe["greedy"]["erased"],
        probe["greedy"]["reset"],
    )
    horizon = 0
    for name in ("clean-0", "clean-1", "clean-2"):
        if greedy.get(name, 0.0) >= g["floor"]:
            horizon += 1
        else:
            break
    through_trace = all(
        erased[name] <= g["lesion"]
        and reset[name] <= g["lesion"]
        and greedy[name] >= erased[name] + g["lesion_margin"]
        and greedy[name] >= reset[name] + g["lesion_margin"]
        for name in protocol["conditions"]
        if CONDITIONS[name].delay > 0
    )
    checks = {
        "horizon": horizon,
        "distractor": greedy.get("distractor-1", 0.0) >= g["floor"],
        "replacement": greedy.get("replacement-1", 0.0) >= g["floor"],
        "order": greedy.get("order-latest-1", 0.0) >= g["floor"],
        "through_trace": through_trace,
        "history_competent": min(history["accuracy"].values()) >= g["history"],
    }
    checks["closure"] = bool(
        checks["horizon"] >= 2
        and checks["distractor"]
        and checks["replacement"]
        and checks["order"]
        and checks["through_trace"]
        and checks["history_competent"]
    )
    return checks


def run_founder(job: tuple[str, int, str, str, str]) -> dict[str, Any]:
    protocol_path, seed, arm, recipe_name, out = job
    import cadence  # noqa: PLC0415

    began = time.time()
    protocol, digest = load_protocol(Path(protocol_path))
    spec = protocol["arms"][arm]
    gated = bool(spec["gated"])
    history = bool(spec.get("history", False))
    recipe = protocol["recipes"][recipe_name]
    directory = Path(out) / recipe_name / arm / f"founder-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    brain = compose(recipe, seed, gated)
    life = Life(brain, gated, protocol["pay"], spec.get("interface", "live"))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    episodes = episodes_of(rng, protocol["conditions"], protocol["training_episodes"], history)
    control = history_control(protocol, seed)
    probes: dict[str, Any] = {}
    segments: dict[str, Any] = {}
    checkpoint = directory / "episodes-0.npz"
    brain.save(checkpoint)
    probes["0"] = evaluate(checkpoint, protocol, gated, seed, history)
    probes["0"]["gates"] = gates_for(probes["0"], protocol, control)
    probe_every = int(protocol["probe_every"])
    tally = Tally()
    for index, episode in enumerate(episodes, start=1):
        life.episode(episode, tally)
        if index % probe_every == 0 or index == len(episodes):
            life.settle_pay(tally)  # the stream pauses at an episode boundary with no pay owed
            checkpoint = directory / f"episodes-{index}.npz"
            brain.save(checkpoint)
            probes[str(index)] = evaluate(checkpoint, protocol, gated, seed, history)
            probes[str(index)]["gates"] = gates_for(probes[str(index)], protocol, control)
            segments[str(index)] = tally.summary()
            tally = Tally()
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "recipe": recipe_name,
                        "episodes": index,
                        "train_success": segments[str(index)]["success_rate"],
                        "greedy": probes[str(index)]["greedy"]["intact"],
                        "gates": probes[str(index)]["gates"],
                    }
                ),
                flush=True,
            )
        if time.time() - began > protocol["caps"]["per_founder_arm_seconds"]:
            segments["capped_at"] = index
            break
    result = {
        "schema": SCHEMA,
        "protocol_sha256": digest,
        "seed": seed,
        "arm": arm,
        "recipe": recipe_name,
        "gated": gated,
        "history": history,
        "interface": spec.get("interface", "live"),
        "genes": {
            k: (v if isinstance(v, (int, float, str, bool, list, dict)) else repr(v))
            for k, v in brain.describe().items()
        },
        "training": segments,
        "probes": probes,
        "history_control": control,
        "random_control": random_control(protocol, seed),
        "cadence_version": cadence.__version__,
        "numpy_version": np.__version__,
        "sources": sources(),
        "seconds": round(time.time() - began, 2),
    }
    (directory / "founder.json").write_text(json.dumps(result, indent=1))
    return result


def sources() -> dict[str, str]:
    import cadence  # noqa: PLC0415

    package = Path(cadence.__file__).resolve().parent
    out = {
        "benchmarks/recall/" + name: sha256(HERE / name)
        for name in ("reward_recall.py", "finite_horizon_inputs.py", "protocol-reward.json")
    }
    out.update(
        {
            "src/cadence/" + str(p.relative_to(package)): sha256(p)
            for p in sorted(package.rglob("*.py"))
        }
    )
    return out


def report(out: Path) -> str:
    lines = []
    bodies = [json.loads(p.read_text()) for p in sorted(out.glob("*/*/founder-*/founder.json"))]
    if not bodies:
        return "no founders"
    protocol = (
        json.loads((out / "protocol.json").read_text())
        if (out / "protocol.json").exists()
        else None
    )
    conditions = (
        protocol["conditions"] if protocol else list(bodies[0]["probes"]["0"]["greedy"]["intact"])
    )
    for recipe in sorted({b["recipe"] for b in bodies}):
        for arm in sorted({b["arm"] for b in bodies if b["recipe"] == recipe}):
            rows = sorted(
                (b for b in bodies if b["recipe"] == recipe and b["arm"] == arm),
                key=lambda b: b["seed"],
            )
            last = [max(b["probes"], key=int) for b in rows]
            lines.append(
                f"### {recipe} / {arm}: founders {[b['seed'] for b in rows]}, final probe at "
                f"{sorted(set(last))} episodes"
            )
            lines.append("")
            lines.append(
                "| condition | greedy intact % [per founder] | sampled intact % | erased % "
                "| reset % | history % | random % |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for name in conditions:

                def col(get: Any, rows: list = rows, last: list = last) -> str:
                    values = [get(b, k) for b, k in zip(rows, last, strict=True)]
                    each = " ".join(f"{v * 100:.0f}" for v in values)
                    return f"{np.mean(values) * 100:.0f} [{each}]"

                lines.append(
                    "| "
                    + " | ".join(
                        [
                            name,
                            col(lambda b, k, n=name: b["probes"][k]["greedy"]["intact"][n]),
                            col(lambda b, k, n=name: b["probes"][k]["sampled"]["intact"][n]),
                            col(lambda b, k, n=name: b["probes"][k]["greedy"]["erased"][n]),
                            col(lambda b, k, n=name: b["probes"][k]["greedy"]["reset"][n]),
                            col(lambda b, k, n=name: b["history_control"]["accuracy"][n]),
                            col(lambda b, k, n=name: b["random_control"][n]),
                        ]
                    )
                    + " |"
                )
            lines.append("")
            gates = [b["probes"][k]["gates"] for b, k in zip(rows, last, strict=True)]
            lines.append(
                "- horizons "
                + str([g["horizon"] for g in gates])
                + "; closure "
                + str(sum(g["closure"] for g in gates))
                + f"/{len(gates)}; distractor "
                + str(sum(g["distractor"] for g in gates))
                + "; replacement "
                + str(sum(g["replacement"] for g in gates))
                + "; order "
                + str(sum(g["order"] for g in gates))
                + "; through the trace "
                + str(sum(g["through_trace"] for g in gates))
            )
            shares = [
                b["probes"][k]["write_share_greedy_intact"] for b, k in zip(rows, last, strict=True)
            ]
            lines.append(
                "- write share at evaluation (greedy, intact), founder means: "
                + ", ".join(
                    f"{kind} {np.mean([s[kind] for s in shares if s[kind] is not None]):.2f}"
                    for kind in KINDS
                    if any(s[kind] is not None for s in shares)
                )
            )
            training = [
                b["training"][k] for b, k in zip(rows, last, strict=True) if k in b["training"]
            ]
            if training:
                shares = {
                    kind: np.mean(
                        [
                            t["write_share"][kind]
                            for t in training
                            if t["write_share"][kind] is not None
                        ]
                    )
                    for kind in KINDS
                    if any(t["write_share"][kind] is not None for t in training)
                }
                share_text = ", ".join(f"{kind} {value:.2f}" for kind, value in shares.items())
                lines.append(
                    "- last training segment, founder means: success "
                    f"{np.mean([t['success_rate'] for t in training]):.2f}, aroused share "
                    f"{np.mean([t['aroused_share'] for t in training]):.2f}, learned moments "
                    f"{np.mean([t['learned_moments'] for t in training]):.0f}, positive dopamine "
                    f"{np.mean([t['positive_dopamine_share'] for t in training]):.2f}; write share "
                    + share_text
                )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--recipes", nargs="*")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--episodes", type=int, help="development override of training_episodes")
    parser.add_argument("--probe-every", type=int, help="development override")
    parser.add_argument("--evaluation-episodes", type=int, help="development override")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.report:
        print(report(args.report))
        return 0
    if args.out is None:
        parser.error("--out is required")
    protocol, _ = load_protocol(args.protocol)
    overrides = {}
    for key, value in (("training_episodes", args.episodes), ("probe_every", args.probe_every)):
        if value is not None:
            protocol[key] = value
            overrides[key] = value
    if args.evaluation_episodes is not None:
        protocol["evaluation"]["episodes_per_condition"] = args.evaluation_episodes
        overrides["evaluation.episodes_per_condition"] = args.evaluation_episodes
    args.out.mkdir(parents=True, exist_ok=False)
    used = args.out / "protocol.json"
    used.write_text(json.dumps(protocol, indent=2))
    (args.out / "overrides.json").write_text(
        json.dumps(
            {"overrides": overrides, "declared_protocol": not overrides, "argv": sys.argv}, indent=1
        )
    )
    seeds = args.seeds or protocol["seeds"]["development"]
    arms = args.arms or list(protocol["arms"])
    recipes = args.recipes or list(protocol["recipes"])
    jobs = [
        (str(used), seed, arm, recipe, str(args.out))
        for recipe in recipes
        for arm in arms
        for seed in seeds
    ]
    began = time.time()
    with get_context("spawn").Pool(max(1, min(args.workers, len(jobs)))) as pool:
        for result in pool.imap_unordered(run_founder, jobs):
            print(
                json.dumps(
                    {
                        "done": [result["recipe"], result["arm"], result["seed"]],
                        "seconds": result["seconds"],
                    }
                ),
                flush=True,
            )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA + "-manifest",
                "jobs": jobs,
                "wall_seconds": round(time.time() - began, 1),
                "sources": sources(),
                "threads": {
                    k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
            },
            indent=1,
        )
    )
    print(report(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
