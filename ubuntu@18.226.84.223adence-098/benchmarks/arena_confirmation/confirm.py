"""Fixed native-arena confirmation with explicit outcome and saved-world custody."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

WORK = {"actual_sweeps": 0, "settle_calls": 0, "failed_settle_calls": 0}


class Refusal(RuntimeError):
    def __init__(self, reading: dict):
        super().__init__("a policy refused, including any successful automatic retry")
        self.partial = {"refusal": reading}


def call(policy: Any, observation: Any, reward: float | None) -> dict:
    before, refusals = WORK.copy(), policy.refusals
    reading = policy.moment(observation, reward)
    reading.update({key: WORK[key] - before[key] for key in WORK})
    reading["refusal_attempts"] = policy.refusals - refusals
    if reading["refused"] or reading["refusal_attempts"]:
        reading["last_error"] = getattr(policy, "last_error", None)
        raise Refusal(reading)
    return reading


def plain(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def restore(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict) and set(value) == {"array", "dtype", "shape"}:
        return np.asarray(value["array"], dtype=value["dtype"]).reshape(value["shape"])
    if isinstance(value, dict):
        return {k: restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [restore(v) for v in value]
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


class Life:
    """The unchanged native nursery world/reward, with its state made explicit."""

    def __init__(self, blueprint: Any, policy: Any, seed: int, relocate: int = 400):
        import numpy as np
        from arena import nursery
        from arena.world import Arena, Robot

        self.nursery, self.np = nursery, np
        self.blueprint, self.policy, self.seed = blueprint, policy, seed
        self.rng = np.random.default_rng(seed)
        self.robot, self.dummy = Robot.build(0, blueprint), Robot.build(1, nursery.DUMMY)
        self.arena = Arena(
            [self.robot, self.dummy],
            radius=10.0,
            zone_end=nursery.RING,
            zone_moments=1,
            seed=seed,
            spawn=False,
        )
        self.arena.t = 1
        self.robot.place_at(0.0, 0.0, self.rng.uniform(-math.pi, math.pi))
        nursery._place_dummy(self.arena, self.robot, self.dummy, self.rng)
        self.reward: float | None = None
        self.stood, self.relocate = 0, relocate

    def snapshot(self) -> dict[str, Any]:
        robots = []
        for robot in self.arena.robots:
            robots.append({**vars(robot), "blueprint": robot.blueprint.to_dict()})
        state = {k: v for k, v in vars(self.arena).items() if k not in {"robots", "rng"}}
        return plain(
            {
                "arena": state,
                "arena_rng": self.arena.rng.bit_generator.state,
                "robots": robots,
                "placement_rng": self.rng.bit_generator.state,
                "stood": self.stood,
                "relocate": self.relocate,
                "reward": self.reward,
                "seed": self.seed,
                "next_observation": self.nursery.observe(self.robot, self.arena),
            }
        )

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any], policy: Any) -> Life:
        from arena.parts import blueprint_from_dict

        value = restore(snapshot)
        blueprint = blueprint_from_dict(value["robots"][0]["blueprint"])
        life = cls(blueprint, policy, value["seed"], value["relocate"])
        vars(life.arena).update(value["arena"])
        for robot, saved in zip(life.arena.robots, value["robots"], strict=True):
            vars(robot).update({**saved, "blueprint": blueprint_from_dict(saved["blueprint"])})
        life.arena.rng.bit_generator.state = value["arena_rng"]
        life.rng.bit_generator.state = value["placement_rng"]
        life.stood, life.reward = value["stood"], value["reward"]
        if life.snapshot() != snapshot:
            raise RuntimeError("world snapshot did not restore exactly")
        return life

    def step(self, *, ablate: bool = False) -> dict[str, Any]:
        n, np = self.nursery, self.np
        x = n.observe(self.robot, self.arena)
        if ablate:
            x[:, :4] = 0.0
        reading = call(self.policy, x, self.reward)
        before = math.hypot(self.dummy.x - self.robot.x, self.dummy.y - self.robot.y)
        px, py = self.robot.x, self.robot.y
        outcome = self.arena.step({0: reading["commands"], 1: [1]})[0]
        self.robot.hp = self.blueprint.hp
        after = math.hypot(self.dummy.x - self.robot.x, self.dummy.y - self.robot.y)
        progress = before - after
        reward = float(np.clip(n.PROGRESS_PAY * progress, -n.PROGRESS_CAP, n.PROGRESS_CAP))
        reward += (outcome["dealt"] - outcome["taken"]) / n.DAMAGE_SCALE
        killed = not self.dummy.alive
        if killed:
            reward += n.KILL_PAY
        self.reward = reward
        self.stood += 1
        if killed or self.stood >= self.relocate:
            n._place_dummy(self.arena, self.robot, self.dummy, self.rng)
            self.stood = 0
        return {
            "commands": reading["commands"],
            "progress_m": progress,
            "dealt": outcome["dealt"],
            "taken": outcome["taken"],
            "kills": int(killed),
            "reward": reward,
            "travelled_m": math.hypot(self.robot.x - px, self.robot.y - py),
            "aroused": int(reading["aroused"]),
            "sweeps": reading["sweeps"],
            "learning_sweeps": reading["learning_sweeps"],
            "policy_ms": reading["ms"],
            "refused": int(reading["refused"]),
            "outside": int(outcome["outside"]),
            "burn": outcome["zone"],
            "moments": 1,
            **{key: reading[key] for key in WORK},
            "refusal_attempts": reading["refusal_attempts"],
        }


def run(life: Life, moments: int, *, ablate: bool = False, progress: Path | None = None) -> dict:
    began = time.perf_counter()
    totals: dict[str, float] = {}
    blocks = []
    for index in range(moments):
        try:
            row = life.step(ablate=ablate)
        except Refusal as error:
            error.partial.update(
                {
                    "completed_total": totals,
                    "blocks": blocks,
                    "seconds": time.perf_counter() - began,
                    "world": life.snapshot(),
                }
            )
            if progress is not None:
                write(progress, error.partial)
            raise
        for name, value in row.items():
            if name != "commands":
                totals[name] = totals.get(name, 0) + value
        if (index + 1) % 1000 == 0:
            blocks.append({"moment": index + 1, "cumulative": totals.copy()})
            if progress is not None:
                write(progress, {"blocks": blocks, "seconds": time.perf_counter() - began})
                print(
                    json.dumps({"moment": index + 1, "progress_m": totals["progress_m"]}),
                    flush=True,
                )
    return {"total": totals, "blocks": blocks, "seconds": time.perf_counter() - began}


def continuation(life: Life, checkpoint: Path, folder: Path, moments: int) -> dict:
    import numpy as np
    from arena.brain import RobotBrain

    began = time.perf_counter()
    snapshot = life.snapshot()
    wrapper = {
        "owed": life.policy.owed,
        "refusals": life.policy.refusals,
        "moments": life.policy.moments,
        "last_error": life.policy.last_error,
    }
    write(folder / "world.json", {"world": snapshot, "wrapper": wrapper})
    loaded = RobotBrain.load(life.blueprint, checkpoint)
    # Exercise the serialized JSON, not just an in-memory clone.
    saved = json.loads((folder / "world.json").read_text())
    vars(loaded).update(saved["wrapper"])
    other = Life.from_snapshot(saved["world"], loaded)
    records = []
    for _ in range(moments):
        left, right = life.step(), other.step()
        comparable_left = {k: v for k, v in left.items() if k != "policy_ms"}
        comparable_right = {k: v for k, v in right.items() if k != "policy_ms"}
        if comparable_left != comparable_right or life.snapshot() != other.snapshot():
            raise RuntimeError("saved continuation differs in action, reward, work or physics")
        records.append({"unbroken": left, "reloaded": right})
    left_path, right_path = folder / "fork-original.npz", folder / "fork-loaded.npz"
    life.policy.save(left_path)
    other.policy.save(right_path)
    with (
        np.load(left_path, allow_pickle=False) as left,
        np.load(right_path, allow_pickle=False) as right,
    ):
        if set(left.files) != set(right.files) or any(
            not np.array_equal(left[name], right[name]) for name in left.files
        ):
            raise RuntimeError("saved continuation brain arrays differ")
    return {
        "moments_per_fork": moments,
        "equal": True,
        "records": records,
        "seconds": time.perf_counter() - began,
        "outcomes_received_after": int(loaded.brain.arousal.rewards),
        "checkpoint_arrays_equal": True,
    }


def fingerprint(blueprint: Any, checkpoint: Path) -> dict:
    import numpy as np
    from arena.brain import FrozenPolicy
    from arena.probe import situations

    from cadence import Brain

    began = time.perf_counter()
    rows, tables = [], []
    for label, x in situations(blueprint):
        policy = FrozenPolicy(blueprint, Brain.load(checkpoint))
        reading = call(policy, x, None)
        state = policy.brain.basal_ganglia.state
        probability = np.asarray(policy.brain.basal_ganglia.probabilities(state))
        tables.append(probability)
        rows.append({"situation": label, **reading})
    stack = np.stack(tables)
    return {
        "rows": rows,
        "distinct": len({tuple(row["commands"]) for row in rows}),
        "sensitivity": float(np.mean(stack.max(axis=0) - stack.min(axis=0))),
        "sweeps": sum(row["sweeps"] for row in rows),
        "learning_sweeps": sum(row["learning_sweeps"] for row in rows),
        **{key: sum(row[key] for row in rows) for key in WORK},
        "seconds": time.perf_counter() - began,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refusal-smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    if args.refusal_smoke and not args.smoke:
        parser.error("refusal injection is only available outside the confirmation cohort")
    case = next(item for item in protocol["cases"] if item["id"] == args.case)
    core, arena = args.cadence.resolve(), args.arena.resolve()
    sys.path[:0] = [str(core / "src"), str(arena)]
    import numpy as np
    from arena import nursery
    from arena.brain import RobotBrain, genes_of, make_policy
    from arena.parts import stock_designs

    import cadence
    from cadence import NeuralGraph

    if Path(cadence.__file__).resolve().parents[2] != core:
        raise RuntimeError("wrong Cadence checkout imported")
    if Path(nursery.__file__).resolve().parents[1] != arena:
        raise RuntimeError("wrong arena checkout imported")

    def sources() -> dict:
        return {
            name: {str(p.relative_to(root)): digest(p) for p in sorted(directory.rglob("*.py"))}
            for name, root, directory in (
                ("cadence", core, core / "src/cadence"),
                ("arena", arena, arena / "arena"),
            )
        }

    def revision(path: Path) -> str:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()

    folder = args.output / (("smoke-" if args.smoke else "") + case["id"])
    folder.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    original = sources()
    expected = json.loads(Path(__file__).with_name("sources.json").read_text())
    original_settle = NeuralGraph.settle_batch

    def measured_settle(graph: Any, *positional: Any, **keywords: Any) -> Any:
        WORK["settle_calls"] += 1
        try:
            result = original_settle(graph, *positional, **keywords)
        except Exception:
            WORK["failed_settle_calls"] += 1
            raise
        WORK["actual_sweeps"] += int(result.steps)
        return result

    NeuralGraph.settle_batch = measured_settle
    receipt: dict[str, Any] = {
        "protocol": protocol,
        "case": case,
        "smoke": args.smoke,
        "status": "running",
        "producer_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(protocol_path),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "numpy": np.__version__,
        "cadence_commit": revision(core),
        "arena_commit": revision(arena),
        "platform": platform.platform(),
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "source_sha256": original,
        "source_admitted": original == expected["source_sha256"],
        "admission_sha256": digest(Path(__file__).with_name("sources.json")),
    }
    write(folder / "receipt.json", receipt)
    try:
        if not receipt["source_admitted"]:
            raise RuntimeError("runtime source does not match the fixed confirmation admission")
        blueprint = replace(
            next(body for body in stock_designs() if body.name == case["body"]),
            seed=900001 if args.smoke else case["founder"],
        )
        train_n = 128 if args.smoke else protocol["training_moments"]
        eval_n = 64 if args.smoke else protocol["evaluation_moments"]
        world = 900002 if args.smoke else case["training_world"]
        receipt.update({"blueprint": blueprint.to_dict(), "resolved_genes": genes_of(blueprint)})
        policy = RobotBrain.newborn(blueprint)
        if args.refusal_smoke:
            policy.brain.learner.config = replace(
                policy.brain.learner.config,
                free_steps=1,
                tolerance=1e-30,
            )
        life = Life(blueprint, policy, world, protocol["relocate"])
        receipt["training"] = run(life, train_n, progress=folder / "progress.json")
        checkpoint = folder / "trained.npz"
        policy.save(checkpoint)
        receipt["custody"] = {
            "executed_actions": train_n,
            "outcomes_received": int(policy.brain.arousal.rewards),
            "pending_actual_reward": life.reward,
            "pending_done": False,
            "pending_action": policy.has_pending(),
            "wrapper_owed": policy.owed,
            "description": "Final executed action awaits this real outcome with next observation",
        }
        if policy.brain.arousal.rewards != train_n - 1 or not policy.has_pending() or policy.owed:
            raise RuntimeError("training outcome custody differs from the declared schedule")
        if int(sum(policy.brain.arousal.sweeps.values())) != receipt["training"]["total"]["sweeps"]:
            raise RuntimeError("training settle work does not reconcile")
        if policy.brain.arousal.learning_sweeps != receipt["training"]["total"]["learning_sweeps"]:
            raise RuntimeError("training learning work does not reconcile")
        receipt["training"]["brain"] = policy.describe()
        receipt["continuation"] = continuation(
            life,
            checkpoint,
            folder,
            protocol["continuation_moments"],
        )
        receipt["fingerprint"] = fingerprint(blueprint, checkpoint)
        receipt["evaluations"] = []
        for seed in protocol["evaluation_worlds"]:
            evaluation = {"world": seed}
            for name, kind, path, ablate in (
                ("trained", "frozen", checkpoint, False),
                ("newborn", "frozen", None, False),
                ("random", "random", None, False),
                ("ablated", "frozen", checkpoint, True),
            ):
                trial = Life(blueprint, make_policy(blueprint, kind, path), seed)
                evaluation[name] = run(trial, eval_n, ablate=ablate)
                if evaluation[name]["total"]["learning_sweeps"]:
                    raise RuntimeError("a frozen or random evaluation performed learning")
            receipt["evaluations"].append(evaluation)
        if args.smoke:
            start = time.perf_counter()
            native = nursery.run_nursery(blueprint, moments=train_n, seed=world, block=train_n)
            total = receipt["training"]["total"]
            for metric, decimals in (("progress_m", 3), ("dealt", 2), ("taken", 2), ("reward", 3)):
                if round(total[metric], decimals) != native["total"][metric]:
                    raise RuntimeError(f"instrument differs from native nursery: {metric}")
            if total["kills"] != native["total"]["kills"]:
                raise RuntimeError("instrument differs from native nursery: kills")
            receipt["native_smoke_control"] = {
                "report": native,
                "seconds": time.perf_counter() - start,
                "equal": True,
            }
        receipt["status"] = "complete"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, Refusal):
            receipt["partial"] = error.partial
        raise
    finally:
        NeuralGraph.settle_batch = original_settle
        receipt["work_all_sections"] = WORK.copy()
        receipt["source_unchanged_after"] = sources() == original
        receipt["producer_unchanged_after"] = (
            digest(Path(__file__)) == receipt["producer_sha256"]
            and digest(protocol_path) == receipt["protocol_sha256"]
            and digest(Path(__file__).with_name("sources.json")) == receipt["admission_sha256"]
        )
        if not receipt["source_unchanged_after"]:
            receipt["status"] = "failed"
            receipt["source_error"] = "runtime source changed during the attempt"
        if not receipt["producer_unchanged_after"]:
            receipt["status"] = "failed"
            receipt["producer_error"] = "producer/protocol changed during the attempt"
        receipt["artifacts"] = {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(folder.iterdir())
            if path.name != "receipt.json"
        }
        receipt["total_seconds"] = time.perf_counter() - began
        write(folder / "receipt.json", receipt)
    if receipt["status"] != "complete":
        raise RuntimeError("the attempt did not complete with stable sources")


if __name__ == "__main__":
    main()
