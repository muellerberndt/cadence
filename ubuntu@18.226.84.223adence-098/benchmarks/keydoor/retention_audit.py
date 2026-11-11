"""Development-only preservation of the stable door skill across a hidden reversal.

The original key-door loop runs unchanged. At declared probe boundaries, a saved copy
starts a fresh stream and greedily faces a door while holding a key. No outcome is
delivered and no learning is allowed. This tests the same usable skill before and after
rule B; it does not ask a frozen policy to infer the unannounced return to rule A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import key_door as kd
import numpy as np

SCHEMA = "key-door-retention/1"
PROTOCOL = Path(__file__).with_name("protocol-retention-development.json")
BOUNDARIES = ("initial", "after_A", "after_B")


def sources() -> list[tuple[str, Path]]:
    return [*kd.sources(), ("retention_audit.py", Path(__file__).resolve())]


def checkpoint_digests(path: Path) -> tuple[str, str]:
    """Ignore archive timestamps; compare all saved state and durable learned state."""
    with np.load(path, allow_pickle=False) as saved:
        if "brain" in saved.files:  # one group per memory, the format since 0.85
            brain = json.loads(str(saved["brain"]))
            store, hippocampus = "memories/hippocampus/", brain["memories"].get("hippocampus")
        else:
            brain = json.loads(str(saved["generic"]))
            store, hippocampus = "episodic/", brain["hippocampus"]
        arrays = {
            name: {
                "dtype": str(saved[name].dtype),
                "shape": list(saved[name].shape),
                "sha256": hashlib.sha256(saved[name].tobytes()).hexdigest(),
            }
            for name in saved.files
        }
        durable = {
            name: value
            for name, value in arrays.items()
            if name
            in (
                "efficacy",
                "log_gain",
                "bias",
                "velocity",
                "velocity_bias",
                "second_moment",
                "second_moment_bias",
                "actor/w_critic",
            )
            or name.startswith(store)
        }
        learner = json.loads(str(saved["meta"]))
        durable["updates"] = [brain["updates"], learner["updates"], learner["contrast_updates"]]
        durable["critic_bias"] = brain["b_critic"]
        durable["hippocampus"] = hippocampus
    return kd.canonical_sha256(arrays), kd.canonical_sha256(durable)


def stable_door(life: kd.BrainLife, directory: Path, boundary: str) -> dict[str, Any]:
    """Read a private branch, retain its checkpoints, and charge every saved copy/solve."""
    directory.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    before = life.brain.save(directory / f"{boundary}-source.npz")
    life.work["checkpoints"] += 1
    original_state, original_durable = checkpoint_digests(before)
    branch = kd.cd.Brain.load(before)
    life.work["checkpoints"] += 1
    life._count_memory(branch, "probe_memory_reads")
    branch.reset()
    try:
        action = int(branch.act(kd.observe(kd.DOOR, True, life.pouch), greedy=True)[0])
    except Exception:
        life._charge_refusal(branch)
        raise
    settlement = branch.last_settlement
    assert settlement is not None
    life.work["probe_sweeps"] += int(settlement["steps"])
    life.work["probes"] += 1
    probability = float(
        branch.basal_ganglia.probabilities(branch.basal_ganglia.state)[0, kd.INTERACT]
    )
    after_branch = branch.save(directory / f"{boundary}-branch.npz")
    after_source = life.brain.save(directory / f"{boundary}-source-after.npz")
    life.work["checkpoints"] += 2
    after_state, _ = checkpoint_digests(after_source)
    _, branch_durable = checkpoint_digests(after_branch)
    if original_state != after_state or original_durable != branch_durable:
        raise RuntimeError("the private stable-skill assay changed living or learned state")
    return {
        "boundary": boundary,
        "action": action,
        "food": int(action == kd.INTERACT),
        "policy_interact": probability,
        "observation": kd.observe(kd.DOOR, True, life.pouch).tolist(),
        "holding": True,
        "done": True,
        "outcomes_delivered": 0,
        "source_state_sha256": original_state,
        "source_after_state_sha256": after_state,
        "durable_before_sha256": original_durable,
        "durable_after_sha256": branch_durable,
        "work": {
            "checkpoints": 4,
            "greedy_calls": 1,
            "sweeps": int(settlement["steps"]),
            "seconds": time.perf_counter() - began,
        },
        "artifacts": [
            {
                "path": p.relative_to(directory.parent).as_posix(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in (before, after_branch, after_source)
        ],
    }


def run_life(
    arm: str, seed: int, delay: int, protocol: dict[str, Any], directory: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Serial adapter around the unchanged chamber; no observation or reward is substituted."""
    original_factory = kd.make_life
    readings: list[dict[str, Any]] = []
    per_phase = (protocol["episodes"] - 1) // protocol["probe_every"] + 2
    boundaries = {1: "initial", per_phase: "after_A", 2 * per_phase: "after_B"}

    def observed_factory(*args: Any) -> kd.BrainLife:
        life = original_factory(*args)
        if not isinstance(life, kd.BrainLife):
            raise ValueError("this development assay declares only the live and blind bodies")
        original_probe = life.probe
        probes = 0

        def observed_probe() -> Any:
            nonlocal probes
            answer = original_probe()
            probes += 1
            if probes in boundaries:
                readings.append(stable_door(life, directory, boundaries[probes]))
            return answer

        life.probe = observed_probe
        return life

    kd.make_life = observed_factory
    try:
        row = kd.run_life(arm, seed, delay, protocol)
    finally:
        kd.make_life = original_factory
    return row, readings


def verify(path: Path, *, current: bool = False, artifacts: bool = False) -> tuple[bool, str]:
    def check(body: dict[str, Any]) -> str | None:
        if body["confirmation"] is not False:
            return "this is a development assay, not confirmation"
        chamber = body["chamber"]
        projected = {"kind": kd.SCHEMA, "body": chamber, "source": stored["source"]}
        projected["digest"] = kd.canonical_sha256(projected)
        with tempfile.TemporaryDirectory() as temporary:
            plain = Path(temporary) / "chamber.json"
            plain.write_text(kd.canonical_json(projected) + "\n")
            valid, reason = kd.verify(plain)
            if not valid:
                return reason
        if len(body["assays"]) != len(chamber["rows"]):
            return "the assay census differs from the continuing lives"
        for row, assays in zip(chamber["rows"], body["assays"], strict=True):
            if "error" in row:
                return "a development life or its private assay failed"
            if [a["boundary"] for a in assays] != list(BOUNDARIES):
                return "the three declared stable-skill boundaries are missing"
            for assay in assays:
                expected = kd.observe(kd.DOOR, True, row["arm"] != "blind").tolist()
                if (
                    assay["observation"] != expected
                    or assay["holding"] is not True
                    or assay["done"] is not True
                ):
                    return "the private branch did not execute the declared keyed-door problem"
                if (
                    assay["action"] not in (0, 1)
                    or assay["food"] != int(assay["action"] == kd.INTERACT)
                    or not 0 <= assay["policy_interact"] <= 1
                ):
                    return "the stable-skill response or actual food outcome is invalid"
                if (
                    assay["outcomes_delivered"] != 0
                    or assay["source_state_sha256"] != assay["source_after_state_sha256"]
                    or assay["durable_before_sha256"] != assay["durable_after_sha256"]
                ):
                    return "the assay changed source custody or learned state"
                work = assay["work"]
                if (
                    work["checkpoints"] != 4
                    or work["greedy_calls"] != 1
                    or type(work["sweeps"]) is not int
                    or work["sweeps"] <= 0
                    or not np.isfinite(work["seconds"])
                    or work["seconds"] < 0
                ):
                    return "invalid stable-skill work counters"
                if artifacts:
                    if len(assay["artifacts"]) != 3:
                        return "the source, branch and unchanged source checkpoints are required"
                    checkpoints = []
                    for artifact in assay["artifacts"]:
                        target = (path.parent / artifact["path"]).resolve()
                        if (
                            not target.is_relative_to(path.parent.resolve())
                            or hashlib.sha256(target.read_bytes()).hexdigest() != artifact["sha256"]
                        ):
                            return "an assay checkpoint is missing or differs"
                        checkpoints.append(target)
                    original, branch, after = map(checkpoint_digests, checkpoints)
                    if (
                        original[0] != assay["source_state_sha256"]
                        or after[0] != assay["source_after_state_sha256"]
                        or original[1] != assay["durable_before_sha256"]
                        or branch[1] != assay["durable_after_sha256"]
                    ):
                        return "the checkpoint state does not match the preservation readings"
                    restored = kd.cd.Brain.load(checkpoints[1])
                    if restored.basal_ganglia.state.steps != work["sweeps"]:
                        return "stable-skill sweeps differ from the saved branch"
                    policy = restored.basal_ganglia.probabilities(restored.basal_ganglia.state)[0]
                    if (
                        int(np.argmax(policy)) != assay["action"]
                        or float(policy[kd.INTERACT]) != assay["policy_interact"]
                    ):
                        return "the executed greedy action does not match its saved branch"
        return None

    try:
        stored = json.loads(path.read_text())
        if stored["kind"] != SCHEMA:
            return False, "wrong stable-skill receipt kind"
        return kd.Receipt.verify(path, sources=sources() if current else None, check=check)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return False, f"cannot verify the stable-skill receipt: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--current", action="store_true")
    parser.add_argument("--artifacts", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        valid, reason = verify(args.verify, current=args.current, artifacts=args.artifacts)
        print(json.dumps({"verified": valid, "reason": reason}))
        return 0 if valid else 1
    if args.out is None:
        parser.error("--out is required for a development run")
    frozen = args.protocol.read_bytes()
    protocol = json.loads(frozen)
    if protocol["rules"] != ["chest", "lamp"] or protocol["assay"]["boundaries"] != list(
        BOUNDARIES
    ):
        parser.error("the development assay declares chest, lamp and three stable-skill boundaries")
    arms, seeds, delays = (
        protocol["assay"]["arms"],
        protocol["seeds"]["development"],
        protocol["delays"],
    )
    if arms != ["live", "blind"]:
        parser.error("the declared body comparison is live and blind")
    kd.validate_plan(protocol, arms, seeds, delays)
    destinations = [
        args.out,
        *(
            args.out.parent / f"{arm}-d{delay}-s{seed}"
            for arm in arms
            for delay in delays
            for seed in seeds
        ),
    ]
    if any(path.exists() for path in destinations):
        parser.error(
            "the receipt and founder directories must be new; existing evidence is retained"
        )
    manifest = kd.source_manifest(sources())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, assays = [], []
    for arm in arms:
        for delay in delays:
            for seed in seeds:
                row, readings = run_life(
                    arm, seed, delay, protocol, args.out.parent / f"{arm}-d{delay}-s{seed}"
                )
                rows.append(row)
                assays.append(readings)
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "seed": seed,
                            "delay": delay,
                            "error": row.get("error"),
                            "stable_food": [r["food"] for r in readings],
                        }
                    ),
                    flush=True,
                )
    chamber = {
        "instrument_revision": kd.INSTRUMENT_REVISION,
        "cadence": kd.cd.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "protocol_sha256": hashlib.sha256(frozen).hexdigest(),
        "protocol_source": frozen.decode(),
        "protocol": protocol,
        "genes_override": None,
        "frozen_protocol": False,
        "arms": arms,
        "seeds": seeds,
        "delays": delays,
        "rows": rows,
        "gates": kd.audit_gates(rows, protocol, eligible=False),
    }
    receipt = kd.Receipt.build(
        SCHEMA, {"confirmation": False, "chamber": chamber, "assays": assays}, sources()
    )
    if receipt.source != manifest:
        raise RuntimeError("a source changed during the development run")
    receipt.write(args.out)
    valid, reason = verify(args.out, current=True, artifacts=True)
    print(json.dumps({"verified": valid, "reason": reason, "receipt": str(args.out)}))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
