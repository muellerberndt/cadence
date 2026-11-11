"""Frozen newborn controls for the fixed three-founder arena policy comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the previous receipt; choose another output path")
    core, arena = args.cadence.resolve(), args.arena.resolve()
    sys.path[:0] = [str(core / "src"), str(arena)]
    import arena.nursery as nursery
    from arena.parts import stock_designs

    import cadence

    if Path(cadence.__file__).resolve().parents[2] != core:
        raise RuntimeError("imported Cadence does not match the declared checkout")
    if Path(nursery.__file__).resolve().parents[1] != arena:
        raise RuntimeError("imported arena does not match the declared checkout")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def sources() -> dict[str, dict[str, str]]:
        return {
            "cadence": {
                str(p.relative_to(core)): digest(p)
                for p in sorted((core / "src/cadence").rglob("*.py"))
            },
            "arena": {
                str(p.relative_to(arena)): digest(p)
                for p in sorted((arena / "arena").rglob("*.py"))
            },
        }

    def revision(path: Path) -> str:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()

    original = sources()
    receipt = {
        "protocol": "arena-policy-frozen-newborn/1",
        "producer": {"file": Path(__file__).name, "sha256": digest(Path(__file__))},
        "command": [sys.executable, *sys.argv],
        "cadence_commit": revision(core),
        "arena_commit": revision(arena),
        "source_sha256": original,
        "python": sys.version,
        "conditions": {
            "policy": "frozen", "brain_path": None, "moments": 1000,
            "block": 500, "relocate": 400, "save_to": None,
            "training_moments": 0, "stage": None,
        },
        "evaluations": [],
        "limitations": [
            "Development comparator added after inspecting the trained evaluations",
            "Fixed founder and world seeds match those trained evaluations; no tuning",
            "Fresh composed brains act greedily without learning or loading prior checkpoints",
            "Actions can change dummy relocation times despite matched initial world seeds",
        ],
    }
    for founder, world in ((11, 19), (12, 19), (13, 20)):
        blueprint = replace(stock_designs()[0], seed=founder)
        result = nursery.run_nursery(
            blueprint, moments=1000, seed=world, policy="frozen", brain_path=None,
            block=500, relocate=400, save_to=None,
        )
        assert all(block["learning_sweeps"] == 0 for block in result["blocks"])
        receipt["evaluations"].append({"brainseed": founder, "evalseed": world, "frozen": result})
        print(json.dumps({"brainseed": founder, "evalseed": world, **result["total"]}), flush=True)
    if sources() != original:
        raise RuntimeError("source changed during the comparison")
    receipt["source_unchanged_after"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
