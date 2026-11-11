"""Read-only bounded continuation of the six reported arena checkpoints.

Run with PYTHONPATH selecting an exact Cadence source and arena root:
    python compare_arena.py --output result.json [--policy random] [--need-zero]
No training campaign, checkpoint writes, brain resets or stage heat changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cadence
import cadence.arousal
from arena.league import League
from arena.royale import Fighter, run_royale


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--policy", choices=("brain", "random"), default="brain")
parser.add_argument("--need-zero", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
league = League(root / "league-evolved")
names = [n for n, _ in sorted(league.data["robots"].items(), key=lambda kv: -kv[1]["elo"])[:6]]
paths = {n: league.brain_path(n) for n in names}
before = {n: digest(p) for n, p in paths.items()}
source = Path(cadence.__file__).resolve().parents[2]
result = {
    "format": "cadence-robot-arena/issue158-development/1",
    "scope": "Exploratory paired continuation; no claim of acquired fighting skill or fresh confirmation.",
    "cadence_source": str(source),
    "cadence_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
    "arousal_sha256": digest(cadence.arousal.__file__),
    "arena_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "checkpoint_sha256": before,
    "policy": args.policy,
    "stage": {"need": 0} if args.need_zero else None,
    "duration": 600,
    "seeds": [77, 78, 79],
    "runs": [],
}
for seed in result["seeds"]:
    fighters = [Fighter(n, league.blueprint(n), args.policy, paths[n] if args.policy == "brain" else None, stage=result["stage"]) for n in names]
    outcome = run_royale(fighters, seed=seed, duration=600, zone_moments=500, workers=0, record=False, save=False)
    result["runs"].append({"seed": seed, "moments": outcome["moments"], "fighters": [dict(name=f.name, **f.stats) for f in fighters]})
    print(seed, [(f.name, f.stats["aroused_share"], f.stats["closing_share"], f.stats["refused"]) for f in fighters], flush=True)
after = {n: digest(p) for n, p in paths.items()}
assert before == after, "A saved league brain was changed"
result["checkpoints_unchanged"] = True
args.output.write_text(json.dumps(result, indent=2) + "\n")
