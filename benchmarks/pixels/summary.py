"""The declared run's gates (cadence#284) from the lives' JSON lines.

    python benchmarks/pixels/summary.py results/*.jsonl

Per founder at the last budget: held-out accuracy, each invariance panel's gap below it, the
brain against the matched baseline and against the blank-screen control under reward.
Standard errors are binomial over a panel's frames.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np

GATE = {"held_out": 0.90, "founders": 5, "gap": 0.05, "baseline": 0.05, "sigmas": 2.0}
INVARIANCE = ("positions", "scales", "distractor")


def load(paths: list[str]) -> list[dict[str, Any]]:
    lives = []
    for path in paths:
        with open(path) as handle:
            for line in handle:
                life = json.loads(line)
                if "error" in life:
                    raise SystemExit(f"a life failed: {life['spec']} {life['error']}")
                lives.append(life)
    return lives


def key(life: dict[str, Any]) -> tuple[str, str, str, str]:
    shape = "x".join(str(k) for k in life["shape"])
    return life["condition"], shape, life["arm"], life.get("control") or "-"


def last(life: dict[str, Any]) -> dict[str, float]:
    return dict(life["readings"][-1]["panels"])


def se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 1e-12) / n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--panel", type=int, default=256)
    parser.add_argument("--brain", default="fields")
    parser.add_argument("--baseline", default="conv_matched")
    args = parser.parse_args()
    groups: dict[tuple[str, str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for life in load(args.paths):
        groups[key(life)][int(life["founder"])] = life
    print("| condition | frame | arm | control | parameters | frames | held_out | positions | "
          "scales | lighting | distractor | invariant (of founders) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in sorted(groups):
        lives = groups[k]
        panels = [last(life) for life in lives.values()]
        mean = {n: float(np.mean([p[n] for p in panels])) for n in panels[0]}
        invariant = sum(
            all(p["held_out"] - p[n] <= GATE["gap"] for n in INVARIANCE) for p in panels
        )
        any_life = next(iter(lives.values()))
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {any_life['parameters']:,} | "
              f"{any_life['readings'][-1]['frames']:,} | "
              + " | ".join(f"{mean[n]:.3f}" for n in
                           ("held_out", "positions", "scales", "lighting", "distractor"))
              + f" | {invariant} of {len(panels)} |")
    print()
    brain = groups.get(("lesson", "84x84x1", args.brain, "-"), {})
    base = groups.get(("lesson", "84x84x1", args.baseline, "-"), {})
    rows = []
    for founder in sorted(brain):
        p = last(brain[founder])
        gaps = {n: p["held_out"] - p[n] for n in INVARIANCE}
        b = last(base[founder])["held_out"] if founder in base else float("nan")
        rows.append((founder, p["held_out"], gaps, b))
        print(f"founder {founder}: held_out {p['held_out']:.3f} (gate .90: "
              f"{p['held_out'] >= GATE['held_out']}), gaps "
              + ", ".join(f"{n} {g:+.3f}" for n, g in gaps.items())
              + f", baseline {b:.3f}")
    if rows:
        passed = sum(r[1] >= GATE["held_out"] for r in rows)
        invariant = sum(all(g <= GATE["gap"] for g in r[2].values()) for r in rows)
        shortfall = float(np.mean([r[3] - r[1] for r in rows]))
        print(f"G1 lessons, held-out at .90: {passed} of {len(rows)} "
              f"({'met' if passed >= GATE['founders'] else 'not met'})")
        print(f"G3 invariance within .05 on {', '.join(INVARIANCE)}: {invariant} of {len(rows)} "
              f"({'met' if invariant >= GATE['founders'] else 'not met'})")
        print(f"G4 baseline minus brain, mean held-out: {shortfall:+.3f} "
              f"({'met' if shortfall <= GATE['baseline'] else 'not met'})")
    reward = groups.get(("reward", "84x84x1", args.brain, "-"), {})
    blank = groups.get(("reward", "84x84x1", args.brain, "blank"), {})
    if reward and blank:
        above = 0
        for founder in sorted(reward):
            p = last(reward[founder])["held_out"]
            q = last(blank[founder])["held_out"] if founder in blank else float("nan")
            margin = GATE["sigmas"] * float(np.hypot(se(p, args.panel), se(q, args.panel)))
            above += p - q > margin
            print(f"reward founder {founder}: {p:.3f} against blank {q:.3f} (margin {margin:.3f})")
        print(f"G2 reward above the blank screen by two standard errors: {above} of {len(reward)} "
              f"({'met' if above >= GATE['founders'] else 'not met'})")


if __name__ == "__main__":
    main()
