"""The declared run's gates (cadence#284) from the lives' JSON lines.

    python benchmarks/pixels/summary.py results/*.jsonl

Per founder at the last budget: held-out accuracy, each invariance panel's gap below it, the
brain against the matched baseline and against the blank-screen control under reward.
Standard errors are binomial over a panel's frames. ``--against ARM`` reads ``--brain`` beside
another brain arm on the founders both lived: each panel's gap below held-out, founder by
founder, and the paired difference with its standard error over the founders.
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


PANELS = "panels"  # or "panels_256", the panels of the first declaration, where a life read both


def last(life: dict[str, Any]) -> dict[str, float]:
    return dict(life["readings"][-1][PANELS])


def se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 1e-12) / n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--panel", type=int, default=256, help="frames per panel read")
    parser.add_argument("--declared", action="store_true",
                        help="read the 256-frame panels of lives that hold both")
    parser.add_argument("--founders", type=int, default=5, help="founders a gate needs")
    parser.add_argument("--brain", default="fields")
    parser.add_argument("--baseline", default="conv_matched")
    parser.add_argument("--blank", default=None,
                        help="the arm whose blank-screen lives are the control (the brain)")
    parser.add_argument("--against", default=None,
                        help="a brain arm the brain is read beside, paired by founder")
    args = parser.parse_args()
    global PANELS
    if args.declared:
        PANELS = "panels_256"
    GATE["founders"] = args.founders
    groups: dict[tuple[str, str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for life in load(args.paths):
        groups[key(life)][int(life["founder"])] = life
    print("| condition | frame | arm | control | parameters | MACs per lesson frame | "
          "MACs per read frame | frames | held_out | positions | "
          "scales | lighting | distractor | invariant (of founders) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in sorted(groups):
        lives = groups[k]
        panels = [last(life) for life in lives.values()]
        mean = {n: float(np.mean([p[n] for p in panels])) for n in panels[0]}
        invariant = sum(
            all(p["held_out"] - p[n] <= GATE["gap"] for n in INVARIANCE) for p in panels
        )
        any_life = next(iter(lives.values()))
        work = [life["macs"] for life in lives.values() if life.get("macs")]
        macs = " | ".join(
            f"{float(np.mean([w[name] for w in work])):.3g}" if work else "-"
            for name in ("lesson", "read")
        )
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {any_life['parameters']:,} | {macs} | "
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
        for name in INVARIANCE:
            within = sum(r[2][name] <= GATE["gap"] for r in rows)
            print(f"{name} alone within .05 of held-out: {within} of {len(rows)}")
    other = groups.get(("lesson", "84x84x1", args.against, "-"), {}) if args.against else {}
    paired = sorted(set(brain) & set(other))
    if paired:
        print()
        print(f"{args.brain} beside {args.against}, {len(paired)} founders (gap: held-out "
              "minus the panel; difference: the gap the brain closes)")
        for name in ("held_out", *INVARIANCE):
            mine = np.array([last(brain[f])[name] for f in paired])
            theirs = np.array([last(other[f])[name] for f in paired])
            if name == "held_out":
                difference = mine - theirs
                label = "held_out, brain minus the other"
            else:
                held = np.array([last(brain[f])["held_out"] for f in paired])
                held_other = np.array([last(other[f])["held_out"] for f in paired])
                difference = (held_other - theirs) - (held - mine)
                label = f"{name} gap, the other minus the brain"
            error = float(np.std(difference, ddof=1) / np.sqrt(len(paired))) if len(
                paired) > 1 else float("nan")
            print(f"{label}: {float(np.mean(difference)):+.4f} (standard error {error:.4f}; "
                  f"{float(np.mean(mine)):.3f} against {float(np.mean(theirs)):.3f})")
    reward = groups.get(("reward", "84x84x1", args.brain, "-"), {})
    blank = groups.get(("reward", "84x84x1", args.blank or args.brain, "blank"), {})
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
