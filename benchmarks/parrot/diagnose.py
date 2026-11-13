"""Development reader: how good the teacher's lessons are, and how well the distance neurons
fit them (``python -m benchmarks.parrot.diagnose --founder 900``)."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from benchmarks.parrot.chamber import place_things, step, truth_distance, truths
from cadence.parrot.brain import Parrot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--founder", type=int, default=900)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--moments", type=int, default=50)
    parser.add_argument("--genes", type=str, default="{}")
    args = parser.parse_args()
    rng = np.random.default_rng(np.random.SeedSequence([args.founder, 332, 1]))
    parrot = Parrot(json.loads(args.genes), seed=args.founder)
    rows = []
    pairs = []
    for k in range(args.scenes):
        scene = place_things(rng, int(rng.integers(3, 7)))
        parrot.reset()
        parrot.taught = []
        parrot.record = []
        for _ in range(args.moments):
            step(parrot, scene, learn=True)
        t = parrot.taught
        for i in range(0, len(t), 2):
            a, b = t[i], t[i + 1]
            same = a["truth"] == b["truth"] and a["truth"] >= 0
            for x in (a, b):
                true = (truth_distance(scene, x["truth"], x["origin"]) if x["truth"] >= 0
                        else math.nan)
                rows.append({"scene": k, "same": same, "binocular": x["binocular"],
                             "angle": x["angle"], "gap": x["gap"], "target": x["target"],
                             "true": true})
        recs = [r for r in parrot.record if r["truth"] >= 0]
        for r in truths(scene, [r for r in recs if r["eye"] == 2]):
            pairs.append((k, r["true_disparity"], r["disparity"], r["true"], r["distance"]))
        both = [abs(math.log(r["distance"] / truth_distance(scene, r["truth"], r["origin"])))
                for r in recs if r["eye"] == 2 and np.isfinite(r["distance"])]
        one = [abs(math.log(r["distance"] / truth_distance(scene, r["truth"], r["origin"])))
               for r in recs if r["eye"] != 2 and np.isfinite(r["distance"])]
        both_m = np.median(both) if both else math.nan
        one_m = np.median(one) if one else math.nan
        print(f"scene {k}: lessons {len(t)} reads both {len(both)} {both_m:.3f}"
              f" one {len(one)} {one_m:.3f}", flush=True)
    same = np.array([r["same"] for r in rows])
    err = np.array([abs(math.log(r["target"] / r["true"])) if np.isfinite(r["true"])
                    else math.nan for r in rows])
    ang = np.array([r["angle"] for r in rows])
    bino = np.array([r["binocular"] for r in rows])
    print("lessons", len(rows), "same thing", same.mean().round(3))
    for lo, hi in ((1, 2), (2, 4), (4, 8), (8, 30)):
        m = same & (ang >= lo) & (ang < hi)
        print(f"  parallax {lo}-{hi} deg: {m.sum()} lessons, target error median "
              f"{np.nanmedian(err[m]) if m.any() else math.nan:.3f}")
    print("  binocular lessons", int((bino & same).sum()), "monocular", int((~bino & same).sum()))
    p = np.array(pairs)
    if len(p):
        derr = p[:, 2] - p[:, 1]
        print("both-eye sightings", len(p), "disparity peak - true: median", np.round(
            np.median(derr), 2), "abs median", np.round(np.median(np.abs(derr)), 2),
            "share within 1 deg", np.round(np.mean(np.abs(derr) < 1), 3))
        rerr = np.abs(np.log(p[:, 4] / p[:, 3]))
        good = np.abs(derr) < 1
        print("  read error where the peak is within 1 deg", np.round(np.median(rerr[good]), 3),
              "elsewhere", np.round(np.median(rerr[~good]), 3) if (~good).any() else None)
        for row in p[np.argsort(-np.abs(derr))][:12]:
            print("  scene {:.0f} true disp {:.2f} peak {:.2f} true {:.3f} read {:.3f}".format(
                *row))
    # the field on binocular codes: read against truth
    print("wrong-thing lessons target/true", np.round(
        [r["target"] / r["true"] for r in rows
         if not r["same"] and np.isfinite(r["true"])][:20], 2))


if __name__ == "__main__":
    main()
