"""Protocol 1 of #332, the gated run: every arm on the twelve fresh founders, and the gates.

``python -m benchmarks.parrot.gated --out runs/gated`` lives each (arm, founder) in its own
process (``founder`` of the chamber) and writes ``<out>/<arm>/founder-<seed>.json``;
``--read`` reads the folder and writes ``<out>/readings.json`` and ``<out>/readings.md``: the
table of every arm with learned parameters and multiply-accumulates, and each gate of
``PROTOCOL.md`` with its founders.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.parrot.chamber import founder

HERE = Path(__file__).resolve().parent
FOUNDERS = list(range(33200000, 33200012))
ARMS = ("selected", "hand-set", "random-wiring", "random-search", "cnn", "vit", "cnn10", "vit10")
WEIGHTS = {"m1": 1.0, "m2": 1.0, "m3": 0.5, "m4": 1.0}


def genes_of(arm: str) -> dict[str, Any]:
    if arm == "selected":
        return dict(json.loads((HERE / "selected.json").read_text())["genes"])
    if arm == "random-search":
        return dict(json.loads((HERE / "random_search.json").read_text())["genes"])
    if arm == "random-wiring":
        return {"wiring": "random"}
    return {}


def _live(job: tuple[str, int, str]) -> str:
    arm, seed, out = job
    target = Path(out) / arm / f"founder-{seed}.json"
    if target.exists():
        return str(target)
    control = arm if arm in ("cnn", "vit", "cnn10", "vit10") else None
    result = founder(seed, genes_of(arm) if control is None else {},
                     silence=arm in ("selected", "hand-set"), control=control)
    result["arm"] = arm
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    part.write_text(json.dumps(result, default=float))
    part.rename(target)
    return str(target)


def terms(r: dict[str, Any]) -> dict[str, float]:
    """The fitness terms of evolve.py from a founder's readings."""
    t = r["tests"]
    behind = t["gap_behind"] if np.isfinite(t["gap_behind"]) else t["gap_seen"]
    looked = [v for v in (t["E3_looked"], t["E3_looked_one"]) if np.isfinite(v)]
    return {"m1": 100.0 * t["gap_seen"], "m2": 100.0 * behind,
            "m3": 1.0 - (t["E2_one"] if np.isfinite(t["E2_one"]) else 0.0),
            "m4": float(np.mean(looked)) if looked else 1.0,
            "cost": float(sum(r["work"]["macs"].values()))}


def fitness(r: dict[str, Any], hand: dict[str, Any]) -> float:
    a, h = terms(r), terms(hand)
    total = sum(w * a[k] / max(h[k], 1e-9) for k, w in WEIGHTS.items())
    return float(-total - 0.2 * a["cost"] / max(h["cost"], 1e-9))


def gate(values: list[float], threshold: float, above: bool = True, least: int = 10
         ) -> dict[str, Any]:
    """A threshold gate: the mean reaches it and ``least`` of the founders do."""
    v = np.asarray(values, dtype=np.float64)
    each = v >= threshold if above else v <= threshold
    mean_ok = bool(np.nanmean(v) >= threshold if above else np.nanmean(v) <= threshold)
    return {"mean": float(np.nanmean(v)), "founders": int(each.sum()), "of": len(v),
            "threshold": threshold, "passes": bool(mean_ok and each.sum() >= least)}


def read(out: Path) -> dict[str, Any]:
    arms: dict[str, dict[int, dict[str, Any]]] = {}
    for path in sorted(out.glob("*/founder-*.json")):
        r = json.loads(path.read_text())
        arms.setdefault(r.get("arm", path.parent.name), {})[int(r["seed"])] = r
    sel, hand = arms.get("selected", {}), arms.get("hand-set", {})
    seeds = sorted(set(sel) & set(hand))
    table = []
    for arm in ARMS:
        rs = [arms.get(arm, {}).get(s) for s in seeds]
        rs = [r for r in rs if r is not None]
        if not rs:
            continue
        t = [r["tests"] for r in rs]
        params = rs[0]["parameters"]
        table.append({
            "arm": arm, "founders": len(rs), "parameters": params,
            "learned": int(sum(params.values())),
            "macs": float(np.mean([sum(r["work"]["macs"].values()) + (r.get("control_macs") or 0)
                                   for r in rs])),
            **{k: float(np.nanmean([x[k] for x in t])) for k in (
                "E1_seen", "E1_behind", "E2_one", "E3_both", "E3_one", "E3_looked",
                "E3_first_quarter", "E3_last_quarter", "bearing")},
            "fitness": float(np.mean([fitness(arms[arm][s], hand[s]) for s in seeds
                                      if s in arms[arm]]))})
    gates: dict[str, Any] = {}
    if seeds:
        T = [sel[s]["tests"] for s in seeds]
        gates["E1 seen"] = gate([t["E1_seen"] for t in T], 0.85)
        gates["E1 behind"] = gate([t["E1_behind"] for t in T], 0.95)
        gates["E2"] = gate([t["E2_one"] for t in T], 0.85)
        gates["E3 learned, error"] = gate([t["E3_both"] for t in T], 0.08, above=False)
        gates["E3 learned, last quarter below first"] = gate(
            [t["E3_first_quarter"] - t["E3_last_quarter"] for t in T], 1e-12)
        ratio = [t["E3_one"] / t["E3_both"] for t in T]
        gates["E3 one eye"] = {**gate(ratio, 1.25, least=0), "above 1": int(
            sum(x > 1 for x in ratio)), "passes": bool(np.mean(ratio) >= 1.25
                                                      and sum(x > 1 for x in ratio) >= 10)}
        found = []
        silenced_x, bearing_ok, e2_ok, others = [], [], [], []
        for s in seeds:
            r = sel[s]
            if "found" not in r:
                continue
            units = r["found"]["regions"].get("distance", 0)
            found.append(units / float(r["genes"]["reach_bins"]))
            i, x, o = r["intact_small"], r["silenced"], r["silenced_others"]
            silenced_x.append(x["E3_both"] / max(i["E3_both"], 1e-9))
            bearing_ok.append(x["bearing"] - i["bearing"])
            e2_ok.append(abs(x["E2_one"] - i["E2_one"]) if np.isfinite(x["E2_one"]) and
                         np.isfinite(i["E2_one"]) else math.nan)
            others.append(o["E3_both"] - (1.25 * i["E3_both"] + 0.01))
        if found:
            gates["E7 found"] = gate(found, 0.8)
            s1 = gate(silenced_x, 2.0)
            s2 = gate(bearing_ok, 1.0, above=False)
            s3 = gate(e2_ok, 0.10, above=False)
            gates["E7 silenced"] = {"error times intact": s1, "bearing change": s2,
                                    "E2 change": s3,
                                    "passes": s1["passes"] and s2["passes"] and s3["passes"]}
            gates["E7 others"] = gate(others, 0.0, above=False)
        fit = {arm: {s: fitness(arms[arm][s], hand[s]) for s in seeds if s in arms.get(arm, {})}
               for arm in ("selected", "hand-set", "random-wiring", "random-search")}
        if fit["selected"] and fit["hand-set"]:
            d = [fit["selected"][s] - fit["hand-set"][s] for s in seeds]
            gates["E4 selected over hand-set"] = {**gate(d, 1e-12, least=8)}
        if fit["random-wiring"]:
            d = [fit["hand-set"][s] - fit["random-wiring"][s] for s in seeds
                 if s in fit["random-wiring"]]
            gates["E4 hand-set over random wiring"] = gate(d, 1e-12)
        if fit["random-search"]:
            d = [fit["selected"][s] - fit["random-search"][s] for s in seeds
                 if s in fit["random-search"]]
            gates["E4 selected over random search"] = {**gate(d, 1e-12, least=0)}
        for control in ("cnn", "vit"):
            if arms.get(control):
                d = [arms[control][s]["tests"]["E3_both"] - sel[s]["tests"]["E3_both"]
                     for s in seeds if s in arms[control]]
                gates[f"E5 below the {control}"] = gate(d, 1e-12)
    return {"founders": seeds, "table": table, "gates": gates}


def markdown(reading: dict[str, Any]) -> str:
    lines = ["| Arm | Learned parameters | MACs a moment | E1 seen | E1 behind | E2 | E3 both "
             "| E3 one eye | E3 first / last quarter | Fitness |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in reading["table"]:
        lines.append(
            f"| {row['arm']} ({row['founders']}) | {row['learned']:,} | {row['macs']:.3g} | "
            f"{row['E1_seen']:.3f} | {row['E1_behind']:.3f} | {row['E2_one']:.3f} | "
            f"{row['E3_both']:.3f} | {row['E3_one']:.3f} | {row['E3_first_quarter']:.3f} / "
            f"{row['E3_last_quarter']:.3f} | {row['fitness']:.2f} |")
    lines += ["", "| Gate | Reading | Passes |", "| --- | --- | --- |"]
    for name, g in reading["gates"].items():
        if "mean" in g:
            text = f"mean {g['mean']:.3f}, {g['founders']} of {g['of']} founders"
        else:
            text = "; ".join(f"{k}: mean {v['mean']:.3f}, {v['founders']} of {v['of']}"
                             for k, v in g.items() if isinstance(v, dict))
        lines.append(f"| {name} | {text} | {'yes' if g['passes'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--founders", type=int, nargs="+", default=FOUNDERS)
    parser.add_argument("--processes", type=int, default=60)
    parser.add_argument("--read", action="store_true")
    args = parser.parse_args()
    if not args.read:
        jobs = [(arm, seed, str(args.out)) for arm in args.arms for seed in args.founders]
        with Pool(args.processes, maxtasksperchild=1) as pool:
            for done in pool.imap_unordered(_live, jobs):
                print("done", done, flush=True)
    reading = read(args.out)
    (args.out / "readings.json").write_text(json.dumps(reading, indent=1, default=float))
    (args.out / "readings.md").write_text(markdown(reading))
    print(markdown(reading))


if __name__ == "__main__":
    main()
