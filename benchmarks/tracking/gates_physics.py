"""Learned physics on the map: its declared gates (PROTOCOL.md, issue #324), read from a folder
of a run's files.

    python benchmarks/tracking/gates_physics.py RUN_FOLDER

Each founder holds ``physics-F.json`` (``mapped.py --physics``). Prints every gate with its mean
over the founders and its standard error, and the readings reported beside the gates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HAND, ERRANDS = "a person's hand", "errands"
MAP = "map, near, reach 0.15 m"


def mean_se(values: list[float]) -> tuple[float, float]:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def physics_at(run: dict[str, Any], watched: int, panel: str = HAND) -> dict[str, Any]:
    (point,) = [p for p in run["curve"] if p["watched"] == watched]
    (table,) = point["readings"][panel].values()
    return table


def gates(folder: Path, last: int = 400) -> dict[str, Any]:
    runs = {int(p.stem.split("-")[1]): json.loads(p.read_text())
            for p in sorted(folder.glob("physics-*.json"))}
    founders = sorted(runs)
    out: dict[str, Any] = {"founders": founders}

    def get(fn: Any) -> tuple[float, float]:
        return mean_se([fn(f) for f in founders])

    hand = {f: runs[f]["readings"][HAND] for f in founders}
    at = {f: physics_at(runs[f], last) for f in founders}
    gap = get(lambda f: hand[f][MAP]["out_of_view"] - at[f]["out_of_view"])
    out["P1: the map less the map with physics, out of view (m)"] = (gap, gap[0] > 2 * gap[1])
    straight = get(lambda f: at[f]["panels"]["straight"]["error"])
    kalman = get(lambda f: hand[f]["Kalman"]["panels"]["straight"]["error"])
    out["P2: straight (m)"] = (straight, straight[0] <= 2.0 * kalman[0])
    out["P2: the Kalman filter, straight (m)"] = (kalman, None)
    stop = get(lambda f: at[f]["panels"]["stop"]["error"])
    stop_map = get(lambda f: hand[f][MAP]["panels"]["stop"]["error"])
    out["P3: stops (m)"] = (stop, stop[0] <= stop_map[0])
    out["P3: the map, stops (m)"] = (stop_map, None)
    held = get(lambda f: at[f]["in90"])
    out["P4: the area at .90 holds the hand, out of view"] = (held, held[0] >= 0.90)
    marks = sorted({p["watched"] for f in founders for p in runs[f]["curve"]})
    above = []
    for mark in marks:
        diff = get(lambda f, m=mark: physics_at(runs[f], m)["out_of_view"]
                   - hand[f][MAP]["out_of_view"])
        above.append(diff[0] > 0.0)
        out[f"P5: {mark} watched, the map with physics less the map, out of view (m)"] = (
            diff, diff[0] <= 0.0)
    out["P5: the curve never goes above the map"] = ((float(not any(above)), 0.0), not any(above))
    # beside the gates
    for name in ("slow", "turn"):
        out[f"beside: {name} (m)"] = (get(lambda f, n=name: at[f]["panels"][n]["error"]), None)
        out[f"beside: the map, {name} (m)"] = (get(lambda f, n=name: hand[f][MAP]["panels"][n][
            "error"]), None)
    out["beside: the map out of view (m)"] = (get(lambda f: hand[f][MAP]["out_of_view"]), None)
    out["beside: Kalman out of view (m)"] = (get(lambda f: hand[f]["Kalman"]["out_of_view"]), None)
    out["beside: last seen out of view (m)"] = (get(lambda f: hand[f]["last seen"]["out_of_view"]),
                                               None)
    out["beside: in view (m)"] = (get(lambda f: at[f]["in_view"]), None)
    out["beside: the area at .90, cells"] = (get(lambda f: at[f]["cells90"]), None)
    out["beside: the area at .95 holds"] = (get(lambda f: at[f]["in95"]), None)
    errands = {f: physics_at(runs[f], last, ERRANDS) for f in founders}
    out["beside: errands out of view (m)"] = (get(lambda f: errands[f]["out_of_view"]), None)
    out["beside: the map, errands out of view (m)"] = (get(
        lambda f: runs[f]["readings"][ERRANDS][MAP]["out_of_view"]), None)
    for key in ("rest", "leave", "ahead", "aside"):
        out[f"beside: learned {key}"] = (get(
            lambda f, k=key: [p for p in runs[f]["curve"] if p["watched"] == last][0]["physics"][
                k]), None)
    return out


def main(argv: list[str] | None = None) -> int:
    folder = Path((argv or sys.argv[1:])[0])
    out = gates(folder)
    print(f"founders: {out.pop('founders')}")
    for name, ((mean, se), passed) in out.items():
        mark = "" if passed is None else ("  pass" if passed else "  FAIL")
        print(f"{name:72} {mean:9.4f} (se {se:.4f}){mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
