"""The map of things' declared gates (PROTOCOL.md), read from a folder of a run's files.

    python benchmarks/tracking/gates_map.py RUN_FOLDER

Each founder holds ``mapped-F.json`` (``mapped.py``) and ``rooms-F.json`` (``rooms.py``). Prints
every gate with its mean over the founders and its standard error, and the readings reported
beside the gates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

MAP = "map, near, reach 0.15 m"


def mean_se(values: list[float]) -> tuple[float, float]:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def gates(folder: Path) -> dict[str, Any]:
    mapped = {int(p.stem.split("-")[1]): json.loads(p.read_text())
              for p in sorted(folder.glob("mapped-*.json"))}
    rooms = {int(p.stem.split("-")[1]): json.loads(p.read_text())
             for p in sorted(folder.glob("rooms-*.json"))}
    founders = sorted(set(mapped) & set(rooms))
    out: dict[str, Any] = {"founders": founders}

    def get(fn: Any) -> tuple[float, float]:
        return mean_se([fn(f) for f in founders])

    r = {f: mapped[f]["readings"] for f in founders}
    for case in ("a person's hand", "own"):
        ratio = get(lambda f, c=case: r[f][c][MAP]["in_view"] / r[f][c][MAP]["camera"])
        out[f"S1 {case}: in view over the camera's error"] = (ratio, ratio[0] <= 1.1)
    moved = get(lambda f: r[f]["stays"]["moved"])
    last = get(lambda f: r[f]["stays"]["last 100"] - r[f]["stays"]["while seen"])
    out["S3: moved over the hidden moments (m)"] = (moved, moved[0] == 0.0)
    out["S3: last 100 less while seen (m)"] = (last, last[0] <= 0.0)
    heard = get(lambda f: r[f]["heard"]["after"] - r[f]["heard"]["cue"])
    out["S4: after the last cue less the cues' own error (m)"] = (heard, heard[0] <= 0.0)
    ms = get(lambda f: r[f]["a person's hand"][MAP]["milliseconds"])
    out["S7: milliseconds a moment natively, one thing"] = (ms, ms[0] < 33.0)
    k = {f: rooms[f]["kinds"] for f in founders}
    seen_error = 0.01 * np.sqrt(np.pi / 2)
    for part in ("in view", "seen 500 before"):
        where = get(lambda f, p=part: float(np.mean([x[p]["where"] for x in k[f]])))
        what = get(lambda f, p=part: float(np.mean([x[p]["what"] for x in k[f]])))
        error = get(lambda f, p=part: float(np.mean([x[p]["error"] for x in k[f]])))
        label = "S9" if part == "in view" else "S9/S10"
        out[f"{label} {part}: where within 5 cm"] = (where, where[0] >= 0.95)
        out[f"{label} {part}: what"] = (what, what[0] >= 0.95)
        out[f"{label} {part}: mean error over the seen error"] = (
            (error[0] / seen_error, error[1] / seen_error), error[0] <= 1.5 * seen_error)
    m = {f: rooms[f]["moved"] for f in founders}
    for key in ("old place held before", "new place reported", "nothing at the old place"):
        share = get(lambda f, key=key: float(np.mean([x[key] for x in m[f]])))
        out[f"S11: {key}"] = (share, share[0] >= 0.95)
    for n in ("8", "32", "128"):
        for key in ("where", "what"):
            share = get(lambda f, n=n, key=key: float(np.mean(
                [x[key] for x in rooms[f]["many"][n]])))
            out[f"S12 {n} things: {key}"] = (share, share[0] >= 0.95)
    a = {f: rooms[f]["arrive"] for f in founders}
    for mark in ("149", "279", "640", "799"):
        found = get(lambda f, mk=mark: float(np.mean([
            1.0 - (x[mk]["extra"] + x[mk]["missed"]) / max(x[mk]["things"] + x[mk]["missed"], 1)
            for x in a[f]])))
        out[f"S13 at {mark}: listed right"] = (found, found[0] >= 0.95)
    gone = get(lambda f: float(np.mean([1 in x["640"]["not known where"] for x in a[f]])))
    out["S13: the thing taken away, not known where"] = (gone, gone[0] >= 0.95)
    state = {f: rooms[f]["state"] for f in founders}
    held = get(lambda f: state[f]["after 1,000 moments"]["where"])
    out["S14: held after 1,000 moments"] = (held, held[0] >= 0.95)
    alike = get(lambda f: float(state[f]["file answers alike"]))
    empty = get(lambda f: float(state[f]["emptied reports nothing"]))
    out["S14: a file answers alike"] = (alike, alike[0] == 1.0)
    out["S14: emptied, no thing"] = (empty, empty[0] == 1.0)
    beside: dict[str, Any] = {}
    for panel in ("a person's hand", "errands", "promise", "promise-folded"):
        for arm in ("last seen", "Kalman", MAP, "map, keep"):
            if arm in r[founders[0]].get(panel, {}):
                beside[f"{panel}, {arm}"] = {
                    "out of view": get(lambda f, p=panel, a=arm: r[f][p][a]["out_of_view"]),
                    **({"area .90 holds": get(lambda f, p=panel, a=arm: r[f][p][a]["in90"]),
                        "area .90 cells": get(lambda f, p=panel, a=arm: r[f][p][a]["cells90"])}
                       if "in90" in r[founders[0]][panel][arm] else {}),
                    **{name: get(lambda f, p=panel, a=arm, n=name: r[f][p][a]["panels"][n][
                        "error"]) for name in r[founders[0]][panel][arm].get("panels", {})}}
    out["beside"] = beside
    return out


def main(argv: list[str] | None = None) -> int:
    folder = Path((argv or sys.argv[1:])[0])
    result = gates(folder)
    print(f"founders: {result['founders']}")
    for name, value in result.items():
        if name in ("founders", "beside"):
            continue
        (m, se), ok = value
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {m:.4f} (se {se:.4f})")
    for name, value in result["beside"].items():
        cells = "  ".join(f"{k} {v[0] * (100 if 'cells' not in k and 'holds' not in k else 1):.2f}"
                          for k, v in value.items())
        print(f"      {name}: {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
