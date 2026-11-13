"""The retina chamber's declared run: every arm's lives on the founders, and the gates read from
their files (cadence#330, RETINA_PROTOCOL.md).

    python retina_gates.py run <out> <first founder> <founders> [jobs]
    python retina_gates.py read <out>

``run`` writes one JSON file a life: ``life_<seed>.json`` (the parrot with eyes and the parrot
fed the camera's reports, practice told), ``reward_<seed>.json`` (both by pistachios, a reading)
and ``<model>_<mode>_<size>_<seed>.json`` for the controls (``size`` is ``matched``, at the
brain's parameter count, or ``large``, at nine times it). A control waits for its founder's
life, whose looks it is shown. ``read`` prints the tables and each gate's verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMETERS = 15104  # the parrot with eyes: the lexicon's pairs and the two category layers'
SIZES = {"matched": PARAMETERS, "large": 9 * PARAMETERS}
CONTROLS = [(m, mode, "matched") for m in ("cnn", "vit") for mode in ("online", "offline")] + [
    (m, "offline", "large") for m in ("cnn", "vit")]
WORDS = ["W1", "W1_named_only", "W2", "W3", "W4_never_seen", "W4_never_taught",
         "W5_by_colour", "W5_by_place"]
READINGS = WORDS + ["W4_out_of_view", "W4_map_rate", "W6_ablated"]


def jobs(out: Path, first: int, count: int) -> list[tuple[Path, list[str], Path | None]]:
    python = sys.executable
    todo: list[tuple[Path, list[str], Path | None]] = []
    for seed in range(first, first + count):
        for name, given in (("life", ""), ("reward", "practice='reward', practice_rooms=60")):
            code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import retina; "
                    f"print(json.dumps(retina.life({seed}, {given}), default=float))")
            todo.append((out / f"{name}_{seed}.json", [python, "-c", code], None))
    for seed in range(first, first + count):
        life = out / f"life_{seed}.json"
        for model, mode, size in CONTROLS:
            code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import retina_controls; "
                    f"looks = json.load(open({str(life)!r}))['eyes']['look_headings']; "
                    f"print(json.dumps(retina_controls.run({seed}, {model!r}, {mode!r}, "
                    f"{SIZES[size]}, looks), default=float))")
            todo.append((out / f"{model}_{mode}_{size}_{seed}.json", [python, "-c", code], life))
    return todo


def run(out: Path, first: int, count: int, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}

    def one(job: tuple[Path, list[str], Path | None]) -> str:
        path, command, needs = job
        if path.exists():
            return f"kept {path.name}"
        while needs is not None and not needs.exists():
            if needs.with_suffix(".err").exists():
                return f"skipped {path.name}: {needs.name} failed"
            time.sleep(10)
        done = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
        if done.returncode:
            (path.with_suffix(".err")).write_text(done.stderr)
            return f"failed {path.name}"
        part = path.with_suffix(".part")
        part.write_text(done.stdout)
        part.rename(path)
        return f"wrote {path.name}"

    with ThreadPoolExecutor(workers) as pool:
        for line in pool.map(one, jobs(out, first, count)):
            print(line, flush=True)


def _reach(values: np.ndarray, threshold: float, side: str) -> tuple[bool, int]:
    """Whether the mean and at least ten of twelve founders (all, with fewer) lie on ``side``
    of ``threshold``; and how many founders do."""
    hits = values >= threshold if side == "at least" else values <= threshold
    mean = values.mean() >= threshold if side == "at least" else values.mean() <= threshold
    return bool(len(values)) and bool(mean) and int(hits.sum()) >= min(10, len(values)), int(
        hits.sum())


def read(out: Path) -> dict[str, Any]:
    lives = [json.loads(p.read_text()) for p in sorted(out.glob("life_*.json"))]
    rewards = [json.loads(p.read_text()) for p in sorted(out.glob("reward_*.json"))]
    arms: dict[str, list[dict[str, Any]]] = {"eyes": [life["eyes"] for life in lives],
                                             "camera": [life["camera"] for life in lives]}
    if rewards:
        arms["eyes, pistachios"] = [life["eyes"] for life in rewards]
        arms["camera, pistachios"] = [life["camera"] for life in rewards]
    for model, mode, size in CONTROLS:
        found = sorted(out.glob(f"{model}_{mode}_{size}_*.json"))
        if found:
            arms[f"{model} {mode} {size}"] = [json.loads(p.read_text()) for p in found]
    head = ["arm", "founders", "parameters", "multiply-accumulates a life"] + READINGS
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for arm, rows in arms.items():
        line = [arm, str(len(rows)), f"{int(rows[0].get('parameters', 0)):,}",
                f"{np.mean([r.get('macs', 0) for r in rows]):.3g}"]
        for name in READINGS:
            values = [r[name] for r in rows if name in r and r[name] == r[name]]
            line.append(f"{np.mean(values):.3f} ({min(values):.2f}-{max(values):.2f})"
                        if values else "")
        print("| " + " | ".join(line) + " |")
    sight = {"R1 found": [life["R1"]["found"] for life in lives],
             "R1 figures that are no thing": [life["R1"]["spurious"] for life in lives],
             "R1 recognised": [life["R1"]["recognised"] for life in lives],
             "R2 in view": [life["R2"]["in_view"] for life in lives],
             "R2 out of view": [life["R2"]["out_of_view"] for life in lives],
             "R2 eyes' place from the true one (median m)": [life["R2"]["eyes_true_median"]
                                                             for life in lives],
             "R3 kind and colour": [life["R3"]["R3_both"] for life in lives],
             "R3 kind": [life["R3"]["R3_kind"] for life in lives],
             "R3 colour": [life["R3"]["R3_colour"] for life in lives],
             "kind categories": [life["eyes"]["categories"]["kinds"] for life in lives],
             "kind categories' purity": [life["R1"]["kinds_purity"] for life in lives],
             "colour categories": [life["eyes"]["categories"]["colours"] for life in lives],
             "colour categories' purity": [life["R1"]["colours_purity"] for life in lives],
             "looks a life": [life["eyes"]["looks"] for life in lives]}
    print("\n| reading | mean | range |\n| --- | --- | --- |")
    for name, values in sight.items():
        if values:
            print(f"| {name} | {np.mean(values):.3f} | {min(values):.3f}-{max(values):.3f} |")
    verdicts: dict[str, Any] = {}

    def gate(name: str, parts: list[tuple[str, np.ndarray, float, str]]) -> None:
        held, said = True, []
        for label, values, threshold, side in parts:
            ok, hits = _reach(values, threshold, side)
            held &= ok
            said.append(f"{label} mean {values.mean():.3f}, {hits} of {len(values)} {side} "
                        f"{threshold:.3g}")
        verdicts[name] = held
        print(f"{name}: {'holds' if held else 'fails'} ({'; '.join(said)})")

    print()
    gate("R1", [("found", np.array(sight["R1 found"]), 0.95, "at least"),
                ("figures that are no thing", np.array(sight["R1 figures that are no thing"]),
                 0.05, "at most")])
    gate("R2", [("in view", np.array(sight["R2 in view"]), 0.95, "at least"),
                ("out of view", np.array(sight["R2 out of view"]), 0.95, "at least")])
    gate("R3", [("kind and colour", np.array(sight["R3 kind and colour"]), 0.90, "at least")])
    # R4: each reading within .05 of the camera's in the mean, and within .10 on ten founders
    parts = []
    eyes, camera = arms["eyes"], arms["camera"]
    for name in WORDS:
        gap = np.array([e[name] - c[name] for e, c in zip(eyes, camera, strict=True)])
        parts.append((f"{name} less the camera's", gap, -0.05, "at least"))
    held, said = True, []
    for label, gap, threshold, _ in parts:
        hits = int((gap >= -0.10).sum())
        ok = bool(len(gap)) and gap.mean() >= threshold and hits >= min(10, len(gap))
        held &= ok
        said.append(f"{label} mean {gap.mean():+.3f}, {hits} of {len(gap)} at -0.10 or more")
    out_gap = np.array([abs(e["W4_out_of_view"] - e["W4_map_rate"]) for e in eyes
                        if e["W4_out_of_view"] == e["W4_out_of_view"]])
    ok, hits = _reach(out_gap, 0.05, "at most")
    held &= ok
    said.append(f"W4 out of view from the map's own rate mean {out_gap.mean():.3f}, {hits} of "
                f"{len(out_gap)} at most 0.05")
    ablated = np.array([e["W6_ablated"] for e in eyes])
    held &= bool(len(ablated)) and ablated.max() <= 0.05
    said.append(f"W6 the lexicon at zero answers W1 at {ablated.max():.3f} at most")
    verdicts["R4"] = held
    print(f"R4: {'holds' if held else 'fails'} ({'; '.join(said)})")
    return verdicts


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 30)
    else:
        read(Path(sys.argv[2]))
