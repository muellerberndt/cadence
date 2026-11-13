"""The words chamber's declared run: every arm's lives on the founders, and the gates read from
their files (cadence#326, PROTOCOL.md).

    python gates.py run <out> <first founder> <founders> [jobs]
    python gates.py read <out>

``run`` writes one JSON file a life: ``brain_<seed>.json`` and
``<model>_<mode>_<size>_<seed>.json`` for the controls (``size`` is ``matched``, at the brain's
parameter count, or ``large``, at nine times it). ``read`` prints the table and each gate's
verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMETERS = 6656  # the brain's learned parameters: one efficacy a reciprocal pair
SIZES = {"matched": PARAMETERS, "large": 9 * PARAMETERS}
CONTROLS = [(m, mode, size) for m in ("lstm", "transformer") for mode in ("online", "offline")
            for size in ("matched",)] + [(m, "offline", "large") for m in ("lstm", "transformer")]

# the declared gates: (reading, threshold); a gate holds where the mean over the founders and at
# least ten of twelve founders reach it
GATES = {
    "W1": [("W1", 0.95), ("W1_named_only", 0.95)],
    "W2": [("W2", 0.90)],
    "W3": [("W3", 0.90)],
    "W4": [("W4_never_seen", 0.95), ("W4_never_taught", 0.95)],
    "W5": [("W5_by_colour", 0.90), ("W5_by_place", 0.90)],
}
READINGS = ["W1", "W1_practised", "W1_named_only", "W2", "W3", "W3_where", "W4_out_of_view",
            "W4_map_rate", "W4_never_seen", "W4_never_taught", "W5_by_colour", "W5_by_place",
            "W6_ablated"]


def jobs(out: Path, first: int, count: int) -> list[tuple[Path, list[str]]]:
    python = sys.executable
    todo = []
    for seed in range(first, first + count):
        todo.append((out / f"brain_{seed}.json", [python, str(HERE / "words.py"), str(seed)]))
        for model, mode, size in CONTROLS:
            code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import controls; "
                    f"print(json.dumps(controls.run({seed}, {model!r}, {mode!r}, "
                    f"{SIZES[size]}), default=float))")
            todo.append((out / f"{model}_{mode}_{size}_{seed}.json", [python, "-c", code]))
    return todo


def run(out: Path, first: int, count: int, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}

    def one(job: tuple[Path, list[str]]) -> str:
        path, command = job
        if path.exists():
            return f"kept {path.name}"
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


def read(out: Path) -> dict[str, Any]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out.glob("*.json")):
        arm = path.stem.rsplit("_", 1)[0]
        arms.setdefault(arm, []).append(json.loads(path.read_text()))
    head = ["arm", "founders", "parameters", "multiply-accumulates", "seconds"] + READINGS
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for arm, lives in arms.items():
        row = [arm, str(len(lives)), str(lives[0].get("parameters", "")),
               f"{np.mean([life.get('macs', 0) for life in lives]):.3g}",
               f"{np.mean([sum(life['seconds'].values()) for life in lives]):.0f}"]
        for name in READINGS:
            values = [life[name] for life in lives if name in life and life[name] == life[name]]
            row.append(f"{np.mean(values):.3f} ({min(values):.2f}-{max(values):.2f})"
                       if values else "")
        print("| " + " | ".join(row) + " |")
    verdicts: dict[str, Any] = {}
    brain = arms.get("brain", [])
    for gate, needs in GATES.items():
        held = True
        parts = []
        for name, threshold in needs:
            values = np.array([life[name] for life in brain])
            reach = int((values >= threshold).sum())
            ok = bool(len(values)) and values.mean() >= threshold and reach >= min(
                10, len(values))
            held &= ok
            parts.append(f"{name} mean {values.mean():.3f}, {reach} of {len(values)} at "
                         f"{threshold}")
        verdicts[gate] = held
        print(f"{gate}: {'holds' if held else 'fails'} ({'; '.join(parts)})")
    gaps = [abs(life["W4_out_of_view"] - life["W4_map_rate"]) for life in brain]
    verdicts["W4_out_of_view"] = bool(gaps) and max(gaps) <= 0.02
    print(f"W4 out of view: {'holds' if verdicts['W4_out_of_view'] else 'fails'} "
          f"(largest gap from the map's own rate {max(gaps) if gaps else float('nan'):.3f})")
    ablated = [life["W6_ablated"] for life in brain]
    verdicts["W6"] = bool(ablated) and max(ablated) <= 0.05
    print(f"W6 ablation: the lexicon's synapses at zero answer W1 at "
          f"{max(ablated) if ablated else float('nan'):.3f} at most")
    return verdicts


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 30)
    else:
        read(Path(sys.argv[2]))
