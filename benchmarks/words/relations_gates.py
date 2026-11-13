"""The relations chamber's declared run: every arm's lives on the founders, and the gates read
from their files (cadence#327, RELATIONS_PROTOCOL.md).

    python relations_gates.py run <out> <first founder> <founders> [jobs]
    python relations_gates.py read <out>

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
PARAMETERS = 2368  # the brain's learned parameters: one efficacy a reciprocal pair
SIZES = {"matched": PARAMETERS, "large": 9 * PARAMETERS}
CONTROLS = [(m, mode, "matched") for m in ("lstm", "transformer")
            for mode in ("online", "offline")] + [(m, "offline", "large")
                                                   for m in ("lstm", "transformer")]

# the declared gates: (reading, threshold); a gate holds where the mean over the founders and at
# least ten of twelve founders reach it
GATES = {
    "R1": [("R1_thing", 0.90), ("R1_you", 0.90)],
    "R2": [("R2_thing", 0.90), ("R2_you", 0.90)],
    "R3": [("R3_thing", 0.90), ("R3_you", 0.90)],
    "R4": [("R4_held", 0.90)],
    "R5": [("R5_new", 0.90)],
}
READINGS = ["R1_thing", "R1_you", "R2_thing", "R2_thing_where", "R2_thing_howfar", "R2_you",
            "R2_you_where_you", "R2_you_howfar_you", "R3_thing", "R3_you", "R4_held", "R5_new",
            "R6_ablated"]


def jobs(out: Path, first: int, count: int) -> list[tuple[Path, list[str]]]:
    python = sys.executable
    todo = []
    for seed in range(first, first + count):
        todo.append((out / f"brain_{seed}.json", [python, str(HERE / "relations.py"),
                                                  str(seed)]))
        for model, mode, size in CONTROLS:
            todo.append((out / f"{model}_{mode}_{size}_{seed}.json",
                         [python, str(HERE / "relations_controls.py"), str(seed), model, mode,
                          str(SIZES[size])]))
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
            path.with_suffix(".err").write_text(done.stderr)
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
            read = np.array([life[name] for life in brain])
            reach = int((read >= threshold).sum())
            ok = bool(len(read)) and bool(read.mean() >= threshold) and reach >= min(10, len(read))
            held &= ok
            parts.append(f"{name} mean {read.mean():.3f}, {reach} of {len(read)} at "
                         f"{threshold}")
        verdicts[gate] = held
        print(f"{gate}: {'holds' if held else 'fails'} ({'; '.join(parts)})")
    ablated = [life["R6_ablated"] for life in brain]
    verdicts["R6"] = bool(ablated) and max(ablated) <= 0.05
    print(f"R6 ablation: the relation groups' synapses at zero answer R1 and R2 at "
          f"{max(ablated) if ablated else float('nan'):.3f} at most")
    return verdicts


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 16)
    else:
        read(Path(sys.argv[2]))
